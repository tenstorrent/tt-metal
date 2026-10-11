# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded HTTP prefix correctness probe for an exclusively owned test server.

This is a mechanism qualification, not AgentX, GPQA, or a production benchmark.
It uses only generated test prompts and the bundle's private checkpoint store.
"""

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def make_prompts(weights):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
    stem = tokenizer.encode(
        "The cedar tree stands beside the river. Remember the word cedar. ", add_special_tokens=False
    )
    prefix = (stem * (4096 // len(stem) + 1))[:4096]
    prompts = {}
    for length in (1, 31, 32, 33):
        suffix = tokenizer.encode(f" Repeat cedar and describe the tree in {length} words. ", add_special_tokens=False)
        prompts[str(length)] = prefix + (suffix * (length // len(suffix) + 1))[:length]
    return dict(prefix_tokens=4096, prompts=prompts, tokenizer_class=type(tokenizer).__name__)


def get(endpoint, path, timeout=10):
    if not endpoint.startswith("http://127.0.0.1:"):
        raise ValueError("Qualification must target its own loopback endpoint")
    with urllib.request.urlopen(endpoint + path, timeout=timeout) as response:
        return response.read()


def wait_ready(endpoint, process, *, timeout, check=lambda: None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        check()
        if process.poll() is not None:
            raise RuntimeError(f"Owned server exited before readiness: {process.returncode}")
        try:
            get(endpoint, "/health")
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError("Owned server did not become ready before its startup deadline")


def metric(endpoint, name):
    rows = get(endpoint, "/metrics").decode().splitlines()
    values = [float(row.rsplit(" ", 1)[1]) for row in rows if row.startswith(name + "{") or row.startswith(name + " ")]
    if not values:
        raise ValueError("Required cache metric is not exposed: " + name)
    return sum(values)


HITS = "vllm:external_prefix_cache_hits_total"


def completion(endpoint, prompt, salt, *, output_tokens=64, logprobs=False):
    payload = dict(
        model="prefix-qualification",
        prompt=prompt,
        max_tokens=output_tokens,
        temperature=0,
        top_k=1,
        seed=20261011,
        cache_salt=salt,
        return_token_ids=True,
        ignore_eos=True,
    )
    if logprobs:
        payload["logprobs"] = 1
    request = urllib.request.Request(
        endpoint + "/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=300) as response:
        value = json.load(response)
    choice = value["choices"][0]
    ids = choice.get("token_ids")
    if not isinstance(ids, list) or len(ids) != output_tokens or value["usage"]["completion_tokens"] != output_tokens:
        raise ValueError("Completion omitted token IDs, truncated, or produced the wrong output count")
    return dict(
        token_ids=ids,
        text=choice["text"],
        elapsed_s=time.monotonic() - started,
        usage=value["usage"],
        finish_reason=choice["finish_reason"],
    )


def hit_completion(endpoint, prompt, salt, **kwargs):
    before = metric(endpoint, HITS)
    result = completion(endpoint, prompt, salt, **kwargs)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        delta = metric(endpoint, HITS) - before
        if delta >= 4096:
            result["external_hit_tokens_delta"] = delta
            return result
        time.sleep(1)
    raise ValueError("Warm output arrived without a measured complete-prefix scheduler hit")


def same(left, right):
    if left["token_ids"] != right["token_ids"] or left["text"] != right["text"]:
        raise ValueError("Cold and restored greedy outputs differ")


def cancel_stream(endpoint, prompt, salt):
    payload = dict(
        model="prefix-qualification",
        prompt=prompt,
        max_tokens=512,
        temperature=0,
        top_k=1,
        seed=20261011,
        cache_salt=salt,
        stream=True,
        ignore_eos=True,
    )
    request = urllib.request.Request(
        endpoint + "/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        for line in response:
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                chunk = json.loads(line[6:])
                if any(choice.get("text") for choice in chunk.get("choices", [])):
                    return dict(cancelled_after_first_output=True)
    raise ValueError("Cancellation probe produced no real output before closing the stream")


def exercise(endpoint, bundle, output):
    from models.demos.qwen38_27b_qb2.demo.prefix_qualification import sha, write
    from models.demos.qwen38_27b_qb2.tt.prefix_storage import AtomicDirectoryStore

    prompts = json.loads((bundle / "prompts.json").read_text())["prompts"]
    store = AtomicDirectoryStore(output / "checkpoint-store", max_bytes=2**31)
    result = dict(passed=False, cases=[], cold={}, warm={}, checkpoint_keys={})
    receipt = output / "first-probe.json"
    write(receipt, result)
    # Different salts force cold baselines but keep the exact same text/state
    # computation. Reuse each salt for its corresponding warm request.
    for length, prompt in prompts.items():
        salt = "suffix-" + length
        before = {path.name for path in store.root.glob("*.checkpoint")}
        cold = completion(endpoint, prompt, salt)
        created = {path.name for path in store.root.glob("*.checkpoint")} - before
        if len(created) != 1:
            raise ValueError("Cold prefill did not publish exactly one 4096-token checkpoint")
        result["checkpoint_keys"][length] = created.pop().removesuffix(".checkpoint")
        warm = hit_completion(endpoint, prompt, salt)
        same(cold, warm)
        result["cold"][length], result["warm"][length] = cold, warm
        result["cases"].append("cold_warm_suffix_" + length)
        write(receipt, result)
    # Concurrent client submissions compare all 64 tokens against isolated
    # baselines. This alone does not prove hardware overlap or physical remaps.
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {
            length: pool.submit(completion, endpoint, prompts[length], "suffix-" + length) for length in ("31", "33")
        }
        concurrent = {length: future.result(timeout=600) for length, future in pending.items()}
    for length, value in concurrent.items():
        same(result["cold"][length], value)
    result["concurrent"] = dict(outputs=concurrent, elapsed_s=time.monotonic() - started)
    result["cases"].append("concurrent_submission_output_isolation")
    # Terminate one real streaming request while a second request completes,
    # then reuse slots. The cancelled request never becomes a reference output.
    with ThreadPoolExecutor(max_workers=2) as pool:
        survivor = pool.submit(completion, endpoint, prompts["33"], "suffix-33")
        result["cancel"] = cancel_stream(endpoint, prompts["31"], "suffix-31")
        same(result["cold"]["33"], survivor.result(timeout=600))
    same(result["cold"]["31"], hit_completion(endpoint, prompts["31"], "suffix-31"))
    result["cases"].append("cancellation_survivor_and_followup_output")
    # Mutate only this test's derived cache file, under its own storage lease.
    # A missing completion marker forces worker failure after partial restore.
    key = result["checkpoint_keys"]["32"]
    blob = store.root / (key + ".checkpoint")
    original = sha(blob)
    with store._lock(), blob.open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    recovered = completion(endpoint, prompts["32"], "suffix-32")
    same(result["cold"]["32"], recovered)
    if sha(blob) != original:
        raise ValueError("Failed external load did not cold-recompute and repair the exact checkpoint")
    result["cases"].append("corrupt_restore_recompute_repair")
    store.discard(key)
    same(result["cold"]["32"], completion(endpoint, prompts["32"], "suffix-32"))
    if sha(blob) != original:
        raise ValueError("Missing checkpoint was not republished exactly")
    result["cases"].append("missing_checkpoint_recompute")
    # Sparse derived fixture fills only the configured accounting quota, never
    # the host filesystem. This proves cache admission failure, not real ENOSPC.
    fixture = store.root / "qualification-quota.partial"
    with store._lock(), fixture.open("xb") as stream:
        stream.truncate(store.max_bytes - store.used_bytes)
    try:
        before = {path.name for path in store.root.glob("*.checkpoint")}
        uncached = completion(endpoint, prompts["32"], "quota-decline")
        same(result["cold"]["32"], uncached)
        if {path.name for path in store.root.glob("*.checkpoint")} != before:
            raise ValueError("Full cache quota admitted another checkpoint")
    finally:
        fixture.unlink()
    result["cases"].append("quota_declines_capture_without_request_failure")
    result.update(
        passed=True,
        preemption_tested=False,
        physical_slot_remap_observed=False,
        latency_measure="nonstream end-to-end response latency, not TTFT",
        limitation="Concurrent submissions and cancellation outputs are checked; physical slot remaps and forced scheduler preemption need separate worker evidence.",
    )
    write(receipt, result)
    return dict(path=str(receipt.relative_to(bundle)), sha256=sha(receipt), passed=True)


def restart_probe(endpoint, bundle, output):
    from models.demos.qwen38_27b_qb2.demo.prefix_qualification import sha, write

    previous = json.loads((output / "first-probe.json").read_text())
    prompts = json.loads((bundle / "prompts.json").read_text())["prompts"]
    restored = hit_completion(endpoint, prompts["32"], "suffix-32")
    same(previous["cold"]["32"], restored)
    receipt = output / "restart-probe.json"
    write(receipt, dict(passed=True, output=restored, persisted_checkpoint_reused=True))
    return dict(path=str(receipt.relative_to(bundle)), sha256=sha(receipt), passed=True)
