"""Synthetic CPU transport/scoring test. NEVER model benchmark results."""
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path.cwd() / "models/autoports/qwen_qwen3_8_flash_next"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output", type=Path, default=ROOT / "doc/benchmark/repair_deadline/runtime_dispatch_mock_full_manifest"
)
OUT = parser.parse_args().output
OUT.mkdir(parents=True, exist_ok=False)
CONFIG = json.loads((ROOT / "doc/benchmark/run_config.json").read_text())
MANIFEST = Path(CONFIG["manifest"])
manifest = json.loads(MANIFEST.read_text())
HOOKS = ROOT / "tools/benchmark_client_hooks"
sys.path.insert(0, str(HOOKS))
os.environ["BENCHMARK_ARTIFACT_DIR"] = str(OUT)
original_argv = sys.orig_argv
sys.orig_argv = ["python", "-m", "benchmark_stage", "evaluate"]
spec = importlib.util.spec_from_file_location("mock_recording_sitecustomize", HOOKS / "sitecustomize.py")
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)
sys.orig_argv = original_argv

import aiohttp
from benchmark_stage.evaluate import evaluate_groups
from benchmark_stage.responses import read_jsonl, scoring_response
from benchmark_stage.subsets import digest
from lm_eval.models.openai_completions import LocalChatCompletion


async def forbidden_network(*args, **kwargs):
    raise AssertionError("Network forbidden in synthetic integration test")


aiohttp.ClientSession.post = forbidden_network
state = dict(started=[], completed=[], active=0, max_active=0, started_request_hashes=[])
original_amodel = LocalChatCompletion.amodel_call


async def fake_amodel(self, session, sem, messages, *, cache_keys=None, generate=True, gen_kwargs=None, **kwargs):
    number = len(state["started"])
    state["started"].append(number)
    state["started_request_hashes"].append(digest(cache_keys[0]))
    payload = self._create_payload(
        self.create_message(messages), generate=generate, gen_kwargs=copy.deepcopy(gen_kwargs), seed=self._seed
    )
    assert payload["messages"] == json.loads(messages[0].prompt)
    async with sem:
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep((31 - number % 32) * 0.001)
        raw = dict(
            id=f"SYNTHETIC-NOT-MODEL-{number:04d}",
            object="chat.completion",
            model="SYNTHETIC-TRANSPORT-ONLY",
            choices=[
                dict(
                    index=0,
                    finish_reason="stop",
                    message=dict(
                        role="assistant",
                        reasoning="Synthetic test reasoning; no model inference.",
                        content=f"SYNTHETIC transport answer {number}.",
                    ),
                )
            ],
            usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        answers = self.parse_generations(raw)
        for result, key in zip(answers, cache_keys or []):
            self.cache_hook.add_partial("generate_until", key, result)
        state["completed"].append(number)
        state["active"] -= 1
        return answers


LocalChatCompletion.amodel_call = fake_amodel
(OUT / "SYNTHETIC_ONLY.txt").write_text(
    "Synthetic CPU transport/scoring integration. No model, HTTP request, hardware, or benchmark accuracy evidence. All scores in this directory are synthetic and must never enter benchmark reports.\n"
)
started = time.monotonic()
try:
    result = evaluate_groups(
        model="SYNTHETIC-TRANSPORT-ONLY",
        base_url="http://no-network.invalid",
        manifest_path=MANIFEST,
        groups=CONFIG["tasks"],
        output=OUT,
        generation=CONFIG["generation"][CONFIG["tasks"][0]],
        shared=True,
    )
finally:
    LocalChatCompletion.amodel_call = original_amodel

dispatch = read_jsonl(OUT / "request_dispatch.jsonl")
assert len(dispatch) == 2 and dispatch[0]["event"] == "dispatch" and dispatch[1]["event"] == "restored"
permutation = dispatch[0]["dispatch_to_original"]
plan = json.loads((HOOKS / "frozen_runtime_estimates.json").read_text())
expected_order = sorted(range(384), key=lambda i: (-plan["jobs"][i]["estimated_completion_tokens"], i))
assert permutation == expected_order
assert dispatch[0]["algorithm"] == plan["algorithm"]
assert (
    hashlib.sha256((OUT / "runtime_estimates.json").read_bytes()).hexdigest() == dispatch[0]["runtime_estimates_sha256"]
)
assert sorted(permutation) == list(range(384))
assert dispatch[0]["task_counts"] == {"gsm8k_cot_zeroshot": 256, "triviaqa": 128}
before = read_jsonl(OUT / "upstream_inputs_before_identity.jsonl")
after = read_jsonl(OUT / "upstream_inputs.jsonl")
identities = read_jsonl(OUT / "request_identity.jsonl")
wire = read_jsonl(OUT / "api_inputs.jsonl")
assert len(before) == len(after) == len(identities) == len(wire) == 384
assert sum(x["serialization_changed"] for x in identities) == 56
assert len({x["effective_request_sha256"] for x in identities}) == 384
assert state["started"] != state["completed"] and state["max_active"] == 32
assert state["started_request_hashes"] == [identities[i]["effective_request_sha256"] for i in permutation]
# Exercise the production audit unchanged, including every duplicate occurrence.
audit_spec = importlib.util.spec_from_file_location("dispatch_input_audit", ROOT / "doc/benchmark/audit_inputs.py")
audit_module = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(audit_module)
input_audit = audit_module.audit_request_identity(OUT, wire)
assert input_audit["requests"] == 384 and input_audit["generation_unchanged"]

wire_equal = []
for a, b in zip(before, after):
    assert (a["task"], a["doc_id"], a["repeats"]) == (b["task"], b["doc_id"], b["repeats"])
    assert a["args"][1] == b["args"][1]
    assert json.loads(a["args"][0][0]) == json.loads(b["args"][0][0])
    pa = hook.create_payload(
        type("PayloadOnly", (), {"model": "SYNTHETIC-TRANSPORT-ONLY", "_max_gen_toks": 8192})(),
        json.loads(a["args"][0][0]),
        generate=True,
        gen_kwargs=copy.deepcopy(a["args"][1]),
        seed=1234,
    )
    pb = hook.create_payload(
        type("PayloadOnly", (), {"model": "SYNTHETIC-TRANSPORT-ONLY", "_max_gen_toks": 8192})(),
        json.loads(b["args"][0][0]),
        generate=True,
        gen_kwargs=copy.deepcopy(b["args"][1]),
        seed=1234,
    )
    assert pa == pb
    wire_equal.append(dict(task=a["task"], local_doc_id=a["doc_id"], payload_sha256=digest(pa)))
# Wire uses the evaluator's configured seed; compare exact original/effective payloads above,
# and compare all other fields against the actual async payload recording.
expected_payloads = []
for a in before:
    p = hook.create_payload(
        type("PayloadOnly", (), {"model": "SYNTHETIC-TRANSPORT-ONLY", "_max_gen_toks": 8192})(),
        json.loads(a["args"][0][0]),
        generate=True,
        gen_kwargs=copy.deepcopy(a["args"][1]),
        seed=1234,
    )
    expected_payloads.append(digest(p))
assert sorted(expected_payloads) == sorted(digest(x) for x in wire)
checks = []
for group in CONFIG["tasks"]:
    path = OUT / group
    responses = read_jsonl(path / "responses.jsonl")
    links = read_jsonl(path / "request_links.jsonl")
    assert len(responses) == len(links) == manifest["groups"][group]["sample_count"]
    bydoc = {}
    for response, link in zip(responses, links):
        assert response["id"] == link["response_id"]
        assert link["group"] == group
        assert (link["task"], link["doc_id"]) not in bydoc
        bydoc[link["task"], link["doc_id"]] = (link, scoring_response(response)[0]["choices"][0]["message"]["content"])
    for task in manifest["groups"][group]["tasks"]:
        samples = read_jsonl(path / f"samples_{task}.jsonl")
        frozen = manifest["tasks"][task]
        hashes = dict(zip(frozen["indices"], frozen["document_sha256"]))
        filters = {x["filter"] for x in samples}
        for filt in filters:
            assert sorted(x["doc_id"] for x in samples if x["filter"] == filt) == frozen["indices"]
        for row in samples:
            link, content = bydoc[task, row["doc_id"]]
            assert digest(row["doc"]) == hashes[row["doc_id"]]
            assert len(row["arguments"]) == 1 and digest(row["arguments"][0]) == link["request_sha256"]
            assert row["resps"] == [[content]]
        checks.append(
            dict(
                task=task,
                documents=len(frozen["indices"]),
                filters=sorted(filters),
                scored_rows=len(samples),
                document_hashes_exact=True,
                request_response_scored_arguments_exact=True,
            )
        )
summary = dict(
    status="passed",
    scope="SYNTHETIC CPU transport integration; scores are NOT model results",
    requests=384,
    serialization_changes=56,
    distinct_original_identities=len({x["original_request_sha256"] for x in identities}),
    distinct_effective_identities=384,
    wire_payloads_exact=True,
    completion_out_of_order=True,
    max_concurrency=state["max_active"],
    checks=checks,
    elapsed_seconds=time.monotonic() - started,
    manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
    helper_sha256=hashlib.sha256((HOOKS / "request_identity.py").read_bytes()).hexdigest(),
    dispatch_sha256=hashlib.sha256((HOOKS / "task_dispatch.py").read_bytes()).hexdigest(),
    runtime_estimate_order_exact=True,
    algorithm=dispatch[0]["algorithm"],
    runtime_estimates_sha256=dispatch[0]["runtime_estimates_sha256"],
    original_scorer_order_restored=True,
    production_input_audit=input_audit,
)
(OUT / "payload_equality.json").write_text(json.dumps(wire_equal, indent=2) + "\n")
(OUT / "completion_order.json").write_text(json.dumps(state, indent=2) + "\n")
(OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
