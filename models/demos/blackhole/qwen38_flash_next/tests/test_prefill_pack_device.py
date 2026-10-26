# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P3d-2 device tests: the multi-user packed slab body (``ttnn/pack.py``) on the single-slot adapter's chain.

``leak_<S>``: cross-user independence, bitwise.  Pack ``(A, B, ...)`` against ``(A, B', ...)`` (the other users' tokens AND
lengths differ), against a pack with only A and empty slots, and against a pack whose empty slots hold other tokens: user A's
logits and every landing source of A (each GDN layer's final recurrent state and last three conv rows, the 12 QSA layers'
K/V rows, compressed rows and kept ``[v|k]`` / raw-key rows over A's whole segment, the PLE snapshot) are compared as int bit
patterns on all four devices.  The other users' own outputs must change (sensitivity).
``slot_<S>``: the same user at slot 0, a middle slot and the last slot (rows re-based): bit-identical logits and sources.
``numerics``: first-token logits of 16+ prompts of 20..1000 tokens, packed by bucket, against the single-user production path
(chain prefill + forced step): top-1 agreement, max abs and KL (a tolerance-class comparison).
``timing``: replay wall time per bucket against a 2048-row slab replay, host input prep per pack, DRAM before / after.

Run:  pytest .../test_prefill_pack_device.py -v -s -k leak_s128   (one id per process)
Env:  as ``test_prefill_padded_pass_device`` (QWEN38_HARDWARE_PROFILE, TT_METAL_TRACE_ALLOC_TRACKING=1, MODEL_WEIGHTS_DIR,
      QWEN38_CACHE_ROOT, TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0).  ``P3D2_OUT`` names a directory for the JSON results.
"""

import contextlib
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_flash_next.demo.long_prompt import load_context_text
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    PROMPT_FILE,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_batch_contract_device import initialize
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import chain_warm_hook
from models.demos.blackhole.qwen38_flash_next.ttnn import pack as pack_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import Qwen38TTNNLayerType

OUT = os.environ.get("P3D2_OUT")
BUCKETS = {"s128": 128, "s256": 256, "s512": 512, "s1024": 1024}
VOCAB = TOKENIZER_SIZE


def save(name: str, payload) -> None:
    if OUT:
        (Path(OUT) / name).write_text(json.dumps(payload, indent=1, default=str))


def dram(mesh) -> dict:
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    banks = int(view.num_banks)
    return {
        "free_bytes_per_device": banks * int(view.total_bytes_free_per_bank),
        "allocated_bytes_per_device": banks * int(view.total_bytes_allocated_per_bank),
    }


def mib(n: int) -> float:
    return round(n / 2**20, 1)


# ----------------------------------------------------------------------------------------------------- the prompts


class Prompts:
    """Text prompts of chosen lengths: chat-rendered book slices (the demo's context text), distinct per call."""

    def __init__(self, checkpoint: Path) -> None:
        self.template = Qwen38OfficialChatTemplate(checkpoint)
        config = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
        text = load_context_text(config, Path(os.environ["QWEN38_CACHE_ROOT"]) / "demo-context")
        self.tokenizer = self.template.tokenizer
        self.book = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        self.cursor = 5000
        self.overhead = len(self.render(self.book[:1]))

    def render(self, ids: list[int]) -> list[int]:
        content = self.tokenizer.decode(ids)
        text = self.template.render([{"role": "user", "content": content}], enable_thinking=False).text
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def make(self, length: int, seed: int | None = None) -> list[int]:
        """A chat prompt of about ``length`` tokens (exactly, after trimming a longer render at the head's end)."""

        start = self.cursor if seed is None else 5000 + 977 * seed
        self.cursor += 613
        for guess in range(max(1, length - self.overhead), length + 8):
            ids = self.render(self.book[start : start + guess])
            if len(ids) >= length:
                return ids[:length] if len(ids) > length + 2 else ids
        raise RuntimeError(f"no prompt of {length} tokens")


# ----------------------------------------------------------------------------------------------------------- harness


class PackHarness:
    """One adapter (single slot) whose chain open runs the pack's allocation and eager warm pass as its warm hook: the
    persistent pack buffers must exist before the chain captures its traces (a buffer allocated after a capture can sit on
    a trace's intermediates and is overwritten by that trace's replays); the pack traces are captured after the open."""

    def __init__(self, mesh_device, checkpoint: Path, buckets: tuple[int, ...]) -> None:
        self.mesh = mesh_device
        self.checkpoint = checkpoint
        self.buckets = tuple(buckets)
        mesh_device.enable_program_cache()
        self.dram = {}
        held = {}

        def hook(chain) -> None:
            tm = chain.built_target.model
            self.dram["chain"] = dram(mesh_device)
            started = time.time()
            held["pack"] = tm.allocate_pack_state(chain.state, chain.slab_state, buckets=self.buckets)
            self.dram["pack_state"] = dram(mesh_device)
            tokens = [[(7 * i + 13) % 5000 + 1 for i in range(40)]]
            for rows in self.buckets:  # eager warm: every program of the body compiles here, misses allowed
                tm.run_pack(held["pack"], rows, chain.state, tokens)
            held["warm_s"] = time.time() - started

        started = time.time()
        with chain_warm_hook(hook):
            self.model = initialize(mesh_device, checkpoint, 1)
        self.model.allocate_kv_cache((1, 1, 64, 128), None, 12)
        self.open_s = time.time() - started
        self.warm_s = held["warm_s"]
        self.pack = held["pack"]
        self.chain = self.model.chain
        self.tm = self.chain.built_target.model
        self.state = self.chain.state
        self.slab = self.chain.slab_state
        assert self.slab is not None and self.slab.rows == pack_module.PACK_ROWS, getattr(self.slab, "rows", None)
        self.dram["after_open"] = dram(mesh_device)
        self.traces: dict[int, tuple[int, object]] = {}
        self.dram["bucket_traces"] = {}
        for rows in self.buckets:
            self.capture(rows)
        # the host reads below compile their own programs
        mesh_device.set_program_cache_misses_allowed(True)
        self.layers = self.tm.layers
        self.cache_layers = [i for i, layer in enumerate(self.layers) if layer.layer_type is Qwen38TTNNLayerType.QSA]
        self.gdn_layers = [i for i, layer in enumerate(self.layers) if layer.layer_type is Qwen38TTNNLayerType.GDN]

    def capture(self, rows: int) -> None:
        """Capture the bucket's trace (the body was warmed eagerly in the chain's warm hook) and verify it."""

        before = dram(self.mesh)
        started = time.time()
        self.traces[rows] = self.tm.capture_prefill_pack(
            self.pack, rows, self.state, guard=lambda label: contextlib.nullcontext()
        )
        after = dram(self.mesh)
        TraceAllocationTracker.verify_before_replay(self.mesh, self.traces[rows][0])
        self.dram["bucket_traces"][rows] = {
            "before_capture": before,
            "after_capture": after,
            "trace_bytes_per_device": before["free_bytes_per_device"] - after["free_bytes_per_device"],
            "capture_s": round(time.time() - started, 2),
        }
        logger.info(f"pack trace S={rows}: {self.dram['bucket_traces'][rows]}")

    def run(self, rows: int, users, *, timing=None, pad_token=pack_module.PACK_PAD_TOKEN_ID, fill=None):
        trace_id, output = self.traces[rows]
        return self.tm.run_pack(
            self.pack, rows, self.state, users, trace_id=trace_id, output=output, pad_token=pad_token, timing=timing
        )

    # -- the landing sources, as digests per (family, layer, device, user) ------------------------------------

    @staticmethod
    def digest(tensor: torch.Tensor) -> str:
        data = tensor.contiguous()
        return hashlib.blake2b(data.view(torch.uint8).numpy().tobytes(), digest_size=12).hexdigest()

    def sources(self, rows: int, count: int, lengths: list[int] | None = None) -> dict:
        """Digests of every user's landing sources after the last replay: ``{(family, layer, device, user): digest}`` for
        the first ``count`` users (the users' whole segments for the cache rows)."""

        bucket = self.pack.buckets[rows]
        users = list(range(count))
        found: dict = {}
        ttnn.synchronize_device(self.mesh)

        def devices(tensor):
            return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(tensor)]

        def head(tensor, row_end):
            sliced = ttnn.slice(
                tensor, (0, 0, 0, 0), (1, 1, row_end, int(tensor.shape[-1])), memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            out = devices(sliced)
            ttnn.deallocate(sliced)
            return out

        for index in self.gdn_layers:
            layer_pack = self.pack.layers[index]
            states = [devices(t) for t in layer_pack.states]
            conv = devices(layer_pack.conv_rows)
            for device in range(4):
                for user in users:
                    launch, head0 = pack_module.gdn_state_slot(bucket, user)
                    found[("gdn_state", index, device, user)] = self.digest(
                        states[launch][device][head0 : head0 + 12].reshape(12, 128, 128)
                    )
                    found[("gdn_conv", index, device, user)] = self.digest(
                        conv[device].reshape(-1, conv[device].shape[-1])[32 * user : 32 * user + 32]
                    )
        for index in self.cache_layers:
            attention = self.state.layers[index].attention
            kept = self.slab.layers[index].attention
            for family, tensor, per_user in (
                ("kv", attention.packed_kv_cache, rows),
                ("compressed", attention.compressed_index_cache, rows // 4),
                ("kept_kv", kept.kept_kv, rows),
                ("kept_raw", kept.kept_raw, rows),
            ):
                total = (
                    pack_module.PACK_ROWS
                    if per_user == rows or family.startswith("kept")
                    else pack_module.PACK_ROWS // 4
                )
                host = head(tensor, total) if family in ("kv", "compressed") else devices(tensor)
                for device in range(4):
                    flat = host[device].reshape(-1, host[device].shape[-1])
                    for user in users:
                        found[(family, index, device, user)] = self.digest(
                            flat[user * per_user : (user + 1) * per_user]
                        )
        ple_index = 1
        for user in users:
            snapshot = devices(self.pack.ple_snapshots[user])
            for device in range(4):
                found[("ple", ple_index, device, user)] = self.digest(snapshot[device])
        return found


def diff_sources(a: dict, b: dict, users_a: list[int], users_b: list[int]) -> dict:
    """Mismatching keys between the sources of user ``users_a[i]`` in ``a`` and ``users_b[i]`` in ``b``, by family."""

    bad: dict[str, int] = {}
    total = 0
    for ua, ub in zip(users_a, users_b):
        for (family, layer, device, user), digest in a.items():
            if user != ua:
                continue
            total += 1
            if b[(family, layer, device, ub)] != digest:
                bad[family] = bad.get(family, 0) + 1
    return {"compared": total, "mismatching_by_family": bad}


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


# ------------------------------------------------------------------------------------------------------- the tests


@pytest.fixture
def harness_factory(mesh_device):
    checkpoint = Path(os.environ["MODEL_WEIGHTS_DIR"])
    built = []

    def make(buckets):
        h = PackHarness(mesh_device, checkpoint, tuple(buckets))
        built.append(h)
        return h

    yield make, checkpoint
    for h in built:
        h.model.release_persistent_capture()


def lengths_for(rows: int, count: int, seed: int, low: int = 3) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(max(low, rows // 8), rows) for _ in range(count)]


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "bucket", [pytest.param(name, id=f"leak_{name}") for name in ("s128", "s256", "s512", "s1024")]
)
@pytest.mark.timeout(3600)
def test_pack_cross_user_leak(mesh_device, harness_factory, bucket):
    make, checkpoint = harness_factory
    rows = BUCKETS[bucket]
    h = make((rows,))
    prompts = Prompts(checkpoint)
    g = h.pack.buckets[rows].users
    a_slot = 1 if g > 1 else 0
    p_a = max(8, (rows * 3) // 4 + 3)

    def users(seed, a_tokens, a_length_fill=None, count=None, empty_fill=None):
        """A pack of ``count`` users with A at ``a_slot`` and others random (seed)."""

        count = g if count is None else count
        out = []
        rng = random.Random(seed)
        for slot in range(count):
            if slot == a_slot:
                out.append(a_tokens)
            else:
                out.append(prompts.make(rng.randint(max(20, rows // 8), rows), seed=seed * 100 + slot)[:rows])
        return out

    a_tokens = prompts.make(p_a, seed=7)[:p_a]
    assert len(a_tokens) == p_a
    results = {"bucket": rows, "users": g, "a_slot": a_slot, "a_tokens": p_a}

    def run(label, packs, **kwargs):
        timing = {}
        logits = h.run(rows, packs, timing=timing, **kwargs)
        sources = h.sources(rows, len(packs))
        logger.info(f"{label}: ran {len(packs)} users, replay {timing['replay_ms']:.1f} ms")
        return logits, sources, packs

    base = run("base", users(1, a_tokens))
    other = run("other users differ", users(2, a_tokens))
    alone = run("A with empty slots", users(3, a_tokens, count=a_slot + 1))
    results["lengths_base"] = [len(u) for u in base[2]]
    results["lengths_other"] = [len(u) for u in other[2]]

    def check(name, left, right):
        logit_equal = bits_equal(left[0][a_slot], right[0][a_slot])
        sources = diff_sources(left[1], right[1], [a_slot], [a_slot])
        results[name] = {"logits_bitwise": logit_equal, **sources}
        logger.info(f"RESULT {bucket} {name}: {results[name]}")
        return logit_equal and not sources["mismatching_by_family"]

    ok = [check("A: base vs other users", base, other), check("A: base vs empty slots", base, alone)]
    # empty slots hold other tokens (their rows are masked as empty): A must not change
    junk = lambda: [[(31 * i + 977) % 150000 + 5 for i in range(rows)] for _ in range(g)]  # noqa: E731
    original = pack_module.pack_token_rows

    def with_junk_fill(bucket_, users_, pad_token=pack_module.PACK_PAD_TOKEN_ID):
        padded = original(bucket_, users_, pad_token)
        filler = junk()
        for slot in range(len(users_), bucket_.users):
            padded[slot] = filler[slot]
        return padded

    pack_module.pack_token_rows = with_junk_fill
    try:
        fill = run("A with junk in the empty slots", users(3, a_tokens, count=a_slot + 1))
    finally:
        pack_module.pack_token_rows = original
    ok.append(check("A: empty slots with other tokens", alone, fill))
    # sensitivity: the other users' own outputs change when their tokens change
    changed = [slot for slot in range(g) if slot != a_slot and not bits_equal(base[0][slot], other[0][slot])]
    source_changed = diff_sources(
        base[1], other[1], [s for s in range(g) if s != a_slot], [s for s in range(g) if s != a_slot]
    )
    results["sensitivity"] = {
        "other_users_with_changed_logits": len(changed),
        "other_users": g - 1,
        "source_digests_changed": source_changed,
    }
    logger.info(f"RESULT {bucket} sensitivity: {results['sensitivity']}")
    # repeat determinism: the same pack twice
    repeat = run("repeat of base", users(1, a_tokens))
    ok.append(check("A: base vs repeat", base, repeat))
    results["repeat_all_users_logits_bitwise"] = all(bits_equal(base[0][s], repeat[0][s]) for s in range(g))
    save(f"leak_{bucket}.json", results)
    assert all(ok), results
    if g > 1:
        assert len(changed) == g - 1, results["sensitivity"]
    assert results["repeat_all_users_logits_bitwise"]


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "bucket", [pytest.param(name, id=f"slot_{name}") for name in ("s128", "s256", "s512", "s1024")]
)
@pytest.mark.timeout(3600)
def test_pack_slot_independence(mesh_device, harness_factory, bucket):
    make, checkpoint = harness_factory
    rows = BUCKETS[bucket]
    h = make((rows,))
    prompts = Prompts(checkpoint)
    g = h.pack.buckets[rows].users
    p_a = max(8, (rows * 5) // 8 + 1)
    a_tokens = prompts.make(p_a, seed=11)[:p_a]
    slots = sorted({0, g // 2, g - 1}) if g > 2 else list(range(g))
    results = {"bucket": rows, "slots": slots, "a_tokens": p_a}
    runs = {}
    for slot in slots:
        rng = random.Random(100 + slot)
        users = [
            a_tokens if s == slot else prompts.make(rng.randint(20, rows), seed=500 + 10 * slot + s)[:rows]
            for s in range(g)
        ]
        logits = h.run(rows, users)
        runs[slot] = (logits, h.sources(rows, g))
    first = slots[0]
    ok = True
    for slot in slots[1:]:
        equal = bits_equal(runs[first][0][first], runs[slot][0][slot])
        sources = diff_sources(runs[first][1], runs[slot][1], [first], [slot])
        results[f"slot {first} vs {slot}"] = {"logits_bitwise": equal, **sources}
        logger.info(f"RESULT {bucket} slot {first} vs {slot}: {results[f'slot {first} vs {slot}']}")
        ok = ok and equal and not sources["mismatching_by_family"]
    save(f"slot_{bucket}.json", results)
    assert ok, results


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(3600)
def test_pack_pad_content_and_bucket_size(mesh_device, harness_factory):
    """Diagnostics for the numerics gap to the single-user path: (a) the pad token of the pad rows (three values) changes
    nothing in a user's real rows (logits, GDN state, conv rows, PLE snapshot, K/V rows of the real tokens); (b) the same
    prompt in a smaller and a larger bucket (other pad counts, other S per segment) gives logits whose KL is the pad / bucket
    sensitivity of the pack itself."""

    make, checkpoint = harness_factory
    h = make((256, 512, 1024))
    prompts = Prompts(checkpoint)
    results = {}
    for length in (200, 300):
        tokens = prompts.make(length, seed=61 + length)[:length]
        rows = pack_module.bucket_for_length(length)
        runs = {}
        for pad in (0, 151643, 12345):
            logits = h.run(rows, [tokens], pad_token=pad)
            runs[pad] = (logits[0], h.sources(rows, 1))
        base = runs[0]
        entry = {"bucket": rows}
        for pad in (151643, 12345):
            other = runs[pad]
            diff = diff_sources(base[1], other[1], [0], [0])["mismatching_by_family"]
            entry[f"pad {pad} vs 0"] = {
                "logits_bitwise": bits_equal(base[0], other[0]),
                "max_abs": float((base[0][:VOCAB] - other[0][:VOCAB]).abs().max()),
                "families_differing": diff,
            }
        by_bucket = {}
        for bucket_rows in (256, 512, 1024):
            if bucket_rows >= length:
                by_bucket[bucket_rows] = h.run(bucket_rows, [tokens])[0]
        entry["bucket_vs_bucket"] = {}
        keys = sorted(by_bucket)
        for a, b in zip(keys, keys[1:]):
            entry["bucket_vs_bucket"][f"{a} vs {b}"] = numerics_row(by_bucket[a], by_bucket[b])
        results[str(length)] = entry
        logger.info(f"RESULT padcheck {length}: {entry}")
    save("padcheck.json", results)


def cache_and_state_rows(h: PackHarness, rows: int) -> dict:
    """The chain's own state after a single-user prefill, per device on the host: the first ``rows`` K/V rows and ``rows // 4``
    compressed rows of the 12 QSA layers and the 36 GDN layers' recurrent state."""

    ttnn.synchronize_device(h.mesh)
    found: dict = {}

    def head(tensor, count):
        sliced = ttnn.slice(
            tensor, (0, 0, 0, 0), (1, 1, count, int(tensor.shape[-1])), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(sliced)]
        ttnn.deallocate(sliced)
        return out

    for index in h.cache_layers:
        attention = h.state.layers[index].attention
        found[("kv", index)] = head(attention.packed_kv_cache, rows)
        found[("compressed", index)] = head(attention.compressed_index_cache, rows // 4)
    for index in h.gdn_layers:
        found[("gdn_state", index)] = [
            ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(h.state.layers[index].attention.recurrent)
        ]
    return found


def pack_user_rows(h: PackHarness, rows: int, user: int, count: int) -> dict:
    """The same quantities of one pack user after ``run``: its segment's first ``count`` K/V rows, its compressed rows and its
    recurrent states (the layouts of ``ttnn/pack.py``)."""

    bucket = h.pack.buckets[rows]
    ttnn.synchronize_device(h.mesh)
    found: dict = {}

    def head(tensor, first, n):
        sliced = ttnn.slice(
            tensor, (0, 0, first, 0), (1, 1, first + n, int(tensor.shape[-1])), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(sliced)]
        ttnn.deallocate(sliced)
        return out

    for index in h.cache_layers:
        attention = h.state.layers[index].attention
        found[("kv", index)] = head(attention.packed_kv_cache, user * rows, count)
        found[("compressed", index)] = head(attention.compressed_index_cache, user * rows // 4, count // 4)
    launch, head0 = pack_module.gdn_state_slot(bucket, user)
    for index in h.gdn_layers:
        buffers = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(h.pack.layers[index].states[launch])]
        found[("gdn_state", index)] = [b[head0 : head0 + 12].reshape(1, 12, 128, 128) for b in buffers]
    return found


def relative_errors(reference: dict, other: dict) -> dict:
    """Per (family, layer) the Frobenius error over the devices, relative to the reference's norm."""

    out = {}
    for key, devices in reference.items():
        num = sum(float((a - b).norm() ** 2) for a, b in zip(devices, other[key]))
        den = sum(float(a.norm() ** 2) for a in devices)
        out[key] = (num / den) ** 0.5 if den else 0.0
    return out


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(3600)
def test_pack_layer_error_profile(mesh_device, harness_factory):
    """Where the pack differs from the single-user path: the relative error of every QSA layer's K/V rows and compressed rows
    and of every GDN layer's final recurrent state, a pack user against the chain after the single-user prefill (cost and
    legacy plans; the plans against each other give the noise floor of the tolerance class)."""

    make, checkpoint = harness_factory
    h = make((128, 512, 1024))
    prompts = Prompts(checkpoint)
    results = {}
    for length, seed in ((300, 30), (90, 24), (700, 33)):
        tokens = prompts.make(length, seed=seed)[:length]
        rows = pack_module.bucket_for_length(length)
        entry = {}
        refs = {}
        for plan in ("legacy", "cost"):
            os.environ[driver_module.PLAN_ENV] = plan
            try:
                h.model.prefill_forward(
                    tokens=torch.tensor([tokens], dtype=torch.int32),
                    page_table=None,
                    kv_cache=None,
                    enable_trace=True,
                    prompt_lens=[length],
                    start_pos=[0],
                    empty_slots=[0],
                )
            finally:
                os.environ.pop(driver_module.PLAN_ENV, None)
            refs[plan] = cache_and_state_rows(
                h, length - 1
            )  # the head's rows: the last token's row is the forced step's
            h.model.release_request(0)
        h.run(rows, [tokens])
        packed = pack_user_rows(h, rows, 0, length - 1)
        for name, errors in (
            ("pack_vs_legacy", relative_errors(refs["legacy"], packed)),
            ("cost_vs_legacy", relative_errors(refs["legacy"], refs["cost"])),
        ):
            by_family: dict = {}
            for (family, layer), value in errors.items():
                by_family.setdefault(family, []).append((layer, round(value, 5)))
            entry[name] = {family: sorted(v) for family, v in by_family.items()}
        results[str(length)] = entry
        logger.info(f"PROFILE {length}: {json.dumps(entry)}")
    save("profile.json", results)


def edge_lengths(rows: int, users: int) -> list[int]:
    """Prompt lengths around every boundary the pack handles: one to ten tokens (conv history rows before the sequence
    start, PLE windows shorter than nine rows), the 4 / 32 / 128-row block edges, and a full segment."""

    if users <= 2:
        return [rows, 1][:users]  # a full segment (no pad rows) beside a one-token prompt
    wanted = [1, 3, 9, rows // 4 + 1, rows - 1, rows, 2, 8, rows // 4, rows // 2, 4, 5, 7, 10, rows // 4 - 1, rows - 5]
    seen: list[int] = []
    for n in wanted:
        if 1 <= n <= rows and n not in seen:
            seen.append(n)
    extra = [n for n in (11, 17, 31, 32, 33, 63, 65, 96, 127, 129, 255) if 1 <= n <= rows and n not in seen]
    return (seen + extra)[:users]


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("bucket", [pytest.param(name, id=f"edge_{name}") for name in ("s128", "s256", "s1024")])
@pytest.mark.timeout(3600)
def test_pack_edge_lengths_and_reversed_slots(mesh_device, harness_factory, bucket):
    """Every user at an edge length; then the same users in the reversed slot order: each user's logits and landing
    sources are bit-identical (also against the single-user path's top-1 for the prompts it takes)."""

    make, checkpoint = harness_factory
    rows = BUCKETS[bucket]
    h = make((rows,))
    prompts = Prompts(checkpoint)
    g = h.pack.buckets[rows].users
    lengths = edge_lengths(rows, g)
    users = []
    for slot, n in enumerate(lengths):
        full = prompts.make(max(n, 40), seed=300 + slot)
        users.append(full[:n])
    count = len(users)
    forward = h.run(rows, users)
    sources_forward = h.sources(rows, count)
    backward = h.run(rows, users[::-1])
    sources_backward = h.sources(rows, count)
    results = {"bucket": rows, "lengths": lengths, "finite": bool(torch.isfinite(forward).all())}
    bad = []
    for slot in range(count):
        other = count - 1 - slot
        logits_equal = bits_equal(forward[slot], backward[other])
        diff = diff_sources(sources_forward, sources_backward, [slot], [other])
        if not logits_equal or diff["mismatching_by_family"]:
            bad.append((slot, lengths[slot], logits_equal, diff))
    results["reversed_slot_mismatches"] = bad
    agree = []
    for slot, tokens in enumerate(users):
        if len(tokens) >= 2:
            single = first_token_logits_single(h, tokens)
            agree.append((len(tokens), int(single[:VOCAB].argmax()) == int(forward[slot][:VOCAB].argmax())))
    results["top1_vs_single"] = agree
    logger.info(f"RESULT edge {bucket}: {results}")
    save(f"edge_{bucket}.json", results)
    assert results["finite"] and not bad, results


def first_token_logits_single(h: PackHarness, prompt: list[int], plan: str = "cost") -> torch.Tensor:
    """The single-user production path: chain prefill of the head and the forced step of the last token (``plan`` the
    prefill plan: ``cost`` is the served default, ``legacy`` the 32 / 128-row chunks and slabs without a padded pass).
    """

    os.environ[driver_module.PLAN_ENV] = plan
    try:
        out, _ = h.model.prefill_forward(
            tokens=torch.tensor([prompt], dtype=torch.int32),
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[len(prompt)],
            start_pos=[0],
            empty_slots=[0],
        )
    finally:
        os.environ.pop(driver_module.PLAN_ENV, None)
    h.model.release_request(0)
    return out.reshape(-1).clone().float()


def numerics_row(single: torch.Tensor, packed: torch.Tensor) -> dict:
    a = single[:VOCAB]
    b = packed[:VOCAB]
    top = torch.topk(a, 2)
    pa = torch.log_softmax(a, dim=0)
    pb = torch.log_softmax(b, dim=0)
    kl = float((pa.exp() * (pa - pb)).sum())
    return {
        "top1_equal": int(a.argmax()) == int(b.argmax()),
        "single_top1": int(a.argmax()),
        "pack_top1": int(b.argmax()),
        "single_margin": float(top.values[0] - top.values[1]),
        "max_abs": float((a - b).abs().max()),
        "kl": kl,
    }


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(5400)
def test_pack_numerics_against_single_user(mesh_device, harness_factory):
    make, checkpoint = harness_factory
    h = make(pack_module.PACK_BUCKETS)
    prompts = Prompts(checkpoint)
    lengths = [20, 33, 47, 64, 90, 127, 128, 150, 200, 256, 300, 400, 511, 512, 600, 700, 800, 900, 1000, 1024, 75, 180]
    if os.environ.get("P3D2_NUM_LENGTHS"):
        lengths = [int(n) for n in os.environ["P3D2_NUM_LENGTHS"].split(",")]
    items = []
    for i, length in enumerate(lengths):
        prompt = prompts.make(length, seed=20 + i)
        prompt = prompt[:length]
        items.append((length, prompt))
    singles = []
    for _n, p in items:
        started = time.time()
        singles.append(first_token_logits_single(h, p))
        logger.info(f"single path: {len(p)} tokens, {time.time() - started:.2f} s")
    # a rerun of two of them: the noise floor of the single path
    rerun_at = (0, len(items) // 2)
    rerun = [first_token_logits_single(h, items[k][1]) for k in rerun_at]
    legacy = [first_token_logits_single(h, p, "legacy") for _n, p in items]
    rows = []
    groups: dict[int, list[int]] = {}
    for index, (_n, prompt) in enumerate(items):
        groups.setdefault(pack_module.bucket_for_length(len(prompt)), []).append(index)
    packed: dict[int, torch.Tensor] = {}
    for bucket_rows, indices in groups.items():
        g = h.pack.buckets[bucket_rows].users
        for first in range(0, len(indices), g):
            chunk = indices[first : first + g]
            logger.info(f"pack S={bucket_rows}: prompts {[len(items[i][1]) for i in chunk]}")
            logits = h.run(bucket_rows, [items[i][1] for i in chunk])
            for slot, i in enumerate(chunk):
                packed[i] = logits[slot]
    for index, (length, prompt) in enumerate(items):
        row = numerics_row(singles[index], packed[index])
        row.update(length=len(prompt), bucket=pack_module.bucket_for_length(len(prompt)))
        row["pack_vs_legacy"] = numerics_row(legacy[index], packed[index])
        row["single_vs_legacy"] = numerics_row(legacy[index], singles[index])
        rows.append(row)
        logger.info(f"NUMERICS {row}")
    floor = [numerics_row(singles[k], r) for k, r in zip(rerun_at, rerun)]
    agree = sum(r["top1_equal"] for r in rows)
    disagree = [r for r in rows if not r["top1_equal"]]
    summary = {
        "prompts": len(rows),
        "top1_agree": agree,
        "max_abs_max": max(r["max_abs"] for r in rows),
        "max_abs_median": statistics.median(r["max_abs"] for r in rows),
        "kl_max": max(r["kl"] for r in rows),
        "kl_median": statistics.median(r["kl"] for r in rows),
        "disagreements": disagree,
        "single_path_rerun_floor": floor,
        "pack_vs_legacy_top1_agree": sum(r["pack_vs_legacy"]["top1_equal"] for r in rows),
        "single_vs_legacy_top1_agree": sum(r["single_vs_legacy"]["top1_equal"] for r in rows),
        "pack_vs_legacy_kl_median": statistics.median(r["pack_vs_legacy"]["kl"] for r in rows),
        "single_vs_legacy_kl_median": statistics.median(r["single_vs_legacy"]["kl"] for r in rows),
        "pack_vs_legacy_kl_max": max(r["pack_vs_legacy"]["kl"] for r in rows),
        "single_vs_legacy_kl_max": max(r["single_vs_legacy"]["kl"] for r in rows),
        "pack_vs_legacy_max_abs_median": statistics.median(r["pack_vs_legacy"]["max_abs"] for r in rows),
        "single_vs_legacy_max_abs_median": statistics.median(r["single_vs_legacy"]["max_abs"] for r in rows),
    }
    logger.info(f"RESULT numerics: {json.dumps(summary, default=str)}")
    save("numerics.json", {"rows": rows, "summary": summary})
    assert all(not math.isnan(r["kl"]) for r in rows)
    assert all(r["single_margin"] < 0.5 for r in disagree), disagree  # a disagreement must be a near tie


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(5400)
def test_pack_timing_and_dram(mesh_device, harness_factory):
    make, checkpoint = harness_factory
    h = make(pack_module.PACK_BUCKETS)
    prompts = Prompts(checkpoint)
    results = {"open_s": round(h.open_s, 1), "dram": h.dram, "buckets": {}}
    free = lambda d: d["free_bytes_per_device"]  # noqa: E731
    results["dram_mib"] = {
        "pack_state": mib(free(h.dram["chain"]) - free(h.dram["pack_state"])),
        "per_bucket_trace": {str(r): mib(v["trace_bytes_per_device"]) for r, v in h.dram["bucket_traces"].items()},
        "all_four_traces": mib(free(h.dram["after_open"]) - free(dram(mesh_device))),
        "pack_state_plus_all_traces": mib(
            free(h.dram["chain"]) - free(h.dram["pack_state"]) + free(h.dram["after_open"]) - free(dram(mesh_device))
        ),
        "free_before_pack_hook": mib(free(h.dram["chain"])),
        "free_after_open": mib(free(h.dram["after_open"])),
        "free_after_all_captures": mib(free(dram(mesh_device))),
    }
    # the 2048-row slab replay (the chain's own trace), for comparison
    slab_ms = []
    tokens = prompts.make(pack_module.PACK_ROWS, seed=3)[: pack_module.PACK_ROWS]
    for _ in range(4):
        h.tm.reset_generic_state_inplace(h.state)
        h.tm.reset_chunk_state_inplace(h.state, h.slab)
        h.tm.write_chunk_inputs(h.slab, tokens, ple_context=None)
        ttnn.synchronize_device(mesh_device)
        started = time.perf_counter()
        ttnn._ttnn_execute_trace(mesh_device, h.chain.slab_trace_id, cq_id=0, blocking=True)
        slab_ms.append((time.perf_counter() - started) * 1e3)
    results["slab_2048_replay_ms"] = [round(x, 1) for x in slab_ms[1:]]
    results["slab_2048_replay_median_ms"] = round(statistics.median(slab_ms[1:]), 1)
    for rows in pack_module.PACK_BUCKETS:
        g = h.pack.buckets[rows].users
        users = [prompts.make(max(8, rows - 5 - 3 * s), seed=40 + s)[:rows] for s in range(g)]
        times = []
        for _ in range(4):
            timing = {}
            h.run(rows, users, timing=timing)
            times.append(timing)
        replay = [t["replay_ms"] for t in times[1:]]
        results["buckets"][str(rows)] = {
            "users": g,
            "replay_ms": [round(x, 1) for x in replay],
            "replay_median_ms": round(statistics.median(replay), 1),
            "prepare_ms": [round(t["prepare_ms"], 2) for t in times[1:]],
            "upload_ms": [round(t["upload_ms"], 2) for t in times[1:]],
            "prepare_parts_median_ms": {
                key: round(statistics.median(t[key] for t in times[1:]), 2)
                for key in ("tokens_ms", "ple_ms", "rope_ms", "images_ms")
            },
        }
        logger.info(f"TIMING S={rows}: {results['buckets'][str(rows)]}")
    logger.info(f"RESULT timing: {json.dumps(results, default=str)}")
    save("timing.json", results)
