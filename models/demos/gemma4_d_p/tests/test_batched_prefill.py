# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched prefill: several requests' chunks in one prefill step.

The requests' CP-local slabs stack request-major along rows; the dense ops run once on the stacked rows, and each
attention layer writes every request's KV into its own slot and attends each request's prefix with that request's
device metadata. One trace per lane count. Projections take one request's chunk fidelity, as the batched runtime does.

Env: G4B_DEEP=1 (correctness test: prefixes up to 32k), G4B_REPS (timed replays), PREFILL_TRACE_DIR (the GPU golden
prompt).
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4_d_p.demo.batched_steps import TRACE_REGION_SIZE, StepRunner, build_batched_model
from models.demos.gemma4_d_p.demo.text_demo_prefill import _hf_model_id, _text_token_stream
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.attention import ring_prefill
from models.demos.gemma4_d_p.tt.attention.ring_prefill import GlobalRingKVCache
from models.demos.gemma4_d_p.tt.runners.kv_validation import (
    BATCHED_MAX_OVERALL_RRMSE,
    BATCHED_MIN_OVERALL_PCC,
    EARLY_LAYERS,
    ERROR_BLOCK_ROWS,
    compare_head,
    read_cache_tensor,
    summarize_metrics,
)

# Batched KV vs the same request prefilled alone. In every layer below EARLY_LAYERS every head of every 2048-token
# block must match: clean steps reach 0.9965, while RoPE positions off by one in one lane of one step give 0.765.
# Deeper layers amplify summation-order differences chaotically (clean steps reach 0.906 min head at layer 57), so
# the whole cache only has to clear kv_validation's coarse floor.
MIN_EARLY_HEAD_PCC = 0.99
GOLDEN_DIR = Path(os.environ.get("PREFILL_TRACE_DIR", "/mnt/models/huggingface/gpu_traces/gemma4_d_p/gutenberg-135"))


def _lane(prompts, slot, start, width):
    """A step lane (slot, start, tokens): tokens [start, start + width) of the slot's prompt."""
    return (slot, start, prompts[slot][start : start + width])


# ── KV readback ────────────────────────────────────────────────────────────────


_staging = {}  # read_cache_tensor's host buffers, one per readback shape


def _read_slot(model, slot, n_tokens, chunk):
    """{layer: {config_id: [n_tokens, width]}} with kv_validation's config ids (global 0-3, sliding K 4-19, V 20-35)."""

    def read(tensor):
        return read_cache_tensor(tensor, slot, n_tokens, staging=_staging, chunk_size=chunk)

    layers = {}
    for layer_idx, layer in enumerate(model.layers):
        cache = layer.self_attn.ring_kv_cache
        if isinstance(cache, GlobalRingKVCache):
            heads = read(cache.kv)
            layers[layer_idx] = {h: heads[h] for h in range(heads.shape[0])}
        else:
            k, v = read(cache.k), read(cache.v)
            layers[layer_idx] = {
                **{4 + h: k[h] for h in range(k.shape[0])},
                **{20 + h: v[h] for h in range(v.shape[0])},
            }
    return layers


def _compare(expected, actual, rows=None):
    """KV PCC metric between two {layer: {config: tensor}} snapshots over rows (a slice, default all).

    Returns overall PCC / RRMSE, the worst head, the worst head of any ERROR_BLOCK_ROWS-row block in the layers below
    EARLY_LAYERS, per-layer PCC and the first layer that isn't bit-identical.
    """
    rows = rows or slice(None)
    comparisons, layer_pcc, worst, first_diff = [], [], None, None
    for layer in sorted(expected):
        layer_comparisons, identical = [], True
        for config_id, golden in expected[layer].items():
            result = compare_head(config_id, golden[rows], actual[layer][config_id][rows])
            for name, comparison in result.items():
                layer_comparisons.append(comparison)
                identical = identical and comparison.identical
                if worst is None or comparison.pcc < worst["pcc"]:
                    worst = dict(layer=layer, config=config_id, cache_type=name, pcc=comparison.pcc)
        if not identical and first_diff is None:
            first_diff = layer
        comparisons.extend(layer_comparisons)
        layer_pcc.append(round(summarize_metrics(layer_comparisons)["pcc"], 8))
    overall = summarize_metrics(comparisons)
    return dict(
        pcc=overall["pcc"],
        relative_rmse=overall["relative_rmse"],
        worst_head=worst,
        early_worst_head=_early_worst_head(expected, actual, rows),
        first_differing_layer=first_diff,
        layer_pcc=layer_pcc,
    )


def _early_worst_head(expected, actual, rows):
    """The lowest-PCC head of any ERROR_BLOCK_ROWS-row block of rows in the layers below EARLY_LAYERS."""
    start, stop, _ = rows.indices(next(iter(expected[0].values())).shape[0])
    worst = None
    for block in range(start, stop, ERROR_BLOCK_ROWS):
        block_rows = slice(block, min(block + ERROR_BLOCK_ROWS, stop))
        for layer in range(EARLY_LAYERS):
            for config_id, golden in expected[layer].items():
                result = compare_head(config_id, golden[block_rows], actual[layer][config_id][block_rows])
                for name, comparison in result.items():
                    if worst is None or comparison.pcc < worst["pcc"]:
                        worst = dict(rows=block, layer=layer, config=config_id, cache_type=name, pcc=comparison.pcc)
    return worst


def _gates(result):
    return (
        result["early_worst_head"]["pcc"] >= MIN_EARLY_HEAD_PCC
        and result["pcc"] >= BATCHED_MIN_OVERALL_PCC
        and result["relative_rmse"] < BATCHED_MAX_OVERALL_RRMSE
    )


def _fmt(result):
    w, e = result["worst_head"], result["early_worst_head"]
    return (
        f"min head in layers 0-{EARLY_LAYERS - 1} {e['pcc']:.6f} "
        f"(layer {e['layer']} {e['cache_type']}, rows {e['rows']}+) / "
        f"overall {result['pcc']:.6f} / RRMSE {result['relative_rmse']:.6f} / min head {w['pcc']:.6f} "
        f"(layer {w['layer']} {w['cache_type']}) / first differing layer {result['first_differing_layer']}"
    )


def _check(failures, label, expected, actual):
    """_compare, logged as a RESULT; a result that misses the gates is appended to failures."""
    result = _compare(expected, actual)
    logger.info(f"[batching] RESULT {label}: {_fmt(result)}")
    if not _gates(result):
        failures.append(f"{label}: {_fmt(result)}")


# ── Correctness ────────────────────────────────────────────────────────────────

# Chunks each request has before the batched steps: slot 0 (the GPU golden prompt) is deepest. G4B_DEEP=1 takes the
# deep set (prefixes to 32k, capacity 64k, 4 steps).
_PREFIX_CHUNKS = {2: (3, 0), 4: (3, 0, 1, 2)}
_DEEP_PREFIX_CHUNKS = {2: (16, 4), 4: (16, 4, 8, 12)}


def _poison(runner, slot, n_chunks, chunk, vocab_size):
    """Overwrite chunks 0..n_chunks-1 of slot with another prompt's KV, so a batched write that never lands leaves KV
    that fails the comparison instead of the reference KV it would have rewritten."""
    generator = torch.Generator().manual_seed(4321 + slot)
    tokens = torch.randint(0, vocab_size, (n_chunks * chunk,), dtype=torch.int32, generator=generator)
    for idx in range(n_chunks):
        runner.step([(slot, idx * chunk, tokens[idx * chunk : (idx + 1) * chunk])])


def _prompts(num_lanes, n_tokens, vocab_size):
    """Different prompt per slot: slot 0 the GPU golden prompt, then windows of the token stream, then random."""
    golden = json.loads((GOLDEN_DIR / "metadata.json").read_text())["token_ids"]
    stream = _text_token_stream(_hf_model_id())[0]
    prompts = [torch.tensor(golden[:n_tokens], dtype=torch.int32)]
    for offset in (60_000, 120_000)[: max(0, num_lanes - 2)]:
        prompts.append(stream[offset : offset + n_tokens].clone())
    if num_lanes > 1:
        generator = torch.Generator().manual_seed(1234)
        prompts.append(torch.randint(0, vocab_size, (n_tokens,), dtype=torch.int32, generator=generator))
    return prompts


@torch.no_grad()
@pytest.mark.timeout(900)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
@pytest.mark.parametrize("num_lanes", [2, 4], ids=lambda b: f"B{b}")
def test_batched_kv_matches_unbatched(mesh_device, num_lanes, reset_seeds):
    """Every slot's KV after batched steps vs the same request prefilled alone, chunk 2048. One more slot, which no
    batched step names, must stay bit-identical."""
    chunk = 2048
    deep = os.environ.get("G4B_DEEP") == "1"
    steps = 4 if deep else 2
    prefix_chunks = (_DEEP_PREFIX_CHUNKS if deep else _PREFIX_CHUNKS)[num_lanes]
    n_chunks = [p + steps for p in prefix_chunks]
    capacity = 65536 if deep else 32768
    assert max(n_chunks) * chunk <= capacity
    untouched = num_lanes
    mesh_config, model_args, model = build_batched_model(mesh_device, chunk, num_lanes + 1, capacity)
    prompts = _prompts(num_lanes, max(n_chunks) * chunk, model_args.vocab_size)
    prompts.append(prompts[0])

    def lane(slot, idx):
        return _lane(prompts, slot, idx * chunk, chunk)

    runner = StepRunner(mesh_device, mesh_config, model, chunk)
    runner.capture_all([[lane(0, 0)], [lane(slot, 0) for slot in range(num_lanes)]])

    def unbatched():
        for slot in range(num_lanes):
            for idx in range(n_chunks[slot]):
                runner.step([lane(slot, idx)])

    def batched():
        for slot in range(num_lanes):
            for idx in range(prefix_chunks[slot]):
                runner.step([lane(slot, idx)])
        for k in range(steps):
            runner.step([lane(slot, prefix_chunks[slot] + k) for slot in range(num_lanes)])

    t0 = time.time()
    unbatched()
    reference = [_read_slot(model, slot, n_chunks[slot] * chunk, chunk) for slot in range(num_lanes)]
    for idx in range(2):
        runner.step([lane(untouched, idx)])
    untouched_before = _read_slot(model, untouched, 2 * chunk, chunk)
    logger.info(f"[batching] reference run + readback {time.time() - t0:.0f}s")

    for slot in range(num_lanes):
        _poison(runner, slot, n_chunks[slot], chunk, model_args.vocab_size)
    failures = []
    t0 = time.time()
    batched()
    for slot in range(num_lanes):
        actual = _read_slot(model, slot, n_chunks[slot] * chunk, chunk)
        _check(failures, f"B={num_lanes} slot={slot} batched", reference[slot], actual)
        del actual
    logger.info(f"[batching] batched run + readback + compare {time.time() - t0:.0f}s")

    result = _compare(untouched_before, _read_slot(model, untouched, 2 * chunk, chunk))
    logger.info(f"[batching] RESULT B={num_lanes} untouched slot={untouched}: {_fmt(result)}")
    if result["first_differing_layer"] is not None:
        failures.append(f"untouched slot {untouched} changed: {_fmt(result)}")
    runner.release()
    assert not failures, "batched KV fails the gates:\n" + "\n".join(failures)


# ── Lanes sliding ring SDPA vs one call per request ────────────────────────────


# name -> (chunks each slot holds before the step, the step's lanes as (slot, chunk index)).
_LANES_SDPA_SCENARIOS = {
    "lane0_deepest": ((3, 0, 1, 2), [(0, 3), (1, 0), (2, 1), (3, 2)]),
    "lane0_empty": ((0, 3, 1, 2), [(0, 0), (1, 3), (2, 1), (3, 2)]),
    # One request's consecutive chunks in one step: lane k attends lanes 0..k-1 through the cache.
    "same_one_request": ((3, 0, 0, 0), [(0, 3), (0, 4), (0, 5), (0, 6)]),
    "same_two_by_two": ((2, 0, 0, 0), [(1, 0), (1, 1), (0, 2), (0, 3)]),
}


@torch.no_grad()
@pytest.mark.timeout(600)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
@pytest.mark.parametrize("scenario", list(_LANES_SDPA_SCENARIOS))
def test_lanes_sdpa_matches_per_request(mesh_device, scenario, reset_seeds, monkeypatch):
    """An eager 2k x 4 step with the sliding layers' attention as one lanes call vs one call per lane gives
    bit-identical hidden states and KV. Both use sliding K split 1, so the per-unit arithmetic matches.

    lane0_empty puts the shortest prefix on lane 0, whose prefix alone drives the call's ring-iteration masks."""
    prefix_chunks, step_chunks = _LANES_SDPA_SCENARIOS[scenario]
    chunk, num_lanes, capacity = 2048, 4, 32768
    mesh_config, model_args, model = build_batched_model(mesh_device, chunk, num_lanes, capacity)
    prompts = _prompts(num_lanes, capacity, model_args.vocab_size)
    # Sliding K split 1 in single-request calls too (a lanes call never splits K).
    chunk_sizes = ring_prefill.ring_sdpa_chunk_sizes

    def k_split_1(q_slab_tokens, sliding, **kwargs):
        q_chunk, k_chunk, k_splits, segmented = chunk_sizes(q_slab_tokens, sliding, **kwargs)
        return q_chunk, k_chunk, 1 if sliding else k_splits, segmented

    monkeypatch.setattr(ring_prefill, "ring_sdpa_chunk_sizes", k_split_1)

    def lane(slot, idx):
        return _lane(prompts, slot, idx * chunk, chunk)

    runner = StepRunner(mesh_device, mesh_config, model, chunk)
    for layout in ((chunk,), (chunk,) * num_lanes):
        runner._ensure(layout)

    def eager(lanes, lanes_sdpa=True, readback=True):
        runner.stage(lanes)
        out = runner._forward(runner.layout(lanes), lanes_sdpa=lanes_sdpa)
        host = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out.cpu())] if readback else None
        out.deallocate(True)
        return host

    for slot, n in enumerate(prefix_chunks):
        for idx in range(n):
            eager([lane(slot, idx)], readback=False)
    step = [lane(slot, idx) for slot, idx in step_chunks]
    filled = [max([n] + [idx + 1 for s, idx in step_chunks if s == slot]) for slot, n in enumerate(prefix_chunks)]
    results = {}
    for mode in (False, True):
        hidden = eager(step, lanes_sdpa=mode)
        kv = [_read_slot(model, slot, n * chunk, chunk) if n else None for slot, n in enumerate(filled)]
        results[mode] = (hidden, kv)
        logger.info(f"[batching] lanes SDPA={mode} step done")
    (h0, kv0), (h1, kv1) = results[False], results[True]
    hidden_equal = all(torch.equal(a, b) for a, b in zip(h0, h1))
    max_diff = max(float((a.float() - b.float()).abs().max()) for a, b in zip(h0, h1))
    logger.info(f"[batching] LANES hidden bit-identical={hidden_equal} max|diff|={max_diff:.3e}")
    for slot in (slot for slot, n in enumerate(filled) if n):  # untouched slots hold nothing to compare
        result = _compare(kv0[slot], kv1[slot])
        logger.info(f"[batching] LANES slot {slot} KV lanes vs per-request: {_fmt(result)}")
        assert result["first_differing_layer"] is None, f"slot {slot} KV differs: {_fmt(result)}"
    assert hidden_equal, f"final hidden states differ, max |diff| {max_diff}"


# ── Several chunks of one request in a step ────────────────────────────────────

# Scenarios: name -> steps; a step is [(slot, chunk index)]. Every scenario rewrites chunks 0..n-1 of its slots.
_SAME_SLOT_SCENARIOS = {
    # One 8k prompt as one step of four 2k lanes.
    "lone_8k": [[(0, 0), (0, 1), (0, 2), (0, 3)]],
    # Three chunks alone, then four lanes of the same request at prefix 6k.
    "lone_after_6k": [[(0, 0)], [(0, 1)], [(0, 2)], [(0, 3), (0, 4), (0, 5), (0, 6)]],
    # Two requests, two lanes each, for two steps.
    "two_by_two": [[(0, 0), (0, 1), (1, 0), (1, 1)], [(0, 2), (0, 3), (1, 2), (1, 3)]],
}


@torch.no_grad()
@pytest.mark.timeout(1200)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_same_request_lanes(mesh_device, reset_seeds):
    """Several lanes of one request in a step: KV vs the request prefilled alone.

    Every lane's KV write runs before any lane's attention, so lane k attends lanes 0..k-1 like earlier chunks.
    """
    chunk, num_lanes, num_slots = 2048, 4, 2
    capacity = 65536
    mesh_config, model_args, model = build_batched_model(mesh_device, chunk, num_slots, capacity)
    prompts = _prompts(2, capacity, model_args.vocab_size)
    prompts[1] = _text_token_stream(_hf_model_id())[0][60_000 : 60_000 + capacity].clone()

    def lane(slot, idx):
        return _lane(prompts, slot, idx * chunk, chunk)

    runner = StepRunner(mesh_device, mesh_config, model, chunk)
    runner.capture_all([[lane(0, 0)], [lane(0, i) for i in range(num_lanes)]])

    n_chunks = {name: {} for name in _SAME_SLOT_SCENARIOS}
    for name, steps in _SAME_SLOT_SCENARIOS.items():
        for step in steps:
            for slot, idx in step:
                n_chunks[name][slot] = max(n_chunks[name].get(slot, 0), idx + 1)
    deepest = {slot: max(n[slot] for n in n_chunks.values() if slot in n) for slot in range(num_slots)}

    for slot in range(num_slots):
        for idx in range(deepest[slot]):
            runner.step([lane(slot, idx)])
    reference = {slot: _read_slot(model, slot, deepest[slot] * chunk, chunk) for slot in range(num_slots)}

    failures = []
    for name, steps in _SAME_SLOT_SCENARIOS.items():
        for slot, n in n_chunks[name].items():
            _poison(runner, slot, n, chunk, model_args.vocab_size)
        for step in steps:
            runner.step([lane(slot, idx) for slot, idx in step])
        for slot, n in n_chunks[name].items():
            actual = _read_slot(model, slot, n * chunk, chunk)
            rows = slice(0, n * chunk)
            expected = {layer: {c: t[rows] for c, t in heads.items()} for layer, heads in reference[slot].items()}
            _check(failures, f"same-slot {name} slot={slot} vs alone", expected, actual)
            del actual
    runner.release()
    assert not failures, "same-request lanes fail the gates:\n" + "\n".join(failures)


# ── Perf ───────────────────────────────────────────────────────────────────────

# name -> each lane's prefix for a B-lane step
_MIXES = {
    "shallow": lambda b: [2048] * b,
    "u32k": lambda b: [32768] * b,
    "mixed": lambda b: [2048, 16384, 65536, 131072][:b],
    "deep": lambda b: [131072] * b,
}


@torch.no_grad()
@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_batched_step_perf(mesh_device, reset_seeds):
    """Traced step time of B requests x 2048 tokens vs B unbatched steps at the same prefixes."""
    chunk_size, lane_counts, capacity = 2048, (1, 2, 4), 262144
    num_slots = max(lane_counts)
    mesh_config, model_args, model = build_batched_model(mesh_device, chunk_size, num_slots, capacity)

    # Each slot gets its own window of real text, prefilled up to the deepest prefix any mix asks for, so the
    # cached prefixes hold real KV. A measured step rewrites its own rows with the same tokens.
    deepest = max(max(mix(num_slots)) for mix in _MIXES.values()) + chunk_size
    stream = _text_token_stream(_hf_model_id())[0]
    stride = min(100_000, (len(stream) - deepest) // max(1, num_slots - 1))
    assert stride > 0, f"token stream ({len(stream)}) too short for {num_slots} windows of {deepest}"
    seqs = [stream[slot * stride : slot * stride + deepest].clone() for slot in range(num_slots)]

    def lane(slot, start):
        return _lane(seqs, slot, start, chunk_size)

    runner = StepRunner(mesh_device, mesh_config, model, chunk_size)
    runner.capture_all([[lane(slot, 0) for slot in range(b)] for b in lane_counts])

    t0 = time.time()
    for slot in range(num_slots):
        for start in range(0, deepest - chunk_size, chunk_size):
            runner.step([lane(slot, start)])
    logger.info(f"[batching] filled {num_slots} slots to {deepest - chunk_size} tokens in {time.time() - t0:.0f}s")

    for mix_name, mix in _MIXES.items():
        for b in lane_counts:
            prefixes = mix(b)
            lanes = [lane(slot, p) for slot, p in enumerate(prefixes)]
            batched_ms = runner.timed(lanes)
            alone_ms = [runner.timed([single]) for single in lanes] if b > 1 else [batched_ms]
            logger.info(
                f"[batching] PERF chunk={chunk_size} B={b} mix={mix_name} prefixes={prefixes} "
                f"batched={batched_ms:.1f}ms unbatched_sum={sum(alone_ms):.1f}ms "
                f"ratio={batched_ms / sum(alone_ms):.3f} per_lane={[round(x, 1) for x in alone_ms]} "
                f"tok/s batched={b * chunk_size * 1000 / batched_ms:.0f} unbatched={b * chunk_size * 1000 / sum(alone_ms):.0f}"
            )
    runner.release()
