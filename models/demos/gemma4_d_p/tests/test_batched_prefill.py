# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Continuous-batching prefill PoC: several requests' chunks in one prefill step.

The requests' CP-local slabs stack request-major along rows; the dense ops run once on the stacked rows, and each
attention layer writes every request's KV into its own slot and runs one ring SDPA per request with that request's
device metadata. One trace per layout (each request's chunk width, in lane order).

    test_batched_kv_matches_unbatched  KV of every slot, batched vs each request alone (plus a repeat control and
                                       slot 0 against the GPU golden); G4B_DEEP=1 for prefixes up to 32k
    test_kv_window_write               update_padded_kv_cache with an input row window vs slice-then-write
    test_lanes_sdpa_matches_per_request  one lanes sliding SDPA vs one call per request: bit-identical (K split 1)
    test_same_request_lanes            several chunks of one request in one step vs the request alone
    test_partial_steps                 fewer requests than lanes: a smaller trace vs padding with a spare lane
    test_lane_widths                   per-request chunk widths in one step (e.g. 4k + 2k + 2k, a lone 8k)
    test_batched_step_perf             step time of B x chunk vs B unbatched steps at several prefix mixes

Env: G4B_OUT (result JSON dir), G4B_STEPS (batched steps in the correctness test), G4B_REPS (timed replays),
G4B_LANES (lane counts in the perf test), G4B_WIDTH_SCENARIOS / G4B_NO_PERF (widths test filters),
G4B_PROJECTION_FIDELITY=hifi2 (perf test: pin the projections at HiFi2), G4B_LANES_SDPA=0 (perf test: sliding
layers one call per request instead of one lanes call).
"""

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import models.demos.gemma4_d_p.tt.attention as attention
import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.demo.text_demo_prefill import _cp_or_replicate_mapper, _hf_model_id, _text_token_stream
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.attention import operations as attention_operations
from models.demos.gemma4_d_p.tt.attention import ring_prefill
from models.demos.gemma4_d_p.tt.attention.ring_prefill import GlobalRingKVCache
from models.demos.gemma4_d_p.tt.common import create_tt_model
from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillLanes
from models.demos.gemma4_d_p.tt.runners.kv_validation import compare_head, load_gpu_cache_heads, summarize_metrics

TRACE_REGION_SIZE = int(os.environ.get("GEMMA4_PREFILL_TRACE_REGION_SIZE", 600_000_000))
MIN_PER_HEAD_PCC = 0.91
MIN_OVERALL_PCC = 0.97
MAX_OVERALL_RRMSE = 0.232
GOLDEN_DIR = Path(os.environ.get("PREFILL_TRACE_DIR", "/mnt/models/huggingface/gpu_traces/gemma4_d_p/gutenberg-135"))


def _out_dir(name):
    directory = Path(os.environ.get("G4B_OUT", "generated/gemma4_batching")) / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_report(subdir, name, report):
    out = _out_dir(subdir) / name
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    logger.info(f"[batching] wrote {out}")


def _witness():
    import models.demos.gemma4_d_p.tt.model as model_module

    logger.info(f"[batching] witness ttnn={ttnn.__file__} models={model_module.__file__}")


class StepRunner:
    """Stacked inputs, per-lane metadata and one trace per layout (each lane's chunk width, in lane order) for a model
    with several KV slots. A lane is (slot, start, tokens); its width is len(tokens)."""

    def __init__(self, mesh_device, mesh_config, model, chunk_size):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.model = model
        self.chunk = chunk_size
        self.inputs = {}
        self.traces = {}
        self.outputs = {}
        # Mixed widths: every step, even one plain chunk, takes the PrefillLanes path (residual spilled across
        # attention). The first compile's all-gathers create their global semaphores lazily wherever L1 is free; with
        # the residual in L1 they land at ~1,440,576, under a 4k-wide global SDPA's CB region (ends at 1,475,712).
        self.always_lanes = False
        self.reps = int(os.environ.get("G4B_REPS", "5"))

    @staticmethod
    def layout(lanes):
        return tuple(len(tokens) for _, _, tokens in lanes)

    def _host(self, seqs):
        """Per-lane 1D int sequences -> the [1, total] host tensor whose CP shard is each rank's stacked slabs."""
        cp = self.mesh_config.cp_degree
        parts = []
        for rank in range(cp):
            for seq in seqs:
                slab = len(seq) // cp
                parts.append(seq[rank * slab : (rank + 1) * slab])
        return ttnn.from_torch(
            torch.cat(parts).reshape(1, -1).to(torch.int32).contiguous(),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=_cp_or_replicate_mapper(self.mesh_config, seq_dim=-1),
        )

    def _ensure(self, layout):
        total = sum(layout)
        if total not in self.inputs:
            zeros = [torch.zeros(total, dtype=torch.int32)]
            self.inputs[total] = tuple(ttnn.to_device(self._host(zeros), device=self.mesh_device) for _ in range(2))
        self.model.lane_prefill_metadata(len(layout))
        if len(layout) > 1:
            self.model.lane_vector_metadata(len(layout))

    def stage(self, lanes):
        """Host refresh of everything a replay reads."""
        tokens, positions = self.inputs[sum(self.layout(lanes))]
        ttnn.copy_host_to_device_tensor(self._host([t for _, _, t in lanes]), tokens)
        starts = [torch.arange(start, start + len(t)) for _, start, t in lanes]
        ttnn.copy_host_to_device_tensor(self._host(starts), positions)
        for metadata, (slot, start, _) in zip(self.model.lane_prefill_metadata(len(lanes)), lanes):
            metadata.update(slot_idx=slot, kv_actual_global=start)
        if len(lanes) > 1:
            self.model.lane_vector_metadata(len(lanes)).update(
                slot_idx=[slot for slot, _, _ in lanes], kv_actual_global=[start for _, start, _ in lanes]
            )

    def _forward(self, layout):
        tokens, positions = self.inputs[sum(layout)]
        self.model.set_prefill_rope_positions(positions)
        embeds = self.model.transform_and_embed_prefill_inputs_device(tokens)
        metadata = self.model.lane_prefill_metadata(len(layout))
        rows = [w // self.mesh_config.cp_degree for w in layout]
        if layout == (self.chunk,) and not self.always_lanes:
            metadata = metadata[0]
        else:
            # Each request's rows (one request wider than the model's chunk, or always_lanes, too), and for several
            # requests the B-element slot / prefix tensors the lanes ring SDPA reads.
            vector = self.model.lane_vector_metadata(len(layout)) if len(layout) > 1 else None
            metadata = PrefillLanes(metadata, rows, vector=vector)
        return self.model(hidden_states=embeds, prefill_metadata=metadata)

    def capture_all(self, lane_sets):
        """Compile every layout first, then capture every trace.

        A compile pass allocates persistent tensors lazily (gather-index constants, per-lane ring buffers). Allocated
        after another trace's capture, they can land on that trace's freed intermediates, and each replay of it then
        overwrites them. So no compile may follow a capture.
        """
        for lanes in lane_sets:
            layout = self.layout(lanes)
            self._ensure(layout)
            t0 = time.time()
            self.stage(lanes)
            out = self._forward(layout)
            ttnn.synchronize_device(self.mesh_device)
            out.deallocate(True)
            logger.info(f"[batching] layout={layout} compile={time.time() - t0:.1f}s")
        for lanes in lane_sets:
            layout = self.layout(lanes)
            self.stage(lanes)
            tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            self.outputs[layout] = self._forward(layout)
            ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
            ttnn.synchronize_device(self.mesh_device)
            self.traces[layout] = tid
            logger.info(f"[batching] layout={layout} captured")

    def step(self, lanes):
        """Stage and replay one step; returns its execute+sync time in ms."""
        self.stage(lanes)
        t0 = time.time()
        ttnn.execute_trace(self.mesh_device, self.traces[self.layout(lanes)], cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh_device)
        return (time.time() - t0) * 1000

    def timed(self, lanes):
        """Median replay time in ms over G4B_REPS replays, after one warm-up."""
        self.step(lanes)
        return statistics.median(self.step(lanes) for _ in range(self.reps))

    def release(self):
        for tid in self.traces.values():
            ttnn.release_trace(self.mesh_device, tid)
        self.traces = {}


def _build(mesh_device, chunk_size, num_slots, capacity):
    mesh_config = MeshConfig(mesh_device)
    _witness()
    model_args, model, _, _ = create_tt_model(
        mesh_config=mesh_config,
        prefill_chunk_size=chunk_size,
        max_batch_size=num_slots,
        max_seq_len=capacity,
        dtype=ttnn.bfloat16,
        hf_model_id=_hf_model_id(),
    )
    # Every lane's metadata and ring semaphores up front, before any activation is live (see lane_prefill_metadata).
    model.lane_prefill_metadata(num_slots)
    return mesh_config, model_args, model


# ── KV readback ────────────────────────────────────────────────────────────────


def _read_cache_heads(tensor, slot, widths, mesh_config):
    """One slot's first sum(widths) tokens of a CP-sharded chunk-major cache, as [heads, tokens, width] in token order.

    widths: the slot's chunk widths in order. Chunk i of width w takes w / CP local rows on every rank, and rank r's
    rows hold its tokens [r * w / CP, (r + 1) * w / CP) of the chunk.
    """
    cp, tp = mesh_config.cp_degree, mesh_config.tp_degree
    n_tokens = sum(widths)
    assert mesh_config.cp_axis == 0 and all(w % cp == 0 for w in widths)
    selected = ttnn.slice(
        tensor,
        (slot, 0, 0, 0),
        (slot + 1, tensor.shape[1], n_tokens // cp, tensor.shape[3]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    row_major = ttnn.untilize(selected, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(selected)
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(row_major.cpu())]
    ttnn.deallocate(row_major)
    local_heads, width = shards[0].shape[1], shards[0].shape[3]
    gathered = torch.empty((local_heads * tp, n_tokens, width), dtype=torch.bfloat16)
    for row in range(cp):
        for column in range(tp):
            shard = shards[row * tp + column][0]
            local, token = 0, 0
            for w in widths:
                slab = w // cp
                gathered[
                    column * local_heads : (column + 1) * local_heads, token + row * slab : token + (row + 1) * slab
                ].copy_(shard[:, local : local + slab])
                local, token = local + slab, token + w
    return gathered


def _read_slot(model, slot, n_tokens, chunk, mesh_config, widths=None):
    """{layer: {config_id: [n_tokens, width]}} with kv_validation's config ids (global 0-3, sliding K 4-19, V 20-35).

    widths: the slot's chunk widths when they differ from n_tokens / chunk chunks of chunk.
    """
    widths = widths or [chunk] * (n_tokens // chunk)
    assert sum(widths) == n_tokens
    layers = {}
    for layer_idx, layer in enumerate(model.layers):
        cache = layer.self_attn.ring_kv_cache
        if isinstance(cache, GlobalRingKVCache):
            heads = _read_cache_heads(cache.kv, slot, widths, mesh_config)
            layers[layer_idx] = {h: heads[h] for h in range(heads.shape[0])}
        else:
            k = _read_cache_heads(cache.k, slot, widths, mesh_config)
            v = _read_cache_heads(cache.v, slot, widths, mesh_config)
            layers[layer_idx] = {
                **{4 + h: k[h] for h in range(k.shape[0])},
                **{20 + h: v[h] for h in range(v.shape[0])},
            }
    return layers


def _compare(expected, actual, rows=None):
    """KV PCC metric between two {layer: {config: tensor}} snapshots over rows (a slice, default all).

    Returns overall PCC / RRMSE, the worst head, per-layer PCC and the first layer that isn't bit-identical.
    """
    rows = rows or slice(None)
    comparisons, layer_pcc, worst, first_diff = [], [], None, None
    for layer in sorted(expected):
        layer_comparisons, identical = [], True
        for config_id, golden in expected[layer].items():
            result = compare_head(config_id, golden[rows].float(), actual[layer][config_id][rows].float())
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
        first_differing_layer=first_diff,
        layer_pcc=layer_pcc,
    )


def _gates(result):
    return (
        result["pcc"] >= MIN_OVERALL_PCC
        and result["relative_rmse"] < MAX_OVERALL_RRMSE
        and result["worst_head"]["pcc"] >= MIN_PER_HEAD_PCC
    )


def _fmt(result):
    w = result["worst_head"]
    return (
        f"overall {result['pcc']:.6f} / RRMSE {result['relative_rmse']:.6f} / min head {w['pcc']:.6f} "
        f"(layer {w['layer']} {w['cache_type']}) / first differing layer {result['first_differing_layer']}"
    )


# ── Correctness ────────────────────────────────────────────────────────────────

# Chunks each request has before the batched steps: slot 0 (the GPU golden prompt) is deepest. G4B_DEEP=1 takes the
# deep set (prefixes to 32k, capacity 64k, 4 steps by default).
_PREFIX_CHUNKS = {2: (3, 0), 4: (3, 0, 1, 2)}
_DEEP_PREFIX_CHUNKS = {2: (16, 4), 4: (16, 4, 8, 12)}


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
@pytest.mark.timeout(10800)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
@pytest.mark.parametrize("fidelity", ["auto", "hifi2"])
@pytest.mark.parametrize("num_lanes", [2, 4], ids=lambda b: f"B{b}")
def test_batched_kv_matches_unbatched(mesh_device, num_lanes, fidelity, reset_seeds, monkeypatch):
    """Every slot's KV after batched steps vs the same request prefilled alone, chunk 2048."""
    if fidelity == "hifi2":
        monkeypatch.setattr(attention_operations, "PROJECTION_FIDELITY_OVERRIDE", ttnn.MathFidelity.HiFi2)
    chunk = 2048
    deep = os.environ.get("G4B_DEEP") == "1"
    steps = int(os.environ.get("G4B_STEPS", "4" if deep else "2"))
    prefix_chunks = (_DEEP_PREFIX_CHUNKS if deep else _PREFIX_CHUNKS)[num_lanes]
    n_chunks = [p + steps for p in prefix_chunks]
    capacity = 65536 if deep else 32768
    assert max(n_chunks) * chunk <= capacity
    mesh_config, model_args, model = _build(mesh_device, chunk, num_lanes, capacity)
    prompts = _prompts(num_lanes, max(n_chunks) * chunk, model_args.vocab_size)

    def lane(slot, idx):
        return (slot, idx * chunk, prompts[slot][idx * chunk : (idx + 1) * chunk])

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

    def snapshot():
        return [_read_slot(model, slot, n_chunks[slot] * chunk, chunk, mesh_config) for slot in range(num_lanes)]

    t0 = time.time()
    unbatched()
    reference = snapshot()
    logger.info(f"[batching] reference run + readback {time.time() - t0:.0f}s")

    report = dict(
        num_lanes=num_lanes,
        fidelity=fidelity,
        chunk=chunk,
        steps=steps,
        prefix_chunks=prefix_chunks,
        slots={},
    )
    failures = []
    for name, run in (("batched", batched), ("control", unbatched)):
        t0 = time.time()
        run()
        for slot in range(num_lanes):
            actual = _read_slot(model, slot, n_chunks[slot] * chunk, chunk, mesh_config)
            batched_rows = slice(prefix_chunks[slot] * chunk, n_chunks[slot] * chunk)
            entry = report["slots"].setdefault(slot, dict(tokens=n_chunks[slot] * chunk))
            entry[name] = _compare(reference[slot], actual)
            entry[f"{name}_stepped_rows"] = _compare(reference[slot], actual, batched_rows)
            logger.info(
                f"[batching] RESULT B={num_lanes} fid={fidelity} slot={slot} {name} all rows: {_fmt(entry[name])}"
            )
            logger.info(
                f"[batching] RESULT B={num_lanes} fid={fidelity} slot={slot} {name} stepped rows "
                f"[{batched_rows.start}, {batched_rows.stop}): {_fmt(entry[f'{name}_stepped_rows'])}"
            )
            if name == "batched":
                for key in ("batched", "batched_stepped_rows"):
                    if not _gates(entry[key]):
                        failures.append(f"slot {slot} {key}: {_fmt(entry[key])}")
                if slot == 0:
                    golden = {layer: load_gpu_cache_heads(GOLDEN_DIR, layer, n_chunks[0] * chunk) for layer in actual}
                    entry["batched_vs_gpu"] = _compare(golden, actual)
                    entry["unbatched_vs_gpu"] = _compare(golden, reference[0])
                    for key in ("unbatched_vs_gpu", "batched_vs_gpu"):
                        logger.info(f"[batching] RESULT B={num_lanes} fid={fidelity} slot=0 {key}: {_fmt(entry[key])}")
            del actual
        logger.info(f"[batching] {name} run + readback + compare {time.time() - t0:.0f}s")

    runner.release()
    _write_report("kv", f"kv_B{num_lanes}_{fidelity}{'_deep' if deep else ''}.json", report)
    assert not failures, "batched KV fails the gates:\n" + "\n".join(failures)


# ── Op check: windowed KV-cache write ───────────────────────────────────────────


@torch.no_grad()
@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_kv_window_write(mesh_device, reset_seeds):
    """update_padded_kv_cache with an input row window writes exactly what slice-then-write does: four lanes of a
    stacked [1, heads, 4 x 256, 256] bfp8 input into four slots at different prefixes, two caches, bit-identical."""
    from models.demos.gemma4_d_p.tt.attention.ring_prefill import (
        init_sliding_ring_kv_cache,
        write_chunk_to_sliding_ring_cache,
    )
    from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillMetadata

    mesh_config = MeshConfig(mesh_device)
    heads, width = 4, 256
    lanes = [(0, 2048 * 3, 256), (1, 0, 256), (2, 2048, 512), (3, 4096, 256)]  # (slot, prefix, rows)
    total = sum(r for _, _, r in lanes)
    caches = [init_sliding_ring_kv_cache(mesh_config, heads, width, max_seq_len=16384, num_users=4) for _ in range(2)]
    host = torch.randn(1, heads, total, width).to(torch.bfloat16)
    stacked = ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    metadata = [PrefillMetadata(mesh_config) for _ in lanes]
    start = 0
    for (slot, prefix, n), meta in zip(lanes, metadata):
        meta.update(slot_idx=slot, kv_actual_global=prefix)
        sliced = ttnn.slice(stacked, (0, 0, start, 0), (1, heads, start + n, width))
        write_chunk_to_sliding_ring_cache(
            caches[0].k, caches[0].v, sliced, sliced, mesh_config, kv_actual_global=0, prefill_metadata=meta
        )
        write_chunk_to_sliding_ring_cache(
            caches[1].k,
            caches[1].v,
            stacked,
            stacked,
            mesh_config,
            kv_actual_global=0,
            prefill_metadata=meta,
            input_rows=(start, n),
        )
        sliced.deallocate(True)
        start += n
    ttnn.synchronize_device(mesh_device)
    for name in ("k", "v"):
        a = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(getattr(caches[0], name).cpu())]
        b = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(getattr(caches[1], name).cpu())]
        assert all(torch.equal(x, y) for x, y in zip(a, b)), f"windowed write differs from slice + write ({name})"
        nonzero = sum(int(x.abs().sum() > 0) for x in a)
        logger.info(f"[batching] KV window write {name}: bit-identical on {len(a)} devices ({nonzero} with data)")


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
@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
@pytest.mark.parametrize("scenario", list(_LANES_SDPA_SCENARIOS))
def test_lanes_sdpa_matches_per_request(mesh_device, scenario, reset_seeds, monkeypatch):
    """One eager 2k x 4 step (prefixes in 2k chunks per slot) with the sliding layers' attention as one lanes call vs
    one call per lane; both with sliding K split 1 so the per-unit arithmetic is the same. The step's final hidden
    states and every slot's KV must be bit-identical. lane0_empty puts the shortest prefix on lane 0, whose prefix
    alone drives the call's ring-iteration masks; the same_* scenarios give one request several lanes."""
    prefix_chunks, step_chunks = _LANES_SDPA_SCENARIOS[scenario]
    chunk, num_lanes, capacity = 2048, 4, 32768
    mesh_config, model_args, model = _build(mesh_device, chunk, num_lanes, capacity)
    prompts = _prompts(num_lanes, capacity, model_args.vocab_size)
    monkeypatch.setattr(ring_prefill, "SLIDING_K_SPLITS_OVERRIDE", 1)

    def lane(slot, idx):
        return (slot, idx * chunk, prompts[slot][idx * chunk : (idx + 1) * chunk])

    runner = StepRunner(mesh_device, mesh_config, model, chunk)
    for layout in ((chunk,), (chunk,) * num_lanes):
        runner._ensure(layout)

    def eager(lanes):
        runner.stage(lanes)
        out = runner._forward(runner.layout(lanes))
        ttnn.synchronize_device(mesh_device)
        host = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out.cpu())]
        out.deallocate(True)
        return host

    for slot, n in enumerate(prefix_chunks):
        for idx in range(n):
            eager([lane(slot, idx)])
    step = [lane(slot, idx) for slot, idx in step_chunks]
    filled = [max([n] + [idx + 1 for s, idx in step_chunks if s == slot]) for slot, n in enumerate(prefix_chunks)]
    results = {}
    for mode in (False, True):
        monkeypatch.setattr(attention, "USE_LANES_SLIDING_SDPA", mode)
        hidden = eager(step)
        kv = [_read_slot(model, slot, n * chunk, chunk, mesh_config) if n else None for slot, n in enumerate(filled)]
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
@pytest.mark.timeout(10800)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_same_request_lanes(mesh_device, reset_seeds, monkeypatch):
    """Lanes of one request in a step (consecutive chunks on one slot): KV vs the request prefilled alone, and the
    step time vs the same chunks as unbatched steps.

    Every lane's KV write runs before any lane's attention, so lane k attends lanes 0..k-1 of the same step like
    an earlier chunk, and every lane keeps the chunk-2048 cache layout. HiFi2 projections, as batched serving runs
    them: at LoFi (the default from 512 rows per device) a request whose every chunk ran batched misses the gates.
    """
    monkeypatch.setattr(attention_operations, "PROJECTION_FIDELITY_OVERRIDE", ttnn.MathFidelity.HiFi2)
    chunk, num_lanes, num_slots = 2048, 4, 2
    capacity = 65536
    mesh_config, model_args, model = _build(mesh_device, chunk, num_slots, capacity)
    prompts = _prompts(2, capacity, model_args.vocab_size)
    prompts[1] = _text_token_stream(_hf_model_id())[0][60_000 : 60_000 + capacity].clone()

    def lane(slot, idx):
        return (slot, idx * chunk, prompts[slot][idx * chunk : (idx + 1) * chunk])

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
    reference = {slot: _read_slot(model, slot, deepest[slot] * chunk, chunk, mesh_config) for slot in range(num_slots)}

    report, failures = dict(chunk=chunk, scenarios={}), []
    for name, steps in _SAME_SLOT_SCENARIOS.items():
        for step in steps:
            runner.step([lane(slot, idx) for slot, idx in step])
        report["scenarios"][name] = entry = {}
        for slot, n in n_chunks[name].items():
            actual = _read_slot(model, slot, n * chunk, chunk, mesh_config)
            rows = slice(0, n * chunk)
            expected = {layer: {c: t[rows] for c, t in heads.items()} for layer, heads in reference[slot].items()}
            entry[slot] = dict(tokens=n * chunk, vs_alone=_compare(expected, actual))
            logger.info(f"[batching] RESULT same-slot {name} slot={slot} vs alone: {_fmt(entry[slot]['vs_alone'])}")
            if not _gates(entry[slot]["vs_alone"]):
                failures.append(f"{name} slot {slot}: {_fmt(entry[slot]['vs_alone'])}")
            if slot == 0:
                golden = {layer: load_gpu_cache_heads(GOLDEN_DIR, layer, n * chunk) for layer in actual}
                entry[slot]["vs_gpu"] = _compare(golden, actual)
                entry[slot]["alone_vs_gpu"] = _compare(golden, expected)
                logger.info(f"[batching] RESULT same-slot {name} slot=0 vs GPU: {_fmt(entry[slot]['vs_gpu'])}")
                logger.info(
                    f"[batching] RESULT same-slot {name} slot=0 alone vs GPU: {_fmt(entry[slot]['alone_vs_gpu'])}"
                )
            del actual

    # Perf: one request's four chunks as one step vs four unbatched steps, at prefix 0 and 32k (slot 0 holds real
    # KV that deep from the reference run plus the fill below).
    for idx in range(deepest[0], capacity // chunk):
        runner.step([lane(0, idx)])
    report["perf"] = []
    for prefix_chunks in (0, 16):
        lanes = [lane(0, prefix_chunks + i) for i in range(num_lanes)]
        batched_ms = runner.timed(lanes)
        alone_ms = [runner.timed([single]) for single in lanes]
        report["perf"].append(
            dict(
                prefix=prefix_chunks * chunk,
                batched_ms=round(batched_ms, 2),
                unbatched_ms=[round(x, 2) for x in alone_ms],
            )
        )
        logger.info(
            f"[batching] PERF same-slot prefix={prefix_chunks * chunk} 4 lanes={batched_ms:.1f}ms "
            f"4 steps={sum(alone_ms):.1f}ms per_step={[round(x, 1) for x in alone_ms]}"
        )
    runner.release()
    _write_report("kv", "same_request_lanes.json", report)
    assert not failures, "same-request lanes fail the gates:\n" + "\n".join(failures)


# ── Partial steps: spare lanes on a scratch slot ───────────────────────────────


@torch.no_grad()
@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_partial_steps(mesh_device, reset_seeds):
    """A 4-lane trace serving 2 requests, with the 2 spare lanes on a scratch slot: the requests' KV vs each alone,
    an untouched slot stays bit-identical, and the step time vs a 2-lane trace.
    """
    chunk, capacity = 2048, 65536
    real, scratch, untouched = (0, 1), 2, 3
    mesh_config, model_args, model = _build(mesh_device, chunk, 4, capacity)
    prompts = _prompts(1, capacity, model_args.vocab_size)
    stream = _text_token_stream(_hf_model_id())[0]
    prompts += [stream[offset : offset + capacity].clone() for offset in (60_000, 120_000, 180_000)]
    pad = torch.zeros(chunk, dtype=torch.int32)

    def lane(slot, idx):
        return (slot, idx * chunk, prompts[slot][idx * chunk : (idx + 1) * chunk])

    def spare():
        return (scratch, 0, pad)

    runner = StepRunner(mesh_device, mesh_config, model, chunk)
    runner.capture_all([[lane(0, 0)], [lane(0, 0), lane(1, 0)], [lane(0, 0), lane(1, 0), spare(), spare()]])

    prefix = {0: 3, 1: 0}
    steps = 2
    n_chunks = {slot: prefix[slot] + steps for slot in real}
    reference = {}
    for slot in real:
        for idx in range(n_chunks[slot]):
            runner.step([lane(slot, idx)])
        reference[slot] = _read_slot(model, slot, n_chunks[slot] * chunk, chunk, mesh_config)
    for idx in range(4):
        runner.step([lane(untouched, idx)])
    untouched_before = _read_slot(model, untouched, 4 * chunk, chunk, mesh_config)

    for slot in real:
        for idx in range(prefix[slot]):
            runner.step([lane(slot, idx)])
    for k in range(steps):
        runner.step([lane(0, prefix[0] + k), lane(1, prefix[1] + k), spare(), spare()])

    report, failures = dict(chunk=chunk, slots={}), []
    for slot in real:
        actual = _read_slot(model, slot, n_chunks[slot] * chunk, chunk, mesh_config)
        report["slots"][slot] = result = _compare(reference[slot], actual)
        logger.info(f"[batching] RESULT partial slot={slot} vs alone: {_fmt(result)}")
        if not _gates(result):
            failures.append(f"slot {slot}: {_fmt(result)}")
    untouched_after = _read_slot(model, untouched, 4 * chunk, chunk, mesh_config)
    report["untouched"] = result = _compare(untouched_before, untouched_after)
    logger.info(f"[batching] RESULT partial untouched slot: {_fmt(result)}")
    if result["first_differing_layer"] is not None:
        failures.append(f"untouched slot changed: {_fmt(result)}")

    for slot in (*real, scratch):
        for idx in range(16 + 1):
            runner.step([lane(slot, idx)] if slot != scratch else [(scratch, idx * chunk, pad)])
    report["perf"] = []
    for name, p in (("shallow", (0, 0)), ("u32k", (16, 16)), ("mixed", (16, 0))):
        two = [lane(0, p[0]), lane(1, p[1])]
        row = dict(
            prefixes=[q * chunk for q in p],
            two_lane_ms=round(runner.timed(two), 2),
            four_lane_two_spare_ms=round(runner.timed(two + [spare(), spare()]), 2),
            alone_ms=[round(runner.timed([x]), 2) for x in two],
        )
        report["perf"].append(row)
        logger.info(f"[batching] PERF partial {name} {row}")
    runner.release()
    _write_report("kv", "partial_steps.json", report)
    assert not failures, "partial steps fail:\n" + "\n".join(failures)


# ── Per-request chunk widths ───────────────────────────────────────────────────

# Scenarios: name -> (steps, compare). A step is [(slot, chunk start, width)]. compare names, per slot, the width
# whose unbatched run is the reference (None: the slot's own widths). The ring cache layout depends on the chunk
# width, so a request keeps one width for its life, except in width_change, which shows what breaks when it doesn't.
_WIDTH_SCENARIOS = {
    # 4k + 2k + 2k in one step, prefixes 8k / 0 / 6k, two steps.
    "mixed_4k_2k_2k": (
        [[(0, 0, 4096)], [(0, 4096, 4096)], [(2, 0, 2048)], [(2, 2048, 2048)], [(2, 4096, 2048)]]
        + [[(0, 8192 + 4096 * k, 4096), (1, 2048 * k, 2048), (2, 6144 + 2048 * k, 2048)] for k in range(2)],
        {0: None, 1: None, 2: None},
    ),
    # 4k + 4k, prefixes 0 / 4k, two steps.
    "pair_4k_4k": (
        [[(1, 0, 4096)]] + [[(0, 4096 * k, 4096), (1, 4096 + 4096 * k, 4096)] for k in range(2)],
        {0: None, 1: None},
    ),
    # A lone 16k prompt as two 8k steps, compared with the same prompt in 2k chunks.
    "lone_8k": ([[(0, 0, 8192)], [(0, 8192, 8192)]], {0: 2048}),
    # Diagnostic: four 2k chunks, then the same request continues at width 8k.
    "width_change": ([[(0, 2048 * k, 2048)] for k in range(4)] + [[(0, 8192, 8192)]], {0: 2048}),
    # Diagnostic, reversed: one 8k chunk, then four 2k chunks.
    "width_change_rev": ([[(0, 0, 8192)]] + [[(0, 8192 + 2048 * k, 2048)] for k in range(4)], {0: 2048}),
}
_GATED_WIDTH_SCENARIOS = ("mixed_4k_2k_2k", "pair_4k_4k", "lone_8k")


@torch.no_grad()
@pytest.mark.timeout(10800)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_lane_widths(mesh_device, reset_seeds):
    """Requests with different chunk widths in one step (each request keeps its width): KV vs each request alone at
    its width, then step times vs the same requests unbatched.
    """
    unit, num_slots, capacity = 2048, 3, 65536
    mesh_config, model_args, model = _build(mesh_device, unit, num_slots, capacity)
    stream = _text_token_stream(_hf_model_id())[0]
    prompts = _prompts(1, capacity, model_args.vocab_size)
    prompts += [stream[offset : offset + capacity].clone() for offset in (60_000, 120_000)]

    def lane(slot, start, width):
        return (slot, start, prompts[slot][start : start + width])

    # Widest first: the first compile's all-gathers place their lazily created global semaphores low in L1 (under
    # the activations live at that point), and every later program's CB region must end below them. A chunk-2048
    # first compile puts them at ~1,440,576, under the 4k global SDPA's CB end (1,475,712).
    layouts = [(8192,), (4096,), (4096, 4096), (4096, 2048, 2048), (2048,) * 4, (2048,)]
    runner = StepRunner(mesh_device, mesh_config, model, unit)
    runner.always_lanes = True
    runner.capture_all([[lane(slot, 0, w) for slot, w in zip((0, 1, 2, 0), layout)] for layout in layouts])

    def alone(slot, width, n_tokens):
        for start in range(0, n_tokens, width):
            runner.step([lane(slot, start, width)])
        return _read_slot(model, slot, n_tokens, width, mesh_config)

    report, failures = dict(unit=unit, scenarios={}), []
    wanted = os.environ.get("G4B_WIDTH_SCENARIOS")
    for name, (steps, compare) in _WIDTH_SCENARIOS.items():
        if wanted and name not in wanted.split(","):
            continue
        widths = {}
        for step in steps:
            for slot, start, width in step:
                widths.setdefault(slot, []).append(width)
        references = {}
        for slot, ref_width in compare.items():
            ref_width = ref_width or widths[slot][0]
            references[slot] = alone(slot, ref_width, sum(widths[slot]))
        for step in steps:
            runner.step([lane(*spec) for spec in step])
        report["scenarios"][name] = entry = {}
        for slot in compare:
            n_tokens = sum(widths[slot])
            actual = _read_slot(model, slot, n_tokens, unit, mesh_config, widths=widths[slot])
            entry[slot] = dict(widths=widths[slot], vs_alone=_compare(references[slot], actual))
            logger.info(
                f"[batching] RESULT widths {name} slot={slot} {widths[slot]} vs alone: {_fmt(entry[slot]['vs_alone'])}"
            )
            if name in _GATED_WIDTH_SCENARIOS and not _gates(entry[slot]["vs_alone"]):
                failures.append(f"{name} slot {slot}: {_fmt(entry[slot]['vs_alone'])}")
            if name.startswith("width_change"):
                # After the width change: the first window (1,024 tokens) needs history written at the other width;
                # the rest of the chunk only needs its own rows.
                for label, rows in (("first 1k after the change", slice(8192, 9216)), ("rest", slice(9216, 16384))):
                    entry[slot][label] = result = _compare(references[slot], actual, rows)
                    pccs = [round(x, 5) for x in result["layer_pcc"][:8]]
                    logger.info(f"[batching] RESULT widths {name} {label}: {_fmt(result)} layers 0-7 {pccs}")
            if slot == 0:
                golden = {layer: load_gpu_cache_heads(GOLDEN_DIR, layer, n_tokens) for layer in actual}
                entry[slot]["vs_gpu"] = _compare(golden, actual)
                entry[slot]["alone_vs_gpu"] = _compare(golden, references[slot])
                logger.info(f"[batching] RESULT widths {name} slot=0 vs GPU: {_fmt(entry[slot]['vs_gpu'])}")
                logger.info(f"[batching] RESULT widths {name} slot=0 alone vs GPU: {_fmt(entry[slot]['alone_vs_gpu'])}")
            del actual
        del references

    # Perf. Every slot holds real text KV to 40k (2k chunks); a timed step rewrites its own rows.
    if os.environ.get("G4B_NO_PERF") == "1":
        runner.release()
        assert not failures, "per-request widths fail the gates:\n" + "\n".join(failures)
        return
    for slot in range(num_slots):
        for start in range(0, 40960, unit):
            runner.step([lane(slot, start, unit)])
    cases = {
        # (slot, offset from the case's prefix, width) per lane
        "lone_8k": [(0, 0, 8192)],
        "lone_4x2k": [(0, 2048 * i, 2048) for i in range(4)],
        "mixed_4k_2k_2k": [(0, 0, 4096), (1, 0, 2048), (2, 0, 2048)],
        "pair_4k_4k": [(0, 0, 4096), (1, 0, 4096)],
    }
    report["perf"] = []
    for prefix in (0, 32768):
        for case, spec in cases.items():
            lanes = [lane(slot, prefix + offset, width) for slot, offset, width in spec]
            own = [[single] for single in lanes]
            unit_steps = [
                [lane(slot, start + unit * i, unit)] for slot, start, t in lanes for i in range(len(t) // unit)
            ]
            batched_ms = runner.timed(lanes)
            own_ms = [runner.timed(single) for single in own]
            unit_ms = [runner.timed(single) for single in unit_steps]
            row = dict(case=case, prefix=prefix, layout=runner.layout(lanes), batched_ms=round(batched_ms, 2))
            row["own_width_alone_ms"] = [round(x, 2) for x in own_ms]
            row["unit_steps_ms"] = [round(x, 2) for x in unit_ms]
            report["perf"].append(row)
            logger.info(
                f"[batching] PERF widths {case} prefix={prefix} layout={row['layout']} step={batched_ms:.1f}ms "
                f"each request alone at its width={sum(own_ms):.1f}ms {row['own_width_alone_ms']} "
                f"as {len(unit_ms)} unbatched 2k steps={sum(unit_ms):.1f}ms"
            )
    runner.release()
    _write_report("kv", "lane_widths.json", report)
    assert not failures, "per-request widths fail the gates:\n" + "\n".join(failures)


# ── Perf ───────────────────────────────────────────────────────────────────────

_MIXES = {
    2048: {
        "shallow": lambda b: [2048] * b,
        "u32k": lambda b: [32768] * b,
        "mixed": lambda b: [2048, 16384, 65536, 131072][:b],
        "deep": lambda b: [131072] * b,
    },
    1024: {
        "shallow": lambda b: [1024] * b,
        "u32k": lambda b: [32768] * b,
        "mixed": lambda b: ([1024, 8192, 32768, 65536] * 2)[:b],
        "deep": lambda b: [98304] * b,
    },
}
_PERF_SHAPES = {2048: ((1, 2, 4), 262144), 1024: ((1, 2, 4, 8), 131072)}


@torch.no_grad()
@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
@pytest.mark.parametrize("chunk_size", [2048, 1024], ids=lambda c: f"c{c}")
def test_batched_step_perf(mesh_device, chunk_size, reset_seeds, monkeypatch):
    """Traced step time of B requests x chunk vs B unbatched steps at the same prefixes."""
    if os.environ.get("G4B_PROJECTION_FIDELITY", "").lower() == "hifi2":
        monkeypatch.setattr(attention_operations, "PROJECTION_FIDELITY_OVERRIDE", ttnn.MathFidelity.HiFi2)
    monkeypatch.setattr(attention, "USE_LANES_SLIDING_SDPA", os.environ.get("G4B_LANES_SDPA", "1") == "1")
    lane_counts, capacity = _PERF_SHAPES[chunk_size]
    lane_counts = tuple(int(b) for b in os.environ.get("G4B_LANES", ",".join(map(str, lane_counts))).split(","))
    num_slots = max(lane_counts)
    mixes = _MIXES[chunk_size]
    mesh_config, model_args, model = _build(mesh_device, chunk_size, num_slots, capacity)

    # Each slot gets its own window of real text, prefilled up to the deepest prefix any mix asks for, so the
    # cached prefixes hold real KV. A measured step rewrites its own rows with the same tokens.
    deepest = max(max(mix(num_slots)) for mix in mixes.values()) + chunk_size
    stream = _text_token_stream(_hf_model_id())[0]
    stride = min(100_000, (len(stream) - deepest) // max(1, num_slots - 1))
    assert stride > 0, f"token stream ({len(stream)}) too short for {num_slots} windows of {deepest}"
    seqs = [stream[slot * stride : slot * stride + deepest].clone() for slot in range(num_slots)]

    def lane(slot, start):
        return (slot, start, seqs[slot][start : start + chunk_size])

    runner = StepRunner(mesh_device, mesh_config, model, chunk_size)
    runner.capture_all([[lane(slot, 0) for slot in range(b)] for b in sorted({1, *lane_counts})])

    t0 = time.time()
    for slot in range(num_slots):
        for start in range(0, deepest - chunk_size, chunk_size):
            runner.step([lane(slot, start)])
    logger.info(f"[batching] filled {num_slots} slots to {deepest - chunk_size} tokens in {time.time() - t0:.0f}s")

    results = []
    for mix_name, mix in mixes.items():
        for b in lane_counts:
            prefixes = mix(b)
            lanes = [lane(slot, p) for slot, p in enumerate(prefixes)]
            batched_ms = runner.timed(lanes)
            alone_ms = [runner.timed([single]) for single in lanes] if b > 1 else [batched_ms]
            row = dict(
                chunk=chunk_size,
                lanes=b,
                mix=mix_name,
                prefixes=prefixes,
                batched_ms=round(batched_ms, 2),
                unbatched_ms=[round(x, 2) for x in alone_ms],
                unbatched_sum_ms=round(sum(alone_ms), 2),
            )
            results.append(row)
            logger.info(
                f"[batching] PERF chunk={chunk_size} B={b} mix={mix_name} prefixes={prefixes} "
                f"batched={batched_ms:.1f}ms unbatched_sum={sum(alone_ms):.1f}ms "
                f"ratio={batched_ms / sum(alone_ms):.3f} per_lane={[round(x, 1) for x in alone_ms]} "
                f"tok/s batched={b * chunk_size * 1000 / batched_ms:.0f} unbatched={b * chunk_size * 1000 / sum(alone_ms):.0f}"
            )
    runner.release()
    _write_report("perf", f"perf_c{chunk_size}_{time.strftime('%Y%m%d_%H%M%S')}.json", results)
