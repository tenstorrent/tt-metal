# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Warmed prefill and traced warmed decode latency for the functional decoder.

These numbers are the stage-01 baseline that the optimized decoder has to beat,
so the measurement conditions matter as much as the values:

* **Warmed.** The first call to any shape compiles kernels and populates the
  program cache. Timing that measures the compiler. Every configuration runs
  warmup iterations that are discarded.
* **Traced decode.** Decode is short enough that host dispatch overhead is a
  large share of wall time, so an eager measurement mostly reports Python.
  Replaying a captured trace is what a serving stack actually does.
* **Device-synchronised.** Dispatch is asynchronous; without an explicit
  synchronise the host would time enqueue calls rather than execution.
* **Median, not mean.** One descheduled iteration should not move the number.

Results are written to ``doc/functional_decoder/`` as CSV so later stages can
diff against them rather than re-deriving a baseline.

Two harness rules the CSVs depend on:

* **Every test here is marked ``models_performance_bare_metal``.** These tests
  overwrite the published CSVs, and ``TT_METAL_WATCHER=10`` inflates device
  timings roughly 8x, so a watcher run over the whole suite silently replaced
  the prefill baseline with 4358 us/token. Watcher runs must deselect them:
  ``TT_METAL_WATCHER=10 pytest ... -m "not models_performance_bare_metal"``.
* **The decode CSVs are rewritten whole, not appended to.** The decode test is
  parametrised over context length, so each parametrisation contributes one
  row; rows accumulate in ``_DECODE_ROWS`` for the life of the process and the
  file is rewritten from scratch each time. Appending stacked eight interleaved
  runs into ``doc/functional_decoder/perf_decode.csv`` before this was fixed.
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ..tt.functional_decoder import (
    DecoderLayerConfig,
    build_expert_sparsity,
    build_rope_cache,
    create_kv_cache,
    decoder_layer_decode,
    decoder_layer_prefill,
    upload_layer_weights,
)
from ..tt.weight_mapping import convert_layer_weights
from .reference import build_reference_layer, layer_state_dict

LAYER_IDX = 0
MAX_SEQ = 4096
BLOCK_SIZE = 32
TRACE_REGION_SIZE = 50331648

PREFILL_LENGTHS = [128, 512, 1024, 2048]
PREFILL_WARMUP, PREFILL_ITERS = 1, 5
DECODE_WARMUP, DECODE_ITERS = 10, 100

DOC_DIR = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder"

DECODE_FIELDS = ["context_len", "median_ms", "min_ms", "max_ms", "tok_per_s_per_layer", "iters"]

# {csv path: [row, ...]} for this process only, so a rerun truncates.
_DECODE_ROWS: dict[Path, list[dict]] = {}


def _write_decode_row(path: Path, row: dict) -> None:
    """Add a row and rewrite the whole file (see module docstring)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _DECODE_ROWS.setdefault(path, [])
    rows[:] = [r for r in rows if r["context_len"] != row["context_len"]] + [row]
    rows.sort(key=lambda r: r["context_len"])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DECODE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def reference():
    return build_reference_layer(LAYER_IDX)


@pytest.fixture(scope="module")
def torch_weights(reference):
    _, hf_config = reference
    return convert_layer_weights(layer_state_dict(LAYER_IDX), hf_config)


def _to_device(t, mesh_device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> Path:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    path = DOC_DIR / name
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {path}")
    return path


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_prefill_latency(mesh_device, reference, torch_weights):
    _, hf_config = reference
    config = DecoderLayerConfig.from_hf(hf_config)
    weights = upload_layer_weights(torch_weights, mesh_device, config)
    cos_cache, sin_cache = build_rope_cache(hf_config, MAX_SEQ, mesh_device)
    sparsity = build_expert_sparsity(mesh_device, config.moe.num_experts)

    rows = []
    for seq_len in PREFILL_LENGTHS:
        torch.manual_seed(0)
        hidden = torch.randn(1, 1, seq_len, hf_config.hidden_size) * 0.02
        tt_in = _to_device(hidden, mesh_device)

        def once():
            out = decoder_layer_prefill(tt_in, weights, config, cos_cache, sin_cache, sparsity)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(out)

        for _ in range(PREFILL_WARMUP):
            once()

        samples = []
        for _ in range(PREFILL_ITERS):
            t0 = time.perf_counter()
            once()
            samples.append((time.perf_counter() - t0) * 1e3)

        median = statistics.median(samples)
        per_tok = median / seq_len * 1e3  # us/token
        logger.info(
            f"prefill S={seq_len:>5}: median {median:8.2f} ms  "
            f"min {min(samples):8.2f}  max {max(samples):8.2f}  ({per_tok:6.1f} us/token)"
        )
        rows.append(
            {
                "seq_len": seq_len,
                "median_ms": round(median, 3),
                "min_ms": round(min(samples), 3),
                "max_ms": round(max(samples), 3),
                "us_per_token": round(per_tok, 2),
                "iters": PREFILL_ITERS,
            }
        )
        ttnn.deallocate(tt_in)

    _write_csv(
        "perf_prefill.csv",
        ["seq_len", "median_ms", "min_ms", "max_ms", "us_per_token", "iters"],
        rows,
    )
    assert all(r["median_ms"] > 0 for r in rows)


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("context_len", [128, 1024, 4096], ids=["ctx128", "ctx1k", "ctx4k"])
def test_decode_latency_traced(mesh_device, reference, torch_weights, context_len):
    """Traced single-token decode latency at several cache depths.

    Swept over context because decode cost is dominated by the SDPA read over
    the cache, so a single depth would not say whether latency is flat or grows
    with the conversation.
    """
    _, hf_config = reference
    config = DecoderLayerConfig.from_hf(hf_config)
    weights = upload_layer_weights(torch_weights, mesh_device, config)
    cos_cache, sin_cache = build_rope_cache(hf_config, MAX_SEQ, mesh_device)
    kv_cache = create_kv_cache(mesh_device, config.attention, max_batch=1, max_seq_len=MAX_SEQ, block_size=BLOCK_SIZE)

    pos = context_len - 1
    torch.manual_seed(0)
    tt_in = _to_device(torch.randn(1, 1, 1, hf_config.hidden_size) * 0.02, mesh_device)
    current_pos = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, device=mesh_device)

    def step():
        return decoder_layer_decode(
            tt_in, weights, config, cos_cache, sin_cache, kv_cache, current_pos, token_index=pos
        )

    step()  # compile outside the capture
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    step()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    for _ in range(DECODE_WARMUP):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)

    samples = []
    for _ in range(DECODE_ITERS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - t0) * 1e3)

    median = statistics.median(samples)
    logger.info(
        f"traced decode ctx={context_len:>5}: median {median:7.3f} ms  "
        f"min {min(samples):7.3f}  max {max(samples):7.3f}  "
        f"({1e3 / median:7.1f} tok/s/layer)"
    )

    _write_decode_row(
        DOC_DIR / "perf_decode.csv",
        {
            "context_len": context_len,
            "median_ms": round(median, 4),
            "min_ms": round(min(samples), 4),
            "max_ms": round(max(samples), 4),
            "tok_per_s_per_layer": round(1e3 / median, 1),
            "iters": DECODE_ITERS,
        },
    )

    ttnn.release_trace(mesh_device, trace_id)
    assert median > 0


# --- optimized path -----------------------------------------------------------
# Same harness, same conditions, writing to doc/optimized_decoder/ so the two
# stages' CSVs are directly diffable rather than needing re-derivation.

OPT_DOC_DIR = Path(__file__).resolve().parents[1] / "doc" / "optimized_decoder"


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_optimized_prefill_latency(mesh_device, reference, torch_weights):
    from ..tt import optimized_decoder as O

    _, hf_config = reference
    config = DecoderLayerConfig.from_hf(hf_config)
    weights = upload_layer_weights(torch_weights, mesh_device, config)
    packed = O.upload_packed_expert_weights(torch_weights, mesh_device, config.moe)
    cos_cache, sin_cache = build_rope_cache(hf_config, MAX_SEQ, mesh_device)
    sparsity = build_expert_sparsity(mesh_device, config.moe.num_experts)

    rows = []
    for seq_len in PREFILL_LENGTHS:
        torch.manual_seed(0)
        tt_in = _to_device(torch.randn(1, 1, seq_len, hf_config.hidden_size) * 0.02, mesh_device)

        def once():
            out = O.decoder_layer_prefill_optimized(tt_in, weights, config, cos_cache, sin_cache, sparsity, packed)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(out)

        for _ in range(PREFILL_WARMUP):
            once()
        samples = []
        for _ in range(PREFILL_ITERS):
            t0 = time.perf_counter()
            once()
            samples.append((time.perf_counter() - t0) * 1e3)

        median = statistics.median(samples)
        logger.info(
            f"OPTIMIZED prefill S={seq_len:>5}: median {median:8.2f} ms ({median / seq_len * 1e3:6.1f} us/token)"
        )
        rows.append(
            {
                "seq_len": seq_len,
                "median_ms": round(median, 3),
                "min_ms": round(min(samples), 3),
                "max_ms": round(max(samples), 3),
                "us_per_token": round(median / seq_len * 1e3, 2),
                "iters": PREFILL_ITERS,
            }
        )
        ttnn.deallocate(tt_in)

    OPT_DOC_DIR.mkdir(parents=True, exist_ok=True)
    with (OPT_DOC_DIR / "perf_prefill.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["seq_len", "median_ms", "min_ms", "max_ms", "us_per_token", "iters"])
        wr.writeheader()
        wr.writerows(rows)
    assert all(r["median_ms"] > 0 for r in rows)


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("context_len", [128, 1024, 4096], ids=["ctx128", "ctx1k", "ctx4k"])
def test_optimized_decode_latency_traced(mesh_device, reference, torch_weights, context_len):
    from ..tt import optimized_decoder as O

    _, hf_config = reference
    config = DecoderLayerConfig.from_hf(hf_config)
    weights = upload_layer_weights(torch_weights, mesh_device, config)
    packed = O.upload_packed_expert_weights(torch_weights, mesh_device, config.moe)
    cos_cache, sin_cache = build_rope_cache(hf_config, MAX_SEQ, mesh_device)
    kv_cache = create_kv_cache(mesh_device, config.attention, max_batch=1, max_seq_len=MAX_SEQ, block_size=BLOCK_SIZE)

    pos = context_len - 1
    torch.manual_seed(0)
    tt_in = _to_device(torch.randn(1, 1, 1, hf_config.hidden_size) * 0.02, mesh_device)
    current_pos = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, device=mesh_device)

    def step():
        return O.decoder_layer_decode_optimized(
            tt_in, weights, config, cos_cache, sin_cache, kv_cache, current_pos, pos, packed_experts=packed
        )

    step()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    step()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    for _ in range(DECODE_WARMUP):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    samples = []
    for _ in range(DECODE_ITERS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - t0) * 1e3)

    median = statistics.median(samples)
    logger.info(
        f"OPTIMIZED traced decode ctx={context_len:>5}: median {median:7.3f} ms ({1e3 / median:7.1f} tok/s/layer)"
    )

    _write_decode_row(
        OPT_DOC_DIR / "perf_decode.csv",
        {
            "context_len": context_len,
            "median_ms": round(median, 4),
            "min_ms": round(min(samples), 4),
            "max_ms": round(max(samples), 4),
            "tok_per_s_per_layer": round(1e3 / median, 1),
            "iters": DECODE_ITERS,
        },
    )
    ttnn.release_trace(mesh_device, trace_id)
    assert median > 0


# --- multichip path -----------------------------------------------------------
# Stage 03, on the full 4-die P300_X2 mesh. Three tests, and the first of them is
# the one that makes the other two mean anything:
#
#   test_multichip_baseline_1x1_*  re-measures the *single-chip* optimized layer,
#       with this harness, in this tree, on one die, and writes it into
#       doc/multichip_decoder/. Stage 02's CSVs already hold a warmed single-chip
#       baseline, but they are also the artifact its own README quotes cell by
#       cell, and re-running them here would move their third significant figure
#       and silently invalidate that document. A stage-owned copy of the baseline
#       is cheaper than a cross-stage prose/artifact mismatch.
#   test_multichip_prefill_latency / test_multichip_decode_latency_traced
#       measure the same lengths on the mesh, so speedup is one CSV cell divided
#       by another rather than a number quoted from anywhere.
#
# All three are marked models_performance_bare_metal: they rewrite published
# CSVs and TT_METAL_WATCHER inflates device timings, so a watcher run must
# deselect them.

MC_DOC_DIR = Path(__file__).resolve().parents[1] / "doc" / "optimized_multichip_decoder"
# Stage 04 note. ``tt/multichip_decoder.py`` is optimized **in place**, so the
# four tests below now measure the stage-04 path. They therefore write into
# ``doc/optimized_multichip_decoder/``; ``doc/multichip_decoder/perf_*.csv`` are
# stage 03's frozen *before* numbers and are deliberately never regenerated --
# re-pointing this constant back would overwrite the baseline half of every
# before/after table in both READMEs. The stage-04 decode before/after is also
# measured in one process by
# ``doc/optimized_multichip_decoder/probes/layer_levers.py``, whose "stage 03"
# leg is a verbatim copy of the committed stage-03 layer body.

# Ring fabric must be set before the mesh opens; the conftest device_params hook
# does that, which is why it is spelled here rather than with set_fabric_config.
MC_DEVICE_PARAMS = {"trace_region_size": TRACE_REGION_SIZE, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}
MC_MESH = (1, 4)

PREFILL_FIELDS = ["seq_len", "median_ms", "min_ms", "max_ms", "us_per_token", "iters"]


def _prefill_sweep(once_factory, lengths=PREFILL_LENGTHS, label="") -> list[dict]:
    rows = []
    for seq_len in lengths:
        once = once_factory(seq_len)
        for _ in range(PREFILL_WARMUP):
            once()
        samples = []
        for _ in range(PREFILL_ITERS):
            t0 = time.perf_counter()
            once()
            samples.append((time.perf_counter() - t0) * 1e3)
        median = statistics.median(samples)
        logger.info(f"{label} prefill S={seq_len:>5}: median {median:8.2f} ms ({median / seq_len * 1e3:6.1f} us/token)")
        rows.append(
            {
                "seq_len": seq_len,
                "median_ms": round(median, 3),
                "min_ms": round(min(samples), 3),
                "max_ms": round(max(samples), 3),
                "us_per_token": round(median / seq_len * 1e3, 2),
                "iters": PREFILL_ITERS,
            }
        )
    return rows


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)
    logger.info(f"wrote {path}")


def _traced_decode_median(mesh_device, step) -> tuple[float, float, float]:
    step()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    step()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    for _ in range(DECODE_WARMUP):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    samples = []
    for _ in range(DECODE_ITERS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - t0) * 1e3)
    ttnn.release_trace(mesh_device, trace_id)
    return statistics.median(samples), min(samples), max(samples)


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_optimized_multichip_baseline_1x1_prefill(mesh_device, reference, torch_weights):
    """Stage-04's own copy of the warmed single-chip prefill baseline.

    ``optimized_decoder.py`` is untouched by stage 04, so this re-measures the
    same code stage 03 did; it is re-run rather than quoted so the speedup
    columns in this stage's README are one CSV cell divided by another taken in
    the same session."""
    from ..tt import optimized_decoder as O

    _, hf_config = reference
    config = DecoderLayerConfig.from_hf(hf_config)
    weights = upload_layer_weights(torch_weights, mesh_device, config)
    packed = O.upload_packed_expert_weights(torch_weights, mesh_device, config.moe)
    cos_cache, sin_cache = build_rope_cache(hf_config, MAX_SEQ, mesh_device)
    sparsity = build_expert_sparsity(mesh_device, config.moe.num_experts)

    def factory(seq_len):
        torch.manual_seed(0)
        tt_in = _to_device(torch.randn(1, 1, seq_len, hf_config.hidden_size) * 0.02, mesh_device)

        def once():
            out = O.decoder_layer_prefill_optimized(tt_in, weights, config, cos_cache, sin_cache, sparsity, packed)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(out)

        return once

    rows = _prefill_sweep(factory, label="BASELINE 1x1")
    _write_rows(MC_DOC_DIR / "perf_baseline_1x1_prefill.csv", PREFILL_FIELDS, rows)
    assert all(r["median_ms"] > 0 for r in rows)


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("context_len", [128, 1024, 4096], ids=["ctx128", "ctx1k", "ctx4k"])
def test_optimized_multichip_baseline_1x1_decode(mesh_device, reference, torch_weights, context_len):
    """Stage-04's own copy of the warmed single-chip traced decode baseline."""
    from ..tt import optimized_decoder as O

    _, hf_config = reference
    config = DecoderLayerConfig.from_hf(hf_config)
    weights = upload_layer_weights(torch_weights, mesh_device, config)
    packed = O.upload_packed_expert_weights(torch_weights, mesh_device, config.moe)
    cos_cache, sin_cache = build_rope_cache(hf_config, MAX_SEQ, mesh_device)
    kv_cache = create_kv_cache(mesh_device, config.attention, max_batch=1, max_seq_len=MAX_SEQ, block_size=BLOCK_SIZE)

    pos = context_len - 1
    torch.manual_seed(0)
    tt_in = _to_device(torch.randn(1, 1, 1, hf_config.hidden_size) * 0.02, mesh_device)
    current_pos = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, device=mesh_device)

    median, lo, hi = _traced_decode_median(
        mesh_device,
        lambda: O.decoder_layer_decode_optimized(
            tt_in, weights, config, cos_cache, sin_cache, kv_cache, current_pos, pos, packed_experts=packed
        ),
    )
    logger.info(f"BASELINE 1x1 traced decode ctx={context_len:>5}: median {median:7.4f} ms")
    _write_decode_row(
        MC_DOC_DIR / "perf_baseline_1x1_decode.csv",
        {
            "context_len": context_len,
            "median_ms": round(median, 4),
            "min_ms": round(lo, 4),
            "max_ms": round(hi, 4),
            "tok_per_s_per_layer": round(1e3 / median, 1),
            "iters": DECODE_ITERS,
        },
    )
    assert median > 0


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [MC_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MC_MESH], ids=["1x4"], indirect=True)
def test_optimized_multichip_prefill_latency(mesh_device, reference, torch_weights):
    from ..tt import multichip_decoder as MC

    _, hf_config = reference
    config = MC.MeshDecoderConfig.from_hf(hf_config)
    ctx = MC.mesh_context(mesh_device)
    weights = MC.upload_multichip_weights(torch_weights, mesh_device, config)
    cos_cache, sin_cache = build_rope_cache(hf_config, MAX_SEQ, mesh_device)
    sparsity = MC.build_local_sparsity(mesh_device, config.local_moe)

    def factory(seq_len):
        torch.manual_seed(0)
        tt_in = ttnn.from_torch(
            torch.randn(1, 1, seq_len, hf_config.hidden_size) * 0.02,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        def once():
            out = MC.decoder_layer_prefill_multichip(tt_in, weights, config, ctx, cos_cache, sin_cache, sparsity)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(out)

        return once

    rows = _prefill_sweep(factory, label="MULTICHIP 1x4")
    _write_rows(MC_DOC_DIR / "perf_prefill.csv", PREFILL_FIELDS, rows)
    assert all(r["median_ms"] > 0 for r in rows)


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [MC_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MC_MESH], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("context_len", [128, 1024, 4096], ids=["ctx128", "ctx1k", "ctx4k"])
def test_optimized_multichip_decode_latency_traced(mesh_device, reference, torch_weights, context_len):
    """Warmed trace replay on the mesh. Same harness as the 1x1 baseline above."""
    from ..tt import multichip_decoder as MC

    _, hf_config = reference
    config = MC.MeshDecoderConfig.from_hf(hf_config)
    ctx = MC.mesh_context(mesh_device)
    weights = MC.upload_multichip_weights(torch_weights, mesh_device, config)
    cos_cache, sin_cache = build_rope_cache(hf_config, MAX_SEQ, mesh_device)
    kv_cache = MC.create_mesh_kv_cache(mesh_device, config, 1, MAX_SEQ, block_size=BLOCK_SIZE)

    pos = context_len - 1
    torch.manual_seed(0)
    tt_in = ttnn.from_torch(
        torch.randn(1, 1, 1, hf_config.hidden_size) * 0.02,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    current_pos = ttnn.from_torch(
        torch.tensor([pos], dtype=torch.int32),
        dtype=ttnn.int32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    median, lo, hi = _traced_decode_median(
        mesh_device,
        lambda: MC.decoder_layer_decode_multichip(
            tt_in, weights, config, ctx, cos_cache, sin_cache, kv_cache, current_pos, pos
        ),
    )
    logger.info(f"MULTICHIP 1x4 traced decode ctx={context_len:>5}: median {median:7.4f} ms")
    _write_decode_row(
        MC_DOC_DIR / "perf_decode.csv",
        {
            "context_len": context_len,
            "median_ms": round(median, 4),
            "min_ms": round(lo, 4),
            "max_ms": round(hi, 4),
            "tok_per_s_per_layer": round(1e3 / median, 1),
            "iters": DECODE_ITERS,
        },
    )
    assert median > 0
