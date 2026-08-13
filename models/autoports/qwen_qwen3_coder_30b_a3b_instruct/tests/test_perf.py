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
