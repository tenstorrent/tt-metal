# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
BGE-M3 fully-on-device pipeline benchmark — trace-capture forward timing.

Same mechanism as ``perf.py`` (trace capture + replay), which removes host-side
dispatch overhead so the measured wall time reflects the *unadulterated forward
pass* on the device. This is NOT the Tracy device profiler — Tracy
(``tracy_perf.py``) is for per-op latency breakdowns; this is for end-to-end
forward timing.

What this adds over ``perf.py``:

  * The ENTIRE production inner loop is captured into ONE trace, all on device:
        1. D2D copy:  source token buffers (device) -> graph input slots (device)
        2. Forward:   encoder + CLS pooling (pooling head fused in the graph)
        3. D2D copy:  pooled embedding (device) -> our output buffer (device)
    Nothing in the timed region touches the host, and there is no device->host
    readback in the loop (the embedding is left on device for a downstream stage).
  * Device-side CLS pooling, which ``perf.py`` does not do.
  * Random token ids drawn uniformly from BGE-M3's full vocab.
  * Parametrized on (batch_size, seq_len, dtype, unroll).

Because the whole pipeline is one trace, ``execute_trace(blocking=True)`` runs it
as a single device command; timing that call with a host clock measures the
device forward-pass time with the per-op dispatch overhead already removed.

Knobs (pytest params):
  * dtype  — ``bf8`` (bfloat8_b, the default config) or ``bf16``.
  * unroll — how many back-to-back pipelines are captured into ONE trace. unroll=1
             is the single-pipeline trace; higher K amortizes the per-replay host
             enqueue+sync across K forwards (amortized ms = replay / K). Reveals
             how much fixed dispatch overhead remains, esp. at small batch.

Output:
  Besides the console log, each run writes its amortized ms/forward into 2D CSV
  grids (rows = seq_len, cols = batch_size) under ``generated/bge_m3_perf/`` —
  one file per metric: ``mean``, ``p50``, ``p99``. The file name encodes the
  unroll factor, dtype, and metric, e.g. ``onchip_perf_unroll4_bf8_p99.csv``.
  Cells fill in as you run more (batch, seq_len) combinations for that
  (unroll, dtype); unrun cells stay blank.

``NUM_ITERATIONS`` (replays per config, default 2000) can be overridden via the
``BGE_PERF_ITERS`` env var — e.g. ``BGE_PERF_ITERS=3`` for a quick functional check.

Usage:
    TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/perf_onchip.py -k "seq512 and bf8 and unroll1" -s
    BGE_PERF_ITERS=3 TT_VISIBLE_DEVICES=0 pytest ...perf_onchip.py -k "seq4096 and bf8 and unroll1 and batch1" -s
    # full grid for one (unroll, dtype) -> a complete CSV:
    TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/perf_onchip.py -k "bf8 and unroll1" -s
"""

import csv
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.wormhole.bge_m3.tt.common import create_tt_model
from models.demos.wormhole.bge_m3.tt.model_config import determine_device_name

# Replays per config. Override with env var for quick checks, e.g.
# BGE_PERF_ITERS=3 pytest ... — defaults to 2000.
NUM_ITERATIONS = int(os.environ.get("BGE_PERF_ITERS", "2000"))
WARMUP_ITERS = 3

# Grid axes — also define the CSV row/column order.
ALL_SEQ_LENS = [128, 512, 1024, 4096]
ALL_BATCHES = [1, 8, 16, 32]

PERF_OUT_DIR = "generated/bge_m3_perf"
_DTYPE_STR = {ttnn.bfloat8_b: "bf8", ttnn.bfloat16: "bf16"}


# ──────────────────────────────────────────────────────────────────────────────
# Inputs — random token ids within BGE-M3's vocab
# ──────────────────────────────────────────────────────────────────────────────


def prepare_random_inputs(vocab_size, batch_size, seq_len, pad_token_id):
    """Random token ids in [0, vocab_size); all positions valid (no padding)."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    # All-valid mask -> RoBERTa position ids = pad_token_id + (1..S)
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    position_ids = (torch.cumsum(mask, dim=1) * mask + pad_token_id).to(torch.long)

    # Additive attention mask [B, 1, S, S]; all-valid -> all zeros.
    additive_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    return {
        "input_ids": input_ids,
        "attention_mask": additive_mask,
        "token_type_ids": token_type_ids,
        "position_ids": position_ids,
    }


def to_device_tensors(inputs, mesh_device, mask_dtype):
    """Materialize the four inputs as persistent device tensors."""

    def ids_to_dev(t):
        return ttnn.from_torch(t.int(), device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    return {
        "input_ids": ids_to_dev(inputs["input_ids"]),
        "attention_mask": ttnn.from_torch(
            inputs["attention_mask"],
            device=mesh_device,
            dtype=mask_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        "token_type_ids": ids_to_dev(inputs["token_type_ids"]),
        "position_ids": ids_to_dev(inputs["position_ids"]),
    }


_IN_KEYS = ("input_ids", "attention_mask", "token_type_ids", "position_ids")


# ──────────────────────────────────────────────────────────────────────────────
# 2D result CSVs — rows = seq_len, cols = batch_size; one file per (unroll, dtype, metric)
# ──────────────────────────────────────────────────────────────────────────────


def _percentile(sorted_vals, q):
    """q-th percentile (q in [0,100]) of an already-sorted list, linear interp."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] * (1 - (pos - lo)) + sorted_vals[hi] * (pos - lo)


def _write_grid_cell(path, seq_len, batch_size, value):
    """Read-modify-write one cell of a (seq_len x batch_size) grid CSV in place."""
    grid = {s: {b: "" for b in ALL_BATCHES} for s in ALL_SEQ_LENS}
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            cols = [int(c) for c in header[1:]] if header else []
            for row in reader:
                if not row:
                    continue
                s = int(row[0])
                for c, cell in zip(cols, row[1:]):
                    if s in grid and c in grid[s]:
                        grid[s][c] = cell
    grid[seq_len][batch_size] = f"{value:.3f}"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len\\batch"] + ALL_BATCHES)
        for s in ALL_SEQ_LENS:
            writer.writerow([s] + [grid[s][b] for b in ALL_BATCHES])


def write_perf_grid_cells(dtype_str, unroll, seq_len, batch_size, metrics):
    """Write this (seq_len, batch) cell into one 2D-grid CSV per metric.

    `metrics` is {name: value_ms}. Each metric gets its own file
    ``onchip_perf_unroll{K}_{dtype}_{name}.csv`` (rows=seq_len, cols=batch_size).
    Existing cells are preserved, so a -k sweep fills each grid incrementally.
    Returns {name: path}.
    """
    os.makedirs(PERF_OUT_DIR, exist_ok=True)
    paths = {}
    for name, value in metrics.items():
        path = os.path.join(PERF_OUT_DIR, f"onchip_perf_unroll{unroll}_{dtype_str}_{name}.csv")
        _write_grid_cell(path, seq_len, batch_size, value)
        paths[name] = path
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Introspection util — dump the compute kernel configs the model was built with
# ──────────────────────────────────────────────────────────────────────────────


def _fmt_ckc(ckc):
    """One-line repr of a (Wormhole)ComputeKernelConfig, or str() for anything else."""
    if ckc is None or not hasattr(ckc, "math_fidelity"):
        return str(ckc)
    return (
        f"fidelity={str(ckc.math_fidelity).split('.')[-1]:<5} "
        f"fp32_dst={str(ckc.fp32_dest_acc_en):<5} "
        f"packer_l1={str(ckc.packer_l1_acc):<5} "
        f"approx={str(ckc.math_approx_mode):<5} "
        f"dst_full_sync={ckc.dst_full_sync_en}"
    )


def dump_compute_kernel_configs(model, log=logger):
    """Log every per-op compute kernel config the model was built with.

    Reads them off encoder layer 0 (all layers are identical) plus the norms.
    These are auto-derived by Optimizations.build from (device, batch, seq_len,
    dtype) — the benchmark never sets them — so this documents exactly which
    fidelity / fp32-accumulation each op uses for the shape being run.
    Returns a {label: WormholeComputeKernelConfig} dict for programmatic use.
    """
    found = {}

    def walk(label, cfg):
        if cfg is None:
            return
        for k in sorted(vars(cfg)):
            if "compute_kernel" in k:
                v = getattr(cfg, k)
                found[f"{label}.{k}"] = v
                log.info(f"    {label}.{k:<26} {_fmt_ckc(v)}")

    log.info("Compute kernel configs (encoder layer 0 — representative of all layers):")
    layer0 = model.layers[0]
    walk("attn", layer0.attention.config)
    walk("mlp", layer0.feed_forward.config)
    walk("attn_norm", getattr(getattr(layer0, "attention_norm", None), "config", None))
    walk("ff_norm", getattr(getattr(layer0, "feed_forward_norm", None), "config", None))
    walk("emb_norm", getattr(getattr(model, "embedding_norm", None), "config", None))
    return found


# ──────────────────────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "unroll",
    [1, 4, 16],
    ids=["unroll1", "unroll4", "unroll16"],
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
    ids=["bf8", "bf16"],
)
@pytest.mark.parametrize("seq_len", ALL_SEQ_LENS, ids=[f"seq{s}" for s in ALL_SEQ_LENS])
@pytest.mark.parametrize("batch_size", ALL_BATCHES, ids=[f"batch{b}" for b in ALL_BATCHES])
@pytest.mark.parametrize(
    "device_params",
    # Larger trace region: the unrolled trace records `unroll` copies of the
    # whole pipeline's command stream, so the buffer must hold up to 16x.
    [{"trace_region_size": 200_000_000, "num_command_queues": 1}],
    indirect=True,
)
def test_bge_m3_onchip_traced_pipeline(mesh_device, batch_size, seq_len, dtype, unroll):
    """Capture `unroll` back-to-back (D2D-in -> encode+pool -> D2D-out) pipelines
    into ONE trace; replay it and report amortized per-forward time.

    unroll=1 reproduces the single-pipeline trace. unroll=K records K pipelines
    back-to-back, so one execute_trace runs K forwards with a single host enqueue
    + sync amortized across all K — driving the residual per-replay dispatch
    overhead toward zero. Amortized ms = (replay time) / K.

    Writes the amortized ms/forward into a 2D grid CSV (rows=seq_len, cols=batch)
    named by (unroll, dtype) — see write_perf_grid_cell.
    """
    device_name = determine_device_name(mesh_device)[0]

    # ── Build model with CLS pooling fused into the graph ────────────────────
    logger.info(f"Building model (on-chip traced pipeline): B{batch_size} S{seq_len} {dtype} {device_name}")
    t0 = time.perf_counter()
    model_args, model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
        dtype=dtype,
        pooling="cls",
    )
    build_time = time.perf_counter() - t0
    mask_dtype = model_args.attention_mask_dtype
    vocab_size = model_args.vocab_size
    logger.info(f"Model built in {build_time:.1f}s  (vocab_size={vocab_size}, mask_dtype={mask_dtype})")
    dump_compute_kernel_configs(model)  # self-document the auto-derived configs for this shape/dtype

    # ── Persistent device buffers (allocated ONCE) ───────────────────────────
    # source:     where request tokens already live on device (upstream producer)
    # graph_in:   the fixed input slots the trace reads
    # out_buffer: our consumer's buffer (a downstream on-device stage reads this)
    host_inputs = prepare_random_inputs(vocab_size, batch_size, seq_len, model_args.pad_token_id)
    source = to_device_tensors(host_inputs, mesh_device, mask_dtype)
    graph_in = to_device_tensors(host_inputs, mesh_device, mask_dtype)

    def pipeline(out_buffer):
        """The on-device pipeline. Returns out_buffer (allocates it if None)."""
        # 1. D2D: source -> graph input slots
        for k in _IN_KEYS:
            ttnn.copy(source[k], graph_in[k])
        # 2. encode + CLS pool  -> [B, 1, 1, D]
        pooled = model.forward(**graph_in)
        # 3. D2D: pooled embedding -> our output buffer
        if out_buffer is None:
            out_buffer = ttnn.clone(pooled, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            ttnn.copy(pooled, out_buffer)
        ttnn.deallocate(pooled)
        return out_buffer

    # ── Warmup (eager) — JIT-compile kernels + allocate out_buffer ───────────
    # Compilation synchronizes the device, which is illegal during trace capture,
    # so it must happen first (eager).
    logger.info("Compiling + warming up (eager)...")
    t0 = time.perf_counter()
    out_buffer = pipeline(None)  # first call compiles and allocates out_buffer
    ttnn.synchronize_device(mesh_device)
    compile_time = time.perf_counter() - t0
    for _ in range(WARMUP_ITERS):
        pipeline(out_buffer)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"Compile (1st run): {compile_time:.2f}s")

    # ── Capture `unroll` pipelines back-to-back as one trace ─────────────────
    # Each iteration overwrites graph_in (from source) and out_buffer, so the K
    # iterations chain through those addresses and run sequentially on device —
    # K real forwards under a single execute_trace.
    logger.info(f"Capturing trace: {unroll}x (D2D-in + encode + CLS pool + D2D-out)...")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(unroll):
        pipeline(out_buffer)  # records all ops into the trace; writes into out_buffer
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # ── Trace warmup (untimed) ───────────────────────────────────────────────
    for _ in range(WARMUP_ITERS):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)

    # ── Benchmark — time the blocking replay of the unrolled trace ───────────
    # Each replay runs `unroll` forwards as one device command with a single
    # host enqueue + sync. Amortized per-forward = replay_time / unroll.
    logger.info(f"Timing {NUM_ITERATIONS} replays of the {unroll}x trace (B{batch_size} S{seq_len})")
    times = []
    for _ in range(NUM_ITERATIONS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
        times.append(time.perf_counter() - t0)

    ttnn.release_trace(mesh_device, trace_id)

    # ── Results ──────────────────────────────────────────────────────────────
    replay_ms = sorted(t * 1000 for t in times)  # per-replay wall (= `unroll` forwards)
    amort_ms = [r / unroll for r in replay_ms]  # amortized per single forward (sorted)
    mean_ms = sum(amort_ms) / len(amort_ms)
    p50_ms = _percentile(amort_ms, 50)
    p99_ms = _percentile(amort_ms, 99)
    total_tokens = batch_size * seq_len

    # ── Write the 2D grid CSVs (rows=seq_len, cols=batch); one file per metric ─
    dtype_str = _DTYPE_STR.get(dtype, str(dtype))
    csv_paths = write_perf_grid_cells(
        dtype_str,
        unroll,
        seq_len,
        batch_size,
        {"mean": mean_ms, "p50": p50_ms, "p99": p99_ms},
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  BGE-M3 on-chip traced pipeline  ({device_name})")
    logger.info("=" * 70)
    logger.info(f"  Batch size:        {batch_size}")
    logger.info(f"  Seq length:        {seq_len}")
    logger.info(f"  dtype:             {dtype}")
    logger.info(f"  Unroll factor:     {unroll}  (forwards per replay)")
    logger.info(f"  Total tokens:      {total_tokens}")
    logger.info(f"  Replays:           {NUM_ITERATIONS}")
    logger.info("  Pipeline:          D2D-in + encode + CLS pool + D2D-out (all on device)")
    logger.info("-" * 70)
    logger.info(f"  Model build time:  {build_time:.1f}s")
    logger.info(f"  Compile (1st run): {compile_time:.2f}s")
    logger.info("-" * 70)
    logger.info(f"  Avg replay time:   {sum(replay_ms) / len(replay_ms):.3f} ms  ({unroll} forwards)")
    logger.info(f"  Amortized / fwd:   mean {mean_ms:.3f} | p50 {p50_ms:.3f} | p99 {p99_ms:.3f} ms")
    logger.info(f"  Mean embeddings/s: {batch_size / (mean_ms / 1000):.1f}")
    logger.info(f"  Mean tokens/s:     {total_tokens / (mean_ms / 1000):.0f}")
    for _name, _p in csv_paths.items():
        logger.info(f"  CSV [{_name}]:        {_p}  (cell = amortized ms/fwd)")
    logger.info("=" * 70)

    # ── Sanity: read the embedding back ONCE (outside timing) ─────────────────
    emb = ttnn.to_torch(out_buffer).float().reshape(batch_size, -1)
    logger.info(f"  output shape: {tuple(emb.shape)}  (batch x embedding_dim, raw CLS pooled)")
    assert emb.shape == (batch_size, 1024), "unexpected pooled embedding shape"
