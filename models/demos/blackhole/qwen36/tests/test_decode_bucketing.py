# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verification for the qwen36 decode-batch bucketing fix (branch atupe/qwen36-bucketing-fix).

Three questions, each answered by measurement rather than by reading the comments:

1. ``test_bucket_selection`` (no device): does the bucket-picking arithmetic in
   ``qwen36_vllm.decode_forward`` choose the right width for the padded batches the
   vLLM runner actually produces?
2. ``test_bucketed_decode_matches_full_width`` (device): is a width-B decode
   numerically equivalent to row 0 of a width-Bmax decode? Bucketing is only a
   valid optimization if it is.
3. ``test_decode_width_scaling`` (device): how much step time does narrowing the
   width actually save at 8k, and which layer type does the saving come from?
   This is the one that explains the residual gap.
"""
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

# n_layers=8 of the real checkpoint = layer_types[:8] = 6 linear_attention + 2 full_attention.
N_LAYERS = 8
N_GDN_IN_SLICE = 6
N_ATTN_IN_SLICE = 2
# Full Qwen3.6-27B: 64 layers, full_attention_interval=4 -> 48 GDN + 16 full attention.
FULL_GDN = 48
FULL_ATTN = 16

BLOCK = 64
# ISL under test. Override with QWEN36_BUCKET_TEST_CTX to compare against a server run at a
# different ISL (the benchmark sweep was trimmed to 4096).
CTX = int(os.environ.get("QWEN36_BUCKET_TEST_CTX", "8192"))
BPU = CTX // BLOCK  # blocks per user


def _pick_bucket(tokens, start_pos, width):
    """Bucket arithmetic lifted verbatim from qwen36_vllm.decode_forward (the code under test)."""
    num_active = int((start_pos != -1).sum()) if start_pos is not None else width
    num_active = max(1, min(num_active, width))
    return min(width, 1 << max(0, (num_active - 1).bit_length()))


def _padded_decode_batch(num_active, width):
    """Reproduce what tt_model_runner.py:907-926 hands to decode_forward."""
    tokens = torch.cat(
        [
            torch.randint(1, 1000, (num_active, 1), dtype=torch.int32),
            torch.zeros(width - num_active, 1, dtype=torch.int32),
        ]
    )
    positions = torch.cat(
        [torch.full((num_active,), 4096, dtype=torch.int32), torch.ones(width - num_active, dtype=torch.int32) * -1]
    )
    return tokens, positions


def test_bucket_selection():
    """The runner pads to max_num_seqs and marks pad rows with position -1."""
    for width in (8, 32):
        for num_active in range(1, width + 1):
            tokens, positions = _padded_decode_batch(num_active, width)
            got = _pick_bucket(tokens, positions, width)
            expect = min(width, 1 << max(0, (num_active - 1).bit_length()))
            assert got == expect, f"width={width} active={num_active}: bucket {got} != {expect}"
            assert got >= num_active, f"bucket {got} drops active rows (active={num_active})"
    # The case this whole exercise is about.
    tokens, positions = _padded_decode_batch(1, 8)
    assert _pick_bucket(tokens, positions, 8) == 1
    logger.info("PASSED: bucket selection picks smallest pow2 >= num_active and never drops active rows")


def _build(mesh_device, bmax):
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=bmax, max_seq_len=CTX, n_layers=N_LAYERS, hf_model="Qwen/Qwen3.6-27B"
    )
    num_blocks = bmax * BPU
    kv_shape = (num_blocks, model.args.n_local_kv_heads, BLOCK, model.args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=bmax)
    page_table = torch.stack(
        [torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(bmax)]
    )  # [bmax, BPU]
    return model, page_table


def _decode_once(model, tokens, positions, page_table):
    dev = model.prepare_inputs_decode(tokens, positions, page_table)
    out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
    return out


@torch.no_grad()
@parametrize_mesh_tp()
def test_bucketed_decode_matches_full_width(mesh_device, reset_seeds, ensure_gc):
    """Each bucket must equal the same rows of a width-8 decode.

    Every run starts from zeroed GDN state and uses the same token, position,
    and page-table row for each active user. This is the correctness
    precondition for bucketing and for active-prefix recurrent-state writes.
    """
    BMAX = 8
    model, page_table = _build(mesh_device, BMAX)
    vocab = model.args.vocab_size

    tok0, pos0 = 42, 4096
    # --- width-8 reference: user 0 real, users 1..7 also real (distinct tokens) ---
    model.reset_tp()
    tokens8 = torch.tensor([[tok0]] + [[100 + u] for u in range(1, BMAX)], dtype=torch.int32)
    pos8 = torch.full((BMAX,), pos0, dtype=torch.int32)
    out8 = _decode_once(model, tokens8, pos8, page_table)
    lg8 = model.process_output_decode(out8, BMAX)[:, 0, :vocab].float()

    for width in (1, 2, 4):
        model.reset_tp()
        out = _decode_once(model, tokens8[:width], pos8[:width], page_table[:width])
        lg = model.process_output_decode(out, width)[:, 0, :vocab].float()
        _, pcc = comp_pcc(lg8[:width], lg, 0.99)
        logger.info(f"width-{width} vs width-8 rows [0:{width}] logits PCC = {pcc}")
        assert float(pcc) >= 0.99, f"bucketed width-{width} does not match full-width rows [0:{width}]: PCC {pcc}"
    logger.info("PASSED: bucket widths 1, 2, and 4 are numerically equivalent to full width")


@torch.no_grad()
@parametrize_mesh_tp()
def test_decode_width_scaling(mesh_device, reset_seeds, ensure_gc):
    """Measure step time vs decode width at 8k, and attribute the delta to GDN vs the rest.

    Uses the branch's own GDN_PHASE_TIMING hooks for the GDN share. Reports a projection
    to the full 64-layer model so the numbers can be compared to the served tok/s.
    """
    os.environ["GDN_PHASE_TIMING"] = "1"
    import models.demos.blackhole.qwen36.tt.gdn.tp as gdn_tp

    BMAX = 8
    ITERS = 20
    model, page_table = _build(mesh_device, BMAX)

    results = {}
    for width in (1, 2, 4, 8):
        model.reset_tp()
        tokens = torch.tensor([[100 + u] for u in range(width)], dtype=torch.int32)
        positions = torch.full((width,), CTX - 64, dtype=torch.int32)
        pt = page_table[:width]

        # warm: first call compiles programs for this width
        _decode_once(model, tokens, positions, pt)
        ttnn.synchronize_device(mesh_device)

        gdn_tp._pt_reset()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            _decode_once(model, tokens, positions, pt)
        ttnn.synchronize_device(mesh_device)
        step_ms = (time.perf_counter() - t0) * 1000.0 / ITERS

        phases = gdn_tp._pt_report()
        gdn_ms = sum(v[0] for v in phases.values()) / ITERS
        results[width] = (step_ms, gdn_ms, {k: v[0] / ITERS for k, v in phases.items()})
        logger.info(f"width={width}: step={step_ms:.2f} ms  gdn={gdn_ms:.2f} ms  phases={results[width][2]}")

    logger.info("=== per-step time for an 8-layer slice (6 GDN + 2 attn) ===")
    for w, (step, gdn, _) in results.items():
        logger.info(f"  width={w}: total={step:.2f} ms   gdn={gdn:.2f} ms   non-gdn={step - gdn:.2f} ms")

    s8, g8, _ = results[8]
    s1, g1, _ = results[1]
    d_total, d_gdn = s8 - s1, g8 - g1
    d_other = d_total - d_gdn
    logger.info(f"8-layer slice: width8-width1 delta total={d_total:.2f} ms (gdn={d_gdn:.2f}, other={d_other:.2f})")
    # Project the per-layer-type deltas onto the real 64-layer model.
    proj = d_gdn / N_GDN_IN_SLICE * FULL_GDN + d_other / N_ATTN_IN_SLICE * FULL_ATTN
    logger.info(
        f"PROJECTED full-64-layer width8->width1 saving = {proj:.1f} ms/step "
        f"(gdn {d_gdn / N_GDN_IN_SLICE * FULL_GDN:.1f} ms, attn+other {d_other / N_ATTN_IN_SLICE * FULL_ATTN:.1f} ms)"
    )
    assert results[1][0] > 0


def _parametrize_traced(max_tp=4, trace_bytes=1073741824):
    """Same mesh/fabric params as parametrize_mesh_tp, plus a trace region (needed to capture)."""
    from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

    shape = _resolve_mesh_shape(max_tp)

    def decorator(fn):
        fn = pytest.mark.parametrize(
            "device_params",
            [
                {
                    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                    "l1_small_size": GDN_CONV1D_L1_SMALL_SIZE,
                    "trace_region_size": trace_bytes,
                }
            ],
            indirect=True,
        )(fn)
        fn = pytest.mark.parametrize("mesh_device", [pytest.param(shape, id=f"{shape[0]}x{shape[1]}")], indirect=True)(
            fn
        )
        return fn

    return decorator


@torch.no_grad()
@_parametrize_traced()
def test_gdn_prefix_write_trace(mesh_device, reset_seeds, ensure_gc):
    """Prove a trace-safe prefix update for the fixed-capacity GDN state.

    The recurrent state is TILE/DRAM ``[Bmax,Nv,Dk,Dv]``. A bucketed decode
    produces only ``[B,Nv,Dk,Dv]``. The current model preserves idle rows by
    slicing them, concatenating them with the new active state, and copying the
    complete Bmax tensor. This test instead height-shards the active state in L1
    and uses ``slice_write`` to update only rows ``[0:B]`` of the original
    fixed-address TILE buffer.

    The sharded input is important: the non-sharded TILE slice_write wrapper
    converts the output through ROW_MAJOR buffers, which is not suitable for a
    trace-persistent state address. The tiled-sharded path writes directly into
    the supplied interleaved TILE output.
    """
    BMAX, NV, DK, DV = 8, 12, 128, 128
    B = int(os.environ.get("QWEN36_PREFIX_WRITE_WIDTH", "1"))
    assert B in (1, 2, 4, 8), f"QWEN36_PREFIX_WRITE_WIDTH must be 1, 2, 4, or 8; got {B}"
    iters = int(os.environ.get("QWEN36_PREFIX_WRITE_ITERS", "100"))
    assert iters > 0

    # 48 cores make every supported width tile-aligned:
    # flattened NHW = B*12*128, shard height = NHW/48 = B*32.
    grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 5),
            )
        }
    )
    nhw = B * NV * DK
    assert nhw % 48 == 0
    shard_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (nhw // 48, DV), ttnn.ShardOrientation.ROW_MAJOR),
    )

    # Give every inactive row a distinct value so an accidental overwrite is
    # visible. Use fp32 because that is the production recurrent-state dtype.
    baseline = torch.arange(BMAX, dtype=torch.float32).reshape(BMAX, 1, 1, 1).expand(BMAX, NV, DK, DV).contiguous()
    active = (100.0 + torch.arange(B, dtype=torch.float32)).reshape(B, 1, 1, 1).expand(B, NV, DK, DV).contiguous()
    expected = baseline.clone()
    expected[:B] = active

    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    state = ttnn.from_torch(
        baseline,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    active_interleaved = ttnn.from_torch(
        active,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    active_sharded = ttnn.to_memory_config(active_interleaved, shard_memcfg)
    baseline_host = ttnn.from_torch(
        baseline,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=rep,
    )

    start = [0, 0, 0, 0]
    end = [B, NV, DK, DV]
    step = [1, 1, 1, 1]

    # Compile and check the eager operation first.
    ttnn.experimental.slice_write(active_sharded, state, start, end, step)
    ttnn.synchronize_device(mesh_device)
    eager = ttnn.to_torch(ttnn.get_device_tensors(state)[0]).reshape(BMAX, NV, DK, DV)
    assert torch.equal(eager, expected), "eager prefix write changed active or inactive GDN rows"

    # Reset the same output buffer, capture against its stable address, then
    # reset once more so correctness depends on the replay.
    ttnn.copy_host_to_device_tensor(baseline_host, state)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    ttnn.experimental.slice_write(active_sharded, state, start, end, step)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.copy_host_to_device_tensor(baseline_host, state)
    ttnn.synchronize_device(mesh_device)

    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    replay = ttnn.to_torch(ttnn.get_device_tensors(state)[0]).reshape(BMAX, NV, DK, DV)
    assert torch.equal(replay, expected), "trace replay changed active or inactive GDN rows"

    # Time only replay of the prefix-write trace. Rewriting the same prefix is
    # idempotent, so no reset is needed between iterations.
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    prefix_ms = (time.perf_counter() - t0) * 1000.0 / iters
    logger.info(f"GDN_PREFIX_WRITE_RESULT width={B} bmax={BMAX} dtype=fp32 " f"cores=48 ms_per_write={prefix_ms:.6f}")

    ttnn.release_trace(mesh_device, tid)


@torch.no_grad()
@_parametrize_traced()
@pytest.mark.parametrize("n_layers", [None], ids=["all64"])
def test_decode_width_scaling_traced(mesh_device, n_layers, reset_seeds, ensure_gc):
    """DEVICE time vs decode width, on the traced path the server actually runs.

    The eager measurement in test_decode_width_scaling is host-dispatch bound, which can
    mask a device-side width effect. Trace replay removes host dispatch, so this isolates
    device time. Run at the FULL layer count so ms/step is directly comparable to the
    served tok/s.
    """
    os.environ.pop("GDN_PHASE_TIMING", None)
    from models.tt_transformers.tt.common import copy_host_to_device

    BMAX = 8
    ITERS = 50
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=BMAX, max_seq_len=CTX, n_layers=n_layers, hf_model="Qwen/Qwen3.6-27B"
    )
    num_blocks = BMAX * BPU
    kv_shape = (num_blocks, model.args.n_local_kv_heads, BLOCK, model.args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    page_table_full = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    n_layers_actual = len(model.layers)
    logger.info(f"model built with {n_layers_actual} layers, ctx={CTX}, Bmax={BMAX}")

    out = {}
    # QWEN36_BUCKET_TEST_WIDTHS restricts the widths measured. Needed to run this harness against a
    # tree WITHOUT the bucketing change, whose decode graph only accepts the full width: width 1
    # there dies in reshape (new_volume == old_volume). "8" then yields the full-width baseline the
    # bucketing saving must be measured against.
    widths = tuple(int(w) for w in os.environ.get("QWEN36_BUCKET_TEST_WIDTHS", "1,8").split(","))
    for width in widths:
        model.reset_tp()
        tokens = torch.tensor([[100 + u] for u in range(width)], dtype=torch.int32)
        positions = torch.full((width,), CTX - 64, dtype=torch.int32)
        pt = page_table_full[:width]

        # compile run (eager) so trace capture records only pre-compiled programs
        dev0 = model.prepare_inputs_decode(tokens, positions, pt)
        model.ttnn_decode_forward(dev0[0], dev0[1], rot_mat_idxs=dev0[2], page_table=dev0[3])
        ttnn.synchronize_device(mesh_device)

        host = model.prepare_decode_inputs_host(tokens, positions, page_table=pt)
        dev = copy_host_to_device(host, mesh_device=mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)  # warm replay
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) * 1000.0 / ITERS
        out[width] = ms
        logger.info(f"TRACED width={width}: {ms:.3f} ms/step device time  ({1000.0 / ms:.2f} tok/s ceiling)")
        ttnn.release_trace(mesh_device, tid)

    if 1 in out and 8 in out:
        d = out[8] - out[1]
        logger.info(
            f"RESULT ({n_layers_actual} layers): traced device step width8={out[8]:.3f} ms, width1={out[1]:.3f} ms, "
            f"saving={d:.3f} ms ({100.0 * d / out[8]:.1f}%)  -> tok/s {1000.0 / out[8]:.2f} vs {1000.0 / out[1]:.2f}"
        )
    assert all(v > 0 for v in out.values())


@torch.no_grad()
@_parametrize_traced()
def test_decode_capacity_width1_traced(mesh_device, reset_seeds, ensure_gc):
    """Capacity-only device cost at live width 1 (true Bmax=1 vs Bmax=8).

    Set ``QWEN36_CAPACITY_TEST_BMAX`` to 1 or 8 in separate pytest processes.
    On-device logits; no sampling replay (both round to 32 slots). Optional:
    ``N_LAYERS``, ``LAYER_INDEX`` (3 = first full-attn), ``EAGER_PROFILE=1``
    for per-op Tracy rows instead of aggregate trace time.
    """
    from tracy import signpost

    from models.tt_transformers.tt.common import copy_host_to_device

    bmax = int(os.environ.get("QWEN36_CAPACITY_TEST_BMAX", "8"))
    assert bmax in (1, 8), f"QWEN36_CAPACITY_TEST_BMAX must be 1 or 8, got {bmax}"
    n_layers_env = os.environ.get("QWEN36_CAPACITY_TEST_N_LAYERS")
    n_layers = int(n_layers_env) if n_layers_env is not None else None
    assert n_layers is None or 1 <= n_layers <= 64, f"QWEN36_CAPACITY_TEST_N_LAYERS must be in [1,64], got {n_layers}"
    layer_index_env = os.environ.get("QWEN36_CAPACITY_TEST_LAYER_INDEX")
    layer_indices = [int(layer_index_env)] if layer_index_env is not None else None
    assert not (n_layers is not None and layer_indices is not None), (
        "Set only one of QWEN36_CAPACITY_TEST_N_LAYERS or " "QWEN36_CAPACITY_TEST_LAYER_INDEX"
    )
    assert (
        layer_indices is None or 0 <= layer_indices[0] < 64
    ), f"QWEN36_CAPACITY_TEST_LAYER_INDEX must be in [0,63], got {layer_indices}"
    trials = int(os.environ.get("QWEN36_CAPACITY_TEST_TRIALS", "5"))
    iters = int(os.environ.get("QWEN36_CAPACITY_TEST_ITERS", "20"))
    warmup = int(os.environ.get("QWEN36_CAPACITY_TEST_WARMUP", "3"))
    eager_profile = os.environ.get("QWEN36_CAPACITY_TEST_EAGER_PROFILE") == "1"
    assert trials > 0 and iters > 0 and warmup >= 0

    # Start decode at CTX and leave enough RoPE/KV capacity for every captured,
    # warmup, and measured replay. Round the page-table width to 8 blocks because
    # each int32 row-major stick must be 32-byte aligned for paged SDPA.
    max_seq_len = CTX + 1 + warmup + trials * iters + 8
    blocks_per_user = (max_seq_len + BLOCK - 1) // BLOCK
    blocks_per_user = ((blocks_per_user + 7) // 8) * 8

    model = Qwen36Model.from_pretrained(
        mesh_device,
        max_batch_size=bmax,
        max_seq_len=max_seq_len,
        n_layers=n_layers,
        layer_indices=layer_indices,
        hf_model="Qwen/Qwen3.6-27B",
    )
    assert model.sampling is not None, "on-device logits require the sampling module"
    sampler_batch = model.sampling.tt_sampling.max_batch_size
    assert sampler_batch == 32, (
        f"capacity comparison expects the same 32-slot sampler, got {sampler_batch} " f"for model capacity {bmax}"
    )

    num_blocks = bmax * blocks_per_user
    kv_shape = (num_blocks, model.args.n_local_kv_heads, BLOCK, model.args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=bmax)
    page_table_full = torch.stack(
        [torch.arange(u * blocks_per_user, (u + 1) * blocks_per_user, dtype=torch.int32) for u in range(bmax)]
    )
    page_table = page_table_full[:1]
    tokens = torch.tensor([[100]], dtype=torch.int32)
    positions = torch.tensor([CTX], dtype=torch.int32)

    model.reset_tp()
    dev0 = model.prepare_inputs_decode(tokens, positions, page_table)
    model.ttnn_decode_forward(
        dev0[0],
        dev0[1],
        rot_mat_idxs=dev0[2],
        page_table=dev0[3],
        on_device_logits=True,
    )
    ttnn.synchronize_device(mesh_device)

    if eager_profile:
        # Use fresh inputs after the compile run, then profile exactly one eager
        # decode. No sampler is invoked: this produces one logits row, not a
        # generated output-token sequence.
        profile_positions = torch.tensor([CTX + 1], dtype=torch.int32)
        profile_dev = model.prepare_inputs_decode(tokens, profile_positions, page_table)
        ttnn.synchronize_device(mesh_device)
        signpost("start")
        t0 = time.perf_counter()
        model.ttnn_decode_forward(
            profile_dev[0],
            profile_dev[1],
            rot_mat_idxs=profile_dev[2],
            page_table=profile_dev[3],
            on_device_logits=True,
        )
        ttnn.synchronize_device(mesh_device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        signpost("stop")
        logger.info(
            f"CAPACITY_EAGER_PROFILE_RESULT bmax={bmax} live_width=1 ctx={CTX} "
            f"layer_indices={model.layer_indices} sampler_batch={sampler_batch} elapsed_ms={elapsed_ms:.3f}"
        )
        return

    host = model.prepare_decode_inputs_host(tokens, positions, page_table=page_table)
    dev = copy_host_to_device(host, mesh_device=mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    model.ttnn_decode_forward(
        dev[0],
        dev[1],
        rot_mat_idxs=dev[2],
        page_table=dev[3],
        on_device_logits=True,
    )
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    for _ in range(warmup):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    samples_ms = []
    # These markers isolate only the measured trace replays in Tracy's ops CSV;
    # compile, capture, and warmup operations remain outside the comparison.
    signpost("start")
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(iters):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        samples_ms.append((time.perf_counter() - t0) * 1000.0 / iters)
    signpost("stop")

    median_ms = statistics.median(samples_ms)
    stdev_ms = statistics.pstdev(samples_ms)
    logger.info(
        f"CAPACITY_RESULT bmax={bmax} live_width=1 ctx={CTX} layer_indices={model.layer_indices} "
        f"sampler_batch={sampler_batch} trials={trials} iters_per_trial={iters} "
        f"samples_ms={[round(v, 3) for v in samples_ms]} median_ms={median_ms:.3f} "
        f"stdev_ms={stdev_ms:.3f} range_ms={max(samples_ms) - min(samples_ms):.3f}"
    )

    ttnn.release_trace(mesh_device, tid)
    assert all(v > 0 for v in samples_ms)


@torch.no_grad()
@_parametrize_traced()
def test_decode_step_host_overhead(mesh_device, reset_seeds, ensure_gc):
    """Time our per-step host prep (prepare + H2D) vs device; remainder is vLLM / async.

    Prep was ~0.03 ms (execute_trace overlaps prior device work); the TPOT gap was vLLM-side.
    Device decode continuity later removed per-step staging (see test_async_decode.py).
    """
    from models.tt_transformers.tt.common import copy_host_to_device

    BMAX, ITERS, WIDTH = 8, 50, 1
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=BMAX, max_seq_len=CTX, n_layers=None, hf_model="Qwen/Qwen3.6-27B"
    )
    num_blocks = BMAX * BPU
    kv_shape = (num_blocks, model.args.n_local_kv_heads, BLOCK, model.args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    page_table = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])[:WIDTH]
    tokens = torch.tensor([[100 + u] for u in range(WIDTH)], dtype=torch.int32)
    positions = torch.full((WIDTH,), CTX - 64, dtype=torch.int32)

    model.reset_tp()
    dev0 = model.prepare_inputs_decode(tokens, positions, page_table)
    model.ttnn_decode_forward(dev0[0], dev0[1], rot_mat_idxs=dev0[2], page_table=dev0[3])
    ttnn.synchronize_device(mesh_device)

    host = model.prepare_decode_inputs_host(tokens, positions, page_table=page_table)
    dev = copy_host_to_device(host, mesh_device=mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # (a) pure device: replay only
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    replay_ms = (time.perf_counter() - t0) * 1000.0 / ITERS

    # (b) device + our per-step host input prep, exactly as always-refresh does it
    t0 = time.perf_counter()
    for k in range(ITERS):
        pos_k = torch.full((WIDTH,), CTX - 64 + k, dtype=torch.int32)
        host_k = model.prepare_decode_inputs_host(tokens, pos_k, page_table=page_table)
        copy_host_to_device(host_tensors=host_k, device_tensors=dev)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    full_ms = (time.perf_counter() - t0) * 1000.0 / ITERS

    ours = full_ms - replay_ms
    logger.info(f"replay only            : {replay_ms:.2f} ms/step")
    logger.info(f"replay + our host prep : {full_ms:.2f} ms/step   (our host cost = {ours:.2f} ms)")
    logger.info(
        f"SERVER TPOT was 46.7 ms at 4k/conc-1 -> vLLM-side overhead ~= "
        f"{46.7 - full_ms:.2f} ms (only async scheduling hides that part)"
    )
    ttnn.release_trace(mesh_device, tid)
    assert replay_ms > 0 and full_ms >= replay_ms


@torch.no_grad()
@_parametrize_traced()
def test_bucketed_on_device_sampling_traces(mesh_device, reset_seeds, ensure_gc):
    """Bucketing + ON-DEVICE sampling, alternating widths. This is the SERVED config.

    The served spec sets ``sample_on_device_mode: decode_only``, and the bucketing branch's
    own comment says this combination is untested. The stated failure mode is that
    ``SamplingGenerator._validate_trace_inputs`` binds a captured sampling trace to ONE logits
    tensor by IDENTITY (generator.py:298), so a per-bucket decode trace (each with its own
    logits tensor) needs a per-bucket sampling trace. ``set_trace_bucket`` is meant to provide
    that. Alternating 1 -> 8 -> 1 -> 8 is what would trip it.
    """
    from models.tt_transformers.tt.common import copy_host_to_device

    BMAX = 8
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=BMAX, max_seq_len=CTX, n_layers=N_LAYERS, hf_model="Qwen/Qwen3.6-27B"
    )
    if model.sampling is None:
        pytest.skip("on-device sampling unsupported on this mesh/vocab; nothing to verify")
    if not hasattr(model.sampling, "set_trace_bucket"):
        pytest.fail("SamplingGenerator lacks set_trace_bucket -- the bucketing branch is not applied")

    num_blocks = BMAX * BPU
    kv_shape = (num_blocks, model.args.n_local_kv_heads, BLOCK, model.args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    page_table_full = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])

    # Allocate GDN state ONCE, up front, at Bmax. reset_tp() -> TPGatedDeltaNet.reset_state()
    # REALLOCATES the state buffers (ttnn.from_torch), which would invalidate the baked addresses
    # of any already-captured trace and hang the device on replay. Both traces must stay live here,
    # so state is allocated once and only ever zeroed in place afterwards.
    model.reset_tp()
    _warm = model.prepare_inputs_decode(
        torch.tensor([[7]] * BMAX, dtype=torch.int32),
        torch.full((BMAX,), CTX - 64, dtype=torch.int32),
        page_table_full,
    )
    model.ttnn_decode_forward(_warm[0], _warm[1], rot_mat_idxs=_warm[2], page_table=_warm[3], on_device_logits=True)
    ttnn.synchronize_device(mesh_device)

    # Per-width: a decode trace (stable logits tensor) + a sampling trace bound to it.
    logits_of, dtrace_of, _keep = {}, {}, []
    for width in (1, BMAX):
        model._reset_gdn_state_for_new_sequence()  # in-place; preserves baked trace addresses
        tokens = torch.tensor([[100 + u] for u in range(width)], dtype=torch.int32)
        positions = torch.full((width,), CTX - 64, dtype=torch.int32)
        pt = page_table_full[:width]

        dev0 = model.prepare_inputs_decode(tokens, positions, pt)
        model.ttnn_decode_forward(dev0[0], dev0[1], rot_mat_idxs=dev0[2], page_table=dev0[3], on_device_logits=True)
        ttnn.synchronize_device(mesh_device)

        host = model.prepare_decode_inputs_host(tokens, positions, page_table=pt)
        dev = copy_host_to_device(host, mesh_device=mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        lg = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=True)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        # The trace bakes these input tensors' addresses; keep them referenced for every replay.
        logits_of[width], dtrace_of[width] = lg, tid
        _keep.append(dev)

        # capture the sampling trace for THIS bucket
        model.sampling.set_trace_bucket(width)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        model.sampling.sample(lg, enable_trace=True)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"captured decode+sampling trace for bucket width={width} (logits shape {list(lg.shape)})")

    # Alternate widths. Without per-bucket namespacing this raises on the first switch.
    for step, width in enumerate((1, BMAX, 1, BMAX, 1)):
        model.sampling.set_trace_bucket(width)
        ttnn.execute_trace(mesh_device, dtrace_of[width], cq_id=0, blocking=False)
        tok = model.sampling.sample(logits_of[width], enable_trace=True)
        ttnn.synchronize_device(mesh_device)
        if isinstance(tok, tuple):  # (tokens, log_probs) when log-probs are on
            tok = tok[0]
        t = ttnn.to_torch(tok, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).reshape(-1)
        ids = t[:width].to(torch.int64)
        assert (
            int(ids.min()) >= 0 and int(ids.max()) < model.args.vocab_size
        ), f"step {step} width {width}: sampled ids out of range: {ids.tolist()}"
        logger.info(f"step {step} width={width}: sampled ids {ids.tolist()[: min(4, width)]} OK")

    for tid in dtrace_of.values():
        ttnn.release_trace(mesh_device, tid)
    logger.info("PASSED: per-bucket sampling traces survive alternating widths with on-device sampling")


@torch.no_grad()
@_parametrize_traced(trace_bytes=1073741824)  # exactly the b8 model spec's trace_region_size
def test_all_buckets_fit_trace_region(mesh_device, reset_seeds, ensure_gc):
    """Do FOUR live decode traces (widths 1,2,4,8) + their sampling traces fit in 1 GB?

    warmup_model_decode captures one decode trace per power-of-2 bucket, and the branch never
    releases the non-last ones. Enabling TT_DECODE_BUCKETING therefore multiplies decode-trace
    memory by 4 against the spec's trace_region_size. An overflow is a hard startup failure, so
    this must be checked before turning bucketing on in the served spec.
    """
    from models.tt_transformers.tt.common import copy_host_to_device

    BMAX = 8
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=BMAX, max_seq_len=CTX, n_layers=None, hf_model="Qwen/Qwen3.6-27B"
    )
    num_blocks = BMAX * BPU
    kv_shape = (num_blocks, model.args.n_local_kv_heads, BLOCK, model.args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    page_table_full = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    on_dev = model.sampling is not None

    # Allocate GDN state once (reset_state reallocates and would invalidate live traces).
    model.reset_tp()
    tids = []
    # A captured trace bakes the ADDRESSES of its device input tensors. Those tensors must stay
    # referenced for the whole test, or Python frees them and replaying the trace reads freed
    # memory and hangs the board. Hold every width's inputs (and logits) alive here.
    keepalive = []
    for width in (1, 2, 4, BMAX):
        model._reset_gdn_state_for_new_sequence()
        tokens = torch.tensor([[100 + u] for u in range(width)], dtype=torch.int32)
        positions = torch.full((width,), CTX - 64, dtype=torch.int32)
        pt = page_table_full[:width]

        dev0 = model.prepare_inputs_decode(tokens, positions, pt)
        model.ttnn_decode_forward(dev0[0], dev0[1], rot_mat_idxs=dev0[2], page_table=dev0[3], on_device_logits=on_dev)
        ttnn.synchronize_device(mesh_device)

        host = model.prepare_decode_inputs_host(tokens, positions, page_table=pt)
        dev = copy_host_to_device(host, mesh_device=mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        lg = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=on_dev)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        tids.append(tid)
        keepalive.append((dev, lg))  # must outlive every replay below
        if on_dev:
            model.sampling.set_trace_bucket(width)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            model.sampling.sample(lg, enable_trace=True)
            ttnn.synchronize_device(mesh_device)
        logger.info(f"bucket width={width}: decode{'+sampling' if on_dev else ''} trace captured and live")

    # Capture only: overflow fails at alloc. Do not replay earlier widths after later setup —
    # H2D alloc with live traces corrupts memory / hangs the board. Replay coverage is in
    # test_bucketed_on_device_sampling_traces (inputs bound before any replay).
    assert len(tids) == 4 and len(keepalive) == 4
    logger.info(
        f"PASSED: all {len(tids)} bucket decode traces (+ sampling traces) captured inside the "
        f"1 GB trace_region_size with no allocator failure -- the b8 spec's budget is sufficient"
    )
    for tid in tids:
        ttnn.release_trace(mesh_device, tid)
