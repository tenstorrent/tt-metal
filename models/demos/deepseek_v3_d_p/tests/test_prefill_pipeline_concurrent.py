# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline parallelism running CONCURRENTLY on one galaxy, and its measured throughput.

The stage geometry is a parameter: `PP` is however many stages of the requested `(sp, tp)` shape tile
the 32-chip galaxy, so `8x1` gives PP=4 and `8x2` gives PP=2.

The `(8,4)` mesh is 8 rows x 4 columns and the SP axis is axis 0, so **each column is exactly an
`(8,1)` PP stage** and each column *pair* is an `(8,2)` stage. `create_submeshes` carves them, and
`pp/probe_submesh.py` verified that the SP-axis collectives ring attention needs work *inside* a
submesh, and that two submeshes are independently usable. That makes real pipeline parallelism
reachable in ONE process, without the multi-process rank driver or the D2D socket transport.

**Traced, because eager cannot pipeline.** The first version of this test ran the stages eagerly and
measured 2,501 tok/s — 10x WORSE than single-rank — because eager dispatch is host-bound: one Python
thread still issues all 36 layers' ops, so four submeshes buy nothing and the pipeline costs exactly
what one monolithic eager forward costs. Each stage is therefore captured as a segmented ttnn trace
(`SubDeviceTraceController`, which splits at the MoE sub-device swaps — a single trace cannot span
them), and an iteration is then 4 cheap replays instead of ~thousands of host op issues. Replays are
issued with `blocking=False` so all four are in flight before the one sync.

How the concurrency is obtained without a D2D op: ttnn enqueues asynchronously, so all four stages are
enqueued on their four submeshes BEFORE anything blocks, using a one-iteration lag. The hand-off goes
through the host (3 hops x [1, 1, W/8, 4096] bf16 = 5 MB each), which is why the read-backs are
deliberately deferred to the END of the iteration:

    iteration t:  stage0(request t)        <- enqueue, submesh 0
                  stage1(h0 from t-1)      <- enqueue, submesh 1
                  ...                      <- one per stage, all enqueued before anything blocks
                  read back h0..h(PP-2)    <- ONE sync point, after all of them are in flight

In steady state one request retires per iteration, so throughput = W / iteration_time, and the
iteration is bounded by the slowest stage rather than by the sum. Independent requests are used rather
than chunks of one request: single-shot stages need no chunked KV bookkeeping, and for throughput the
two are equivalent (the pipeline is full either way).

Knobs: PP_WINDOW (tokens per request, default 5120), PP_ITERS (default 12; the first PP fill the
pipeline and are discarded).
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.mistral_small4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

GALAXY_CHIPS = 32
TOTAL_LAYERS = 36
W = int(os.environ.get("PP_WINDOW", 5120))
ITERS = int(os.environ.get("PP_ITERS", 12))
# "host": hand the activation over through the host (D2H + re-shard + H2D). Correct, but the composed
# activation is [1, 1, W, hidden] = 42 MB at W=5120, so three hops move ~250 MB per iteration and that
# dominates everything else. "none": skip the hand-off and only sync, isolating the pipeline's own
# ceiling -- what a real device-to-device transport would approach. The gap between the two IS the
# value of building that transport. This is a ceiling for the TRANSPORT only: the downstream ranks
# replay against a fixed input, which must be seeded with a real activation
# (_seed_downstream_inputs) or the MoE gate degenerates and the "ceiling" reads far too low.
HANDOFF = os.environ.get("PP_HANDOFF", "host")
# Stage geometry -> PP falls out of it, since the stages tile the galaxy. (8,1) is SP=8/TP=1: no TP
# collectives at all, four stages. (4,2) halves the sequence split to buy TP=2 and is a measured
# failure (worse than no PP) -- kept only so it cannot be "rediscovered". (8,2) holds SP=8 and halves
# the TP width rather than deleting it, giving PP=2: half the pipeline fill/drain, and unlike (8,1) it
# retains collective work, which is what scales with board power.
SHAPES = {"8x1": (8, 1), "4x2": (4, 2), "8x2": (8, 2)}


def _stage_plan(pp_shape):
    """(sp, tp, PP, layers_per_stage) for a stage shape, from how many stages tile the galaxy."""
    sp, tp = SHAPES[pp_shape]
    pp = GALAXY_CHIPS // (sp * tp)
    assert TOTAL_LAYERS % pp == 0, f"{TOTAL_LAYERS} layers do not divide into {pp} stages"
    return sp, tp, pp, TOTAL_LAYERS // pp


def _compose(sm, sp, tp):
    """Gather a stage output [1,1,W/sp,hidden/tp] back to the full [1,1,W,hidden]."""
    return ttnn.ConcatMesh2dToTensor(sm, dims=(2, 3), mesh_shape=(sp, tp))


def _shard(sm, sp, tp):
    """Re-shard a full [1,1,W,hidden] onto the next stage: seq on the SP axis, hidden on the TP axis."""
    return ttnn.ShardTensor2dMesh(sm, dims=(2, 3), mesh_shape=(sp, tp))


def _seed_downstream_inputs(ctrls, outs, inputs, subs, sp, tp, pp):
    """Put a REAL activation in every downstream rank's input, once, before timing.

    Under HANDOFF != "host" these inputs are never refreshed, so whatever they hold is what every
    iteration forwards. Zeros are the pathological choice: a zero hidden state gives every token
    identical gate logits, so top-k selects the SAME experts for all of them and the whole batch lands
    on one chip (with EP > 1 it must converge across the expert axis too, idling the other group).
    Measured cost of that artefact: ~1.1x at (8,1) and ~2.5x at (8,2) -- enough to invert a conclusion.

    Stage 0's own traced output is the seed: a genuine activation with realistic per-token variation,
    so routing spreads the way it does in production. It comes from 9 layers deep rather than each
    rank's true depth, so magnitudes are approximate; what the gate keys on is that tokens differ from
    one another, which this has and zeros do not.
    """
    ctrls[0].replay(blocking=False)
    ttnn.synchronize_device(subs[0])
    seed = ttnn.to_torch(outs[0], mesh_composer=_compose(subs[0], sp, tp)).to(torch.bfloat16)
    for r in range(1, pp):
        dev = ttnn.from_torch(
            seed,
            device=subs[r],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_shard(subs[r], sp, tp),
        )
        ttnn.copy(dev, inputs[r])
        ttnn.deallocate(dev)
        ttnn.synchronize_device(subs[r])


@pytest.mark.parametrize("pp_shape", list(SHAPES), ids=list(SHAPES))
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE
                ),
                "l1_small_size": 768,
                # One stage's segmented trace lives on each device; without a reserved region the
                # segments come out of general DRAM and trace_bytes() reads 0.
                "trace_region_size": 512 * 1024 * 1024,
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("variant", ["mistral_small4"], indirect=True, ids=["mistral4"])
@pytest.mark.timeout(0)
def test_mistral4_pp4_concurrent_throughput(
    variant, config_only, mesh_device, device_params, weight_cache_path, pp_shape
):
    if weight_cache_path is None:
        pytest.skip(f"pretrained TTNN cache unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    sp, tp, PP, LAYERS_PER_STAGE = _stage_plan(pp_shape)
    config = config_only
    config.max_seq_len = W
    assert (
        W // sp
    ) % 64 == 0, f"masked_bincount needs tokens/chip ({W}/{sp} = {W//sp}) to be a multiple of its 64-core grid"

    subs = mesh_device.create_submeshes(ttnn.MeshShape(sp, tp))
    assert len(subs) == PP, f"expected {PP} ({sp},{tp}) submeshes from {mesh_device.shape}, got {len(subs)}"
    logger.info(
        f"PP={PP} x ({sp},{tp}): carved {len(subs)} stage submeshes of {LAYERS_PER_STAGE} layers; window={W}, "
        f"tokens/chip={W//sp}, hidden/tp={config.hidden_size//tp}, iters={ITERS}"
    )
    cache_path = weight_cache_path / f"{sp}x{tp}"

    stages, kvs, ctrls = [], [], []
    try:
        for r, sm in enumerate(subs):
            stages.append(
                TtPrefillTransformer(
                    mesh_device=sm,
                    config=config,
                    model_cfg=MistralSmall4Config,
                    state_dict={},
                    num_layers=LAYERS_PER_STAGE,
                    seq_len=W,
                    dispatch_buffer_capacity_factor=8,
                    num_links=1,
                    topology=ttnn.Topology.Linear,
                    sp_axis=0,
                    tp_axis=1,
                    is_balanced=False,
                    gate_fallback_mode=GateComputeMode.GPT_DEVICE,
                    weight_cache_path=cache_path,
                    lm_head_is_column_parallel=True,
                    routing_use_l1_small_for_semaphores=True,
                    first_layer_idx=r * LAYERS_PER_STAGE,
                    is_first_rank=(r == 0),
                    # HEADLESS on purpose, every stage: the norm/LM-head/sample tail ends in a blocking
                    # host readback, which cannot live inside a trace. Throughput does not need the
                    # token, and a real PP prefill's last stage is headless anyway (the runtime has
                    # kv_only_last_layer for exactly this). So all four stages are blocks-only and
                    # capture identically.
                    is_last_rank=False,
                )
            )
            kvs.append(
                init_mla_kv_cache(
                    cache_format=MlaKvCacheFormat.BFP8_TILE,
                    hf_config=config,
                    mesh_device=sm,
                    seq_len=W,
                    mesh_shape=(sp, tp),
                    sp_axis=0,
                    num_kvpe_cache_layers=LAYERS_PER_STAGE,
                )
            )
        logger.info(f"built {PP} stages of {LAYERS_PER_STAGE} layers")

        # Persistent, captured-address inputs. Stage 0 takes SP-sharded token ids; stages 1..3 take a
        # hidden activation, refreshed in place between replays (a host->device write cannot be inside a
        # trace, so the copies happen between replays, never during capture).
        torch.manual_seed(0)
        tokens = torch.randint(0, config.vocab_size, (1, W), dtype=torch.int64)
        inputs = [
            ttnn.from_torch(
                tokens.reshape(sp, 1, W // sp),
                device=subs[0],
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(subs[0], dims=(0, None), mesh_shape=(sp, tp)),
            )
        ]
        hidden_host = torch.zeros(1, 1, W, config.hidden_size, dtype=torch.bfloat16)
        for r in range(1, PP):
            inputs.append(
                ttnn.from_torch(
                    hidden_host,
                    device=subs[r],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=_shard(subs[r], sp, tp),
                )
            )

        # Warm (compiles programs with the controller idle), then capture, per stage.
        outs = []
        for r, (st, sm) in enumerate(zip(stages, subs)):
            c = SubDeviceTraceController(sm)
            ctrls.append(c)
            st.set_trace_controller(c)
            st.forward(inputs[r], kvs[r], actual_isl=W)  # warmup
            ttnn.synchronize_device(sm)
            c.begin_capture()
            out = st.forward(inputs[r], kvs[r], actual_isl=W)
            c.end_capture()
            ttnn.synchronize_device(sm)
            outs.append(out)
            logger.info(
                f"  stage {r}: captured {c.num_segments} segments, {c.trace_bytes()/(1024*1024):.1f} MB, "
                f"out {list(out.shape)}"
            )

        if HANDOFF != "host":
            _seed_downstream_inputs(ctrls, outs, inputs, subs, sp, tp, PP)

        # --- pipelined replay: all stages in flight, then ONE sync ---
        times = []
        for t in range(ITERS):
            t0 = time.perf_counter()
            for c in ctrls:
                c.replay(blocking=False)  # non-blocking: keeps the 4 submeshes overlapped
            if HANDOFF == "host":
                handoffs = [
                    ttnn.to_torch(outs[r], mesh_composer=_compose(subs[r], sp, tp)).to(torch.bfloat16)
                    for r in range(PP - 1)
                ]  # sync point (D2H) -- 42 MB per hop at W=5120
                for r in range(1, PP):  # H2D for the next iteration's inputs
                    dev = ttnn.from_torch(
                        handoffs[r - 1],
                        device=subs[r],
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=_shard(subs[r], sp, tp),
                    )
                    ttnn.copy(dev, inputs[r])
                    ttnn.deallocate(dev)
            else:
                # No hand-off: just wait for all four stages. Each still does its full 9 layers of real
                # work on real weights; only the cross-stage data movement is omitted, so this is the
                # pipeline ceiling a device-to-device transport would chase.
                for sm in subs:
                    ttnn.synchronize_device(sm)
            dt = time.perf_counter() - t0
            times.append(dt)
            logger.info(f"  iter {t:>2}: {dt*1000:8.1f} ms" + ("   (filling)" if t < PP else ""))

        steady = times[PP:]
        best, med = min(steady), sorted(steady)[len(steady) // 2]
        logger.success(
            f"PP={PP} x ({sp},{tp}) TRACED handoff={HANDOFF}, window={W}: steady min {best*1000:.1f} ms, med {med*1000:.1f} ms "
            f"-> {W/best:,.0f} tok/s (min) / {W/med:,.0f} tok/s (med)"
        )
        print(
            f"PP{PP}_TRACED shape={pp_shape} handoff={HANDOFF} window={W} min_ms={best*1000:.1f} med_ms={med*1000:.1f} "
            f"tok_s_min={W/best:.0f} tok_s_med={W/med:.0f}"
        )
    finally:
        for c in ctrls:
            c.release()
        for st in stages:
            st.set_trace_controller(None)
            st.release_sub_device_managers()


# ======================================================================================================
# Long context: the same 4-submesh pipeline, but CHUNKED, so the KV accumulates and the numbers are
# directly comparable to the single-rank 102,400 / 261,120 measurements.
#
# The single-shot test above measures throughput with an EMPTY KV, so it models a chunk-0 rate. Here
# each stage keeps its own 9 layers' KV for the WHOLE context and chunk c flows through stages 0..3 over
# four iterations, so stage r's attention sees everything it wrote for chunks < c -- which is exactly
# chunked prefill, pipelined.
#
# One trace still serves every chunk: the per-chunk scalars (slot_id / actual_start / actual_end) cannot
# be host arguments on a captured program, so they live in 1-element uint32 DRAM tensors that the
# metadata ops read on-device, refreshed in place between replays.
# ======================================================================================================

CONTEXT = int(os.environ.get("PP_CONTEXT", 102400))


def _meta1(val, sm=None):
    """One 1-element uint32 metadata scalar; device-resident when `sm` is given, else host-side."""
    t = torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1)
    kw = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    if sm is not None:
        kw.update(device=sm, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(sm))
    else:
        kw.update(mesh_mapper=ttnn.ReplicateTensorToMesh(_MESH_FOR_HOST_META[0]))
    return ttnn.from_torch(t, **kw)


_MESH_FOR_HOST_META = [None]  # set once per test; host-side from_torch still wants a mapper


@pytest.mark.parametrize("pp_shape", ["8x1", "8x2"], ids=["8x1", "8x2"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE
                ),
                "l1_small_size": 768,
                "trace_region_size": 512 * 1024 * 1024,
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("variant", ["mistral_small4"], indirect=True, ids=["mistral4"])
@pytest.mark.timeout(0)
def test_mistral4_pp4_concurrent_longctx(variant, config_only, mesh_device, device_params, weight_cache_path, pp_shape):
    if weight_cache_path is None:
        pytest.skip(f"pretrained TTNN cache unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    sp, tp, PP, LAYERS_PER_STAGE = _stage_plan(pp_shape)
    CHUNK = W
    n_chunks = CONTEXT // CHUNK
    assert CONTEXT % CHUNK == 0, f"PP_CONTEXT {CONTEXT} must be a multiple of CHUNK {CHUNK}"
    assert (CHUNK // sp) % 64 == 0, f"masked_bincount needs chunk/sp ({CHUNK//sp}) to be a multiple of 64"

    config = config_only
    config.max_seq_len = CONTEXT
    subs = mesh_device.create_submeshes(ttnn.MeshShape(sp, tp))
    assert len(subs) == PP
    _MESH_FOR_HOST_META[0] = subs[0]
    cache_path = weight_cache_path / f"{sp}x{tp}"
    logger.info(
        f"PP={PP} x ({sp},{tp}) CHUNKED: {LAYERS_PER_STAGE} layers/stage, context={CONTEXT}, "
        f"chunk={CHUNK}, {n_chunks} chunks, "
        f"tokens/chip={CHUNK//sp}, handoff={HANDOFF}"
    )

    stages, kvs, ctrls, metas, outs = [], [], [], [], []
    try:
        for r, sm in enumerate(subs):
            stages.append(
                TtPrefillTransformer(
                    mesh_device=sm,
                    config=config,
                    model_cfg=MistralSmall4Config,
                    state_dict={},
                    num_layers=LAYERS_PER_STAGE,
                    seq_len=CHUNK,  # per-chunk size -> MoE/FFN dispatch buffers
                    max_seq_len=CONTEXT,  # KV ring buffer + RoPE tables span the whole context
                    dispatch_buffer_capacity_factor=8,
                    num_links=1,
                    topology=ttnn.Topology.Linear,
                    sp_axis=0,
                    tp_axis=1,
                    is_balanced=False,
                    gate_fallback_mode=GateComputeMode.GPT_DEVICE,
                    weight_cache_path=cache_path,
                    lm_head_is_column_parallel=True,
                    routing_use_l1_small_for_semaphores=True,
                    is_chunked=True,
                    slot_num=1,
                    first_layer_idx=r * LAYERS_PER_STAGE,
                    is_first_rank=(r == 0),
                    is_last_rank=False,  # headless: blocks only, so every stage is capturable
                )
            )
            kvs.append(
                init_mla_kv_cache(
                    cache_format=MlaKvCacheFormat.BFP8_TILE,
                    hf_config=config,
                    mesh_device=sm,
                    seq_len=CONTEXT,
                    mesh_shape=(sp, tp),
                    sp_axis=0,
                    num_kvpe_cache_layers=LAYERS_PER_STAGE,
                )
            )

        torch.manual_seed(0)
        tokens = torch.randint(0, config.vocab_size, (1, CHUNK), dtype=torch.int64)
        inputs, hidden_host = [], torch.zeros(1, 1, CHUNK, config.hidden_size, dtype=torch.bfloat16)
        for r, sm in enumerate(subs):
            if r == 0:
                inputs.append(
                    ttnn.from_torch(
                        tokens.reshape(sp, 1, CHUNK // sp),
                        device=sm,
                        dtype=ttnn.uint32,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=ttnn.ShardTensor2dMesh(sm, dims=(0, None), mesh_shape=(sp, tp)),
                    )
                )
            else:
                inputs.append(
                    ttnn.from_torch(
                        hidden_host,
                        device=sm,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=_shard(sm, sp, tp),
                    )
                )
            metas.append((_meta1(0, sm), _meta1(0, sm), _meta1(CHUNK, sm)))

        # host-side scalars for every chunk position, copied in between replays
        host_meta = [(_meta1(0), _meta1(c * CHUNK), _meta1((c + 1) * CHUNK)) for c in range(n_chunks)]

        for r, (st, sm) in enumerate(zip(stages, subs)):
            c = SubDeviceTraceController(sm)
            ctrls.append(c)
            st.set_trace_controller(c)

            def _fwd():
                return st.forward(
                    inputs[r],
                    kvs[r],
                    actual_isl=CHUNK,
                    actual_start=None,
                    actual_end=None,
                    cache_user_id=0,
                    metadata=metas[r],
                )

            _fwd()  # warm the metadata-variant programs
            ttnn.synchronize_device(sm)
            c.begin_capture()
            outs.append(_fwd())
            c.end_capture()
            ttnn.synchronize_device(sm)
            logger.info(f"  stage {r}: {c.num_segments} segments, {c.trace_bytes()/(1024*1024):.1f} MB")

        if HANDOFF != "host":
            _seed_downstream_inputs(ctrls, outs, inputs, subs, sp, tp, PP)

        # --- pipelined chunk loop: chunk (t-r) is in stage r at iteration t ---
        total, per_iter = 0.0, []
        for t in range(n_chunks + PP - 1):
            t0 = time.perf_counter()
            active = []
            for r in range(PP):
                c = t - r
                if 0 <= c < n_chunks:
                    for dst, src in zip(metas[r], host_meta[c]):
                        ttnn.copy_host_to_device_tensor(src, dst)
                    active.append(r)
            for r in active:
                ctrls[r].replay(blocking=False)
            if HANDOFF == "host":
                hs = [
                    ttnn.to_torch(outs[r], mesh_composer=_compose(subs[r], sp, tp)).to(torch.bfloat16)
                    for r in active
                    if r < PP - 1
                ]
                for i, r in enumerate([x for x in active if x < PP - 1]):
                    dev = ttnn.from_torch(
                        hs[i],
                        device=subs[r + 1],
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=_shard(subs[r + 1], sp, tp),
                    )
                    ttnn.copy(dev, inputs[r + 1])
                    ttnn.deallocate(dev)
            else:
                for r in active:
                    ttnn.synchronize_device(subs[r])
            dt = time.perf_counter() - t0
            total += dt
            per_iter.append(dt)
            if t < 3 or t % 10 == 0 or t >= n_chunks + PP - 3:
                logger.info(f"  iter {t:>2} (stages {active}): {dt*1000:8.1f} ms")

        steady = per_iter[PP - 1 : n_chunks]  # full-pipeline iterations only
        med = sorted(steady)[len(steady) // 2] if steady else float("nan")
        logger.success(
            f"PP={PP} x ({sp},{tp}) CHUNKED, context={CONTEXT}, {n_chunks} chunks, handoff={HANDOFF}: "
            f"total {total:.2f} s -> {CONTEXT/total:,.0f} tok/s; steady-state median {med*1000:.1f} ms/chunk "
            f"-> {CHUNK/med:,.0f} tok/s"
        )
        print(
            f"PP{PP}_LONGCTX shape={pp_shape} handoff={HANDOFF} context={CONTEXT} chunks={n_chunks} "
            f"total_s={total:.2f} tok_s_total={CONTEXT/total:.0f} med_ms={med*1000:.1f} "
            f"tok_s_steady={CHUNK/med:.0f}"
        )
    finally:
        for c in ctrls:
            c.release()
        for st in stages:
            st.set_trace_controller(None)
            st.release_sub_device_managers()
