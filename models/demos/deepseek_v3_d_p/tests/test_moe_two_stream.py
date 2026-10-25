# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Two-chunk pipeline (E3), MoE side: per-stream MoE resources.

The pipeline runs the SAME layer objects for two chunks (streams) whose ops interleave. TtMoe.forward(stream=s)
selects stream s's TT_CCL (tt_ccl.ccl_stream): its CCL semaphore pools, persistent CCL buffers (gate partials AG),
shared-expert RS intermediate + keepalive, the HRE/combine overlap semaphores + forwarding buffer, and the gate's
device padding config. This test builds the real GLM-5.3 L10 model from the weight cache and, for every MoE layer,
captures ONE trace that alternates the two streams back to back:

    for L in moe_layers:  out[A][L] = moe_L(x_A, stream=0) ; out[B][L] = moe_L(x_B, stream=1)

a second trace with the same calls all on stream 0 (today's single-stream path), and a third issuing each stream from
its own thread with no explicit stream (the E3 runner's style: the stream is thread-local). Every trace is replayed and
every (layer, stream) output must be bitwise identical to the single-stream trace and to the eager single-stream
forward (PCC is logged too). Inputs are two different chunks of the golden trace's hidden states (RMS-normalised),
each with its own chunk metadata tensors, as the pipeline will have.

Env: MOE2S_LAYERS (model depth, default 10), MOE2S_MAX_MOE (cap on MoE layers, default all), MOE2S_REPLAYS (2).
Run with MLA_SEQ_SHARDED_Q=1 (best-v6 config).
"""

import copy
import hashlib
import os
import threading
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests._reuse import acquire_transformer, finish_transformer
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.test_prefill_transformer_chunked import (
    CHUNK,
    GLM_KV_CACHE_FORMAT,
    GLM_L1_SMALL_SIZE,
    GLM_TRACE_REGION_SIZE,
    SEQ_CACHE,
    _load_layer_rows,
    _resolve_trace_dir,
    _sparse_cache_format,
)
from models.demos.deepseek_v3_d_p.tt.mla.indexer import resolve_has_indexer
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt import tt_ccl as tt_ccl_module
from models.demos.deepseek_v3_d_p.tt.tt_ccl import all_tt_ccls, get_tt_ccl, per_axis_topology, prepare_ccl_streams
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

NUM_STREAMS = 2


def _host_inputs(variant, emb_dim, first_moe_layer):
    """Two chunks (streams) of MoE-block inputs [CHUNK, emb]: the golden trace's hidden state entering the first
    MoE layer (decoder_output of the layer before), RMS-normalised, chunk s = tokens [s*CHUNK, (s+1)*CHUNK); plus
    the matching residuals. Falls back to seeded noise when the trace is absent."""
    xs, residuals = [], []
    try:
        trace_dir = _resolve_trace_dir(variant)
    except FileNotFoundError:
        trace_dir = None
    for s in range(NUM_STREAMS):
        h = None
        if trace_dir is not None and trace_dir.exists():
            layer = first_moe_layer - 1
            h = _load_layer_rows(
                trace_dir,
                variant.prefill_trace_layout,
                "hidden_states",
                layer,
                f"decoder_output_layer_{layer}",
                s * CHUNK,
                (s + 1) * CHUNK,
            ).float()
        if h is None or h.shape[0] != CHUNK:
            gen = torch.Generator().manual_seed(1234 + s)
            h = torch.randn(CHUNK, emb_dim, generator=gen)
        x = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        xs.append(x)
        residuals.append(h)
    return xs, residuals


def _time_trace(mesh_device, fn, n_inner=8, n_replay=10):
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(n_inner):
        ttnn.deallocate(fn())
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n_replay):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ts.append((time.perf_counter() - t0) / (n_replay * n_inner) * 1e6)
    ttnn.release_trace(mesh_device, tid)
    return sorted(ts)[1]


def _pcr_probe(mesh_device, moe, forward):
    """MOE2S_PCR_PROBE=1: capture the post_combine_reduce inputs of one real MoE forward (real routing, the overlapped
    HRE+combine output) and time PCR alone, with the model's dispatch table and with an all-local one, and count the
    active (token, slot) pairs the skip leaves. Explains the model's ~245 us vs ~92 us for a 2-of-8-active R0."""
    stash = {}
    orig = moe.reduce_module.forward

    def spy(combine_output, weights=None, indices=None, expert_dispatch_table=None):
        if not stash:
            stash.update(
                combine=ttnn.clone(combine_output),
                weights=ttnn.clone(weights),
                indices=ttnn.clone(indices),
                table=expert_dispatch_table,
            )
        return orig(combine_output, weights=weights, indices=indices, expert_dispatch_table=expert_dispatch_table)

    moe.reduce_module.forward = spy
    try:
        ttnn.deallocate(forward())
        ttnn.synchronize_device(mesh_device)
    finally:
        moe.reduce_module.forward = orig
    c, w, i, tbl = stash["combine"], stash["weights"], stash["indices"], stash["table"]
    logger.info(
        f"[pcr-probe] combine {tuple(c.shape)} {c.layout} {c.dtype} {c.memory_config()}; weights {tuple(w.shape)} "
        f"{w.layout} {w.dtype}; indices {tuple(i.shape)} {i.layout} {i.dtype} {i.memory_config()}; table "
        f"{tuple(tbl.shape)} {tbl.layout} {tbl.dtype}"
    )
    dev_i = ttnn.get_device_tensors(i)
    dev_t = ttnn.get_device_tensors(tbl)
    for d in (0, 1, 2, 3, 13, 31):
        idx = ttnn.to_torch(dev_i[d]).long().reshape(-1, i.shape[-1])
        table = ttnn.to_torch(dev_t[d]).long().flatten()
        ok = idx < table.numel()
        active = torch.zeros_like(idx, dtype=torch.bool)
        active[ok] = table[idx[ok]] != -1
        logger.info(
            f"[pcr-probe] dev {d}: table entries {table.numel()} local {(table != -1).sum().item()}; "
            f"active slots/token {active.float().sum(-1).mean().item():.2f} (tokens {idx.shape[0]}); "
            f"max idx {idx.max().item()}"
        )

    def pcr(table):
        return ttnn.experimental.deepseek_prefill.post_combine_reduce(
            c, w, i, table, expert_dim=3, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    logger.info(f"[pcr-probe] PCR model table: {_time_trace(mesh_device, lambda: pcr(tbl)):.1f} us")
    all_local = ttnn.from_torch(
        torch.zeros(tuple(tbl.shape), dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    logger.info(f"[pcr-probe] PCR all-local table: {_time_trace(mesh_device, lambda: pcr(all_local)):.1f} us")
    none_local = ttnn.from_torch(
        torch.full(tuple(tbl.shape), -1, dtype=torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    logger.info(f"[pcr-probe] PCR none-local table: {_time_trace(mesh_device, lambda: pcr(none_local)):.1f} us")


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha1(t.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:16]


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=moe_fabric_payload(GLM53Config),
                l1_small_size=GLM_L1_SMALL_SIZE,
                trace_region_size=GLM_TRACE_REGION_SIZE,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_moe_two_stream(
    variant, config_only, mesh_device, device_params, weight_cache_path, num_links, monkeypatch
):
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    num_layers = int(os.environ.get("MOE2S_LAYERS", "10"))
    max_moe = int(os.environ.get("MOE2S_MAX_MOE", "0")) or None
    replays = max(1, int(os.environ.get("MOE2S_REPLAYS", "2")))
    topology = per_axis_topology(device_params["fabric_config"])
    sp_axis, tp_axis = 0, 1
    sp, tp = tuple(mesh_device.shape)
    assert (sp, tp) == (8, 4), f"this test targets mesh-8x4, got {tuple(mesh_device.shape)}"

    config = copy.deepcopy(config_only)
    config.max_seq_len = SEQ_CACHE
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and rope_scaling.get("factor", 1.0) == 1.0:
        rope_scaling["original_max_position_embeddings"] = SEQ_CACHE
    has_indexer = resolve_has_indexer(config)
    cache_format = _sparse_cache_format(has_indexer, GLM_KV_CACHE_FORMAT)
    effective_cache_path = weight_cache_path / f"{sp}x{tp}"
    experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp * tp)
    assert TtPrefillTransformer.check_cache_complete(
        effective_cache_path,
        num_layers,
        experts_per_chip=experts_per_chip,
        first_k_dense=variant.model_config.NUM_DENSE_LAYERS,
    ), f"TTNN cache incomplete for {num_layers} layers at {effective_cache_path}"

    transformer = acquire_transformer(
        TtPrefillTransformer,
        mesh_device,
        variant,
        config,
        model_cfg=variant.model_config,
        state_dict={},
        num_layers=num_layers,
        seq_len=CHUNK,
        max_seq_len=SEQ_CACHE,
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        weight_cache_path=effective_cache_path,
        is_chunked=True,
        slot_num=1,
        kv_only_last_layer=False,
        routing_use_l1_small_for_semaphores=True,
        tp_shard_kv=has_indexer,
        **({"sparse_kv_cache_format": cache_format} if has_indexer else {}),
    )
    moes = [(i, layer.ffn) for i, layer in enumerate(transformer.layers) if isinstance(layer.ffn, TtMoe)]
    assert moes, f"no MoE layer in the first {num_layers} layers"
    if max_moe is not None:
        moes = moes[:max_moe]
    layer_ids = [i for i, _ in moes]
    logger.info(f"[moe2s] MoE layers {layer_ids}, overlap={moes[0][1].overlap_routed_expert_with_combine}")

    emb_dim = config.hidden_size
    host_x, host_res = _host_inputs(variant, emb_dim, layer_ids[0])
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, -1))
    compose = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(sp, tp), dims=(0, -1))
    rep = ttnn.ReplicateTensorToMesh(mesh_device)

    def upload(t):
        return ttnn.from_torch(
            t.reshape(sp, CHUNK // sp, emb_dim).to(torch.bfloat16),
            mesh_mapper=shard,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def meta1(val):
        return ttnn.from_torch(
            torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=rep,
        )

    # Persistent per-stream inputs (outside every trace; each forward frees its input, so it gets a clone).
    xs = [upload(t) for t in host_x]
    residuals = [upload(t) for t in host_res]
    metas = [(meta1(0), meta1(s * CHUNK), meta1((s + 1) * CHUNK)) for s in range(NUM_STREAMS)]

    def moe_call(moe, s_data, stream):
        out, _ = moe(
            ttnn.clone(xs[s_data]),
            return_intermediates=False,
            actual_isl=CHUNK,
            padding_side="right",
            metadata=metas[s_data],
            residual=residuals[s_data],
            stream=stream,
        )
        return out

    # "runner": the E3 two-stream runner's style -- each stream issues from its own thread, no explicit stream
    # argument; the stream comes from a thread-local (utils.pipeline.current_stream on the E3 branch, patched here).
    runner_tls = threading.local()
    monkeypatch.setattr(tt_ccl_module, "_pipeline_current_stream", lambda: getattr(runner_tls, "stream", 0))

    def in_stream_thread(stream, fn):
        out = {}

        def body():
            runner_tls.stream = stream
            try:
                out["value"] = fn()
            except BaseException as e:  # re-raised in the caller
                out["error"] = e

        t = threading.Thread(target=body)
        t.start()
        t.join()
        if "error" in out:
            raise out["error"]
        return out["value"]

    def run_all(stream_of):
        """One forward per (layer, data stream) in pipeline order: layer-major, A then B. stream_of(s) = "thread"
        issues chunk s from a thread whose runner stream is s, with stream=None."""
        outs = {}
        for i, moe in moes:
            for s in range(NUM_STREAMS):
                if stream_of(s) == "thread":
                    outs[(i, s)] = in_stream_thread(s, lambda: moe_call(moe, s, None))
                else:
                    outs[(i, s)] = moe_call(moe, s, stream_of(s))
        return outs

    def to_host(outs):
        host = {k: ttnn.to_torch(v, mesh_composer=compose) for k, v in outs.items()}
        for v in outs.values():
            ttnn.deallocate(v)
        return host

    failures = []

    def check(tag, ref, got):
        """Bitwise per (layer, chunk); PCC and digests logged either way."""
        for k in ref:
            same = torch.equal(ref[k], got[k])
            pcc = _pcc(ref[k], got[k])
            logger.info(
                f"[moe2s] {tag} layer {k[0]} chunk {'AB'[k[1]]}: {'bitwise' if same else 'DIFFERS'} pcc={pcc:.7f} "
                f"ref={_digest(ref[k])} got={_digest(got[k])}"
            )
            if not same:
                failures.append(f"{tag} layer {k[0]} chunk {'AB'[k[1]]} pcc={pcc:.7f}")

    controller = SubDeviceTraceController(mesh_device)
    transformer.set_trace_controller(controller)
    try:
        # Stream 1's TT_CCL and MoE resources before any warm-up/capture (their creation synchronizes the mesh).
        prepare_ccl_streams(mesh_device, NUM_STREAMS)
        for _, moe in moes:
            for s in range(NUM_STREAMS):
                moe.prepare_stream(s)
        assert get_tt_ccl(mesh_device, 1) is not get_tt_ccl(mesh_device, 0)
        if moes[0][1].overlap_routed_expert_with_combine:
            addr = [
                {
                    ttnn.get_global_semaphore_address(sem)
                    for sem in get_tt_ccl(mesh_device, s).get_combine_overlap_semaphores()
                }
                for s in range(NUM_STREAMS)
            ]
            assert not (addr[0] & addr[1]), "stream 1 shares combine semaphores with stream 0"

        # Eager: the single-stream reference (stream 0 for both chunks), then stream 1 for both (warms stream 1's
        # programs and lazily created per-stream buffers, e.g. the gate's device padding config).
        if os.environ.get("MOE2S_PCR_PROBE", "0") == "1":
            _pcr_probe(mesh_device, moes[0][1], lambda: moe_call(moes[0][1], 0, 0))
            return
        eager = to_host(run_all(lambda s: 0))
        ttnn.synchronize_device(mesh_device)
        # The two chunks must really differ, or a stream mix-up would be invisible.
        for i in layer_ids:
            assert not torch.equal(eager[(i, 0)], eager[(i, 1)]), f"layer {i}: chunks A and B give the same output"
        check("eager stream1 vs eager stream0", eager, to_host(run_all(lambda s: 1)))
        ttnn.synchronize_device(mesh_device)

        single_r0 = None
        for name, stream_of in (("single", lambda s: 0), ("alternating", lambda s: s), ("runner", lambda s: "thread")):
            controller.release()
            controller = SubDeviceTraceController(mesh_device)
            transformer.set_trace_controller(controller)
            controller.begin_capture()
            outs = run_all(stream_of)
            controller.end_capture()
            ttnn.synchronize_device(mesh_device)
            for ccl in all_tt_ccls(mesh_device):  # allocated inside the capture: the trace owns that address
                ccl.set_shared_rs_input_keepalive(None)
            logger.info(
                f"[moe2s] {name}: captured {len(outs)} MoE forwards = {controller.num_segments} segments, "
                f"{controller.trace_bytes() / 2**20:.1f} MB"
            )
            for r in range(replays):
                controller.replay()
                ttnn.synchronize_device(mesh_device)
                got = {k: ttnn.to_torch(v, mesh_composer=compose) for k, v in outs.items()}
                check(f"trace {name} r{r} vs eager", eager, got)
                if name == "single" and r == 0:
                    single_r0 = got
                else:
                    check(f"trace {name} r{r} vs trace single r0", single_r0, got)
                del got
            for v in outs.values():
                ttnn.deallocate(v)
    finally:
        if controller._capturing:  # close the open segment so release() runs and the real error surfaces
            controller.end_capture()
        controller.release()
        transformer.set_trace_controller(None)

    finish_transformer(transformer)
    assert not failures, f"{len(failures)} mismatches: {failures[:8]}"
