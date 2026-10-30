# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One Kimi-K3 MoE layer on one real input: TtMoe vs the MoE block, untraced and traced, on exactly the same x.

test_k3_layer_capture_ttmoe (K3_MOE_IMPL unset): the model's first K3_BENCH_LAYER + 1 layers prefilled chunk by chunk
over the 1M golden trace's tokens (TtMoe, real weights from the CI ttnn cache) up to chunk K3_BENCH_CHUNK; on that
chunk, layer K3_BENCH_LAYER's MoE input (ffn_norm's output) and TtMoe's output are saved to $K3_BENCH_DIR/x_L<l>_c<c>.pt,
the layer's decoder output scored against the trace; then the model's own TtMoe of that layer is timed on that x:
K3_BENCH_N eager forwards (synced each, the input copy off the clock) and K3_BENCH_REPLAYS traced replays (one at a
time, host wall, as test_kimi_moe_perf times them).

test_k3_layer_block (K3_MOE_IMPL=block): the MoE block on the saved x ($K3_BENCH_X) - K3_BLOCK_WEIGHTS=weka the layer's
real weights (K3_BLOCK_CHECK=1: PCC against the block's host model and the routing's skew), K3_BLOCK_WEIGHTS=ci on a
test_kimi_moe_perf MOE_X_DUMP (x [8, 640, H] and its gate) - its output against TtMoe's when the file has one, then the
same eager and traced timing.
"""

import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import SP_AXIS, TP_AXIS, _device_params

CHUNK = 5120
LAYER = int(os.environ.get("K3_BENCH_LAYER", "12"))
CHUNK_IDX = int(os.environ.get("K3_BENCH_CHUNK", "11"))
EAGER_N = int(os.environ.get("K3_BENCH_N", "20"))
REPLAYS = int(os.environ.get("K3_BENCH_REPLAYS", "20"))
TRACE_REGION = 64 * 1024 * 1024

PLACEMENTS = [
    pytest.param(
        (8, 4),
        {**_device_params(), "trace_region_size": TRACE_REGION},
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="8x4",
    )
]


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _composed(mesh_device, t):
    """[1, 1, S/SP, H/TP] (or [1, S/SP, H/TP]) over the mesh -> [S, H] in the mesh's row order."""
    shape = tuple(mesh_device.shape)
    dims = [0, 0]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    if len(t.shape) == 3:
        t = ttnn.unsqueeze(t, 0)
    y = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=shape))
    return y.reshape(-1, y.shape[-1])


def _summary(tag, kind, ms):
    rest = sorted(ms[1:]) or ms
    logger.info(
        f"K3_BENCH {tag} {kind}: first {ms[0]:.3f} ms, median {statistics.median(rest):.3f} ms, min {rest[0]:.3f} ms "
        f"over {len(rest)} after it"
    )


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
def test_k3_layer_capture_ttmoe(mesh_device, device_params):
    assert os.environ.get("K3_MOE_IMPL") != "block", "the capture runs the model's own TtMoe: K3_MOE_IMPL unset"
    from models.common.utility_functions import comp_pcc
    from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import load_attn_res_state_dict
    from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_1M, resolve_checkpoint, resolve_trace
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import _model_state_dict
    from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
    from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk
    from models.demos.deepseek_v3_d_p.tt.attn_res.weights import load_attn_res_weights
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import TtAttnResResidual
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import cache_root
    from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache
    from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

    checkpoint, trace = resolve_checkpoint(), resolve_trace(TRACE_1M)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_CKPT and the 1M golden trace")
    out_dir = Path(os.environ.get("K3_BENCH_DIR", "generated/k3_bench"))
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint)
    root = resolve_model_root(checkpoint)
    num_layers, total = LAYER + 1, CHUNK * (CHUNK_IDX + 1)
    config = kimi_k3_hf_config(max_seq=total)
    cache = cache_root(checkpoint, tuple(mesh_device.shape), TP_AXIS)

    attn_res = TtAttnRes(
        mesh_device,
        hidden_size=KimiK3Config.EMB_SIZE,
        eps=KimiK3Config.RMS_NORM_EPS,
        tp_axis=TP_AXIS,
        weights=load_attn_res_weights(
            mesh_device,
            load_attn_res_state_dict(checkpoint, num_layers, root),
            None,
            num_layers=num_layers,
            tensor_parallel_axis=TP_AXIS,
            prefix=root,
        ),
    )

    def residual_factory(hidden, block_residual=None):
        w = attn_res.weights
        return TtAttnResResidual(TtAttnResWalk(attn_res, hidden, list(w.pre), list(w.post), w.output, num_layers))

    t0 = time.time()
    model = TtKimiK3Transformer(
        mesh_device,
        config,
        KimiK3Config,
        _model_state_dict(checkpoint, num_layers, root, cache),
        num_layers=num_layers,
        seq_len=CHUNK,
        residual_factory=residual_factory,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=total,
        is_chunked=True,
        weight_cache_path=cache,
        tp_shard_kv=True,
    )
    logger.info(f"K3_BENCH model L{num_layers} built in {time.time() - t0:.0f} s")
    kvpe = None
    if model.schedule.num_mla_layers:
        kvpe = allocate_mla_kvpe_cache(
            mesh_device=mesh_device,
            hf_config=config,
            max_seq_len=total,
            mesh_shape=tuple(mesh_device.shape),
            sp_axis=SP_AXIS,
            num_layers=model.schedule.num_mla_layers,
            num_users=1,
            tp_axis=TP_AXIS,
        )

    blk = model.layers[LAYER]
    assert blk.is_moe and not blk.moe_block, f"layer {LAYER} is not a TtMoe layer"
    tt_moe = blk.ffn
    orig = blk._ffn_path
    cap = {}

    def ffn_tap(ffn_norm_out, **kw):
        if not cap.get("arm"):
            return orig(ffn_norm_out, **kw)
        cap["mc"] = ffn_norm_out.memory_config()
        cap["x_dev"] = ttnn.clone(ffn_norm_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cap["x"] = _composed(mesh_device, ffn_norm_out).to(torch.bfloat16)
        cap["kw"] = dict(actual_isl=kw["actual_isl"], padding_side=kw["padding_side"])
        cap["ctx"] = (kw["ctx"].actual_start, kw["ctx"].cache_user_id, kw["ctx"].metadata)
        out = orig(ffn_norm_out, **kw)
        cap["y"] = _composed(mesh_device, out).to(torch.bfloat16)
        return out

    blk._ffn_path = ffn_tap
    hidden = {}
    try:
        for chunk in range(CHUNK_IDX + 1):
            start = chunk * CHUNK
            if chunk == 0:
                model.reset_streams()
            last = chunk == CHUNK_IDX
            cap["arm"] = last
            tokens_tt = prepare_prefill_input_tensor(
                trace.token_ids(CHUNK, start)[0].tolist(),
                mesh_device,
                tuple(mesh_device.shape)[SP_AXIS],
                False,
                tuple(mesh_device.shape),
                SP_AXIS,
            )
            t0 = time.perf_counter()
            out = model.forward(
                tokens_tt,
                kvpe_cache=kvpe,
                actual_start=start,
                layer_tap=(lambda i, h: hidden.__setitem__(i, _composed(mesh_device, h))) if last else None,
            )
            if out is not None:
                ttnn.deallocate(out)
            ttnn.synchronize_device(mesh_device)
            logger.info(f"K3_BENCH chunk {chunk} (tokens {start}..{start + CHUNK}): {time.perf_counter() - t0:.2f} s")
        blk._ffn_path = orig

        start = CHUNK_IDX * CHUNK
        golden = trace.decoder_output(LAYER, start, start + CHUNK)
        _, msg = comp_pcc(golden, hidden[LAYER][:CHUNK], 0.98)
        logger.info(f"K3_BENCH layer {LAYER} chunk {CHUNK_IDX}: decoder output vs the 1M trace {msg}")
        logger.info(
            f"K3_BENCH captured x {tuple(cap['x'].shape)} (|max| {cap['x'].abs().max():.3g}, std "
            f"{cap['x'].float().std():.3g}), memory {cap['mc'].buffer_type}; forward kwargs {cap['kw']}, ctx "
            f"(actual_start, cache_user_id, metadata is None) {cap['ctx'][:2]}, {cap['ctx'][2] is None}"
        )
        path = out_dir / f"x_L{LAYER}_c{CHUNK_IDX}.pt"
        torch.save({"x": cap["x"], "y": cap["y"], "layer": LAYER, "chunk": CHUNK_IDX}, path)
        logger.info(f"K3_BENCH saved {path}")

        actual_start, cache_user_id, metadata = cap["ctx"]

        def forward(x):
            return tt_moe(
                ttnn.squeeze(x, dim=0),
                return_intermediates=False,
                actual_start=actual_start,
                metadata=metadata,
                cache_user_id=cache_user_id,
                **cap["kw"],
            )

        x_dev = cap["x_dev"]
        walls = []
        for i in range(EAGER_N):
            x = ttnn.clone(x_dev, memory_config=cap["mc"])
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter_ns()
            y, _ = forward(x)
            ttnn.synchronize_device(mesh_device)
            walls.append((time.perf_counter_ns() - t0) / 1e6)
            if i == 0:
                logger.info(
                    f"K3_BENCH ttmoe rerun vs the in-model call: PCC {_pcc(_composed(mesh_device, y), cap['y']):.5f}"
                )
            ttnn.deallocate(y)
        _summary(f"ttmoe L{LAYER} c{CHUNK_IDX}", "eager", walls)

        # traced: test_kimi_moe_perf's measure - a one-op trace refreshes x's buffer, the forward's segments replayed
        controller = SubDeviceTraceController(mesh_device)
        tt_moe.set_trace_controller(controller)
        copy_tid, capturing = None, False
        try:
            y, _ = forward(ttnn.clone(x_dev, memory_config=cap["mc"]))
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(y)
            copy_tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            x = ttnn.clone(x_dev, memory_config=cap["mc"])
            ttnn.end_trace_capture(mesh_device, copy_tid, cq_id=0)
            controller.begin_capture()
            capturing = True
            y, _ = forward(x)
            controller.end_capture()
            capturing = False
            replays = []
            for _ in range(REPLAYS):
                ttnn.execute_trace(mesh_device, copy_tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                t0 = time.perf_counter_ns()
                controller.replay()
                replays.append((time.perf_counter_ns() - t0) / 1e6)
            logger.info(
                f"K3_BENCH ttmoe traced vs the in-model call: PCC {_pcc(_composed(mesh_device, y), cap['y']):.5f}"
            )
            _summary(f"ttmoe L{LAYER} c{CHUNK_IDX}", "traced", replays)
        finally:
            if capturing:
                controller.end_capture()
            controller.release()
            if copy_tid is not None:
                ttnn.release_trace(mesh_device, copy_tid)
            tt_moe.set_trace_controller(None)
            tt_moe.release_sub_device_manager()
    finally:
        blk._ffn_path = orig
        if model.kda_states is not None:
            model.kda_states.deallocate()


def _fp32_reference(m, w, flat, x, y_block, y_ttmoe, path, tag):
    """The layer in fp32 on the host with its own fp32 routing (the block's model: score, top-K over score + bias, the
    picks' scores normalized x route_scale; the experts from TtMoe's bfp4 cache as the block holds them) -> both device
    outputs scored against it; the reference saved."""
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.moe_block_ffn import host_moe

    t0 = time.time()
    rt = m.router
    assert (
        w.router["weight"].shape[0] == m.num_experts
    ), f"router rows {w.router['weight'].shape[0]} != E (a folded router)"
    xf = x.float()
    lg = xf @ w.router["weight"].float().T
    sc = torch.sqrt(torch.nn.functional.softplus(lg)) if rt.score_func == "sqrtsoftplus" else torch.sigmoid(lg)
    _, ids = torch.topk(sc + w.router["bias"].float(), m.top_k, dim=-1)
    wts = sc.gather(1, ids)
    wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * rt.route_scale
    y_ref = host_moe(xf, w, m, ids, wts, flat=flat)
    torch.save({"y": y_ref.to(torch.bfloat16), "ids": ids.to(torch.int16), "wts": wts}, path)
    msg = f"K3_BENCH {tag}: fp32 reference ({rt.score_func}, host routing, {time.time() - t0:.0f} s): block PCC {_pcc(y_block, y_ref):.5f}"
    if y_ttmoe is not None:
        msg += f", TtMoe PCC {_pcc(y_ttmoe, y_ref):.5f}"
    nrm = y_ref.float().norm()
    msg += f"; rel err block {(y_block.float() - y_ref).norm() / nrm:.4f}"
    if y_ttmoe is not None:
        msg += f", TtMoe {(y_ttmoe.float() - y_ref).norm() / nrm:.4f}"
    logger.info(msg + f"; saved {path}")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
def test_k3_layer_block(mesh_device, device_params):
    assert os.environ.get("K3_MOE_IMPL") == "block", "K3_MOE_IMPL=block (the fabric payload the block's combine needs)"
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.moe_block_ffn import TtK3MoEBlockFfn

    d = torch.load(os.environ["K3_BENCH_X"])
    H = KimiK3Config.EMB_SIZE
    x = d["x"].reshape(-1, H).to(torch.bfloat16)
    T = x.shape[0]
    assert T == CHUNK, f"x has {T} rows, the bench runs one {CHUNK}-token chunk"
    layer_idx = int(d.get("layer", 1))
    kind = os.environ.get("K3_BLOCK_WEIGHTS", "random")
    tag = f"block L{layer_idx} c{d.get('chunk', '-')} {kind}"
    shape = tuple(mesh_device.shape)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=(2, 3))
    host_x = ttnn.from_torch(x.view(1, 1, T, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
    x_dev = ttnn.to_device(host_x, mesh_device)

    t0 = time.time()
    ffn = TtK3MoEBlockFfn(mesh_device, KimiK3Config, layer_idx)
    ref = os.environ.get("K3_BENCH_REF") == "1" and kind == "weka"
    if ref:  # (held here: the adapter drops its host experts once checked)
        w_host, flat = ffn._weights(), ffn.flat
        assert flat is not None, "K3_BENCH_REF=1 needs the host experts: K3_BLOCK_CHECK=1"
    y = ffn(x_dev)  # (K3_BLOCK_CHECK=1: this call checked against the host model; the routing's skew logged)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"K3_BENCH {tag}: built and first call in {time.time() - t0:.0f} s")
    y_block = _composed(mesh_device, y).to(torch.bfloat16)
    if d.get("y") is not None:
        logger.info(f"K3_BENCH {tag}: output vs TtMoe's on the same x: PCC {_pcc(y_block, d['y']):.5f}")
    out_dir = Path(os.environ.get("K3_BENCH_DIR", "generated/k3_bench"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"L{layer_idx}_c{d.get('chunk', '-')}_{kind}"
    torch.save(
        {"y": y_block, "layer": layer_idx, "chunk": d.get("chunk"), "kind": kind}, out_dir / f"y_block_{stem}.pt"
    )
    logger.info(f"K3_BENCH {tag}: block output saved to {out_dir / f'y_block_{stem}.pt'}")
    ttnn.deallocate(y)
    if ref:
        _fp32_reference(ffn.model, w_host, flat, x, y_block, d.get("y"), out_dir / f"y_ref_{stem}.pt", tag)
        del w_host, flat

    walls = []
    for _ in range(EAGER_N):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter_ns()
        y = ffn(x_dev)
        ttnn.synchronize_device(mesh_device)
        walls.append((time.perf_counter_ns() - t0) / 1e6)
        ttnn.deallocate(y)
    _summary(tag, "eager", walls)

    block, layer, _ = ffn._block(T)
    x_in = block.input_buffer()
    ttnn.copy_host_to_device_tensor(host_x, x_in)
    run = (lambda: block(x_in)) if layer is None else (lambda: block(x_in, layer=layer))
    y = run()
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(y)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    y = run()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    replays = []
    for _ in range(REPLAYS):
        ttnn.copy_host_to_device_tensor(host_x, x_in)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter_ns()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        replays.append((time.perf_counter_ns() - t0) / 1e6)
    if d.get("y") is not None:
        logger.info(f"K3_BENCH {tag} traced: output vs TtMoe's: PCC {_pcc(_composed(mesh_device, y), d['y']):.5f}")
    _summary(tag, "traced", replays)
    ttnn.release_trace(mesh_device, tid)
    ttnn.deallocate(y)


def _glm_params():
    from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

    from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
    from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params

    return torus_xy_device_params(
        fabric_payload_size=fabric_payload(GLM53Config.EMB_SIZE), l1_small_size=0, trace_region_size=TRACE_REGION
    )


ML_MODEL = os.environ.get("MOE_ML_MODEL", "k3")
ML_PLACEMENTS = [
    pytest.param(
        (8, 4),
        _glm_params() if ML_MODEL == "glm" else {**_device_params(), "trace_region_size": TRACE_REGION},
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="8x4",
    )
]


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device, device_params", ML_PLACEMENTS, indirect=True)
def test_block_layers_traced(mesh_device, device_params):
    """Several MoE layers on one block, one after another (each its own MoELayer: the per-layer weight switch), eager then
    captured in ONE trace and replayed - every layer's traced output against its eager one. MOE_ML_MODEL=k3 (K3_MOE_IMPL
    =block, K3_BLOCK_WEIGHTS real_gate / weka) or glm (GLM-5.3, weka: $GLM53_HF_MODEL + its TtMoe cache)."""
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.moe_block_ffn import TtK3MoEBlockFfn

    layers = [int(v) for v in os.environ.get("MOE_ML_LAYERS", "3,4,5,6" if ML_MODEL == "glm" else "1,2,4,5").split(",")]
    if ML_MODEL == "glm":
        from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config as cfg

        cache = Path(os.environ["TT_GLM53_PREFILL_TTNN_CACHE"]) / "glm_5_3_bh_32dev"
        hits = sorted(cache.rglob(f"layer_{layers[0]}.routed_expert.local_0_gate_dtype_*.tensorbin"))
        assert hits, f"no GLM routed cache under {cache}"
        kw = dict(cache_path=hits[0].parent)
    else:
        assert os.environ.get("K3_MOE_IMPL") == "block"
        cfg, kw = KimiK3Config, {}
    H = cfg.EMB_SIZE
    shape = tuple(mesh_device.shape)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=(2, 3))
    x = (torch.randn(CHUNK, H, generator=torch.Generator().manual_seed(11)) * 0.5).to(torch.bfloat16)
    x_dev = ttnn.from_torch(
        x.view(1, 1, CHUNK, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper, device=mesh_device
    )
    ffns = [TtK3MoEBlockFfn(mesh_device, cfg, L, **kw) for L in layers]

    def seq():
        return [f(x_dev) for f in ffns]

    eager = []
    for _ in range(2):
        ys = seq()
        ttnn.synchronize_device(mesh_device)
        eager = [_composed(mesh_device, y) for y in ys]
        for y in ys:
            ttnn.deallocate(y)
    logger.info(
        f"K3_ML eager: {len(layers)} layers {layers} ok, |y| max "
        + " ".join(f"{float(e.abs().max()):.3g}" for e in eager)
    )
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    ys = seq()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for it in range(3):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        got = [_composed(mesh_device, y) for y in ys]
        pccs = [_pcc(g, e) for g, e in zip(got, eager)]
        logger.info(f"K3_ML traced replay {it}: per-layer PCC vs eager " + " ".join(f"{p:.5f}" for p in pccs))
        assert all(p > 0.999 for p in pccs), f"traced layers differ from eager: {pccs}"
    ttnn.release_trace(mesh_device, tid)
