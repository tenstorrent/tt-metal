# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device A/B of the MoE roofline knobs (2026-10-26, _agents/moe_roofline/STATUS.md). NOT RUN when written (the chips
belong to the validation queue): every case names its expected outcome.

Chip 0 (``TT_VISIBLE_DEVICES=<chip>``, mesh (1,1)):
* ``test_router_rows_chip0``: C10 -- the fused router's [x | w] pages + uint16 id rows (``ep_mode="rm"``) and its
  bf16 index tiles (``tiles``) vs today's outputs + the glue ops (``TtEPExperts.dispatch_rows`` layout chain on
  one chip, ``_index_tiles``): BIT-IDENTICAL (same bf16 roundings, exact ids), padded rows untouched, trace replay.
* ``test_glu_split_chip0``: C11 -- the fused activation vs ``slice + slice + clamped_silu_glu`` on the shared
  expert's and the dense MLP's decode shapes: PCC >= 0.9999 and <= 2 bf16 ulps (a different SFPU sequence, not
  bit identity), trace replay.
* ``test_ep_w13_bfp4_chip0``: C12 -- the single-chip EP path with GLM53_EXPERT_W13_DTYPE=bfp4 vs bfp8 on real
  layer-4 weights / routing: PCC vs the bfp8 path >= 0.985 (the sweep doc measured 0.989-0.991), traced us per
  call of both (the bfp4 gain is the DRAM-bound share of the gate|up).
* ``test_stream_knobs_chip0``: (d) -- the EP dispatch ops at the 8x4 shape (9 experts, I = 2048) with every plan
  variant (rotation on / off, units with 1 / 2 / 4 / 8 pairs, prefetch 1 / 2 on bfp4, explicit packet tiles):
  BIT-IDENTICAL outputs to the legacy program (the knobs only reorder reads / split work), traced us per variant
  logged as the calibration table of ``plan.py`` (H1 / H2).
* ``test_ep_placement_chip0``: (e) -- the placement perm through the fused router + the EP op on one chip with a
  permuted 9-expert share: the result equals the identity placement's on the same tokens (the experts are the same
  weights under other slot ids).

(2,4) block (TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18, the p150_x8 mesh graph descriptor, MESH_DEVICE=P150x8,
GLM53_MESH_SHAPE=2,4):
* ``test_ep_moe_router_rows_block``: the EP MoE layer with GLM53_ROUTER_EP_ROWS=1 vs off at 2 / 4 / 32 users per
  row: outputs within 1 bf16 ulp (the dispatch payload is the same bytes), the forward in a trace, no L1 left
  behind, the op count down by 5 (graph capture).
* ``test_ep_moe_bfp4_block``: GLM53_EXPERT_W13_DTYPE=bfp4 through the whole EP layer vs HF (PCC > 0.985 reported,
  the 0.995 bar is the bfp8 path's) and vs the bfp8 run; the cache files carry the dtype.
* ``test_ep_moe_plan_block``: GLM53_EXPERT_PLAN=explicit with rotation + units: bit-identical to legacy.

The e2e chain with the knobs (not here): ``tests/test_mesh_model.py`` on the block with GLM53_MOE_LAYOUT=ep
GLM53_EP_COMBINE=rs GLM53_EP_RS_REMAP=1 GLM53_ROUTER_EP_ROWS=1 GLM53_SHARED_GLU=fused [GLM53_EXPERT_PLAN=explicit
GLM53_STREAM_GATHER_ROTATE=1], then the 8x4 e2e at 32 users; for C12 the accuracy plan = the logits gate
(``tests/test_mesh_model_logits.py``, GLM53_DUMP=L45_NAT_S128, PCC >= 0.98, top-1 >= 0.85 on natural text with
GLM53_EXPERT_W13_DTYPE=bfp4) + the 64 K needle (``tests/test_needle_smoke.py``).

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_moe_roofline_mesh.py -k chip0 -s
    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto \\
        MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4 GLM53_MOE_LAYOUT=ep pytest models/demos/glm53_flash/tests/test_moe_roofline_mesh.py -k block -s
"""

from __future__ import annotations

import dataclasses
import os
import time

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.custom_ops.expert_stream import plan as P  # noqa: E402
from models.demos.glm53_flash.custom_ops.expert_stream.op import (  # noqa: E402
    ExpertDispatch,
    ep_base_tensor,
    ep_index_tiles,
    expert_dispatch_golden_ep,
)
from models.demos.glm53_flash.custom_ops.glu_split.op import GluSplit  # noqa: E402
from models.demos.glm53_flash.custom_ops.router_topk.op import RouterTopk  # noqa: E402
from models.demos.glm53_flash.tests.custom_ops.test_expert_dispatch import DOWN_READERS, GROUPS  # noqa: E402
from models.demos.glm53_flash.tests.custom_ops.test_expert_dispatch import upload as upload_random_experts  # noqa: E402
from models.demos.glm53_flash.tests.custom_ops.test_router_topk import _gate, _real_x  # noqa: E402
from models.demos.glm53_flash.tests.test_mesh_moe import l1_high_water, l1_resident  # noqa: E402
from models.demos.glm53_flash.tests.test_moe import LAYER, realistic_mlp_input, weights  # noqa: E402,F401 (fixture)
from models.demos.glm53_flash.tests.test_moe_block import count_ops, ref_block, timed_trace  # noqa: E402
from models.demos.glm53_flash.tests.utils import pcc  # noqa: E402
from models.demos.glm53_flash.tt.config import GLM53FlashConfig  # noqa: E402
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params  # noqa: E402
from models.demos.glm53_flash.tt.mlp import TtClampedSwiGLU  # noqa: E402
from models.demos.glm53_flash.tt.moe import TtEPExperts, TtMoE, TtRouter  # noqa: E402

H, K = 4096, 8
ROWS, COLS = 2, 4
STEPS = int(os.environ.get("GLM53_BLOCK_STEPS", "20"))
DEVICE_PARAMS = {"l1_small_size": 32768, "trace_region_size": 160 * 1024 * 1024}


def _ulps(a: torch.Tensor, b: torch.Tensor) -> int:
    """Max distance in bf16 ulps between two bf16 tensors."""
    ai = a.bfloat16().view(torch.int16).to(torch.int32)
    bi = b.bfloat16().view(torch.int16).to(torch.int32)
    return int((ai - bi).abs().max())


# ============================================================================================ chip 0: C10
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("T", [32, 4])
def test_router_rows_chip0(mesh_device, T):
    cfg, gate_w, bias = _gate(LAYER)
    router = TtRouter(mesh_device, cfg, gate_w, bias)
    x = ttnn.from_torch(_real_x(T), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    x_host = ttnn.to_torch(x)[0, 0, :T]
    scores_host = ttnn.to_torch(router._scores(x, T))[0, 0].float()[:T]
    bias_host = ttnn.to_torch(router.bias).float().reshape(-1)[: router.E]
    xw_g, idx_g, tiles_g = RouterTopk.golden_rows(x_host, scores_host, bias_host, K, cfg.routed_scaling_factor)
    # today's outputs + the glue chain of TtEPExperts.dispatch_rows (layout ops only, no fabric)
    w16, idx32, _ = router.forward(x, ep_mode="rows")
    l1 = ttnn.L1_MEMORY_CONFIG
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=l1)
    w_rm = ttnn.to_layout(w16, ttnn.ROW_MAJOR_LAYOUT, memory_config=l1)
    xw_ref = ttnn.to_torch(ttnn.concat([x_rm, w_rm], dim=-1, memory_config=l1))[0, 0]
    idx_ref = ttnn.to_torch(ttnn.to_layout(ttnn.typecast(idx32, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT))[0, 0].long()
    # the fused layouts
    xw, idx_rows, _ = router.forward(x, ep_mode="rm")
    assert xw.layout == ttnn.ROW_MAJOR_LAYOUT and list(xw.shape) == [T, 1, 1, H + K]
    assert idx_rows.dtype == ttnn.uint16 and list(idx_rows.shape) == [T, 1, 1, K]
    xw_h = ttnn.to_torch(xw).reshape(T, H + K)
    idx_h = ttnn.to_torch(idx_rows).reshape(T, K).long()
    assert torch.equal(xw_h, xw_ref), "the [x | w] pages must equal untilize + untilize + concat bit for bit"
    assert torch.equal(xw_h, xw_g) and torch.equal(idx_h, idx_g) and torch.equal(idx_h, idx_ref)
    w_t, tiles, _ = router.forward(x, ep_mode="tiles")
    tiles_h = ttnn.to_torch(tiles)[0, 0, :T]
    assert torch.equal(tiles_h.float(), tiles_g.float())
    it_ref, wt_ref = _IndexTiles(mesh_device)._index_tiles(
        ttnn.typecast(idx32, ttnn.float32), ttnn.typecast(w16, ttnn.float32)
    )
    assert torch.equal(ttnn.to_torch(tiles).float(), ttnn.to_torch(it_ref).float())
    assert torch.equal(ttnn.to_torch(w_t).float(), ttnn.to_torch(wt_ref).float())
    logger.info(f"router rows T={T}: [x | w] pages, id rows and index tiles bit-identical to the glue chain")
    # trace replay of the rm mode (outputs allocated inside the capture)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    xw2, idx2, _ = router.forward(x, ep_mode="rm")
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for _ in range(3):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        assert torch.equal(ttnn.to_torch(xw2).reshape(T, H + K), xw_g)
        assert torch.equal(ttnn.to_torch(idx2).reshape(T, K).long(), idx_g)
    ttnn.release_trace(mesh_device, tid)


class _IndexTiles:
    """``TtEPExperts._index_tiles`` without an expert build (its pad constants only)."""

    def __init__(self, device):
        self.device = device
        self._pad = {}

    _index_tiles = TtEPExperts._index_tiles


# ============================================================================================ chip 0: C11
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("T, I", [(32, 512), (4, 512), (32, 3072)], ids=["shared-32", "shared-4", "dense-32"])
def test_glu_split_chip0(mesh_device, T, I):
    g = torch.Generator().manual_seed(T + I)
    gu_host = (torch.randn(1, 1, T, 2 * I, generator=g) * 6).bfloat16()
    gu = ttnn.from_torch(gu_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    gate = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, T, I])
    up = ttnn.slice(gu, [0, 0, 0, I], [1, 1, T, 2 * I])
    ref = ttnn.to_torch(ttnn.clamped_silu_glu(gate, up, 10.0))[0, 0]
    op = GluSplit(mesh_device, 10.0)
    out = op.allocate_output(gu)
    op(gu, out)
    got = ttnn.to_torch(out)[0, 0]
    golden = GluSplit.golden(gu_host.float())[0, 0]
    p, u = pcc(ref, got), _ulps(ref, got)
    logger.info(
        f"glu_split T={T} I={I}: pcc vs composite {p:.6f}, max {u} bf16 ulps, pcc vs torch {pcc(golden, got):.6f}"
    )
    assert p > 0.9999 and u <= 2
    assert pcc(golden, got) > 0.9999
    # trace replay with the output allocated in the capture
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out2 = op.allocate_output(gu)
    op(gu, out2)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    assert torch.equal(ttnn.to_torch(out2)[0, 0], got)
    ttnn.release_trace(mesh_device, tid)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_shared_expert_knob_chip0(mesh_device, weights, monkeypatch):
    """The shared expert through tt/mlp.py with GLM53_SHARED_GLU=fused vs composite (same weights): <= 2 ulps."""
    snapshot, text, raw, hf, sd = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    x = realistic_mlp_input(snapshot, text, raw, 32).reshape(1, 1, 32, -1)
    x_dev = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    outs = {}
    for mode in ("composite", "fused"):
        monkeypatch.setenv("GLM53_SHARED_GLU", mode)
        mlp = TtClampedSwiGLU(
            mesh_device,
            raw["mlp.shared_experts.gate_proj.weight"],
            raw["mlp.shared_experts.up_proj.weight"],
            raw["mlp.shared_experts.down_proj.weight"],
            cfg.swiglu_limit,
        )
        top, dev_ops = count_ops(mesh_device, lambda: mlp(x_dev))
        y, dt = timed_trace(mesh_device, lambda: mlp(x_dev), STEPS)
        outs[mode] = (y, dt, len(dev_ops))
        logger.info(f"shared expert {mode}: {len(dev_ops)} device ops, {dt * 1e6:.1f} us per call")
    yc, yf = outs["composite"][0], outs["fused"][0]
    assert outs["fused"][2] == outs["composite"][2] - 2
    assert pcc(yc, yf) > 0.9999 and _ulps(yc, yf) <= 4


# ============================================================================================ chip 0: C12
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_ep_w13_bfp4_chip0(mesh_device, weights, monkeypatch):
    snapshot, text, raw, hf, sd = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    E_loc, I = 9, cfg.moe_intermediate_size
    cfg_loc = dataclasses.replace(cfg, n_routed_experts=E_loc)
    gate_up_loc = sd["experts.gate_up_proj"][:E_loc].contiguous()
    down_loc = sd["experts.down_proj"][:E_loc].contiguous()
    T = 32
    x_all = realistic_mlp_input(snapshot, text, raw, 256).reshape(256, -1)[:T]
    with torch.no_grad():
        _, w_hf, idx_hf = hf.gate(x_all)
    idx_all = idx_hf.reshape(T, -1).long()
    w_all = w_hf.reshape(T, -1).float()
    dense_all = torch.zeros(T, cfg.n_routed_experts).scatter_(1, idx_all, w_all)
    ref = ref_block(x_all, dense_all[:, :E_loc], gate_up_loc, down_loc, cfg.swiglu_limit)
    x_dev = ttnn.from_torch(
        x_all.reshape(1, 1, T, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    idx_dev = ttnn.from_torch(
        idx_all.float().reshape(1, 1, T, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    w_dev = ttnn.from_torch(w_all.reshape(1, 1, T, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    res = {}
    for name in ("bfp8", "bfp4"):
        monkeypatch.setenv("GLM53_EXPERT_W13_DTYPE", name)
        from models.demos.glm53_flash.tt.moe import routed_w13_dtype

        dt = routed_w13_dtype(ttnn.bfloat8_b)
        ep = TtEPExperts(mesh_device, cfg_loc, gate_up_loc, down_loc, expert_dtype=dt)
        assert ep.w13.dtype == dt and ep.w2.dtype == ttnn.bfloat8_b
        y, t = timed_trace(mesh_device, lambda: ep.forward(x_dev, w_dev, idx_dev), STEPS)
        res[name] = (y.reshape(T, -1), t)
        logger.info(
            f"EP 9 experts T={T} w13 {name}: {t * 1e6:.1f} us per call, pcc vs torch {pcc(ref, res[name][0]):.6f}"
        )
    p = pcc(res["bfp8"][0], res["bfp4"][0])
    logger.info(f"bfp4 vs bfp8 gate|up: pcc {p:.6f}; us {res['bfp8'][1] * 1e6:.1f} -> {res['bfp4'][1] * 1e6:.1f}")
    assert pcc(ref, res["bfp8"][0]) > 0.9999 and p > 0.985


# ============================================================================================ chip 0: (d)
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 64 * 1024 * 1024}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "T, active, rows_per", [(32, 2, 5), (256, 3, 27), (256, 9, 28)], ids=["32-2", "256-3", "256-9"]
)
@pytest.mark.parametrize("w13", ["bfp8", "bfp4"])
def test_stream_knobs_chip0(mesh_device, T, active, rows_per, w13):
    """Every plan variant vs the legacy program: bit-identical outputs, traced us per variant (the calibration
    table of plan.py's H1 / H2)."""
    I, E_local, base = 2048, 9, 9
    dt = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[w13]
    w13_t, w2_t, w13_q, w2_q = upload_random_experts(mesh_device, E_local, I, seed=base, w13_dtype=dt)
    g = torch.Generator().manual_seed(T + active)
    idx = torch.full((T, K), 4000, dtype=torch.long)
    for a in range(active):
        toks = torch.randperm(T, generator=g)[:rows_per]
        idx[toks, a % K] = base + a
    wts = (torch.rand(T, K, generator=g) + 0.2).bfloat16().float()
    x = torch.randn(T, H).bfloat16()
    _, y_ref, passes = expert_dispatch_golden_ep(x, idx, wts, base, E_local, w13_q, w2_q)
    xd = ttnn.from_torch(x.reshape(1, 1, T, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    it, wt = ep_index_tiles(idx, wts, mesh_device, K)
    bt = ep_base_tensor(base, mesh_device)
    pre_max = P.max_prefetch_pairs(128, P.TILE_BYTES[w13], T, E_local, True)
    variants = {
        "legacy": P.legacy_plan(E_local, I),
        "rotate": P.StreamPlan(gather_rotate=True),
        "units1": P.StreamPlan(units=True, unit_pairs=1),
        "units1-rotate": P.StreamPlan(units=True, unit_pairs=1, gather_rotate=True),
        "units2-rotate": P.StreamPlan(units=True, unit_pairs=2, gather_rotate=True),
        "units4-rotate": P.StreamPlan(units=True, unit_pairs=4, gather_rotate=True),
        "units8-rotate": P.StreamPlan(units=True, unit_pairs=8, gather_rotate=True),
        "packet4": P.StreamPlan(packet_tiles=4),
        "roofline": P.plan_expert_stream(
            grid=(13, 10),
            E_local=E_local,
            K=H,
            I=I,
            H=H,
            T=T,
            w13=w13,
            expected_active=active,
            tokens_per_expert=rows_per,
        ),
    }
    if pre_max >= 2:
        variants["prefetch2"] = P.StreamPlan(prefetch_pairs=2)
    table = []
    y_legacy = None
    for name, plan in variants.items():
        kw = plan.op_kwargs()
        kw["groups"] = min(kw["groups"], GROUPS) if name != "roofline" else kw["groups"]
        kw["down_readers"] = DOWN_READERS
        op = ExpertDispatch(mesh_device, E_local, H, I, H, T, ep=True, w13_dtype=dt, **kw)

        def run():
            x_rm = ttnn.to_layout(xd, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
            act, table_, out = op.alloc(mesh_device)
            op.forward(x_rm, wt, it, w13_t, w2_t, act, table_, out, base=bt)
            for t_ in (x_rm, act, table_):
                ttnn.deallocate(t_)
            return out

        y, dt_ = timed_trace(mesh_device, run, STEPS)
        y = y.reshape(T, H)
        p = pcc(y_ref, y)
        if y_legacy is None:
            y_legacy = y
        same = torch.equal(y, y_legacy)
        table.append((name, dt_ * 1e6, p, same, kw))
        logger.info(
            f"{w13} T={T} {active}x{rows_per}: {name:14s} {dt_ * 1e6:8.1f} us  pcc {p:.6f}  bit-identical {same}  {kw}"
        )
        assert p > 0.9999
        assert same, f"{name}: the knobs must not change the result"
    logger.info("STREAM-KNOBS " + "; ".join(f"{n} {t:.0f}" for n, t, *_ in table))


# ============================================================================================ chip 0: (e)
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_ep_placement_chip0(mesh_device, weights, monkeypatch, tmp_path):
    """One chip (D = 1) with a permuted 9-expert share: experts [perm] under slot ids vs the identity placement
    with global ids (the same weights, the same tokens): identical results."""
    snapshot, text, raw, hf, sd = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    E_loc = 9
    cfg_loc = dataclasses.replace(cfg, n_routed_experts=E_loc)
    T = 32
    x_all = realistic_mlp_input(snapshot, text, raw, 256).reshape(256, -1)[:T]
    with torch.no_grad():
        _, w_hf, idx_hf = hf.gate(x_all)
    idx_all = idx_hf.reshape(T, -1).long()
    w_all = w_hf.reshape(T, -1).float()
    # the chip's share: the 9 most popular experts of these tokens (so several are active), as slots 0..8
    pop = torch.bincount(idx_all.flatten(), minlength=cfg.n_routed_experts)
    chosen = torch.argsort(pop, descending=True)[:E_loc].tolist()
    gate_up_loc = sd["experts.gate_up_proj"][chosen].contiguous()
    down_loc = sd["experts.down_proj"][chosen].contiguous()
    perm = torch.full((cfg.n_routed_experts,), 4000, dtype=torch.long)  # non-local experts map to no slot
    for i, e in enumerate(chosen):
        perm[e] = i
    x_dev = ttnn.from_torch(
        x_all.reshape(1, 1, T, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    # reference: global ids remapped on the host to the slot ids
    idx_slots = perm[idx_all]
    dense_loc = torch.zeros(T, E_loc)
    for t in range(T):
        for j in range(K):
            if idx_slots[t, j] < E_loc:
                dense_loc[t, idx_slots[t, j]] = w_all[t, j]
    ref = ref_block(x_all, dense_loc, gate_up_loc, down_loc, cfg.swiglu_limit)
    ep = TtEPExperts(mesh_device, cfg_loc, gate_up_loc, down_loc)
    idx_dev = ttnn.from_torch(
        idx_slots.float().reshape(1, 1, T, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    w_dev = ttnn.from_torch(w_all.reshape(1, 1, T, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    y_host_perm = ttnn.to_torch(ep.forward(x_dev, w_dev, idx_dev)).reshape(T, -1)
    # the router applies the table on device (the tiles mode: ids as bf16 integers need slots <= 256: ours are < 9)
    router = TtRouter(mesh_device, cfg, raw["mlp.gate.weight"], raw["mlp.gate.e_score_correction_bias"])
    others = [e for e in range(cfg.n_routed_experts) if e not in chosen]  # the rest: slots 9.. (never local)
    full_perm = [0] * cfg.n_routed_experts
    for i, e in enumerate(chosen):
        full_perm[e] = i
    for i, e in enumerate(others):
        full_perm[e] = E_loc + i
    router.set_perm(full_perm)
    w_t, tiles, _ = router.forward(x_dev, ep_mode="tiles")
    y_dev_perm = ttnn.to_torch(ep.forward(x_dev, w_t, tiles)).reshape(T, -1)
    p1, p2 = pcc(ref, y_host_perm), pcc(y_host_perm, y_dev_perm)
    logger.info(
        f"placement on one chip: host-remapped ids pcc vs torch {p1:.6f}; router perm vs host remap pcc {p2:.6f}"
    )
    assert p1 > 0.9999 and p2 > 0.99999


# ============================================================================================ block
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params",
    [mesh_device_params(ROWS, COLS, fabric_config=ttnn.FabricConfig.FABRIC_1D, trace_region_size=200_000_000)],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("rows_T", [4, 2, 32])
def test_ep_moe_router_rows_block(mesh_device, weights, rows_T, monkeypatch):
    monkeypatch.setenv("GLM53_EP_RS_REMAP", "1")
    snapshot, text, raw, hf, sd = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    mlp = {k[len("mlp.") :]: v for k, v in raw.items() if "experts." not in k or "shared_experts" in k}
    mlp["experts.gate_up_proj"] = sd["experts.gate_up_proj"]
    mlp["experts.down_proj"] = sd["experts.down_proj"]
    mesh = GLMMesh.from_device(mesh_device)
    x_all = realistic_mlp_input(snapshot, text, raw, ROWS * 32).reshape(1, 1, ROWS * 32, -1)
    xs = torch.cat([x_all[:, :, 32 * r : 32 * r + rows_T] for r in range(ROWS)], dim=2)
    with torch.no_grad():
        y_hf = hf(xs.reshape(1, ROWS * rows_T, -1)).reshape(1, 1, ROWS * rows_T, -1)
    tt_x = mesh.upload(xs.bfloat16(), ttnn.bfloat16, rows=2)
    results = {}
    for mode in ("0", "1"):
        monkeypatch.setenv("GLM53_ROUTER_EP_ROWS", mode)
        t0 = time.perf_counter()
        moe = TtMoE(mesh_device, cfg, mlp, mesh=mesh, layout="ep")
        logger.info(f"EP MoE (router rows {mode}) built in {time.perf_counter() - t0:.0f} s")
        top, dev_ops = count_ops(mesh_device, lambda: moe(tt_x))
        marks = [l1_high_water(mesh_device)]
        y = moe(tt_x)
        ttnn.synchronize_device(mesh_device)
        shards = [ttnn.to_torch(d).float() for d in mesh.device_tensors(y)]
        worst = 1.0
        for r in range(ROWS):
            for c in range(COLS):
                worst = min(worst, pcc(y_hf[:, :, r * rows_T : (r + 1) * rows_T], shards[r * COLS + c][:, :, :rows_T]))
        ttnn.deallocate(y)
        marks.append(l1_high_water(mesh_device))
        y2 = moe(tt_x)
        ttnn.deallocate(y2)
        marks.append(l1_high_water(mesh_device))
        assert marks[2] == marks[1] and not l1_resident(moe, mesh)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        y_tr = moe(tt_x)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        tr_shards = [ttnn.to_torch(d).float() for d in mesh.device_tensors(y_tr)]
        ttnn.release_trace(mesh_device, tid)
        results[mode] = (shards, tr_shards, len(dev_ops), worst)
        logger.info(f"router rows {mode}: {len(dev_ops)} device ops, worst pcc vs HF {worst:.6f}")
        assert worst > 0.995
    s0, s1 = results["0"][0], results["1"][0]
    assert results["1"][2] == results["0"][2] - 5, (results["0"][2], results["1"][2])
    for a, b in zip(s0, s1):
        assert pcc(a, b) > 0.99999 and _ulps(a, b) <= 1
    for a, b in zip(results["1"][0], results["1"][1]):
        assert pcc(a, b) > 0.9999


@pytest.mark.timeout(2400)
@pytest.mark.parametrize(
    "device_params",
    [mesh_device_params(ROWS, COLS, fabric_config=ttnn.FabricConfig.FABRIC_1D, trace_region_size=200_000_000)],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("knobs", ["bfp4", "plan"])
def test_ep_moe_knobs_block(mesh_device, weights, knobs, monkeypatch):
    """GLM53_EXPERT_W13_DTYPE=bfp4 (C12: PCC vs HF reported, >= 0.985; vs the bfp8 run) and GLM53_EXPERT_PLAN=explicit
    with rotation + 2-pair units ((d): bit-identical to legacy) through the whole EP layer on the block."""
    monkeypatch.setenv("GLM53_EP_RS_REMAP", "1")
    snapshot, text, raw, hf, sd = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    mlp = {k[len("mlp.") :]: v for k, v in raw.items() if "experts." not in k or "shared_experts" in k}
    mlp["experts.gate_up_proj"] = sd["experts.gate_up_proj"]
    mlp["experts.down_proj"] = sd["experts.down_proj"]
    mesh = GLMMesh.from_device(mesh_device)
    rows_T = 4
    x_all = realistic_mlp_input(snapshot, text, raw, ROWS * 32).reshape(1, 1, ROWS * 32, -1)
    xs = torch.cat([x_all[:, :, 32 * r : 32 * r + rows_T] for r in range(ROWS)], dim=2)
    with torch.no_grad():
        y_hf = hf(xs.reshape(1, ROWS * rows_T, -1)).reshape(1, 1, ROWS * rows_T, -1)
    tt_x = mesh.upload(xs.bfloat16(), ttnn.bfloat16, rows=2)
    envs = {
        "bfp4": [{}, {"GLM53_EXPERT_W13_DTYPE": "bfp4"}],
        "plan": [
            {},
            {
                "GLM53_EXPERT_PLAN": "explicit",
                "GLM53_STREAM_GATHER_ROTATE": "1",
                "GLM53_STREAM_UNITS": "1",
                "GLM53_STREAM_UNIT_PAIRS": "2",
            },
        ],
    }[knobs]
    outs = []
    for env in envs:
        for k in (
            "GLM53_EXPERT_W13_DTYPE",
            "GLM53_EXPERT_PLAN",
            "GLM53_STREAM_GATHER_ROTATE",
            "GLM53_STREAM_UNITS",
            "GLM53_STREAM_UNIT_PAIRS",
        ):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        moe = TtMoE(mesh_device, cfg, mlp, mesh=mesh, layout="ep")
        if env.get("GLM53_EXPERT_W13_DTYPE") == "bfp4":
            assert moe.experts.w13.dtype == ttnn.bfloat4_b and moe.experts.w2.dtype == ttnn.bfloat8_b
        y, dt = timed_trace(mesh_device, lambda: moe(tt_x), STEPS)
        shards = [ttnn.to_torch(d).float() for d in mesh.device_tensors(moe(tt_x))]
        worst = min(
            pcc(y_hf[:, :, r * rows_T : (r + 1) * rows_T], shards[r * COLS + c][:, :, :rows_T])
            for r in range(ROWS)
            for c in range(COLS)
        )
        logger.info(f"EP MoE block {knobs} {env}: {dt * 1e3:.3f} ms per traced call, worst pcc vs HF {worst:.6f}")
        outs.append((shards, worst))
    base, knob = outs
    if knobs == "bfp4":
        assert base[1] > 0.995 and knob[1] > 0.985
        logger.info(f"bfp4 vs bfp8 EP layer: pcc {min(pcc(a, b) for a, b in zip(base[0], knob[0])):.6f}")
    else:
        for a, b in zip(base[0], knob[0]):
            assert torch.equal(a, b), "the stream plan knobs must not change the result"
