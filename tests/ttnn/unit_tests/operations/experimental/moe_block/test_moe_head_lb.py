# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.moe_block.moe_head on a LoudBox (head_tail_lb_cases.py: SP 2 x TP 4, FABRIC_2D, seeded random
weights), the op on its own - accuracy, determinism and device time, each its own test.

Against torch on the same weights (the shared expert's on the bf8 grid, the router's bf16):
  * se: the shared expert's output (GLM: the owners' rows [S, H]; K3: the residual's layout [rows, H / TP]);
  * the routing rows: each token's top-K expert set and its gate weights (sigmoid scores + bias select, the scores
    normalized x route_scale) - almost every token's set (near ties may flip), the weights of the tokens whose set
    matches;
  * the planner's counts: each chip's experts' counts = the routing rows' picks (a histogram);
  * the dispatch buffer's own slots: GLM x itself (bit-exact), K3 the latent rows x @ latent_down^T.
Perf baselines: moe_head_lb_perf_baselines.json."""

import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from op_checks import (  # noqa: E402
    BLOCK_OPS,
    PERF_RUNS,
    DeterminismMarker,
    PerfBaselines,
    median_us,
    traced_op_device_times,
)
from head_tail_lb_cases import (  # noqa: E402
    CALLS,
    MODELS,
    PARAMS,
    SP,
    TP,
    Weights,
    _from_owner,
    _from_residual,
    _host_ints,
    _pcc,
    _q,
    _router_ref,
    _shared_ref,
    _skip_unless_lb,
    _to_residual,
    _x_host,
    mb,
)

BASELINES = PerfBaselines(HERE / "moe_head_lb_perf_baselines.json")
HEAD_OPS = {"moe_head": BLOCK_OPS["moe_head"]}


# ---------------------------------------------------------------------------------------------------- head
class Head:
    def __init__(self, mesh, case):
        self.mesh, self.case = mesh, case
        self.model = model = MODELS[case.model]
        self.dtypes = mb.MoEDtypes()
        cfg = mb.MoEBlockConfig(chunk_tokens=case.chunk, num_links=2, persist_l1_small=True)
        self.res = mb.MoEBlockResources(mesh, model.dims(), cfg, self.dtypes)
        self.A = self._accumulators()
        fit = mb.plan_moe_head(self.res, self.dtypes, self.A, None)
        assert fit.ok, fit.reason
        lay = fit.planned[0].layout
        logger.info(
            f"{case.id}: head plan GY {lay.engine_rows} MT {lay.row_tiles_per_core} NCH {lay.row_chunks}; A {self.A}"
        )
        self.w = Weights(model, case.seed)
        f32 = lambda t: None if t is None else ttnn.from_torch(t, dtype=ttnn.float32)  # noqa: E731
        w = self.w
        self.tw = mb.prepare_moe_head_weights(
            f32(w.gate),
            f32(w.up),
            f32(w.down),
            f32(w.latent_down),
            f32(w.router),
            f32(w.bias),
            resources=self.res,
            dtypes=self.dtypes,
            accumulators=self.A,
        )
        self.x_host = _x_host(case, model)
        self.x = _to_residual(mesh, self.x_host)

    def _accumulators(self):
        """The block planner's accumulator count (the head plans the combine's routing for it); a model whose experts
        a chip exceed the flat op's 64 on this mesh (K3 on 8 chips) takes the Galaxy's 11."""
        try:
            plan = mb.moe_block_plan(
                self.res,
                dtypes=self.dtypes,
                accumulators=None,
                max_tokens_per_expert=max(self.case.chunk, 256),
                layout=mb.FlatLayoutOptions(),
                combine=mb.CombineConfig(),
            )
            return plan.accumulators
        except Exception as e:  # noqa: BLE001
            logger.info(f"{self.case.id}: no block plan ({str(e).splitlines()[0][:160]}): accumulators 11")
            return 11

    def __call__(self):
        return mb.moe_head(self.x, self.tw, resources=self.res, dtypes=self.dtypes, accumulators=self.A)

    def close(self):
        self.res.release()


@pytest.mark.timeout(1800)
@PARAMS
def test_moe_head_accuracy(mesh_device, device_params, case):
    _skip_unless_lb(mesh_device)
    head = Head(mesh_device, case)
    model, chunk = head.model, case.chunk
    S, N, K = chunk // (SP * TP), SP * TP, model.K
    out = head()
    x = head.x_host.float()
    try:
        logger.info(f"MOE_HEAD accuracy {case.id}:")
        # the shared expert
        se_ref = _shared_ref(model, head.w, x)
        per_chip = ttnn.get_device_tensors(out.shared_expert_output)[0].shape
        se = (
            _from_owner(out.shared_expert_output, S)
            if per_chip[-1] == model.H
            else _from_residual(mesh_device, out.shared_expert_output, chunk)
        )
        _pcc("se", se_ref, se, 0.99)
        # the routing rows: a token's K ids, then its K gate weights as bf16 bits
        rows = torch.cat(
            [ttnn.to_torch(d).reshape(-1, 2 * K).to(torch.int64)[:S] for d in ttnn.get_device_tensors(out.routing_rows)]
        )
        ids = rows[:, :K] & 0xFFFF
        ws = ((rows[:, K:] & 0xFFFF) << 16).to(torch.int32).view(torch.float32)
        ids, order = torch.sort(ids, dim=-1)
        ws = torch.gather(ws, 1, order)
        ref_ids, ref_ws = _router_ref(model, head.w, x)
        same = (ids == ref_ids).all(-1)
        frac = float(same.float().mean())
        logger.info(f"  routing: {frac:.4f} of tokens pick the reference's set ({int((~same).sum())} differ)")
        assert frac >= 0.98, f"routing: only {frac:.4f} of tokens pick the reference's expert set"
        _pcc("gate weights (matching tokens)", ref_ws[same], ws[same], 0.999)
        # the planner's counts: chip i's experts' counts = the picks' histogram
        EPC = model.E // N
        hist = torch.bincount(ids.reshape(-1), minlength=model.E)
        counts = torch.cat(
            [
                ttnn.to_torch(d).reshape(-1)[i * EPC : (i + 1) * EPC].to(torch.int64)
                for i, d in enumerate(ttnn.get_device_tensors(out.counts))
            ]
        )
        assert torch.equal(counts, hist), "the planner's counts are not the routing rows' picks"
        assert int(counts.sum()) == chunk * K
        # the dispatch buffer's own slots
        xb = []
        for me, d in enumerate(ttnn.get_device_tensors(out.dispatch_buffer)):
            t = ttnn.to_torch(d)
            xb.append(t.reshape(-1, t.shape[-1])[me * S : (me + 1) * S])
        xb = torch.cat(xb)
        if model.latent:
            _pcc("xbuf (latent rows)", x @ _q(head.w.latent_down, head.dtypes.shared_weights).T, xb, 0.99)
        else:
            assert torch.equal(xb.to(torch.bfloat16), head.x_host), "xbuf: the own slots are not x"
            logger.info("  xbuf: own slots = x, bit-exact")
    finally:
        head.close()


@pytest.mark.timeout(1800)
@PARAMS
def test_moe_head_determinism(mesh_device, device_params, case):
    _skip_unless_lb(mesh_device)
    head = Head(mesh_device, case)
    det = DeterminismMarker(mesh_device)
    ints = ("routing_rows", "route_words", "counts", "regions", "token_index", "table", "plan")
    first = None
    try:
        for call in range(CALLS):
            out = head()
            # (xbuf: the head writes only its own slots - the rest stays as the first call left it: all comparable)
            xb = ttnn.to_layout(out.dispatch_buffer, ttnn.TILE_LAYOUT)
            det.add({"se": out.shared_expert_output, "xbuf": xb})
            got = {n: _host_ints(getattr(out, n)) for n in ints}
            if first is None:
                first = got
            else:
                for n in ints:
                    for chip, (a, b) in enumerate(zip(first[n], got[n])):
                        assert torch.equal(a, b), f"{case.id}: call {call}: {n} differs on chip {chip}"
        det.assert_identical(f"moe_head {case.id} (se, xbuf; + {', '.join(ints)} on the host)")
    finally:
        head.close()


@pytest.mark.timeout(1800)
@PARAMS
def test_moe_head_perf(mesh_device, device_params, case):
    _skip_unless_lb(mesh_device)
    head = Head(mesh_device, case)
    try:
        head()  # (warm: program build)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        head()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        t = traced_op_device_times(mesh_device, tid, PERF_RUNS, HEAD_OPS)["moe_head"]
        ttnn.release_trace(mesh_device, tid)
    finally:
        head.close()
    us = median_us(t)
    logger.info(f"PERF moe_head {case.id}: median {us:.1f} us over {len(t)} replays")
    BASELINES.check(f"moe_head {case.id}", us, case.margin)
