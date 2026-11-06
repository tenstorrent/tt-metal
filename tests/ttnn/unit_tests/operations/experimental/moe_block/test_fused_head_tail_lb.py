# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.moe_block.fused_head and fused_tail on a LoudBox: SP 2 x TP 4 (FABRIC_2D), seeded random weights,
each op on its own (no dispatch, no experts, no combine) - accuracy, determinism and device time, each its own test.

A case is a model config (MODELS: GLM-5.3, Kimi-K3 - the block's dims, the host reference's activation) at a test
config (the tokens a mesh row, 640 or 384: chunk = SP x that - a multiple of 32 x TP, the head's rows rule).

fused_head, against torch on the same weights (the shared expert's on the bf8 grid, the router's bf16):
  * se: the shared expert's output (GLM: the owners' rows [S, H]; K3: the residual's layout [rows, H / TP]);
  * the routing rows: each token's top-K expert set and its gate weights (sigmoid scores + bias select, the scores
    normalized x route_scale) - almost every token's set (near ties may flip), the weights of the tokens whose set
    matches;
  * the planner's counts: each chip's experts' counts = the routing rows' picks (a histogram);
  * the dispatch buffer's own slots: GLM x itself (bit-exact), K3 the latent rows x @ latent_down^T.
fused_tail, from random owner rows and shared-expert output:
  * GLM (all-to-all): out = y_own + se in the residual's layout [rows, H / TP];
  * K3 (engine): out = rmsnorm(y_own) * norm @ latent_up^T + se.
Determinism: CALLS calls on the same inputs, float outputs folded on the device (op_checks.DeterminismMarker), the
small integer outputs (routing rows, route words, plan tensors) compared on the host.
Perf: op_checks.PERF_RUNS (5) traced replays on the real-time program profiler, the op's slowest chip, median, against
its baseline in fused_head_tail_perf_baselines.json (per machine: the LoudBox's, or a Galaxy's 2x4 tray;
MOE_PERF_RECORD=1 records it)."""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
sys.path.insert(0, str(HERE))
from op_checks import (  # noqa: E402
    BLOCK_OPS,
    PERF_RUNS,
    DeterminismMarker,
    PerfBaselines,
    median_us,
    traced_op_device_times,
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


weights_py = _load(
    "flat_routed_expert_weights",
    REPO / "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/flat_routed_expert_weights.py",
)
mb = ttnn.experimental.moe_block

SP, TP = 2, 4
CALLS = 10  # (determinism calls; perf: PERF_RUNS)
BASELINES = PerfBaselines(HERE / "fused_head_tail_perf_baselines.json")
HEAD_OPS = {"fused_head": BLOCK_OPS["fused_head"]}
TAIL_OPS = {"fused_tail": BLOCK_OPS["fused_tail"]}


# ---------------------------------------------------------------------------------------------------- model configs
@dataclass(frozen=True)
class Model:
    E: int
    K: int
    H: int
    I: int  # routed expert hidden (the head plans the combine for it)
    Is: int  # shared expert hidden
    act: str  # weights_py.act_ref name (the shared expert's, here also the routed experts')
    score: str  # "Sigmoid" | "SqrtSoftplus"
    route_scale: float
    latent: int | None = None

    def dims(self):
        if self.act == "situ":
            act = mb.ExpertActivation(kind=mb.ExpertActivationKind.Situ, beta=4.0, linear_beta=25.0)
        elif self.act == "clamped_silu":
            act = mb.ExpertActivation(kind=mb.ExpertActivationKind.ClampedSilu, limit=10.0)
        else:
            act = mb.ExpertActivation(kind=mb.ExpertActivationKind.Silu)
        kw = dict(latent_dim=self.latent, latent_norm=True) if self.latent else {}
        return mb.MoEModelDims(
            num_experts=self.E,
            top_k=self.K,
            emb_dim=self.H,
            routed_expert_hidden_dim=self.I,
            routed_expert_activation=act,
            router=mb.RouterSpec(score_func=getattr(mb.RouterScoreFunc, self.score), route_scale=self.route_scale),
            shared_expert_hidden_dim=self.Is,
            shared_expert_activation=act,
            **kw,
        )

    @property
    def routed_width(self):
        return self.latent or self.H


MODELS = {
    "glm53": Model(E=256, K=8, H=6144, I=2048, Is=2048, act="silu", score="Sigmoid", route_scale=2.5),
    "k3": Model(E=896, K=16, H=7168, I=3072, Is=6144, act="situ", score="Sigmoid", route_scale=1.0, latent=3584),
}


# ---------------------------------------------------------------------------------------------------- test configs
@dataclass(frozen=True)
class Case:
    model: str
    row_tokens: int  # tokens a mesh row (chunk = SP x row_tokens)
    margin: float = 0.05  # (the baselines: fused_head_tail_perf_baselines.json)
    seed: int = 3

    @property
    def chunk(self):
        return SP * self.row_tokens

    @property
    def id(self):
        return f"{self.model}-row{self.row_tokens}"


# (Kimi-K3 is not a case: a latent model needs a torus axis (MoEBlockResources fails fast - on the plain 2D fabric its
# first call hangs), and the LoudBox's 2x4 has none)
CASES = [Case("glm53", 640), Case("glm53", 384)]


def _payload(model):
    return (2 * model.routed_width + 32 + 63) // 64 * 64  # (moe_impl.moe_block_fabric_payload)


def _device_params(model):
    rc = ttnn._ttnn.fabric.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = _payload(model)
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        "fabric_router_config": rc,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        "l1_small_size": 16384,
        "trace_region_size": 64 << 20,
    }


def _params():
    return [pytest.param((SP, TP), _device_params(MODELS[c.model]), c, id=c.id) for c in CASES]


PARAMS = pytest.mark.parametrize("mesh_device, device_params, case", _params(), indirect=["mesh_device", "device_params"])


def _skip_unless_lb(mesh_device):
    if ttnn.get_arch_name() != "blackhole":
        pytest.skip("Blackhole only")
    if tuple(mesh_device.shape) != (SP, TP):
        pytest.skip(f"needs a {SP}x{TP} mesh")


# ---------------------------------------------------------------------------------------------------- host side
def _q(w, dtype):
    """w on a device dtype's grid (the reference's weights as the op holds them)."""
    return ttnn.to_torch(ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


class Weights:
    """Seeded random weights of a case, torch Linear layouts ([out, in])."""

    def __init__(self, model: Model, seed: int):
        g = torch.Generator().manual_seed(seed)
        r = lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc  # noqa: E731
        H, Is = model.H, model.Is
        self.gate, self.up, self.down = r(Is, H, sc=H**-0.5), r(Is, H, sc=H**-0.5), r(H, Is, sc=Is**-0.5)
        self.router, self.bias = r(model.E, H, sc=H**-0.5), r(model.E, sc=0.01)
        self.latent_down = self.latent_up = self.latent_norm = None
        if model.latent:
            L = model.latent
            self.latent_down = r(L, H, sc=H**-0.5)
            self.latent_up = r(H, L, sc=L**-0.5)
            self.latent_norm = 1.0 + r(L, sc=0.1)


def _shared_ref(model, w, x):
    """x [T, H] (bf16 values) -> the shared expert's output [T, H], fp32, on the bf8 weights."""
    b8 = ttnn.bfloat8_b
    g, u, d = _q(w.gate, b8), _q(w.up, b8), _q(w.down, b8)
    return weights_py.act_ref(x @ g.T, x @ u.T, model.act) @ d.T


def _router_ref(model, w, x):
    """x [T, H] -> (ids [T, K] sorted, weights [T, K] in the ids' order), fp32 on the bf16 router weights."""
    logits = x @ _q(w.router, ttnn.bfloat16).T
    if model.score == "Sigmoid":
        s = torch.sigmoid(logits)
    else:
        s = torch.sqrt(torch.nn.functional.softplus(logits))
    sel = torch.topk(s + w.bias, model.K, dim=-1).indices
    ids, order = torch.sort(sel, dim=-1)
    ws = torch.gather(s, 1, ids)
    ws = ws / ws.sum(-1, keepdim=True) * model.route_scale
    return ids, ws


def _rmsnorm(z, gamma, eps):
    return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps) * gamma


def _x_host(case, model):
    g = torch.Generator().manual_seed(1000 + case.seed)
    return torch.randn(case.chunk, model.H, generator=g).to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------------- mesh layouts
def _to_residual(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """[chunk, W] -> [1, 1, chunk / SP, W / TP] a chip (the residual's layout)."""
    return ttnn.from_torch(
        t.reshape(1, 1, *t.shape),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(SP, TP), dims=(2, 3)),
    )


def _to_owner(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT):
    """[chunk, W] -> [1, 1, S, W] a chip, the owners' rows (chip (r, c) owns tokens (r TP + c) S ..)."""
    return ttnn.from_torch(
        t.reshape(1, 1, *t.shape),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
    )


def _from_residual(mesh, t, chunk):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=(SP, TP), dims=(2, 3))).reshape(
        chunk, -1
    )


def _from_owner(t, S):
    """per-chip [.., S', W] (the first S rows the chip's own tokens) -> [chunk, W] in token order."""
    return torch.cat([ttnn.to_torch(d).reshape(-1, d.shape[-1])[:S] for d in ttnn.get_device_tensors(t)], dim=0)


def _host_ints(t):
    """an integer output's per-chip host copies (determinism on the host: small tensors)."""
    return [ttnn.to_torch(d).clone() for d in ttnn.get_device_tensors(t)]


def _pcc(name, ref, got, floor):
    ok, pcc = comp_pcc(ref.float(), got.float(), floor)
    logger.info(f"  {name}: pcc {pcc}")
    assert ok, f"{name}: pcc {pcc} < {floor}"


# ---------------------------------------------------------------------------------------------------- head
class Head:
    def __init__(self, mesh, case):
        self.mesh, self.case = mesh, case
        self.model = model = MODELS[case.model]
        self.dtypes = mb.MoEDtypes()
        cfg = mb.MoEBlockConfig(chunk_tokens=case.chunk, num_links=2, persist_l1_small=True)
        self.res = mb.MoEBlockResources(mesh, model.dims(), cfg, self.dtypes)
        self.A = self._accumulators()
        fit = mb.plan_fused_head(self.res, self.dtypes, self.A, None)
        assert fit.ok, fit.why
        lay = fit.planned[0].lay
        logger.info(f"{case.id}: head plan GY {lay.GY} MT {lay.MT} NCH {lay.NCH} xdram {lay.xdram}; A {self.A}")
        assert not lay.xdram, f"{case.id}: the head took its DRAM fallback (x_full not wired in this test)"
        self.w = Weights(model, case.seed)
        f32 = lambda t: None if t is None else ttnn.from_torch(t, dtype=ttnn.float32)  # noqa: E731
        w = self.w
        self.tw = mb.prepare_fused_head_weights(
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
        return mb.fused_head(self.x, self.tw, resources=self.res, dtypes=self.dtypes, accumulators=self.A)

    def close(self):
        self.res.release()


@pytest.mark.timeout(1800)
@PARAMS
def test_fused_head_accuracy(mesh_device, device_params, case):
    _skip_unless_lb(mesh_device)
    head = Head(mesh_device, case)
    model, chunk = head.model, case.chunk
    S, N, K = chunk // (SP * TP), SP * TP, model.K
    out = head()
    x = head.x_host.float()
    try:
        logger.info(f"FUSED_HEAD accuracy {case.id}:")
        # the shared expert
        se_ref = _shared_ref(model, head.w, x)
        per_chip = ttnn.get_device_tensors(out.se)[0].shape
        se = _from_owner(out.se, S) if per_chip[-1] == model.H else _from_residual(mesh_device, out.se, chunk)
        _pcc("se", se_ref, se, 0.99)
        # the routing rows: a token's K ids, then its K gate weights as bf16 bits
        rows = torch.cat(
            [ttnn.to_torch(d).reshape(-1, 2 * K).to(torch.int64)[:S] for d in ttnn.get_device_tensors(out.rows)]
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
        for me, d in enumerate(ttnn.get_device_tensors(out.xbuf)):
            t = ttnn.to_torch(d)
            xb.append(t.reshape(-1, t.shape[-1])[me * S : (me + 1) * S])
        xb = torch.cat(xb)
        if model.latent:
            _pcc("xbuf (latent rows)", x @ _q(head.w.latent_down, head.dtypes.projection_weights).T, xb, 0.99)
        else:
            assert torch.equal(xb.to(torch.bfloat16), head.x_host), "xbuf: the own slots are not x"
            logger.info("  xbuf: own slots = x, bit-exact")
    finally:
        head.close()


@pytest.mark.timeout(1800)
@PARAMS
def test_fused_head_determinism(mesh_device, device_params, case):
    _skip_unless_lb(mesh_device)
    head = Head(mesh_device, case)
    det = DeterminismMarker(mesh_device)
    ints = ("rows", "route", "counts", "regions", "token_index", "table", "plan")
    first = None
    try:
        for call in range(CALLS):
            out = head()
            # (xbuf: the head writes only its own slots - the rest stays as the first call left it: all comparable)
            xb = ttnn.to_layout(out.xbuf, ttnn.TILE_LAYOUT)
            det.add({"se": out.se, "xbuf": xb})
            got = {n: _host_ints(getattr(out, n)) for n in ints}
            if first is None:
                first = got
            else:
                for n in ints:
                    for chip, (a, b) in enumerate(zip(first[n], got[n])):
                        assert torch.equal(a, b), f"{case.id}: call {call}: {n} differs on chip {chip}"
        det.assert_identical(f"fused_head {case.id} (se, xbuf; + {', '.join(ints)} on the host)")
    finally:
        head.close()


@pytest.mark.timeout(1800)
@PARAMS
def test_fused_head_perf(mesh_device, device_params, case):
    _skip_unless_lb(mesh_device)
    head = Head(mesh_device, case)
    try:
        head()  # (warm: program build)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        head()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        t = traced_op_device_times(mesh_device, tid, PERF_RUNS, HEAD_OPS)["fused_head"]
        ttnn.release_trace(mesh_device, tid)
    finally:
        head.close()
    us = median_us(t)
    logger.info(f"PERF fused_head {case.id}: median {us:.1f} us over {len(t)} replays")
    BASELINES.check(f"fused_head {case.id}", us, case.margin)


# ---------------------------------------------------------------------------------------------------- tail
class Tail:
    def __init__(self, mesh, case):
        self.mesh, self.case = mesh, case
        self.model = model = MODELS[case.model]
        cfg = mb.MoEBlockConfig(chunk_tokens=case.chunk, num_links=2, persist_l1_small=True)
        self.res = mb.MoEBlockResources(mesh, model.dims(), cfg, mb.MoEDtypes())
        self.w = Weights(model, case.seed)
        g = torch.Generator().manual_seed(2000 + case.seed)
        chunk = case.chunk
        self.y_host = torch.randn(chunk, model.routed_width, generator=g).to(torch.bfloat16)
        self.se_host = (torch.randn(chunk, model.H, generator=g) * 0.5).to(torch.bfloat16)
        self.y = _to_owner(mesh, self.y_host)
        self.w_up = None
        if model.latent:
            # (K3: the shared expert's output in the residual's layout)
            self.se = _to_residual(mesh, self.se_host)
            f32 = lambda t: ttnn.from_torch(t, dtype=ttnn.float32)  # noqa: E731
            self.w_up = mb.prepare_fused_tail_weights(
                f32(self.w.latent_up), f32(self.w.latent_norm), resources=self.res, dtype=ttnn.bfloat8_b
            )
        else:
            # (GLM: the owners' rows)
            self.se = _to_owner(mesh, self.se_host, layout=ttnn.TILE_LAYOUT)

    def reference(self):
        y, se = self.y_host.float(), self.se_host.float()
        if not self.model.latent:
            return y + se
        # (the op folds the norm's weight into W_up before its bf8 packing: the reference too)
        eps = self.model.dims().norm_eps
        up = _q(self.w.latent_up * self.w.latent_norm[None, :], ttnn.bfloat8_b)
        return _rmsnorm(y, 1.0, eps) @ up.T + se

    def __call__(self):
        return mb.fused_tail(self.y, self.se, resources=self.res, w_up=self.w_up)

    def close(self):
        self.res.release()


@pytest.mark.timeout(1800)
@PARAMS
def test_fused_tail_accuracy(mesh_device, device_params, case):
    _skip_unless_lb(mesh_device)
    tail = Tail(mesh_device, case)
    try:
        out = tail()
        got = _from_residual(mesh_device, out, case.chunk).float()
        ref = tail.reference()
        logger.info(f"FUSED_TAIL accuracy {case.id}:")
        _pcc("out", ref, got, 0.999 if tail.model.latent is None else 0.99)
        if tail.model.latent is None:
            exact = float((got.to(torch.bfloat16) == ref.to(torch.bfloat16)).float().mean())
            logger.info(f"  out = bf16(y_own + se) on {exact:.4f} of the elements")
            assert torch.allclose(got, ref, rtol=1e-2, atol=1e-2), "out != y_own + se"
    finally:
        tail.close()


@pytest.mark.timeout(1800)
@PARAMS
def test_fused_tail_determinism(mesh_device, device_params, case):
    _skip_unless_lb(mesh_device)
    tail = Tail(mesh_device, case)
    det = DeterminismMarker(mesh_device)
    try:
        for _ in range(CALLS):
            out = tail()
            det.add({"out": out})
            ttnn.deallocate(out)
        det.assert_identical(f"fused_tail {case.id}")
    finally:
        tail.close()


@pytest.mark.timeout(1800)
@PARAMS
def test_fused_tail_perf(mesh_device, device_params, case):
    _skip_unless_lb(mesh_device)
    tail = Tail(mesh_device, case)
    try:
        ttnn.deallocate(tail())  # (warm: program build)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = tail()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        t = traced_op_device_times(mesh_device, tid, PERF_RUNS, TAIL_OPS)["fused_tail"]
        ttnn.release_trace(mesh_device, tid)
        ttnn.deallocate(out)
    finally:
        tail.close()
    us = median_us(t)
    logger.info(f"PERF fused_tail {case.id}: median {us:.1f} us over {len(t)} replays")
    BASELINES.check(f"fused_tail {case.id}", us, case.margin)


# ---------------------------------------------------------------------------------------------------- token loops
@pytest.mark.timeout(1800)
@PARAMS
def test_fused_tail_loops_one_call(mesh_device, device_params, case):
    """MoEBlockConfig.token_loops 2: the tail over both loops' rows in one call (all_loops, its kernels loop) is
    bit-identical to a call a loop on each loop's rows (y_own, the shared expert's output and the output: loop 1's rows
    after loop 0's)."""
    _skip_unless_lb(mesh_device)
    k = 2
    if case.row_tokens % (32 * TP):
        pytest.skip(f"{case.id}: a loop's rows a mesh row ({case.row_tokens}) not a multiple of {32 * TP}")
    model = MODELS[case.model]
    chunk = k * case.chunk  # (a loop: the case's rows)
    cfg = mb.MoEBlockConfig(chunk_tokens=chunk, num_links=2, persist_l1_small=True)
    cfg.token_loops, cfg.chunk_splits = k, 1
    res = mb.MoEBlockResources(mesh_device, model.dims(), cfg, mb.MoEDtypes())
    try:
        w = Weights(model, case.seed)
        g = torch.Generator().manual_seed(2000 + case.seed)
        y = _to_owner(mesh_device, torch.randn(chunk, model.routed_width, generator=g).to(torch.bfloat16))
        se_host = (torch.randn(chunk, model.H, generator=g) * 0.5).to(torch.bfloat16)
        w_up = None
        if model.latent:
            se = _to_residual(mesh_device, se_host)
            f32 = lambda t: ttnn.from_torch(t, dtype=ttnn.float32)  # noqa: E731
            w_up = mb.prepare_fused_tail_weights(
                f32(w.latent_up), f32(w.latent_norm), resources=res, dtype=ttnn.bfloat8_b
            )
        else:
            se = _to_owner(mesh_device, se_host, layout=ttnn.TILE_LAYOUT)

        def rows(t, i):
            n, wd = int(t.shape[-2]) // k, int(t.shape[-1])
            return ttnn.slice(t, [0, 0, i * n, 0], [1, 1, (i + 1) * n, wd], [1, 1, 1, 1])

        ref = ttnn.concat([mb.fused_tail(rows(y, i), rows(se, i), resources=res, w_up=w_up) for i in range(k)], dim=2)
        for _ in range(3):  # (repeated: the loop words re-armed between calls)
            got = mb.fused_tail(y, se, resources=res, w_up=w_up, all_loops=True)
            diff = sum(
                int((ttnn.to_torch(a) != ttnn.to_torch(b)).sum())
                for a, b in zip(ttnn.get_device_tensors(got), ttnn.get_device_tensors(ref))
            )
            logger.info(f"FUSED_TAIL one call {case.id} x {k} loops: {diff} elements differ from a call a loop")
            assert diff == 0
    finally:
        res.release()
