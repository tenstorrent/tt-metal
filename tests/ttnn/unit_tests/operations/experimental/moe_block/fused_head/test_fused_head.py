# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.moe_block.fused_head on device, bit-exact against saved reference outputs (fabric ties split by
the source's parity, as main's). A Galaxy (the cases' mesh); skipped without one or without the saved cases
($MOE_BLOCK_REFS/fused_head/<case>/):

    MOE_BLOCK_REFS=/data/mstaletovic/jobs/port/refs pytest tests/ttnn/unit_tests/operations/experimental/moe_block/fused_head/test_fused_head.py

Compared per chip, bit for bit: the shared expert's output se, rows, route, the combine plan (counts, regions,
token_index, table, plan) and the chip's own slots of the dispatch buffer xbuf (the head writes only those; a latent
model's are the latent rows). The cases' bf8 weights are on the bf8 grid: the reference's fp32 -> bf16 -> bf8 upload
and this op's fp32 -> bf8 packing give the same device weights. Run to run, every output bit-identical
(test_fused_head_deterministic)."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

import ttnn

HERE = Path(__file__).resolve().parent
mb = ttnn.experimental.moe_block


def _harness():
    spec = importlib.util.spec_from_file_location("ref_harness", HERE.parent / "ref_harness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hz = _harness()
CASES = hz.list_cases("fused_head") if os.environ.get("MOE_BLOCK_REFS") else []
pytestmark = pytest.mark.skipif(not CASES, reason="no fused_head reference cases (set MOE_BLOCK_REFS)")

# (E, K, H, I, shared I, shared activation, score, route scale, latent): the reference's MoEModels
MODELS = {
    "glm53": (256, 8, 6144, 2048, 2048, "Silu", "Sigmoid", 2.5, None),
    "glm53_flash": (288, 8, 4096, 2048, 2048, ("ClampedSilu", 10.0), "Sigmoid", 2.5, None),
    "k27": (384, 8, 7168, 2048, 2048, "Silu", "Sigmoid", 2.827, None),
    "k3": (896, 16, 7168, 3072, 6144, ("Situ", 4.0, 25.0), "Sigmoid", 1.0, 3584),
    "dsv41_flash": (384, 6, 5120, 2304, 2304, ("ClampedSilu", 10.0), "SqrtSoftplus", 1.5, None),
}


def _act(spec):
    if isinstance(spec, str):
        return mb.ExpertActivation(kind=getattr(mb.ExpertActivationKind, spec))
    if spec[0] == "ClampedSilu":
        return mb.ExpertActivation(kind=mb.ExpertActivationKind.ClampedSilu, limit=spec[1])
    return mb.ExpertActivation(kind=mb.ExpertActivationKind.Situ, beta=spec[1], linear_beta=spec[2])


def dims_of(name):
    E, K, H, I, Is, act, score, scale, latent = MODELS[name]
    return mb.MoEModelDims(
        num_experts=E,
        top_k=K,
        emb_dim=H,
        routed_expert_hidden_dim=I,
        routed_expert_activation=_act(act),
        router=mb.RouterSpec(score_func=getattr(mb.RouterScoreFunc, score), route_scale=scale),
        shared_expert_hidden_dim=Is,
        shared_expert_activation=_act(act),
        latent_dim=latent,
    )


class Head:
    """A case's head on the mesh: resources, weights, x."""

    def __init__(self, mesh, meta, ref):
        cfg = meta["config"]
        self.mesh, self.R, self.C = mesh, *meta["mesh_shape"]
        self.res = mb.MoEBlockResources(
            mesh,
            dims_of(meta["model"]),
            mb.MoEBlockConfig(chunk_tokens=cfg["chunk_tokens"], num_links=cfg["num_links"]),
        )
        self.dtypes = mb.MoEDtypes()
        self.A = cfg["accumulators"]
        self.l1 = cfg.get("l1_free")  # (the free L1 the reference's head planned on: the same layout)
        fit = mb.plan_fused_head(self.res, self.dtypes, self.A, self.l1)
        assert fit.ok, fit.why
        lay = fit.planned[0].lay
        got = {k: getattr(lay, k) for k in ("GY", "MT", "NCH", "KB", "GX", "compact", "xdram")}
        want = None if cfg["layout"] is None else {k: cfg["layout"][k] for k in got}
        assert want is None or got == want, f"planned {got}, the reference built {cfg['layout']}"
        assert not lay.xdram, "the DRAM fallback's cases need x_full (not dumped yet)"
        w = torch.load(hz.case_dir("fused_head", meta["case"]) / "weights.pt")
        host = lambda k: ttnn.from_torch(w[k].float(), dtype=ttnn.float32) if k in w else None  # noqa: E731
        self.w = mb.prepare_fused_head_weights(
            host("gate"),
            host("up"),
            host("down"),
            host("latent_down"),
            host("router_weight"),
            host("router_bias"),
            resources=self.res,
            dtypes=self.dtypes,
            accumulators=self.A,
            l1_free=self.l1,
        )
        self.x = hz.to_mesh(ref, "x", mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def __call__(self):
        return mb.fused_head(
            self.x, self.w, resources=self.res, dtypes=self.dtypes, accumulators=self.A, l1_free=self.l1
        )


@pytest.fixture(scope="module", params=CASES)
def case(request):
    meta, ref = hz.load_case("fused_head", request.param)
    R, C = meta["mesh_shape"]
    if ttnn.get_num_devices() < R * C:
        pytest.skip(f"needs a {R}x{C} mesh")
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, meta["config"].get("fabric", "FABRIC_2D_TORUS_XY")))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(R, C), l1_small_size=0)  # (as the dump)
    head = Head(mesh, meta, ref)
    yield meta, ref, head
    head.res.release()
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_fused_head_bit_exact(case):
    meta, ref, head = case
    out = head()
    got = {}
    # (a latent model's latent rows go to xbuf)
    names = ["se", "rows", "route", "counts", "regions", "token_index", "table", "plan"]
    for n in names:
        for rc, t in hz.from_mesh(getattr(out, n), (head.R, head.C)).items():
            shape = ref[rc][n].shape  # (the reference's rank: the same elements)
            got.setdefault(rc, {})[n] = t.reshape(shape) if t.numel() == ref[rc][n].numel() else t
    # (the dispatch buffer: only this chip's own slots me S .. (me + 1) S are the head's)
    S = meta["config"]["chunk_tokens"] // (head.R * head.C)
    xb = hz.from_mesh(out.xbuf, (head.R, head.C))
    want = {}
    for (r, c), t in xb.items():
        me = r * head.C + c
        got[(r, c)]["xbuf_own"] = t.reshape(-1, t.shape[-1])[me * S : (me + 1) * S]
        exp = ref[(r, c)]["xbuf"]
        want[(r, c)] = dict(ref[(r, c)], xbuf_own=exp.reshape(-1, exp.shape[-1])[me * S : (me + 1) * S])
    hz.assert_bit_exact(got, want, names + ["xbuf_own"])


def test_fused_head_deterministic(case):
    """Three calls on the same x (the first may build the program, the others hit the cache): every output bit-identical
    call to call, independent of the reference."""
    _, _, head = case
    names = ["se", "rows", "route", "counts", "regions", "token_index", "table", "plan", "xbuf"]
    calls = []
    for _ in range(3):
        out = head()
        got = {}
        for n in names:
            for rc, t in hz.from_mesh(getattr(out, n), (head.R, head.C)).items():
                got.setdefault(rc, {})[n] = t.clone()
        calls.append(got)
    for call in (1, 2):
        hz.assert_bit_exact(calls[call], calls[0])
