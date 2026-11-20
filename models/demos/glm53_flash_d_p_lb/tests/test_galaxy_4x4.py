# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The two paths a Galaxy 4x4 (SP 4 x TP 4) adds to the LoudBox model, against references, no checkpoint:

- test_indexer_ring_modes: the indexer's SP-axis ring (GLM_INDEXER_RING_AXIS: cache striped over the mesh rows,
  replicated along them, queries sub-sharded over TP) and, where its stripe is tile aligned, the full-mesh ring, vs
  the replicated-cache indexer (no ring) on the same random weights (reference/fake_weights.py) and inputs, over two
  chunks: the selected token ids per query row must match exactly.
- test_experts_ag_rows: the all-gather MoE with more than two mesh rows (local reduce over all T tokens, reduce-scatter
  over axis 0) in the residual layouts (row share, own rows with the full-mesh gather, replicated) vs an fp32 host reference of the routed experts: GLM's 288 experts (18 per
  chip, the route plan's unaligned case) with random weights in the spec's expert dtype (bfp4 on random Gaussian
  weights: PCC 0.97966 / coef 1.000 on the 4x4 and on the 2x4's two-row fused send-back alike), generated per expert from its seed (host memory
  stays small; the reference regenerates them).

    TT_VISIBLE_DEVICES=<4x4mid carve> TT_MESH_GRAPH_DESC_PATH=$PWD/models/demos/deepseek_v3_d_p/experimental_descriptors/single_bh_galaxy_4x4_torus_xy_graph_descriptor.textproto \\
    BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec_galaxy_4x4.yaml \\
    scripts/run_safe_pytest.sh --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_galaxy_4x4.py -s

GLM_G44_CHUNKS (default "5120"): the indexer chunk sizes (two chunks each, from 0).
"""

import os

import pytest
import torch

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
CHUNKS = [int(v) for v in os.environ.get("GLM_G44_CHUNKS", "5120").split(",")]


def _rows_to_mesh(mesh, t):
    """host [S, W] -> chip (r, c) holds rows (r C + c) S/n .. (the split residual layout), bf16 TILE."""
    import ttnn

    rows, cols = tuple(mesh.shape)
    return ttnn.from_torch(
        t.reshape(rows, cols, t.shape[0] // (rows * cols), t.shape[1]).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(rows, cols)),
    )


def _mesh_rows_to_host(t):
    """per chip [.., s, W] in row-major chip order -> host [n s, W]."""
    import ttnn

    return torch.cat([ttnn.to_torch(d).reshape(-1, d.shape[-1]) for d in ttnn.get_device_tensors(t)])


@mesh_parametrize
@pytest.mark.parametrize("chunk", CHUNKS)
def test_indexer_ring_modes(mesh_device, chunk):
    import ttnn
    from models.demos.glm53_flash_d_p.reference.fake_weights import FakeLoader
    from models.demos.glm53_flash_d_p.tt import indexer as ix

    loader = FakeLoader()
    cfg = loader.cfg
    S.hooks().apply_device_settings(S)
    layer = int(S.get("block_types.dsa_moe.representative"))
    n = mesh_device.get_num_devices()
    max_seq = 2 * chunk
    modes = {"replicated": (False, None), "axis": (True, "1")}
    if chunk // n % (ix.KP * ttnn.TILE_SIZE) == 0:
        modes["mesh"] = (True, "0")

    torch.manual_seed(0)
    xs = [torch.randn(chunk, cfg.hidden_size) for _ in range(2)]
    qs = [torch.randn(chunk, cfg.q_lora_rank) for _ in range(2)]
    out = {}
    for name, (ring, axis) in modes.items():
        ix.RING_AXIS = axis
        idx = ix.build_indexer(mesh_device, loader, cfg, layer, max_seq, [chunk], ring=ring)
        if ring:
            assert idx.ring_axis == (axis == "1"), (name, idx.ring_axis)
        got = []
        for i, start in enumerate((0, chunk)):
            x, q = _rows_to_mesh(mesh_device, xs[i]), _rows_to_mesh(mesh_device, qs[i])
            ids = idx(x, q, start, q_local=True, x_local=True)
            got.append(_mesh_rows_to_host(ids).to(torch.int64) & 0xFFFFFFFF)
            for t in (x, q, ids):
                ttnn.deallocate(t)
        out[name] = got
        del idx
    ix.RING_AXIS = os.environ.get("GLM_INDEXER_RING_AXIS")

    fails = []
    for name in modes:
        if name == "replicated":
            continue
        for i, (a, b) in enumerate(zip(out[name], out["replicated"])):
            same = (a.sort(-1).values == b.sort(-1).values).all(-1)
            frac = float(same.float().mean())
            print(f"[indexer chunk {chunk} #{i}] {name} vs replicated: {frac * 100:.3f}% rows identical", flush=True)
            if frac < 1.0:
                fails.append(f"{name} chunk #{i}: {int((~same).sum())} of {same.numel()} rows differ")
    assert not fails, "; ".join(fails)


class _RandomExperts:
    """Expert e's HF weights ((out, in), the values bf16-exact), random from seed e: generated on access."""

    def __init__(self, n, H, I):
        self.n, self.H, self.I = n, H, I

    def __len__(self):
        return self.n

    def __getitem__(self, e):
        g = torch.Generator().manual_seed(1000 + e)
        H, I = self.H, self.I
        bf = lambda t: t.to(torch.bfloat16).float()  # noqa: E731
        return {
            "gate_proj": bf(torch.randn(I, H, generator=g) * H**-0.5),
            "up_proj": bf(torch.randn(I, H, generator=g) * H**-0.5),
            "down_proj": bf(torch.randn(H, I, generator=g) * I**-0.5),
        }


def _experts_reference(x, idx, w, experts, limit=10.0):
    """x [S, H], idx / w [S, K] -> sum_k w[t, k] * (silu(min(x g^T, 10)) * clamp(x u^T, +-10)) d^T, fp32."""
    y = torch.zeros_like(x)
    for e in range(len(experts)):
        t, k = (idx == e).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        ew, xe = experts[e], x[t]
        h = torch.nn.functional.silu((xe @ ew["gate_proj"].T).clamp(max=limit))
        h = h * (xe @ ew["up_proj"].T).clamp(-limit, limit)
        y.index_add_(0, t, (h @ ew["down_proj"].T) * w[t, k, None])
    return y


@mesh_parametrize
def test_experts_ag_rows(mesh_device):
    import ttnn
    from models.demos.glm53_flash_d_p.reference.fake_weights import FakeLoader
    from models.demos.glm53_flash_d_p.tt.common import replicate, replicated_to_host
    from models.demos.glm53_flash_d_p.tt.experts_ag import TtExpertsAg

    hooks = S.hooks()
    hooks.apply_device_settings(S)

    cfg = FakeLoader().cfg
    rows, cols = tuple(mesh_device.shape)
    E, K, H, I = cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.hidden_size, cfg.moe_intermediate_size
    s = int(S.get("target.chunk"))
    torch.manual_seed(0)
    weights = _RandomExperts(E, H, I)
    x = torch.randn(s, H).to(torch.bfloat16).float()
    idx = torch.rand(s, E).argsort(-1)[:, :K]
    w = torch.rand(s, K).to(torch.bfloat16).float()
    dense = torch.zeros(s, E).scatter_(1, idx, w)
    want = _experts_reference(x, idx, w, weights)

    mod = TtExpertsAg(
        mesh_device, 0, weights, E, H, I, top_k=K, max_seq_len=s, weights_dtype=hooks.experts_dtype(S), cache=False
    )
    xb, db = x.reshape(1, 1, s, H).to(torch.bfloat16), dense.reshape(1, 1, s, E).to(torch.bfloat16)
    by_row = lambda t: ttnn.from_torch(  # noqa: E731  mesh row r holds rows [r S/R, (r+1) S/R) on all its chips
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(2, None)),
    )
    got = {}
    xs, ds = by_row(xb), by_row(db)
    y = mod(xs, dense=ds, split=True)
    got["split"] = _mesh_rows_to_host(y).float()
    ttnn.deallocate(y)
    xo, do = _rows_to_mesh(mesh_device, x), _rows_to_mesh(mesh_device, dense)  # own rows (GLM_MOE_FULL_MESH)
    y = mod(xo, dense=do, split=True, full_mesh=True)
    got["full_mesh"] = _mesh_rows_to_host(y).float()
    ttnn.deallocate(y)
    xr, dr = replicate(mesh_device, xb), replicate(mesh_device, db)
    y = mod(xr, dense=dr)
    got["replicated"] = replicated_to_host(y).reshape(s, H).float()
    ttnn.deallocate(y)

    fails = []
    for lay, g in got.items():
        gc, wc = (g - g.mean()).double(), (want - want.mean()).double()
        pcc = float((gc * wc).sum() / (gc.norm() * wc.norm()))
        coef = float((g.double() * want.double()).sum() / (want.double() ** 2).sum())
        rel = float((g - want).norm() / want.norm())
        print(f"[experts {rows}x{cols} {lay}] pcc={pcc:.6f} rel={rel:.5f} coef={coef:.5f}", flush=True)
        if not (torch.isfinite(g).all() and pcc >= 0.975 and abs(coef - 1) <= 0.03):
            fails.append(f"{lay}: pcc {pcc:.6f} coef {coef:.5f}")
    assert not fails, "; ".join(fails)
