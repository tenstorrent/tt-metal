# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""fused_decode GemmaBoundary (tt/fused_decode/boundary.py) vs torch on the QB2 1x4 mesh (WS5).

entry -> attn -> moe on random per-device partial sums (the fabric all-reduce is real): every output against the
torch reference of the Gemma 4 decoder sandwich norms, and bit-identical on every device. Producers are BF16 TILE
tensors in L1, interleaved and width-sharded (the o_proj output layout).
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4_26b_a4b_qb2.tt.fused_decode.boundary import (
    GemmaBoundary,
    flat_from_tile_rows,
    flat_to_tile_rows,
)

H, EPS = 2816, 1e-6


def rms(v, g=None):
    v = v.float()
    out = v * torch.rsqrt(v.pow(2).mean() + EPS)
    return out if g is None else out * g.float()


def rows0(t):
    return [ttnn.to_torch(x).float().reshape(-1, t.padded_shape[-1])[0, :H] for x in ttnn.get_device_tensors(t)]


@pytest.mark.parametrize("pregathered", [False, True], ids=["gather", "pregathered"])
@pytest.mark.parametrize("round_n", [True, False], ids=["round_n", "fp32_n"])
@pytest.mark.parametrize("sharded", [False, True], ids=["interleaved", "sharded"])
def test_gemma_boundary(qb2_mesh, sharded, round_n, pregathered):
    mesh = qb2_mesh
    torch.manual_seed(0)
    b = GemmaBoundary(mesh, H, EPS, shared_dtype=ttnn.bfloat16, round_n=round_n)

    def tile_rows(rows, replicate=False):
        x = torch.zeros(len(rows), 1, 32, H)
        for i, r in enumerate(rows):
            x[i, 0, 0] = r
        return ttnn.from_torch(
            x.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if replicate else ttnn.ShardTensorToMesh(mesh, dim=0),
        )

    g = {k: (1 + 0.3 * torch.randn(H)).bfloat16() for k in ("in", "pa", "e", "f1", "f2", "f", "nx")}
    vf = torch.randn(H) * 0.05
    scalar = 0.73
    emb = (torch.randn(H) * 3).bfloat16()
    o_parts = [torch.randn(H).bfloat16() for _ in range(4)]
    s_parts = [torch.randn(H).bfloat16() for _ in range(4)]
    r_parts = [(torch.randn(H) * 0.5).bfloat16() for _ in range(4)]
    o_t, s_t, r_t = tile_rows(o_parts), tile_rows(s_parts), tile_rows(r_parts)
    if sharded:
        o_t = ttnn.to_memory_config(
            o_t,
            ttnn.create_sharded_memory_config(
                (32, H // 8),
                ttnn.CoreGrid(x=8, y=1),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
        )

    def check(name, got, ref, min_pcc):
        for dev, v in enumerate(got):
            assert torch.equal(v, got[0]), f"{name}: device {dev} differs from device 0"
        ok, pcc = comp_pcc(ref, got[0], min_pcc)
        print(f"BOUNDARY {name} {pcc}")
        assert ok, f"{name}: {pcc}"

    # entry: residual = the embedded token, x = input_norm(residual)
    x = b.entry(tile_rows([emb], replicate=True), b.gammas([g["in"]]))
    assert torch.equal(b.residual().bfloat16(), emb)
    check("entry_x", rows0(x), rms(emb, g["in"]), 0.9999)

    # attn: res += post_attention_norm(sum of the o_proj partials); n = norm(res)
    router, expert, shared = b.attn(o_t, b.gammas([g["pa"], g["e"]]), b.fp32_vector(vf))
    res1 = (emb.float() + rms(sum(p.float() for p in o_parts), g["pa"])).bfloat16()
    n = rms(res1)
    if round_n:
        n = n.bfloat16().float()
    check("attn_router_in", rows0(router), n * vf, 0.9999)
    check("attn_expert_in", rows0(expert), n * g["e"].float(), 0.9999)
    check("attn_shared_in", rows0(shared), n, 0.9999)
    res1_dev = flat_from_tile_rows(ttnn.to_torch(ttnn.get_device_tensors(b.res[1])[0]).float(), H)
    check("attn_residual", [res1_dev], res1.float(), 0.9999)

    # moe: res = (res + ff_norm(ff1_norm(sum shared) + ff2_norm(sum routed))) * layer_scalar; x = next_norm(res)
    g_moe = b.gammas([g["f1"], g["f2"], g["f"], g["nx"]])
    if pregathered:
        # the producers' flat partials, written straight into the boundary's partial buffer (moe_decode flat_out)
        T = b.T
        flat = torch.stack(
            [
                flat_to_tile_rows(torch.cat([s_parts[d].float(), torch.zeros(T * 1024 - H), r_parts[d].float()]), 2 * T)
                for d in range(4)
            ]
        )[:, None]
        staged = ttnn.from_torch(
            flat.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=b.part.memory_config(),
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        ttnn.copy(staged, b.part)
        x = b.moe_pregathered(g_moe, scalar)
    else:
        x = b.moe(s_t, r_t, g_moe, scalar)
    c = rms(sum(p.float() for p in s_parts), g["f1"]) + rms(sum(p.float() for p in r_parts), g["f2"])
    res2 = ((res1.float() + rms(c, g["f"])) * scalar).bfloat16()
    check("moe_residual", [b.residual()], res2.float(), 0.9999)
    check("moe_x", rows0(x), rms(res2, g["nx"]), 0.9999)
