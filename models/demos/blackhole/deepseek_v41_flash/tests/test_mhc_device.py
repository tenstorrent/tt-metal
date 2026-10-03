# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device test: V4.1 mHC (mixes / collapse / expand) vs the checkpoint's own Block methods, real layer weights."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@pytest.mark.parametrize("which", ["attn", "ffn"])
@torch.no_grad()
def test_mhc(mesh_device, which):
    torch.manual_seed(0)
    rows, cols = tuple(mesh_device.shape)
    per_dev = 4
    T = per_dev * rows
    blk = R.build_layer(0, max_batch_size=2, max_seq_len=64)
    tok = torch.randint(1000, 100000, (T, 1))
    h, pm = R.embed_tokens(tok)  # [T, 1, 4, 5120]; real embedding streams
    x = h.reshape(T, 4, 5120)
    # make the streams differ so the mixing is exercised
    x = x + 0.02 * torch.randn_like(x)
    fn = getattr(blk, f"hc_{which}_fn")
    base = getattr(blk, f"hc_{which}_base")
    scale = getattr(blk, f"hc_{which}_scale")

    # reference
    ref_pre, ref_post, ref_comb = blk.hc_mixes(x.reshape(1, T, 4, 5120), fn, scale, base)
    sub_out = torch.randn(1, T, 5120).to(torch.bfloat16)  # stand-in for the sublayer output
    ref_collapse = blk.hc_pre(x.reshape(1, T, 4, 5120), ref_pre)
    ref_new = blk.hc_post(sub_out, x.reshape(1, T, 4, 5120), ref_post, ref_comb)

    mhc = DSV41MHC(mesh_device, fn.data, base.data, scale.data)
    shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=(rows, cols))
    up = lambda t, dt: ttnn.from_torch(
        t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    tt_x = up(x.reshape(1, 1, T, 4 * 5120).float(), ttnn.float32)
    tt_sub = up(sub_out.reshape(1, 1, T, 5120).float(), ttnn.float32)
    pre, post, comb = mhc.mixes(tt_x)
    new = mhc.expand(tt_sub, tt_x, post, comb)
    col = mhc.collapse(tt_x, pre)

    comp = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=(rows, cols))
    first = lambda t: ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=(rows, cols))
    )

    def rows_only(t):  # tokens sharded over rows, replicated over cols: take column 0 of each row
        devs = ttnn.get_device_tensors(t)
        return torch.cat([ttnn.to_torch(devs[r * cols]) for r in range(rows)], dim=2)

    d_pre, d_post, d_comb = (
        rows_only(pre).reshape(T, 4),
        rows_only(post).reshape(T, 4),
        rows_only(comb).reshape(T, 4, 4),
    )
    d_new, d_col = rows_only(new).reshape(T, 4, 5120), rows_only(col).reshape(T, 5120)
    res = {
        "pre": R.pcc(d_pre, ref_pre.reshape(T, 4)),
        "post": R.pcc(d_post, ref_post.reshape(T, 4)),
        "comb": R.pcc(d_comb, ref_comb.reshape(T, 4, 4)),
        "collapse": R.pcc(d_col, ref_collapse.reshape(T, 5120)),
        "expand": R.pcc(d_new, ref_new.reshape(T, 4, 5120)),
    }
    print(f"mhc[{which}] PCC:", {k: round(v, 5) for k, v in res.items()})
    assert all(v > 0.999 for v in res.values()), res
