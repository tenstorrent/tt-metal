# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device test of ``paged_ops.paged_gather_rows`` (APC snapshot read of an fp8 / bf16 pool): scatter random rows (including subnormal e4m3 values, |x| < 2^-6, which the device typecast flushes)
through ``paged_scatter_rows`` and gather them back: fp8 == torch's e4m3 round trip bit for bit, bf16 == identity; SKIP ids leave the output untouched; scatter(gather(x)) restores the same bytes."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P

SKIP = 0xFFFFFFFF


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.fp8_e4m3, ttnn.bfloat16], ids=["fp8", "bf16"])
@torch.no_grad()
def test_gather_roundtrip(mesh_device, dtype):
    md = mesh_device
    R, N = 4096, 3000
    torch.manual_seed(0)
    pool_shape = [1, 1, R, 512]
    if dtype == ttnn.bfloat16:
        pool = ttnn.zeros(pool_shape, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
    else:
        pool = ttnn.allocate_tensor_on_device(
            ttnn.Shape(pool_shape), dtype, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )

    class _Pool:  # the two attributes the ops read
        pass

    pl = _Pool()
    pl.pool, pl.dtype, pl.device = pool, dtype, lambda: md
    x = torch.randn(N, 512) * 2
    x[:, :64] *= 2**-7  # subnormal range of e4m3
    x = x.to(torch.bfloat16)
    ids = torch.randperm(R)[:N].to(torch.int32)
    d_ids = ttnn.from_torch(ids.reshape(1, N), device=md, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    d_src = ttnn.from_torch(x, device=md, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    P.paged_scatter_rows(pl, d_src, d_ids)
    out = P.paged_gather_rows(pl, d_ids)
    got = ttnn.to_torch(out).reshape(N, 512)
    ref = x.to(torch.float8_e4m3fn).to(torch.bfloat16) if dtype == ttnn.fp8_e4m3 else x
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), "gather differs from the e4m3 / bf16 round trip"
    # SKIP ids: the rows keep their previous content
    ids2 = ids.clone().to(torch.int64)
    ids2[::2] = SKIP
    d_ids2 = ttnn.from_torch(
        ids2.to(torch.int32).reshape(1, N), device=md, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    out2 = P.paged_gather_rows(
        pl,
        d_ids2,
        out=ttnn.from_torch(
            torch.full((1, 1, N, 512), 7.0), device=md, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        ),
    )
    got2 = ttnn.to_torch(out2).reshape(N, 512)
    assert torch.equal(got2[1::2].view(torch.int16), ref[1::2].view(torch.int16))
    assert (got2[::2] == 7.0).all()
