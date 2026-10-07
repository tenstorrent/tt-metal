# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58723): would swapping the operands of a block-float x bf16 multiply (so that the block-float one
is SrcB and HiFi2 is exact) keep main's bits? Prints, per case, how many output elements of multiply(a, b) and multiply(b, a)
differ. Run with EB_R3_NO_HIFI2=1 (both orders at HiFi4) and without it (b, a takes the HiFi2 rule)."""
import os
import pytest
import torch
import ttnn
from test_eb_fid import DT, _mem


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


CASES = [
    ("bfp8_bf16_dram", "bfp8", "dram", None),
    ("bfp8_bf16_l1", "bfp8", "l1", None),
    ("bfp8_bf16_hs8", "bfp8", "hs8", None),
    ("bfp8_bf16_bs64", "bfp8", "bs64", None),
    ("bfp4_bf16_hs8", "bfp4", "hs8", None),
    ("bfp8_bf16_dram_fp32out", "bfp8", "dram", ttnn.float32),
    ("bfp8_bf16_l1_fp32out", "bfp8", "l1", ttnn.float32),
]


@pytest.mark.parametrize("cid, da, mem, out_dt", CASES, ids=[c[0] for c in CASES])
def test_swap(device, cid, da, mem, out_dt):
    torch.manual_seed(5)
    shape = (1, 1, 4096, 1280) if mem == "bs64" else (1, 1, 1024, 1024)
    a = torch.randn(shape, dtype=torch.bfloat16) * 3
    b = torch.randn(shape, dtype=torch.bfloat16) * 3
    mc = _mem(mem, shape)
    ta = ttnn.from_torch(a, dtype=DT[da], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    kw = dict(fast_and_approximate_mode=True, memory_config=mc)
    if out_dt is not None:
        kw["dtype"] = out_dt
    ab = ttnn.to_torch(ttnn.multiply(ta, tb, **kw))
    ba = ttnn.to_torch(ttnn.multiply(tb, ta, **kw))
    exact = ttnn.to_torch(ta).float() * ttnn.to_torch(tb).float()
    exact = exact.to(ab.dtype)
    n = ab.numel()
    d_swap = int((ab.view(torch.int16 if ab.dtype == torch.bfloat16 else torch.int32) != ba.view(torch.int16 if ba.dtype == torch.bfloat16 else torch.int32)).sum())
    d_ab = int((ab != exact).sum())
    d_ba = int((ba != exact).sum())
    mode = "both HiFi4" if os.environ.get("EB_R3_NO_HIFI2") else "b*a at HiFi2"
    print(f"\nSWAP {cid} [{mode}] elements {n}: a*b vs b*a differ {d_swap}; a*b vs rounded exact differ {d_ab}; b*a vs rounded exact differ {d_ba}")
