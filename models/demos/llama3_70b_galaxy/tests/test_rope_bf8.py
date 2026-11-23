# rotary_embedding_llama with dtype=bfloat8_b (packs the rotation straight to bf8) vs bf16 rotation + typecast, at the
# DP4 x TP8 prefill shape (8 q heads, M rows, head dim 128), one Blackhole chip: equality and time.
#   pytest models/demos/llama3_70b_galaxy/tests/test_rope_bf8.py -s       env: RB_M (4096,16384)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_common import get_rot_transformation_mat


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_rope_bf8(device):
    D = 128
    for M in [int(m) for m in (os.environ.get("RB_M") or "4096,16384").split(",")]:
        torch.manual_seed(0)
        x = torch.randn(1, 8, M, D)
        cos = torch.randn(1, 1, M, D)
        sin = torch.randn(1, 1, M, D)
        trans = get_rot_transformation_mat(32)
        kw = dict(device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tx, tc, ts, tt_ = (ttnn.from_torch(t, **kw) for t in (x, cos, sin, trans))

        def ref():
            y = ttnn.experimental.rotary_embedding_llama(tx, tc, ts, tt_, is_decode_mode=False)
            z = ttnn.typecast(y, ttnn.bfloat8_b)
            ttnn.deallocate(y)
            return z

        def new():
            return ttnn.experimental.rotary_embedding_llama(tx, tc, ts, tt_, is_decode_mode=False, dtype=ttnn.bfloat8_b)

        a, b = ttnn.to_torch(ref()).float(), ttnn.to_torch(new()).float()
        pcc = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
        out = []
        for fn in (ref, new):
            ttnn.deallocate(fn())
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            for _ in range(20):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(device)
            out.append((time.perf_counter() - t0) / 20 * 1e3)
        logger.info(
            f"ROPEBF8 M={M}: bf16+typecast {out[0]:.3f} ms, dtype=bf8 {out[1]:.3f} ms, pcc {pcc:.6f}, "
            f"max diff {(a - b).abs().max():.4f}, dtype {b.dtype}"
        )
