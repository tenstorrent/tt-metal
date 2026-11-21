# Stock ttnn ops while the weight stream's sub-devices are loaded: which ones run, and where.
import pytest
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.stream.stream import Stream


@pytest.mark.parametrize("op", ["add", "transpose", "layernorm", "embedding"])
def test_stream_stock(device, op):
    st = Stream(device)
    try:
        a = ttnn.from_torch(torch.randn(1, 1, 32, 5120), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        if op == "add":
            y = ttnn.add(a, a)
        elif op == "transpose":
            b = ttnn.from_torch(torch.randn(1, 24, 1, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            y = ttnn.transpose(b, 1, 2)
        elif op == "layernorm":
            y = ttnn.rms_norm(a, weight=None)
        else:
            w = ttnn.from_torch(
                torch.randn(1000, 5120), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            i = ttnn.from_torch(torch.tensor([[3]], dtype=torch.int32), dtype=ttnn.uint32, device=device)
            y = ttnn.embedding(i, w)
        ttnn.synchronize_device(device)
        print(f"STOCK {op} ok {list(y.shape)}", flush=True)
    finally:
        st.close()
