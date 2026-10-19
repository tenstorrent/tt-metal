# Scratch op-level cases (not for merge): ttnn.tilize on the tilize_block paths #60672 and its fix change.
import pytest
import torch
import ttnn

CASES = {
    "int32_512x1024": (ttnn.int32, (1, 1, 512, 1024)),
    "uint32_512x1024": (ttnn.uint32, (1, 1, 512, 1024)),
    "fp32_512x1024": (ttnn.float32, (1, 1, 512, 1024)),
    "int32_2048x64": (ttnn.int32, (1, 1, 2048, 64)),
    "bf16_4096x32": (ttnn.bfloat16, (1, 1, 4096, 32)),
    "bf16_to_bfp8_2048x64": (ttnn.bfloat16, (1, 1, 2048, 64)),
}


@pytest.mark.parametrize("case", list(CASES))
def test_tz(device, case):
    dt, shape = CASES[case]
    if dt in (ttnn.int32, ttnn.uint32):
        t = torch.randint(0, 1000, shape, dtype=torch.int32)
    else:
        t = torch.randn(shape, dtype=torch.float32 if dt == ttnn.float32 else torch.bfloat16)
    x = ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    if case.startswith("bf16_to_bfp8"):
        y = ttnn.tilize(x, dtype=ttnn.bfloat8_b, use_multicore=True)
        assert torch.allclose(ttnn.to_torch(y).float(), t.float(), rtol=0.05, atol=0.05)
        return
    y = ttnn.tilize(x, use_multicore=True)
    out = ttnn.to_torch(y)
    assert torch.equal(out.to(t.dtype), t)
