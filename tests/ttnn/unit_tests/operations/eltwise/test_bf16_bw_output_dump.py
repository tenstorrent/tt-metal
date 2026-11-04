import base64
import os
import zlib

import pytest
import torch
import ttnn
from loguru import logger

OPS = [op for op in os.environ.get("DUMP_OPS", "").split(":") if op]
GRADS = ("ones", "random0")


@pytest.mark.parametrize("grad", GRADS)
@pytest.mark.parametrize("op", OPS)
def test_bf16_bw_output_dump(device, op, grad):
    """Every BF16 input x with a fixed grad through ttnn.<op>(grad, x); logs each output's words."""
    x = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(1, 1, 256, 256)
    if grad == "ones":
        g = torch.ones_like(x)
    else:
        g = torch.randn(x.shape, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_g = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for index, out in enumerate(getattr(ttnn, op)(tt_g, tt_x)):
        words = ttnn.to_torch(out).to(torch.bfloat16).flatten().view(torch.int16).numpy().astype("<u2")
        logger.info(f"DUMP {op}:{grad}:{index} {base64.b64encode(zlib.compress(words.tobytes(), 9)).decode()}")
