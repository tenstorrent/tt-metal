import base64
import json
import os
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
import ttnn
from loguru import logger

OPS = [op for op in os.environ.get("DUMP_OPS", "").split(":") if op]
# {op: [TT-NN public name, args, kwargs]}, base64 JSON so an env field that splits on commas keeps it whole.
SPEC = json.loads(base64.b64decode(os.environ["DUMP_SPEC_B64"])) if os.environ.get("DUMP_SPEC_B64") else {}


@pytest.mark.parametrize("op", OPS)
def test_bf16_output_dump(device, op):
    """Every BF16 input through ttnn.<op>; saves src/res (the --silicon-outputs format) and logs the words."""
    public, args, kwargs = SPEC.get(op, [op, [], {}])
    bits = torch.arange(1 << 16, dtype=torch.int32)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 256, 256)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    words = ttnn.to_torch(getattr(ttnn, public)(tt_x, *args, **kwargs)).flatten().view(torch.int16).numpy().astype("<u2")
    out = Path(os.environ.get("DUMP_DIR", "generated/bf16_output_dump"))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{op}-{os.environ.get('DUMP_TAG', 'run')}.npz", src=np.arange(65536, dtype=np.uint16), res=words)
    logger.info(f"DUMP {op} {base64.b64encode(zlib.compress(words.tobytes(), 9)).decode()}")
