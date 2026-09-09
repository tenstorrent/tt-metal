"""Baseline: the 18 legal (dtype, output_dtype) pairs on the CURRENT code,
with SUPPORTED widened in-memory only. Evidence before implementation."""
import torch, ttnn
from ttnn.operations import tilize as T

DT = [ttnn.bfloat16, ttnn.float32, ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8]
OD = [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8]
T.SUPPORTED["dtype"] = DT
T.SUPPORTED["output_dtype"] = OD

INT = (ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8)
FLOAT_OUT = (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat4_b)
WIDTH = {ttnn.uint32: 4, ttnn.int32: 4, ttnn.uint16: 2, ttnn.uint8: 1}


def legal(i, o):
    if i in INT:
        return o in INT and WIDTH[i] == WIDTH[o]
    return o in FLOAT_OUT


TORCH = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
    ttnn.bfloat4_b: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.uint32: torch.int32,
    ttnn.uint16: torch.int32,
    ttnn.int32: torch.int32,
    ttnn.uint8: torch.uint8,
}


def mk(dt, shape):
    if dt == ttnn.uint8:
        return torch.randint(0, 100, shape, dtype=torch.uint8)
    if dt in (ttnn.uint32, ttnn.uint16):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    if dt == ttnn.int32:
        return torch.randint(-1000, 1000, shape, dtype=torch.int32)
    if dt == ttnn.float32:
        return torch.randn(shape, dtype=torch.float32)
    return torch.randn(shape).bfloat16()


shape = (1, 1, 64, 128)
torch.manual_seed(0)
for i in DT:
    for o in OD:
        if not legal(i, o):
            continue
        try:
            x = mk(i, shape)
            tt = ttnn.from_torch(
                x, dtype=i, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            out = T.tilize(tt, dtype=o)
            got = ttnn.to_torch(out)
            exp = x.to(TORCH[o])
            d = (got.to(torch.float64) - exp.to(torch.float64)).abs()
            print(
                f"{str(i):24s} -> {str(o):24s}  max_abs={float(d.max()):.6g}  nmismatch={int((got!=exp).sum())}/{got.numel()}"
            )
        except Exception as e:
            print(f"{str(i):24s} -> {str(o):24s}  EXC {type(e).__name__}: {str(e)[:180]}")
