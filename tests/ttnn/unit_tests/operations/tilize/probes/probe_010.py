# Characterize the dtype axis WITHOUT widening SUPPORTED: dispatch straight past
# validate() (the same door _bench_tilize.py uses) and see what the existing,
# dtype-generic program descriptor actually produces. Data for Refinement 7.
import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch


def src(shape, dt):
    torch.manual_seed(42)
    if dt in (ttnn.uint8, ttnn.uint16, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    if dt == ttnn.int32:
        return torch.randint(-1000, 1000, shape, dtype=torch.int32)
    if dt == ttnn.float32:
        return torch.randn(shape, dtype=torch.float32)
    return torch.randn(shape).bfloat16()


def pcc(a, b):
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    if torch.allclose(a, b):
        return 1.0, 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1]), float((a - b).abs().max())


shape = (1, 1, 64, 128)
cases = [
    (ttnn.float32, None),
    (ttnn.uint32, None),
    (ttnn.uint8, None),
    (ttnn.uint16, None),
    (ttnn.int32, None),
    (ttnn.bfloat16, ttnn.float32),
    (ttnn.float32, ttnn.bfloat16),
    (ttnn.bfloat16, ttnn.bfloat8_b),
]
for dt, out_dt in cases:
    t = src(shape, dt)
    tag = f"{dt}->{out_dt or dt}"
    try:
        tt_in = ttnn.from_torch(
            t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = _dispatch(tt_in, dtype=out_dt)
        got = ttnn.to_torch(out)
        p, m = pcc(t, got)
        print(f"RESULT {tag}: pcc={p:.6f} maxdiff={m} exact={p==1.0}")
    except Exception as e:
        print(f"RESULT {tag}: EXC {type(e).__name__}: {str(e)[:200]}")
