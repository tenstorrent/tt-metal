import torch, ttnn
import ttnn.operations.g6_expand.g6_expand_program_descriptor as pd
from ttnn.operations.g6_expand import g6_expand, last_plan

device = ttnn.open_device(device_id=0)


def run(shape, sizes, cap=None, eighths=None, rows=None):
    old = (pd.COL_BLOCK_CAP, pd.READER_WRITE_EIGHTHS, pd.REPLICATE_CPU_ROWS)
    if cap:
        pd.COL_BLOCK_CAP = cap
    if eighths is not None:
        pd.READER_WRITE_EIGHTHS = eighths
    if rows is not None:
        pd.REPLICATE_CPU_ROWS = rows
    pd._cache.clear()
    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    tx = ttnn.fill_implicit_tile_padding(
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device), 1000.0
    )
    got = ttnn.to_torch(g6_expand(tx, sizes))
    exp = x.expand(*[d if s == -1 else s for d, s in zip(shape, sizes)]).contiguous()
    ok = torch.equal(got.view(torch.int16), exp.view(torch.int16))
    print("CASE", shape, sizes, cap, eighths, rows, "OK" if ok else "MISMATCH", dict(last_plan))
    pd.COL_BLOCK_CAP, pd.READER_WRITE_EIGHTHS, pd.REPLICATE_CPU_ROWS = old
    pd._cache.clear()


run([2, 3, 1, 100], [-1, -1, 70, -1])
run([1, 1, 1, 512], [1, 1, 512, 512], cap=1)
run([1, 1, 1, 500], [1, 1, 40, 500], cap=1)
run([4, 1, 1, 200], [-1, -1, 333, -1], cap=2, eighths=6, rows=1)
run([1, 1, 1, 16], [1, 1, 5, 16])
run([1, 1, 1, 4096], [1, 1, 64, 4096], rows=32, eighths=0)
ttnn.close_device(device)
