"""Do the ported head-op variants still run on the p150 against current main?

The batched-barrier kernels were written in May and have never been compiled
against this tt-metal. Run every variant once and compare against the stock op,
which is the reference the p150 sweeps used.
"""

import torch

import ttnn
from models.demos.wormhole.bge_m3.tt.custom_ops.fused_concat_heads import (
    bge_concat_heads_headsplit,
    bge_concat_heads_stock,
    bge_concat_heads_tracka,
)
from models.demos.wormhole.bge_m3.tt.custom_ops.fused_qkv_heads import (
    bge_qkv_heads_headsplit,
    bge_qkv_heads_scatter,
    bge_qkv_heads_stock,
    bge_qkv_heads_tracka,
)

BATCH, SEQ, HEADS, HEAD_DIM = 1, 512, 16, 64
HIDDEN = HEADS * HEAD_DIM

d = ttnn.open_device(device_id=0)
torch.manual_seed(0)

qkv_host = torch.randn((BATCH, 1, SEQ, 3 * HIDDEN), dtype=torch.bfloat16)
qkv = ttnn.from_torch(
    qkv_host, device=d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
)


def check(name, fn):
    try:
        q, k, v = fn(qkv, num_heads=HEADS)
        shapes = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
        finite = all(bool(torch.isfinite(ttnn.to_torch(t).float()).all()) for t in (q, k, v))
        print("  %-26s -> q%s finite=%s" % (name, shapes[0], finite))
        return ttnn.to_torch(q).float()
    except NotImplementedError as exc:
        print("  %-26s -> SKIPPED (stub): %s" % (name, str(exc)[:38]))
    except Exception as exc:
        print("  %-26s -> FAILED: %s" % (name, str(exc).split("\n")[0][:48]))
    return None


print("QKV head-split variants, B%d/S%d:" % (BATCH, SEQ))
ref = check("bge_qkv_heads_stock", bge_qkv_heads_stock)
for nm, fn in (("bge_qkv_heads_tracka", bge_qkv_heads_tracka), ("bge_qkv_heads_headsplit", bge_qkv_heads_headsplit)):
    got = check(nm, fn)
    if ref is not None and got is not None:
        print("      max abs diff vs stock: %.3e" % (ref - got).abs().max().item())
check("bge_qkv_heads_scatter", bge_qkv_heads_scatter)

print("\nConcat-heads variants:")
ctx_host = torch.randn((BATCH, HEADS, SEQ, HEAD_DIM), dtype=torch.bfloat16)
ctx = ttnn.from_torch(
    ctx_host, device=d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
for nm, fn in (
    ("bge_concat_heads_stock", bge_concat_heads_stock),
    ("bge_concat_heads_tracka", bge_concat_heads_tracka),
    ("bge_concat_heads_headsplit", bge_concat_heads_headsplit),
):
    try:
        out = fn(ctx) if nm.endswith("stock") else fn(ctx, head_groups=1)
        print(
            "  %-26s -> %s finite=%s" % (nm, tuple(out.shape), bool(torch.isfinite(ttnn.to_torch(out).float()).all()))
        )
    except TypeError:
        try:
            out = fn(ctx)
            print("  %-26s -> %s" % (nm, tuple(out.shape)))
        except Exception as exc:
            print("  %-26s -> FAILED: %s" % (nm, str(exc).split("\n")[0][:48]))
    except Exception as exc:
        print("  %-26s -> FAILED: %s" % (nm, str(exc).split("\n")[0][:48]))

ttnn.close_device(d)
