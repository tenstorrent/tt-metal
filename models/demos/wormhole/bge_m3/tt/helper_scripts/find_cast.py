"""Find the per-layer BF16->BFP8 typecast by tracing every ttnn.typecast call.

The Tracy capture shows 24 BF16 => BFP8 typecasts at B8, 522 us, one per layer,
sitting between the head split and SDPA. Reading the source did not settle which
call site produces them, so this wraps ttnn.typecast and reports the caller.
"""

import collections
import traceback

import torch

import ttnn

real_typecast = ttnn.typecast
seen = collections.Counter()


def traced_typecast(tensor, dtype=None, *args, **kwargs):
    target = dtype if dtype is not None else (args[0] if args else None)
    frame = None
    for entry in reversed(traceback.extract_stack()[:-1]):
        if "bge_m3" in entry.filename:
            frame = entry
            break
    src = getattr(tensor, "dtype", None)
    key = (
        "%s:%d %s" % (frame.filename.split("bge_m3/")[-1], frame.lineno, frame.name) if frame else "unknown",
        str(src).split(".")[-1],
        str(target).split(".")[-1],
    )
    seen[key] += 1
    return (
        real_typecast(tensor, dtype, *args, **kwargs) if dtype is not None else real_typecast(tensor, *args, **kwargs)
    )


ttnn.typecast = traced_typecast

from models.demos.wormhole.bge_m3.tt.common import create_tt_model

device = ttnn.open_device(device_id=0)
args, model, _ = create_tt_model(mesh_device=device, max_batch_size=8, max_seq_len=512, dtype=ttnn.bfloat8_b)
torch.manual_seed(0)
tokens = torch.randint(0, 1000, (8, 512), dtype=torch.int32)

seen.clear()
model.forward(ttnn.from_torch(tokens, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT))

print("  ttnn.typecast call sites during one forward:")
for (site, src, dst), n in seen.most_common():
    print("    x%-4d %-14s -> %-10s %s" % (n, src, dst, site))
ttnn.close_device(device)
