import torch, ttnn
import importlib

m = importlib.import_module("ttnn.operations.qkv_proj_heads_rope.qkv_proj_heads_rope")
import sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/qkv_proj_heads_rope")
from test_qkv_proj_heads_rope import make_inputs, reference, pcc, relative_rms

D, L = ttnn.BufferType.DRAM, ttnn.BufferType.L1
for ax in ("buffer_type", "out_buffer_type", "table_buffer_type"):
    m.SUPPORTED[ax] = [L, D]
m.SUPPORTED["alignment"] = ["tile_aligned", "h_non_aligned"]
dev = ttnn.open_device(device_id=0)
mc = {D: ttnn.DRAM_MEMORY_CONFIG, L: ttnn.L1_MEMORY_CONFIG}


def run(b, s, st, xb, ob, tb):
    x, w, bb, c, sn = make_inputs(b, s, st)
    t = lambda a, buf: ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc[buf]
    )
    outs = m.qkv_proj_heads_rope(t(x, xb), t(w, D), t(bb, D), t(c, tb), t(sn, tb), memory_config=mc[ob])
    ref = reference(x, w, bb, c, sn)
    res = []
    for o, r in zip(outs, ref):
        a = ttnn.to_torch(o)
        full = ttnn.to_torch(ttnn.reshape(o, ttnn.Shape(list(o.padded_shape)), ttnn.Shape(list(o.padded_shape))))
        pad = full[..., s:, :]
        res.append((round(pcc(a, r), 7), round(relative_rms(a, r), 5), int((pad != 0).sum())))
    print("PROBE", b, s, st, xb, ob, tb, res)


run(8, 512, 512, D, D, D)
run(4, 256, 512, D, L, D)
run(2, 37, 37, L, L, L)
run(13, 300, 300, D, D, D)
run(16, 8, 8, L, L, L)
run(4096, 1, 1, D, D, D)
ttnn.close_device(dev)
