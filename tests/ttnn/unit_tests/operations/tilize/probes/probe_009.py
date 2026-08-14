import torch, ttnn
from eval.golden_tests.tilize import helpers as H

device = ttnn.open_device(device_id=0)
_L1, _DRAM = ttnn.BufferType.L1, ttnn.BufferType.DRAM
_ROW = ttnn.ShardOrientation.ROW_MAJOR
_H, _W, _B = (
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
)


def _crs(*r):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for (s, e) in r})


def _il(b):
    return {"kind": "interleaved", "buffer": b}


def _sh(b, g, s, o, sc):
    return {"kind": "sharded", "buffer": b, "grid": g, "shard_shape": s, "orientation": o, "scheme": sc}


SC = {
    # (a) tile-aligned-width narrow page source + pad -> interleaved out
    "a_w64_pad_il": {
        "input_shape": [1, 1, 100, 128],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "pad_mode": "explicit",
        "output_padded_shape": [1, 1, 128, 128],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (1, 1))), (50, 64), _ROW, _B),
        "out": _il(_DRAM),
    },
    # (b) same but sharded (local) out
    "b_w64_pad_H": {
        "input_shape": [1, 1, 100, 128],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "pad_mode": "explicit",
        "output_padded_shape": [1, 1, 128, 128],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (1, 1))), (50, 64), _ROW, _B),
        "out": _sh(_L1, _crs(((0, 0), (3, 0))), (32, 128), _ROW, _H),
    },
    # (c) NON-32B-multiple shard width (50 elems = 100 B page) + pad -> interleaved
    "c_w50_pad_il": {
        "input_shape": [1, 1, 100, 100],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "pad_mode": "explicit",
        "output_padded_shape": [1, 1, 128, 128],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (1, 1))), (50, 50), _ROW, _B),
        "out": _il(_DRAM),
    },
    # (d) NON-32B shard width, NO pad -> interleaved  (aligned W=100 impossible; use W=96 shard 48)
    "d_w48_nopad_il": {
        "input_shape": [1, 1, 64, 96],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (1, 1))), (32, 48), _ROW, _B),
        "out": _il(_DRAM),
    },
}
for name, s in SC.items():
    try:
        mc = H.build_mem_config(device, s["in"])
        x = H.make_torch_input(ttnn.bfloat16, s["input_shape"])
        t = H.create_ttnn_input_tensor(x, device, dtype=ttnn.bfloat16, memory_config=mc)
        print(f"[{name}] page={t.buffer_page_size()} padded={list(t.padded_shape)}")
        del t
    except Exception as e:
        print(f"[{name}] tensor build FAIL {type(e).__name__}: {str(e)[:150]}")
        continue
    try:
        H.run_tilize((s,), device=device, dtype=ttnn.bfloat16)
        print(f"[{name}] PASS")
    except Exception as e:
        print(f"[{name}] FAIL {type(e).__name__}: {str(e)[:160]}")
ttnn.close_device(device)
