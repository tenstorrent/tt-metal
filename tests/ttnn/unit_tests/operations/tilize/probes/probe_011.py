import torch, ttnn
from eval.golden_tests.tilize import helpers as H

device = ttnn.open_device(device_id=0)
_L1, _DRAM = ttnn.BufferType.L1, ttnn.BufferType.DRAM
_ROW = ttnn.ShardOrientation.ROW_MAJOR
_H, _B = ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED


def _crs(*r):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for (s, e) in r})


def _sh(b, g, s, o, sc):
    return {"kind": "sharded", "buffer": b, "grid": g, "shard_shape": s, "orientation": o, "scheme": sc}


SC = {
    # unaligned (100 B) narrow page + pad -> must be a typed refusal, not wrong data
    "c_w50_pad_il": {
        "input_shape": [1, 1, 100, 100],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "pad_mode": "explicit",
        "output_padded_shape": [1, 1, 128, 128],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (1, 1))), (50, 50), _ROW, _B),
        "out": {"kind": "interleaved", "buffer": _DRAM},
    },
    "c_w50_pad_H": {
        "input_shape": [1, 1, 100, 100],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "pad_mode": "explicit",
        "output_padded_shape": [1, 1, 128, 128],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (1, 1))), (50, 50), _ROW, _B),
        "out": _sh(_L1, _crs(((0, 0), (3, 0))), (32, 128), _ROW, _H),
    },
    # tile-aligned narrow page + pad: still correct
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
}
for name, s in SC.items():
    try:
        H.run_tilize((s,), device=device, dtype=ttnn.bfloat16)
        print(f"[{name}] PASS")
    except NotImplementedError as e:
        print(f"[{name}] REFUSED (typed): {str(e)[:140]}")
    except Exception as e:
        print(f"[{name}] FAIL {type(e).__name__}: {str(e)[:160]}")
ttnn.close_device(device)
