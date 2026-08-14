import torch, ttnn
import importlib

T = importlib.import_module("ttnn.operations.tilize.tilize")
from eval.golden_tests.tilize import helpers as H

device = ttnn.open_device(device_id=0)

# Lift the pad x sharded EXCLUSIONS in-place (the eventual Refinement-2 edit).
T.EXCLUSIONS[:] = [e for e in T.EXCLUSIONS if not ("pad_mode" in e and "shard_api" in e)]

_L1, _DRAM = ttnn.BufferType.L1, ttnn.BufferType.DRAM
_ROW = ttnn.ShardOrientation.ROW_MAJOR
_HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def _crs(*ranges):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for (s, e) in ranges})


def _il(b):
    return {"kind": "interleaved", "buffer": b}


def _sh(b, g, s, o, sc):
    return {"kind": "sharded", "buffer": b, "grid": g, "shard_shape": s, "orientation": o, "scheme": sc}


SCEN = {
    "nd2nd_uneven": {
        "input_shape": [3, 50, 96],
        "use_multicore": True,
        "shard_api": "nd",
        "pad_mode": "explicit",
        "output_padded_shape": [3, 64, 96],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (1, 0))), (2, 50, 96), _ROW, None),
        "out": _sh(_L1, _crs(((0, 0), (1, 0))), (1, 64, 96), _ROW, None),
    },
    "nd2il": {
        "input_shape": [3, 50, 96],
        "use_multicore": True,
        "shard_api": "nd",
        "pad_mode": "explicit",
        "output_padded_shape": [3, 64, 96],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (1, 0))), (2, 50, 96), _ROW, None),
        "out": _il(_L1),
    },
    "il2nd": {
        "input_shape": [3, 50, 96],
        "use_multicore": True,
        "shard_api": "nd",
        "pad_mode": "explicit",
        "output_padded_shape": [3, 64, 96],
        "pad_value": 0.0,
        "in": _il(_DRAM),
        "out": _sh(_L1, _crs(((0, 0), (1, 0))), (1, 64, 96), _ROW, None),
    },
    "lh2il": {
        "input_shape": [3, 100, 128],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "pad_mode": "explicit",
        "output_padded_shape": [3, 128, 128],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (1, 0))), (150, 128), _ROW, _HEIGHT),
        "out": _il(_L1),
    },
    "il2lh": {
        "input_shape": [3, 100, 64],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "pad_mode": "explicit",
        "output_padded_shape": [3, 128, 64],
        "pad_value": 10.2,
        "in": _il(_DRAM),
        "out": _sh(_L1, _crs(((0, 0), (0, 1))), (192, 64), _ROW, _HEIGHT),
    },
    "nd2lh": {
        "input_shape": [3, 50, 64],
        "use_multicore": True,
        "shard_api": "nd",
        "pad_mode": "explicit",
        "output_padded_shape": [3, 64, 64],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (0, 1))), (2, 50, 64), _ROW, None),
        "out": _sh(_L1, _crs(((0, 0), (0, 1))), (96, 64), _ROW, _HEIGHT),
    },
    "lh2nd": {
        "input_shape": [3, 100, 128],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "pad_mode": "explicit",
        "output_padded_shape": [3, 128, 128],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (1, 0))), (150, 128), _ROW, _HEIGHT),
        "out": _sh(_L1, _crs(((0, 0), (1, 0))), (3, 96, 96), _ROW, None),
    },
}

for name, s in SCEN.items():
    # page-size introspection on the input side
    try:
        mc = H.build_mem_config(device, s["in"])
        x = H.make_torch_input(ttnn.bfloat16, s["input_shape"])
        t = H.create_ttnn_input_tensor(x, device, dtype=ttnn.bfloat16, memory_config=mc)
        print(f"[{name}] in page_bytes={t.buffer_page_size()} shape={list(t.shape)} padded={list(t.padded_shape)}")
        del t
    except Exception as e:
        print(f"[{name}] page probe FAILED: {type(e).__name__}: {e}")
    try:
        H.run_tilize((s,), device=device, dtype=ttnn.bfloat16)
        print(f"[{name}] PASS")
    except Exception as e:
        msg = str(e)
        print(f"[{name}] FAIL {type(e).__name__}: {msg[:400]}")

ttnn.close_device(device)
