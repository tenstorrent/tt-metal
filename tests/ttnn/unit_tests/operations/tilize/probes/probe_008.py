import torch, ttnn, importlib
from eval.golden_tests.tilize import helpers as H

PD = importlib.import_module("ttnn.ttnn.operations.tilize.tilize_program_descriptor") if False else None
device = ttnn.open_device(device_id=0)

_L1, _DRAM = ttnn.BufferType.L1, ttnn.BufferType.DRAM
_ROW, _COL = ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR
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
    # --- NEW: narrow-page cross-spec gather (was silently wrong) ---
    "B2x2->H4": {
        "input_shape": [1, 1, 128, 128],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (1, 1))), (64, 64), _ROW, _B),
        "out": _sh(_L1, _crs(((0, 0), (3, 0))), (32, 128), _ROW, _H),
    },
    "W4->H4": {
        "input_shape": [1, 1, 64, 512],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (3, 0))), (64, 128), _ROW, _W),
        "out": _sh(_L1, _crs(((0, 0), (1, 0))), (32, 512), _ROW, _H),
    },
    "B_col->H": {
        "input_shape": [1, 1, 128, 128],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (1, 1))), (64, 64), _COL, _B),
        "out": _sh(_L1, _crs(((0, 0), (3, 0))), (32, 128), _ROW, _H),
    },
    "ndW->il": {
        "input_shape": [1, 1, 64, 512],
        "use_multicore": True,
        "shard_api": "nd",
        "in": _sh(_L1, _crs(((0, 0), (3, 0))), (1, 1, 64, 128), _ROW, None),
        "out": _sh(_L1, _crs(((0, 0), (1, 0))), (1, 1, 32, 512), _ROW, None),
    },
    "W->Wdiff": {
        "input_shape": [1, 1, 64, 512],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (3, 0))), (64, 128), _ROW, _W),
        "out": _sh(_L1, _crs(((0, 0), (1, 0))), (64, 256), _ROW, _W),
    },
    # narrow page + PADDED (cross of both new paths)
    "Bnarrow_pad->H": {
        "input_shape": [1, 1, 100, 100],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "pad_mode": "explicit",
        "output_padded_shape": [1, 1, 128, 128],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (1, 1))), (50, 50), _ROW, _B),
        "out": _sh(_L1, _crs(((0, 0), (3, 0))), (32, 128), _ROW, _H),
    },
    # --- padded x sharded (the lifted EXCLUSIONS) ---
    "pad_nd2nd": {
        "input_shape": [3, 50, 96],
        "use_multicore": True,
        "shard_api": "nd",
        "pad_mode": "explicit",
        "output_padded_shape": [3, 64, 96],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (1, 0))), (2, 50, 96), _ROW, None),
        "out": _sh(_L1, _crs(((0, 0), (1, 0))), (1, 64, 96), _ROW, None),
    },
    "pad_nd2il": {
        "input_shape": [3, 50, 96],
        "use_multicore": True,
        "shard_api": "nd",
        "pad_mode": "explicit",
        "output_padded_shape": [3, 64, 96],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (1, 0))), (2, 50, 96), _ROW, None),
        "out": _il(_L1),
    },
    "pad_il2nd": {
        "input_shape": [3, 50, 96],
        "use_multicore": True,
        "shard_api": "nd",
        "pad_mode": "explicit",
        "output_padded_shape": [3, 64, 96],
        "pad_value": 0.0,
        "in": _il(_DRAM),
        "out": _sh(_L1, _crs(((0, 0), (1, 0))), (1, 64, 96), _ROW, None),
    },
    "pad_lh2il": {
        "input_shape": [3, 100, 128],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "pad_mode": "explicit",
        "output_padded_shape": [3, 128, 128],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (1, 0))), (150, 128), _ROW, _H),
        "out": _il(_L1),
    },
    "pad_il2lh": {
        "input_shape": [3, 100, 64],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "pad_mode": "explicit",
        "output_padded_shape": [3, 128, 64],
        "pad_value": 10.2,
        "in": _il(_DRAM),
        "out": _sh(_L1, _crs(((0, 0), (0, 1))), (192, 64), _ROW, _H),
    },
    "pad_nd2lh": {
        "input_shape": [3, 50, 64],
        "use_multicore": True,
        "shard_api": "nd",
        "pad_mode": "explicit",
        "output_padded_shape": [3, 64, 64],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (0, 1))), (2, 50, 64), _ROW, None),
        "out": _sh(_L1, _crs(((0, 0), (0, 1))), (96, 64), _ROW, _H),
    },
    "pad_lh2nd": {
        "input_shape": [3, 100, 128],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "pad_mode": "explicit",
        "output_padded_shape": [3, 128, 128],
        "pad_value": 10.2,
        "in": _sh(_L1, _crs(((0, 0), (1, 0))), (150, 128), _ROW, _H),
        "out": _sh(_L1, _crs(((0, 0), (1, 0))), (3, 96, 96), _ROW, None),
    },
    "pad_auto_sh": {
        "input_shape": [1, 1, 50, 64],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "pad_mode": "auto",
        "pad_value": -7.5,
        "in": _il(_DRAM),
        "out": _sh(_L1, _crs(((0, 0), (1, 0))), (32, 64), _ROW, _H),
    },
    # --- REGRESSION representatives (prior phases) ---
    "reg_il": {
        "input_shape": [1, 1, 64, 128],
        "use_multicore": True,
        "shard_api": "none",
        "in": _il(_DRAM),
        "out": _il(_DRAM),
    },
    "reg_il_pad": {
        "input_shape": [1, 1, 50, 50],
        "use_multicore": True,
        "shard_api": "none",
        "pad_mode": "explicit",
        "output_padded_shape": [1, 1, 128, 128],
        "pad_value": -18.0,
        "in": _il(_DRAM),
        "out": _il(_DRAM),
    },
    "reg_same_H": {
        "input_shape": [1, 1, 512, 64],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (3, 0))), (128, 64), _ROW, _H),
        "out": _sh(_L1, _crs(((0, 0), (3, 0))), (128, 64), _ROW, _H),
    },
    "reg_same_W": {
        "input_shape": [1, 1, 64, 512],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (3, 0))), (64, 128), _ROW, _W),
        "out": _sh(_L1, _crs(((0, 0), (3, 0))), (64, 128), _ROW, _W),
    },
    "reg_same_B": {
        "input_shape": [1, 1, 128, 128],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (1, 1))), (64, 64), _COL, _B),
        "out": _sh(_L1, _crs(((0, 0), (1, 1))), (64, 64), _COL, _B),
    },
    "reg_cross_il2H": {
        "input_shape": [1, 1, 128, 64],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _il(_DRAM),
        "out": _sh(_L1, _crs(((0, 0), (3, 0))), (32, 64), _ROW, _H),
    },
    "reg_cross_H2il": {
        "input_shape": [1, 1, 128, 64],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (3, 0))), (32, 64), _ROW, _H),
        "out": _il(_DRAM),
    },
    "reg_H4toH2": {
        "input_shape": [1, 1, 128, 64],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (3, 0))), (32, 64), _ROW, _H),
        "out": _sh(_L1, _crs(((0, 0), (1, 0))), (64, 64), _ROW, _H),
    },
    "reg_Wnarrow2il": {
        "input_shape": [1, 1, 64, 512],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (3, 0))), (64, 128), _ROW, _W),
        "out": _il(_DRAM),
    },
}
fails = []
for name, s in SC.items():
    for dt in (ttnn.bfloat16, ttnn.float32):
        try:
            H.run_tilize((s,), device=device, dtype=dt)
            print(f"[{name}/{dt}] PASS")
        except Exception as e:
            print(f"[{name}/{dt}] FAIL {type(e).__name__}: {str(e)[:200]}")
            fails.append(name)
print("FAILS:", sorted(set(fails)))
ttnn.close_device(device)
