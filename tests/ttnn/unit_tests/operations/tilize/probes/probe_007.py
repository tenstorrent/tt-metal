import torch, ttnn, traceback
from eval.golden_tests.tilize import helpers as H

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
    # 1. narrow source page: WIDTH-sharded RM in -> interleaved TILE out
    "W_in->il": {
        "input_shape": [1, 1, 64, 512],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (3, 0))), (64, 128), _ROW, _W),
        "out": _il(_DRAM),
    },
    # 2. cross-scheme: WIDTH in -> HEIGHT out
    "W_in->H_out": {
        "input_shape": [1, 1, 64, 512],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (3, 0))), (64, 128), _ROW, _W),
        "out": _sh(_L1, _crs(((0, 0), (3, 0))), (16, 512), _ROW, _H),
    },
    # 3. BLOCK in -> interleaved out (narrow page)
    "B_in->il": {
        "input_shape": [1, 1, 128, 128],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (1, 1))), (64, 64), _ROW, _B),
        "out": _il(_DRAM),
    },
    # 4. HEIGHT in (full width) -> WIDTH out : cross-scheme, wide source page ok
    "H_in->W_out": {
        "input_shape": [1, 1, 64, 512],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (1, 0))), (32, 512), _ROW, _H),
        "out": _sh(_L1, _crs(((0, 0), (3, 0))), (64, 128), _ROW, _W),
    },
    # 5. uneven height shard grid in -> interleaved out
    "Huneven->il": {
        "input_shape": [1, 1, 160, 64],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (2, 0))), (64, 64), _ROW, _H),
        "out": _il(_DRAM),
    },
    # 6. interleaved -> uneven height shard grid out
    "il->Huneven": {
        "input_shape": [1, 1, 160, 64],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _il(_DRAM),
        "out": _sh(_L1, _crs(((0, 0), (2, 0))), (64, 64), _ROW, _H),
    },
    # 7. HEIGHT 4core -> HEIGHT 2core (the R1 cross-spec cell, sanity)
    "H4->H2": {
        "input_shape": [1, 1, 128, 64],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (3, 0))), (32, 64), _ROW, _H),
        "out": _sh(_L1, _crs(((0, 0), (1, 0))), (64, 64), _ROW, _H),
    },
    # 8. BLOCK in -> BLOCK out different grid
    "B2x2->B4x1": {
        "input_shape": [1, 1, 128, 128],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (1, 1))), (64, 64), _ROW, _B),
        "out": _sh(_L1, _crs(((0, 0), (3, 0))), (32, 128), _ROW, _H),
    },
}
for name, s in SC.items():
    try:
        mc = H.build_mem_config(device, s["in"])
        x = H.make_torch_input(ttnn.bfloat16, s["input_shape"])
        t = H.create_ttnn_input_tensor(x, device, dtype=ttnn.bfloat16, memory_config=mc)
        pb = t.buffer_page_size()
        del t
    except Exception as e:
        pb = f"ERR {e}"
    try:
        H.run_tilize((s,), device=device, dtype=ttnn.bfloat16)
        print(f"[{name}] in_page={pb} PASS")
    except Exception as e:
        print(f"[{name}] in_page={pb} FAIL {type(e).__name__}: {str(e)[:200]}")
ttnn.close_device(device)
