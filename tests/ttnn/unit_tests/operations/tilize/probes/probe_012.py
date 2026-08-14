import torch, ttnn
from eval.golden_tests.tilize import helpers as H
from ttnn.operations.tilize import tilize, validate
from ttnn.operations.tilize import tilize_program_descriptor as pd

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


def placements(shape, in_mc, out_mc, dtype=ttnn.bfloat16, **padkw):
    t = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=in_mc,
    )
    plan = validate(t, out_mc, dtype=dtype, **padkw)
    o = ttnn.allocate_tensor_on_device(
        ttnn.Shape(plan.target), plan.out_dtype, ttnn.TILE_LAYOUT, device, plan.out_memory_config
    )
    d = pd.create_program_descriptor(t, o, plan)
    return (
        d.kernels[0].compile_time_args[1],
        d.kernels[1].compile_time_args[0],
        d.kernels[0].compile_time_args[0],
        d.kernels[0].compile_time_args[14],
        d.cbs[0].has_buffer(),
        d.cbs[1].has_buffer(),
        d.kernels[0].core_ranges.num_cores(),
    )


# uneven HEIGHT grid: 160 rows / 3 shards of 64 -> 64,64,32
un_h = ttnn.MemoryConfig(_H, _L1, ttnn.ShardSpec(_crs(((0, 0), (2, 0))), (64, 64), _ROW))
print("uneven same-spec:", placements([1, 1, 160, 64], un_h, un_h))
print("uneven in -> dram:", placements([1, 1, 160, 64], un_h, ttnn.DRAM_MEMORY_CONFIG))
print("dram -> uneven out:", placements([1, 1, 160, 64], ttnn.DRAM_MEMORY_CONFIG, un_h))
# padded + sharded out (should be ACCESSOR/LOCAL)
sh2 = ttnn.MemoryConfig(_H, _L1, ttnn.ShardSpec(_crs(((0, 0), (1, 0))), (32, 64), _ROW))
print("pad -> sharded out:", placements([1, 1, 50, 64], ttnn.DRAM_MEMORY_CONFIG, sh2, pad_value=-7.5))
# narrow-page cross-spec
bl = ttnn.MemoryConfig(_B, _L1, ttnn.ShardSpec(_crs(((0, 0), (1, 1))), (64, 64), _ROW))
h4 = ttnn.MemoryConfig(_H, _L1, ttnn.ShardSpec(_crs(((0, 0), (3, 0))), (32, 128), _ROW))
print("block -> height:", placements([1, 1, 128, 128], bl, h4))
# hot path must be untouched: regime R_ALIGNED (0), src_row_pages 1
print("interleaved hot path:", placements([1, 1, 64, 128], ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG))

SC = {
    "un_same": {
        "input_shape": [1, 1, 160, 64],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (2, 0))), (64, 64), _ROW, _H),
        "out": _sh(_L1, _crs(((0, 0), (2, 0))), (64, 64), _ROW, _H),
    },
    "un_in2dram": {
        "input_shape": [1, 1, 160, 64],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (2, 0))), (64, 64), _ROW, _H),
        "out": _il(_DRAM),
    },
    "dram2un": {
        "input_shape": [1, 1, 160, 64],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _il(_DRAM),
        "out": _sh(_L1, _crs(((0, 0), (2, 0))), (64, 64), _ROW, _H),
    },
    "un_cross": {
        "input_shape": [1, 1, 160, 64],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (2, 0))), (64, 64), _ROW, _H),
        "out": _sh(_L1, _crs(((0, 0), (1, 0))), (96, 64), _ROW, _H),
    },
    "un_pad": {
        "input_shape": [1, 1, 150, 64],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "pad_mode": "auto",
        "pad_value": 3.5,
        "in": _il(_DRAM),
        "out": _sh(_L1, _crs(((0, 0), (2, 0))), (64, 64), _ROW, _H),
    },
    "nd_un": {
        "input_shape": [3, 64, 96],
        "use_multicore": True,
        "shard_api": "nd",
        "in": _sh(_L1, _crs(((0, 0), (1, 0))), (2, 64, 96), _ROW, None),
        "out": _sh(_L1, _crs(((0, 0), (1, 0))), (2, 64, 96), _ROW, None),
    },
}
for name, s in SC.items():
    for dt in (ttnn.bfloat16, ttnn.float32):
        try:
            H.run_tilize((s,), device=device, dtype=dt)
            print(f"[{name}/{dt}] PASS")
        except Exception as e:
            print(f"[{name}/{dt}] FAIL {type(e).__name__}: {str(e)[:180]}")
ttnn.close_device(device)
