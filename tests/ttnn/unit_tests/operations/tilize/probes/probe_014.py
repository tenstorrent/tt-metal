import tests.ttnn.unit_tests.operations.tilize._bench_tilize as B
import torch, ttnn

device = ttnn.open_device(device_id=0)
_L1 = ttnn.BufferType.L1
_ROW = ttnn.ShardOrientation.ROW_MAJOR


def _crs(n):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))})


def _crs2(x, y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})


def H(shape, n, grid=None):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _L1, ttnn.ShardSpec(grid or _crs(n), (shape[-2] // n, shape[-1]), _ROW)
    )


def W(shape, n, grid=None):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, _L1, ttnn.ShardSpec(grid or _crs(n), (shape[-2], shape[-1] // n), _ROW)
    )


S = [1, 1, 1024, 256]
CASES = {
    # source-local (writer streams whole tile pages), varying core count
    "H2->dram": (H(S, 2), ttnn.DRAM_MEMORY_CONFIG),
    "H8->dram": (H(S, 8), ttnn.DRAM_MEMORY_CONFIG),
    "H32->dram": (H(S, 32, _crs2(8, 4)), ttnn.DRAM_MEMORY_CONFIG),
    "W8->dram": (W(S, 8), ttnn.DRAM_MEMORY_CONFIG),  # narrow local source shard (wt_chunk=1)
    # destination-local, narrow destination shard in W
    "dram->W8": (None, W(S, 8)),
    "dram->W32": (None, W(S, 32, _crs2(8, 4))),
}
for name, (inc, outc) in CASES.items():
    row = {}
    for zc in (1, 0):
        try:
            row[zc] = B._measure(
                device,
                S,
                ttnn.bfloat16,
                in_mem_config=inc,
                out_mem_config=outc,
                levers=dict(zero_copy=zc),
                label=f"{name}/zc={zc}",
            )
        except Exception as e:
            row[zc] = float("nan")
            print(f"ERR {name} zc={zc}: {type(e).__name__} {str(e)[:90]}")
    print(f"RESULT {name}: on={row[1]} off={row[0]} ratio={row[0]/row[1]:.2f}x")
ttnn.close_device(device)
