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
    "narrowW4->H8": (W(S, 4), H(S, 8)),
    "narrowW4->H32": (W(S, 4), H(S, 32, _crs2(8, 4))),
    "fullH4->W8": (H(S, 4), W(S, 8)),
    "fullH8->H2": (H(S, 8), H(S, 2)),
    "dram->H8": (None, H(S, 8)),
    "narrowW8->H8": (W(S, 8), H(S, 8)),
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
            row[zc] = f"ERR {type(e).__name__}: {str(e)[:90]}"
    on, off = row[1], row[0]
    try:
        print(f"RESULT {name}: on={on} off={off} ratio={off/on:.2f}x")
    except Exception:
        print(f"RESULT {name}: on={on} off={off}")
ttnn.close_device(device)
