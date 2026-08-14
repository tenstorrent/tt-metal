import torch, ttnn
from eval.golden_tests.tilize import helpers as H

device = ttnn.open_device(device_id=0)
_L1 = ttnn.BufferType.L1
_ROW = ttnn.ShardOrientation.ROW_MAJOR
_B = ttnn.TensorMemoryLayout.BLOCK_SHARDED


def _crs(*r):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for (s, e) in r})


def _sh(b, g, s, o, sc):
    return {"kind": "sharded", "buffer": b, "grid": g, "shard_shape": s, "orientation": o, "scheme": sc}


# vary shard width (elements) for bf16 -> page bytes = 2*w
for W, sw in [(128, 64), (96, 48), (96, 24), (100, 50), (128, 32), (96, 12)]:
    s = {
        "input_shape": [1, 1, 64, W],
        "use_multicore": True,
        "shard_api": "legacy_2d",
        "in": _sh(_L1, _crs(((0, 0), (1, 1))), (32, sw), _ROW, _B),
        "out": {"kind": "interleaved", "buffer": ttnn.BufferType.DRAM},
    }
    try:
        mc = H.build_mem_config(device, s["in"])
        x = H.make_torch_input(ttnn.bfloat16, s["input_shape"])
        t = H.create_ttnn_input_tensor(x, device, dtype=ttnn.bfloat16, memory_config=mc)
        pb = t.buffer_page_size()
        cta = list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        del t
    except Exception as e:
        print(f"W={W} sw={sw}: build FAIL {str(e)[:120]}")
        continue
    try:
        H.run_tilize((s,), device=device, dtype=ttnn.bfloat16)
        r = "PASS"
    except Exception as e:
        r = f"FAIL {str(e)[:80]}"
    print(f"W={W} sw={sw} page={pb} pages/row={ -(-W//sw) } cta={cta} -> {r}")
ttnn.close_device(device)
