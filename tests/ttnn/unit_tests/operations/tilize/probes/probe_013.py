import torch, ttnn, traceback
from ttnn.operations.tilize import tilize
from eval.golden_tests.tilize.helpers import make_torch_input, _TORCH_DTYPE, _transition_tolerance
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc

dev = ttnn.open_device(device_id=0)
pairs = []
F = [ttnn.bfloat16, ttnn.float32]
for i in F:
    for o in [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat4_b]:
        pairs.append((i, o))
pairs += [
    (ttnn.uint32, ttnn.uint32),
    (ttnn.uint32, ttnn.int32),
    (ttnn.int32, ttnn.int32),
    (ttnn.int32, ttnn.uint32),
    (ttnn.uint16, ttnn.uint16),
    (ttnn.uint8, ttnn.uint8),
]
shape = [1, 1, 64, 128]
for i, o in pairs:
    try:
        torch.manual_seed(0)
        x = make_torch_input(i, shape)
        t = ttnn.from_torch(x, dtype=i, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y = tilize(t, ttnn.DRAM_MEMORY_CONFIG, dtype=o)
        out = ttnn.to_torch(y)
        exp = x.to(_TORCH_DTYPE[o])
        mode, th = _transition_tolerance(i, o)
        ok, msg = comp_pcc(exp, out, th) if mode == "pcc" else comp_equal(exp, out)
        print("PAIR", i, o, y.dtype, "OK" if ok else "FAIL", msg[:120])
    except Exception as e:
        print("PAIR", i, o, "EXC", str(e)[:300])
ttnn.close_device(dev)
