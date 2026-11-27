import torch, ttnn, sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/g6_typecast")
from test_g6_typecast_sources import _fp32_input, _bits16
from ttnn.operations.g6_typecast import g6_typecast

dev = ttnn.open_device(device_id=0)
x = _fp32_input((1, 1, 32, 32))
t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
got = ttnn.to_torch(g6_typecast(t, ttnn.bfloat16))
exp = x.to(torch.bfloat16)
bad = (_bits16(got) != _bits16(exp)).flatten()
xi = x.flatten().view(torch.int32)[bad]
print("R in", [hex(v & 0xFFFFFFFF) for v in xi.tolist()])
print(
    "R got",
    [hex(v & 0xFFFF) for v in _bits16(got).flatten()[bad].tolist()],
    "exp",
    [hex(v & 0xFFFF) for v in _bits16(exp).flatten()[bad].tolist()],
)
ttnn.close_device(dev)
