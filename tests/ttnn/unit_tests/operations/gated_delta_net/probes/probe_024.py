import torch, ttnn, importlib
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs

pd = importlib.import_module("ttnn.operations.gated_delta_net.gated_delta_net_program_descriptor")
op = importlib.import_module("ttnn.operations.gated_delta_net.gated_delta_net")
device = ttnn.open_device(device_id=0)
tensors = make_inputs(((1, 1, 2560), (1, 12, 128, 128)))
keys = []
for _ in range(2):
    dev = [
        ttnn.from_torch(
            tensors[n],
            dtype=ttnn.float32 if n == "recurrent_state" else ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        for n in ORDER
    ]
    keys.append(op._cache_key(tuple(dev), None, None, 1e-6, compute_config()))
for i, (x, y) in enumerate(zip(*keys)):
    if x != y:
        print("KEYDIFF", i, x, y)
print("KEYEQ", keys[0] == keys[1], hash(keys[0]) == hash(keys[1]))
ttnn.close_device(device)
