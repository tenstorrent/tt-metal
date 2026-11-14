import torch, ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    compute_config,
    make_inputs,
    reference,
    to_device,
)

device = ttnn.open_device(device_id=0)
inputs = ((1, 1, 256), (1, 3, 32, 64))
torch.manual_seed(42)
t = make_inputs(inputs)
exp = reference(*(t[n] for n in ORDER))
out, cs, rs = gated_delta_net(*to_device(t, device), compute_kernel_config=compute_config())
o = ttnn.to_torch(out).float().reshape(3, 64)
e = exp[0].reshape(3, 64)
print("RATIO", (o / e).mean(1), (o / e).std(1))
# compute unnormalized o from reference state
z = t["z"].reshape(3, 64)
g = z * torch.sigmoid(z)
w = t["norm_weight"]
oraw = e / (w * g)
print("ms", (oraw**2).mean(1), (oraw**2).sum(1))
ttnn.close_device(device)
