import torch, ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    compute_config,
    make_inputs,
    reference,
)
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_bf16 import _rel_metrics
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_perf import _to_device

device = ttnn.open_device(device_id=0)
for T in (64, 128, 512):
    torch.manual_seed(0)
    t = make_inputs(((1, T, 2560), (1, 12, 128, 128)))
    for n in ORDER:
        if n != "recurrent_state":
            t[n] = t[n].to(torch.bfloat16).to(torch.float32)
    exp = reference(*(t[n] for n in ORDER))
    out, cs, rs = (
        ttnn.to_torch(x).float()
        for x in gated_delta_net(*_to_device(t, device, ttnn.bfloat16), compute_kernel_config=compute_config())
    )
    print(
        f"T={T}",
        "out",
        _rel_metrics(out, exp[0]),
        "state",
        _rel_metrics(rs, exp[2]),
        "cs exact",
        torch.equal(cs, exp[1]),
    )
ttnn.close_device(device)
