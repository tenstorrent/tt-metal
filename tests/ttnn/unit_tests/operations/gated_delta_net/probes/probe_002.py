import torch, ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    make_inputs,
    reference,
    compute_config,
)
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_bf16 import _rel_metrics, _to_device

device = ttnn.open_device(device_id=0)
for inputs in [
    ((1, 1, 2560), (1, 12, 128, 128)),
    ((1, 16, 2560), (16, 12, 128, 128)),
    ((1, 64, 2560), (1, 12, 128, 128)),
    ((1, 10, 256), (2, 3, 32, 64)),
]:
    torch.manual_seed(0)
    t = make_inputs(inputs)
    for n in ORDER:
        if n != "recurrent_state":
            t[n] = t[n].to(torch.bfloat16).float()
    exp = reference(*(t[n] for n in ORDER))
    out, cs, rs = gated_delta_net(*_to_device(t, device, None), compute_kernel_config=compute_config())
    o = _rel_metrics(ttnn.to_torch(out).float(), exp[0])
    s = _rel_metrics(ttnn.to_torch(rs).float(), exp[2])
    print(
        f"ACC {inputs} out pcc={o[0]:.6f} rms={o[1]:.5f} state pcc={s[0]:.7f} rms={s[1]:.6f} cs_exact={torch.equal(ttnn.to_torch(cs).float(), exp[1])}"
    )
ttnn.close_device(device)
