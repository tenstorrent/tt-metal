import torch, ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    compute_config,
    make_inputs,
    reference,
)
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_perf import _to_device
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_bf16 import _rel_metrics

device = ttnn.open_device(device_id=0)
for inputs, dt, name in [
    (((1, 1, 2560), (1, 12, 128, 128)), ttnn.bfloat16, "decode_b1"),
    (((1, 16, 2560), (16, 12, 128, 128)), ttnn.bfloat16, "decode_b16"),
    (((1, 96, 2560), (1, 12, 128, 128)), ttnn.bfloat16, "prefill_t96"),
    (((1, 1, 2560), (1, 12, 128, 128)), ttnn.float32, "decode_b1_f32"),
]:
    torch.manual_seed(0)
    t = make_inputs(inputs)
    if dt == ttnn.bfloat16:
        for n in ORDER:
            if n != "recurrent_state":
                t[n] = t[n].to(torch.bfloat16).to(torch.float32)
    exp = reference(*(t[n] for n in ORDER))
    out, cs, rs = gated_delta_net(*_to_device(t, device, dt), compute_kernel_config=compute_config())
    o, s = ttnn.to_torch(out).float(), ttnn.to_torch(rs).float()
    print(
        "ACC",
        name,
        "out PCC %.6f rms %.2e" % _rel_metrics(o, exp[0]),
        "state PCC %.7f rms %.2e" % _rel_metrics(s, exp[2]),
    )
ttnn.close_device(device)
