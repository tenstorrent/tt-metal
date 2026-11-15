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
try:
    for name, inputs, dt in [
        ("b1", ((1, 1, 2560), (1, 12, 128, 128)), ttnn.bfloat16),
        ("b8", ((1, 8, 2560), (8, 12, 128, 128)), ttnn.bfloat16),
        ("b16", ((1, 16, 2560), (16, 12, 128, 128)), ttnn.bfloat16),
        ("b1_f32", ((1, 1, 2560), (1, 12, 128, 128)), ttnn.float32),
    ]:
        torch.manual_seed(0)
        t = make_inputs(inputs)
        if dt == ttnn.bfloat16:
            for n in ORDER:
                if n != "recurrent_state":
                    t[n] = t[n].to(torch.bfloat16).to(torch.float32)
        exp = reference(*(t[n] for n in ORDER))
        out, cs, rs = gated_delta_net(*_to_device(t, device, dt), compute_kernel_config=compute_config())
        o, c, r = (ttnn.to_torch(x).to(torch.float32) for x in (out, cs, rs))
        po, ro_ = _rel_metrics(o, exp[0])
        ps, rs_ = _rel_metrics(r, exp[2])
        print(
            f"ACC {name} out PCC {po:.7f} relRMS {ro_:.2e} state PCC {ps:.7f} relRMS {rs_:.2e} cs_exact {torch.equal(c, exp[1])}"
        )
finally:
    ttnn.close_device(device)
