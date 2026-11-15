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
for inputs in (((1, 8, 2560), (8, 12, 128, 128)), ((1, 16, 2560), (16, 12, 128, 128))):
    torch.manual_seed(0)
    t = make_inputs(inputs)
    for n in ORDER:
        if n != "recurrent_state":
            t[n] = t[n].to(torch.bfloat16).to(torch.float32)
    exp = reference(*(t[n] for n in ORDER))
    outs = gated_delta_net(*_to_device(t, device, ttnn.bfloat16), compute_kernel_config=compute_config())
    o, cs, rs = (ttnn.to_torch(x).to(torch.float32) for x in outs)
    po, ro = _rel_metrics(o, exp[0])
    ps, rss = _rel_metrics(rs, exp[2])
    print(
        f"METRIC B={inputs[1][0]} out PCC {po:.7f} relRMS {ro:.2e} state PCC {ps:.7f} relRMS {rss:.2e} cs_exact {torch.equal(cs, exp[1])}"
    )
ttnn.close_device(device)
