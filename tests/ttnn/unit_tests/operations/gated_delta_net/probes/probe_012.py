import torch, ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    compute_config,
    make_inputs,
    reference,
)
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_bf16 import _rel_metrics

device = ttnn.open_device(device_id=0)
for B in (8, 16):
    torch.manual_seed(0)
    t = make_inputs(((1, B, 2560), (B, 12, 128, 128)))
    for n in ORDER:
        if n != "recurrent_state":
            t[n] = t[n].to(torch.bfloat16).to(torch.float32)
    exp = reference(*(t[n] for n in ORDER))
    dev = [
        ttnn.from_torch(
            t[n],
            dtype=ttnn.float32 if n == "recurrent_state" else ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        for n in ORDER
    ]
    out, cs, rs = gated_delta_net(*dev, compute_kernel_config=compute_config())
    o, c, r = (ttnn.to_torch(x).float() for x in (out, cs, rs))
    po, ro = _rel_metrics(o, exp[0])
    ps, rss = _rel_metrics(r, exp[2])
    print(f"B={B} out PCC {po:.7f} rms {ro:.3e} state PCC {ps:.7f} rms {rss:.3e} cs_exact {torch.equal(c, exp[1])}")
ttnn.close_device(device)
