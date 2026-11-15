import torch, ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from ttnn.operations.gated_delta_net import gated_delta_net_program_descriptor as gdn_pd
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    compute_config,
    make_inputs,
    reference,
)
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_bf16 import _rel_metrics

device = ttnn.open_device(device_id=0)
try:
    for inputs, dt, name in [
        (((1, 4096, 2560), (1, 12, 128, 128)), ttnn.bfloat16, "model_prefill_4096"),
        (((1, 4096, 640), (1, 3, 128, 128)), ttnn.bfloat16, "one_call_4096_hk1_bf16"),
        (((1, 2048, 256), (1, 3, 32, 64)), ttnn.float32, "small_f32_t2048"),
    ]:
        torch.manual_seed(0)
        t = make_inputs(inputs)
        if dt == ttnn.bfloat16:
            for n in ORDER:
                if n != "recurrent_state":
                    t[n] = t[n].to(torch.bfloat16).to(torch.float32)
        exp = reference(*(t[n] for n in ORDER))
        dev = [
            ttnn.from_torch(
                t[n], dtype=ttnn.float32 if n == "recurrent_state" else dt, layout=ttnn.TILE_LAYOUT, device=device
            )
            for n in ORDER
        ]
        out, cs, rs = gated_delta_net(*dev, compute_kernel_config=compute_config())
        o, c, r = (ttnn.to_torch(x).float() for x in (out, cs, rs))
        po, ro = _rel_metrics(o, exp[0])
        ps, rss = _rel_metrics(r, exp[2])
        print(
            f"ACC {name}: output PCC {po:.7f} relRMS {ro:.2e}; state PCC {ps:.7f} relRMS {rss:.2e}; conv_state exact {torch.equal(c, exp[1])}"
        )
finally:
    ttnn.close_device(device)
