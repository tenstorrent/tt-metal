import torch, ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from eval.golden_tests.gated_delta_net import helpers as H
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import compute_config
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_bf16 import _rel_metrics

device = ttnn.open_device(device_id=0)
try:
    for name, inputs, gen, dt in [
        ("model_prefill_4096", ((1, 4096, 2560), (1, 12, 128, 128)), None, ttnn.bfloat16),
        ("state_carry_one_call_f32", ((1, 4096, 640), (1, 3, 128, 128)), "state_carry", ttnn.float32),
        ("chip_t70_f32", ((1, 70, 2560), (1, 12, 128, 128)), None, ttnn.float32),
    ]:
        torch.manual_seed(0)
        t = H.make_inputs(inputs, dtype=dt, input_gen=gen)
        exp = H.pytorch_gated_delta_net(*(t[n] for n in H.TENSOR_NAMES))
        dev = H.input_tensors_on_device(t, device, dtype=dt)
        out, cs, rs = gated_delta_net(*dev, compute_kernel_config=compute_config())
        o, c, r = (ttnn.to_torch(x).float() for x in (out, cs, rs))
        po, ro = _rel_metrics(o, exp[0].float())
        ps, rs_ = _rel_metrics(r, exp[2].float())
        print(
            "RES",
            name,
            "out PCC %.6f relRMS %.2e | state PCC %.7f relRMS %.2e | conv exact %s"
            % (po, ro, ps, rs_, torch.equal(c, exp[1].float())),
        )
finally:
    ttnn.close_device(device)
