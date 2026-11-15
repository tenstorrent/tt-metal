import torch, ttnn
from eval.golden_tests.gated_delta_net import helpers as H
from ttnn.operations.gated_delta_net import gated_delta_net_program_descriptor as gdn_pd
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_bf16 import _rel_metrics

device = ttnn.open_device(device_id=0)
try:
    for offload in (True, False):
        gdn_pd.PREP_OFFLOAD = offload
        for inputs, dt, calls, name in [
            (((1, 768, 640), (1, 3, 128, 128)), ttnn.bfloat16, [512, 256], "state_carry_bf16_512+256"),
            (((1, 4096, 640), (1, 3, 128, 128)), ttnn.bfloat16, None, "state_carry_bf16_4096"),
        ]:
            torch.manual_seed(0)
            t = H.make_inputs(inputs, dtype=dt, input_gen="state_carry")
            res = H.run_schedule(
                t, device=device, dtype=dt, calls=calls, state_tolerance=(0.0, 1.0), output_tolerance=(0.0, 1.0)
            )
            for i, r in enumerate(res):
                ps, rss = _rel_metrics(r["recurrent_state_out"].float(), r["expected"][2].float())
                po, ro = _rel_metrics(r["output"].float(), r["expected"][0].float())
                print(
                    f"ACC offload={offload} {name} call{i}: output PCC {po:.7f} relRMS {ro:.2e}; state relRMS {rss:.2e}"
                )
finally:
    ttnn.close_device(device)
