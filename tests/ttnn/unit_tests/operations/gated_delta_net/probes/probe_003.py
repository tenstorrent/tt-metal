import sys, torch, ttnn
import ttnn.operations.gated_delta_net

m = sys.modules["ttnn.operations.gated_delta_net.gated_delta_net"]
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    make_inputs,
    reference,
    to_device,
    compute_config,
    check_call,
)

m.SUPPORTED["sequence_mode"].append("batched_prefill")
device = ttnn.open_device(device_id=0)
try:
    for inputs in [
        ((1, 10, 256), (2, 3, 32, 64)),
        ((1, 96, 256), (3, 3, 32, 64)),
        ((1, 66, 256), (2, 3, 32, 64)),
        ((1, 64, 2560), (2, 12, 128, 128)),
    ]:
        torch.manual_seed(1)
        t = make_inputs(inputs)
        exp = reference(*(t[n] for n in ORDER))
        for poison in (None, float("inf")):
            try:
                check_call(
                    m.gated_delta_net(*to_device(t, device, poison=poison), compute_kernel_config=compute_config()),
                    exp,
                    inputs,
                )
                print("CASE", inputs, poison, "PASS")
            except AssertionError as e:
                print("CASE", inputs, poison, "FAIL", str(e)[:200])
finally:
    ttnn.close_device(device)
