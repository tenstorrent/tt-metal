import torch, ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from ttnn.operations.gated_delta_net import gated_delta_net_program_descriptor as gdn_pd
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs

orig = gdn_pd.check_l1_footprint


def spy(role, footprint, budget):
    print("FOOTPRINT", role, footprint, budget)
    return orig(role, footprint, budget)


gdn_pd.check_l1_footprint = spy
device = ttnn.open_device(device_id=0)
try:
    for dt in (ttnn.bfloat16, ttnn.float32):
        gdn_pd.clear_program_cache()
        tensors = make_inputs(((1, 1, 2560), (1, 12, 128, 128)))
        dev = [
            ttnn.from_torch(
                tensors[n],
                dtype=ttnn.float32 if n == "recurrent_state" else dt,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for n in ORDER
        ]
        print("DTYPE", dt)
        gated_delta_net(*dev, compute_kernel_config=compute_config())
finally:
    ttnn.close_device(device)
