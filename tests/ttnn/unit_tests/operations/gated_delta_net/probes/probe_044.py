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
    for inputs in [((1, 1, 2560), (1, 12, 128, 128)), ((1, 3, 512), (3, 4, 32, 64))]:
        for dt in [ttnn.bfloat16, ttnn.float32]:
            for seed in [0, 1]:
                torch.manual_seed(seed)
                tensors = make_inputs(inputs)
                if dt == ttnn.bfloat16:
                    for n in ORDER:
                        if n != "recurrent_state":
                            tensors[n] = tensors[n].to(torch.bfloat16).to(torch.float32)
                exp = reference(*(tensors[n] for n in ORDER))
                res = {}
                for raw in (False, True):
                    gdn_pd.QK_RAW_HANDOFF = raw
                    gdn_pd.clear_program_cache()
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
                    o, cs, s = gated_delta_net(*dev, compute_kernel_config=compute_config())
                    o, cs, s = (ttnn.to_torch(t).float() for t in (o, cs, s))
                    res[raw] = (_rel_metrics(o, exp[0]), _rel_metrics(s, exp[2]), torch.equal(cs, exp[1]))
                print(
                    "RES",
                    inputs[1],
                    dt,
                    seed,
                    {
                        k: (
                            f"out pcc {v[0][0]:.7f} rms {v[0][1]:.3e}",
                            f"state pcc {v[1][0]:.8f} rms {v[1][1]:.3e}",
                            v[2],
                        )
                        for k, v in res.items()
                    },
                )
finally:
    ttnn.close_device(device)
