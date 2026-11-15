import sys, torch, ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gdn13/ref.pt"
import os

OUT = os.environ.get("GDN_OUT", "/tmp/gdn13/ref.pt")
device = ttnn.open_device(device_id=0)
cases = {
    "decode_b1_bf16": (((1, 1, 2560), (1, 12, 128, 128)), ttnn.bfloat16),
    "decode_b16_bf16": (((1, 16, 2560), (16, 12, 128, 128)), ttnn.bfloat16),
    "decode_b1_f32": (((1, 1, 2560), (1, 12, 128, 128)), ttnn.float32),
    "prefill_t128": (((1, 128, 2560), (1, 12, 128, 128)), ttnn.bfloat16),
    "prefill_t96": (((1, 96, 2560), (1, 12, 128, 128)), ttnn.bfloat16),
    "batched_b2_t40": (((1, 80, 2560), (2, 12, 128, 128)), ttnn.bfloat16),
    "small_f32": (((1, 64, 384), (1, 4, 32, 32)), ttnn.float32),
}
res = {}
for name, (inputs, dt) in cases.items():
    torch.manual_seed(0)
    tensors = make_inputs(inputs)
    for in_place in (False, True):
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
        kw = dict(conv_state_out=dev[4], recurrent_state_out=dev[5]) if in_place else {}
        outs = gated_delta_net(*dev, compute_kernel_config=compute_config(), **kw)
        res[(name, in_place)] = [ttnn.to_torch(t) for t in outs]
torch.save(res, OUT)
print("SAVED", len(res))
ttnn.close_device(device)
