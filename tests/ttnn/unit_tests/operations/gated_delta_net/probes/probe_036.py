import os, torch, ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs

T = int(os.environ["P_T"])
B = int(os.environ["P_B"])
mode = os.environ["P_MODE"]
inputs = ((1, T * B if T > 1 else B, 2560), (B, 12, 128, 128))
device = ttnn.open_device(device_id=0)
up = lambda t, d: ttnn.from_torch(
    t, dtype=d, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
torch.manual_seed(0)
x = make_inputs(inputs)
w = {n: up(x[n], ttnn.float32 if n == "recurrent_state" else ttnn.bfloat16) for n in ORDER}
a = up(torch.randn(512, 2048), ttnn.bfloat16)
b = up(torch.randn(2048, 2048), ttnn.bfloat16)
cs, rs = w["conv_state"], w["recurrent_state"]
for i in range(12):
    args = {**w, "conv_state": cs, "recurrent_state": rs}
    kw = dict(conv_state_out=cs, recurrent_state_out=rs) if mode == "inplace" else {}
    out, cs, rs = gated_delta_net(*(args[n] for n in ORDER), compute_kernel_config=compute_config(), **kw)
    ttnn.experimental.minimal_matmul(a, b)
    ttnn.matmul(a, b)
    ttnn.synchronize_device(device)
    print("call", i, "ok", flush=True)
print("PROBE_OK")
ttnn.close_device(device)
