# Separate-process reading (round 3 re-review, 2026-10-09 10:27 UTC): one case under one form per process, with RC_ALT
# fixed for the whole process (the measurement build adds RC_ALT_<value> to every kernel's defines; sdpa_flash_decode.cpp
# drops REDUCE_ROW_BLOCK under RC_ALT_BOFF1/BOFF2 and REDUCE_POW2_SCALER under RC_ALT_OFF1/OFF2). The program cache is
# on (one form per process). Each call after two warm-up calls gets a signpost "case|form|k".
# usage: RC_ALT=<form> python -m tracy -r -p --no-web-server -o OUT sep.py <case> <form>
import os, sys
import torch, ttnn
from tracy import signpost

case, form = sys.argv[1], sys.argv[2]
assert os.environ.get("RC_ALT") == form
dev = ttnn.open_device(device_id=0, l1_small_size=32768)
fid = {"acc": ttnn.MathFidelity.HiFi4, "perf": ttnn.MathFidelity.HiFi2}[case]
b, nh, nkv, s, d = 32, 32, 8, 1024, 128
g = torch.Generator().manual_seed(1024)
q = ttnn.from_torch(torch.randn((1, b, nh, d), generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
k = ttnn.from_torch(torch.randn((b, nkv, s, d), generator=g), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
v = ttnn.from_torch(torch.randn((b, nkv, s, d), generator=g), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=dev.compute_with_storage_grid_size(), q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False)
c = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True)
pos = [s - 1 - 7 * i for i in range(b)]
fn = lambda: ttnn.transformer.scaled_dot_product_attention_decode(q, k, v, cur_pos=pos, is_causal=True, program_config=pc, compute_kernel_config=c)
for _ in range(2):
    fn(); ttnn.synchronize_device(dev)
for i in range(int(os.environ.get("SEP_K", "10"))):
    ttnn.synchronize_device(dev)
    signpost(header=f"{case}|{form}|{i}")
    fn()
ttnn.synchronize_device(dev)
ttnn.ReadDeviceProfiler(dev)
print(f"CASE {case} {form} ok", flush=True)
ttnn.close_device(dev)
