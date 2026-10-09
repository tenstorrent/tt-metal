# In-process alternation 3 (05, 17:40 UTC; the review of 2026-10-09 08:01 UTC): one process runs each case call by call
# under four forms, two with the edit on and two with it off, each pair identical code under two no-op kernel defines (an
# A/A copy on each side). The measurement build adds RC_ALT_<value of RC_ALT> to every kernel's defines
# (tt_metal/impl/kernels/kernel.cpp) and the edits read them: SDPA decode's REDUCE_POW2_SCALER is dropped under
# RC_ALT_OFF1/OFF2, its REDUCE_ROW_BLOCK under RC_ALT_BOFF1/BOFF2, the SCALAR MAX unpack keeps the source clear under
# RC_ALT_SOFF1/SOFF2 (the form of the PR without #58708), and the pool factory keeps the requested fidelity when RC_ALT
# starts with POFF. The program cache is off, so each call builds its program under the current form.
# usage: python -m tracy -r -p --no-web-server -o OUT alt3.py <run> [case,...]
import os, sys
import torch, ttnn
from tracy import signpost

run = sys.argv[1]
want = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else None
dev = ttnn.open_device(device_id=0, l1_small_size=32768)
dev.disable_and_clear_program_cache()
cfgk = lambda fid, fp32: ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32)
SD = ("AA1", "OFF1", "AA2", "OFF2")
BL = ("BA1", "BOFF1", "BA2", "BOFF2")
SC = ("SA1", "SOFF1", "SA2", "SOFF2")
PO = ("PA1", "POFF1", "PA2", "POFF2")

def sdpa(b, nh, nkv, s, d, fid, fp32):
    g = torch.Generator().manual_seed(1024)
    q = ttnn.from_torch(torch.randn((1, b, nh, d), generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    k = ttnn.from_torch(torch.randn((b, nkv, s, d), generator=g), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    v = ttnn.from_torch(torch.randn((b, nkv, s, d), generator=g), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=dev.compute_with_storage_grid_size(), q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False)
    c = cfgk(fid, fp32); pos = [s - 1 - 7 * i for i in range(b)]
    return lambda: ttnn.transformer.scaled_dot_product_attention_decode(q, k, v, cur_pos=pos, is_causal=True, program_config=pc, compute_kernel_config=c)

def scalar(op, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    x = ttnn.from_torch(torch.randn((1, 1, 32, 32), generator=g) * 4, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
    return lambda: getattr(ttnn, op)(x, dim=[-2, -1], keepdim=True)

def pool(n, h, w, c, k):
    g = torch.Generator().manual_seed(59143 + h * 7 + c)
    x = torch.randn((n, c, h, w), generator=g).permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)
    return lambda: ttnn.avg_pool2d(input_tensor=t, batch_size=n, input_h=h, input_w=w, channels=c, kernel_size=[k, k], stride=[k, k],
                                   padding=[0, 0], ceil_mode=False, count_include_pad=True, divisor_override=None, deallocate_input=False,
                                   dtype=ttnn.bfloat16, output_layout=ttnn.TILE_LAYOUT, compute_kernel_config=cc)

CASES = [
    # tt_transformers' SDPA decode at batch 32, 1024 positions, 32/8 heads, head dim 128, chunks auto:
    # accuracy mode (HiFi4, fp32 DEST) toggles the exact fidelity, performance mode (HiFi2, fp32 DEST) the row block
    ("sdpad_acc_define", SD, lambda: sdpa(32, 32, 8, 1024, 128, ttnn.MathFidelity.HiFi4, True)),
    ("sdpad_perf_rowblock", BL, lambda: sdpa(32, 32, 8, 1024, 128, ttnn.MathFidelity.HiFi2, True)),
    ("sdpad_acc_rowblock", BL, lambda: sdpa(32, 32, 8, 1024, 128, ttnn.MathFidelity.HiFi4, True)),
    # the generic reduce's single-core SCALAR path (a one-tile tensor over H and W): the SCALAR MAX unpack of #58708
    ("scmax_bf16", SC, lambda: scalar("max", ttnn.bfloat16, 1)),
    ("scmin_bf16", SC, lambda: scalar("min", ttnn.bfloat16, 2)),
    ("scmax_fp32", SC, lambda: scalar("max", ttnn.float32, 3)),
    ("scmin_fp32", SC, lambda: scalar("min", ttnn.float32, 4)),
    # the average pool's HiFi2 (the build with #59143's unpack): RT-DETR's 2x2 downsamples and Gemma 3's 4x4 projector
    ("pool_rtd160", PO, lambda: pool(1, 160, 160, 256, 2)),
    ("pool_rtd80", PO, lambda: pool(1, 80, 80, 512, 2)),
    ("pool_rtd40", PO, lambda: pool(1, 40, 40, 1024, 2)),
    ("pool_gem64", PO, lambda: pool(1, 64, 64, 1152, 4)),
]
print(f"GRID {dev.compute_with_storage_grid_size().x}x{dev.compute_with_storage_grid_size().y}", flush=True)
K = int(os.environ.get("ALT_K", "10"))
for name, forms, mk in CASES:
    if want and name not in want:
        continue
    try:
        fn = mk()
        for f in forms:  # build and warm each form
            os.environ["RC_ALT"] = f
            fn(); ttnn.synchronize_device(dev)
        for k in range(K):
            for f in (forms if k % 2 == 0 else forms[2:] + forms[:2]):
                os.environ["RC_ALT"] = f
                ttnn.synchronize_device(dev)
                signpost(header=f"{name}|{f}|{k}")
                fn()
        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
        print(f"CASE {name} ok", flush=True)
    except Exception as e:
        print(f"FAILED {name}: {str(e)[:300]}", flush=True)
os.environ.pop("RC_ALT", None)
ttnn.close_device(dev)
