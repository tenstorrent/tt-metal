# ttnn matmul callers at production-like shapes, each case signposted, warmed up and run N times (device time A/B).
import os, sys
import numpy as np
import torch
import ttnn
from tracy import signpost

if not (os.environ.get("HWLOCK_HELD") or os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    sys.exit("not under hwlock")
N = int(os.environ.get("MM_N", "8"))
ONLY = os.environ.get("MM_ONLY", "")
dev = ttnn.open_device(device_id=0, l1_small_size=32768)
grid = dev.compute_with_storage_grid_size()
D = ttnn.DRAM_MEMORY_CONFIG
F = ttnn.MathFidelity


def ckc(fid, fp32=False, l1acc=True, full=False):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=l1acc, dst_full_sync_en=full)


def tt(shape, dt, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(torch.randn(shape), dtype=dt, layout=layout, device=dev, memory_config=D)


def case(name, f):
    if ONLY and ONLY not in name:
        return
    try:
        for _ in range(2):
            f()
        ttnn.synchronize_device(dev)
        signpost(name)
        for _ in range(N):
            f()
        ttnn.synchronize_device(dev)
        print("case done", name, flush=True)
    except Exception as e:  # keep going so one unsupported case does not lose the others
        print("case FAIL", name, str(e).splitlines()[0][:200] if str(e) else type(e).__name__, flush=True)


def bmm(name, m, k, n, wdt, fid, fp32=False, bias=False, act=None, odt=None, full=False):
    a, w = tt((1, 1, m, k), ttnn.bfloat16), tt((1, 1, k, n), wdt)
    b = tt((1, 1, 1, n), ttnn.bfloat16) if bias else None
    kw = dict(dtype=odt) if odt is not None else {}
    if bias or act:
        f = lambda: ttnn.linear(a, w, bias=b, activation=act, compute_kernel_config=ckc(fid, fp32, full=full), core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x), memory_config=D, **kw)
    else:
        f = lambda: ttnn.matmul(a, w, compute_kernel_config=ckc(fid, fp32, full=full), core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x), memory_config=D, **kw)
    case(name, f)


def mmm(name, m, k, n, wdt, fid, kb=8):
    a, w = tt((m, k), ttnn.bfloat16), tt((k, n), wdt)
    cfg = ttnn.MinimalMatmulConfig(M_block_size=8, K_block_size=kb, N_block_size=8, subblock_h=2, subblock_w=4, compute_with_storage_grid_size=grid)
    case(name, lambda: ttnn.experimental.minimal_matmul(a, w, compute_kernel_config=ckc(fid), config=cfg))


def conv(name, b, cin, cout, h, w, kh, s, p, fid, act_block_h=None):
    if ONLY and ONLY not in name:
        return
    x = ttnn.from_torch(torch.randn(1, 1, b * h * w, cin), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=D)
    wt = ttnn.from_torch(torch.randn(cout, cin, kh, kh), dtype=ttnn.bfloat16)
    cc = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat8_b)
    if act_block_h:
        cc.act_block_h_override = act_block_h
    kw = dict(in_channels=cin, out_channels=cout, batch_size=b, input_height=h, input_width=w, kernel_size=(kh, kh), stride=(s, s),
              padding=(p, p), dilation=(1, 1), groups=1, device=dev, conv_config=cc, compute_config=ckc(fid, l1acc=True), dtype=ttnn.bfloat16)
    state = {"w": wt}
    def f():
        o, [state["w"], _] = ttnn.conv2d(input_tensor=x, weight_tensor=state["w"], bias_tensor=None, return_weights_and_bias=True, **kw)
        ttnn.deallocate(o)
    case(name, f)


def prefill(name, b, nh, nkv, s, d, qc, kc, dt, fid, fp32):
    if ONLY and ONLY not in name:
        return
    q, k, v = tt((b, nh, s, d), dt), tt((b, nkv, s, d), dt), tt((b, nkv, s, d), dt)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False)
    case(name, lambda: ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, program_config=pc, compute_kernel_config=ckc(fid, fp32, False)))


def decode(name, b, s, nh=8, nkv=1, d=128):
    if ONLY and ONLY not in name:
        return
    q = ttnn.from_torch(torch.randn(1, b, nh, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)
    k, v = tt((b, nkv, s, d), ttnn.bfloat8_b), tt((b, nkv, s, d), ttnn.bfloat8_b)
    pos = np.linspace(0, s - 1, b, dtype=np.int32).tolist()
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=256, exp_approx_mode=False)
    case(name, lambda: ttnn.transformer.scaled_dot_product_attention_decode(
        q, k, v, cur_pos=pos, scale=d ** -0.5, program_config=pc, compute_kernel_config=ckc(F.HiFi2, False, False), memory_config=D))


bmm("b_2k5k5k_bfp8_hifi2", 2048, 5120, 5120, ttnn.bfloat8_b, F.HiFi2)
bmm("b_4k4k4k_bf16_hifi2", 4096, 4096, 4096, ttnn.bfloat16, F.HiFi2)
bmm("b_4k1k4k_bfp8_hifi4", 4096, 1024, 4096, ttnn.bfloat8_b, F.HiFi4)
bmm("b_2k2k8k_bfp8_lofi", 2048, 2048, 8192, ttnn.bfloat8_b, F.LoFi)
bmm("b_2k4k14k_bfp4_lofi", 2048, 4096, 14336, ttnn.bfloat4_b, F.LoFi)
bmm("b_2k5k5k_bfp8_lofi", 2048, 5120, 5120, ttnn.bfloat8_b, F.LoFi)
bmm("b_dec_32x4kx14k_bfp8_hifi2", 32, 4096, 14336, ttnn.bfloat8_b, F.HiFi2)
bmm("b_fused_2k4k4k_bias_gelu", 2048, 4096, 4096, ttnn.bfloat8_b, F.HiFi2, bias=True, act="gelu")
bmm("b_8k1k1k_bf16_hifi4_fp32", 8192, 1024, 1024, ttnn.bfloat16, F.HiFi4, fp32=True)
bmm("b_2k4k4k_bf16_hifi2_outbfp8", 2048, 4096, 4096, ttnn.bfloat16, F.HiFi2, odt=ttnn.bfloat8_b)
bmm("b_2k2k8k_bfp8_lofi_outbfp8", 2048, 2048, 8192, ttnn.bfloat8_b, F.LoFi, odt=ttnn.bfloat8_b)
bmm("b_dec_32x4kx14k_bfp8_hifi2_outbfp8", 32, 4096, 14336, ttnn.bfloat8_b, F.HiFi2, odt=ttnn.bfloat8_b)
bmm("b_2k4k4k_bf16_hifi2_full", 2048, 4096, 4096, ttnn.bfloat16, F.HiFi2, full=True)
bmm("b_2k2k8k_bfp8_lofi_full", 2048, 2048, 8192, ttnn.bfloat8_b, F.LoFi, full=True)
mmm("mm_2k4k14k_bfp8_hifi2", 2048, 4096, 14336, ttnn.bfloat8_b, F.HiFi2)
mmm("mm_2k4k4k_bf16_hifi2", 2048, 4096, 4096, ttnn.bfloat16, F.HiFi2)
mmm("mm_2k4k4k_bfp8_lofi_kb1", 2048, 4096, 4096, ttnn.bfloat8_b, F.LoFi, kb=1)
conv("c_rn50_l2_3x3", 8, 128, 128, 28, 28, 3, 1, 1, F.LoFi)
conv("c_rn50_l3_1x1", 8, 256, 1024, 14, 14, 1, 1, 0, F.LoFi)
prefill("p_l70b_s8k", 1, 8, 1, 8192, 128, 256, 256, ttnn.bfloat16, F.HiFi2, False)
prefill("p_gqa_s4k", 1, 32, 8, 4096, 128, 256, 512, ttnn.bfloat16, F.HiFi2, False)
prefill("p_fp32acc_s2k", 1, 8, 1, 2048, 128, 128, 256, ttnn.bfloat16, F.HiFi4, True)
decode("d_b32_s2k", 32, 2048)
decode("d_b8_s8k", 8, 8192)
for dtn, dt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b), ("fp32", ttnn.float32)):
    sa = tt((1, 8, 1024, 1024), dt)
    case(f"o_softmax_{dtn}", lambda: ttnn.softmax(sa, dim=-1))
    la, lw = tt((1, 1, 4096, 4096), dt), tt((1, 1, 32, 4096), ttnn.bfloat16)
    case(f"o_layer_norm_{dtn}", lambda: ttnn.layer_norm(la, weight=lw))
ttnn.close_device(dev)
print("ALL DONE", flush=True)
