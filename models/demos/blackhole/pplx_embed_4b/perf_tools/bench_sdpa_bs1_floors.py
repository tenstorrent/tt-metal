# bs1 SDPA at the model's call (profile attributes at 3164ba1): Q [1, 32, 512, 128], K / V [1, 8, 512, 128] bfp8 in L1
# interleaved, pack_gqa_heads, output_heads_concat, q192 / k512 on 11x8 (88 cores), LoFi, exp approx, scale 1/sqrt(128).
# Run once per kernel tree of sdpa_kernel_variants.py (base / conly / dmonly / dmread / dmwrite / zones ...):
#   cd <dir without a ttnn/ tree>; TT_METAL_KERNEL_PATH=<out>/<variant> TT_METAL_CACHE=<out>/cache_<variant> \
#     TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=<prof> python bench_sdpa_bs1_floors.py
#   device_kernel_us.py <prof>
# Prints the output's PCC against torch (a patched variant shows ~0: proof the patch compiled in) and wall us per call.
# SDPA_GRID=x,y SDPA_Q=<q chunk> SDPA_K=<k chunk> override the config.
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common_traced import make_traced  # noqa: E402


def main():
    gx, gy = (int(v) for v in os.getenv("SDPA_GRID", "11,8").split(","))
    qc, kc = int(os.getenv("SDPA_Q", "192")), int(os.getenv("SDPA_K", "512"))
    D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 << 20)
    traced = make_traced(D)
    L1, B8 = ttnn.L1_MEMORY_CONFIG, ttnn.bfloat8_b
    try:
        torch.manual_seed(0)
        qh, kh, vh = torch.randn(1, 32, 512, 128), torch.randn(1, 8, 512, 128), torch.randn(1, 8, 512, 128)
        q, k, v = (
            ttnn.from_torch(t, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1) for t in (qh, kh, vh)
        )
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            q_chunk_size=qc,
            k_chunk_size=kc,
            exp_approx_mode=True,
        )
        ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        fn = lambda: ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=128**-0.5,
            program_config=pc,
            compute_kernel_config=ck,
            memory_config=L1,
            pack_gqa_heads=True,
            output_heads_concat=True,
        )
        o = ttnn.to_torch(fn()).float().reshape(1, 512, 32, 128).transpose(1, 2)
        qq, kk, vv = (ttnn.to_torch(t).float() for t in (q, k, v))
        kk, vv = kk.repeat_interleave(4, dim=1), vv.repeat_interleave(4, dim=1)
        gold = torch.softmax(qq @ kk.transpose(-1, -2) * 128**-0.5, dim=-1) @ vv
        pcc = torch.corrcoef(torch.stack([o.flatten(), gold.flatten()]))[0, 1].item()
        print(f"RES sdpa bs1 {gx}x{gy} q{qc}/k{kc} pcc {pcc:.5f} wall {traced(fn):.1f} us", flush=True)
    finally:
        ttnn.close_device(D)


if __name__ == "__main__":
    main()
