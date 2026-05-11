"""Bench gate+up fusion for the qwen3_tts MLP decode path.

Compares:
  A. Unfused — two DRAM-sharded matmuls (gate, up) + ttnn.mul[SILU]   (today's path)
  B. Fused   — one DRAM-sharded matmul producing [M, 2N], then split, mul[SILU]

Reports device time per iter for each, plus PCC of the post-mul output vs torch.

Run:
  python models/demos/qwen3_tts/tests/bench_fused_gate_up.py --block code_predictor
  python models/demos/qwen3_tts/tests/bench_fused_gate_up.py --block talker
"""
import argparse
import time

import torch
import torch.nn.functional as F

import ttnn
from models.demos.qwen3_tts.tt.dram_sharded_matmul import (
    build_dram_sharded_weight,
    dram_sharded_program_config,
    find_grid_k_n,
    width_sharded_l1_memcfg,
)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    am, bm = a - a.mean(), b - b.mean()
    return (am @ bm / (am.norm() * bm.norm() + 1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=["talker", "code_predictor"], default="code_predictor")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    if args.block == "talker":
        K, N = 2048, 6144
    else:
        K, N = 1024, 3072

    print(f"[bench] block={args.block} K={K} N={N}, iters={args.iters} (warmup={args.warmup})")

    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
    device.enable_program_cache()

    torch.manual_seed(0)
    # HF-style weights are [out, in]; the model code transposes to [K, N] before sharding.
    gate_w_hf = torch.randn(N, K, dtype=torch.bfloat16) / (K**0.5)
    up_w_hf = torch.randn(N, K, dtype=torch.bfloat16) / (K**0.5)
    x_torch = torch.randn(1, 1, 32, K, dtype=torch.bfloat16)

    # Torch reference: SiLU(gate) * up
    gate_ref = x_torch.float() @ gate_w_hf.t().float()
    up_ref = x_torch.float() @ up_w_hf.t().float()
    ref_post = F.silu(gate_ref) * up_ref  # [1,1,32,N]

    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # ===== UNFUSED PATH =====
    gate_kn = gate_w_hf.transpose(-2, -1).contiguous()  # [K, N]
    up_kn = up_w_hf.transpose(-2, -1).contiguous()
    gate_dram, k_g, n_g = build_dram_sharded_weight(gate_kn, device)
    up_dram, _, n_u = build_dram_sharded_weight(up_kn, device)
    K_t, N_t = k_g // 32, n_g // 32
    rows, cols = find_grid_k_n(K_t, N_t)
    cores = rows * cols
    progcfg_unfused = dram_sharded_program_config(m=32, k=k_g, n=n_g, num_cores=cores)
    in0_memcfg = width_sharded_l1_memcfg(m_tiles=1, k_tiles=K_t, num_cores_x=cols, num_cores_y=rows)
    out_memcfg = width_sharded_l1_memcfg(m_tiles=1, k_tiles=N_t, num_cores_x=cols, num_cores_y=rows)
    print(f"[bench]   unfused: cores={cores} ({rows}x{cols})  K_t={K_t} N_t={N_t}")

    x_tt = ttnn.from_torch(x_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x_sharded = ttnn.to_memory_config(x_tt, in0_memcfg)

    def run_unfused():
        gate_out = ttnn.linear(
            x_sharded, gate_dram, compute_kernel_config=cfg, program_config=progcfg_unfused, memory_config=out_memcfg
        )
        up_out = ttnn.linear(
            x_sharded, up_dram, compute_kernel_config=cfg, program_config=progcfg_unfused, memory_config=out_memcfg
        )
        h = ttnn.mul(gate_out, up_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=out_memcfg)
        ttnn.deallocate(gate_out)
        ttnn.deallocate(up_out)
        return h

    # ===== FUSED PATH =====
    # Fused weight: [K, 2N] = concat(gate_kn, up_kn) along N.
    fused_kn = torch.cat([gate_kn, up_kn], dim=-1).contiguous()  # [K, 2N]
    fused_dram, k_f, n_f = build_dram_sharded_weight(fused_kn, device)
    K_tf, N_tf = k_f // 32, n_f // 32
    rows_f, cols_f = find_grid_k_n(K_tf, N_tf)
    cores_f = rows_f * cols_f
    progcfg_fused = dram_sharded_program_config(m=32, k=k_f, n=n_f, num_cores=cores_f)
    out_memcfg_fused = width_sharded_l1_memcfg(m_tiles=1, k_tiles=N_tf, num_cores_x=cols_f, num_cores_y=rows_f)
    print(f"[bench]   fused:   cores={cores_f} ({rows_f}x{cols_f})  K_t={K_tf} N_t={N_tf} (2N tiles)")

    # Reuse x_sharded from above (same K, same in0_memcfg).
    def run_fused():
        # One big matmul → [1,1,32,2N], width-sharded across cores_f.
        gu = ttnn.linear(
            x_sharded,
            fused_dram,
            compute_kernel_config=cfg,
            program_config=progcfg_fused,
            memory_config=out_memcfg_fused,
        )
        # Move to interleaved L1 so we can slice cleanly along width.
        gu_il = ttnn.to_memory_config(gu, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(gu)
        gate_il = ttnn.slice(gu_il, [0, 0, 0, 0], [1, 1, 32, N], memory_config=ttnn.L1_MEMORY_CONFIG)
        up_il = ttnn.slice(gu_il, [0, 0, 0, N], [1, 1, 32, 2 * N], memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(gu_il)
        # Reshard back to width-sharded for downstream down_proj-compat layout.
        gate_sh = ttnn.to_memory_config(gate_il, out_memcfg)
        up_sh = ttnn.to_memory_config(up_il, out_memcfg)
        ttnn.deallocate(gate_il)
        ttnn.deallocate(up_il)
        h = ttnn.mul(gate_sh, up_sh, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=out_memcfg)
        ttnn.deallocate(gate_sh)
        ttnn.deallocate(up_sh)
        return h

    # ===== PCC =====
    h_unfused = run_unfused()
    h_unfused_torch = ttnn.to_torch(h_unfused).float()
    p_unfused = pcc(h_unfused_torch, ref_post)
    print(f"[bench] PCC(unfused, torch_ref) = {p_unfused:.6f}")
    ttnn.deallocate(h_unfused)

    h_fused = run_fused()
    h_fused_torch = ttnn.to_torch(h_fused).float()
    p_fused = pcc(h_fused_torch, ref_post)
    print(f"[bench] PCC(fused,   torch_ref) = {p_fused:.6f}")
    p_pair = pcc(h_unfused_torch, h_fused_torch)
    print(f"[bench] PCC(unfused, fused)     = {p_pair:.6f}")
    ttnn.deallocate(h_fused)

    # ===== TIMING =====
    def time_path(fn, label, iters, warmup):
        for _ in range(warmup):
            o = fn()
            ttnn.deallocate(o)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        outs = []
        for _ in range(iters):
            outs.append(fn())
        ttnn.synchronize_device(device)
        elapsed = (time.perf_counter() - t0) * 1e6 / iters
        for o in outs:
            ttnn.deallocate(o)
        print(f"[bench] {label}: {elapsed:.1f} µs/iter")
        return elapsed

    t_unf = time_path(run_unfused, "unfused", args.iters, args.warmup)
    t_fus = time_path(run_fused, "fused  ", args.iters, args.warmup)
    print(f"[bench] speedup: {t_unf / t_fus:.2f}x  (saved {t_unf - t_fus:.1f} µs/iter)")

    pcc_pass = p_unfused > 0.99 and p_fused > 0.99 and p_pair > 0.999
    print(f"[bench] {'PASS' if pcc_pass else 'FAIL'}")
    ttnn.close_device(device)
    return 0 if pcc_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
