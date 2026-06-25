# SPDX-License-Identifier: Apache-2.0
"""PCC check: padded-weight matmul_decode MLP vs reference ttnn.linear.

gate/up N padded 4096→6144, down K padded 4096→6144. Zero-padding means
extra terms contribute 0 to the matmul sum → PCC should be ~1.0.

Run:
  python _bench_mlp_pad_pcc.py --device-id 23
"""
import argparse

import torch
import ttnn

MLP_K = 1024  # hidden
MLP_N_ORIG = 4096  # intermediate (original)
MLP_N_PAD = 6144  # intermediate (padded)
M = 32  # action horizon padded


def pearson_cc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=23)
    args = ap.parse_args()

    dev = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=24576, trace_region_size=134_217_728)
    grid = dev.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    maxc = gx * gy

    torch.manual_seed(42)

    # Reference weights (PyTorch, fp32)
    gate_w = torch.randn(MLP_K, MLP_N_ORIG) * 0.02
    up_w = torch.randn(MLP_K, MLP_N_ORIG) * 0.02
    down_w = torch.randn(MLP_N_ORIG, MLP_K) * 0.02  # [K_down=4096, N_down=1024]
    x = torch.randn(M, MLP_K) * 0.1

    # --- Reference: ttnn.linear (DRAM weights, unpadded) ---
    def mk_dram(t):
        return ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)

    g_ref = mk_dram(gate_w)
    u_ref = mk_dram(up_w)
    d_ref = mk_dram(down_w)
    x_tt = ttnn.from_torch(x.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    gate_out = ttnn.linear(
        x_tt, g_ref, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=ckc
    )
    gate_out = ttnn.gelu(gate_out, fast_and_approximate_mode=True, memory_config=ttnn.L1_MEMORY_CONFIG)
    up_out = ttnn.linear(
        x_tt, u_ref, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=ckc
    )
    h_ref = ttnn.multiply(gate_out, up_out, memory_config=ttnn.L1_MEMORY_CONFIG)
    out_ref_tt = ttnn.linear(
        h_ref, d_ref, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=ckc
    )
    out_ref = ttnn.to_torch(out_ref_tt).float()
    for t in [g_ref, u_ref, d_ref, gate_out, up_out, h_ref, out_ref_tt]:
        ttnn.deallocate(t)

    # --- Padded matmul_decode: gate/up N→6144, down K→6144 ---
    def pad_n(w, target_n):
        p = torch.zeros(w.shape[0], target_n - w.shape[1])
        return torch.cat([w, p], dim=-1)

    def pad_k(w, target_k):
        p = torch.zeros(target_k - w.shape[0], w.shape[1])
        return torch.cat([w, p], dim=0)

    gate_p = pad_n(gate_w, MLP_N_PAD)  # [1024, 6144]
    up_p = pad_n(up_w, MLP_N_PAD)  # [1024, 6144]
    down_p = pad_k(down_w, MLP_N_PAD)  # [6144, 1024]

    def n_cores(n):
        c = n // 32
        while c > maxc:
            c //= 2
        return c

    def b_wsh(w_t, nc):
        K, N = w_t.shape
        cfg = ttnn.create_sharded_memory_config(
            (K, N // nc),
            core_grid=ttnn.num_cores_to_corerangeset(nc, grid, True),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.from_torch(
            w_t.bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=cfg
        )

    def a_wsh(t, K):
        cfg = ttnn.create_sharded_memory_config(
            (M, K // 2),
            core_grid=ttnn.num_cores_to_corerangeset(2, grid, True),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.to_memory_config(t, cfg)

    nc_gu = n_cores(MLP_N_PAD)  # 96
    nc_dn = n_cores(MLP_K)  # 32  (N_down=1024)
    # K_blocks=1 for all: B shard has full K height → use full_width_sharded (partial=False)
    partial_gu = False
    partial_dn = False

    g_pad = b_wsh(gate_p, nc_gu)
    u_pad = b_wsh(up_p, nc_gu)
    d_pad = b_wsh(down_p, nc_dn)

    x_il = ttnn.from_torch(
        x.bfloat16(), layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile((32, 32)), device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    _a = a_wsh(x_il, MLP_K)
    gate_md = ttnn.matmul_decode(_a, g_pad, partial_width_sharded=partial_gu)
    gate_md = ttnn.gelu(gate_md, fast_and_approximate_mode=True, memory_config=gate_md.memory_config())
    up_md = ttnn.matmul_decode(_a, u_pad, partial_width_sharded=partial_gu)
    ttnn.deallocate(_a)
    h_md = ttnn.multiply(gate_md, up_md, memory_config=gate_md.memory_config())
    ttnn.deallocate(gate_md)
    ttnn.deallocate(up_md)
    h_il = ttnn.sharded_to_interleaved(h_md, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(h_md)
    _a_dn = a_wsh(h_il, MLP_N_PAD)  # [M, 6144] sharded 2c
    ttnn.deallocate(h_il)
    out_md_tt = ttnn.matmul_decode(_a_dn, d_pad, partial_width_sharded=partial_dn)
    ttnn.deallocate(_a_dn)
    out_il = ttnn.sharded_to_interleaved(out_md_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(out_md_tt)
    out_pad = ttnn.to_torch(out_il).float()
    for t in [g_pad, u_pad, d_pad, x_il, out_il]:
        ttnn.deallocate(t)

    pcc = pearson_cc(out_ref, out_pad).item()
    max_err = (out_ref - out_pad).abs().max().item()
    print(f"\ngate/up N {MLP_N_ORIG}→{MLP_N_PAD}, down K {MLP_N_ORIG}→{MLP_N_PAD}")
    print(f"nc_gate/up={nc_gu} partial={partial_gu}  nc_down={nc_dn} partial={partial_dn}")
    print(f"PCC  = {pcc:.6f}  (target ≥ 0.999)")
    print(f"MaxE = {max_err:.4f}")
    assert pcc >= 0.998, f"PCC {pcc:.4f} below threshold"
    print("PASS")

    ttnn.CloseDevice(dev)


if __name__ == "__main__":
    main()
