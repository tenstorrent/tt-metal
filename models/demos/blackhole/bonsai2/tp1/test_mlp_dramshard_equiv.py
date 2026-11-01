# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_1C_MLP_DRAMSHARD equivalence on one gate/up matmul: the DRAM-sharded config against the 1D path it
replaces (ttnn.linear auto config, the decode compute config), same bfp4 weight, same x. Reports bit-exact
fraction and max |diff| per variant, so a change in agreement can be traced to a specific setting.

    pytest models/demos/blackhole/bonsai2/tp1/test_mlp_dramshard_equiv.py -s
"""
import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

K, N = 5120, 17408


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_mlp_dramshard_equiv(mesh_device):
    torch.manual_seed(0)
    x_t = torch.randn(1, 1, 1, K)
    w_t = torch.randn(K, N) * 0.02

    def ckc(l1_acc=True, fid=ttnn.MathFidelity.LoFi):
        return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, fp32_dest_acc_en=True, packer_l1_acc=l1_acc)

    w_il = ttnn.from_torch(w_t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    w_ds = ttnn.to_memory_config(w_il, tpc.create_dram_sharded_mem_config(K, N))
    act_cfg = tpc.create_activation_shard_config(K)
    x_il = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                           memory_config=ttnn.L1_MEMORY_CONFIG)
    x_sh = ttnn.to_memory_config(x_il, act_cfg)
    pc = tpc.create_dram_sharded_matmul_program_config(1, K, N)

    def old(act):
        y = ttnn.linear(x_il, w_il, activation=act, compute_kernel_config=ckc(), memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.to_torch(y).float().reshape(-1)[:N], y.dtype

    def new(act, workers=2, l1_acc=True, kb=None):
        cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=kb or pc.in0_block_w,
            per_core_M=pc.per_core_M,
            per_core_N=pc.per_core_N,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if act else None,
            num_workers_per_dram_bank=workers,
        )
        y = ttnn.linear(x_sh, w_ds, compute_kernel_config=ckc(l1_acc), program_config=cfg,
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
        return ttnn.to_torch(ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)).float().reshape(-1)[:N], y.dtype

    ref = (x_t.reshape(1, K).double() @ ttnn.to_torch(w_il).double()).reshape(-1)
    for act in (None, "silu"):
        o, odt = old(act)
        r = torch.nn.functional.silu(ref) if act else ref
        print(f"\n== act={act}  old dtype {odt}  old vs float64: max|d| {(o - r).abs().max():.3e}")
        for label, kw in [("w2 l1acc", {}), ("w1 l1acc", {"workers": 1}), ("w2 no-l1acc", {"l1_acc": False}),
                          ("w2 kb1", {"kb": 1})]:
            n, ndt = new(act, **kw)
            exact = (n == o).float().mean().item()
            print(f"  {label:<12} dtype {ndt}  bit-exact vs old {100 * exact:6.2f}%  max|d| vs old {(n - o).abs().max():.3e}"
                  f"  vs float64 {(n - r).abs().max():.3e}")
        if act:
            # SiLU outside the matmul, on the bf16 matmul output: standalone, and fused into the gate*up mul.
            cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=pc.in0_block_w, per_core_M=pc.per_core_M, per_core_N=pc.per_core_N,
                fused_activation=None, num_workers_per_dram_bank=2)
            y = ttnn.linear(x_sh, w_ds, compute_kernel_config=ckc(), program_config=cfg,
                            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
            ones = ttnn.ones_like(y)
            for label, z in [("silu op", ttnn.silu(y, memory_config=y.memory_config())),
                             ("mul a-act", ttnn.mul(y, ones, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                                                    memory_config=y.memory_config()))]:
                n = ttnn.to_torch(ttnn.sharded_to_interleaved(z, ttnn.L1_MEMORY_CONFIG)).float().reshape(-1)[:N]
                print(f"  {label:<12} bit-exact vs old {100 * (n == o).float().mean().item():6.2f}%"
                      f"  max|d| vs old {(n - o).abs().max():.3e}  vs float64 {(n - r).abs().max():.3e}")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("wdtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_mlp_dramshard_down_equiv(mesh_device, wdtype):
    """down_proj: DRAM-sharded (in0 = the 32-core x 17-tile width shard of SiLU(gate)*up) vs _down_decode_progcfg."""
    from models.demos.blackhole.qwen36.tt.mlp import _down_decode_progcfg

    torch.manual_seed(0)
    Kd, Nd = N, K
    h_t = torch.randn(1, 1, 1, Kd)
    w_t = torch.randn(Kd, Nd) * 0.02
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)
    w_il = ttnn.from_torch(w_t, dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    w_ds = ttnn.to_memory_config(w_il, tpc.create_dram_sharded_mem_config(Kd, Nd))
    h_il = ttnn.from_torch(h_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                           memory_config=ttnn.L1_MEMORY_CONFIG)
    h_sh = ttnn.to_memory_config(h_il, tpc.create_activation_shard_config(Kd))
    o = ttnn.to_torch(ttnn.linear(h_il, w_il, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG,
                                  program_config=_down_decode_progcfg(Nd))).float().reshape(-1)[:Nd]
    pc = tpc.create_dram_sharded_matmul_program_config(1, Kd, Nd)
    for workers in (1, 2):
        cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=(Kd // 32) // 32, per_core_M=pc.per_core_M, per_core_N=pc.per_core_N,
            fused_activation=None, num_workers_per_dram_bank=workers)
        y = ttnn.linear(h_sh, w_ds, compute_kernel_config=ckc, program_config=cfg,
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
        n = ttnn.to_torch(ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)).float().reshape(-1)[:Nd]
        print(f"down {wdtype} w{workers}: bit-exact vs 1D {100 * (n == o).float().mean().item():6.2f}%  max|d| {(n - o).abs().max():.3e}")
