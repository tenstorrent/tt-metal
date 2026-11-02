# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""M1 conv_silu against the model's own decode conv chain (ttnn mul + K-1 mac + silu, bf16), whole row,
real-scale random inputs. Reports bit-exact fraction and max|diff| for several compute configs, and each
against a float64 reference.

    pytest models/demos/blackhole/bonsai2/tp1/test_m1_vs_chain.py -s
"""
import importlib.util
import os

import pytest
import torch

import ttnn

_spec = importlib.util.spec_from_file_location(
    "conv_silu_op", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops", "conv_silu", "op.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

KEY_DIM, VALUE_DIM, NV, GQA, K = 2048, 6144, 48, 3, 4
C = 2 * KEY_DIM + VALUE_DIM


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_m1_vs_chain(mesh_device):
    torch.manual_seed(0)
    bf = torch.bfloat16
    qkv = torch.randn(1, 1, C).to(bf)
    state = torch.randn(K, 1, C).to(bf)
    taps = (torch.randn(K, 1, C) * 0.3).to(bf)

    def dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                               memory_config=ttnn.L1_MEMORY_CONFIG)

    # The model's chain on the row form (CONV_HM is bit-identical to it): taps[j] * state[j] summed, silu.
    st = [dev(state[j : j + 1]) for j in range(K)]
    tp = [dev(taps[j : j + 1]) for j in range(K)]
    acc = ttnn.multiply(st[0], tp[0])
    for j in range(1, K):
        acc = ttnn.mac(st[j], tp[j], acc)
    chain = ttnn.to_torch(ttnn.silu(acc)).float().reshape(-1)
    ref = torch.nn.functional.silu((state.double() * taps.double()).sum(0)).reshape(-1)
    print(f"\\nchain vs float64: max|d| {(chain - ref).abs().max():.3e}")

    diag = dev(_mod.tap_diagonals(taps))
    cfgs = {
        "fp32acc (as built)": ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True),
        "fp32acc HiFi4": ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, math_fidelity=ttnn.MathFidelity.HiFi4),
        "bf16 dst HiFi4": ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, math_fidelity=ttnn.MathFidelity.HiFi4),
        "bf16 dst LoFi": ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, math_fidelity=ttnn.MathFidelity.LoFi),
    }
    cfgs = {f"{n} round={r}": (c, r) for n, c in cfgs.items() if "bf16 dst LoFi" not in n for r in (False, True)}
    for name, (cfg, rnd) in cfgs.items():
        conv, shift = _mod.conv_silu(dev(qkv), dev(state), diag, KEY_DIM, VALUE_DIM, NV, GQA, compute_config=cfg,
                                     chain_round=rnd)
        m1 = ttnn.to_torch(conv).float().reshape(-1)
        sh = ttnn.to_torch(shift).float()
        want_sh = torch.cat([state[1:], qkv], 0).float()
        print(f"{name:<28} vs chain: bit-exact {100 * (m1 == chain).float().mean():6.2f}%  max|d| {(m1 - chain).abs().max():.3e}"
              f"  | vs float64 {(m1 - ref).abs().max():.3e}  | shift exact {bool((sh == want_sh).all())}")
