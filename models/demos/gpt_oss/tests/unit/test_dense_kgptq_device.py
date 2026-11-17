# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The batch-32 dense MoE experts in K-GPTQ (experts/weights.py dense_kgptq_dir): bfloat4_b weights whose 16-value
exponent groups run along K are held exactly only when every tile is stored transposed within itself
(ttnn.Tile transpose_tile), and the batched decode's two DRAM-sharded matmuls (experts/decode.py DenseDecodePlan, the
program configs decode_forward_dense runs) read such tiles correctly.

- packing: K-grouped bfloat4_b values survive a transposed-tile pack unchanged; the normal pack re-rounds many.
- gate (x [32, 2880] @ [2880, 32 * 736]) and down (act [32, 32 * 736] @ [32 * 736, 2880]) with transposed tiles
  against torch on the stored values: PCC >= 0.999.

Run on Blackhole (one chip is enough):
    pytest models/demos/gpt_oss/tests/unit/test_dense_kgptq_device.py
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.gpt_oss.tt.experts.decode import DenseDecodePlan
from models.demos.gpt_oss.tt.experts.dense_layout import dram_shard_tiles
from models.demos.gpt_oss.tt.experts.weights import _dram_width_sharded
from models.demos.gpt_oss.tt.fused_decode.config import DENSE_MOE_WORKERS_PER_BANK

H, IL_PAD, E, TOKENS = 2880, 736, 32, 32
TRANSPOSED = ttnn.Tile((32, 32), transpose_tile=True)

pytestmark = pytest.mark.skipif(os.environ.get("CI") == "true", reason="Blackhole only; run locally.")


def k_grouped(w):
    """w [K, N] rounded on the bfloat4_b grid whose exponent groups run along K (the K-GPTQ grid): one transposed-tile
    pack and unpack on host."""
    t = ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, tile=TRANSPOSED)
    return ttnn.to_torch(t).float()


def test_transposed_tiles_hold_k_grouped_bfp4():
    torch.manual_seed(0)
    w = k_grouped(torch.randn(256, 256) * 0.02)
    again = ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, tile=TRANSPOSED)).float()
    normal = ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
    assert torch.equal(again, w), "a transposed-tile pack re-rounded K-grouped values"
    changed = (normal != w).float().mean().item()
    assert changed > 0.05, f"the normal pack changed only {changed:.1%} of K-grouped values (expected many)"


def _weight(device, w, k, n):
    shard = dram_shard_tiles(n // 32, device.dram_grid_size().x, DENSE_MOE_WORKERS_PER_BANK)
    return ttnn.from_torch(
        w,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        tile=TRANSPOSED,
        device=device,
        memory_config=_dram_width_sharded(device, k, n, shard),
    )


@pytest.mark.skipif(not is_blackhole(), reason="the batch-32 dense MoE runs on Blackhole")
def test_dense_matmuls_with_transposed_tiles(device):
    torch.manual_seed(0)
    plan = DenseDecodePlan(device, H, IL_PAD, E, TOKENS)
    n_act = E * IL_PAD

    # gate / up: x [32, H] (bf8, L1 width-sharded) @ [H, E * 736]
    x = torch.randn(TOKENS, H)
    w = k_grouped(torch.randn(H, n_act) * 0.02)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=plan.input_memory_config
    )
    w_tt = _weight(device, w, H, n_act)
    out = ttnn.matmul(
        x_tt,
        w_tt,
        program_config=plan.gate_up_program_config,
        memory_config=plan.act_memory_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=plan.compute_kernel_config,
    )
    ref = ttnn.to_torch(x_tt).double() @ w.double()
    ok, pcc = comp_pcc(ref, ttnn.to_torch(out).double(), 0.999)
    assert ok, f"gate matmul with transposed tiles: PCC {pcc}"

    # down: act [32, E * 736] (bf16, L1 width-sharded) @ [E * 736, H]
    a = torch.randn(TOKENS, n_act) * 0.1
    d = k_grouped(torch.randn(n_act, H) * 0.02)
    a_tt = ttnn.from_torch(
        a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=plan.act_memory_config
    )
    d_tt = _weight(device, d, n_act, H)
    out = ttnn.matmul(
        a_tt,
        d_tt,
        program_config=plan.down_program_config,
        memory_config=plan.out_memory_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=plan.compute_kernel_config,
    )
    ref = ttnn.to_torch(a_tt).double() @ d.double()
    ok, pcc = comp_pcc(ref, ttnn.to_torch(out).double(), 0.999)
    assert ok, f"down matmul with transposed tiles: PCC {pcc}"
