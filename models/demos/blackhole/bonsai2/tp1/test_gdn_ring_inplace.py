# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Step 1: the fused recurrence writes the new recurrent state IN PLACE (no separate Copy).

The 15.8 us `Copy` after FusedRecurrentGatedDeltaRule exists only because the op allocates its state
output. The op already has an in-place mode -- `initial_state_block_idx` ("ring") -- where the state
output aliases the caller's buffer: the writer stores block (t*BH + h) as it goes. At T=1 every head
starts and ends at block h, so a [BH,Dk,Dv] ring with idx = arange(BH) is exactly the decode case, and
the arithmetic is identical (per_token = per_token || last = true either way).

This test pins the contract the decode path relies on:
  1. the returned state IS the input buffer (same address),
  2. the values match the non-ring call bit for bit-ish (same kernel, same tiles -> tiny PCC is 1.0),
  3. `ttnn.reshape([B,Nv,Dk,Dv] -> [BH,Dk,Dv])` is a metadata view, so rec_state needs no copy to
     become the ring and no copy to become rec_state again.

Run: MESH_DEVICE=P150 pytest models/demos/blackhole/bonsai2/tp1/test_gdn_ring_inplace.py -s
"""
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.gdn_fla_ref import make_gdn_inputs, pcc
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    fused_recurrent_gated_delta_rule_ttnn,
)

H, DK, DV = 48, 128, 128  # the Bonsai TP=1 GDN shape: Nv heads after GQA expansion
SCALE = DK**-0.5


def _to_dev(mesh_device, t, dtype=ttnn.float32):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _dev0(mesh_device, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:1]


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("seed", [0, 1])
def test_ring_inplace_matches_allocating(mesh_device, seed):
    q, k, v, beta, g = make_gdn_inputs(T=1, H=H, Dk=DK, Dv=DV, seed=seed)
    s0 = torch.randn(1, H, DK, DV, dtype=torch.float32) * 0.1

    q_tt, k_tt, v_tt = (_to_dev(mesh_device, x) for x in (q, k, v))
    beta_tt, g_tt = _to_dev(mesh_device, beta), _to_dev(mesh_device, g)

    # (a) today's path: the op allocates a fresh state tensor, the caller copies it back.
    s_alloc = _to_dev(mesh_device, s0)
    o_a, h_a = fused_recurrent_gated_delta_rule_ttnn(
        q_tt, k_tt, v_tt, beta_tt, g_tt, scale=SCALE, initial_state=s_alloc, device=mesh_device
    )
    assert h_a.buffer_address() != s_alloc.buffer_address(), "non-ring mode must allocate its own state"
    h_a_dev = _dev0(mesh_device, h_a)

    # (b) ring mode on the same shape: state output aliases the input buffer.
    s_ring = _to_dev(mesh_device, s0)
    assert list(s_ring.shape) == [1, H, DK, DV]
    ring = ttnn.reshape(s_ring, (H, DK, DV))  # metadata view of the live rec_state buffer
    assert ring.buffer_address() == s_ring.buffer_address(), "the [B,Nv,Dk,Dv] -> [BH,Dk,Dv] reshape must alias"
    idx = ttnn.from_torch(
        torch.arange(H, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    o_b, h_b = fused_recurrent_gated_delta_rule_ttnn(
        q_tt,
        k_tt,
        v_tt,
        beta_tt,
        g_tt,
        scale=SCALE,
        initial_state=ring,
        device=mesh_device,
        output_per_token_state=True,
        initial_state_block_idx=idx,
    )
    assert h_b.buffer_address() == ring.buffer_address(), "ring mode must write the state in place"
    assert h_b.buffer_address() == s_ring.buffer_address(), "and that is the caller's rec_state buffer"
    h_b_dev = ttnn.to_torch(ttnn.reshape(s_ring, (1, H, DK, DV))).float()

    p_h = pcc(h_b_dev, h_a_dev)
    p_o = pcc(_dev0(mesh_device, o_b), _dev0(mesh_device, o_a))
    logger.info(f"seed={seed}: in-place vs allocating state PCC {p_h:.8f}, output PCC {p_o:.8f}")
    assert p_h > 0.99999, f"in-place state differs from the allocating path: PCC {p_h}"
    assert p_o > 0.99999, f"in-place output differs from the allocating path: PCC {p_o}"

    # The state really advanced (not a no-op that left s0 in place).
    adv = pcc(h_b_dev, s0)
    logger.info(f"seed={seed}: state vs input s0 PCC {adv:.6f} (must be < 0.999 to prove it moved)")
    assert adv < 0.999, "state did not change: the in-place write did not land"
