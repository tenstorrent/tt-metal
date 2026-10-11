# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the pooled synthetic experts are exactly what the reference says they are.

The whole point of building experts on device by sign flips is that the reference can name
every expert's weights exactly -- the dequantized pool times that expert's signs. This pins it
bit for bit on every chip, for every slot, at the real K3 shapes and placement, and records
how long the build takes (plan §5.4's cost question).
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.synthetic_moe import PooledRoutedExperts
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import routed_expert_weight_memory_config

pytestmark = run_for_blackhole()

HIDDEN = KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE
INTER = KimiK3Config.MOE_INTERMEDIATE_SIZE
EXPERTS = KimiK3Config.NUM_ROUTED_EXPERTS


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((2, 4), id="LB-2x4"), pytest.param((8, 4), id="GH-8x4")],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_pooled_experts_are_exact_sign_flips(mesh_device, device_params):
    rows, cols = mesh_device.shape
    if rows * cols != mesh_device.get_num_devices():
        pytest.skip("mesh does not match this box")
    experts_per_chip = EXPERTS // (rows * cols)
    provider = PooledRoutedExperts(EXPERTS, 16, HIDDEN, INTER, seed=11)

    start = time.perf_counter()
    gate, up, down = provider.build_device_weights(
        mesh_device,
        experts_per_chip,
        ttnn.bfloat4_b,
        lambda n: routed_expert_weight_memory_config(mesh_device, n, dram_nd_sharded=True),
    )
    ttnn.synchronize_device(mesh_device)
    build_s = time.perf_counter() - start
    logger.info(f"{rows}x{cols}: built {experts_per_chip} slots/chip from 16 pools in {build_s:.1f} s")

    reference = provider.reference_experts()
    # Every expert's sign vector differs from every other's (host side, so cheap to check fully).
    assert torch.unique(reference.sign, dim=0).shape[0] == EXPERTS, "two experts share a sign vector"
    table = ExpertMapping.create_global_expert_idx_table(experts_per_chip, rows, cols)
    assert reference.down.abs().max() > 0, "dequantized pools are empty"
    # Gate and up are the pool's own tensors, shared by every slot on that pool, so each pool is checked
    # once on every chip. That is complete only while every slot holds its pool's tensor OBJECT -- the
    # identity assert below fails if the provider ever builds per-slot copies. Down is per (chip, slot).
    pools = provider.pools
    for slot in range(experts_per_chip):
        assert gate[slot] is gate[slot % pools] and up[slot] is up[slot % pools], f"slot {slot} not on its pool"
    for pool in range(pools):
        for chip, (g_tt, u_tt) in enumerate(
            zip(ttnn.get_device_tensors(gate[pool]), ttnn.get_device_tensors(up[pool]), strict=True)
        ):
            assert torch.equal(ttnn.to_torch(g_tt).float().T, reference.gate[pool]), f"pool {pool} chip {chip}: gate"
            assert torch.equal(ttnn.to_torch(u_tt).float().T, reference.up[pool]), f"pool {pool} chip {chip}: up"
    for slot in range(experts_per_chip):
        down_chips = [ttnn.to_torch(d_tt).float().T for d_tt in ttnn.get_device_tensors(down[slot])]
        for chip, got_down in enumerate(down_chips):
            row, col = divmod(chip, cols)
            want = reference.expert(int(table[col, row, slot]))["down_proj"]
            # Equal values; bit patterns may differ ONLY in the sign of a zero (the device's bf4 repack
            # writes +0 where sign * Q(pool) gives -0), which no matmul can tell apart.
            assert torch.equal(got_down, want), f"slot {slot} chip {chip}: down is not sign * Q(pool)"
            differing_bits = got_down.view(torch.int32) != want.view(torch.int32)
            assert (want[differing_bits] == 0).all(), f"slot {slot} chip {chip}: non-zero bit mismatch"
        # Distinct experts on distinct chips in the same slot.
        assert not torch.equal(down_chips[0], down_chips[-1]), f"slot {slot}: first and last chip hold the same expert"

    for tensor in down:
        ttnn.deallocate(tensor)
    for tensor in {id(t): t for t in gate + up}.values():
        ttnn.deallocate(tensor)
