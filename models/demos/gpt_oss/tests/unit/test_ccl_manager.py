# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CCLManager semaphores vs the worker cores the CCL ops pick.

reduce_scatter_minimal_async / all_gather_async choose their worker cores from the whole sub-device grid (11x10 on
Blackhole); a global semaphore exists, and is set to its initial value, only on the cores it was created on. With
semaphores created on an 8x8 grid, the workers outside it read whatever an earlier op or process left at the
semaphore address, so they can start before the data they wait for has arrived (wrong or NaN rows; the intermittent
NaN of the gate v3 L=1024 case). The test poisons all free L1 with NaN first (in this process, so the result does not
depend on what a previous process left there), then checks the prefill TP all-reduce (MeshConfig.allreduce) against
the torch sum at the shapes where the 8x8 semaphores gave wrong rows (128 / 512 / 1024 on Blackhole 1x4).

The second case does the same for the prefill attention TP sum the model runs on Blackhole: ttnn.all_reduce of the
unpadded [1, 1, S, 2880] o_proj output in its prefill dtype (bf8), with the arguments of
attention/operations.py:apply_output_projection_allreduce_prefill.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric, skip_1x4_on_galaxy
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.utils.general_utils import get_default_num_links


def _poison_free_l1(mesh_device):
    """Fill every free L1 page with NaN, then free it again (the values stay behind)."""
    held = []
    for shape in [(1024, 1024), (256, 256), (32, 1024), (32, 32)]:
        while True:
            try:
                held.append(
                    ttnn.full(
                        shape,
                        fill_value=float("nan"),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh_device,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                )
            except Exception:
                break
    ttnn.synchronize_device(mesh_device)
    assert held, "no L1 was poisoned"
    for t in held:
        ttnn.deallocate(t)


@skip_1x4_on_galaxy
@parametrize_mesh_with_fabric([(1, 4)])
def test_ccl_manager_allreduce_after_l1_poison(mesh_device, device_params):
    rows, cols = mesh_device.shape
    if rows != 1 or cols < 2:
        pytest.skip("single-row mesh with more than one device (the TP axis is the row)")
    _poison_free_l1(mesh_device)
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device))
    mesh_config = MeshConfig(tuple(mesh_device.shape), decode=ModeConfig(tp=cols, ep=rows))

    torch.manual_seed(0)
    width = 2944  # attention o_proj output width (hidden 2880 padded per device to a tile multiple at TP=4)
    failures = []
    for seq_len in (128, 512, 1024, 2048):
        for rep in range(2):  # both semaphore sets of the ping-pong
            x = torch.randn(cols, 1, seq_len, width).to(torch.bfloat16)  # device j holds x[j]
            tt_x = ttnn.from_torch(
                x,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 0)),
            )
            out = mesh_config.allreduce(tt_x, ccl_manager, pad_size=0, axis=mesh_config.tp_axis)
            got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
            ttnn.deallocate(out)
            ref = x.float().sum(0)  # [1, S, W], replicated on every device after the all-reduce
            err = (got - ref).abs()  # [cols, 1, S, W]
            bad_rows = (~torch.isfinite(err) | (err > 0.25)).any(-1).flatten(0, 1).any(0)  # [S]
            if bool(bad_rows.any()):
                failures.append(
                    f"S={seq_len} rep={rep}: {int(bad_rows.sum())} of {seq_len} rows wrong "
                    f"(non-finite values: {int((~torch.isfinite(got)).sum())})"
                )
    assert not failures, "TP all-reduce wrong after L1 poison:\n" + "\n".join(failures)


@skip_1x4_on_galaxy
@parametrize_mesh_with_fabric([(1, 4)])
def test_prefill_attention_allreduce_after_l1_poison(mesh_device, device_params):
    rows, cols = mesh_device.shape
    if rows != 1 or cols < 2:
        pytest.skip("single-row mesh with more than one device (the TP axis is the row)")
    _poison_free_l1(mesh_device)
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device))
    mesh_config = MeshConfig(tuple(mesh_device.shape), decode=ModeConfig(tp=cols, ep=rows))

    torch.manual_seed(0)
    width = 2880  # unpadded hidden size: the o_proj_prefill output width
    dtype = ttnn.bfloat8_b  # prefill o_proj output / all-reduce payload dtype on this path (attention/prefill.py)
    failures = []
    for seq_len in (128, 1024, 2048, 8192):
        for rep in range(2):  # both semaphore sets of the ping-pong
            x = torch.randn(cols, 1, seq_len, width)  # device j holds x[j]
            tt_x = ttnn.from_torch(
                x,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 0)),
            )
            # The inputs as the devices hold them (bf8), so the reference sum sees the same values.
            x_q = ttnn.to_torch(tt_x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
            # Same call and arguments as apply_output_projection_allreduce_prefill.
            out = ttnn.all_reduce(
                tt_x,
                num_links=ccl_manager.num_links,
                topology=ccl_manager.topology,
                cluster_axis=mesh_config.tp_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(tt_x)
            got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
            ttnn.deallocate(out)
            ref = x_q.sum(0)  # [1, S, W], replicated on every device after the all-reduce
            # bf8 rounds the payload and the partial sums (~1% of a row's norm); a row that misses or repeats one
            # device's partial is off by ~50%, a row read before its data arrived is NaN (poison) or unrelated.
            rel = (got - ref).norm(dim=-1) / ref.norm(dim=-1)  # [cols, 1, S]
            bad_rows = (~torch.isfinite(got).all(-1) | (rel > 0.05)).flatten(0, 1).any(0)  # [S]
            if bool(bad_rows.any()):
                failures.append(
                    f"S={seq_len} rep={rep}: {int(bad_rows.sum())} of {seq_len} rows wrong "
                    f"(non-finite values: {int((~torch.isfinite(got)).sum())}, max row rel err "
                    f"{float(rel.nan_to_num(float('inf')).max()):.4f})"
                )
            else:
                logger.info(f"S={seq_len} rep={rep}: max row rel err {float(rel.max()):.4f}")
    assert not failures, "prefill attention all_reduce wrong after L1 poison:\n" + "\n".join(failures)
