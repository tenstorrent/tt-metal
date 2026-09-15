# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.all_gather_matmul_async``  (MULTI-DEVICE).

Model call site (modules/attention/attention_1d.py):
  * L1188  _fused_all_gather_wo_decode — Ring-topology fused all-gather + WO matmul in
           the decode path. All-gathers the concatenated attention heads along dim=3,
           then matmuls with the WO weight [Q_DIM, DIM].

This op only makes sense on a multi-device mesh (it gathers a width-sharded tensor
across devices before the matmul), so it is parametrized on a (1, 2) mesh and skips
on the single-device emulator. Semaphores/kwargs are taken from the call site via the
shared TT_CCL helper. A full torch reference is impractical (CCL + sharded matmul), so
we assert output shape / dtype only.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.modules.tt_ccl import (
    CCL_CHUNKS_PER_SYNC,
    CCL_NUM_BUFFERS_PER_CHANNEL,
    CCL_NUM_WORKERS_PER_LINK,
    get_tt_ccl,
)
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U

# CCL + matmul fused op — needs a multi-device mesh. The (1, 2) parametrization below
# is skipped automatically by the ttnn_mesh_device fixture on single-device systems
# (N150 / 2-node emulator present as one device), so it stays out of the emulator run
# while still exercising the op on real multi-device systems (N300 / T3K).


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("batch", U.DECODE_BATCHES)
def test_all_gather_matmul_async(ttnn_mesh_device, reset_seeds, batch):
    mesh = ttnn_mesh_device
    if tuple(mesh.shape) == (1, 1):
        pytest.skip("multi-device op")

    num_devices = mesh.get_num_devices()

    # attn_output_cat: concatenated heads, width-sharded across devices on the last dim.
    m = U.TILE * ((batch + U.TILE - 1) // U.TILE)  # tile-padded decode rows
    x_torch = U.torch_rand((1, 1, m, U.Q_DIM))
    # WO weight [Q_DIM, DIM], column-sharded across the mesh (dim=-1), matching the model.
    w_torch = U.torch_rand((U.Q_DIM, U.DIM))

    # Input is sharded on the gather dim (last), weight sharded on its output dim.
    x = U.to_tt(x_torch, mesh, mesh_mapper=3)
    w = U.to_tt(w_torch, mesh, mesh_mapper=-1)

    tt_ccl = get_tt_ccl(mesh)

    # Kwarg structure copied from attention_1d.py:1188.
    _, dense_out = ttnn.experimental.all_gather_matmul_async(
        x,
        w,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        all_gather_core_grid_offset=(0, 4),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=1,
        memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        chunks_per_sync=CCL_CHUNKS_PER_SYNC,
        num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
        num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
    )

    # After all-gather over dim=3 (K = Q_DIM full) and matmul with the column-sharded WO, each device
    # produces its output-column slice; auto_compose concatenates them back to full [1,1,m,DIM] = x @ WO.
    ref = x_torch.float() @ w_torch.float()
    U.assert_shape_dtype(dense_out, shape=(1, 1, m, U.DIM), mesh_device=mesh)
    U.assert_pcc(ref, dense_out, pcc=0.99, mesh_device=mesh)
