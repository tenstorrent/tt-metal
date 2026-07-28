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

# TODO: verify on device — requires an actual 1x2 (Ring) mesh; the program_config and
# sharded memory configs the model uses (decode_all_gather_matmul_*) are omitted here,
# so this exercises the auto-config path of the op.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.modules.tt_ccl import (
    CCL_CHUNKS_PER_SYNC,
    CCL_NUM_BUFFERS_PER_CHANNEL,
    CCL_NUM_WORKERS_PER_LINK,
    get_tt_ccl,
)
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

# CCL collective — multi-device Ring topology only (T3K/Galaxy). NOT called by the
# model on a single-device N150 or the 2-node emulator (use_fused requires
# topology == Ring; single device binds the plain ttnn.linear WO path instead).
# Skipped so it drops cleanly out of an emulator run.
# Remove this mark to exercise it on a real multi-device Ring mesh.
pytestmark = pytest.mark.skip(reason="CCL op: multi-device Ring only; not used on single-device N150/emulator")


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

    # After all-gather over dim=3 (K = Q_DIM full) and matmul with WO, output width = DIM.
    U.assert_shape_dtype(dense_out, shape=(1, 1, m, U.DIM), mesh_device=mesh)
