# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.to_device``.

Model call site (modules/rope/rope_1d.py:169):
    rot_idxs = ttnn.to_device(rot_idxs, cfg.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

The model uses this to move a host-resident rotation-index tensor onto the mesh
device (DRAM) before the cos/sin embedding lookup. This is a value-preserving
placement op, so the reference is the original tensor read back.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U
from models.experimental.llama32_1b_quasar.utility_functions import nearest_32


@U.with_default_mesh()
@pytest.mark.parametrize(
    "shape",
    # rot_idxs: prepare_rot_idxs pads [batch] -> [1, nearest_32(batch)] (rope_1d.py), i.e. 32 for both decode batches.
    [pytest.param((1, nearest_32(b)), id=f"rot_idxs-batch{b}") for b in U.DECODE_BATCHES]
    + [pytest.param((1, 1, seq, U.DIM), id=f"activation-seq{seq}") for seq in U.PREFILL_SEQ_LENS],
)
@pytest.mark.parametrize("uint", [True, False], ids=["uint32", "bf16"])
def test_to_device(ttnn_mesh_device, reset_seeds, shape, uint):
    mesh = ttnn_mesh_device

    if uint:
        torch_in = torch.randint(0, 1000, shape, dtype=torch.int32)
        dtype = ttnn.uint32
        layout = ttnn.ROW_MAJOR_LAYOUT
    else:
        torch_in = U.torch_rand(shape)
        dtype = ttnn.bfloat16
        layout = ttnn.TILE_LAYOUT

    # Build a HOST tensor (no device=), mirroring prepare_rot_idxs(on_host=True).
    host_t = ttnn.from_torch(
        torch_in,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
    )

    dev_t = ttnn.experimental.quasar.to_device(host_t, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    U.assert_pcc(torch_in, dev_t, pcc=0.999, mesh_device=mesh)
