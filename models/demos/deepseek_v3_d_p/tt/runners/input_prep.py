# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""MLA-family (DeepSeek-V3 / Kimi) prefill input prep.

The model-agnostic engine helpers (mesh open, H2D service, trace loading) live in
the common package at ``models.demos.common.prefill.runners.runner_utils``. What
remains here is the one piece of model-specific glue the runtime needs:
``prepare_prefill_input_tensor`` (the SP-sharded chunk input), which backs
``TtPrefillRuntime.make_chunk_input``.

KV-cache PCC validation + golden loaders live in
``models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation``; the host-pull KV
diagnostics used only by tests live in ``tests/test_runner_utils.py``.
"""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reorder_tensor_chunks


def prepare_prefill_input_tensor(
    token_ids: list[int],
    mesh_device: ttnn.MeshDevice,
    sp_factor: int,
    is_balanced: bool,
    mesh_shape: tuple,
    sp_axis: int,
) -> ttnn.Tensor:
    """Shard and upload token IDs to device as a prefill input tensor.

    Produces an SP-sharded uint32 ROW_MAJOR DRAM tensor of shape
    [sp_factor, 1, len(token_ids) // sp_factor] — the format expected by
    TtPrefillTransformer.forward.
    """
    isl_per_chip = len(token_ids) // sp_factor
    if is_balanced:
        chunk_order = create_balanced_chunk_order(sp_factor)
        t = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        t = reorder_tensor_chunks(t, chunk_order, seq_dim=2)
        token_ids_sharded = t.squeeze(0).squeeze(-1).reshape(sp_factor, 1, isl_per_chip)
    else:
        token_ids_sharded = torch.tensor(token_ids, dtype=torch.int64).reshape(sp_factor, 1, isl_per_chip)
    return ttnn.from_torch(
        token_ids_sharded,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(sp_axis, None)),
    )
