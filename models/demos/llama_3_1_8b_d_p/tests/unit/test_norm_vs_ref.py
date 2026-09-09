# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm on the target mesh vs the torch reference.

Target mesh (8, 4), random weights, identical on both sides. Structure follows
`minimax_m3/tests/unit/test_norm_vs_ref.py`.

Both forms are covered, because which one runs is a construction-time decision and a bring-up that
only ever tests one has not tested the layout it ships with:

* **single-pass** — `ttnn.rms_norm` over full emb, gain replicated.
* **distributed** — `rms_norm_pre_all_gather` -> TP all-gather of sum(x^2) -> `rms_norm_post_all_gather`,
  gain sharded on dim -2.

There is deliberately no Gemma-fold case. Llama is plain `x_normed * weight`; the donor's
`(1 + weight)` fold is asserted OFF in the constructor, and
`tests/torch_ref/test_llama_reference.py::test_rms_norm_matches_upstream` pins that on the oracle side.
"""

import pytest
import torch

import ttnn
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE, RefRMSNorm
from models.demos.llama_3_1_8b_d_p.tt.rms_norm import RMSNorm

from ..test_factory import ACT_DTYPE, assert_pcc, parametrize_target_mesh

SEQ = 512


def _reference(config, seq_len, seed=0):
    """fp16 torch RMSNorm at real width, with a gain drawn around 1.0 where trained gains sit."""
    torch.manual_seed(seed)
    x = torch.randn(1, 1, seq_len, config.hidden_size, dtype=REF_DTYPE)
    weight = (1.0 + 0.1 * torch.randn(config.hidden_size)).to(REF_DTYPE)
    norm = RefRMSNorm(config.hidden_size, config.rms_norm_eps)
    with torch.no_grad():
        norm.weight.copy_(weight)
        out = norm(x)
    return x, weight, out


@parametrize_target_mesh()
@pytest.mark.parametrize("is_distributed", [False, True], ids=["single_pass", "distributed"])
def test_norm_vs_ref(mesh_device, device_params, config, hf_config, mesh_config, ccl_manager, topology_name, is_distributed):
    """Norm output vs the torch reference, for both the single-pass and distributed forms."""
    x, weight, golden = _reference(config, SEQ)

    norm = RMSNorm(
        mesh_device,
        hf_config,
        {"weight": weight},
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        is_distributed=is_distributed,
    )

    # Sequence shards on the SP rows; the feature dim is full emb on the single-pass path and
    # emb/tp on the distributed one (which normalizes each TP column's slice).
    dims = [2, None] if not is_distributed else [2, 3]
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ACT_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
    )
    tt_out = norm.forward(tt_x)
    ttnn.synchronize_device(mesh_device)

    concat_dims = (2, 3) if is_distributed else (2, None)
    if is_distributed:
        got = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)
            ),
        )
    else:
        # Replicated on the TP cols: gather the sequence over the SP rows, take one column.
        got = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 0), mesh_shape=tuple(mesh_device.shape)),
        )[:1]

    assert_pcc(f"rms_norm[{'distributed' if is_distributed else 'single_pass'}]", golden, got, topology_name)


@parametrize_target_mesh()
def test_norm_rejects_gemma_fold(mesh_device, device_params, config, hf_config, mesh_config, ccl_manager):
    """A config that asks for the Gemma `(1 + weight)` fold must be rejected, not quietly honoured.

    The donor reads `use_gemma_norm` off the config and folds `+1` into the gain at load time. Llama
    has no such fold, and a fold applied by accident is a pure accuracy loss with nothing to raise.
    """
    hf_config.use_gemma_norm = True
    torch.manual_seed(0)
    with pytest.raises(AssertionError, match="Gemma"):
        RMSNorm(
            mesh_device,
            hf_config,
            {"weight": torch.ones(config.hidden_size, dtype=REF_DTYPE)},
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
        )
