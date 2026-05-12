# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full DeltaNet BLOCK on mesh (8×4 BH Galaxy or 2×2 submesh) — PCC vs HF.

Validates: projections + conv1d + kernel + GroupRMSNormGated + out_proj all running
with V-head row-axis tensor parallelism. Each row handles n_v/rows V-heads.
"""
import sys

import pytest
import torch

import ttnn

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

from models.demos.qwen3_6_27b.reference.hf_loader import load_qwen36_config, load_qwen36_tensors
from models.demos.qwen3_6_27b.tests.ttnn.test_deltanet_block_e2e import _reconstruct_hf_qkvz_ba


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


@pytest.mark.parametrize("submesh_shape", [(8, 4)])  # full BH GLX (fabric needs the full system mesh)
def test_deltanet_block_mesh_pcc(submesh_shape):
    SYS_ROWS, SYS_COLS = 8, 4
    rows, cols = submesh_shape
    # Fabric requires opening the FULL system mesh.
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    parent_mesh = ttnn.open_mesh_device(ttnn.MeshShape(SYS_ROWS, SYS_COLS))
    if (rows, cols) != (SYS_ROWS, SYS_COLS):
        mesh = parent_mesh.create_submesh(ttnn.MeshShape(rows, cols))
    else:
        mesh = parent_mesh
    try:
        torch.manual_seed(42)
        cfg_dict = load_qwen36_config()
        hf_cfg = Qwen3NextConfig(**cfg_dict["text_config"])

        # Load real layer-0 weights
        prefix = "model.language_model.layers.0.linear_attn"
        keys = [
            f"{prefix}.{k}"
            for k in [
                "in_proj_qkv.weight",
                "in_proj_a.weight",
                "in_proj_b.weight",
                "in_proj_z.weight",
                "conv1d.weight",
                "A_log",
                "dt_bias",
                "norm.weight",
                "out_proj.weight",
            ]
        ]
        weights = load_qwen36_tensors(keys)

        # HF reference
        hf_block = Qwen3NextGatedDeltaNet(hf_cfg, layer_idx=0).eval()
        in_proj_qkvz, in_proj_ba = _reconstruct_hf_qkvz_ba(
            prefix,
            weights,
            n_v=hf_cfg.linear_num_value_heads,
            n_k=hf_cfg.linear_num_key_heads,
            hd_k=hf_cfg.linear_key_head_dim,
            hd_v=hf_cfg.linear_value_head_dim,
        )
        hf_block.load_state_dict(
            {
                "in_proj_qkvz.weight": in_proj_qkvz,
                "in_proj_ba.weight": in_proj_ba,
                "conv1d.weight": weights[f"{prefix}.conv1d.weight"].float(),
                "A_log": weights[f"{prefix}.A_log"].float(),
                "dt_bias": weights[f"{prefix}.dt_bias"].float(),
                "norm.weight": weights[f"{prefix}.norm.weight"].float(),
                "out_proj.weight": weights[f"{prefix}.out_proj.weight"].float(),
            }
        )

        # Build our mesh block
        from models.demos.qwen3_6_27b.tt.linear_attention import TtDeltaNetBlock

        tt_block = TtDeltaNetBlock(mesh, weights, prefix, hf_cfg)

        # Input — replicated hidden state across mesh
        B, T, H = 1, 32, hf_cfg.hidden_size
        hidden = torch.randn(B, T, H, dtype=torch.float32) * 0.02

        # HF forward
        with torch.no_grad():
            hf_out = hf_block(hidden, cache_position=torch.arange(T))
            if isinstance(hf_out, tuple):
                hf_out = hf_out[0]

        # TT forward — input replicated
        hidden_tt = ttnn.from_torch(
            hidden,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.create_mesh_mapper(mesh, ttnn.MeshMapperConfig(row_dim=None, col_dim=None)),
        )
        tt_out_tt = tt_block(hidden_tt)
        # Output is identical across all chips (after all_reduce). Gather all shards;
        # they should match. Use a composer that lets us inspect; take the first chip's copy.
        tt_out_full = ttnn.to_torch(
            tt_out_tt,
            mesh_composer=ttnn.create_mesh_composer(mesh, ttnn.MeshComposerConfig([0, 1])),
        ).float()
        # Shape [rows*B, cols*T, H]. All chips have the same data → take [:B, :T].
        tt_back = tt_out_full[:B, :T, :]

        print(f"  shapes: tt={tt_back.shape}, hf={hf_out.shape}")
        pcc = _pcc(tt_back, hf_out)
        print(f"  Mesh {rows}x{cols} DeltaNet block PCC: {pcc:.6f}")
        assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99"
    finally:
        if mesh is not parent_mesh:
            ttnn.close_mesh_device(mesh)
        ttnn.close_mesh_device(parent_mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
