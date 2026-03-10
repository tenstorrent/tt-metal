# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Test OLMo GQA Attention TTNN implementation against PyTorch reference.

Run with:
    export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think
    pytest models/demos/llama3_70b_galaxy/tests/test_olmo_attention.py -v
"""

import os
import pytest
import torch
import ttnn

from models.demos.llama3_70b_galaxy.reference.functional import attention_forward
from models.demos.llama3_70b_galaxy.reference.yarn_rope import (
    YaRNConfig,
    precompute_yarn_freqs,
)
from models.common.utility_functions import comp_pcc


def get_olmo_weights():
    """Load OLMo weights from HF_MODEL."""
    hf_model = os.getenv("HF_MODEL")
    if not hf_model:
        pytest.skip("HF_MODEL not set")

    import glob
    import json
    from safetensors.torch import load_file

    base_path = os.path.expanduser(hf_model)
    if os.path.exists(os.path.join(base_path, "snapshots")):
        snapshot_dirs = glob.glob(os.path.join(base_path, "snapshots", "*"))
        if snapshot_dirs:
            base_path = snapshot_dirs[0]

    config_path = os.path.join(base_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    safetensor_files = sorted(glob.glob(os.path.join(base_path, "model-*.safetensors")))
    if not safetensor_files:
        pytest.skip("No safetensor files found")

    state_dict = load_file(safetensor_files[0])
    return config, state_dict


@pytest.fixture
def olmo_config_and_weights():
    """Fixture to load OLMo config and weights."""
    return get_olmo_weights()


class TestOlmoAttentionReference:
    """Test OLMo attention reference implementation."""

    def test_attention_shapes(self, olmo_config_and_weights):
        """Test attention output shapes match input."""
        config, state_dict = olmo_config_and_weights

        dim = config["hidden_size"]  # 5120
        n_heads = config["num_attention_heads"]  # 40
        n_kv_heads = config["num_key_value_heads"]  # 8
        head_dim = dim // n_heads  # 128

        batch, seq_len = 1, 128

        # Create test input
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, dim, dtype=torch.float32)

        # Get weights
        q_proj = state_dict["model.layers.0.self_attn.q_proj.weight"].float()
        k_proj = state_dict["model.layers.0.self_attn.k_proj.weight"].float()
        v_proj = state_dict["model.layers.0.self_attn.v_proj.weight"].float()
        o_proj = state_dict["model.layers.0.self_attn.o_proj.weight"].float()

        # Precompute YaRN RoPE
        yarn_config = YaRNConfig.from_olmo()
        cos, sin, mscale = precompute_yarn_freqs(yarn_config, seq_len=seq_len)

        # Reference forward
        output = attention_forward(
            x,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            cos,
            sin,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            mscale=mscale,
        )

        assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        print(f"OLMo Attention reference test passed")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  n_heads: {n_heads}, n_kv_heads: {n_kv_heads}, head_dim: {head_dim}")
        print(f"  GQA ratio: {n_heads // n_kv_heads}:1")
        print(f"  mscale: {mscale}")


@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],  # Galaxy TG mesh shape
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
class TestOlmoAttentionTTNN:
    """Test OLMo Attention on TTNN device using TtLlamaAttention."""

    def test_olmo_attention_prefill(self, mesh_device, olmo_config_and_weights):
        """Test OLMo Attention TTNN vs PyTorch reference PCC."""
        from models.demos.llama3_70b_galaxy.tt.llama_attention import TtLlamaAttention
        from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
        from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
        from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
        from models.demos.llama3_70b_galaxy.tt.llama_common import (
            precompute_freqs_yarn,
            gather_cos_sin,
            get_rot_transformation_mat,
        )
        from loguru import logger

        config, _ = olmo_config_and_weights
        dim = config["hidden_size"]
        n_heads = config["num_attention_heads"]
        n_kv_heads = config["num_key_value_heads"]
        head_dim = dim // n_heads
        dtype = ttnn.bfloat8_b
        batch_size = 1
        seq_len = 128

        # Load OLMo model config
        model_args = TtOlmoModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=seq_len)
        model_args.n_layers = 1
        model_args.use_prefetcher = False
        state_dict = model_args.load_state_dict()

        logger.info(f"OLMo Attention Config: n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
        logger.info(f"OLMo yarn_attention_factor: {model_args.yarn_attention_factor}")

        # Setup prefetcher and CCL
        prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="prefill")
        mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
        tt_ccl = TT_CCL(
            mesh_device, model_args, prefetcher_setup.worker_sub_device_id, mode="prefill", is_qwen=False, is_olmo=True
        )

        # Setup RoPE transformation matrices
        # The transformation matrix is used for interleaving in rotary embedding
        prefill_trans_mat_torch = get_rot_transformation_mat(dhead=head_dim)
        transformation_mat_prefill = ttnn.from_torch(
            prefill_trans_mat_torch,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        transformation_mats = {"decode": transformation_mat_prefill, "prefill": transformation_mat_prefill}

        # Create TTNN Attention model
        model_args.WEIGHTS_DTYPE = dtype
        tt_model = TtLlamaAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            layer_num=0,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=model_args,
            prefetcher_setup=prefetcher_setup,
            tt_ccl=tt_ccl,
        )

        # Create test input
        torch.manual_seed(42)
        torch_input = torch.randn(1, 1, seq_len, dim)

        # Compute YaRN RoPE cos/sin for TTNN
        ttnn_cos, ttnn_sin, _ = precompute_freqs_yarn(
            dim=head_dim,
            end=model_args.max_seq_len * 2,
            theta=model_args.rope_theta,
            scaling_factor=model_args.rope_scaling_factor,
            original_max_position_embeddings=model_args.original_max_position_embeddings,
            beta_fast=model_args.yarn_beta_fast,
            beta_slow=model_args.yarn_beta_slow,
            attention_factor=model_args.yarn_attention_factor,
        )
        position_ids = torch.arange(seq_len)
        cos_gathered, sin_gathered = gather_cos_sin(position_ids, ttnn_cos, ttnn_sin)

        rot_mats_tt = [
            ttnn.from_torch(
                cos_gathered,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            ttnn.from_torch(
                sin_gathered,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
        ]

        # TTNN forward
        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 3),
                mesh_shape=model_args.cluster_shape,
            ),
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        tt_output = tt_model.forward_prefill(tt_input, rot_mats_tt, user_id=0, batch_size=batch_size)

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        tt_output_torch = tt_output_torch[:, :1, :, :dim]

        # Reference forward
        yarn_config = YaRNConfig.from_olmo()
        ref_cos, ref_sin, mscale = precompute_yarn_freqs(yarn_config, seq_len=seq_len)

        wq = state_dict["layers.0.attention.wq.weight"].float()
        wk = state_dict["layers.0.attention.wk.weight"].float()
        wv = state_dict["layers.0.attention.wv.weight"].float()
        wo = state_dict["layers.0.attention.wo.weight"].float()

        ref_input = torch_input[:, :1, :, :dim].squeeze(0).squeeze(0).float()  # [seq_len, dim]
        ref_input = ref_input.unsqueeze(0)  # [1, seq_len, dim]

        ref_output = attention_forward(
            ref_input,
            wq,
            wk,
            wv,
            wo,
            ref_cos,
            ref_sin,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            mscale=mscale,
        )

        # Compare PCC
        # Lower threshold for attention due to bfloat8_b quantization and multi-device all-gather
        pcc_required = 0.95
        passing, pcc_message = comp_pcc(ref_output.unsqueeze(0), tt_output_torch, pcc_required)

        logger.info(f"OLMo Attention PCC: {pcc_message}")

        tt_ccl.close()

        assert passing, f"OLMo Attention prefill PCC {pcc_message} < {pcc_required}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
