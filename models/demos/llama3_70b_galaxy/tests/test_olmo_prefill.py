# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3.1-32B Simple Prefill Test.

Tests prefill forward pass without full model wrapper.

Run with:
    export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think
    export LINE_RS=1
    pytest models/demos/llama3_70b_galaxy/tests/test_olmo_prefill.py -v -x
"""

import os
import torch
import pytest
from loguru import logger
import ttnn

from models.demos.llama3_70b_galaxy.tt.llama_common import (
    precompute_freqs_yarn,
    gather_cos_sin,
    get_rot_transformation_mat,
)
from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_attention import TtLlamaAttention
from models.demos.llama3_70b_galaxy.tt.llama_mlp import TtLlamaMLP
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL, tt_distributed_rmsnorm
from models.demos.llama3_70b_galaxy.reference.functional import (
    decoder_block_forward,
)
from models.demos.llama3_70b_galaxy.reference.yarn_rope import YaRNConfig, precompute_yarn_freqs
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

    # Load all safetensor shards
    safetensor_files = sorted(glob.glob(os.path.join(base_path, "model-*.safetensors")))
    if not safetensor_files:
        pytest.skip("No safetensor files found")

    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(f))

    return config, state_dict


@pytest.fixture
def olmo_config_and_weights():
    """Fixture to load OLMo config and weights."""
    return get_olmo_weights()


@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],  # Galaxy TG mesh shape
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
    ],
    indirect=True,
)
class TestOlmoPrefill:
    """Test OLMo prefill forward pass."""

    def test_prefill_single_layer(self, mesh_device, olmo_config_and_weights):
        """Test single layer prefill: RMSNorm -> Attention -> Add -> RMSNorm -> MLP -> Add."""
        config, raw_state_dict = olmo_config_and_weights

        # Config
        dim = config["hidden_size"]  # 5120
        n_heads = config["num_attention_heads"]  # 40
        n_kv_heads = config["num_key_value_heads"]  # 8
        head_dim = dim // n_heads  # 128
        intermediate_size = config["intermediate_size"]  # 27648
        dtype = ttnn.bfloat8_b
        batch_size = 1
        seq_len = 128
        layer_num = 0

        logger.info(f"OLMo Prefill Config: dim={dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}")
        logger.info(f"  intermediate_size={intermediate_size}, seq_len={seq_len}")

        # Load OLMo model config (for TTNN)
        model_args = TtOlmoModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=seq_len * 2)
        model_args.n_layers = 1
        model_args.use_prefetcher = False
        state_dict = model_args.load_state_dict()

        # Setup prefetcher and CCL
        prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="prefill")
        mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
        tt_ccl = TT_CCL(
            mesh_device, model_args, prefetcher_setup.worker_sub_device_id, mode="prefill", is_qwen=False, is_olmo=True
        )

        # Setup RoPE transformation matrices
        prefill_trans_mat_torch = get_rot_transformation_mat(dhead=head_dim)
        transformation_mat_prefill = ttnn.from_torch(
            prefill_trans_mat_torch,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        transformation_mats = {"decode": transformation_mat_prefill, "prefill": transformation_mat_prefill}

        # Create TTNN Attention
        model_args.WEIGHTS_DTYPE = dtype
        tt_attention = TtLlamaAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=model_args,
            prefetcher_setup=prefetcher_setup,
            tt_ccl=tt_ccl,
        )

        # Create TTNN MLP (uses default state_dict_prefix from args)
        tt_mlp = TtLlamaMLP(
            mesh_device=mesh_device,
            args=model_args,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            layer_num=layer_num,
            dtype=dtype,
            model_config=model_args.get_model_config(),
            prefetcher_setup=prefetcher_setup,
            tt_ccl=tt_ccl,
        )

        # Create norm weights (distributed across devices)
        # Gamma needs to be reshaped and sharded like in DistributedNorm
        attn_norm_weight = state_dict[f"layers.{layer_num}.attention_norm.weight"]
        ffn_norm_weight = state_dict[f"layers.{layer_num}.ffn_norm.weight"]

        # Reshape to 4D and shard across dim 2
        attn_norm_weight_4d = attn_norm_weight.unsqueeze(0).view(1, 1, dim // 32, 32)
        attn_norm_weight_tt = ttnn.as_tensor(
            attn_norm_weight_4d,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=list(mesh_device.shape)),
        )

        ffn_norm_weight_4d = ffn_norm_weight.unsqueeze(0).view(1, 1, dim // 32, 32)
        ffn_norm_weight_tt = ttnn.as_tensor(
            ffn_norm_weight_4d,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=list(mesh_device.shape)),
        )

        # Compute YaRN RoPE cos/sin
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

        # Create test input
        torch.manual_seed(42)
        pt_input = torch.randn(batch_size, seq_len, dim)

        # ===== TTNN Forward =====
        logger.info("Running TTNN prefill forward...")

        # Prepare input tensor (sharded across devices)
        tt_input = model_args.prepare_residual_tensor_prefill(pt_input)
        logger.info(f"  Input shape: {tt_input.shape}, dtype: {tt_input.dtype}")

        # Step 1: Attention norm
        logger.info("  Step 1: Attention RMSNorm...")
        attn_norm_out, _ = tt_distributed_rmsnorm(
            tt_input,
            epsilon=1e-6,
            gamma=attn_norm_weight_tt,
            mesh_device=mesh_device,
            compute_kernel_config=model_args.compute_kernel_config_lofi,
            tt_ccl=tt_ccl,
        )
        logger.info(f"    attn_norm_out shape: {attn_norm_out.shape}")

        # Step 2: Attention
        logger.info("  Step 2: Attention...")
        attn_out = tt_attention.forward_prefill(attn_norm_out, rot_mats_tt, user_id=0, batch_size=batch_size)
        logger.info(f"    attn_out shape: {attn_out.shape}")

        # Step 3: Residual add
        logger.info("  Step 3: Residual add (h = x + attn_out)...")
        h = ttnn.add(tt_input, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_input.deallocate(True)
        attn_out.deallocate(True)
        logger.info(f"    h shape: {h.shape}")

        # Step 4: FFN norm
        logger.info("  Step 4: FFN RMSNorm...")
        ffn_norm_out, _ = tt_distributed_rmsnorm(
            h,
            epsilon=1e-6,
            gamma=ffn_norm_weight_tt,
            mesh_device=mesh_device,
            compute_kernel_config=model_args.compute_kernel_config_lofi,
            tt_ccl=tt_ccl,
        )
        logger.info(f"    ffn_norm_out shape: {ffn_norm_out.shape}")

        # Step 5: MLP
        logger.info("  Step 5: MLP...")
        mlp_out = tt_mlp.forward(ffn_norm_out, mode="prefill", batch_size=batch_size)
        logger.info(f"    mlp_out shape: {mlp_out.shape}")

        # Step 6: Final residual add
        logger.info("  Step 6: Final residual add (out = h + mlp_out)...")
        tt_out = ttnn.add(h, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h.deallocate(True)
        mlp_out.deallocate(True)
        logger.info(f"    tt_out shape: {tt_out.shape}")

        # Convert to torch
        tt_out_torch = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        tt_out_torch = tt_out_torch[:, :1, :seq_len, :dim]  # Trim padding
        logger.info(f"  TTNN output shape: {tt_out_torch.shape}")

        # ===== Reference Forward =====
        logger.info("Running reference forward...")

        # Get weights in HF naming
        prefix = f"model.layers.{layer_num}"
        wq = raw_state_dict[f"{prefix}.self_attn.q_proj.weight"].float()
        wk = raw_state_dict[f"{prefix}.self_attn.k_proj.weight"].float()
        wv = raw_state_dict[f"{prefix}.self_attn.v_proj.weight"].float()
        wo = raw_state_dict[f"{prefix}.self_attn.o_proj.weight"].float()
        w1 = raw_state_dict[f"{prefix}.mlp.gate_proj.weight"].float()
        w2 = raw_state_dict[f"{prefix}.mlp.down_proj.weight"].float()
        w3 = raw_state_dict[f"{prefix}.mlp.up_proj.weight"].float()
        attn_norm_w = raw_state_dict[f"{prefix}.post_attention_layernorm.weight"].float()
        ffn_norm_w = raw_state_dict[f"{prefix}.post_feedforward_layernorm.weight"].float()

        # YaRN RoPE for reference
        yarn_config = YaRNConfig.from_olmo()
        ref_cos, ref_sin, mscale = precompute_yarn_freqs(yarn_config, seq_len=seq_len)

        ref_input = pt_input.float()  # [batch, seq_len, dim]

        ref_out = decoder_block_forward(
            ref_input,
            attention_norm_weight=attn_norm_w,
            ffn_norm_weight=ffn_norm_w,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            w1=w1,
            w2=w2,
            w3=w3,
            cos=ref_cos,
            sin=ref_sin,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            sliding_window=None,  # No sliding window for layer 0
            mscale=mscale,
            norm_eps=1e-6,
        )
        logger.info(f"  Reference output shape: {ref_out.shape}")

        # ===== Compare PCC =====
        # Lower threshold due to bfloat8_b and multi-device
        pcc_required = 0.95
        passing, pcc_message = comp_pcc(ref_out.unsqueeze(1), tt_out_torch, pcc_required)

        logger.info(f"OLMo Prefill Single Layer PCC: {pcc_message}")

        tt_ccl.close()

        assert passing, f"OLMo Prefill PCC {pcc_message} < {pcc_required}"
        logger.info("OLMo Prefill Single Layer Test PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
