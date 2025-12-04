"""
Block-by-block profiling tests for OpenVLA optimization.
Each test profiles a single component to avoid profiler buffer overflow.
"""

import os
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

import ttnn
from models.tt_transformers.tt.multimodal.open_vla import (
    OpenVLAConfig,
    TTOpenVLAForActionPrediction,
    PrismaticVisionBackbone,
    TTNNPrismaticProjector,
)
from ttnn.model_preprocessing import preprocess_model_parameters


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": 81920,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "T3K": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), (1, 1))
    ],
    indirect=True,
)
def test_block1_dinov2_only(mesh_device):
    """Profile DinoV2 vision backbone only."""
    print("\n=== Profiling Block 1: DinoV2 Vision Backbone ===")

    # Create vision backbone
    vision_backbone = PrismaticVisionBackbone(
        use_fused_vision_backbone=True,
        image_sizes=[224, 224],
        timm_model_ids=["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
        timm_override_act_layers=[None, None],
        ttnn_device=mesh_device,
    )

    # Create dummy image input
    dummy_img = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)

    # Preprocess for TTNN
    dummy_img = torch.permute(dummy_img, (0, 2, 3, 1))
    dummy_img = torch.nn.functional.pad(dummy_img, (0, 1, 0, 0, 0, 0, 0, 0))
    pixel_values1 = ttnn.from_torch(dummy_img, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)

    # Profile DinoV2 forward pass
    print("Running DinoV2 forward pass...")
    patches = vision_backbone.ttnn_featurizer(pixel_values1)[:, 5:, :]
    ttnn.synchronize_device(mesh_device)

    print(f"DinoV2 output shape: {patches.shape}")
    print("=== Block 1 Complete ===\n")


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": 81920,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "T3K": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), (1, 1))
    ],
    indirect=True,
)
def test_block2_siglip_only(mesh_device):
    """Profile SigLIP vision backbone only."""
    print("\n=== Profiling Block 2: SigLIP Vision Backbone ===")

    # Create vision backbone
    vision_backbone = PrismaticVisionBackbone(
        use_fused_vision_backbone=True,
        image_sizes=[224, 224],
        timm_model_ids=["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
        timm_override_act_layers=[None, None],
        ttnn_device=mesh_device,
    )

    # Create dummy image input
    dummy_img = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)

    # Preprocess for TTNN
    dummy_img = torch.permute(dummy_img, (0, 2, 3, 1))
    dummy_img = torch.nn.functional.pad(dummy_img, (0, 1, 0, 0, 0, 0, 0, 0))
    pixel_values2 = ttnn.from_torch(dummy_img, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)

    # Profile SigLIP forward pass
    print("Running SigLIP forward pass...")
    patches_fused = vision_backbone.ttnn_fused_featurizer(pixel_values2)
    ttnn.synchronize_device(mesh_device)

    print(f"SigLIP output shape: {patches_fused.shape}")
    print("=== Block 2 Complete ===\n")


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": 81920,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "T3K": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), (1, 1))
    ],
    indirect=True,
)
def test_block3_projector_only(mesh_device):
    """Profile Projector only."""
    print("\n=== Profiling Block 3: Projector ===")

    # Create dummy vision features (concatenated DinoV2 + SigLIP)
    # DinoV2: 256 patches x 1024 dim
    # SigLIP: 256 patches x 1152 dim
    # Total: 256 patches x 2176 dim
    vision_dim = 2176
    llm_dim = 4096
    num_patches = 256

    dummy_features = torch.randn(1, num_patches, vision_dim, dtype=torch.bfloat16)
    dummy_features_tt = ttnn.from_torch(
        dummy_features, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )

    # Create projector
    from models.tt_transformers.tt.multimodal.open_vla import PrismaticProjector

    projector = PrismaticProjector(
        use_fused_vision_backbone=True,
        vision_dim=vision_dim,
        llm_dim=llm_dim,
    ).to(torch.bfloat16)

    # Preprocess for TTNN
    projector_params = preprocess_model_parameters(
        initialize_model=lambda: projector,
        device=mesh_device,
    )

    from models.tt_transformers.tt.multimodal.open_vla import TTNNPrismaticProjector

    ttnn_projector = TTNNPrismaticProjector(
        use_fused_vision_backbone=True,
        vision_dim=vision_dim,
        llm_dim=llm_dim,
        ttnn_device=mesh_device,
        params=projector_params,
    )

    # Profile projector forward pass
    print("Running Projector forward pass...")
    projected = ttnn_projector.forward(dummy_features_tt)
    ttnn.synchronize_device(mesh_device)

    print(f"Projector output shape: {projected.shape}")
    print("=== Block 3 Complete ===\n")


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": 81920,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "T3K": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), (1, 1))
    ],
    indirect=True,
)
def test_block4_llm_prefill_only(mesh_device):
    """Profile LLM prefill only (no vision)."""
    print("\n=== Profiling Block 4: LLM Prefill ===")

    from models.tt_transformers.tt.multimodal.open_vla import OpenVLALanguageModel

    os.environ["HF_MODEL"] = "meta-llama/Llama-2-7b-hf"

    # Create language model
    language_model = OpenVLALanguageModel(mesh_device)

    # Short prompt for prefill only
    prompt = "Hello"

    # Just do prefill, no decode
    from models.tt_transformers.tt.common import preprocess_inputs_prefill

    (
        input_tokens_prefill_pt,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    ) = preprocess_inputs_prefill(
        [prompt],
        language_model.tokenizer,
        language_model.model_args,
        False,
        1,  # Only 1 token to generate
        max_prefill_len=512,
    )

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(1, -1)

    # Profile prefill only
    print("Running LLM prefill...")
    logits = language_model.generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=language_model.page_table,
        kv_cache=language_model.tt_kv_cache,
        prompt_lens=decoding_pos,
    )
    ttnn.synchronize_device(mesh_device)

    print(f"Prefill logits shape: {logits.shape}")
    print("=== Block 4 Complete ===\n")


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": 81920,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "T3K": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), (1, 1))
    ],
    indirect=True,
)
def test_block5_llm_decode_only(mesh_device):
    """Profile LLM decode only (single step)."""
    print("\n=== Profiling Block 5: LLM Decode (1 step) ===")

    from models.tt_transformers.tt.multimodal.open_vla import OpenVLALanguageModel

    os.environ["HF_MODEL"] = "meta-llama/Llama-2-7b-hf"

    # Create language model
    language_model = OpenVLALanguageModel(mesh_device)

    # Dummy token for decode
    dummy_token = torch.tensor([[1]], dtype=torch.long)
    current_pos = torch.tensor([10])

    # Profile single decode step
    print("Running LLM decode (1 step)...")
    logits = language_model.generator.decode_forward_text(
        dummy_token,
        current_pos,
        page_table=language_model.page_table,
        kv_cache=language_model.tt_kv_cache,
        sampling_params=None,
    )
    ttnn.synchronize_device(mesh_device)

    print(f"Decode logits shape: {logits.shape}")
    print("=== Block 5 Complete ===\n")
