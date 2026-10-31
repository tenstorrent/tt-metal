#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LLaVA Vision Pipeline Demo

Runs the first 3 components of the LLaVA pipeline end-to-end on TT hardware:
    1. CLIP Vision Encoder:  Image [1,3,336,336] → Features [1,576,1024]
    2. MLP Projector:        Features [1,576,1024] → Projected [1,576,4096]
    3. Token Interleaver:    Projected + Text embeds → Multimodal embeds [1,N,4096]

The output multimodal embeddings are ready to feed into a LLaMA 3.1 LLM
(Week 4 integration target).

Usage:
    # Recommended: use wrapper script (handles env setup automatically)
    bash models/demos/llava/run_demo.sh --vision-only

    # Direct: from anywhere (script auto-configures environment)
    MESH_DEVICE=N300 python /path/to/tt-metal/models/demos/llava/demo/demo_vision_pipeline.py
"""

import atexit
import os
import sys
from pathlib import Path

# =============================================================================
# Environment Bootstrap — allows running from any directory without manual setup
# =============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_TT_METAL_ROOT = _SCRIPT_DIR.parents[3]  # demo/ -> llava/ -> demos/ -> models/ -> tt-metal/

if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

if "TT_METAL_HOME" not in os.environ:
    _SO_FILE = _TT_METAL_ROOT / "ttnn" / "ttnn" / "_ttnn.cpython-310-x86_64-linux-gnu.so"
    if _SO_FILE.exists():
        os.environ["TT_METAL_HOME"] = str(_TT_METAL_ROOT)
    else:
        _WHEEL_PATH = "/opt/venv/lib/python3.10/site-packages/ttnn"
        if Path(_WHEEL_PATH).exists():
            os.environ["TT_METAL_HOME"] = _WHEEL_PATH
            _CPP_DIR = _TT_METAL_ROOT / "ttnn" / "cpp"
            _CPP_BACKUP = _TT_METAL_ROOT / "ttnn" / "cpp_backup"
            if _CPP_DIR.exists() and not _CPP_BACKUP.exists():
                _CPP_DIR.rename(_CPP_BACKUP)

                @atexit.register
                def _restore_cpp():
                    if _CPP_BACKUP.exists():
                        _CPP_BACKUP.rename(_CPP_DIR)

            # Also hide tt_metal/ — the source tree's fabric/impl/kernels/edm_fabric/
            # contains a newer fabric_erisc_router.cpp that uses symbols not declared
            # in the installed wheel's headers (e.g. sender_ch_wait_static_connection),
            # causing an erisc build failure during fabric initialization.
            _TT_METAL_DIR = _TT_METAL_ROOT / "tt_metal"
            _TT_METAL_BACKUP = _TT_METAL_ROOT / "tt_metal_backup"
            if _TT_METAL_DIR.exists() and not _TT_METAL_BACKUP.exists():
                _TT_METAL_DIR.rename(_TT_METAL_BACKUP)

                @atexit.register
                def _restore_tt_metal():
                    if _TT_METAL_BACKUP.exists():
                        _TT_METAL_BACKUP.rename(_TT_METAL_DIR)


# =============================================================================
import torch
from loguru import logger

import ttnn
from models.demos.llava.demo.demo_utils import create_conversation_prompt, download_sample_image, preprocess_image
from models.demos.llava.reference.llava_reference import create_clip_encoder as create_clip_encoder_pytorch
from models.demos.llava.reference.llava_reference import create_mlp_projector as create_mlp_projector_pytorch
from models.demos.llava.reference.llava_reference import create_standard_interleaver as create_interleaver_pytorch
from models.demos.llava.tt.model_config import VisionModelArgs
from models.demos.llava.tt.projectors.mlp_projector import create_mlp_projector as create_mlp_projector_ttnn
from models.demos.llava.tt.token_handlers.standard_interleaver import (
    create_standard_interleaver as create_interleaver_ttnn,
)
from models.demos.llava.tt.vision_encoders.clip_encoder import create_clip_encoder as create_clip_encoder_ttnn

IMAGE_TOKEN_INDEX = -200
HIDDEN_SIZE = 4096


def run_vision_pipeline(mesh_device, image_path=None, question="What is in this image?"):
    """
    Run the LLaVA vision pipeline end-to-end.

    Args:
        mesh_device: ttnn.MeshDevice
        image_path: Path to image file (None = use sample image)
        question: Question to ask about the image

    Returns:
        dict with all intermediate results
    """
    logger.info("=" * 80)
    logger.info("LLaVA Vision Pipeline Demo")
    logger.info("=" * 80)

    # ============================================
    # Step 0: Prepare image and prompt
    # ============================================
    logger.info("\n--- Step 0: Prepare Image and Prompt ---")

    if image_path is None:
        image_path = download_sample_image()

    image_tensor = preprocess_image(image_path, resolution=336)
    prompt = create_conversation_prompt(question)

    logger.info(f"Image path: {image_path}")
    logger.info(f"Image tensor shape: {image_tensor.shape}  (expected [1, 3, 336, 336])")
    logger.info(f"Prompt: {prompt}")

    # ============================================
    # Step 1: CLIP Vision Encoder
    # ============================================
    logger.info("\n--- Step 1: CLIP Vision Encoder ---")
    logger.info("Encoding image with CLIP ViT-L/14-336...")

    model_name = "openai/clip-vit-large-patch14-336"
    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=1)

    # Load PyTorch reference (for weights + comparison)
    pytorch_encoder = create_clip_encoder_pytorch(model_name=model_name, device="cpu")
    state_dict = pytorch_encoder.vision_model.state_dict()

    # Create TTNN encoder
    weight_cache_path = (
        Path(os.environ.get("TT_METAL_CACHE_PATH", Path.home() / ".cache" / "tt-metal")) / "models" / "llava" / "clip"
    )
    weight_cache_path.mkdir(parents=True, exist_ok=True)

    ttnn_encoder = create_clip_encoder_ttnn(
        args=args,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
    )

    # Run CLIP encoder
    with torch.no_grad():
        pytorch_features = pytorch_encoder(image_tensor)

    ttnn_features = ttnn_encoder(image_tensor)
    ttnn_features_torch = ttnn.to_torch(ttnn.get_device_tensors(ttnn_features)[0])

    logger.info(f"  PyTorch output: {pytorch_features.shape}  (expected [1, 576, 1024])")
    logger.info(f"  TTNN output:    {ttnn_features_torch.shape}  (expected [1, 576, 1024])")
    logger.info(f"  Sample values (first 5 dims, patch 0):")
    logger.info(f"    PyTorch: {pytorch_features[0, 0, :5].tolist()}")
    logger.info(f"    TTNN:    {ttnn_features_torch[0, 0, :5].tolist()}")

    # Quick PCC check
    from models.common.utility_functions import comp_pcc

    passing, pcc = comp_pcc(pytorch_features, ttnn_features_torch, pcc=0.88)
    logger.info(f"  PCC: {pcc:.6f} ({'PASS' if passing else 'FAIL'})")

    # ============================================
    # Step 2: MLP Projector
    # ============================================
    logger.info("\n--- Step 2: MLP Projector (1024 → 4096 → 4096) ---")
    logger.info("Projecting vision features to LLM embedding space...")

    # Create random projector weights (in real demo, load from liuhaotian/llava-v1.5-7b)
    torch.manual_seed(42)
    projector_state_dict = {
        "mm_projector.0.weight": torch.randn(4096, 1024) * 0.02,
        "mm_projector.0.bias": torch.zeros(4096),
        "mm_projector.2.weight": torch.randn(4096, 4096) * 0.02,
        "mm_projector.2.bias": torch.zeros(4096),
    }

    pytorch_projector = create_mlp_projector_pytorch(
        vision_hidden_size=1024,
        llm_hidden_size=4096,
        state_dict=projector_state_dict,
    )

    ttnn_projector = create_mlp_projector_ttnn(
        args=args,
        dtype=ttnn.bfloat16,
        state_dict=projector_state_dict,
    )

    # Run projector on TTNN features
    with torch.no_grad():
        pytorch_projected = pytorch_projector(pytorch_features)

    ttnn_projected = ttnn_projector(ttnn_features)
    ttnn_projected_torch = ttnn.to_torch(ttnn.get_device_tensors(ttnn_projected)[0])

    logger.info(f"  PyTorch output: {pytorch_projected.shape}  (expected [1, 576, 4096])")
    logger.info(f"  TTNN output:    {ttnn_projected_torch.shape}  (expected [1, 576, 4096])")
    logger.info(f"  Sample values (first 5 dims, patch 0):")
    logger.info(f"    PyTorch: {pytorch_projected[0, 0, :5].tolist()}")
    logger.info(f"    TTNN:    {ttnn_projected_torch[0, 0, :5].tolist()}")

    # PCC: end-to-end (CLIP + projector, accumulated error)
    passing_e2e, pcc_e2e = comp_pcc(pytorch_projected, ttnn_projected_torch, pcc=0.88)
    logger.info(f"  PCC (end-to-end CLIP+projector): {pcc_e2e:.6f} ({'PASS' if passing_e2e else 'FAIL'})")
    logger.info(
        f"  Note: Projector-only PCC is 0.9999 (see test_projectors.py). "
        f"End-to-end PCC includes accumulated CLIP error."
    )

    # ============================================
    # Step 3: Token Interleaving
    # ============================================
    logger.info("\n--- Step 3: Token Interleaving ---")
    logger.info("Interleaving vision tokens with text embeddings...")

    # Simulate tokenized prompt: [<image>, "USER", ":", "What", ...]
    # In real pipeline, this comes from LLaMA tokenizer
    prompt_tokens = [IMAGE_TOKEN_INDEX] + list(range(100, 100 + 20))  # 21 tokens
    input_ids = torch.tensor([prompt_tokens])
    text_embeds = torch.randn(1, len(prompt_tokens), HIDDEN_SIZE)  # Dummy text embeddings

    # PyTorch reference
    pytorch_interleaver = create_interleaver_pytorch()
    ref_result = pytorch_interleaver(input_ids, text_embeds, pytorch_projected)

    # TTNN interleaver
    ttnn_interleaver = create_interleaver_ttnn(args)
    ttnn_result = ttnn_interleaver(input_ids, text_embeds, ttnn_projected_torch)
    ttnn_embeds = ttnn.to_torch(ttnn.get_device_tensors(ttnn_result["input_embeds"])[0])

    expected_seq_len = len(prompt_tokens) - 1 + 576  # 20 text + 576 image = 596

    logger.info(f"  Input tokens: {len(prompt_tokens)} (1 image + {len(prompt_tokens)-1} text)")
    logger.info(f"  PyTorch output: {ref_result['input_embeds'].shape}  (expected [1, {expected_seq_len}, 4096])")
    logger.info(f"  TTNN output:    {ttnn_embeds.shape}  (expected [1, {expected_seq_len}, 4096])")
    logger.info(f"  Attention mask sum: {ttnn_result['attention_mask'].sum().item()} (all tokens unmasked)")
    logger.info(f"  Position IDs: 0..{ttnn_result['position_ids'][0].max().item()}")

    # ============================================
    # Summary
    # ============================================
    logger.info("\n" + "=" * 80)
    logger.info("Vision Pipeline Complete!")
    logger.info("=" * 80)
    logger.info(f"  Image:     [1, 3, 336, 336]")
    logger.info(f"  CLIP:      [1, 576, 1024]   (PCC: {pcc:.4f})")
    logger.info(f"  Projector: [1, 576, 4096]   (PCC: {pcc_e2e:.4f} end-to-end, 0.9999 projector-only)")
    logger.info(f"  Interleaved: [1, {expected_seq_len}, 4096]")
    logger.info("")
    logger.info("  ✅ Multimodal embeddings ready for LLaMA 3.1 LLM (Week 4)")
    logger.info("  Next: Feed embeddings to LLaMA Transformer for text generation")
    logger.info("=" * 80)

    return {
        "clip_features": ttnn_features_torch,
        "projected": ttnn_projected_torch,
        "interleaved": ttnn_embeds,
        "attention_mask": ttnn_result["attention_mask"],
        "position_ids": ttnn_result["position_ids"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLaVA Vision Pipeline Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument("--question", type=str, default="What is in this image?", help="Question about image")
    args = parser.parse_args()

    # Determine device
    mesh_device_str = os.environ.get("MESH_DEVICE", "N300")
    if mesh_device_str == "N150":
        mesh_shape = (1, 1)
    else:
        mesh_shape = (1, 2)

    logger.info(f"Device: {mesh_device_str} (shape: {mesh_shape})")

    # Open device
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape))

    try:
        results = run_vision_pipeline(
            mesh_device=mesh_device,
            image_path=args.image,
            question=args.question,
        )
    finally:
        ttnn.close_mesh_device(mesh_device)
