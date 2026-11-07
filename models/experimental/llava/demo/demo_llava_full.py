#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LLaVA Full Demo - Image + Prompt → Text

Hybrid demo combining TTNN vision pipeline with PyTorch LLM for text generation.

Pipeline:
    1. TTNN CLIP Encoder:     Image → Vision features [1, 576, 1024]
    2. PyTorch MLP Projector: Features → Projected [1, 576, llm_hidden_size]
    3. Token Interleaver:     Projected + Text embeds → Multimodal embeds
    4. PyTorch LLM (Qwen2):   Multimodal embeds → Generated text

The vision encoder runs on Tenstorrent hardware (n150/n300).
The LLM runs on CPU using Qwen2-1.5B-Instruct (non-gated, ~3GB).

Usage:
    # Recommended: use wrapper script (handles env setup automatically)
    bash models/demos/llava/run_demo.sh --image /path/to/image.jpg --question "What color is the sky?"

    # Direct: from tt-metal root
    MESH_DEVICE=N300 python models/demos/llava/demo/demo_llava_full.py

    # Direct: from anywhere (script auto-configures environment)
    python /path/to/tt-metal/models/demos/llava/demo/demo_llava_full.py --image img.jpg

    # On n150 (single chip)
    MESH_DEVICE=N150 python models/demos/llava/demo/demo_llava_full.py
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

# Add tt-metal root to sys.path so `from models.demos...` works
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

# Set TT_METAL_HOME if not already set by the user
if "TT_METAL_HOME" not in os.environ:
    _SO_FILE = _TT_METAL_ROOT / "ttnn" / "ttnn" / "_ttnn.cpython-310-x86_64-linux-gnu.so"
    if _SO_FILE.exists():
        os.environ["TT_METAL_HOME"] = str(_TT_METAL_ROOT)
    else:
        # No source build — use the pre-installed wheel and hide source-tree
        # kernels that would clash with the installed wheel's headers.
        _WHEEL_PATH = "/opt/venv/lib/python3.10/site-packages/ttnn"
        if Path(_WHEEL_PATH).exists():
            os.environ["TT_METAL_HOME"] = _WHEEL_PATH
            # Hide source-tree kernels (incompatible with installed wheel)
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
# Now safe to import ttnn and project modules
# =============================================================================
import torch
from loguru import logger

# Compatibility shim for ttnn builds lacking ttnn.tools.trace_allocation_tracker
# (see models/demos/llava/ttnn_tools_shim.py; a no-op unless trace-alloc
# tracking is explicitly enabled).
import models.demos.llava.ttnn_tools_shim  # noqa: F401
import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llava.demo.demo_utils import download_sample_image, preprocess_image
from models.demos.llava.reference.llava_reference import create_clip_encoder as create_clip_encoder_pytorch
from models.demos.llava.tt.llava_model import enable_fabric_if_needed, release_mesh_device
from models.demos.llava.tt.model_config import VisionModelArgs
from models.demos.llava.tt.vision_encoders.clip_encoder import create_clip_encoder as create_clip_encoder_ttnn

IMAGE_TOKEN_INDEX = -200


def load_llava_model(model_name="aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning"):
    """Load full trained LLaVA model (projector + LLaMA LLM) from HuggingFace.

    Uses LLaVA-MORE with LLaMA 3.1 8B backbone (non-gated, ~16GB).
    The vision encoder (CLIP) runs separately on TTNN hardware.
    """
    import glob

    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    logger.info(f"Loading LLaVA model: {model_name}")

    # LLaVA-MORE uses custom llava_llama architecture not in standard transformers,
    # so we load weights directly from safetensors
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json", "tokenizer*"])

    # Load all safetensors files
    safetensor_files = sorted(glob.glob(f"{model_path}/*.safetensors"))
    state_dict = {}
    for f in safetensor_files:
        sd = load_file(f)
        state_dict.update(sd)
        logger.info(f"  Loaded {len(sd)} weights from {os.path.basename(f)}")

    # Load tokenizer — use LLaMA 3.1 tokenizer (LLaVA-MORE uses custom llava_llama type)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.1-8B-Instruct")

    # Extract projector weights
    proj_sd = {
        "mm_projector.0.weight": state_dict["model.mm_projector.0.weight"],
        "mm_projector.0.bias": state_dict["model.mm_projector.0.bias"],
        "mm_projector.2.weight": state_dict["model.mm_projector.2.weight"],
        "mm_projector.2.bias": state_dict["model.mm_projector.2.bias"],
    }

    # Create a simple projector module for CPU fallback
    import torch.nn as nn

    class SimpleProjector(nn.Module):
        def __init__(self, sd):
            super().__init__()
            self.linear_1 = nn.Linear(1024, 4096)
            self.linear_2 = nn.Linear(4096, 4096)
            self.linear_1.weight.data = sd["mm_projector.0.weight"]
            self.linear_1.bias.data = sd["mm_projector.0.bias"]
            self.linear_2.weight.data = sd["mm_projector.2.weight"]
            self.linear_2.bias.data = sd["mm_projector.2.bias"]

        def forward(self, x):
            import torch.nn.functional as F

            return self.linear_2(F.gelu(self.linear_1(x)))

    projector = SimpleProjector(proj_sd)
    projector.eval()
    projector.requires_grad_(False)

    llm_hidden_size = 4096  # LLaMA 3.1 8B

    logger.info(f"  LLM: LLaMA 3.1 8B, hidden_size={llm_hidden_size}")
    logger.info(f"  Projector: mlp2x_gelu (1024 → 4096 → 4096)")
    logger.info(f"  Total weights: {len(state_dict)}")

    return state_dict, projector, tokenizer, llm_hidden_size, model_path


def load_moe_llava_model(model_name="LanguageBind/MoE-LLaVA-Qwen-1.8B-4e"):
    """Load MoE-LLaVA-Qwen-1.8B-4e (projector + Qwen1 MoE LLM) from HuggingFace.

    The checkpoint is Qwen1-style: the LLM, vision tower and projector all nest
    under `transformer.*` (not LLaVA's `model.*`), and weights ship as
    pytorch_model-*.bin shards (no safetensors). The projector is
    mlp2x_gelu: Linear(1024 -> 2048) -> GELU -> Linear(2048 -> 2048).

    The QWenTokenizer is loaded from the HF hub (LanguageBind/
    MoE-LLaVA-Qwen-1.8B-4e; remote code auto-mapped to Qwen/Qwen-1_8B) — the
    >500KB qwen.tiktoken vocab is not vendored in the repo.
    """
    import glob

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    logger.info(f"Loading MoE-LLaVA model: {model_name}")

    model_path = snapshot_download(model_name, allow_patterns=["*.bin", "*.json", "qwen.tiktoken"])

    # Load all pytorch_model-*.bin shards (this checkpoint has no safetensors)
    bin_files = sorted(glob.glob(f"{model_path}/pytorch_model*.bin"))
    safetensor_files = sorted(glob.glob(f"{model_path}/*.safetensors"))
    state_dict = {}
    if bin_files:
        for f in bin_files:
            try:
                sd = torch.load(f, map_location="cpu", weights_only=True)
            except Exception:
                sd = torch.load(f, map_location="cpu", weights_only=False)
            state_dict.update(sd)
            logger.info(f"  Loaded {len(sd)} weights from {os.path.basename(f)}")
    else:
        for f in safetensor_files:
            sd = load_file(f)
            state_dict.update(sd)
            logger.info(f"  Loaded {len(sd)} weights from {os.path.basename(f)}")

    # Qwen1 tokenizer from the hub (remote code via Qwen/Qwen-1_8B; qwen.tiktoken
    # vocab comes with the snapshot download above)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Extract projector weights (transformer.mm_projector.image_spatial_proj.*)
    proj_sd = {
        "mm_projector.0.weight": state_dict["transformer.mm_projector.image_spatial_proj.0.weight"],
        "mm_projector.0.bias": state_dict["transformer.mm_projector.image_spatial_proj.0.bias"],
        "mm_projector.2.weight": state_dict["transformer.mm_projector.image_spatial_proj.2.weight"],
        "mm_projector.2.bias": state_dict["transformer.mm_projector.image_spatial_proj.2.bias"],
    }

    # Projector module for CPU fallback: mlp2x_gelu 1024 -> 2048 -> 2048
    import torch.nn as nn

    class SimpleProjector(nn.Module):
        def __init__(self, sd):
            super().__init__()
            self.linear_1 = nn.Linear(1024, 2048)
            self.linear_2 = nn.Linear(2048, 2048)
            self.linear_1.weight.data = sd["mm_projector.0.weight"]
            self.linear_1.bias.data = sd["mm_projector.0.bias"]
            self.linear_2.weight.data = sd["mm_projector.2.weight"]
            self.linear_2.bias.data = sd["mm_projector.2.bias"]

        def forward(self, x):
            import torch.nn.functional as F

            return self.linear_2(F.gelu(self.linear_1(x)))

    projector = SimpleProjector(proj_sd)
    projector.eval()
    projector.requires_grad_(False)

    llm_hidden_size = 2048  # Qwen-1.8B

    logger.info(f"  LLM: Qwen-1.8B MoE (4 experts, top-2, 12 of 24 layers), hidden_size={llm_hidden_size}")
    logger.info(f"  Projector: mlp2x_gelu (1024 → 2048 → 2048)")
    logger.info(f"  Total weights: {len(state_dict)}")

    return state_dict, projector, tokenizer, llm_hidden_size, model_path


def get_text_embeddings(model, input_ids):
    """Get text token embeddings from LLM's embedding layer"""
    with torch.no_grad():
        embeds = model.model.embed_tokens(input_ids)
    return embeds


def generate_text_from_embeds(
    model,
    input_embeds,
    attention_mask,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
):
    """Generate text from input embeddings using the LLM"""
    with torch.no_grad():
        output = model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=model.config.eos_token_id,
            eos_token_id=model.config.eos_token_id,
        )
    return output


def generate_text_only(model, tokenizer, prompt_text, max_new_tokens=128):
    """Generate text from text-only input (no image) for baseline comparison"""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=model.config.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)


def run_full_demo(
    mesh_device,
    image_path=None,
    question="What is in this image?",
    max_new_tokens=128,
):
    """
    Run the full LLaVA demo: image + prompt → text.

    Uses TTNN CLIP encoder on Tenstorrent hardware for vision,
    and trained LLaVA-MORE projector + LLaMA 3.1 LLM on CPU for text generation.

    Args:
        mesh_device: ttnn.MeshDevice
        image_path: Path to image (None = sample image)
        question: Question about the image
        max_new_tokens: Max tokens to generate

    Returns:
        Generated text
    """
    logger.info("=" * 80)
    logger.info("LLaVA Full Demo - Image + Prompt → Text")
    logger.info("=" * 80)

    # ============================================
    # Step 0: Load trained LLaVA-MORE model and prepare image
    # ============================================
    logger.info("\n--- Step 0: Load Models ---")

    llava_state_dict, projector, tokenizer, llm_hidden_size, model_path = load_llava_model()

    # Load LLaMA 3.1 model for CPU inference (text generation)
    from transformers import AutoModelForCausalLM

    logger.info("Loading LLaMA 3.1 8B for CPU inference...")
    llm = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16)
    llm.eval()
    llm.requires_grad_(False)

    if image_path is None:
        image_path = download_sample_image()
    image_tensor = preprocess_image(image_path, resolution=336)

    logger.info(f"Image: {image_path}")
    logger.info(f"Question: {question}")
    logger.info(f"LLM hidden size: {llm_hidden_size}")

    # ============================================
    # Step 1: TTNN CLIP Vision Encoder
    # ============================================
    logger.info("\n--- Step 1: CLIP Vision Encoder (TTNN) ---")

    model_name = "openai/clip-vit-large-patch14-336"
    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=1)

    pytorch_encoder = create_clip_encoder_pytorch(model_name=model_name, device="cpu")
    state_dict = pytorch_encoder.vision_model.state_dict()

    weight_cache_path = (
        Path(os.environ.get("TT_METAL_CACHE_PATH", Path.home() / ".cache" / "tt-metal")) / "models" / "llava" / "clip"
    )
    weight_cache_path.mkdir(parents=True, exist_ok=True)

    ttnn_encoder = create_clip_encoder_ttnn(args=args, state_dict=state_dict, weight_cache_path=weight_cache_path)

    # Run CLIP on TTNN hardware
    ttnn_features = ttnn_encoder(image_tensor)
    clip_features = ttnn.to_torch(ttnn.get_device_tensors(ttnn_features)[0])

    # Also run PyTorch reference for PCC
    with torch.no_grad():
        pytorch_features = pytorch_encoder(image_tensor)
    passing, pcc = comp_pcc(pytorch_features, clip_features, pcc=0.88)

    logger.info(f"  CLIP output: {clip_features.shape}  (PCC: {pcc:.4f})")

    # ============================================
    # Step 2: TTNN MLP Projector (trained weights, on n300)
    # ============================================
    logger.info("\n--- Step 2: MLP Projector (TTNN, trained weights) ---")

    # Extract trained projector weights and convert to our state_dict format
    proj_sd = {
        "mm_projector.0.weight": projector.linear_1.weight.data,
        "mm_projector.0.bias": projector.linear_1.bias.data,
        "mm_projector.2.weight": projector.linear_2.weight.data,
        "mm_projector.2.bias": projector.linear_2.bias.data,
    }

    # Create TTNN projector (runs on n300)
    from models.demos.llava.tt.projectors.mlp_projector import create_mlp_projector as create_mlp_projector_ttnn

    ttnn_projector = create_mlp_projector_ttnn(
        args=args,
        dtype=ttnn.bfloat16,
        state_dict=proj_sd,
    )

    # Run projector on n300 — feed TTNN CLIP output directly (stays on device)
    ttnn_projected = ttnn_projector(ttnn_features)
    projected = ttnn.to_torch(ttnn.get_device_tensors(ttnn_projected)[0])

    logger.info(f"  Projector output: {projected.shape}  (on n300)")

    # ============================================
    # Step 3: Tokenize prompt and interleave
    # ============================================
    logger.info("\n--- Step 3: Tokenize & Interleave ---")

    # Use LLaMA 3.1 chat template with <image> placeholder
    # LLaVA-MORE uses the same <image> token as LLaVA 1.5
    # We tokenize manually: expand <image> to 576 placeholder tokens
    IMAGE_TOKEN_ID = 128256  # Use a token ID beyond vocab for image placeholder
    # Actually, LLaVA uses a special token. Let's use the tokenizer's pad or a high ID
    # and replace those positions with image features
    # For LLaMA 3.1, we use the chat template
    messages = [
        {"role": "user", "content": f"<image>\n{question}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Replace <image> with a repeatable token, then tokenize
    # We use token ID 0 (pad/bos) as placeholder — will be overwritten with image features
    PLACEHOLDER_ID = 0
    text_without_image = text.replace("<image>", "")
    text_parts = text.split("<image>")

    ids_before = tokenizer(text_parts[0], return_tensors="pt", add_special_tokens=False).input_ids
    ids_after = tokenizer(text_parts[1], return_tensors="pt", add_special_tokens=False).input_ids

    # Create 576 placeholder tokens for image
    image_placeholders = torch.full((1, 576), PLACEHOLDER_ID, dtype=torch.long)
    input_ids = torch.cat([ids_before, image_placeholders, ids_after], dim=1)

    logger.info(f"  Prompt tokens: {input_ids.shape[1]} (includes 576 image placeholders)")
    logger.info(f"  Image features to insert: {projected.shape[1]}")

    # Get text embeddings for ALL tokens (image positions will be overwritten)
    with torch.no_grad():
        all_embeds = llm.model.embed_tokens(input_ids)
    logger.info(f"  All embeds: {all_embeds.shape}")

    # Replace image placeholder positions with TTNN-projected image features
    image_mask = input_ids[0] == PLACEHOLDER_ID
    # Only replace the 576 image positions (not any real pad tokens in text)
    # Find the contiguous block of PLACEHOLDER_ID
    placeholder_indices = torch.where(image_mask)[0]
    if len(placeholder_indices) >= 576:
        # Replace exactly 576 positions
        replace_indices = placeholder_indices[:576]
        input_embeds = all_embeds.clone()
        input_embeds[0, replace_indices, :] = projected[0].to(input_embeds.dtype)
        n_replaced = 576
    else:
        input_embeds = all_embeds.clone()
        n_replaced = 0
        logger.warning(f"  Only {len(placeholder_indices)} placeholders found (expected 576)")

    logger.info(f"  Replaced {n_replaced} image positions with TTNN features")

    attention_mask = torch.ones(1, input_embeds.shape[1], dtype=torch.long)
    total_len = input_embeds.shape[1]

    logger.info(f"  Final embeds: {input_embeds.shape}")
    logger.info(f"  Total sequence length: {total_len} ({total_len - 576} text + 576 image)")

    # ============================================
    # Step 4: Generate text with LLaMA 3.1 LLM
    # ============================================
    logger.info("\n--- Step 4: Text Generation (LLaMA 3.1 8B, CPU) ---")

    # Cast to LLM dtype
    input_embeds = input_embeds.to(llm.dtype)

    logger.info(f"  Generating up to {max_new_tokens} tokens...")
    with torch.no_grad():
        output_ids = llm.generate(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode generated text (skip the input tokens)
    generated_tokens = output_ids[0][total_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    logger.info(f"  Generated {len(generated_tokens)} tokens")

    # ============================================
    # Result
    # ============================================
    logger.info("\n" + "=" * 80)
    logger.info("Multimodal Response (TTNN vision + trained LLaVA LLM):")
    logger.info("=" * 80)
    logger.info(generated_text.strip())
    logger.info("=" * 80)
    logger.info(f"\nPipeline Summary:")
    logger.info(f"  Image:       [1, 3, 336, 336]")
    logger.info(f"  CLIP (TTNN): [1, 576, 1024]   PCC: {pcc:.4f}")
    logger.info(f"  Projector:   [1, 576, {llm_hidden_size}]  (trained weights)")
    logger.info(f"  Interleaved: [1, {total_len}, {llm_hidden_size}]")
    logger.info(f"  Generated:   {len(generated_tokens)} tokens")
    logger.info("=" * 80)

    return generated_text.strip()


def run_full_demo_ttnn_llm(
    mesh_device,
    llava_state_dict,
    projector,
    tokenizer,
    llm_hidden_size,
    image_path=None,
    question="What is in this image?",
    max_new_tokens=64,
    llm_backbone="llama3.1",
):
    """
    Run LLaVA demo with ALL components on n300 using load/unload pattern.

    Thin wrapper around ``LlavaModel`` (see ``models/demos/llava/tt/llava_model.py``)
    which encapsulates the full pipeline:
        CLIP (TTNN) → Projector (TTNN) → Interleaver (host) → LLM (TTNN)

    Supports both backbones:
        - "llama3.1": LLaVA-MORE LLaMA 3.1 8B (hidden 4096, tensor-parallel)
        - "qwen-moe": MoE-LLaVA-Qwen-1.8B-4e (hidden 2048, expert-parallel MoE FFN)

    Memory strategy: CLIP and the LLM cannot coexist in n300 DRAM
    simultaneously, so ``LlavaModel`` runs CLIP+Projector first, then
    closes/reopens the device for the LLM. Because the model closes the
    initial ``mesh_device`` internally, the caller must NOT close it again
    — ``LlavaModel.close()`` handles final cleanup.

    Args:
        mesh_device: ttnn.MeshDevice (will be closed/reopened internally)
        llava_state_dict: Full LLaVA model state dict (projector + LLM weights)
        projector: Unused (kept for backward compat; weights come from llava_state_dict)
        tokenizer: HuggingFace tokenizer for the LLM
        llm_hidden_size: LLM hidden size (4096 llama3.1, 2048 qwen-moe)
        image_path: Path to image (None = sample image)
        question: Question about the image
        max_new_tokens: Max tokens to generate
        llm_backbone: "llama3.1" or "qwen-moe"

    Returns:
        Generated text
    """
    from models.demos.llava.tt.llava_model import LlavaModel

    logger.info("=" * 80)
    logger.info(f"LLaVA Full Demo - ALL ON n300 (load/unload pattern, {llm_backbone})")
    logger.info("=" * 80)

    if image_path is None:
        image_path = download_sample_image()

    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=1, llm_backbone=llm_backbone)
    weight_cache_path = (
        Path(os.environ.get("TT_METAL_CACHE_PATH", Path.home() / ".cache" / "tt-metal")) / "models" / "llava"
    )

    # LlavaModel owns the close/reopen lifecycle for the device.
    # Note: the mesh_device passed here will be closed inside _encode_vision()
    # and a fresh one opened for the LLM phase.
    model = LlavaModel(
        args=args,
        dtype=ttnn.bfloat16,
        llava_state_dict=llava_state_dict,
        tokenizer=tokenizer,
        weight_cache_path=weight_cache_path,
        llm_backbone=llm_backbone,
    )

    try:
        results, pcc_info = model.forward(
            images=[image_path],
            prompts=[question],
            max_new_tokens=max_new_tokens,
            return_pcc=True,
        )
        generated_text = results[0]
        clip_pcc = pcc_info.get("sample_0_clip_pcc")
    finally:
        model.cleanup()

    logger.info("\n" + "=" * 80)
    logger.info("Multimodal Response (ALL ON n300):")
    logger.info("=" * 80)
    logger.info(generated_text)
    logger.info("=" * 80)
    logger.info(f"\nPipeline Summary (ALL n300, load/unload pattern):")
    logger.info(f"  Phase 1: CLIP (n300) + Projector (n300) → image features")
    logger.info(f"  Phase 2: Prepare multimodal embeddings (host)")
    logger.info(f"  Phase 3: LLM (n300, {llm_backbone}) → text")
    if clip_pcc is not None:
        logger.info(f"  CLIP PCC: {clip_pcc:.4f}")
    logger.info("=" * 80)

    return generated_text


def run_full_demo_moe(
    mesh_device,
    image_path=None,
    question="What is in this image?",
    max_new_tokens=64,
    use_ttnn_llm=False,
):
    """
    Run the MoE-LLaVA-Qwen-1.8B-4e demo: image + prompt → text.

    Vision (CLIP + projector) always runs on TTNN hardware; the MoE LLM runs
    either on TTNN (--use-ttnn-llm; expert-parallel on n300) or on CPU via the
    DeepSpeed-stubbed PyTorch reference (default).

    TTNN path:      CLIP (TTNN) → Projector (TTNN, 1024→2048) → Interleave (host) → Qwen1-MoE LLM (TTNN)
    CPU path:       CLIP (TTNN) → Projector (TTNN) → Interleave (host) → MoE-LLaVA (CPU PyTorch)

    Args:
        mesh_device: ttnn.MeshDevice
        image_path: Path to image (None = sample image)
        question: Question about the image
        max_new_tokens: Max tokens to generate
        use_ttnn_llm: If True, run the MoE LLM on TTNN (n300); else CPU PyTorch

    Returns:
        Generated text
    """
    logger.info("=" * 80)
    logger.info("MoE-LLaVA-Qwen-1.8B-4e Demo - Image + Prompt → Text")
    logger.info("=" * 80)

    llava_state_dict, projector, tokenizer, llm_hidden_size, model_path = load_moe_llava_model()

    if use_ttnn_llm:
        # All-TTNN path — LlavaModel handles CLIP → Projector → Interleave → MoE LLM
        return run_full_demo_ttnn_llm(
            mesh_device=mesh_device,
            llava_state_dict=llava_state_dict,
            projector=projector,
            tokenizer=tokenizer,
            llm_hidden_size=llm_hidden_size,
            image_path=image_path,
            question=question,
            max_new_tokens=max_new_tokens,
            llm_backbone="qwen-moe",
        )

    # --- Hybrid path: TTNN vision + CPU MoE-LLaVA generation ---
    if image_path is None:
        image_path = download_sample_image()
    image_tensor = preprocess_image(image_path, resolution=336)

    logger.info(f"Image: {image_path}")
    logger.info(f"Question: {question}")

    # Step 1: TTNN CLIP vision encoder
    model_name = "openai/clip-vit-large-patch14-336"
    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=1, llm_backbone="qwen-moe")

    pytorch_encoder = create_clip_encoder_pytorch(model_name=model_name, device="cpu")
    state_dict = pytorch_encoder.vision_model.state_dict()

    weight_cache_path = (
        Path(os.environ.get("TT_METAL_CACHE_PATH", Path.home() / ".cache" / "tt-metal")) / "models" / "llava" / "clip"
    )
    weight_cache_path.mkdir(parents=True, exist_ok=True)

    ttnn_encoder = create_clip_encoder_ttnn(args=args, state_dict=state_dict, weight_cache_path=weight_cache_path)

    ttnn_features = ttnn_encoder(image_tensor)
    clip_features = ttnn.to_torch(ttnn.get_device_tensors(ttnn_features)[0])

    with torch.no_grad():
        pytorch_features = pytorch_encoder(image_tensor)
    passing, pcc = comp_pcc(pytorch_features, clip_features, pcc=0.88)
    logger.info(f"  CLIP output: {clip_features.shape}  (PCC: {pcc:.4f})")

    # Step 2: TTNN MLP projector (trained weights, 1024 → 2048 → 2048)
    from models.demos.llava.tt.projectors.mlp_projector import create_mlp_projector as create_mlp_projector_ttnn

    proj_sd = {
        "mm_projector.0.weight": projector.linear_1.weight.data,
        "mm_projector.0.bias": projector.linear_1.bias.data,
        "mm_projector.2.weight": projector.linear_2.weight.data,
        "mm_projector.2.bias": projector.linear_2.bias.data,
    }
    ttnn_projector = create_mlp_projector_ttnn(args=args, dtype=ttnn.bfloat16, state_dict=proj_sd)
    ttnn_projected = ttnn_projector(ttnn_features)
    projected = ttnn.to_torch(ttnn.get_device_tensors(ttnn_projected)[0])
    logger.info(f"  Projector output: {projected.shape}  (on device)")

    # Step 3: Tokenize with the MoE-LLaVA phi-style conversation template and interleave
    from models.demos.llava.tt.llava_model import build_moe_llava_prompt

    prompt_text = build_moe_llava_prompt(question)
    text_parts = prompt_text.split("<image>")
    ids_before = tokenizer(text_parts[0], return_tensors="pt", add_special_tokens=False).input_ids
    ids_after = tokenizer(text_parts[1], return_tensors="pt", add_special_tokens=False).input_ids

    PLACEHOLDER_ID = 0
    image_placeholders = torch.full((1, 576), PLACEHOLDER_ID, dtype=torch.long)
    input_ids = torch.cat([ids_before, image_placeholders, ids_after], dim=1)
    logger.info(f"  Prompt tokens: {input_ids.shape[1]} (includes 576 image placeholders)")

    # Step 4: Load MoE-LLaVA on CPU (DeepSpeed stubbed out — pure top-2 routing)
    from models.demos.llava.reference.qwen_moe_reference import load_model as load_moe_reference_model

    logger.info("Loading MoE-LLaVA-Qwen-1.8B-4e on CPU (DeepSpeed-stubbed reference)...")
    cpu_llm, moe_config = load_moe_reference_model("LanguageBind/MoE-LLaVA-Qwen-1.8B-4e", dtype=torch.float32)
    cpu_llm.eval()

    with torch.no_grad():
        all_embeds = cpu_llm.transformer.wte(input_ids)
    input_embeds = all_embeds.clone()
    placeholder_indices = torch.where(input_ids[0] == PLACEHOLDER_ID)[0][:576]
    input_embeds[0, placeholder_indices, :] = projected[0].to(input_embeds.dtype)

    attention_mask = torch.ones(1, input_embeds.shape[1], dtype=torch.long)
    total_len = input_embeds.shape[1]

    # Step 5: Generate on CPU
    logger.info(f"  Generating up to {max_new_tokens} tokens on CPU...")
    with torch.no_grad():
        output_ids = cpu_llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output_ids[0][total_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    logger.info("\n" + "=" * 80)
    logger.info("Multimodal Response (TTNN vision + CPU MoE-LLaVA):")
    logger.info("=" * 80)
    logger.info(generated_text.strip())
    logger.info("=" * 80)
    logger.info(f"\nPipeline Summary:")
    logger.info(f"  Image:       [1, 3, 336, 336]")
    logger.info(f"  CLIP (TTNN): [1, 576, 1024]   PCC: {pcc:.4f}")
    logger.info(f"  Projector:   [1, 576, {llm_hidden_size}]  (trained weights, on device)")
    logger.info(f"  Interleaved: [1, {total_len}, {llm_hidden_size}]")
    logger.info(f"  LLM:         Qwen1-MoE 1.8B-4e (CPU PyTorch, DeepSpeed stubbed)")
    logger.info(f"  Generated:   {len(generated_tokens)} tokens")
    logger.info("=" * 80)

    return generated_text.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLaVA Full Demo - Image + Prompt -> Text")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument("--question", type=str, default="What is in this image?", help="Question about image")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument(
        "--use-ttnn-llm",
        action="store_true",
        help="Use TTNN LLM on n300 (default: CPU LLM). CLIP+Projector always on n300.",
    )
    parser.add_argument(
        "--vision-encoder",
        type=str,
        default="clip",
        choices=["clip", "siglip"],
        help="Vision encoder type: 'clip' (CLIP ViT-L/14-336, default) or 'siglip' (SigLIP SO400M @ 384)",
    )
    parser.add_argument(
        "--llm-backbone",
        type=str,
        default="llama3.1",
        choices=["llama3.1", "qwen-moe"],
        help="LLM backbone: 'llama3.1' (LLaVA-MORE LLaMA 3.1 8B, default) or "
        "'qwen-moe' (MoE-LLaVA-Qwen-1.8B-4e; TTNN LLM with --use-ttnn-llm, else CPU PyTorch)",
    )
    args = parser.parse_args()

    if args.use_ttnn_llm and args.llm_backbone == "llama3.1":
        parser.error("--use-ttnn-llm is supported only with --llm-backbone qwen-moe; LLaMA TP is deprecated")

    # Determine device
    mesh_device_str = os.environ.get("MESH_DEVICE", "N300")
    if mesh_device_str == "N150":
        mesh_shape = (1, 1)
    else:
        mesh_shape = (1, 2)

    logger.info(f"Device: {mesh_device_str} (shape: {mesh_shape})")
    logger.info(f"LLM: {'TTNN (n300)' if args.use_ttnn_llm else 'CPU'}")
    logger.info(f"Vision encoder: {args.vision_encoder}")
    logger.info(f"LLM backbone: {args.llm_backbone}")

    # Open device. Multi-chip meshes run the LLM with CCL collectives
    # (tensor/expert parallelism), so fabric must be enabled before open.
    enable_fabric_if_needed(mesh_shape)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape))
    # The TTNN-LLM path closes/reopens the device internally (load/unload
    # pattern in LlavaModel); the CPU-LLM path leaves the device open.
    # Guard the outer close so we never double-close a device LlavaModel
    # already released on the TTNN path.
    device_open = True

    try:
        if args.llm_backbone == "qwen-moe":
            # MoE-LLaVA-Qwen-1.8B-4e path (TTNN vision; TTNN or CPU MoE LLM)
            result = run_full_demo_moe(
                mesh_device=mesh_device,
                image_path=args.image,
                question=args.question,
                max_new_tokens=args.max_tokens,
                use_ttnn_llm=args.use_ttnn_llm,
            )
            if args.use_ttnn_llm:
                # run_full_demo_moe → run_full_demo_ttnn_llm closed the device
                device_open = False
        elif args.use_ttnn_llm:
            # Load LLaVA-MORE model for weights
            llava_state_dict, projector, tokenizer, llm_hidden_size, model_path = load_llava_model()

            result = run_full_demo_ttnn_llm(
                mesh_device=mesh_device,
                llava_state_dict=llava_state_dict,
                projector=projector,
                tokenizer=tokenizer,
                llm_hidden_size=llm_hidden_size,
                image_path=args.image,
                question=args.question,
                max_new_tokens=args.max_tokens,
                llm_backbone="llama3.1",
            )
            # run_full_demo_ttnn_llm closed the device via LlavaModel
            device_open = False
        else:
            result = run_full_demo(
                mesh_device=mesh_device,
                image_path=args.image,
                question=args.question,
                max_new_tokens=args.max_tokens,
            )
    finally:
        if device_open:
            release_mesh_device(mesh_device)
