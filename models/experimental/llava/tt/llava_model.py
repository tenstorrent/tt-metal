# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LLaVA Model Orchestrator - Full Pipeline Implementation

Wires together all LLaVA components into a single reusable model:
    CLIP Vision Encoder → MLP Projector → Token Interleaver → LLM

Supports two LLM backbones:
    - llama3.1: LLaVA-MORE (LLaMA 3.1 8B, hidden 4096, tensor-parallel)
    - qwen-moe: MoE-LLaVA-Qwen-1.8B-4e (24 layers, 12 MoE w/ 4 experts top-2,
      hidden 2048, expert-parallel FFN + tensor-parallel attention)

Uses a load/unload pattern for n300 memory management: CLIP and the
LLM cannot coexist in DRAM simultaneously, so the vision pipeline runs
first, the device is closed/reopened, then the LLM loads for prefill+decode.

Architecture:
    Phase 1 (Vision):  Image → CLIP → Projector → [B, 576, llm_hidden] features on host
    Phase 2 (Host):    Tokenize prompt → embed → interleave image features
    Phase 3 (LLM):     Load LLM (TP across chips) → chunked prefill → decode

Reference:
    - models/demos/llava/demo/demo_llava_full.py (original inline implementation)
    - models/demos/llava/tt/llm_wrapper.py (LlavaLlmTransformer)
    - models/demos/qwen25_vl/tt/model.py (multimodal LLM pattern)
"""

import gc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import comp_pcc
from models.demos.llava.demo.demo_utils import preprocess_image
from models.demos.llava.reference.llava_reference import create_clip_encoder as create_clip_encoder_pytorch
from models.demos.llava.tt.model_config import VisionModelArgs
from models.demos.llava.tt.projectors.mlp_projector import create_mlp_projector as create_mlp_projector_ttnn
from models.demos.llava.tt.vision_encoders.clip_encoder import create_clip_encoder as create_clip_encoder_ttnn

IMAGE_TOKEN_INDEX = -200
NUM_PATCHES = 576
PLACEHOLDER_ID = 0  # Token ID used for image placeholder positions (overwritten with image features)


def mesh_shape_from_env() -> Tuple[int, int]:
    """Device shape follows MESH_DEVICE: N150 -> (1, 1), otherwise (1, 2) (n300)."""
    return (1, 1) if os.environ.get("MESH_DEVICE", "N300") == "N150" else (1, 2)


def enable_fabric_if_needed(mesh_shape: Tuple[int, int]) -> None:
    """Enable the 1D fabric before opening a multi-chip mesh.

    The MoE EP/TP path issues CCL collectives (tt_all_reduce in attention, MLP
    and MoE expert combine); these require the fabric router kernels, which are
    only started when a fabric config is set prior to open_mesh_device.
    """
    if mesh_shape[0] * mesh_shape[1] > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)


def disable_fabric_if_needed(mesh_shape: Tuple[int, int]) -> None:
    """Reset the fabric config after closing a multi-chip mesh."""
    if mesh_shape[0] * mesh_shape[1] > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def release_mesh_device(mesh_device: ttnn.MeshDevice) -> None:
    """Close a mesh device and reset the fabric config it was opened with."""
    mesh_shape = tuple(mesh_device.shape)
    ttnn.close_mesh_device(mesh_device)
    disable_fabric_if_needed(mesh_shape)


def build_moe_llava_prompt(question: str) -> str:
    """Build the MoE-LLaVA prompt with a <image> placeholder.

    Mirrors conv_templates["qwen"] (-> conv_phi, SeparatorStyle.TWO) from
    MoE-LLaVA/moellava/conversation.py + predict.py:
        system + " " + "USER: " + "<image>\\n{question}" + " " + "ASSISTANT:"
    """
    return (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        f"USER: <image>\n{question} ASSISTANT:"
    )


class LlavaModel(LightweightModule):
    """
    Full LLaVA model: image + prompt → generated text.

    Encapsulates the complete LLaVA pipeline on Tenstorrent n300/n150 hardware:
        CLIP encoder (TTNN) → MLP projector (TTNN) → token interleaving → LLaMA 3.1 LLM (TTNN)

    Uses a load/unload device pattern: CLIP+Projector run first, then the device
    is closed/reopened to free DRAM for the 32-layer LLaMA LLM (tensor-parallel).

    Args:
        args: VisionModelArgs with mesh_device and configuration
        dtype: ttnn data type for weights (ttnn.bfloat16 for vision, ttnn.bfloat8_b for LLM)
        llava_state_dict: Full LLaVA model state dict (contains mm_projector + LLaMA LLM weights)
        tokenizer: HuggingFace tokenizer for LLaMA 3.1 (used for prompt tokenization + decoding)
        weight_cache_path: Path for TTNN weight caching
        clip_model_name: HuggingFace model name for CLIP weights (default: openai/clip-vit-large-patch14-336)
        llm_max_seq_len: Maximum sequence length for LLM (default: 1024)
        llm_model_config: Config directory name in model_params/ (default: Llama-3.1-8B-Instruct)
        cpu_embed_model_name: HuggingFace model for CPU embedding lookup (default: unsloth/Llama-3.1-8B-Instruct)
        llm_dtype: ttnn data type for LLM weights (default: ttnn.bfloat8_b)

    Example:
        model = LlavaModel(args, ttnn.bfloat16, llava_state_dict, tokenizer, cache_path)
        output = model.forward([image], ["What is in this image?"])
        # → ["A cat sitting on a laptop."]
    """

    def __init__(
        self,
        args: VisionModelArgs,
        dtype: ttnn.DataType,
        llava_state_dict: Dict[str, Any],
        tokenizer,
        weight_cache_path: Path,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        llm_max_seq_len: int = 1024,
        llm_model_config: Optional[str] = None,
        cpu_embed_model_name: Optional[str] = None,
        llm_dtype: ttnn.DataType = ttnn.bfloat8_b,
        llm_backbone: Optional[str] = None,
    ):
        self.args = args
        self.mesh_device = args.mesh_device
        self.dtype = dtype
        self.llm_dtype = llm_dtype
        self.llava_state_dict = llava_state_dict
        self.tokenizer = tokenizer
        self.weight_cache_path = Path(weight_cache_path)
        self.clip_model_name = clip_model_name
        self.llm_max_seq_len = llm_max_seq_len
        # Backbone selection ("llama3.1" for LLaVA-MORE, "qwen-moe" for
        # MoE-LLaVA-Qwen-1.8B-4e). Defaults to args.llm_backbone so a
        # VisionModelArgs(llm_backbone="qwen-moe") configures everything.
        self.llm_backbone = llm_backbone or getattr(args, "llm_backbone", "llama3.1")
        assert self.llm_backbone in ("llama3.1", "qwen-moe"), f"Unknown LLM backbone: {self.llm_backbone}"
        if llm_model_config is None:
            llm_model_config = "Qwen1-1.8B-moe" if self.llm_backbone == "qwen-moe" else "Llama-3.1-8B-Instruct"
        if cpu_embed_model_name is None:
            cpu_embed_model_name = (
                "LanguageBind/MoE-LLaVA-Qwen-1.8B-4e"
                if self.llm_backbone == "qwen-moe"
                else "unsloth/Llama-3.1-8B-Instruct"
            )
        self.llm_model_config = llm_model_config
        self.cpu_embed_model_name = cpu_embed_model_name

        # LLM hidden size from args (LLaMA 3.1 8B = 4096, Qwen-1.8B = 2048)
        self.llm_hidden_size = args.projector_hidden_size

        # Extract projector state dict from llava_state_dict
        self.projector_sd = self._extract_projector_state_dict(llava_state_dict)

        # Lazy-loaded components
        self._cpu_embed_model = None

        logger.info(
            f"LlavaModel initialized: clip={clip_model_name}, llm={llm_model_config}, "
            f"backbone={self.llm_backbone}, hidden={self.llm_hidden_size}, max_seq={llm_max_seq_len}"
        )

    def _extract_projector_state_dict(self, llava_state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract MLP projector weights from the full LLaVA state dict.

        LLaVA-MORE (LLaMA 3.1) stores projector weights as:
            model.mm_projector.0.weight, model.mm_projector.0.bias
            model.mm_projector.2.weight, model.mm_projector.2.bias
        MoE-LLaVA (Qwen1) nests the projector inside the transformer under
        image_spatial_proj:
            transformer.mm_projector.image_spatial_proj.0.weight, ...
        """
        proj_sd = {}
        for prefix in (
            "model.mm_projector.",
            "transformer.mm_projector.image_spatial_proj.",
            "transformer.mm_projector.",
        ):
            for idx in ("0", "2"):
                for kind in ("weight", "bias"):
                    k = f"{prefix}{idx}.{kind}"
                    if k in llava_state_dict:
                        proj_sd[f"mm_projector.{idx}.{kind}"] = llava_state_dict[k]
        if len(proj_sd) < 4:
            raise KeyError(
                f"Could not find all projector weights in llava_state_dict. "
                f"Looked for model./transformer.mm_projector.{{0,2}}.{{weight,bias}}. "
                f"Found {list(proj_sd.keys())}."
            )
        return proj_sd

    def _load_cpu_embed_model(self):
        """Lazily load CPU embedding lookup (Phase 2).

        The TTNN LLM accepts pre-computed embeddings (not token IDs), so we need
        a CPU embedding table to convert tokenized text to embeddings before
        sending to the device. This model is only used for embed_tokens, not generation.

        - llama3.1: loads the full LLaMA 3.1 8B HF model (embed_tokens only used)
        - qwen-moe: builds a lightweight nn.Embedding directly from the
          MoE-LLaVA checkpoint's transformer.wte.weight — no extra download.
        """
        if self._cpu_embed_model is not None:
            return self._cpu_embed_model

        if self.llm_backbone == "qwen-moe":
            wte = self.llava_state_dict.get("transformer.wte.weight")
            if wte is None:
                raise KeyError("transformer.wte.weight not in llava_state_dict (MoE-LLaVA Qwen embedding)")
            import torch.nn as nn

            embedding = nn.Embedding(wte.shape[0], wte.shape[1], dtype=torch.float32)
            embedding.weight.data = wte.to(torch.float32)

            class _EmbedShim:
                """Exposes .model.embed_tokens like a HF causal LM."""

                class _Model:
                    pass

                def __init__(self, emb):
                    self.model = self._Model()
                    self.model.embed_tokens = emb

            self._cpu_embed_model = _EmbedShim(embedding)
            logger.info(f"Built Qwen embedding table from checkpoint: {tuple(wte.shape)}")
            return self._cpu_embed_model

        from transformers import AutoModelForCausalLM

        logger.info(f"Loading CPU LLaMA for embedding lookup: {self.cpu_embed_model_name}")
        self._cpu_embed_model = AutoModelForCausalLM.from_pretrained(
            self.cpu_embed_model_name, torch_dtype=torch.bfloat16
        )
        self._cpu_embed_model.eval()
        self._cpu_embed_model.requires_grad_(False)
        return self._cpu_embed_model

    def forward(
        self,
        images: List,
        prompts: List[str],
        max_new_tokens: int = 64,
        return_pcc: bool = False,
    ) -> List[str]:
        """
        Run full LLaVA inference: image + prompt → generated text.

        Executes the load/unload pattern internally:
            Phase 1: CLIP + Projector on device → vision features
            Phase 2: Host-side tokenization + embedding + interleaving
            Phase 3: LLM on device (tensor-parallel) → prefill + decode → text

        Args:
            images: List of PIL.Image or image file paths (one per prompt)
            prompts: List of question/instruction strings (one per image)
            max_new_tokens: Maximum tokens to generate per prompt
            return_pcc: If True, returns (texts, pcc_info) with CLIP PCC validation

        Returns:
            List[str]: Generated text for each (image, prompt) pair.
            If return_pcc=True, returns (List[str], dict) with PCC metrics.
        """
        assert len(images) == len(prompts), f"images ({len(images)}) and prompts ({len(prompts)}) must match"
        results = []
        pcc_info = {}

        for i, (image, prompt) in enumerate(zip(images, prompts)):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"LlavaModel.forward: sample {i+1}/{len(images)}")
            logger.info(f"{'=' * 80}")

            text, clip_pcc = self._forward_single(image, prompt, max_new_tokens)
            results.append(text)
            if clip_pcc is not None:
                pcc_info[f"sample_{i}_clip_pcc"] = clip_pcc

        if return_pcc:
            return results, pcc_info
        return results

    def _forward_single(self, image, prompt: str, max_new_tokens: int) -> Tuple[str, Optional[float]]:
        """Run inference for a single (image, prompt) pair.

        Device lifecycle per sample:
            - Vision phase uses the current self.mesh_device (from constructor for
              sample 0, freshly opened for samples 1+) and closes it at the end.
            - LLM phase opens a fresh device, runs prefill+decode, then closes it.
        """
        # Ensure we have a device for vision encoding. For sample 0 this is the
        # device passed in via args; for subsequent samples we open a fresh one.
        if self.mesh_device is None:
            self.mesh_device = self._open_device()
        self.args.mesh_device = self.mesh_device

        # Phase 1: Vision encoding (CLIP + Projector on device) — closes device
        projected_features, clip_pcc = self._encode_vision(image)
        self.mesh_device = None
        self.args.mesh_device = None

        # Phase 2: Prepare multimodal embeddings on host (no device needed)
        input_embeds, total_len, seq_len = self._prepare_multimodal_embeddings(projected_features, prompt)

        # Phase 3: LLM inference — open fresh device, prefill + decode, close
        self.mesh_device = self._open_device()
        self.args.mesh_device = self.mesh_device
        try:
            generated_text = self._run_llm(input_embeds, total_len, seq_len, max_new_tokens)
        finally:
            release_mesh_device(self.mesh_device)
            self.mesh_device = None
            self.args.mesh_device = None

        return generated_text, clip_pcc

    def _encode_vision(self, image) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Phase 1: Run CLIP encoder + MLP projector on device.

        Loads CLIP from HuggingFace, runs on TTNN hardware, projects through
        trained MLP projector, extracts features to host, then closes the device
        to free DRAM for the LLM.

        Args:
            image: PIL.Image or path to image file

        Returns:
            (projected_features, clip_pcc): [1, 576, 4096] torch tensor + optional PCC
        """
        logger.info("\n=== Phase 1: Vision Encoding (CLIP + Projector on device) ===")

        # Keep args.mesh_device in sync so the CLIP/projector builders pick up the
        # right device handle (they read args.mesh_device, not self.mesh_device).
        self.args.mesh_device = self.mesh_device

        # Preprocess image
        image_tensor = preprocess_image(image, resolution=self.args.vision_resolution)

        # Load CLIP (PyTorch reference for PCC + TTNN for inference)
        pytorch_encoder = create_clip_encoder_pytorch(model_name=self.clip_model_name, device="cpu")
        clip_state_dict = pytorch_encoder.vision_model.state_dict()

        clip_cache = self.weight_cache_path / "clip"
        clip_cache.mkdir(parents=True, exist_ok=True)

        ttnn_encoder = create_clip_encoder_ttnn(
            args=self.args, state_dict=clip_state_dict, weight_cache_path=clip_cache
        )

        # Run CLIP on TTNN
        ttnn_features = ttnn_encoder(image_tensor)
        clip_features = ttnn.to_torch(ttnn.get_device_tensors(ttnn_features)[0])

        # PCC check against PyTorch reference
        clip_pcc = None
        with torch.no_grad():
            pytorch_features = pytorch_encoder(image_tensor)
        passing, pcc = comp_pcc(pytorch_features, clip_features, pcc=0.88)
        clip_pcc = pcc
        logger.info(f"  CLIP output: {clip_features.shape}  (PCC: {pcc:.4f})")

        # Run MLP Projector on device (trained weights)
        logger.info("  Loading MLP Projector (trained weights) on device...")
        ttnn_projector = create_mlp_projector_ttnn(args=self.args, dtype=self.dtype, state_dict=self.projector_sd)
        ttnn_projected = ttnn_projector(ttnn_features)
        projected = ttnn.to_torch(ttnn.get_device_tensors(ttnn_projected)[0])
        logger.info(f"  Projector output: {projected.shape}")

        # Phase 1 cleanup: free all vision resources
        logger.info("  Deallocating CLIP + Projector (freeing DRAM for LLM)...")
        del ttnn_features, ttnn_projected, ttnn_encoder, ttnn_projector, pytorch_encoder, clip_state_dict
        gc.collect()

        # Close device to guarantee all CLIP memory is freed
        release_mesh_device(self.mesh_device)
        logger.info("  Device closed (CLIP memory freed)")

        return projected, clip_pcc

    def _prepare_multimodal_embeddings(
        self, projected_features: torch.Tensor, prompt: str
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Phase 2: Prepare multimodal embeddings on host.

        Tokenizes the prompt using LLaMA 3.1 chat template, replaces <image>
        placeholder positions with projected vision features, and pads to
        tile-aligned sequence length.

        Args:
            projected_features: [1, 576, 4096] vision features from Phase 1
            prompt: User question/instruction text

        Returns:
            (input_embeds_padded, total_len, seq_len): Padded embeddings + lengths
        """
        logger.info("\n=== Phase 2: Prepare Multimodal Embeddings (host) ===")

        if self.llm_backbone == "qwen-moe":
            # MoE-LLaVA uses the phi-style conversation template
            # (conv_templates["qwen"] -> conv_phi in MoE-LLaVA/moellava/conversation.py)
            text = build_moe_llava_prompt(prompt)
        else:
            # Build LLaMA 3.1 chat template with <image> placeholder
            messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Split around <image> and tokenize text parts
        text_parts = text.split("<image>")
        ids_before = self.tokenizer(text_parts[0], return_tensors="pt", add_special_tokens=False).input_ids
        ids_after = self.tokenizer(text_parts[1], return_tensors="pt", add_special_tokens=False).input_ids

        # Create 576 placeholder tokens for image positions
        image_placeholders = torch.full((1, NUM_PATCHES), PLACEHOLDER_ID, dtype=torch.long)
        input_ids = torch.cat([ids_before, image_placeholders, ids_after], dim=1)

        # Get text embeddings from CPU LLaMA embedding table
        cpu_llm = self._load_cpu_embed_model()
        with torch.no_grad():
            all_embeds = cpu_llm.model.embed_tokens(input_ids)

        # Replace image placeholder positions with projected features
        image_mask = input_ids[0] == PLACEHOLDER_ID
        placeholder_indices = torch.where(image_mask)[0][:NUM_PATCHES]
        input_embeds = all_embeds.clone()
        input_embeds[0, placeholder_indices, :] = projected_features[0].to(input_embeds.dtype)

        total_len = input_embeds.shape[1]
        logger.info(f"  Multimodal embeds: {input_embeds.shape} ({total_len - NUM_PATCHES} text + {NUM_PATCHES} image)")

        # Pad to next multiple of 32 (tile size)
        pad_to = ((total_len + 31) // 32) * 32
        pad_len = pad_to - total_len
        if pad_len > 0:
            pad_embeds = torch.zeros(1, pad_len, self.llm_hidden_size, dtype=input_embeds.dtype)
            input_embeds_padded = torch.cat([input_embeds, pad_embeds], dim=1)
        else:
            input_embeds_padded = input_embeds
        seq_len = input_embeds_padded.shape[1]
        logger.info(f"  Padded embeds: {input_embeds_padded.shape} (padded to {seq_len})")

        return input_embeds_padded, total_len, seq_len

    def _open_device(self) -> ttnn.MeshDevice:
        """Open a fresh mesh device matching the current MESH_DEVICE env."""
        mesh_shape = mesh_shape_from_env()
        enable_fabric_if_needed(mesh_shape)
        new_device = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape))
        logger.info(f"  Device opened (shape: {mesh_shape})")
        return new_device

    def _run_llm(self, input_embeds: torch.Tensor, total_len: int, seq_len: int, max_new_tokens: int) -> str:
        """
        Phase 3: Run the LLM on device (tensor-parallel; MoE adds expert-parallel FFN).

        Loads the TTNN LLM with tensor parallelism across n300 chips, runs
        chunked prefill (128 tokens per chunk), then autoregressive decode.

        Args:
            input_embeds: [1, seq_len, hidden] padded multimodal embeddings
            total_len: Actual (unpadded) sequence length
            seq_len: Padded sequence length (multiple of 32)
            max_new_tokens: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        if self.llm_backbone == "qwen-moe":
            from models.demos.llava.tt.llm_wrapper import create_qwen_moe_llm as create_llm
        else:
            from models.demos.llava.tt.llm_wrapper import create_llava_llm as create_llm

        logger.info(f"\n=== Phase 3: LLM Inference ({self.llm_model_config} on device) ===")

        # Load TTNN LLM (tensor-parallel across n300; MoE FFN expert-parallel)
        logger.info("  Loading TTNN LLM...")
        tt_model, tt_model_args, tt_kv_cache, page_table = create_llm(
            self.mesh_device,
            self.llava_state_dict,
            dtype=self.llm_dtype,
            max_seq_len=self.llm_max_seq_len,
            model_config=self.llm_model_config,
        )
        logger.info(
            f"  TTNN LLM: {tt_model_args.n_layers} layers, dim={tt_model_args.dim}, "
            f"n_kv_heads={tt_model_args.n_kv_heads}"
        )

        # === Chunked Prefill (128 tokens per chunk) ===
        logger.info("\n  --- Chunked Prefill ---")
        chunk_size = 128
        last_token_idx = total_len - 1
        total_chunks = (seq_len + chunk_size - 1) // chunk_size
        logger.info(f"  seq_len={seq_len}, chunk_size={chunk_size}, chunks={total_chunks}, last_token={last_token_idx}")

        logits = None
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_embeds = input_embeds[:, chunk_start:chunk_end, :]

            # Slice page table for this chunk's blocks
            block_start = chunk_start // 32
            block_end = (chunk_end + 31) // 32
            chunk_pt = page_table[:, block_start:block_end]

            # Pre-computed multimodal embeds: bypass embd(tokens) and mirror
            # prepare_inputs_prefill's device-side pieces (embed dtype/layout,
            # per-chunk RoPE slice, replicated page tables) directly.
            # tt_transformers' prepare_inputs_prefill only accepts token ids.
            x = ttnn.from_torch(
                chunk_embeds,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            x = ttnn.unsqueeze_to_4D(x)

            mat_len = tt_model.rope_setup.cos_matrix_prefill.shape[2]
            slice_end = min(mat_len, chunk_start + (chunk_end - chunk_start))
            cos_slice = tt_model.rope_setup.cos_matrix_prefill[:, :, chunk_start:slice_end, :]
            sin_slice = tt_model.rope_setup.sin_matrix_prefill[:, :, chunk_start:slice_end, :]
            rot_mats_global = [cos_slice, sin_slice]

            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            tt_chunk_page_table = ttnn.from_torch(
                chunk_pt,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            tt_chunk_start_idx = None
            if chunk_start > 0:
                tt_chunk_start_idx = ttnn.from_torch(
                    torch.tensor([chunk_start], dtype=torch.int32),
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )

            # Only extract logits from the last chunk
            is_last_chunk = chunk_end >= seq_len
            if is_last_chunk:
                last_token_in_chunk = last_token_idx - chunk_start
                get_last_token = (last_token_in_chunk // 32) * 32
            else:
                get_last_token = -1

            chunk_num = chunk_start // chunk_size + 1
            logger.info(
                f"  Chunk {chunk_num}/{total_chunks}: [{chunk_start}:{chunk_end}] get_last_token={get_last_token}"
            )

            tt_out = tt_model.ttnn_prefill_forward(
                x,
                rot_mats_global=rot_mats_global,
                page_table=tt_page_table,
                chunk_page_table=tt_chunk_page_table,
                chunk_start_idx=tt_chunk_start_idx,
                get_last_token=get_last_token,
                kv_cache=tt_kv_cache,
                batch_size=1,
            )

            if is_last_chunk:
                logits = tt_model.process_output_prefill(tt_out.cpu(), last_token_idx=last_token_idx % 32)
            else:
                ttnn.deallocate(tt_out, force=True)

        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
        logger.info(f"  First token: {next_token.item()} ({self.tokenizer.decode(next_token)})")

        # === Decode (autoregressive generation) ===
        logger.info("\n  --- Decode ---")
        generated_tokens = [next_token.item()]
        current_pos = torch.tensor([total_len], dtype=torch.int32)

        for i in range(max_new_tokens - 1):
            # Prepare decode inputs (handles embedding lookup, RoPE, page table)
            tt_tokens, tt_current_pos, tt_rot_mat_idxs, tt_pt_decode = tt_model.prepare_inputs_decode(
                next_token,
                current_pos,
                page_table,
            )

            # Run decode forward
            tt_logits, _ = tt_model.ttnn_decode_forward(
                tt_tokens,
                tt_current_pos,
                rot_mat_idxs=tt_rot_mat_idxs,
                page_table=tt_pt_decode,
                kv_cache=tt_kv_cache,
            )

            # Process output
            decode_logits = tt_model.process_output_decode(tt_logits, B=1)
            next_token = torch.argmax(decode_logits[:, -1, :], dim=-1)
            generated_tokens.append(next_token.item())

            if next_token.item() in [self.tokenizer.eos_token_id, 2]:
                break

            current_pos += 1
            if (i + 1) % 10 == 0:
                logger.info(f"  Generated {i+2} tokens...")

        generated_text = self.tokenizer.decode(torch.tensor(generated_tokens), skip_special_tokens=True)

        logger.info(f"\n  Generated {len(generated_tokens)} tokens")
        logger.info(f"  Response: {generated_text.strip()[:200]}")

        # Cleanup LLM
        del tt_model, tt_kv_cache, page_table
        gc.collect()

        return generated_text.strip()

    def cleanup(self):
        """Release all resources (CPU embed model, device, etc.)."""
        if self.mesh_device is not None:
            try:
                release_mesh_device(self.mesh_device)
            except Exception:
                pass
            self.mesh_device = None
            self.args.mesh_device = None
        if self._cpu_embed_model is not None:
            del self._cpu_embed_model
            self._cpu_embed_model = None
        gc.collect()


def create_llava_model(
    args: VisionModelArgs,
    dtype: ttnn.DataType,
    llava_state_dict: Dict[str, Any],
    tokenizer,
    weight_cache_path: Path,
    **kwargs,
) -> LlavaModel:
    """
    Factory function to create a LlavaModel orchestrator.

    Args:
        args: VisionModelArgs with mesh_device and configuration
        dtype: ttnn data type for vision weights
        llava_state_dict: Full LLaVA model state dict (projector + LLM weights)
        tokenizer: HuggingFace tokenizer for LLaMA 3.1
        weight_cache_path: Path for TTNN weight caching
        **kwargs: Additional arguments passed to LlavaModel constructor

    Returns:
        LlavaModel instance
    """
    return LlavaModel(args, dtype, llava_state_dict, tokenizer, weight_cache_path, **kwargs)
