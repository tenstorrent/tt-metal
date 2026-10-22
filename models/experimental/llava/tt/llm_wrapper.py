# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
LLaVA LLM Wrapper - TTNN Implementation

Subclasses tt_transformers.Transformer to accept pre-computed multimodal
embeddings instead of token IDs. This enables the LLaVA pipeline:
    CLIP (TTNN) → Projector (TTNN) → Interleaver → LLM (TTNN)

The override pattern follows models/demos/qwen25_vl/tt/model.py, which
bypasses the embedding lookup in prepare_inputs_prefill and feeds
pre-embedded tensors directly to the transformer blocks.

Reference:
    - models/demos/qwen25_vl/tt/model.py (Transformer subclass pattern)
    - models/tt_transformers/tt/model.py (base Transformer)
    - models/tt_transformers/tt/load_checkpoints.py (weight conversion)
"""

from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta, standardize_hf_keys
from models.tt_transformers.tt.model import Transformer as TTTransformer


class LlavaLlmTransformer(TTTransformer):
    """
    LLaVA LLM Transformer - accepts pre-computed multimodal embeddings.

    Subclasses the base tt_transformers Transformer and overrides
    prepare_inputs_prefill to accept embedding tensors [B, seq_len, hidden_dim]
    instead of token IDs. This is the Qwen2.5-VL pattern.

    The base Transformer.forward() already accepts embedded tensors — the
    embedding lookup (self.embd) is only called in prepare_inputs_prefill,
    so overriding this method is sufficient.
    """

    def prepare_inputs_prefill(self, tokens, rot_mats=None, start_pos=0, page_table=None, chunk_page_table=None):
        """
        Prepare prefill inputs from pre-computed embeddings.

        Args:
            tokens: torch.Tensor [B, seq_len, hidden_dim] — pre-computed embeddings
            rot_mats: Optional [(cos, sin)] torch tensors. If None, uses model's rope_setup.
            start_pos: Start position for RoPE
            page_table: Optional page table for paged attention
            chunk_page_table: Optional chunk page table

        Returns:
            (tokens_embd, tt_rot_mats, tt_page_table, tt_chunk_page_table)
        """
        assert tokens.dim() == 3, f"Expected 3D tensor [B, seq_len, hidden_dim], got {tokens.dim()}D"
        S = tokens.shape[-2]

        # Convert embeddings to ttnn tensor on device
        # Shape: [B, seq_len, hidden_dim] → [B, 1, seq_len, hidden_dim] for ttnn
        # Use ShardTensor2dMesh for tensor parallelism (shard hidden_dim across chips)
        # This matches the weight sharding in the base Transformer class
        tokens_embd = ttnn.from_torch(
            tokens.unsqueeze(1),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=self.mesh_device,
                dims=(None, 3),
                mesh_shape=self.args.cluster_shape,
            ),
        )

        # Get RoPE matrices — RoPE is replicated (not sharded) since it operates
        # per-head, and each chip has its own subset of heads
        if rot_mats is not None:
            # Use provided rot_mats (multimodal RoPE pattern)
            cos_matrix = ttnn.from_torch(
                rot_mats[0].expand(rot_mats[0].shape[0], -1, -1, -1),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.rope_setup.datatype,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            sin_matrix = ttnn.from_torch(
                rot_mats[1].expand(rot_mats[1].shape[0], -1, -1, -1),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=self.rope_setup.datatype,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            tt_rot_mats = [
                cos_matrix[:, :, start_pos : start_pos + S, :],
                sin_matrix[:, :, start_pos : start_pos + S, :],
            ]
        else:
            # Use model's built-in RoPE (standard 1D positions)
            slice_end = start_pos + S
            tt_rot_mats = [
                self.rope_setup.cos_matrix_prefill[:, :, start_pos:slice_end, :],
                self.rope_setup.sin_matrix_prefill[:, :, start_pos:slice_end, :],
            ]

        # Convert page table
        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return tokens_embd, tt_rot_mats, tt_page_table, tt_chunk_page_table


def extract_llama_state_dict_from_llava(llava_state_dict, n_heads=32, n_kv_heads=8, head_dim=128):
    """
    Extract LLaMA text model weights from LLaVA model state dict and convert to Meta format.

    Supports both LLaVA 1.5 (Vicuna/LLaMA-2) and LLaVA-MORE (LLaMA 3.1).
    The key format is the same: model.layers.*, model.embed_tokens, lm_head, model.norm.

    Args:
        llava_state_dict: State dict from LLaVA model (or safetensors)
        n_heads: Number of attention heads (32 for both LLaMA-2 and 3.1)
        n_kv_heads: Number of KV heads (32 for LLaMA-2, 8 for LLaMA 3.1)
        head_dim: Head dimension (128 for both)

    Returns:
        state_dict in Meta format (keys: tok_embeddings, layers.*, output, etc.)
    """
    # Extract only LLaMA text model keys (skip vision_tower and mm_projector)
    llama_sd = {}
    for k, v in llava_state_dict.items():
        # Skip vision tower and projector weights
        if "vision_tower" in k or "mm_projector" in k:
            continue
        # LLaMA text model keys start with "model.layers.", "model.embed_tokens", "model.norm", or "lm_head"
        if (
            k.startswith("model.layers.")
            or k.startswith("model.embed_tokens")
            or k.startswith("model.norm")
            or k == "lm_head.weight"
        ):
            llama_sd[k] = v

    logger.info(f"Extracted {len(llama_sd)} LLaMA weights from LLaVA model")

    # Convert to Meta format
    # standardize_hf_keys: maps model.embed_tokens → lm_head, etc.
    llama_sd = standardize_hf_keys(llama_sd)

    # convert_hf_to_meta: splits QKV, permutes, maps key names
    llama_sd = convert_hf_to_meta(llama_sd, head_dim, n_heads, n_kv_heads)

    logger.info(f"Converted to Meta format: {len(llama_sd)} weights")
    logger.info(f"Sample keys: {list(llama_sd.keys())[:5]}")

    return llama_sd


def create_llava_llm(
    mesh_device, llava_state_dict, dtype=ttnn.bfloat8_b, max_seq_len=1024, model_config="Llama-3.1-8B-Instruct"
):
    """
    Create a TTNN LLaMA Transformer on n300 from LLaVA model weights.

    Uses tensor parallelism (ShardTensor2dMesh) to shard the LLM across both
    n300 chips, allowing the full 32-layer model to fit alongside CLIP.

    Args:
        mesh_device: ttnn.MeshDevice
        llava_state_dict: State dict from LLaVA model (LLaVA 1.5 or LLaVA-MORE)
        dtype: ttnn data type for weights
        max_seq_len: Maximum sequence length
        model_config: Config directory name in model_params/ (default: LLaMA 3.1)

    Returns:
        (model, model_args, tt_kv_cache, page_table)
    """
    import os

    from models.tt_transformers.tt.common import PagedAttentionConfig
    from models.tt_transformers.tt.model_config import ModelArgs

    # Set HF_MODEL to the LLaMA config directory
    llava_config_dir = str(
        Path(__file__).resolve().parents[4] / "models" / "tt_transformers" / "model_params" / model_config
    )
    os.environ["HF_MODEL"] = llava_config_dir

    logger.info(f"Creating LLaVA LLM on n300 (config: {model_config})")

    # Create ModelArgs — loads config from our config.json
    model_args = ModelArgs(
        mesh_device=mesh_device,
        instruct=True,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )

    # Fix rope_theta if not loaded (newer transformers nests it in rope_parameters)
    if model_args.rope_theta is None:
        if "Llama-3.1" in model_config:
            model_args.rope_theta = 500000.0  # LLaMA 3.1
        else:
            model_args.rope_theta = 10000.0  # LLaMA-2
        logger.info(f"Set rope_theta to {model_args.rope_theta}")

    # Extract and convert LLaMA weights
    state_dict = extract_llama_state_dict_from_llava(
        llava_state_dict,
        n_heads=model_args.n_heads,
        n_kv_heads=model_args.n_kv_heads,
        head_dim=model_args.head_dim,
    )

    # Create paged attention config
    paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)

    # Create weight cache path
    weight_cache_path = model_args.weight_cache_path(dtype)

    # Create the Transformer (our subclass with tensor parallelism)
    model = LlavaLlmTransformer(
        args=model_args,
        dtype=dtype,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=paged_attention_config,
    )

    # Get KV cache
    tt_kv_cache = [l.attention.layer_past for l in model.layers]

    # Create page table
    page_table = torch.arange(0, paged_attention_config.max_num_blocks, dtype=torch.int32).unsqueeze(0)

    logger.info(
        f"LLaVA LLM created: {model_args.n_layers} layers, dim={model_args.dim}, n_kv_heads={model_args.n_kv_heads}"
    )
    logger.info(f"Tensor parallelism: cluster_shape={model_args.cluster_shape}")

    return model, model_args, tt_kv_cache, page_table


def extract_qwen_moe_state_dict_from_llava(llava_state_dict, moe_layer_ids, n_layers=24, num_heads=16, head_dim=128):
    """
    Extract the Qwen1 LLM weights from a MoE-LLaVA-Qwen-1.8B-4e state dict and
    convert to tt-metal Meta format.

    The MoE-LLaVA checkpoint nests the LLM under `transformer.*` (Qwen1 naming,
    NOT LLaVA's `model.*`): transformer.h.{l}.attn.c_attn.weight (fused QKV),
    transformer.h.{l}.mlp.{wg,experts.{e}.{w1,w2,c_proj}} for MoE layers,
    transformer.h.{l}.mlp.{w1,w2,c_proj} for dense layers, transformer.wte,
    transformer.ln_f, lm_head. The vision tower (transformer.image_tower.*) and
    projector (transformer.mm_projector.*) are skipped here — they are consumed
    by the CLIP encoder and MLP projector modules directly.

    Args:
        llava_state_dict: Full state dict from LanguageBind/MoE-LLaVA-Qwen-1.8B-4e
        moe_layer_ids: Layer indices with MoE FFN (even indices for -4e)
        n_layers: Number of transformer layers (24)
        num_heads: Attention heads (16, MHA — n_kv_heads == n_heads)
        head_dim: Head dimension (128)

    Returns:
        state_dict in Meta format (layers.*, embed_tokens, norm, lm_head)
    """
    from models.demos.llava.tt.qwen1_moe_checkpoint import convert_qwen_moe_to_meta

    # Keep only the LLM keys (skip vision tower + projector), then hand a copy
    # to the converter (it consumes keys via pop()).
    llm_sd = {
        k: v
        for k, v in llava_state_dict.items()
        if not (k.startswith("transformer.image_tower.") or k.startswith("transformer.mm_projector."))
    }
    missing = [k for k in (f"transformer.h.{l}.attn.c_attn.weight" for l in range(n_layers)) if k not in llm_sd]
    if missing:
        raise KeyError(
            f"State dict does not look like MoE-LLaVA-Qwen (missing {missing[:3]}). "
            f"Sample keys: {list(llava_state_dict.keys())[:5]}"
        )

    meta_sd, leftover = convert_qwen_moe_to_meta(
        dict(llm_sd),
        n_layers=n_layers,
        moe_layer_ids=moe_layer_ids,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    # Audit leftovers: only non-persistent buffers (e.g. rotary inv_freq) and
    # layers beyond the requested layer count (truncated PCC tests) may remain.
    def _allowed_leftover(key: str) -> bool:
        if "inv_freq" in key or "masked_bias" in key:
            return True
        if key.startswith("transformer.h."):
            try:
                return int(key.split(".")[2]) >= n_layers
            except (IndexError, ValueError):
                return False
        return False

    unexpected = [k for k in leftover if not _allowed_leftover(k)]
    if unexpected:
        raise KeyError(f"Unconverted MoE-LLaVA LLM keys: {unexpected[:10]}")

    n_moe = sum(1 for l in moe_layer_ids if f"layers.{l}.block_sparse_moe.gate.weight" in meta_sd)
    logger.info(
        f"Converted MoE-LLaVA Qwen LLM: {len(meta_sd)} weights, {n_moe} MoE layers "
        f"(of {n_layers}), {sum(1 for k in meta_sd if 'feed_forward' in k)} dense-FFN weights"
    )
    return meta_sd


def create_qwen_moe_llm(
    mesh_device,
    moe_llava_state_dict,
    dtype=ttnn.bfloat8_b,
    max_seq_len=1024,
    model_config="Qwen1-1.8B-moe",
    n_layers=None,
):
    """
    Create a TTNN Qwen1-MoE Transformer on n300 from MoE-LLaVA-Qwen-1.8B-4e weights.

    The MoE FFN is expert-parallel across the 2 n300 chips (2 experts per chip,
    gate columns swapped so each chip's top-k sees its local experts first,
    followed by a 2-chip all-reduce) — see models/tt_transformers/tt/mixtral_moe.py.
    Attention/hidden stay tensor-parallel via ShardTensor2dMesh like the LLaMA path.

    Args:
        mesh_device: ttnn.MeshDevice
        moe_llava_state_dict: State dict from LanguageBind/MoE-LLaVA-Qwen-1.8B-4e
        dtype: ttnn data type for weights
        max_seq_len: Maximum sequence length
        model_config: Config directory name in model_params/
        n_layers: Optional layer-count override for truncated PCC tests
            (moe_layer_ids are truncated to layers < n_layers)

    Returns:
        (model, model_args, tt_kv_cache, page_table)
    """
    import os

    from models.tt_transformers.tt.common import PagedAttentionConfig
    from models.tt_transformers.tt.model_config import ModelArgs

    # Set HF_MODEL to the Qwen1-MoE config directory
    llava_config_dir = str(
        Path(__file__).resolve().parents[4] / "models" / "tt_transformers" / "model_params" / model_config
    )
    os.environ["HF_MODEL"] = llava_config_dir

    logger.info(f"Creating MoE-LLaVA Qwen LLM on n300 (config: {model_config})")

    # Create ModelArgs — reads MoE fields (num_experts, moe_layer_ids,
    # use_hf_rope) from our Qwen1-1.8B-moe config.json
    model_args = ModelArgs(
        mesh_device=mesh_device,
        instruct=True,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )
    if n_layers is not None and n_layers != model_args.n_layers:
        model_args.n_layers = n_layers
        model_args.moe_layer_ids = [l for l in model_args.moe_layer_ids if l < n_layers]

    # Verify the MoE config was loaded correctly
    assert model_args.is_mixture_of_experts, "Qwen1-1.8B-moe config missing MoE fields"
    assert model_args.num_experts == 4, f"Expected 4 experts, got {model_args.num_experts}"
    assert model_args.moe_layer_ids == [
        l for l in range(0, model_args.n_layers, 2)
    ], f"Expected even MoE layers, got {model_args.moe_layer_ids}"
    assert model_args.hidden_dim == 5504, f"Expected Qwen1 half-intermediate 5504, got {model_args.hidden_dim}"

    # Extract and convert the Qwen1 MoE weights to Meta format
    state_dict = extract_qwen_moe_state_dict_from_llava(
        moe_llava_state_dict,
        moe_layer_ids=model_args.moe_layer_ids,
        n_layers=model_args.n_layers,
        num_heads=model_args.n_heads,
        head_dim=model_args.head_dim,
    )

    # Create paged attention config
    paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)

    # Create the Transformer (LlavaLlmTransformer handles pre-embedded inputs)
    weight_cache_path = model_args.weight_cache_path(dtype)
    model = LlavaLlmTransformer(
        args=model_args,
        dtype=dtype,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=paged_attention_config,
    )

    # Sanity-check the per-layer MoE/dense split actually built
    from models.tt_transformers.tt.mixtral_moe import TtMoeLayer

    moe_layers = [l for l in model.layers if isinstance(l.feed_forward, TtMoeLayer)]
    assert len(moe_layers) == len(
        model_args.moe_layer_ids
    ), f"Expected {len(model_args.moe_layer_ids)} MoE layers, built {len(moe_layers)}"

    tt_kv_cache = [l.attention.layer_past for l in model.layers]
    page_table = torch.arange(0, paged_attention_config.max_num_blocks, dtype=torch.int32).unsqueeze(0)

    logger.info(
        f"MoE-LLaVA Qwen LLM created: {model_args.n_layers} layers "
        f"({len(moe_layers)} MoE), dim={model_args.dim}, hidden={model_args.hidden_dim}"
    )
    logger.info(f"Expert parallelism: {model_args.num_experts} experts / {model_args.num_devices} chips")

    return model, model_args, tt_kv_cache, page_table
