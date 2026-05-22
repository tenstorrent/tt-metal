# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5-27B Transformer model.

Thin subclass of the framework Transformer that wires up:
- Qwen35PartialRopeSetup for partial RoPE (64 of 256 dims)
- Per-layer attention dispatch (full_attention -> Qwen35Attention, linear_attention -> TtGatedDeltaNet)
- Post-construction attention weight loading via set_weights()

The framework handles: embedding, RMSNorm (with offset), MLP, LM head, decoder blocks.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.attention import Qwen35Attention
from models.demos.qwen35_27b.tt.gdn import TtGatedDeltaNet
from models.demos.qwen35_27b.tt.model_config import ATTN_CHUNK_SIZE, GDN_CHUNK_SIZE, Qwen35ModelArgs
from models.demos.qwen35_27b.tt.rope import Qwen35PartialRopeSetup
from models.tt_transformers.tt.model import Transformer as TTTransformer
from models.tt_transformers.tt.model_config import Mode


class Transformer(TTTransformer):
    """Qwen3.5-27B Transformer.

    Uses the framework Transformer with:
    - Qwen35PartialRopeSetup for partial RoPE
    - Qwen35Attention for full_attention layers
    - TtGatedDeltaNet for linear_attention layers
    """

    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher=None,
    ):
        # Build with Qwen35Attention as default, then swap GDN layers after

        def attention_wrapper(**kwargs):
            if args.layer_types[kwargs["layer_num"]] == "linear_attention":
                return TtGatedDeltaNet(**kwargs)
            else:
                return Qwen35Attention(**kwargs)

        super().__init__(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            attention_class=attention_wrapper,
            rope_setup_class=Qwen35PartialRopeSetup,
            prefetcher=prefetcher,
        )

    # ── L1 State Management ────────────────────────────────────────────

    def enable_l1_state(self):
        """Enable L1 INTERLEAVED state for GDN layers with rolling window.

        Loads first 3 GDN layers' rec_states to L1. During forward(), swaps
        groups of 3 around attention layers (pattern: 3 GDN + 1 ATTN repeating).
        """
        self._l1_state_enabled = True
        self._l1_window = 3  # matches layer pattern: 3 GDN + 1 ATTN

        # Build GDN layer index list
        self._gdn_indices = [i for i in range(self.args.n_layers) if self.args.layer_types[i] == "linear_attention"]

        # Save DRAM backup refs and load first window to L1
        for i, idx in enumerate(self._gdn_indices):
            gdn = self.layers[idx].attention
            if gdn.rec_states is None:
                gdn.reset_state()
            gdn._dram_state = gdn.rec_states  # keep DRAM ref

        # Load first 3 GDN layers to L1
        for idx in self._gdn_indices[: self._l1_window]:
            gdn = self.layers[idx].attention
            l1_state = ttnn.to_memory_config(gdn._dram_state, ttnn.L1_MEMORY_CONFIG)
            gdn.rec_states = l1_state

        self._l1_current_start = 0  # which GDN group is in L1 (0-based GDN index)
        logger.info(f"L1 state enabled: {len(self._gdn_indices)} GDN layers, window={self._l1_window}")

    def _swap_l1_state(self, old_start, new_start):
        """Swap GDN state: save current L1 group to DRAM, load new group from DRAM to L1.

        Uses pre-allocated DRAM buffers (_dram_state) to avoid creating new tensors.
        L1 buffers are deallocated and re-allocated for new group.
        """
        W = self._l1_window

        # Save old group: L1 → DRAM (copy back to pre-allocated DRAM buffer)
        for j in range(W):
            gi = old_start + j
            if gi >= len(self._gdn_indices):
                break
            gdn = self.layers[self._gdn_indices[gi]].attention
            if gdn.rec_states.memory_config().buffer_type == ttnn.BufferType.L1:
                # Copy L1 data back to the DRAM backup
                ttnn.to_memory_config(gdn.rec_states, ttnn.DRAM_MEMORY_CONFIG, output_tensor=gdn._dram_state)
                ttnn.deallocate(gdn.rec_states)
                gdn.rec_states = gdn._dram_state

        # Load new group: DRAM → L1
        for j in range(W):
            gi = new_start + j
            if gi >= len(self._gdn_indices):
                break
            gdn = self.layers[self._gdn_indices[gi]].attention
            l1_state = ttnn.to_memory_config(gdn._dram_state, ttnn.L1_MEMORY_CONFIG)
            gdn.rec_states = l1_state

    def forward(
        self,
        x,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    ):
        """Forward with rolling-window L1 state for GDN layers.

        Only intercepts the layer loop to add L1 swaps. All other logic
        (embedding, norm, LM head) delegated to parent.
        """
        if mode is None:
            mode = Mode.DECODE

        logger.info(f"Is l1 state enabled: {getattr(self, '_l1_state_enabled', False)}")
        if not getattr(self, "_l1_state_enabled", False) or mode != Mode.DECODE:
            return super().forward(
                x,
                current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=user_id,
                mode=mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                get_last_token=get_last_token,
                kv_cache=kv_cache,
            )

        # Save current L1 states back to DRAM before parent forward
        # (parent uses its own layer loop without swaps)
        W = self._l1_window
        current_block = getattr(self, "_l1_current_start", 0)

        # Install swap hooks on each layer boundary
        gdn_set = set(self._gdn_indices)
        gdn_counter_wrapper = [0]  # mutable for closure
        current_block_wrapper = [current_block]

        # Ensure block 0 is loaded
        if current_block_wrapper[0] != 0:
            self._swap_l1_state(current_block_wrapper[0] * W, 0)
            current_block_wrapper[0] = 0

        # Temporarily wrap each layer to inject swaps
        original_forwards = {}
        for i, layer in enumerate(self.layers):
            if i in gdn_set:
                original_forwards[i] = layer.forward
                gdn_idx = gdn_counter_wrapper[0]

                def make_wrapped_forward(orig_fwd, layer_i, gdn_i):
                    def wrapped_forward(*args, **kwargs):
                        needed_block = gdn_i // W
                        if needed_block != current_block_wrapper[0]:
                            self._swap_l1_state(current_block_wrapper[0] * W, needed_block * W)
                            current_block_wrapper[0] = needed_block
                        return orig_fwd(*args, **kwargs)

                    return wrapped_forward

                layer.forward = make_wrapped_forward(layer.forward, i, gdn_counter_wrapper[0])
                gdn_counter_wrapper[0] += 1

        # Call parent forward (handles everything: layer loop, norm, LM head)
        try:
            result = super().forward(
                x,
                current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=user_id,
                mode=mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                get_last_token=get_last_token,
                kv_cache=kv_cache,
            )
        finally:
            # Restore original forwards
            for i, orig in original_forwards.items():
                self.layers[i].forward = orig
            self._l1_current_start = current_block_wrapper[0]

        return result

    def prefill_layer_chunked(
        self,
        tokens_tensor,
        chunk_size=None,
        *,
        use_paged=False,
        page_table=None,
        page_table_torch=None,
        kv_caches=None,
        paged_attention_config=None,
        user_id=0,
    ):
        """Prefill long sequences using layer-at-a-time chunked processing.

        Args:
            tokens_tensor: [1, seq_len] torch tensor of token IDs
            chunk_size: GDN chunk size. Default: args.prefill_len_cutoff (when
              use_paged=False) or GDN_CHUNK_SIZE (when use_paged=True).
            use_paged: enable paged KV cache path.
            page_table: full ttnn int32 page table (shape [B, max_num_blocks]),
              persistent on device. Required when use_paged=True.
            page_table_torch: the same page table as a torch.int32 tensor. Used
              to slice per-chunk views cheaply on host. Required when use_paged=True.
            kv_caches: List[Optional[(k_paged, v_paged)]], one entry per layer,
              from allocate_paged_kv_caches(). Required when use_paged=True.
            paged_attention_config: PagedAttentionConfig. Required when use_paged=True.
            user_id: int batch row for paged fill. Default 0.
        """
        if use_paged:
            assert page_table is not None, "use_paged=True requires page_table (ttnn tensor)"
            assert page_table_torch is not None, "use_paged=True requires page_table_torch"
            assert kv_caches is not None, "use_paged=True requires kv_caches list"
            assert paged_attention_config is not None, "use_paged=True requires paged_attention_config"
            if chunk_size is None:
                chunk_size = GDN_CHUNK_SIZE
        else:
            if chunk_size is None:
                chunk_size = self.args.prefill_len_cutoff

        seq_len = tokens_tensor.shape[-1]

        if use_paged:
            max_supported = paged_attention_config.block_size * paged_attention_config.max_num_blocks
            assert seq_len <= max_supported, (
                f"seq_len {seq_len} exceeds paged capacity "
                f"({paged_attention_config.block_size} x {paged_attention_config.max_num_blocks} "
                f"= {max_supported})"
            )

        # -- Step 1: Embed tokens + RoPE via framework --
        prefill_inputs = self.prepare_inputs_prefill(tokens_tensor)
        x = prefill_inputs[0]
        rot_mats_full = prefill_inputs[1]
        cos_full = rot_mats_full[0]
        sin_full = rot_mats_full[1]

        # -- Step 2: Init prefill states --
        # For paged attention layers we skip the static [B, 1, max_seq_len, HD]
        # KV allocation (reset_state) -- their KV lives in the external kv_caches.
        # GDN layers still reset (they use reset_state for conv-state init).
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.attention
            layer_type = self.args.layer_types[layer_idx]
            is_paged_attn = use_paged and layer_type == "full_attention"
            if hasattr(attn, "reset_state") and not is_paged_attn:
                attn.reset_state()
            if hasattr(attn, "_init_prefill_states"):
                attn._init_prefill_states()

        block_size = paged_attention_config.block_size if use_paged else None

        # -- Step 3: Layer loop with chunking --
        n_layers = len(self.layers)
        for layer_idx in range(n_layers):
            layer = self.layers[layer_idx]
            layer_type = self.args.layer_types[layer_idx]
            is_last_layer = layer_idx == n_layers - 1
            is_attention = layer_type == "full_attention"

            # Attention: full-seq (non-paged) or ATTN_CHUNK_SIZE (paged).
            # GDN: always chunk_size (existing behavior, orthogonal to paging).
            if is_attention:
                layer_chunk_size = ATTN_CHUNK_SIZE if use_paged else seq_len
            else:
                layer_chunk_size = chunk_size

            chunk_outputs = []
            x_last = None
            for chunk_start in range(0, seq_len, layer_chunk_size):
                chunk_end = min(chunk_start + layer_chunk_size, seq_len)

                if chunk_end == seq_len and len(chunk_outputs) == 0:
                    x_chunk = x
                else:
                    x_chunk = ttnn.slice(
                        x,
                        (0, 0, chunk_start, 0),
                        (1, 1, chunk_end, x.shape[-1]),
                    )

                # RoPE chunk slice (attention only -- GDN doesn't use RoPE).
                if is_attention:
                    cos_chunk = cos_full[:, :, chunk_start:chunk_end, :]
                    sin_chunk = sin_full[:, :, chunk_start:chunk_end, :]
                    rot_mats = [cos_chunk, sin_chunk]
                else:
                    rot_mats = None

                # Paged attention args (attention + paged only).
                if use_paged and is_attention:
                    chunk_block_start = chunk_start // block_size
                    chunk_block_end = (chunk_end + block_size - 1) // block_size
                    chunk_page_table_torch = page_table_torch[:, chunk_block_start:chunk_block_end]
                    chunk_page_table_tt = ttnn.from_torch(
                        chunk_page_table_torch,
                        device=self.mesh_device,
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    )
                    kv_cache_for_layer = kv_caches[layer_idx]
                    assert (
                        kv_cache_for_layer is not None
                    ), f"use_paged=True: layer {layer_idx} is attention but kv_caches[{layer_idx}] is None"
                else:
                    chunk_page_table_tt = None
                    kv_cache_for_layer = None

                logger.info(f"    Chunk [{chunk_start}:{chunk_end}] x_chunk.shape={x_chunk.shape}")
                out_chunk = layer(
                    x_chunk,
                    current_pos=None,
                    rot_mats_global=rot_mats,
                    mode=Mode.PREFILL,
                    page_table=page_table if (use_paged and is_attention) else None,
                    chunk_page_table=chunk_page_table_tt,
                    chunk_start_idx=chunk_start if (use_paged and is_attention) else None,
                    kv_cache=kv_cache_for_layer,
                    user_id=user_id,
                )

                if is_last_layer and chunk_end == seq_len:
                    last_tok_in_chunk = chunk_end - chunk_start - 1
                    get_last = (last_tok_in_chunk // 32) * 32
                    x_last = ttnn.slice(
                        out_chunk,
                        (0, 0, get_last, 0),
                        (1, 1, get_last + 32, out_chunk.shape[-1]),
                    )

                chunk_outputs.append(out_chunk)

            if len(chunk_outputs) == 1:
                x_new = chunk_outputs[0]
            else:
                x_new = ttnn.concat(chunk_outputs, dim=2)
                for c in chunk_outputs:
                    ttnn.deallocate(c)

            ttnn.deallocate(x)
            x = x_new

            logger.info(f"  Layer {layer_idx}/{n_layers} ({layer_type}) done, x.shape={x.shape}")

        # -- Step 4: Replicate prefill states to batch --
        for layer in self.layers:
            attn = layer.attention
            if hasattr(attn, "replicate_kv_cache_to_batch"):
                attn.replicate_kv_cache_to_batch()
            if hasattr(attn, "replicate_prefill_state_to_batch"):
                attn.replicate_prefill_state_to_batch()

        # -- Step 5: Final norm on last token --
        ttnn.deallocate(x)
        x_normed = self.norm(x_last, mode=Mode.PREFILL)

        return x_normed


def allocate_paged_kv_caches(
    model_args,
    paged_attention_config,
    mesh_device,
) -> List[Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
    """Allocate paged KV caches, one per layer.

    Returns a list with len == n_layers. GDN ("linear_attention") layers get
    None (they have no KV cache); full_attention layers get a (k_cache, v_cache)
    tuple, each shape (max_num_blocks, n_local_kv_heads, block_size, head_dim),
    dtype bfloat16, DRAM interleaved, replicated across the mesh.
    """
    block_size = paged_attention_config.block_size
    max_num_blocks = paged_attention_config.max_num_blocks
    n_local_kv_heads = model_args.n_local_kv_heads
    head_dim = model_args.head_dim

    kv_caches: List[Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]] = []
    for layer_idx, layer_type in enumerate(model_args.layer_types):
        if layer_type == "linear_attention":
            kv_caches.append(None)
            continue

        assert layer_type == "full_attention", f"Unexpected layer type {layer_type!r} at layer {layer_idx}"

        def _alloc():
            return ttnn.from_torch(
                torch.zeros(
                    max_num_blocks,
                    n_local_kv_heads,
                    block_size,
                    head_dim,
                    dtype=torch.bfloat16,
                ),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        kv_caches.append((_alloc(), _alloc()))

    assert len(kv_caches) == len(
        model_args.layer_types
    ), f"kv_caches length {len(kv_caches)} != layer count {len(model_args.layer_types)}"
    return kv_caches


def create_qwen35_model(
    mesh_device,
    model_path=None,
    max_batch_size=32,
    max_seq_len=131072,
    dtype=ttnn.bfloat8_b,
    paged_attention_config=None,
    use_paged_kv_cache=False,
    prefetcher=None,
    n_layers=None,
):
    """Factory function to create a fully initialized Qwen3.5 Transformer.

    Args:
        mesh_device: ttnn mesh device (TP=4)
        model_path: Path to Qwen3.5-27B-FP8 weights (default: ~/models/Qwen3.5-27B-FP8)
        max_batch_size: Maximum batch size
        max_seq_len: Maximum sequence length
        dtype: Weight dtype (default bfloat8_b)
        n_layers: Override number of layers (default: all layers from config)

    Returns:
        Transformer instance with all weights loaded
    """
    if model_path is None:
        # Honor MODEL_WEIGHTS_DIR (Docker convention) and HF_MODEL so callers
        # like the vLLM adapter don't need to plumb the path through every call
        # site. Falls back to the historical default for e2e-test entry points
        # that don't set either env var.
        model_path = (
            os.environ.get("MODEL_WEIGHTS_DIR")
            or os.environ.get("HF_MODEL")
            or os.path.expanduser("~/models/Qwen3.5-27B-FP8")
        )
    model_path = os.path.expanduser(model_path)

    # ModelArgs requires HF_MODEL env var
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    # logger.info("Loading Qwen3.5 state dict...")
    # state_dict = load_qwen35_state_dict(model_path)

    logger.info("Creating Qwen35ModelArgs...")
    args = Qwen35ModelArgs(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len)
    state_dict = args.load_state_dict()

    if n_layers is not None:
        logger.info(f"Overriding n_layers: {args.n_layers} -> {n_layers} (layer_types: {args.layer_types[:n_layers]})")
        args.n_layers = n_layers
        args.layer_types = args.layer_types[:n_layers]

    weight_cache_path = Path(os.path.expanduser("~/models/Qwen3.5-27B-mesh-tp4/framework"))
    os.makedirs(weight_cache_path, exist_ok=True)

    logger.info("Building Transformer...")
    model = Transformer(
        args=args,
        dtype=dtype,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=paged_attention_config,
        use_paged_kv_cache=use_paged_kv_cache,
        prefetcher=prefetcher,
    )

    logger.info("Qwen3.5 model ready")
    return model
