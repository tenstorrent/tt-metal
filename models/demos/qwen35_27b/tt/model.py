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

import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.attention import Qwen35Attention
from models.demos.qwen35_27b.tt.gdn import TtGatedDeltaNet
from models.demos.qwen35_27b.tt.model_config import (
    GDN_CONV_KERNEL_SIZE,
    Qwen35ModelArgs,
    _replicate,
    _shard_small,
    _shard_w,
    load_qwen35_state_dict,
    prepare_attn_qg,
    prepare_conv_taps,
    prepare_gdn_qkv,
    replicate_kv_weight,
)
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
        super().__init__(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            attention_class=Qwen35Attention,
            rope_setup_class=Qwen35PartialRopeSetup,
            prefetcher=prefetcher,
        )

        # Replace attention on linear_attention layers with TtGatedDeltaNet
        for i in range(args.n_layers):
            if args.layer_types[i] == "linear_attention":
                self.layers[i].attention = TtGatedDeltaNet(
                    mesh_device=mesh_device,
                    tt_ccl=self.tt_ccl,
                    args=args,
                    state_dict=state_dict,
                    weight_cache_path=weight_cache_path,
                    layer_num=i,
                    dtype=dtype,
                    transformation_mats=self.trans_mats_dict,
                    configuration=args,
                    paged_attention_config=paged_attention_config,
                    use_paged_kv_cache=use_paged_kv_cache,
                    prefetcher=prefetcher,
                )

        # Load attention-specific mesh weights and wire them up
        cache_dir = str(weight_cache_path / "attention_mesh")
        self._load_and_wire_attention_weights(state_dict, mesh_device, cache_dir, args)

    def _load_and_wire_attention_weights(self, state_dict, mesh, cache_dir, args):
        """Load attention mesh tensors and call set_weights() on each layer's attention."""
        os.makedirs(cache_dir, exist_ok=True)
        tp = args.num_devices

        for i in range(args.n_layers):
            layer_type = args.layer_types[i]
            p = f"layers.{i}."
            ld = os.path.join(cache_dir, f"layer_{i:02d}")
            os.makedirs(ld, exist_ok=True)
            tw = {}

            if layer_type == "full_attention":
                qg_reordered = prepare_attn_qg(state_dict, p, args.n_heads, args.head_dim, tp)
                tw["wqkv"] = _shard_w(
                    qg_reordered,
                    mesh,
                    dim=-1,
                    memory_config=args.attn_qg_weight_memcfg,
                    cache_path=os.path.join(ld, "wqkv"),
                )
                k_weight = state_dict[p + "attention.wk.weight"]
                v_weight = state_dict[p + "attention.wv.weight"]
                if args.kv_replication:
                    k_weight = replicate_kv_weight(k_weight, args.n_kv_heads, tp, args.head_dim)
                    v_weight = replicate_kv_weight(v_weight, args.n_kv_heads, tp, args.head_dim)
                tw["wk"] = _shard_w(
                    k_weight,
                    mesh,
                    dim=-1,
                    memory_config=args.attn_k_weight_memcfg,
                    cache_path=os.path.join(ld, "wk"),
                )
                tw["wv"] = _shard_w(
                    v_weight,
                    mesh,
                    dim=-1,
                    memory_config=args.attn_v_weight_memcfg,
                    cache_path=os.path.join(ld, "wv"),
                )
                tw["wo"] = _shard_w(
                    state_dict[p + "attention.wo.weight"],
                    mesh,
                    dim=0,
                    memory_config=args.attn_wo_weight_memcfg,
                    cache_path=os.path.join(ld, "wo"),
                )
                tw["q_norm"] = _replicate(
                    state_dict[p + "attention.q_norm.weight"],
                    mesh,
                    os.path.join(ld, "q_norm"),
                )
                tw["k_norm"] = _replicate(
                    state_dict[p + "attention.k_norm.weight"],
                    mesh,
                    os.path.join(ld, "k_norm"),
                )
                self.layers[i].attention.set_weights(tw)
                logger.info(f"  Layer {i:2d}/{args.n_layers} (full_attention) weights loaded")

            elif layer_type == "linear_attention":
                # Fused QKV+Z weight
                qkv_reordered = prepare_gdn_qkv(state_dict, p, tp)
                z_weight = state_dict[p + "linear_attn.in_proj_z.weight"]
                qkv_per = args.gdn_qkv_dim_tp
                z_per = args.gdn_z_dim_tp
                fused_parts = []
                for d in range(tp):
                    fused_parts.append(
                        torch.cat(
                            [
                                qkv_reordered[d * qkv_per : (d + 1) * qkv_per, :],
                                z_weight[d * z_per : (d + 1) * z_per, :],
                            ],
                            dim=0,
                        )
                    )
                qkvz_fused = torch.cat(fused_parts, dim=0)
                tw["qkvz"] = _shard_w(
                    qkvz_fused,
                    mesh,
                    dim=-1,
                    memory_config=args.gdn_qkvz_weight_memcfg,
                    cache_path=os.path.join(ld, "qkvz"),
                )

                # Fused A+B projection: concat per-device shards of A and B
                a_w = state_dict[p + "linear_attn.in_proj_a.weight"]
                b_w = state_dict[p + "linear_attn.in_proj_b.weight"]
                a_per = args.gdn_nv_tp
                b_per = args.gdn_nv_tp
                ab_parts = []
                for d in range(tp):
                    ab_parts.append(
                        torch.cat(
                            [
                                a_w[d * a_per : (d + 1) * a_per, :],
                                b_w[d * b_per : (d + 1) * b_per, :],
                            ],
                            dim=0,
                        )
                    )
                ab_fused = torch.cat(ab_parts, dim=0)
                tw["ab"] = _shard_w(
                    ab_fused,
                    mesh,
                    dim=-1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_path=os.path.join(ld, "ab"),
                )

                # Output projection (row-parallel)
                tw["out"] = _shard_w(
                    state_dict[p + "linear_attn.out_proj.weight"],
                    mesh,
                    dim=0,
                    memory_config=args.gdn_out_weight_memcfg,
                    cache_path=os.path.join(ld, "out"),
                )

                # Per-head params
                tw["A_log"] = _shard_small(
                    state_dict[p + "linear_attn.A_log"].float(),
                    mesh,
                    os.path.join(ld, "A_log"),
                )
                tw["dt_bias"] = _shard_small(
                    state_dict[p + "linear_attn.dt_bias"].float(),
                    mesh,
                    os.path.join(ld, "dt_bias"),
                )
                tw["norm_w"] = _replicate(
                    state_dict[p + "linear_attn.norm.weight"].float(),
                    mesh,
                    os.path.join(ld, "norm_w"),
                )

                # Conv taps
                taps = prepare_conv_taps(state_dict, p, tp)
                tw["conv_taps"] = [
                    _shard_small(taps[j], mesh, os.path.join(ld, f"conv_tap_{j}")) for j in range(GDN_CONV_KERNEL_SIZE)
                ]

                self.layers[i].attention.set_weights(tw)
                logger.info(f"  Layer {i:2d}/{args.n_layers} (linear_attention) weights loaded")

            else:
                logger.warning(f"  Layer {i:2d}: unknown type '{layer_type}'")

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

    def prefill_layer_chunked(self, tokens_tensor, chunk_size=None):
        """Prefill long sequences using layer-at-a-time chunked processing.

        Args:
            tokens_tensor: [1, seq_len] torch tensor of token IDs
            chunk_size: base chunk size (default: prefill_len_cutoff, 512 on BH)

        Returns:
            x_normed: last-token output after final norm, ready for LM head
        """
        if chunk_size is None:
            chunk_size = self.args.prefill_len_cutoff  # 512 on BH, 1024 on WH
        seq_len = tokens_tensor.shape[-1]

        # ── Step 1: Embed tokens + RoPE via framework ────────────────
        # Use the framework's prepare_inputs_prefill to handle embedding
        # and RoPE correctly (mesh sharding, tile layout, etc.)
        prefill_inputs = self.prepare_inputs_prefill(tokens_tensor)
        x = prefill_inputs[0]  # [1, 1, seq_len, dim] (mesh-sharded embedding)
        rot_mats_full = prefill_inputs[1]  # [cos, sin] each [1, 1, seq_len, rope_dim]
        cos_full = rot_mats_full[0]
        sin_full = rot_mats_full[1]

        # ── Step 3: Init GDN prefill states ──────────────────────────
        for layer in self.layers:
            attn = layer.attention
            if hasattr(attn, "reset_state"):
                attn.reset_state()
            if hasattr(attn, "_init_prefill_states"):
                attn._init_prefill_states()

        # ── Step 4: Layer loop with chunking ─────────────────────────
        n_layers = len(self.layers)
        for layer_idx in range(n_layers):
            layer = self.layers[layer_idx]
            layer_type = self.args.layer_types[layer_idx]
            is_last_layer = layer_idx == n_layers - 1
            is_attention = layer_type == "full_attention"

            # Attention layers process full sequence (need full KV context for causal SDPA).
            # GDN layers chunk (recurrent state carries over between chunks).
            layer_chunk_size = seq_len if is_attention else chunk_size

            chunk_outputs = []
            for chunk_start in range(0, seq_len, layer_chunk_size):
                chunk_end = min(chunk_start + layer_chunk_size, seq_len)

                if chunk_end == seq_len and len(chunk_outputs) == 0:
                    # Single chunk covers full sequence — no slice needed
                    x_chunk = x
                else:
                    x_chunk = ttnn.slice(
                        x,
                        (0, 0, chunk_start, 0),
                        (1, 1, chunk_end, x.shape[-1]),
                    )

                # Prepare rot_mats for this chunk
                if is_attention:
                    cos_chunk = cos_full[:, :, chunk_start:chunk_end, :]
                    sin_chunk = sin_full[:, :, chunk_start:chunk_end, :]
                    rot_mats = [cos_chunk, sin_chunk]
                else:
                    rot_mats = None

                # Run layer on chunk
                logger.info(f"    Chunk [{chunk_start}:{chunk_end}] x_chunk.shape={x_chunk.shape}")
                out_chunk = layer(
                    x_chunk,
                    current_pos=None,
                    rot_mats_global=rot_mats,
                    mode=Mode.PREFILL,
                )

                # On last layer, extract last token from last chunk before concat
                if is_last_layer and chunk_end == seq_len:
                    last_tok_in_chunk = chunk_end - chunk_start - 1
                    get_last = (last_tok_in_chunk // 32) * 32
                    x_last = ttnn.slice(
                        out_chunk,
                        (0, 0, get_last, 0),
                        (1, 1, get_last + 32, out_chunk.shape[-1]),
                    )

                chunk_outputs.append(out_chunk)

            # Concatenate chunks back to full sequence
            if len(chunk_outputs) == 1:
                x_new = chunk_outputs[0]
            else:
                x_new = ttnn.concat(chunk_outputs, dim=2)
                for c in chunk_outputs:
                    ttnn.deallocate(c)

            ttnn.deallocate(x)
            x = x_new

            logger.info(f"  Layer {layer_idx}/{n_layers} ({layer_type}) done, x.shape={x.shape}")

        # ── Step 5: Replicate prefill states to batch ────────────────
        for layer in self.layers:
            attn = layer.attention
            if hasattr(attn, "replicate_kv_cache_to_batch"):
                attn.replicate_kv_cache_to_batch()
            if hasattr(attn, "replicate_prefill_state_to_batch"):
                attn.replicate_prefill_state_to_batch()

        # ── Step 6: Final norm on last token ─────────────────────────
        ttnn.deallocate(x)
        x_normed = self.norm(x_last, mode=Mode.PREFILL)

        return x_normed


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
        model_path = os.path.expanduser("~/models/Qwen3.5-27B-FP8")

    # ModelArgs requires HF_MODEL env var
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    logger.info("Loading Qwen3.5 state dict...")
    state_dict = load_qwen35_state_dict(model_path)

    logger.info("Creating Qwen35ModelArgs...")
    args = Qwen35ModelArgs(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len)

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
