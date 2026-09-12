# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Tenstorrent Bounty #18287: Bring up microsoft/phi-1 on Wormhole (N150/N300)
# Architecture: PhiForCausalLM (HuggingFace transformers >= 4.37)
#   - 24 decoder layers, parallel residual connections
#   - Partial RoPE: rotary_dim=32 out of head_dim=64 (num_attention_heads=32)
#   - NewGELU activation (gelu_new / tanh-approximate GELU)
#
# tt-transformers reuse analysis (bounty #18287 Stage 2 requires leveraging tt_transformers
# where possible; each module below was evaluated against it individually rather than assumed
# reusable or unreusable wholesale):
#   - Decode-time attention (TTPhi1Attention, decode branch): REUSED. Calls
#     ttnn.transformer.scaled_dot_product_attention_decode directly -- the same op
#     tt_transformers/tt/attention.py uses for its own KV-cache decode reads -- instead of a
#     hand-rolled ttnn.slice + non-causal SDPA, which is both non-tile-aligned-safe (ttnn.slice
#     on TILE_LAYOUT requires the output height to be a multiple of 32) and duplicated logic.
#   - RoPE (TTPhi1RoPE): NOT reused, deliberately. tt_transformers/tt/rope.py's
#     Phi3RotaryEmbedding implements Phi-3's long-context scaling (long_factor/short_factor
#     arrays + a scaling_factor multiplier) -- an unrelated mechanism to Phi-1's simple partial
#     rotary_dim=32-of-64 -- and would silently corrupt the rotation math if substituted in.
#     HfRotarySetup/RotarySetup are mesh-device sharded-serving infrastructure built on
#     ttnn.experimental.rotary_embedding_hf, a different op than the one already verified
#     correct here (ttnn.experimental.rotary_embedding). No safe drop-in exists for Phi-1's
#     specific rotary configuration.
#   - Attention/MLP projections, parallel-residual decoder layer: NOT reused. tt_transformers'
#     Attention/MLP modules are deeply coupled to mesh-device, paged-attention, multi-batch
#     serving infrastructure (program configs, prefetcher, CCL all-reduce) that a single-device,
#     single-sequence demo has no use for and would require substantially gutting to adapt.
#     Phi-1's parallel-residual layer (x + attn(LN(x)) + mlp(LN(x))) also differs structurally
#     from tt_transformers' sequential-residual decoder layer.

import math
import typing

import torch

import ttnn


def precompute_freqs(rotary_dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for RoPE (sin/cos caches).

    Returns cos and sin tensors of shape [1, 1, max_seq_len, rotary_dim].
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_seq_len, rotary_dim/2]
    # Rotate-half convention: [c0..c_{d/2-1}, c0..c_{d/2-1}], matching ttnn's rotary_embedding kernel
    # (half_Wt split), NOT the interleaved [c0, c0, c1, c1, ...] convention.
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, rotary_dim]
    cos_cache = emb.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, rotary_dim]
    sin_cache = emb.sin().unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, rotary_dim]
    return cos_cache, sin_cache


class TTPhi1RoPE:
    """Rotary Position Embedding setup for Phi-1 on Tenstorrent hardware.

    Precomputes cos/sin caches and provides methods to retrieve position-sliced
    rotation matrices for both prefill and decode modes.

    Uses ttnn.experimental.rotary_embedding (standard HF RoPE kernel).
    """

    def __init__(
        self,
        device: ttnn.Device,
        head_dim: int = 64,
        rotary_dim: int = 32,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        # Precompute cos/sin on CPU then transfer to device DRAM
        cos_cache_torch, sin_cache_torch = precompute_freqs(rotary_dim, max_seq_len, base)

        # ROW_MAJOR_LAYOUT: ttnn.slice on TILE_LAYOUT requires the output height AND slice-start
        # to be tile-aligned (multiple of 32) -- decode calls this with seq_len=1 and an arbitrary
        # start_pos, which can never satisfy either constraint. ROW_MAJOR_LAYOUT slicing has no
        # such alignment requirement (verified against slice_device_operation.cpp), so the master
        # cache is stored row-major and the sliced-out result is converted to TILE_LAYOUT
        # afterward, since ttnn.experimental.rotary_embedding requires TILE inputs for cos/sin.
        self.cos_cache = ttnn.from_torch(
            cos_cache_torch.to(torch.bfloat16),
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.sin_cache = ttnn.from_torch(
            sin_cache_torch.to(torch.bfloat16),
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def get_rot_mats(self, start_pos: int, seq_len: int):
        """Get position-sliced cos/sin matrices for the given sequence range.

        Returns (cos_slice, sin_slice) tensors ready for ttnn.experimental.rotary_embedding.
        """
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(
                f"RoPE/KV-cache position range [{start_pos}, {start_pos + seq_len}) exceeds "
                f"max_seq_len={self.max_seq_len}. Increase max_seq_len or shorten the prompt/generation."
            )
        cos_slice_rm = ttnn.slice(
            self.cos_cache,
            [0, 0, start_pos, 0],
            [1, 1, start_pos + seq_len, self.rotary_dim],
        )
        sin_slice_rm = ttnn.slice(
            self.sin_cache,
            [0, 0, start_pos, 0],
            [1, 1, start_pos + seq_len, self.rotary_dim],
        )
        cos_slice = ttnn.to_layout(cos_slice_rm, ttnn.TILE_LAYOUT)
        sin_slice = ttnn.to_layout(sin_slice_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(cos_slice_rm)
        ttnn.deallocate(sin_slice_rm)
        return cos_slice, sin_slice


class TTPhi1Attention:
    """Multi-Head Attention block for Phi-1 (microsoft/phi-1).

    Architecture:
        - num_attention_heads = 32, head_dim = 64 (hidden_size=2048)
        - Partial RoPE: rotary_dim=32 out of head_dim=64
        - Uses ttnn.transformer.scaled_dot_product_attention (SDPA) for HW acceleration
        - RoPE applied via ttnn.experimental.rotary_embedding on the first 32 dims of Q/K
    """

    def __init__(
        self,
        device: ttnn.Device,
        state_dict: typing.Dict[str, torch.Tensor],
        base_address: str,
        n_heads: int = 32,
        hidden_size: int = 2048,
        rotary_dim: int = 32,
        dtype: ttnn.DataType = ttnn.bfloat16,
        max_seq_len: int = 2048,
    ):
        self.device = device
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_heads  # 2048 // 32 = 64
        self.rotary_dim = rotary_dim
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        # Load Wqkv combined projection weights (MixFormer format)
        if f"{base_address}.self_attn.Wqkv.weight" in state_dict:
            wqkv_weight = state_dict[f"{base_address}.self_attn.Wqkv.weight"]
            wqkv_bias = state_dict[f"{base_address}.self_attn.Wqkv.bias"]
        elif f"{base_address}.mixer.Wqkv.weight" in state_dict:
            wqkv_weight = state_dict[f"{base_address}.mixer.Wqkv.weight"]
            wqkv_bias = state_dict[f"{base_address}.mixer.Wqkv.bias"]
        elif f"{base_address}.self_attn.q_proj.weight" in state_dict:
            # Native HF PhiForCausalLM: separate q/k/v projections
            q_w = state_dict[f"{base_address}.self_attn.q_proj.weight"]
            k_w = state_dict[f"{base_address}.self_attn.k_proj.weight"]
            v_w = state_dict[f"{base_address}.self_attn.v_proj.weight"]
            wqkv_weight = torch.cat([q_w, k_w, v_w], dim=0)  # [3*hidden, hidden]

            q_b = state_dict[f"{base_address}.self_attn.q_proj.bias"]
            k_b = state_dict[f"{base_address}.self_attn.k_proj.bias"]
            v_b = state_dict[f"{base_address}.self_attn.v_proj.bias"]
            wqkv_bias = torch.cat([q_b, k_b, v_b], dim=0)  # [3*hidden]
        else:
            raise KeyError(
                f"Could not find QKV projection weights in state_dict for {base_address}. "
                f"Checked: self_attn.Wqkv, mixer.Wqkv, self_attn.q_proj"
            )

        self.wqkv = ttnn.from_torch(
            wqkv_weight.T.contiguous(),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bqkv = ttnn.from_torch(
            wqkv_bias.reshape(1, 1, 1, -1).contiguous(),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Output projection (out_proj for MixFormer / dense for native HF)
        out_w_key, out_b_key = None, None
        for prefix in [
            f"{base_address}.self_attn.out_proj",
            f"{base_address}.mixer.out_proj",
            f"{base_address}.self_attn.dense",
        ]:
            if f"{prefix}.weight" in state_dict:
                out_w_key = f"{prefix}.weight"
                out_b_key = f"{prefix}.bias"
                break
        if out_w_key is None:
            raise KeyError(f"Could not find output projection weights for {base_address}")

        self.out_proj = ttnn.from_torch(
            state_dict[out_w_key].T.contiguous(),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bout_proj = ttnn.from_torch(
            state_dict[out_b_key].reshape(1, 1, 1, -1).contiguous(),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cache_shape = (1, n_heads, max_seq_len, self.head_dim)
        self.k_cache = ttnn.from_torch(
            torch.zeros(cache_shape, dtype=torch.bfloat16),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.v_cache = ttnn.from_torch(
            torch.zeros(cache_shape, dtype=torch.bfloat16),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(
        self,
        x: ttnn.Tensor,
        cos_cache: typing.Optional[ttnn.Tensor] = None,
        sin_cache: typing.Optional[ttnn.Tensor] = None,
        current_pos: int = 0,
        use_cache: bool = False,
    ) -> ttnn.Tensor:
        """Forward pass for Phi-1 attention with Partial RoPE.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            cos_cache: Cosine rotation matrix [1, 1, seq_len, rotary_dim]
            sin_cache: Sine rotation matrix [1, 1, seq_len, rotary_dim]
        """
        # Project Q, K, V
        qkv = ttnn.linear(x, self.wqkv, bias=self.bqkv, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Ensure rank-3 for split_query_key_value_and_split_heads kernel
        if len(qkv.shape) == 4 and qkv.shape[1] == 1:
            qkv_new = ttnn.reshape(qkv, (qkv.shape[0], qkv.shape[2], qkv.shape[3]))
            ttnn.deallocate(qkv)
            qkv = qkv_new

        # Split QKV -> separate Q, K, V with heads split
        # Output shapes: [batch, n_heads, seq_len, head_dim]
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            num_heads=self.n_heads,
            transpose_key=False,
        )
        ttnn.deallocate(qkv)

        # Apply Partial RoPE: only first rotary_dim=32 dimensions of Q and K
        if cos_cache is not None and sin_cache is not None:
            # Slice via ROW_MAJOR_LAYOUT: ttnn.slice on TILE_LAYOUT requires the output height
            # (dim -2, the seq dimension here) to be a multiple of TILE_HEIGHT (32). During decode
            # q/k have seq_len=1, so this fails regardless of which dim is nominally being
            # restricted (verified against slice_device_operation.cpp). ROW_MAJOR_LAYOUT has no
            # such constraint. Convert back to TILE_LAYOUT afterward since both
            # ttnn.experimental.rotary_embedding and ttnn.concat (which requires all inputs to
            # share the same layout) need TILE inputs here.
            q_rm = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
            q_rot_rm = ttnn.slice(q_rm, [0, 0, 0, 0], [q.shape[0], q.shape[1], q.shape[2], self.rotary_dim])
            q_pass_rm = ttnn.slice(
                q_rm, [0, 0, 0, self.rotary_dim], [q.shape[0], q.shape[1], q.shape[2], self.head_dim]
            )
            ttnn.deallocate(q_rm)
            q_rot = ttnn.to_layout(q_rot_rm, ttnn.TILE_LAYOUT)
            ttnn.deallocate(q_rot_rm)
            q_pass = ttnn.to_layout(q_pass_rm, ttnn.TILE_LAYOUT)
            ttnn.deallocate(q_pass_rm)

            # Apply RoPE to the rotary portion using the real cos/sin caches
            q_rot_emb = ttnn.experimental.rotary_embedding(
                q_rot, cos_cache, sin_cache, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            ttnn.deallocate(q_rot)

            # Concatenate rotated + pass-through
            q_new = ttnn.concat([q_rot_emb, q_pass], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(q_rot_emb)
            ttnn.deallocate(q_pass)
            ttnn.deallocate(q)
            q = q_new

            # Same for K
            k_rm = ttnn.to_layout(k, ttnn.ROW_MAJOR_LAYOUT)
            k_rot_rm = ttnn.slice(k_rm, [0, 0, 0, 0], [k.shape[0], k.shape[1], k.shape[2], self.rotary_dim])
            k_pass_rm = ttnn.slice(
                k_rm, [0, 0, 0, self.rotary_dim], [k.shape[0], k.shape[1], k.shape[2], self.head_dim]
            )
            ttnn.deallocate(k_rm)
            k_rot = ttnn.to_layout(k_rot_rm, ttnn.TILE_LAYOUT)
            ttnn.deallocate(k_rot_rm)
            k_pass = ttnn.to_layout(k_pass_rm, ttnn.TILE_LAYOUT)
            ttnn.deallocate(k_pass_rm)

            k_rot_emb = ttnn.experimental.rotary_embedding(
                k_rot, cos_cache, sin_cache, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            ttnn.deallocate(k_rot)

            k_new = ttnn.concat([k_rot_emb, k_pass], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(k_rot_emb)
            ttnn.deallocate(k_pass)
            ttnn.deallocate(k)
            k = k_new

        if use_cache:
            seq_len_here = q.shape[2]
            if seq_len_here > 1:
                # Prefill: seed the cache with this call's full K/V starting at current_pos, then
                # attend over the full sequence exactly as before (unchanged SDPA call/is_causal=True).
                # current_pos must be tile-aligned (fill_cache's update_idx is divided by TILE_HEIGHT).
                # Only current callers ever prefill at current_pos=0; guard defensively in case a future
                # caller (e.g. multi-turn/context-extension) ever prefills at a non-zero position.
                if current_pos % 32 != 0:
                    raise ValueError(
                        f"fill_cache requires a tile-aligned update_idx (multiple of 32), got current_pos={current_pos}"
                    )
                ttnn.fill_cache(self.k_cache, k, batch_idx=0, update_idx=current_pos)
                ttnn.fill_cache(self.v_cache, v, batch_idx=0, update_idx=current_pos)
                attn_out = ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, is_causal=True, memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            else:
                # Decode: write this single new token's K/V at current_pos, then attend the new
                # query against the valid cached range using ttnn's decode-specific SDPA kernel.
                #
                # NOTE: this replaces an earlier version that read the cache via
                # ttnn.slice(self.k_cache, ..., [1, n_heads, current_pos+1, head_dim]) followed by
                # a regular (non-causal) scaled_dot_product_attention call. That was a real,
                # hardware-crashing bug: ttnn.slice on TILE_LAYOUT tensors requires the output
                # height AND start index to be tile-aligned (slice_device_operation.cpp: "Can only
                # slice tilized tensor with height begin index aligned to tiles"), but
                # current_pos+1 is essentially never a multiple of 32. scaled_dot_product_attention_decode
                # instead takes the full K/V cache plus cur_pos directly and reads the valid range
                # internally, avoiding the alignment requirement entirely. This is also the same
                # decode op tt_transformers/tt/attention.py uses for its own KV-cache reads.
                ttnn.update_cache(self.k_cache, k, current_pos, batch_offset=0)
                ttnn.update_cache(self.v_cache, v, current_pos, batch_offset=0)

                # scaled_dot_product_attention_decode requires Q as [1, B, NH, D] (not our usual
                # [B, NH, S, D]) and in DRAM when unsharded (sdpa_decode_device_operation.cpp).
                # K/V caches already match its expected [B, NH, S, D] DRAM/TILE contract unchanged.
                q_dram = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
                q_decode = ttnn.reshape(q_dram, (1, 1, self.n_heads, self.head_dim))
                ttnn.deallocate(q_dram)

                attn_out_decode = ttnn.transformer.scaled_dot_product_attention_decode(
                    q_decode,
                    self.k_cache,
                    self.v_cache,
                    is_causal=True,
                    cur_pos=[current_pos],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(q_decode)

                # Back to [batch, n_heads, seq=1, head_dim] to match the prefill branch's
                # attn_out convention before concatenate_heads.
                attn_out = ttnn.reshape(attn_out_decode, (1, self.n_heads, 1, self.head_dim))
                ttnn.deallocate(attn_out_decode)
        else:
            # Existing no-cache path, byte-for-byte identical to current behavior.
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        # Free intermediate Q, K, V from L1
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Concatenate heads back: [batch, n_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        attn_out_concat = ttnn.transformer.concatenate_heads(
            attn_out,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_out)

        # Fix rank if concatenate_heads returns rank-4
        if len(attn_out_concat.shape) == 4 and attn_out_concat.shape[1] == 1:
            attn_out_3d = ttnn.reshape(
                attn_out_concat,
                (attn_out_concat.shape[0], attn_out_concat.shape[2], attn_out_concat.shape[3]),
            )
            ttnn.deallocate(attn_out_concat)
            attn_out_concat = attn_out_3d

        # Output linear projection
        output = ttnn.linear(attn_out_concat, self.out_proj, bias=self.bout_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out_concat)
        return output


class TTPhi1MLP:
    """MLP block for Phi-1: fc1 -> NewGELU -> fc2.

    Architecture:
        - intermediate_size = 8192 (4x expansion of hidden_size=2048)
        - Activation: NewGELU (tanh-approximate GELU, `hidden_act="gelu_new"`)
    """

    def __init__(
        self,
        device: ttnn.Device,
        state_dict: typing.Dict[str, torch.Tensor],
        base_address: str,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.dtype = dtype

        # Resolve fc1/fc2 keys across naming conventions
        fc1_w_key, fc1_b_key, fc2_w_key, fc2_b_key = None, None, None, None
        for fc1_prefix, fc2_prefix in [
            (f"{base_address}.mlp.fc1", f"{base_address}.mlp.fc2"),
            (f"{base_address}.mlp.c_fc", f"{base_address}.mlp.c_proj"),
        ]:
            if f"{fc1_prefix}.weight" in state_dict:
                fc1_w_key = f"{fc1_prefix}.weight"
                fc1_b_key = f"{fc1_prefix}.bias"
                fc2_w_key = f"{fc2_prefix}.weight"
                fc2_b_key = f"{fc2_prefix}.bias"
                break
        if fc1_w_key is None:
            raise KeyError(f"Could not find MLP projection weights for {base_address}")

        self.fc1 = ttnn.from_torch(
            state_dict[fc1_w_key].T.contiguous(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.bfc1 = ttnn.from_torch(
            state_dict[fc1_b_key].reshape(1, 1, 1, -1).contiguous(),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.fc2 = ttnn.from_torch(
            state_dict[fc2_w_key].T.contiguous(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.bfc2 = ttnn.from_torch(
            state_dict[fc2_b_key].reshape(1, 1, 1, -1).contiguous(),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # fc1 projection + NewGELU activation (tanh-approximate GELU)
        h = ttnn.linear(x, self.fc1, bias=self.bfc1, memory_config=ttnn.L1_MEMORY_CONFIG)
        h_act = ttnn.gelu(h, fast_and_approximate_mode=True, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h)
        # fc2 projection
        out = ttnn.linear(h_act, self.fc2, bias=self.bfc2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h_act)

        # ttnn.linear can return rank-4 output ([batch, 1, seq, dim]) even for rank-3 input in some
        # configs (same class of promotion the qkv projection guards against in TTPhi1Attention);
        # normalize back to rank-3 so the residual add against attn_out (always rank-3) can't shape-mismatch.
        if len(out.shape) == 4 and out.shape[1] == 1:
            out_3d = ttnn.reshape(out, (out.shape[0], out.shape[2], out.shape[3]))
            ttnn.deallocate(out)
            out = out_3d

        return out


class TTPhi1DecoderLayer:
    """Single Transformer Decoder Layer for Phi-1.

    Architecture: Parallel residual connections:
        output = input + attn(LayerNorm(input)) + mlp(LayerNorm(input))

    Both attention and MLP operate on the same layer-normed input in parallel,
    then their results are summed with the residual identity.
    """

    def __init__(
        self,
        device: ttnn.Device,
        state_dict: typing.Dict[str, torch.Tensor],
        base_address: str,
        layer_num: int,
        n_heads: int = 32,
        hidden_size: int = 2048,
        rotary_dim: int = 32,
        dtype: ttnn.DataType = ttnn.bfloat16,
        max_seq_len: int = 2048,
    ):
        self.device = device

        # Resolve layer address across HF naming variants
        candidates = [
            f"{base_address}.layers.{layer_num}",
            f"model.layers.{layer_num}",
            f"transformer.h.{layer_num}",
            f"{base_address}.{layer_num}",
        ]
        resolved_address = None
        for cand in candidates:
            if any(k.startswith(f"{cand}.") for k in state_dict.keys()):
                resolved_address = cand
                break
        if resolved_address is None:
            resolved_address = f"{base_address}.layers.{layer_num}"

        self.base_address = resolved_address

        # Input LayerNorm (input_layernorm / ln / ln_1)
        ln_w_key, ln_b_key = None, None
        for ln_prefix in [
            f"{self.base_address}.input_layernorm",
            f"{self.base_address}.ln",
            f"{self.base_address}.ln_1",
        ]:
            if f"{ln_prefix}.weight" in state_dict:
                ln_w_key = f"{ln_prefix}.weight"
                ln_b_key = f"{ln_prefix}.bias"
                break
        if ln_w_key is None:
            raise KeyError(f"Could not find LayerNorm weights for {self.base_address}")

        self.ln_weight = ttnn.from_torch(
            state_dict[ln_w_key].reshape(1, 1, 1, -1).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.ln_bias = ttnn.from_torch(
            state_dict[ln_b_key].reshape(1, 1, 1, -1).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.self_attn = TTPhi1Attention(
            device,
            state_dict,
            self.base_address,
            n_heads=n_heads,
            hidden_size=hidden_size,
            rotary_dim=rotary_dim,
            dtype=dtype,
            max_seq_len=max_seq_len,
        )
        self.mlp = TTPhi1MLP(device, state_dict, self.base_address, hidden_size=hidden_size, dtype=dtype)

    def __call__(
        self,
        x: ttnn.Tensor,
        cos_cache: typing.Optional[ttnn.Tensor] = None,
        sin_cache: typing.Optional[ttnn.Tensor] = None,
        current_pos: int = 0,
        use_cache: bool = False,
    ) -> ttnn.Tensor:
        # Layer Normalization
        normed_x = ttnn.layer_norm(
            x, weight=self.ln_weight, bias=self.ln_bias, epsilon=1e-5, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        # Parallel Attention & MLP on the same normed input
        attn_out = self.self_attn(
            normed_x, cos_cache=cos_cache, sin_cache=sin_cache, current_pos=current_pos, use_cache=use_cache
        )
        mlp_out = self.mlp(normed_x)
        ttnn.deallocate(normed_x)

        # Parallel Residual: x + attn_out + mlp_out
        residual_sum = ttnn.add(attn_out, mlp_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(mlp_out)

        output = ttnn.add(x, residual_sum, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(residual_sum)
        return output


class TTPhi1Model:
    """Core Phi-1 Transformer backbone (num_hidden_layers=24).

    Stacks token embeddings, 24 TTPhi1DecoderLayers, and final LayerNorm.
    Initializes RoPE cache (TTPhi1RoPE) for real cos/sin position embeddings.
    """

    def __init__(
        self,
        device: ttnn.Device,
        state_dict: typing.Dict[str, torch.Tensor],
        base_address: str = "model",
        num_hidden_layers: int = 24,
        n_heads: int = 32,
        hidden_size: int = 2048,
        rotary_dim: int = 32,
        max_position_embeddings: int = 2048,
        dtype: ttnn.DataType = ttnn.bfloat16,
        max_seq_len: int = 2048,
    ):
        self.device = device
        self.base_address = base_address
        self.num_hidden_layers = num_hidden_layers
        self.dtype = dtype

        # 1. Token Embeddings
        embed_key = None
        for candidate in [
            f"{base_address}.embed_tokens.weight",
            "transformer.embd.wte.weight",
            "transformer.wte.weight",
            f"{base_address}.wte.weight",
            "model.embed_tokens.weight",
        ]:
            if candidate in state_dict:
                embed_key = candidate
                break
        if embed_key is None:
            raise KeyError(f"Could not find token embedding matrix in state_dict for {base_address}")

        self.embed_tokens_torch = state_dict[embed_key]
        self.embed_tokens_device = ttnn.from_torch(
            self.embed_tokens_torch.to(torch.bfloat16).unsqueeze(0).unsqueeze(0),
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # 2. RoPE Cache (real cos/sin, not dummies)
        self.rope = TTPhi1RoPE(
            device=device,
            head_dim=hidden_size // n_heads,
            rotary_dim=rotary_dim,
            max_seq_len=max_position_embeddings,
            dtype=dtype,
        )

        # 3. Stack Decoder Layers
        self.layers = []
        for layer_num in range(num_hidden_layers):
            self.layers.append(
                TTPhi1DecoderLayer(
                    device=device,
                    state_dict=state_dict,
                    base_address=base_address,
                    layer_num=layer_num,
                    n_heads=n_heads,
                    hidden_size=hidden_size,
                    rotary_dim=rotary_dim,
                    dtype=dtype,
                    max_seq_len=max_seq_len,
                )
            )

        # 4. Final LayerNorm
        norm_w_key, norm_b_key = None, None
        for candidate_prefix in [
            f"{base_address}.final_layernorm",
            f"{base_address}.norm",
            "transformer.ln_f",
            "transformer.ln",
            "model.final_layernorm",
        ]:
            if f"{candidate_prefix}.weight" in state_dict and f"{candidate_prefix}.bias" in state_dict:
                norm_w_key = f"{candidate_prefix}.weight"
                norm_b_key = f"{candidate_prefix}.bias"
                break

        if norm_w_key is None:
            raise KeyError("Could not find final LayerNorm weights in state_dict.")

        self.final_norm_weight = ttnn.from_torch(
            state_dict[norm_w_key].reshape(1, 1, 1, -1).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.final_norm_bias = ttnn.from_torch(
            state_dict[norm_b_key].reshape(1, 1, 1, -1).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def __call__(
        self,
        x: typing.Union[ttnn.Tensor, torch.Tensor],
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> ttnn.Tensor:
        """Forward pass through the Phi-1 backbone.

        Args:
            x: Token IDs (torch.long) or pre-embedded hidden states
            start_pos: Starting position for RoPE cache slicing (for decode mode)
        """
        # Handle input embedding
        if isinstance(x, torch.Tensor):
            if x.dtype in [torch.long, torch.int, torch.int64, torch.int32]:
                x_device = ttnn.from_torch(x.to(torch.uint32), device=self.device)
                hidden_state = ttnn.embedding(x_device, self.embed_tokens_device, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(x_device)
                seq_len = x.shape[-1]
            else:
                hidden_state = ttnn.from_torch(
                    x.to(torch.bfloat16),
                    dtype=self.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                seq_len = x.shape[-2] if x.dim() >= 2 else x.shape[0]
            can_deallocate = True
        else:
            hidden_state = x
            seq_len = x.shape[-2] if len(x.shape) >= 3 else x.shape[-1]
            can_deallocate = False

        # Ensure rank-3 [batch, seq_len, hidden_size]
        if len(hidden_state.shape) == 4 and hidden_state.shape[1] == 1:
            hidden_state_new = ttnn.reshape(
                hidden_state, (hidden_state.shape[0], hidden_state.shape[2], hidden_state.shape[3])
            )
            if can_deallocate:
                ttnn.deallocate(hidden_state)
            hidden_state = hidden_state_new
            can_deallocate = True

        # Get RoPE cos/sin caches for this sequence
        cos_cache, sin_cache = self.rope.get_rot_mats(start_pos, seq_len)

        # Forward through all decoder layers
        for i, layer in enumerate(self.layers):
            prev_hidden_state = hidden_state
            hidden_state = layer(
                hidden_state,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                current_pos=start_pos,
                use_cache=use_cache,
            )
            if (i > 0 or can_deallocate) and isinstance(prev_hidden_state, ttnn.Tensor):
                ttnn.deallocate(prev_hidden_state)

        # Free RoPE slices
        ttnn.deallocate(cos_cache)
        ttnn.deallocate(sin_cache)

        # Final LayerNorm
        output = ttnn.layer_norm(
            hidden_state,
            weight=self.final_norm_weight,
            bias=self.final_norm_bias,
            epsilon=1e-5,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden_state)
        return output


class TTPhi1ForCausalLM:
    """Top-level Causal LM model for microsoft/phi-1.

    Wraps TTPhi1Model backbone and applies LM Head projection to vocabulary (vocab_size=51200).
    """

    def __init__(
        self,
        device: ttnn.Device,
        state_dict: typing.Dict[str, torch.Tensor],
        base_address: str = "model",
        num_hidden_layers: int = 24,
        n_heads: int = 32,
        hidden_size: int = 2048,
        rotary_dim: int = 32,
        max_position_embeddings: int = 2048,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self.model = TTPhi1Model(
            device=device,
            state_dict=state_dict,
            base_address=base_address,
            num_hidden_layers=num_hidden_layers,
            n_heads=n_heads,
            hidden_size=hidden_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            dtype=dtype,
            max_seq_len=max_position_embeddings,
        )

        # LM Head (lm_head.linear for MixFormer, lm_head for native HF)
        lm_w_key, lm_b_key = None, None
        for candidate_prefix in ["lm_head.linear", "lm_head", "model.lm_head"]:
            if f"{candidate_prefix}.weight" in state_dict:
                lm_w_key = f"{candidate_prefix}.weight"
                if f"{candidate_prefix}.bias" in state_dict:
                    lm_b_key = f"{candidate_prefix}.bias"
                break

        if lm_w_key is None:
            # Fallback: tied embeddings
            for cand in ["model.embed_tokens.weight", "transformer.embd.wte.weight"]:
                if cand in state_dict:
                    lm_w_key = cand
                    break

        if lm_w_key is None:
            raise KeyError("Could not find lm_head weight matrix in state_dict.")

        lm_w = state_dict[lm_w_key]  # [vocab_size, hidden_size] e.g. [51200, 2048]
        self.lm_head_weight = ttnn.from_torch(lm_w.T.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        if lm_b_key is not None and lm_b_key in state_dict:
            self.lm_head_bias = ttnn.from_torch(
                state_dict[lm_b_key].reshape(1, 1, 1, -1).contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        else:
            self.lm_head_bias = None

    def __call__(
        self,
        x: typing.Union[ttnn.Tensor, torch.Tensor],
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> ttnn.Tensor:
        hidden_states = self.model(x, start_pos=start_pos, use_cache=use_cache)
        logits = ttnn.linear(
            hidden_states,
            self.lm_head_weight,
            bias=self.lm_head_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden_states)

        # Same rank-promotion guard as TTPhi1Attention/TTPhi1MLP: ttnn.linear can return rank-4
        # output even for rank-3 input in some configs. A rank-4 logits tensor would break
        # demo.py's strict PCC shape check and the generation loop's slicing/argmax/cat chain.
        if len(logits.shape) == 4 and logits.shape[1] == 1:
            logits_3d = ttnn.reshape(logits, (logits.shape[0], logits.shape[2], logits.shape[3]))
            ttnn.deallocate(logits)
            logits = logits_3d

        return logits
