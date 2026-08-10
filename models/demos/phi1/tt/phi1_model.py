# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Tenstorrent Bounty #18287: Bring up microsoft/phi-1 on Wormhole (N150/N300)
# Modular implementation inheriting design patterns from tt_transformers

import typing

import torch

import ttnn


class TTPhi1Attention:
    """
    Multi-Head Attention block for Phi-1 (`microsoft/phi-1`).
    Supports Partial RoPE (Rotary Position Embedding applied to a fraction of head dimensions)
    and routes through ttnn.scaled_dot_product_attention (SDPA) for hardware acceleration on Tensix cores.
    """

    def __init__(
        self,
        device: ttnn.Device,
        state_dict: typing.Dict[str, torch.Tensor],
        base_address: str,
        n_heads: int = 16,
        hidden_size: int = 2048,
        rotary_dim: int = 32,  # Phi-1 uses partial RoPE (e.g. 32 out of 128 head_dim)
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_heads
        self.rotary_dim = rotary_dim
        self.dtype = dtype

        # Load Wqkv combined or separate projection weights
        if f"{base_address}.self_attn.Wqkv.weight" in state_dict:
            wqkv_weight = state_dict[f"{base_address}.self_attn.Wqkv.weight"]
            wqkv_bias = state_dict[f"{base_address}.self_attn.Wqkv.bias"]
        elif f"{base_address}.self_attn.q_proj.weight" in state_dict:
            q_w = state_dict[f"{base_address}.self_attn.q_proj.weight"].view(
                self.n_heads, self.head_dim, self.hidden_size
            )
            k_w = state_dict[f"{base_address}.self_attn.k_proj.weight"].view(
                self.n_heads, self.head_dim, self.hidden_size
            )
            v_w = state_dict[f"{base_address}.self_attn.v_proj.weight"].view(
                self.n_heads, self.head_dim, self.hidden_size
            )
            wqkv_weight = torch.cat([q_w, k_w, v_w], dim=1).view(3 * self.hidden_size, self.hidden_size)

            q_b = state_dict[f"{base_address}.self_attn.q_proj.bias"].view(self.n_heads, self.head_dim)
            k_b = state_dict[f"{base_address}.self_attn.k_proj.bias"].view(self.n_heads, self.head_dim)
            v_b = state_dict[f"{base_address}.self_attn.v_proj.bias"].view(self.n_heads, self.head_dim)
            wqkv_bias = torch.cat([q_b, k_b, v_b], dim=1).view(3 * self.hidden_size)
        else:
            raise KeyError(f"Could not find Wqkv or q_proj/k_proj/v_proj in state_dict for {base_address}.self_attn")

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

        # Output projection (`out_proj` vs `dense`)
        if f"{base_address}.self_attn.out_proj.weight" in state_dict:
            out_weight = state_dict[f"{base_address}.self_attn.out_proj.weight"]
            out_bias = state_dict[f"{base_address}.self_attn.out_proj.bias"]
        elif f"{base_address}.self_attn.dense.weight" in state_dict:
            out_weight = state_dict[f"{base_address}.self_attn.dense.weight"]
            out_bias = state_dict[f"{base_address}.self_attn.dense.bias"]
        else:
            raise KeyError(f"Could not find out_proj or dense in state_dict for {base_address}.self_attn")

        self.out_proj = ttnn.from_torch(
            out_weight.T.contiguous(),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bout_proj = ttnn.from_torch(
            out_bias.reshape(1, 1, 1, -1).contiguous(),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x: ttnn.Tensor, rotary_pos_emb: typing.Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        # Project Q, K, V
        qkv = ttnn.linear(x, self.wqkv, bias=self.bqkv, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Ensure exact rank 3 (`[batch, seq_len, 3*hidden_size]`) required by split_query_key_value_and_split_heads C++ kernel
        if len(qkv.shape) == 4 and qkv.shape[1] == 1:
            qkv_new = ttnn.reshape(qkv, (qkv.shape[0], qkv.shape[2], qkv.shape[3]))
            ttnn.deallocate(qkv)
            qkv = qkv_new

        # Split QKV into separate tensors or heads
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            num_heads=self.n_heads,
        )
        ttnn.deallocate(qkv)

        # Apply Partial RoPE if provided
        if rotary_pos_emb is not None:
            # Slicing along the head_dim which is at dim 3
            q_rot = ttnn.slice(q, [0, 0, 0, 0], [q.shape[0], q.shape[1], q.shape[2], self.rotary_dim])
            q_pass = ttnn.slice(q, [0, 0, 0, self.rotary_dim], [q.shape[0], q.shape[1], q.shape[2], q.shape[3]])
            q_rot_new = ttnn.apply_rotary_position_embedding(q_rot, rotary_pos_emb, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(q_rot)

            q_new = ttnn.concat([q_rot_new, q_pass], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(q_rot_new)
            ttnn.deallocate(q_pass)
            ttnn.deallocate(q)
            q = q_new

            k_rot = ttnn.slice(k, [0, 0, 0, 0], [k.shape[0], k.shape[1], k.shape[2], self.rotary_dim])
            k_pass = ttnn.slice(k, [0, 0, 0, self.rotary_dim], [k.shape[0], k.shape[1], k.shape[2], k.shape[3]])
            k_rot_new = ttnn.apply_rotary_position_embedding(k_rot, rotary_pos_emb, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(k_rot)

            k_new = ttnn.concat([k_rot_new, k_pass], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(k_rot_new)
            ttnn.deallocate(k_pass)
            ttnn.deallocate(k)
            k = k_new

        # Hardware-accelerated Scaled Dot Product Attention (`ttnn.transformer.scaled_dot_product_attention`)
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Deallocate intermediate Q, K, V activations to prevent Tensix L1 memory exhaustion
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Concatenate heads back to `hidden_size`
        attn_out_concatenated = ttnn.transformer.concatenate_heads(
            attn_out,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_out)

        # Ensure rank matches input before final linear projection
        if len(attn_out_concatenated.shape) == 4 and attn_out_concatenated.shape[1] == 1:
            attn_out_conc_new = ttnn.reshape(
                attn_out_concatenated,
                (attn_out_concatenated.shape[0], attn_out_concatenated.shape[2], attn_out_concatenated.shape[3]),
            )
            ttnn.deallocate(attn_out_concatenated)
            attn_out_concatenated = attn_out_conc_new

        # Final linear projection
        output = ttnn.linear(
            attn_out_concatenated, self.out_proj, bias=self.bout_proj, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(attn_out_concatenated)
        return output


class TTPhi1MLP:
    """
    MLP block for Phi-1 (`fc1` -> NewGELU / GELU -> `fc2`).
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

        if f"{base_address}.mlp.fc1.weight" in state_dict:
            fc1_weight = state_dict[f"{base_address}.mlp.fc1.weight"]
            fc1_bias = state_dict[f"{base_address}.mlp.fc1.bias"]
            fc2_weight = state_dict[f"{base_address}.mlp.fc2.weight"]
            fc2_bias = state_dict[f"{base_address}.mlp.fc2.bias"]
        elif f"{base_address}.mlp.c_fc.weight" in state_dict:
            fc1_weight = state_dict[f"{base_address}.mlp.c_fc.weight"]
            fc1_bias = state_dict[f"{base_address}.mlp.c_fc.bias"]
            fc2_weight = state_dict[f"{base_address}.mlp.c_proj.weight"]
            fc2_bias = state_dict[f"{base_address}.mlp.c_proj.bias"]
        else:
            raise KeyError(f"Could not find fc1/fc2 or c_fc/c_proj in state_dict for {base_address}.mlp")

        self.fc1 = ttnn.from_torch(
            fc1_weight.T.contiguous(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.bfc1 = ttnn.from_torch(
            fc1_bias.reshape(1, 1, 1, -1).contiguous(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.fc2 = ttnn.from_torch(
            fc2_weight.T.contiguous(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.bfc2 = ttnn.from_torch(
            fc2_bias.reshape(1, 1, 1, -1).contiguous(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # fc1 projection + GELU activation
        h = ttnn.linear(x, self.fc1, bias=self.bfc1, memory_config=ttnn.L1_MEMORY_CONFIG)
        h_act = ttnn.gelu(h, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h)
        # fc2 projection
        out = ttnn.linear(h_act, self.fc2, bias=self.bfc2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h_act)
        return out


class TTPhi1DecoderLayer:
    """
    Single Transformer Layer for Phi-1 (`microsoft/phi-1`).
    Crucial Architecture Note: Phi-1 uses parallel residual connections:
      output = input + attn(input_norm) + mlp(input_norm)
    """

    def __init__(
        self,
        device: ttnn.Device,
        state_dict: typing.Dict[str, torch.Tensor],
        base_address: str,
        layer_num: int,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device

        # Resolve exact base address for this decoder layer across HF naming variations
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

        # Input LayerNorm (`input_layernorm` vs `ln_1`)
        if f"{self.base_address}.input_layernorm.weight" in state_dict:
            ln_weight = state_dict[f"{self.base_address}.input_layernorm.weight"]
            ln_bias = state_dict[f"{self.base_address}.input_layernorm.bias"]
        elif f"{self.base_address}.ln_1.weight" in state_dict:
            ln_weight = state_dict[f"{self.base_address}.ln_1.weight"]
            ln_bias = state_dict[f"{self.base_address}.ln_1.bias"]
        else:
            raise KeyError(f"Could not find input_layernorm or ln_1 in state_dict for {self.base_address}")

        self.ln_weight = ttnn.from_torch(
            ln_weight.reshape(1, 1, 1, -1).contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.ln_bias = ttnn.from_torch(
            ln_bias.reshape(1, 1, 1, -1).contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.self_attn = TTPhi1Attention(device, state_dict, self.base_address, dtype=dtype)
        self.mlp = TTPhi1MLP(device, state_dict, self.base_address, dtype=dtype)

    def __call__(self, x: ttnn.Tensor, rotary_pos_emb: typing.Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        # Layer Normalization
        normed_x = ttnn.layer_norm(x, weight=self.ln_weight, bias=self.ln_bias, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Parallel Attention & MLP evaluation
        attn_out = self.self_attn(normed_x, rotary_pos_emb=rotary_pos_emb)
        mlp_out = self.mlp(normed_x)
        ttnn.deallocate(normed_x)

        # Parallel Residual Sum: x + attn_out + mlp_out
        residual_sum = ttnn.add(attn_out, mlp_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(mlp_out)

        output = ttnn.add(x, residual_sum, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(residual_sum)
        return output


class TTPhi1Model:
    """
    Core Phi-1 Transformer backbone (`num_hidden_layers=24`).
    Stacks token embeddings, 24 TTPhi1DecoderLayers, and final LayerNorm.
    """

    def __init__(
        self,
        device: ttnn.Device,
        state_dict: typing.Dict[str, torch.Tensor],
        base_address: str = "model",
        num_hidden_layers: int = 24,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.base_address = base_address
        self.num_hidden_layers = num_hidden_layers
        self.dtype = dtype

        # 1. Token Embeddings
        embed_key = None
        for candidate in [
            f"{base_address}.embed_tokens.weight",
            "transformer.wte.weight",
            f"{base_address}.wte.weight",
            "model.embed_tokens.weight",
        ]:
            if candidate in state_dict:
                embed_key = candidate
                break
        if embed_key is None:
            raise KeyError(
                f"Could not find token embedding matrix in state_dict. Checked candidates for {base_address}.embed_tokens"
            )

        self.embed_tokens_torch = state_dict[embed_key]
        self.embed_tokens_device = ttnn.from_torch(
            self.embed_tokens_torch.to(torch.bfloat16).unsqueeze(0).unsqueeze(0),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # 2. Stack Decoder Layers (0 to num_hidden_layers - 1)
        self.layers = []
        for layer_num in range(num_hidden_layers):
            self.layers.append(
                TTPhi1DecoderLayer(
                    device=device, state_dict=state_dict, base_address=base_address, layer_num=layer_num, dtype=dtype
                )
            )

        # 3. Final LayerNorm (`final_layernorm` vs `norm` vs `ln_f`)
        norm_w_key, norm_b_key = None, None
        for candidate_prefix in [
            f"{base_address}.final_layernorm",
            f"{base_address}.norm",
            "transformer.ln_f",
            "model.final_layernorm",
        ]:
            if f"{candidate_prefix}.weight" in state_dict and f"{candidate_prefix}.bias" in state_dict:
                norm_w_key = f"{candidate_prefix}.weight"
                norm_b_key = f"{candidate_prefix}.bias"
                break

        if norm_w_key is None or norm_b_key is None:
            raise KeyError("Could not find final_layernorm weights (`weight`/`bias`) in state_dict.")

        norm_w = state_dict[norm_w_key].reshape(1, 1, 1, -1).contiguous()
        norm_b = state_dict[norm_b_key].reshape(1, 1, 1, -1).contiguous()
        self.final_norm_weight = ttnn.from_torch(norm_w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.final_norm_bias = ttnn.from_torch(norm_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(
        self, x: typing.Union[ttnn.Tensor, torch.Tensor], rotary_pos_emb: typing.Optional[ttnn.Tensor] = None
    ) -> ttnn.Tensor:
        # Handle input embedding if raw torch tensor or token ids are passed
        if isinstance(x, torch.Tensor):
            if x.dtype in [torch.long, torch.int, torch.int64, torch.int32]:
                # On-device embedding to prevent PCIe bottleneck
                x_device = ttnn.from_torch(x.to(torch.uint32), device=self.device)
                hidden_state = ttnn.embedding(x_device, self.embed_tokens_device, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(x_device)
            else:
                embedded = x.to(torch.bfloat16)
                hidden_state = ttnn.from_torch(
                    embedded,
                    dtype=self.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            can_deallocate = True
        else:
            hidden_state = x
            can_deallocate = False

        # Ensure hidden_state is strictly rank 3 (`[batch, seq_len, hidden_size]`) across all decoder layers
        if len(hidden_state.shape) == 4 and hidden_state.shape[1] == 1:
            hidden_state_new = ttnn.reshape(
                hidden_state, (hidden_state.shape[0], hidden_state.shape[2], hidden_state.shape[3])
            )
            if can_deallocate:
                ttnn.deallocate(hidden_state)
            hidden_state = hidden_state_new
            can_deallocate = True

        # Sequential evaluation across all stacked decoder layers
        for i, layer in enumerate(self.layers):
            prev_hidden_state = hidden_state
            hidden_state = layer(hidden_state, rotary_pos_emb=rotary_pos_emb)
            if (i > 0 or can_deallocate) and isinstance(prev_hidden_state, ttnn.Tensor):
                ttnn.deallocate(prev_hidden_state)

        # Apply final LayerNorm
        output = ttnn.layer_norm(
            hidden_state, weight=self.final_norm_weight, bias=self.final_norm_bias, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(hidden_state)
        return output


class TTPhi1ForCausalLM:
    """
    Top-level Causal LM model for microsoft/phi-1.
    Wraps TTPhi1Model backbone and applies LM Head projection to vocabulary (`vocab_size=51200`).
    """

    def __init__(
        self,
        device: ttnn.Device,
        state_dict: typing.Dict[str, torch.Tensor],
        base_address: str = "model",
        num_hidden_layers: int = 24,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self.model = TTPhi1Model(
            device=device,
            state_dict=state_dict,
            base_address=base_address,
            num_hidden_layers=num_hidden_layers,
            dtype=dtype,
        )

        # LM Head (`lm_head.weight` & `.bias` or `lm_head.linear.weight`)
        lm_w_key, lm_b_key = None, None
        for candidate_prefix in ["lm_head.linear", "lm_head", "model.lm_head"]:
            if f"{candidate_prefix}.weight" in state_dict:
                lm_w_key = f"{candidate_prefix}.weight"
                if f"{candidate_prefix}.bias" in state_dict:
                    lm_b_key = f"{candidate_prefix}.bias"
                break

        if lm_w_key is None:
            # Fallback check if weight sharing with embed_tokens is used
            for cand in ["model.embed_tokens.weight", "transformer.wte.weight"]:
                if cand in state_dict:
                    lm_w_key = cand
                    break

        if lm_w_key is None:
            raise KeyError("Could not find lm_head weight matrix in state_dict.")

        lm_w = state_dict[lm_w_key]  # shape: (vocab_size, hidden_size) e.g. (51200, 2048)
        lm_w_transposed = lm_w.T.contiguous()
        self.lm_head_weight = ttnn.from_torch(lm_w_transposed, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        if lm_b_key is not None and lm_b_key in state_dict:
            lm_b = state_dict[lm_b_key].reshape(1, 1, 1, -1).contiguous()
            self.lm_head_bias = ttnn.from_torch(lm_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        else:
            self.lm_head_bias = None

    def __call__(
        self, x: typing.Union[ttnn.Tensor, torch.Tensor], rotary_pos_emb: typing.Optional[ttnn.Tensor] = None
    ) -> ttnn.Tensor:
        hidden_states = self.model(x, rotary_pos_emb=rotary_pos_emb)
        logits = ttnn.linear(
            hidden_states, self.lm_head_weight, bias=self.lm_head_bias, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(hidden_states)
        return logits
