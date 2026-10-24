# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

# ============================================================
# Configuration Builder (reuse helpers from vision)
# ============================================================


def _get_optimal_subblock(per_core_M, per_core_N, max_product=8):
    """Find largest out_subblock_h × out_subblock_w that fits in register file."""
    best_h, best_w = 1, 1
    for h in range(1, per_core_M + 1):
        if per_core_M % h != 0:
            continue
        for w in range(1, per_core_N + 1):
            if per_core_N % w != 0:
                continue
            if h * w > max_product:
                continue
            if not (w == per_core_N or h == 1):
                continue
            if h * w > best_h * best_w:
                best_h, best_w = h, w
    return best_h, best_w


def _make_block_sharded_memcfg(core_grid, shard_height, shard_width):
    """Create a block-sharded memory config for the given core grid and shard shape."""
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1),
                    )
                }
            ),
            [shard_height, shard_width],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def build_text_encoder_configs(config, device, batch):
    """
    Build all memory configs and program configs for the text encoder.

    CLIP text encoder specs (ViT-B/32):
        hidden_size=512, intermediate_size=2048, num_heads=8, head_dim=64
        max_position_embeddings=77, padded to 96 (seqL_t=3)

    Args:
        config: HuggingFace CLIPTextConfig
        device: ttnn device (single ASIC)
        batch: batch size (must satisfy: batch * seqL_t % grid_y == 0)

    Returns:
        memory_configs: dict of ttnn.MemoryConfig
        program_configs: dict of ttnn program configs
    """
    TILE = 32
    device_grid = device.compute_with_storage_grid_size()

    # --- Tile dimension calculations ---
    dim = config.hidden_size  # 512
    dim_t = dim // TILE  # 16 tiles
    dim_t_x = dim_t // device_grid.x  # 2 tiles per column

    seqL = config.max_position_embeddings  # 77
    seqL_padded = ((seqL - 1) // TILE + 1) * TILE  # 96
    seqL_t = seqL_padded // TILE  # 3 tiles

    total_M_tiles = batch * seqL_t

    # Find the largest grid_y <= device grid that evenly divides total_M_tiles
    grid_y = device_grid.y
    while grid_y > 1 and total_M_tiles % grid_y != 0:
        grid_y -= 1

    per_core_M = total_M_tiles // grid_y

    class _Grid:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    core_grid = _Grid(device_grid.x, grid_y)

    # --- Per-core output widths in tiles ---
    per_core_N_hidden = dim_t_x  # 2  (512/32/8)
    per_core_N_mlp = dim_t_x * 4  # 8  (2048/32/8)

    # Text: dim_t_x = 2 (512/32/8), so fused QKV per-core width = 6 tiles
    per_core_N_qkv_fused = 3 * dim_t_x  # 3*2 = 6 tiles wide per core

    # --- Subblock calculations ---
    fc1_sub_h, fc1_sub_w = _get_optimal_subblock(per_core_M, per_core_N_mlp)
    fc2_sub_h, fc2_sub_w = _get_optimal_subblock(per_core_M, per_core_N_hidden)
    qkv_sub_h, qkv_sub_w = _get_optimal_subblock(per_core_M, per_core_N_hidden)
    out_sub_h, out_sub_w = _get_optimal_subblock(per_core_M, per_core_N_hidden)
    qkv_fused_sub_h, qkv_fused_sub_w = _get_optimal_subblock(per_core_M, per_core_N_qkv_fused)

    # --- Memory configs ---
    memory_configs = {
        "qkv_output": _make_block_sharded_memcfg(
            core_grid,
            per_core_M * TILE,
            per_core_N_qkv_fused * TILE,  # 6*32 = 192 cols per core
        ),
        # Hidden-sized block shard [batch×seqL_padded, 512]
        "hidden": _make_block_sharded_memcfg(
            core_grid,
            per_core_M * TILE,
            per_core_N_hidden * TILE,
        ),
        # MLP intermediate [batch×seqL_padded, 2048]
        "mlp_intermediate": _make_block_sharded_memcfg(
            core_grid,
            per_core_M * TILE,
            per_core_N_mlp * TILE,
        ),
    }

    # --- Program configs ---
    program_configs = {
        "qkv_fused": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=dim_t_x,
            out_subblock_h=qkv_fused_sub_h,
            out_subblock_w=qkv_fused_sub_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N_qkv_fused,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "layer_norm": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=dim_t_x,
            block_h=per_core_M,
            block_w=dim_t_x,
            inplace=False,
        ),
        "qkv_proj": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=dim_t_x,
            out_subblock_h=qkv_sub_h,
            out_subblock_w=qkv_sub_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N_hidden,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "out_proj": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=dim_t_x,
            out_subblock_h=out_sub_h,
            out_subblock_w=out_sub_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N_hidden,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "ff1": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=dim_t_x,
            out_subblock_h=fc1_sub_h,
            out_subblock_w=fc1_sub_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N_mlp,
            transpose_mcast=False,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
        ),
        "ff2": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=per_core_N_mlp,
            out_subblock_h=fc2_sub_h,
            out_subblock_w=fc2_sub_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N_hidden,
            transpose_mcast=False,
            fused_activation=None,
        ),
        "sdpa": ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            exp_approx_mode=False,
            q_chunk_size=96,
            k_chunk_size=96,
        ),
        "compute_kernel": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        ),
    }

    return memory_configs, program_configs


# ============================================================
# Text Embeddings
# ============================================================


class TtCLIPTextEmbeddings:
    """
    CLIP text embeddings: token_embedding + position_embedding.
    No patch embedding, no CLS token — just standard transformer embeddings.
    """

    def __init__(self, config, parameters, device, dtype=ttnn.bfloat8_b):
        self.device = device
        self.dtype = dtype
        self.embed_dim = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings

        # Embedding weights stay in row-major bfloat16 — ttnn.embedding lookups
        # expect ROW_MAJOR with bfloat16 storage regardless of the model's
        # activation dtype.
        self.token_embedding_weight = ttnn.from_torch(
            parameters.token_embedding.weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.position_ids = ttnn.from_torch(
            torch.arange(config.max_position_embeddings).unsqueeze(0),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        self.position_embedding_weight = ttnn.from_torch(
            parameters.position_embedding.weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] token IDs (uint32)
        Returns:
            embeddings: [batch, seq_len, hidden_size] in TILE layout
        """
        seq_len = input_ids.shape[1]

        # Token embeddings: [batch, seq_len] → [batch, seq_len, 512]
        token_embeds = ttnn.embedding(
            input_ids,
            self.token_embedding_weight,
            layout=ttnn.TILE_LAYOUT,
        )

        # Position embeddings: [1, max_pos, 512] → slice to [1, seq_len, 512]
        position_ids = ttnn.slice(self.position_ids, [0, 0], [1, seq_len])
        position_embeds = ttnn.embedding(
            position_ids,
            self.position_embedding_weight,
            layout=ttnn.TILE_LAYOUT,
        )

        embeddings = ttnn.add(token_embeds, position_embeds)
        ttnn.deallocate(token_embeds)

        return embeddings


# ============================================================
# Text MLP
# ============================================================


class TtCLIPTextMLP:
    def __init__(self, config, parameters, device, memory_configs, program_configs, dtype=ttnn.bfloat8_b):
        self.dtype = dtype
        self.memory_configs = memory_configs
        self.program_configs = program_configs

        self.fc1_weight = ttnn.from_torch(
            parameters.fc1.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.fc1_bias = ttnn.from_torch(
            parameters.fc1.bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.fc2_weight = ttnn.from_torch(
            parameters.fc2.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.fc2_bias = ttnn.from_torch(
            parameters.fc2.bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        fc1_weight = ttnn.to_memory_config(self.fc1_weight, ttnn.L1_MEMORY_CONFIG)
        fc1_bias = ttnn.to_memory_config(self.fc1_bias, ttnn.L1_MEMORY_CONFIG)

        intermediate = ttnn.linear(
            hidden_states,
            fc1_weight,
            bias=fc1_bias,
            memory_config=self.memory_configs["mlp_intermediate"],
            program_config=self.program_configs["ff1"],
            dtype=self.dtype,
            compute_kernel_config=self.program_configs["compute_kernel"],
        )

        ttnn.deallocate(hidden_states)
        ttnn.deallocate(fc1_weight)
        ttnn.deallocate(fc1_bias)

        fc2_weight = ttnn.to_memory_config(self.fc2_weight, ttnn.L1_MEMORY_CONFIG)
        fc2_bias = ttnn.to_memory_config(self.fc2_bias, ttnn.L1_MEMORY_CONFIG)

        output = ttnn.linear(
            intermediate,
            fc2_weight,
            bias=fc2_bias,
            memory_config=self.memory_configs["hidden"],
            program_config=self.program_configs["ff2"],
            dtype=self.dtype,
            compute_kernel_config=self.program_configs["compute_kernel"],
        )

        ttnn.deallocate(fc2_weight)
        ttnn.deallocate(fc2_bias)
        ttnn.deallocate(intermediate)

        return output


# ============================================================
# Text Attention
# ============================================================
'''

class TtCLIPTextAttention:
    """
    CLIP text attention with causal masking.

    Key difference from vision: is_causal=True in SDPA.
    Uses separate Q/K/V projections (same as vision) to avoid
    split_query_key_value_and_split_heads batch/head constraints.
    """

    def __init__(self, config, parameters, device, memory_configs, program_configs, dtype=ttnn.bfloat8_b):
        self.dtype = dtype
        self.num_heads = config.num_attention_heads  # 8
        self.embed_dim = config.hidden_size  # 512
        self.head_dim = config.hidden_size // self.num_heads  # 64
        self.scale = 1.0 / (self.head_dim**0.5)
        self.memory_configs = memory_configs
        self.program_configs = program_configs

        # Separate Q/K/V weights in DRAM
        self.q_proj_weight = ttnn.from_torch(
            parameters.q_proj.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.q_proj_bias = ttnn.from_torch(
            parameters.q_proj.bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.k_proj_weight = ttnn.from_torch(
            parameters.k_proj.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.k_proj_bias = ttnn.from_torch(
            parameters.k_proj.bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.v_proj_weight = ttnn.from_torch(
            parameters.v_proj.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.v_proj_bias = ttnn.from_torch(
            parameters.v_proj.bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.out_proj_weight = ttnn.from_torch(
            parameters.out_proj.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.out_proj_bias = ttnn.from_torch(
            parameters.out_proj.bias.reshape(1, -1),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Stage Q/K/V weights to L1
        q_weight = ttnn.to_memory_config(self.q_proj_weight, ttnn.L1_MEMORY_CONFIG)
        q_bias = ttnn.to_memory_config(self.q_proj_bias, ttnn.L1_MEMORY_CONFIG)
        k_weight = ttnn.to_memory_config(self.k_proj_weight, ttnn.L1_MEMORY_CONFIG)
        k_bias = ttnn.to_memory_config(self.k_proj_bias, ttnn.L1_MEMORY_CONFIG)
        v_weight = ttnn.to_memory_config(self.v_proj_weight, ttnn.L1_MEMORY_CONFIG)
        v_bias = ttnn.to_memory_config(self.v_proj_bias, ttnn.L1_MEMORY_CONFIG)

        # Separate Q/K/V projections: [batch, seq, 512] → [batch, seq, 512]
        query = ttnn.linear(
            hidden_states,
            q_weight,
            bias=q_bias,
            memory_config=self.memory_configs["hidden"],
            dtype=self.dtype,
            program_config=self.program_configs["qkv_proj"],
            compute_kernel_config=self.program_configs["compute_kernel"],
        )
        key = ttnn.linear(
            hidden_states,
            k_weight,
            bias=k_bias,
            memory_config=self.memory_configs["hidden"],
            dtype=self.dtype,
            program_config=self.program_configs["qkv_proj"],
            compute_kernel_config=self.program_configs["compute_kernel"],
        )
        value = ttnn.linear(
            hidden_states,
            v_weight,
            bias=v_bias,
            memory_config=self.memory_configs["hidden"],
            dtype=self.dtype,
            program_config=self.program_configs["qkv_proj"],
            compute_kernel_config=self.program_configs["compute_kernel"],
        )

        ttnn.deallocate(hidden_states)
        ttnn.deallocate(q_weight)
        ttnn.deallocate(q_bias)
        ttnn.deallocate(k_weight)
        ttnn.deallocate(k_bias)
        ttnn.deallocate(v_weight)
        ttnn.deallocate(v_bias)

        # Move to interleaved for reshape/permute
        query = ttnn.to_memory_config(query, ttnn.L1_MEMORY_CONFIG)
        key = ttnn.to_memory_config(key, ttnn.L1_MEMORY_CONFIG)
        value = ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG)

        # Reshape to multi-head: [batch, seq, 512] → [batch, 8, seq, 64]
        query = ttnn.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        query = ttnn.permute(query, (0, 2, 1, 3))

        key = ttnn.reshape(key, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = ttnn.permute(key, (0, 2, 1, 3))

        value = ttnn.reshape(value, (batch_size, seq_len, self.num_heads, self.head_dim))
        value = ttnn.permute(value, (0, 2, 1, 3))

        # Scaled dot-product attention with CAUSAL MASK
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            is_causal=True,
            scale=self.scale,
            program_config=self.program_configs["sdpa"],
            compute_kernel_config=self.program_configs["compute_kernel"],
        )

        ttnn.deallocate(query)
        ttnn.deallocate(key)
        ttnn.deallocate(value)

        # Reshape back: [batch, 8, seq, 64] → [batch, seq, 512]
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.embed_dim))

        # Reshard to block for out_proj
        attn_output = ttnn.to_memory_config(attn_output, self.memory_configs["hidden"])

        # Stage out_proj weights to L1
        out_weight = ttnn.to_memory_config(self.out_proj_weight, ttnn.L1_MEMORY_CONFIG)
        out_bias = ttnn.to_memory_config(self.out_proj_bias, ttnn.L1_MEMORY_CONFIG)

        output = ttnn.linear(
            attn_output,
            out_weight,
            bias=out_bias,
            memory_config=self.memory_configs["hidden"],
            dtype=self.dtype,
            program_config=self.program_configs["out_proj"],
            compute_kernel_config=self.program_configs["compute_kernel"],
        )

        ttnn.deallocate(attn_output)
        ttnn.deallocate(out_weight)
        ttnn.deallocate(out_bias)

        return output
'''


class TtCLIPTextAttention:
    def __init__(self, config, parameters, device, memory_configs, program_configs, dtype=ttnn.bfloat8_b):
        self.num_heads = config.num_attention_heads  # 8
        self.embed_dim = config.hidden_size  # 512
        self.head_dim = config.hidden_size // self.num_heads  # 64
        self.scale = 1.0 / (self.head_dim**0.5)
        self.memory_configs = memory_configs
        self.program_configs = program_configs

        # Fuse Q/K/V weights along the output dimension
        # Each is [512, 512], fused is [512, 1536]
        qkv_weight = torch.cat(
            [
                parameters.q_proj.weight.T,
                parameters.k_proj.weight.T,
                parameters.v_proj.weight.T,
            ],
            dim=-1,
        )

        qkv_bias = torch.cat(
            [
                parameters.q_proj.bias,
                parameters.k_proj.bias,
                parameters.v_proj.bias,
            ],
            dim=-1,
        )

        self.qkv_weight = ttnn.from_torch(
            qkv_weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.qkv_bias = ttnn.from_torch(
            qkv_bias,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.out_proj_weight = ttnn.from_torch(
            parameters.out_proj.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.out_proj_bias = ttnn.from_torch(
            parameters.out_proj.bias.reshape(1, -1),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        qkv_weight = ttnn.to_memory_config(self.qkv_weight, ttnn.L1_MEMORY_CONFIG)
        qkv_bias = ttnn.to_memory_config(self.qkv_bias, ttnn.L1_MEMORY_CONFIG)

        # Single fused matmul: [batch*seq, 512] -> [batch*seq, 1536]
        fused = ttnn.linear(
            hidden_states,
            qkv_weight,
            bias=qkv_bias,
            memory_config=self.memory_configs["qkv_output"],
            dtype=ttnn.bfloat8_b,
            program_config=self.program_configs["qkv_fused"],
        )

        ttnn.deallocate(hidden_states)
        ttnn.deallocate(qkv_weight)
        ttnn.deallocate(qkv_bias)

        # Reshard to interleaved for the split op
        fused = ttnn.to_memory_config(fused, ttnn.L1_MEMORY_CONFIG)

        # Split fused into Q, K, V in multi-head format
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused,
            num_heads=self.num_heads,
            transpose_key=False,
        )
        ttnn.deallocate(fused)

        # Causal SDPA — the one difference from vision
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            is_causal=True,
            scale=self.scale,
            program_config=self.program_configs["sdpa"],
            compute_kernel_config=self.program_configs["compute_kernel"],
        )

        ttnn.deallocate(query)
        ttnn.deallocate(key)
        ttnn.deallocate(value)

        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.embed_dim))
        attn_output = ttnn.to_memory_config(attn_output, self.memory_configs["hidden"])

        out_weight = ttnn.to_memory_config(self.out_proj_weight, ttnn.L1_MEMORY_CONFIG)
        out_bias = ttnn.to_memory_config(self.out_proj_bias, ttnn.L1_MEMORY_CONFIG)

        output = ttnn.linear(
            attn_output,
            out_weight,
            bias=out_bias,
            memory_config=self.memory_configs["hidden"],
            dtype=ttnn.bfloat8_b,
            program_config=self.program_configs["out_proj"],
        )

        ttnn.deallocate(attn_output)
        ttnn.deallocate(out_weight)
        ttnn.deallocate(out_bias)

        return output


# ============================================================
# Text Encoder Layer
# ============================================================


class TtCLIPTextEncoderLayer:
    def __init__(self, config, parameters, device, memory_configs, program_configs, dtype=ttnn.bfloat8_b):
        self.config = config
        self.dtype = dtype
        self.memory_configs = memory_configs
        self.program_configs = program_configs

        TILE = 32

        self.self_attn = TtCLIPTextAttention(
            config,
            parameters.self_attn,
            device,
            memory_configs,
            program_configs,
            dtype=dtype,
        )

        self.layer_norm1_weight = ttnn.from_torch(
            parameters.layer_norm1.weight.reshape(1, config.hidden_size // TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self.layer_norm1_bias = ttnn.from_torch(
            parameters.layer_norm1.bias.reshape(1, config.hidden_size // TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        self.mlp = TtCLIPTextMLP(
            config,
            parameters.mlp,
            device,
            memory_configs,
            program_configs,
            dtype=dtype,
        )

        self.layer_norm2_weight = ttnn.from_torch(
            parameters.layer_norm2.weight.reshape(1, config.hidden_size // TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self.layer_norm2_bias = ttnn.from_torch(
            parameters.layer_norm2.bias.reshape(1, config.hidden_size // TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        residual = hidden_states

        ln1_w = ttnn.to_memory_config(self.layer_norm1_weight, ttnn.L1_MEMORY_CONFIG)
        ln1_b = ttnn.to_memory_config(self.layer_norm1_bias, ttnn.L1_MEMORY_CONFIG)

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=ln1_w,
            bias=ln1_b,
            epsilon=self.config.layer_norm_eps,
            memory_config=self.memory_configs["hidden"],
            program_config=self.program_configs["layer_norm"],
        )

        ttnn.deallocate(ln1_w)
        ttnn.deallocate(ln1_b)

        hidden_states = self.self_attn(hidden_states)

        hidden_states = ttnn.add(
            residual,
            hidden_states,
            memory_config=self.memory_configs["hidden"],
            dtype=self.dtype,
        )
        ttnn.deallocate(residual)

        residual = hidden_states

        ln2_w = ttnn.to_memory_config(self.layer_norm2_weight, ttnn.L1_MEMORY_CONFIG)
        ln2_b = ttnn.to_memory_config(self.layer_norm2_bias, ttnn.L1_MEMORY_CONFIG)

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=ln2_w,
            bias=ln2_b,
            epsilon=self.config.layer_norm_eps,
            memory_config=self.memory_configs["hidden"],
            program_config=self.program_configs["layer_norm"],
        )

        ttnn.deallocate(ln2_w)
        ttnn.deallocate(ln2_b)

        hidden_states = self.mlp(hidden_states)

        hidden_states = ttnn.add(
            residual,
            hidden_states,
            memory_config=self.memory_configs["hidden"],
            dtype=self.dtype,
        )
        ttnn.deallocate(residual)

        return hidden_states


# ============================================================
# Text Model
# ============================================================


class TtCLIPTextModel:
    """
    Optimized CLIP text encoder.

    Architecture: token_embedding + position_embedding → 12 × encoder layers → final_layer_norm
    Pooling (EOS token extraction) is handled by the parent CLIP model, not here.
    This model returns the full sequence output [batch, seq_len, hidden_size].
    """

    def __init__(self, config, parameters, device, memory_configs, program_configs, dtype=ttnn.bfloat8_b):
        self.device = device
        self.config = config
        self.dtype = dtype
        self.memory_configs = memory_configs
        self.program_configs = program_configs

        TILE = 32

        self.embeddings = TtCLIPTextEmbeddings(config, parameters.embeddings, device, dtype=dtype)

        self.encoder_layers = []
        for lix in range(config.num_hidden_layers):
            layer = TtCLIPTextEncoderLayer(
                config,
                parameters.encoder.layers[lix],
                device,
                memory_configs,
                program_configs,
                dtype=dtype,
            )
            self.encoder_layers.append(layer)

        # Final layer norm (text has no pre_layernorm, only final)
        self.final_layer_norm_weight = ttnn.from_torch(
            parameters.final_layer_norm.weight.reshape(1, config.hidden_size // TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self.final_layer_norm_bias = ttnn.from_torch(
            parameters.final_layer_norm.bias.reshape(1, config.hidden_size // TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    def __call__(self, input_ids: ttnn.Tensor, **kwargs) -> ttnn.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] token IDs (uint32)
        Returns:
            hidden_states: [batch, seq_len, hidden_size] — full sequence output
                           Pooling (EOS extraction) handled by parent CLIP model.
        """
        hidden_states = self.embeddings(input_ids)

        hidden_states = ttnn.to_memory_config(hidden_states, self.memory_configs["hidden"])

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)

        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.final_layer_norm_weight,
            bias=self.final_layer_norm_bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return hidden_states
