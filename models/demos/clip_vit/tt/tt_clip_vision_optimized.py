import torch

import ttnn

# ============================================================
# Configuration Builder
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


def build_vision_encoder_configs(config, device, batch):
    """
    Build all memory configs and program configs for the vision encoder.

    Args:
        config: HuggingFace CLIPVisionConfig
        device: ttnn device (single ASIC)
        batch: batch size (must satisfy: batch * seqL_t % grid_y == 0)

    Returns:
        memory_configs: dict of ttnn.MemoryConfig
        program_configs: dict of ttnn program configs
    """
    TILE = 32
    device_grid = device.compute_with_storage_grid_size()

    # --- Tile dimension calculations ---
    dim = config.hidden_size  # 768
    dim_t = dim // TILE  # 24 tiles
    dim_t_x = dim_t // device_grid.x  # 3 tiles per column

    patch_count = config.image_size // config.patch_size
    seqL = patch_count * patch_count + 1  # 50 for ViT-B/32 (49 patches + CLS)
    seqL_padded = ((seqL - 1) // TILE + 1) * TILE  # 64
    seqL_t = seqL_padded // TILE  # 2 tiles

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
    per_core_N_hidden = dim_t_x  # 3  (768/32/8)
    per_core_N_mlp = dim_t_x * 4  # 12 (3072/32/8)
    per_core_N_qkv = 3 * dim_t // core_grid.x  # 9  (2304/32/8)

    # --- Subblock calculations ---
    fc1_sub_h, fc1_sub_w = _get_optimal_subblock(per_core_M, per_core_N_mlp)
    fc2_sub_h, fc2_sub_w = _get_optimal_subblock(per_core_M, per_core_N_hidden)
    qkv_sub_h, qkv_sub_w = _get_optimal_subblock(per_core_M, per_core_N_qkv)
    out_sub_h, out_sub_w = _get_optimal_subblock(per_core_M, per_core_N_hidden)

    # --- Memory configs ---
    memory_configs = {
        # Hidden-sized block shard [batch×seqL_padded, 768]
        "hidden": _make_block_sharded_memcfg(
            core_grid,
            per_core_M * TILE,
            per_core_N_hidden * TILE,
        ),
        # QKV output [batch×seqL_padded, 2304]
        "qkv_output": _make_block_sharded_memcfg(
            core_grid,
            per_core_M * TILE,
            per_core_N_qkv * TILE,
        ),
        # MLP intermediate [batch×seqL_padded, 3072]
        "mlp_intermediate": _make_block_sharded_memcfg(
            core_grid,
            per_core_M * TILE,
            per_core_N_mlp * TILE,
        ),
    }

    # --- Program configs ---
    program_configs = {
        "layer_norm": ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            subblock_w=dim_t_x,
            block_h=per_core_M,
            block_w=dim_t_x,
            inplace=False,
        ),
        "qkv_linear": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=dim_t_x,
            out_subblock_h=qkv_sub_h,
            out_subblock_w=qkv_sub_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N_qkv,
            transpose_mcast=False,
            fused_activation=None,
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
            q_chunk_size=64,
            k_chunk_size=64,
        ),
        "compute_kernel": ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        ),
    }

    return memory_configs, program_configs


class TtCLIPVisionEmbeddings:
    def __init__(self, config, parameters, device, dtype=ttnn.bfloat8_b):
        self.device = device
        self.dtype = dtype
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        core_grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=core_grid.y, x=core_grid.x)

        patch_weight = parameters.patch_embedding.weight
        out_channels, c, kh, kw = patch_weight.shape

        patch_weight = torch.nn.functional.pad(patch_weight, (0, 0, 0, 0, 0, 4 - c))
        patch_weight = torch.permute(patch_weight, (2, 3, 1, 0))
        patch_weight = torch.reshape(patch_weight, (kh * kw * 4, out_channels))

        self.patch_embedding_weight = ttnn.from_torch(
            patch_weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.position_ids = ttnn.from_torch(
            torch.arange(self.num_positions).unsqueeze(0),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        self.position_embedding_weight = ttnn.from_torch(
            parameters.position_embedding.weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.class_embedding = ttnn.from_torch(
            parameters.class_embedding.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.position_embeddings = ttnn.embedding(
            self.position_ids,
            self.position_embedding_weight,
            layout=ttnn.TILE_LAYOUT,
        )

    def __call__(self, pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, n_channels, height, width = pixel_values.shape

        pixel_values = self._preprocess_pixels(pixel_values)

        patch_embeds = ttnn.linear(
            pixel_values,
            self.patch_embedding_weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.dtype,
            core_grid=self.core_grid,
        )
        ttnn.deallocate(pixel_values)

        patch_embeds = ttnn.to_layout(patch_embeds, layout=ttnn.ROW_MAJOR_LAYOUT)
        patch_embeds = ttnn.reshape(patch_embeds, (batch_size, self.num_patches, self.embed_dim))
        patch_embeds = ttnn.to_layout(patch_embeds, layout=ttnn.TILE_LAYOUT)

        class_embeds = ttnn.repeat(self.class_embedding, ttnn.Shape([batch_size, 1, 1]))
        embeddings = ttnn.concat([class_embeds, patch_embeds], dim=1)
        ttnn.deallocate(patch_embeds)
        ttnn.deallocate(class_embeds)

        embeddings = ttnn.add(embeddings, self.position_embeddings)

        return embeddings

    def _preprocess_pixels(self, pixel_values):
        batch_size, n_channels, height, width = pixel_values.shape
        pixel_values = ttnn.permute(pixel_values, [0, 2, 3, 1])
        pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.ROW_MAJOR_LAYOUT)
        pixel_values = ttnn.pad(pixel_values, padding=((0, 0), (0, 0), (0, 0), (0, 1)), value=0)
        pixel_values = ttnn.reshape(pixel_values, (batch_size, height, width // self.patch_size, 4 * self.patch_size))
        pixel_values = ttnn.fold(pixel_values, stride_h=self.patch_size, stride_w=1)
        pixel_values = ttnn.reallocate(pixel_values)
        pixel_values = ttnn.to_memory_config(pixel_values, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
        return pixel_values


class TtCLIPVisionMLP:
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

    def __call__(self, hidden_states: ttnn.Tensor, prefetch_fn=None) -> ttnn.Tensor:
        fc1_weight = ttnn.to_memory_config(self.fc1_weight, ttnn.L1_MEMORY_CONFIG)
        fc1_bias = ttnn.to_memory_config(self.fc1_bias, ttnn.L1_MEMORY_CONFIG)

        intermediate = ttnn.linear(
            hidden_states,
            fc1_weight,
            bias=fc1_bias,
            memory_config=self.memory_configs["mlp_intermediate"],
            program_config=self.program_configs["ff1"],
            compute_kernel_config=self.program_configs["compute_kernel"],
            dtype=self.dtype,
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
            compute_kernel_config=self.program_configs["compute_kernel"],
            dtype=self.dtype,
        )

        ttnn.deallocate(fc2_weight)
        ttnn.deallocate(fc2_bias)
        ttnn.deallocate(intermediate)

        return output


class TtCLIPVisionAttention:
    def __init__(self, config, parameters, device, memory_configs, program_configs, dtype=ttnn.bfloat8_b):
        self.dtype = dtype
        self.num_heads = config.num_attention_heads
        self.embed_dim = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = 1.0 / (self.head_dim**0.5)
        self.memory_configs = memory_configs
        self.program_configs = program_configs

        self.q_proj_weight = ttnn.from_torch(
            parameters.q_proj.weight.T, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.q_proj_bias = ttnn.from_torch(parameters.q_proj.bias, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        self.k_proj_weight = ttnn.from_torch(
            parameters.k_proj.weight.T, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.k_proj_bias = ttnn.from_torch(parameters.k_proj.bias, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        self.v_proj_weight = ttnn.from_torch(
            parameters.v_proj.weight.T, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.v_proj_bias = ttnn.from_torch(parameters.v_proj.bias, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
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

        q_weight = ttnn.to_memory_config(self.q_proj_weight, ttnn.L1_MEMORY_CONFIG)
        q_bias = ttnn.to_memory_config(self.q_proj_bias, ttnn.L1_MEMORY_CONFIG)
        k_weight = ttnn.to_memory_config(self.k_proj_weight, ttnn.L1_MEMORY_CONFIG)
        k_bias = ttnn.to_memory_config(self.k_proj_bias, ttnn.L1_MEMORY_CONFIG)
        v_weight = ttnn.to_memory_config(self.v_proj_weight, ttnn.L1_MEMORY_CONFIG)
        v_bias = ttnn.to_memory_config(self.v_proj_bias, ttnn.L1_MEMORY_CONFIG)

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

        query = ttnn.to_memory_config(query, ttnn.L1_MEMORY_CONFIG)
        key = ttnn.to_memory_config(key, ttnn.L1_MEMORY_CONFIG)
        value = ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG)

        query = ttnn.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        query = ttnn.permute(query, (0, 2, 1, 3))

        key = ttnn.reshape(key, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = ttnn.permute(key, (0, 2, 1, 3))

        value = ttnn.reshape(value, (batch_size, seq_len, self.num_heads, self.head_dim))
        value = ttnn.permute(value, (0, 2, 1, 3))

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            is_causal=False,
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
            dtype=self.dtype,
            program_config=self.program_configs["out_proj"],
            compute_kernel_config=self.program_configs["compute_kernel"],
        )

        ttnn.deallocate(attn_output)
        ttnn.deallocate(out_weight)
        ttnn.deallocate(out_bias)

        return output


class TtCLIPVisionEncoderLayer:
    def __init__(self, config, parameters, device, memory_configs, program_configs, dtype=ttnn.bfloat8_b):
        self.config = config
        self.dtype = dtype
        self.memory_configs = memory_configs
        self.program_configs = program_configs

        TILE = 32

        self.self_attn = TtCLIPVisionAttention(
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

        self.mlp = TtCLIPVisionMLP(
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

    def __call__(self, hidden_states: ttnn.Tensor, prefetch_fn=None) -> ttnn.Tensor:
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

        hidden_states = self.mlp(hidden_states, prefetch_fn=prefetch_fn)

        hidden_states = ttnn.add(
            residual,
            hidden_states,
            memory_config=self.memory_configs["hidden"],
            dtype=self.dtype,
        )

        ttnn.deallocate(residual)

        return hidden_states


class TtCLIPVisionModel:
    def __init__(self, config, parameters, device, memory_configs, program_configs, dtype=ttnn.bfloat8_b):
        self.device = device
        self.config = config
        self.dtype = dtype
        self.memory_configs = memory_configs
        self.program_configs = program_configs

        TILE = 32

        self.embeddings = TtCLIPVisionEmbeddings(config, parameters.embeddings, device, dtype=dtype)

        self.pre_layernorm_weight = ttnn.from_torch(
            parameters.pre_layrnorm.weight.reshape(1, config.hidden_size // TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self.pre_layernorm_bias = ttnn.from_torch(
            parameters.pre_layrnorm.bias.reshape(1, config.hidden_size // TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        self.encoder_layers = []
        for lix in range(config.num_hidden_layers):
            layer = TtCLIPVisionEncoderLayer(
                config,
                parameters.encoder.layers[lix],
                device,
                memory_configs,
                program_configs,
                dtype=dtype,
            )
            self.encoder_layers.append(layer)

        self.post_layernorm_weight = ttnn.from_torch(
            parameters.post_layernorm.weight.reshape(1, config.hidden_size // TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self.post_layernorm_bias = ttnn.from_torch(
            parameters.post_layernorm.bias.reshape(1, config.hidden_size // TILE, TILE),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    def __call__(self, pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states = self.embeddings(pixel_values)
        ttnn.deallocate(pixel_values)

        hidden_states = ttnn.to_memory_config(hidden_states, self.memory_configs["hidden"])

        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.pre_layernorm_weight,
            bias=self.pre_layernorm_bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=self.memory_configs["hidden"],
            program_config=self.program_configs["layer_norm"],
        )

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)

        batch_size = hidden_states.shape[0]
        hidden_size = hidden_states.shape[2]

        pooled_output = ttnn.slice(
            hidden_states,
            [0, 0, 0],
            [batch_size, 1, hidden_size],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden_states)

        pooled_output = ttnn.reshape(pooled_output, (batch_size, hidden_size))

        pooled_output = ttnn.layer_norm(
            pooled_output,
            weight=self.post_layernorm_weight,
            bias=self.post_layernorm_bias,
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return pooled_output
