# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional TTNN decoder layer for zai-org/GLM-4.7-Flash (Glm4MoeLiteForCausalLM).

Single-chip (1x1 mesh) implementation of the two GLM-4.7-Flash decoder layer kinds:

- ``dense``: layer 0 — MLA attention + dense SwiGLU MLP (intermediate 10240).
- ``moe``:   layers 1..46 — MLA attention + 64-expert top-4 sigmoid-routed MoE
  (moe_intermediate 1536) + 1 shared expert.

Attention is DeepSeek-style MLA computed in the *absorbed* form, with a paged
compressed-latent KV cache of width kv_lora_rank + qk_rope_head_dim = 576 and a
single KV head. This is the only cache contract under which the advertised
202752-token context survives to a single 32 GB chip for all 47 layers
(20 heads x 256 x 2 x bf16 = 20.5 KB/token/layer for an MHA cache vs 1.15 KB
here).

Prefill/decode contract
=======================

``prefill_forward(x, *, kv_cache, page_table, user_id=0, seq_len=None)``
    x: ttnn.Tensor [1, 1, S, 2048], bf16 TILE, on device. S is the *logical*
    sequence length (any 1 <= S <= max supported context; padding to the paged
    block size is handled internally and pad rows are never attended by valid
    queries). Fills the user's rows of the paged latent cache for positions
    [0, S) and returns hidden states [1, 1, S, 2048] (logical length restored).
    Processes the sequence in ``prefill_chunk_size`` chunks end-to-end, using
    ``ttnn.experimental.paged_fill_cache`` + ``ttnn.transformer.chunked_flash_mla_prefill``.

``decode_forward(x, *, kv_cache, page_table, cur_pos_tensor, rot_idxs)``
    x: ttnn.Tensor [1, 1, B, 2048] (B == max_batch_size), decode layout
    (seq dim 1, users on dim 2). cur_pos_tensor: int32 [B] device tensor of
    current positions p (the new token's 0-based position; the latent for it is
    written at p via ``paged_update_cache`` and attention covers [0, p]).
    rot_idxs: uint32 [1, B] device tensor, normally equal to cur_pos, used for
    the on-device RoPE cos/sin lookup. Returns [1, 1, B, 2048]. The pass is
    fully on-device (no torch / from_torch / to_torch) and trace-capturable:
    positions are tensors, shapes are static.

Weight loading boundary: ``FunctionalDecoder.from_state_dict`` takes the HF
per-layer state dict (keys relative to ``model.layers.<i>.``, canonical HF
checkpoint layout with per-expert ``mlp.experts.<e>.{gate,up,down}_proj.weight``)
and performs all torch-side conversion there. No torch at runtime.

Numerics notes:
- q_a_layernorm / kv_a_layernorm use eps=1e-6 (HF class default), the
  input/post-attention norms use config rms_norm_eps=1e-5.
- Router: logits and score arithmetic in fp32 (fp32 dest acc + fp32 tensors),
  top-4 selection via bf16 ttnn.topk of the biased scores. bf16 rounding is
  monotone, so selection can only differ from fp32 on exact bf16 ties.
- softmax scale is qk_head_dim**-0.5 = 256**-0.5 (HF contract), passed
  explicitly to the flash MLA ops (which see 576-wide latent vectors).
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32


class PagedCacheConfig:
    """Paged latent-cache geometry. block_size must be a multiple of 32."""

    def __init__(self, block_size: int, max_num_blocks: int):
        if block_size % TILE != 0:
            raise ValueError(f"block_size must be a multiple of {TILE}, got {block_size}")
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks

    @classmethod
    def for_context(cls, max_context: int, batch_size: int, block_size: int = 64):
        blocks_per_user = -(-max_context // block_size)
        return cls(block_size=block_size, max_num_blocks=blocks_per_user * batch_size)


def _ck(device, fidelity, fp32_acc):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )


def _rot_transformation_mat():
    import torch

    dhead = TILE
    m = torch.zeros(1, 1, dhead, dhead)
    m[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    m[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return m


def _meta_cos_sin(rope_theta: float, dim: int, max_pos: int):
    """Meta-interleaved cos/sin tables [1, 1, max_pos, dim]: [t1, t1, t2, t2, ...].

    GLM-4.7-Flash uses rope_interleave=True (pairs (x0,x1),(x2,x3),... each rotated
    by one frequency) — exactly the meta layout that rotary_embedding_llama's
    transformation matrix implements.
    """
    import torch

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [max_pos, dim//2]
    cos = torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2)
    sin = torch.stack((freqs.sin(), freqs.sin()), dim=-1).flatten(-2)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


class RopeSetup:
    """On-device rope tables + decode/prefill rotary_embedding_llama inputs."""

    def __init__(self, device, rope_theta: float, dim: int, max_pos: int, batch_size: int):
        self.device = device
        self.dim = dim
        self.max_pos = max_pos
        self.batch_size = batch_size

        cos, sin = _meta_cos_sin(rope_theta, dim, max_pos)
        self.cos_matrix = ttnn.from_torch(cos, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        self.sin_matrix = ttnn.from_torch(sin, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        # Prefill transformation matrix: single tile, DRAM.
        self.trans_mat_prefill = ttnn.from_torch(
            _rot_transformation_mat(), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

        # Decode: one core per batch slot, shard [32, 32] / [32, dim].
        grid = device.compute_with_storage_grid_size()
        assert batch_size <= grid.x * grid.y, f"batch {batch_size} exceeds core grid"
        self.batch_grid = ttnn.num_cores_to_corerangeset(batch_size, grid, row_wise=True)
        # Keep the decode transformation matrix persistent in DRAM only: a
        # persistent L1-sharded tensor clashes with the static circular-buffer
        # region of the large prefill flash kernels (observed at batch 32).
        # decode_mats() shards it into L1 transiently per decode call.
        trans_mat = _rot_transformation_mat().repeat(1, 1, batch_size, 1)
        self.trans_mem_decode = ttnn.create_sharded_memory_config(
            shape=(TILE, TILE),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.trans_mat_decode_dram = ttnn.from_torch(
            trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.decode_cs_mem = ttnn.create_sharded_memory_config(
            shape=(TILE, dim),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def prefill_mats(self, start: int, end: int):
        """cos/sin slices for absolute positions [start, end); both tile-aligned."""
        cos = ttnn.slice(self.cos_matrix, [0, 0, start, 0], [1, 1, end, self.dim])
        sin = ttnn.slice(self.sin_matrix, [0, 0, start, 0], [1, 1, end, self.dim])
        return cos, sin, self.trans_mat_prefill

    def decode_mats(self, rot_idxs):
        """cos/sin for per-user positions; rot_idxs: uint32 [1, B] on device."""
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, B, dim]
        sin = ttnn.unsqueeze_to_4D(sin)
        cos = ttnn.transpose(cos, 1, 2)  # [1, B, 1(32), dim]
        sin = ttnn.transpose(sin, 1, 2)
        cos = ttnn.to_memory_config(cos, self.decode_cs_mem)
        sin = ttnn.to_memory_config(sin, self.decode_cs_mem)
        trans = ttnn.to_memory_config(self.trans_mat_decode_dram, self.trans_mem_decode)
        return cos, sin, trans


class FunctionalDecoder(LightweightModule):
    """One GLM-4.7-Flash decoder layer (dense or moe kind) on a 1x1 mesh."""

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        max_batch_size=1,
        max_context=None,
        expert_dtype=ttnn.bfloat8_b,
        weight_dtype=ttnn.bfloat16,
        prefill_chunk_size=2048,
        paged_config=None,
    ):
        """Build the layer from an HF per-layer state dict (keys relative to
        ``model.layers.<layer_idx>.``, canonical checkpoint layout)."""
        import torch

        self = cls()
        self.device = mesh_device
        self.layer_idx = layer_idx
        mlp_layer_types = getattr(hf_config, "mlp_layer_types", None)
        if mlp_layer_types is not None:
            self.layer_kind = "moe" if mlp_layer_types[layer_idx] == "sparse" else "dense"
        else:
            self.layer_kind = "dense" if layer_idx < hf_config.first_k_dense_replace else "moe"

        # ---- config ----
        self.hidden = hf_config.hidden_size
        self.num_heads = hf_config.num_attention_heads
        self.q_lora_rank = hf_config.q_lora_rank
        self.kv_lora_rank = hf_config.kv_lora_rank
        self.qk_nope = hf_config.qk_nope_head_dim
        self.qk_rope = hf_config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope + self.qk_rope
        self.v_head_dim = hf_config.v_head_dim
        self.kvpe_dim = self.kv_lora_rank + self.qk_rope
        self.rms_eps = hf_config.rms_norm_eps
        self.lora_norm_eps = 1e-6  # HF Glm4MoeLiteRMSNorm default for q_a/kv_a norms
        self.scale = self.qk_head_dim**-0.5
        self.max_batch = max_batch_size
        self.max_context = max_context or hf_config.max_position_embeddings
        self.prefill_chunk_size = prefill_chunk_size
        self.n_experts = getattr(hf_config, "n_routed_experts", 0)
        self.top_k = getattr(hf_config, "num_experts_per_tok", 0)
        self.routed_scaling = getattr(hf_config, "routed_scaling_factor", 1.0)

        if prefill_chunk_size % TILE != 0:
            raise ValueError(f"prefill_chunk_size must be a multiple of {TILE}")

        self.paged_config = paged_config or PagedCacheConfig.for_context(self.max_context, max_batch_size)
        if prefill_chunk_size % self.paged_config.block_size != 0:
            raise ValueError("prefill_chunk_size must be a multiple of the paged block_size")

        dev = mesh_device
        self.ck_hifi4 = _ck(dev, ttnn.MathFidelity.HiFi4, True)
        self.ck_hifi2 = _ck(dev, ttnn.MathFidelity.HiFi2, True)
        # Prefill flash needs fp32 dest accumulation: with bf16 accumulators the
        # chunked flash softmax drifts as K grows (measured on the dense layer at
        # 202k context: end-window PCC 0.9936 with bf16 acc vs 0.9997 with fp32
        # acc; middle 0.9981 vs 0.99994). The decode flash op is exact without it
        # (0.99999 at position 202751), so decode keeps the cheaper config.
        self.ck_flash_prefill = _ck(dev, ttnn.MathFidelity.HiFi4, True)
        self.ck_flash_decode = _ck(dev, ttnn.MathFidelity.HiFi4, False)
        # Bound cores per head/batch so the decode reduction tree fits
        # (sdpa_decode: num_tree_reduction_rounds <= MAX_TREE_REDUCTION_ROUNDS).
        self.flash_decode_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
            max_cores_per_head_batch=8,
        )

        def linear_w(key, dtype=weight_dtype):
            w = state_dict[key].to(torch.float32)
            return ttnn.from_torch(
                w.T.contiguous(),
                device=dev,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def norm_w(key):
            w = state_dict[key].to(torch.float32)
            w = w.reshape(1, 1, w.shape[0] // TILE, TILE)
            return ttnn.from_torch(
                w, device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        # ---- attention weights ----
        self.wq_a = linear_w("self_attn.q_a_proj.weight")
        self.q_norm_w = norm_w("self_attn.q_a_layernorm.weight")
        self.wq_b = linear_w("self_attn.q_b_proj.weight")
        self.wkv_a = linear_w("self_attn.kv_a_proj_with_mqa.weight")
        self.kv_norm_w = norm_w("self_attn.kv_a_layernorm.weight")
        self.wo = linear_w("self_attn.o_proj.weight")

        kv_b = state_dict["self_attn.kv_b_proj.weight"].to(torch.float32)
        kv_b = kv_b.reshape(self.num_heads, self.qk_nope + self.v_head_dim, self.kv_lora_rank)
        w_uk = kv_b[:, : self.qk_nope, :].unsqueeze(0).contiguous()  # [1, nh, qk_nope, kv_lora]
        w_uv_t = kv_b[:, self.qk_nope :, :].transpose(-1, -2).unsqueeze(0).contiguous()  # [1, nh, kv_lora, v_head]
        self.w_uk = ttnn.from_torch(
            w_uk, device=dev, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.w_uv_t = ttnn.from_torch(
            w_uv_t, device=dev, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        self.input_norm_w = norm_w("input_layernorm.weight")
        self.post_norm_w = norm_w("post_attention_layernorm.weight")

        # ---- mlp weights ----
        if self.layer_kind == "dense":
            self.mlp_gate = linear_w("mlp.gate_proj.weight")
            self.mlp_up = linear_w("mlp.up_proj.weight")
            self.mlp_down = linear_w("mlp.down_proj.weight")
        else:
            # fp32 router weight: HF computes router logits in fp32 and top-4
            # selection is sensitive to sub-bf16 score gaps.
            self.gate_w = linear_w("mlp.gate.weight", dtype=ttnn.float32)  # [hidden, n_experts]
            bias = state_dict["mlp.gate.e_score_correction_bias"].to(torch.float32).reshape(1, 1, 1, self.n_experts)
            self.gate_bias = ttnn.from_torch(
                bias, device=dev, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            gate_stack = torch.stack(
                [state_dict[f"mlp.experts.{e}.gate_proj.weight"].to(torch.float32).T for e in range(self.n_experts)]
            ).unsqueeze(0)
            up_stack = torch.stack(
                [state_dict[f"mlp.experts.{e}.up_proj.weight"].to(torch.float32).T for e in range(self.n_experts)]
            ).unsqueeze(0)
            down_stack = torch.stack(
                [state_dict[f"mlp.experts.{e}.down_proj.weight"].to(torch.float32).T for e in range(self.n_experts)]
            ).unsqueeze(0)
            common = dict(
                device=dev, dtype=expert_dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            self.experts_gate = ttnn.from_torch(gate_stack.contiguous(), **common)  # [1, E, hidden, inter]
            self.experts_up = ttnn.from_torch(up_stack.contiguous(), **common)
            self.experts_down = ttnn.from_torch(down_stack.contiguous(), **common)  # [1, E, inter, hidden]
            self.shared_gate = linear_w("mlp.shared_experts.gate_proj.weight")
            self.shared_up = linear_w("mlp.shared_experts.up_proj.weight")
            self.shared_down = linear_w("mlp.shared_experts.down_proj.weight")

            # All-ones prefill sparsity for the dense all-expert prefill path.
            g_max = prefill_chunk_size // TILE
            ones = torch.ones(1, g_max, 1, self.n_experts)
            self.prefill_sparsity_ones = ttnn.from_torch(
                ones, device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            # ones src for the scatter of top-k selections (sized for the largest
            # token count seen by the router: a prefill chunk or the decode batch)
            self.scatter_ones = ttnn.from_torch(
                torch.ones(1, 1, max(max_batch_size, prefill_chunk_size, TILE), self.top_k),
                device=dev,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )

        # ---- rope ----
        rope_params = getattr(hf_config, "rope_parameters", None) or {}
        rope_theta = rope_params.get("rope_theta", getattr(hf_config, "rope_theta", None))
        if rope_theta is None:
            raise ValueError("rope_theta not found in hf_config")
        if rope_params.get("rope_type", "default") != "default":
            raise ValueError(f"unsupported rope_type {rope_params.get('rope_type')}: only default RoPE is implemented")
        self.rope = RopeSetup(dev, float(rope_theta), self.qk_rope, self.max_context, max_batch_size)

        # decode kvpe shard configs (one core per user). paged_update_cache's
        # static CBs scale with B*Wt per core (output CB), which together with
        # the input shard exceeds Blackhole L1 above 16 users at kvpe_dim 576,
        # so the update runs in groups of <= 16 users.
        grid_sz = dev.compute_with_storage_grid_size()
        self.kvpe_update_groups = []
        for g0 in range(0, max_batch_size, 16):
            g1 = min(g0 + 16, max_batch_size)
            mem = ttnn.create_sharded_memory_config(
                shape=(TILE, self.kvpe_dim),
                core_grid=ttnn.num_cores_to_corerangeset(g1 - g0, grid_sz, row_wise=True),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.kvpe_update_groups.append((g0, g1, mem))
        self.rope_in_decode_mem = ttnn.create_sharded_memory_config(
            shape=(TILE, self.qk_rope),
            core_grid=self.rope.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return self

    # ------------------------------------------------------------------ cache

    def allocate_kv_cache(self, dtype=ttnn.bfloat16):
        """Paged latent cache [max_num_blocks, 1, block_size, 576]."""
        return ttnn.zeros(
            (self.paged_config.max_num_blocks, 1, self.paged_config.block_size, self.kvpe_dim),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ------------------------------------------------------------------ norms

    def _rms(self, x, w, eps):
        return ttnn.rms_norm(x, epsilon=eps, weight=w, compute_kernel_config=self.ck_hifi4)

    # ------------------------------------------------------------------ shared mlp pieces

    def _swiglu_linear(self, x, w_gate, w_up, w_down, ck):
        g = ttnn.linear(x, w_gate, compute_kernel_config=ck)
        u = ttnn.linear(x, w_up, compute_kernel_config=ck)
        h = ttnn.multiply(ttnn.silu(g), u)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        out = ttnn.linear(h, w_down, compute_kernel_config=ck)
        ttnn.deallocate(h)
        return out

    # ------------------------------------------------------------------ sparse matmul config

    def _sparse_pc(self, m, n, k, cores=(8, 4), in0_block_w=8, out_subblock_w=1):
        """MatmulMultiCoreReuseMultiCast1DProgramConfig for ttnn.sparse_matmul,
        following models/demos/gpt_oss/tt/experts/config.py:_build_matmul_config
        (in0_block_w snapped to a divisor of Kt; out_subblock_w to a divisor of
        per_core_N so in1_num_subblocks is never 0)."""
        core_x, core_y = cores
        num_cores = core_x * core_y
        Nt = -(-n // TILE)
        per_core_N = -(-Nt // num_cores)
        Kt = -(-k // TILE)
        if Kt % in0_block_w != 0:
            divisors = [d for d in range(2, in0_block_w + 1) if Kt % d == 0]
            in0_block_w = max(divisors) if divisors else Kt
        out_subblock_w = min(out_subblock_w, 8)
        if per_core_N % out_subblock_w != 0:
            divisors = [d for d in range(1, out_subblock_w) if per_core_N % d == 0]
            out_subblock_w = max(divisors)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            out_block_h=1,
            out_block_w=out_subblock_w,
            per_core_M=max(TILE, m) // TILE,
            per_core_N=per_core_N,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    # ------------------------------------------------------------------ router

    def _routing_weights(self, x_flat):
        """Dense routing weights [1, 1, T, E]: score * top4mask, normalized, *1.8.

        x_flat: [1, 1, T, hidden]. Score arithmetic in fp32; selection via bf16
        topk of the biased scores (monotone rounding: only exact bf16 ties can
        differ from the fp32 reference selection).
        """
        x_f32 = ttnn.typecast(x_flat, ttnn.float32)
        logits = ttnn.linear(
            x_f32, self.gate_w, compute_kernel_config=self.ck_hifi4, dtype=ttnn.float32
        )  # [1,1,T,E] fp32
        ttnn.deallocate(x_f32)
        scores = ttnn.sigmoid_accurate(logits)
        ttnn.deallocate(logits)
        biased = ttnn.add(scores, self.gate_bias)  # fp32, bcast over rows
        # ttnn.topk needs bf16. Center the biased scores per token first (rank
        # preserving in fp32) so the bf16 rounding resolution applies to the
        # score *spread*, not to spread + offset; top-4 selection then only
        # differs from the fp32 reference on sub-ulp ties.
        row_mean = ttnn.mean(biased, dim=-1, keepdim=True)
        centered = ttnn.subtract(biased, row_mean)
        ttnn.deallocate(row_mean)
        ttnn.deallocate(biased)
        biased_bf16 = ttnn.typecast(centered, ttnn.bfloat16)
        ttnn.deallocate(centered)
        _, idx = ttnn.topk(biased_bf16, k=self.top_k, dim=-1, sorted=True)
        T = idx.shape[2]
        src = self.scatter_ones
        if src.shape[2] != T:
            src = ttnn.slice(self.scatter_ones, [0, 0, 0, 0], [1, 1, T, self.top_k])
        mask_bf16 = ttnn.scatter(ttnn.zeros_like(biased_bf16), dim=-1, index=idx, src=src)
        ttnn.deallocate(biased_bf16)
        mask = ttnn.typecast(mask_bf16, ttnn.float32)
        ttnn.deallocate(mask_bf16)
        ttnn.deallocate(idx)
        picked = ttnn.multiply(scores, mask)
        ttnn.deallocate(scores)
        ttnn.deallocate(mask)
        denom = ttnn.sum(picked, dim=-1, keepdim=True)  # [1,1,T,1]
        denom = ttnn.add(denom, 1e-20)
        inv = ttnn.reciprocal(denom)
        ttnn.deallocate(denom)
        weights = ttnn.multiply(picked, inv)  # bcast over last dim
        ttnn.deallocate(picked)
        ttnn.deallocate(inv)
        weights = ttnn.multiply(weights, self.routed_scaling)
        weights_bf16 = ttnn.typecast(weights, ttnn.bfloat16)
        ttnn.deallocate(weights)
        return weights_bf16  # [1, 1, T, E] bf16

    # ------------------------------------------------------------------ moe

    def _moe_prefill(self, x):
        """x: [1, 1, S, hidden] (S multiple of 32). Dense all-expert compute."""
        S = x.shape[2]
        G = S // TILE
        routing = self._routing_weights(x)  # [1,1,S,E]

        x_g = ttnn.reshape(x, (1, G, TILE, self.hidden))
        sparsity = self.prefill_sparsity_ones
        if sparsity.shape[1] != G:
            sparsity = ttnn.slice(self.prefill_sparsity_ones, [0, 0, 0, 0], [1, G, 1, self.n_experts])
        nnz = G * self.n_experts

        inter = self.experts_gate.shape[-1]

        def expert_mm(w):
            out = ttnn.sparse_matmul(
                x_g,
                w,
                sparsity=sparsity,
                nnz=nnz,
                program_config=self._sparse_pc(TILE, inter, self.hidden),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.ck_hifi2,
                dtype=ttnn.bfloat16,
            )  # [1, G, 1, E, 32, N]
            out = ttnn.transpose(out, 1, 3)  # [1, E, 1, G, 32, N]
            return ttnn.reshape(out, (1, self.n_experts, S, out.shape[-1]))

        gate = expert_mm(self.experts_gate)
        up = expert_mm(self.experts_up)
        h = ttnn.multiply(ttnn.silu(gate), up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        rw = ttnn.permute(routing, (0, 3, 2, 1))  # [1, E, S, 1]
        ttnn.deallocate(routing)
        sparsity_e = ttnn.slice(self.prefill_sparsity_ones, [0, 0, 0, 0], [1, 1, 1, self.n_experts])
        split = 1024
        routed = None
        for s0 in range(0, S, split):
            s1 = min(s0 + split, S)
            h_s = ttnn.slice(h, [0, 0, s0, 0], [1, self.n_experts, s1, inter]) if S > split else h
            down = ttnn.sparse_matmul(
                h_s,
                self.experts_down,
                sparsity=sparsity_e,
                nnz=self.n_experts,
                is_input_a_sparse=True,
                program_config=self._sparse_pc(s1 - s0, self.hidden, inter),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.ck_hifi2,
                dtype=ttnn.bfloat16,
            )  # [1, E, s, hidden]
            if h_s is not h:
                ttnn.deallocate(h_s)
            rw_s = ttnn.slice(rw, [0, 0, s0, 0], [1, self.n_experts, s1, 1]) if S > split else rw
            down = ttnn.multiply(down, rw_s)
            if rw_s is not rw:
                ttnn.deallocate(rw_s)
            part = ttnn.sum(down, dim=1, keepdim=True)  # [1, 1, s, hidden]
            ttnn.deallocate(down)
            if routed is None:
                routed = part
            else:
                new_routed = ttnn.concat([routed, part], dim=2)
                ttnn.deallocate(routed)
                ttnn.deallocate(part)
                routed = new_routed
        ttnn.deallocate(h)
        ttnn.deallocate(rw)
        # sparsity_e is a slice of prefill_sparsity_ones (aliases it when
        # prefill_chunk_size == 32); no explicit deallocate.

        shared = self._swiglu_linear(x, self.shared_gate, self.shared_up, self.shared_down, self.ck_hifi4)
        out = ttnn.add(routed, shared)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        return out

    def _moe_decode(self, x):
        """x: [1, 1, B, hidden]. Sparse compute over the union of the batch's experts."""
        B = x.shape[2]
        routing = self._routing_weights(x)  # [1,1,B,E] bf16
        union = ttnn.max(routing, dim=2, keepdim=True)  # [1,1,1,E]
        sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(union)

        inter = self.experts_gate.shape[-1]

        def expert_mm(w):
            out = ttnn.sparse_matmul(
                x,
                w,
                sparsity=sparsity,
                nnz=None,
                program_config=self._sparse_pc(B, inter, self.hidden),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.ck_hifi2,
                dtype=ttnn.bfloat16,
            )  # [1, 1, 1, E, B, N]
            return ttnn.reshape(out, (1, self.n_experts, B, out.shape[-1]))

        gate = expert_mm(self.experts_gate)
        up = expert_mm(self.experts_up)
        h = ttnn.multiply(ttnn.silu(gate), up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        down = ttnn.sparse_matmul(
            h,
            self.experts_down,
            sparsity=sparsity,
            nnz=None,
            is_input_a_sparse=True,
            program_config=self._sparse_pc(B, self.hidden, inter),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.ck_hifi2,
            dtype=ttnn.bfloat16,
        )  # [1, E, B, hidden]
        ttnn.deallocate(h)
        ttnn.deallocate(sparsity)
        rw = ttnn.permute(routing, (0, 3, 2, 1))  # [1, E, B, 1]
        ttnn.deallocate(routing)
        down = ttnn.multiply(down, rw)
        ttnn.deallocate(rw)
        routed = ttnn.sum(down, dim=1, keepdim=True)  # [1, 1, B, hidden]
        ttnn.deallocate(down)

        shared = self._swiglu_linear(x, self.shared_gate, self.shared_up, self.shared_down, self.ck_hifi4)
        out = ttnn.add(routed, shared)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        return out

    def _mlp(self, x, mode):
        if self.layer_kind == "dense":
            return self._swiglu_linear(x, self.mlp_gate, self.mlp_up, self.mlp_down, self.ck_hifi4)
        if mode == "prefill":
            return self._moe_prefill(x)
        return self._moe_decode(x)

    # ------------------------------------------------------------------ attention: prefill

    def _attn_prefill_chunk(self, x, kv_cache, page_table, user_id, chunk_start):
        """x: [1, 1, S_c, hidden] normed chunk at absolute positions
        [chunk_start, chunk_start + S_c). Fills the cache and returns attention
        output [1, 1, S_c, hidden]."""
        S_c = x.shape[2]
        cos, sin, trans = self.rope.prefill_mats(chunk_start, chunk_start + S_c)

        # --- KV path ---
        kv_a = ttnn.linear(x, self.wkv_a, compute_kernel_config=self.ck_hifi4)  # [1,1,S,576]
        kv_nope = ttnn.slice(kv_a, [0, 0, 0, 0], [1, 1, S_c, self.kv_lora_rank])
        kv_rope = ttnn.slice(kv_a, [0, 0, 0, self.kv_lora_rank], [1, 1, S_c, self.kvpe_dim])
        ttnn.deallocate(kv_a)
        kv_nope = self._rms(kv_nope, self.kv_norm_w, self.lora_norm_eps)
        kv_rope = ttnn.experimental.rotary_embedding_llama(kv_rope, cos, sin, trans, is_decode_mode=False)
        kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [1,1,S,576]
        ttnn.deallocate(kv_nope)
        ttnn.deallocate(kv_rope)

        # Fill this chunk's blocks of the user's paged cache.
        block = self.paged_config.block_size
        start_block = chunk_start // block
        end_block = (chunk_start + S_c) // block
        # NB: ttnn.slice of the page table can alias the input buffer (full-range
        # or view fast paths); page-table slices are never explicitly deallocated,
        # they are released when the Python reference drops.
        chunk_pt = ttnn.slice(page_table, [0, start_block], [page_table.shape[0], end_block])
        ttnn.experimental.paged_fill_cache(kv_cache, kvpe, page_table=chunk_pt, batch_idx=user_id)

        # --- Q path ---
        q = ttnn.linear(x, self.wq_a, compute_kernel_config=self.ck_hifi4)  # [1,1,S,q_lora]
        q = self._rms(q, self.q_norm_w, self.lora_norm_eps)
        q = ttnn.linear(q, self.wq_b, compute_kernel_config=self.ck_hifi4)  # [1,1,S,nh*qk_head]
        q = ttnn.reshape(q, (1, S_c, self.num_heads, self.qk_head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))  # [1, nh, S, qk_head]
        q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, self.num_heads, S_c, self.qk_nope])
        q_rope = ttnn.slice(q, [0, 0, 0, self.qk_nope], [1, self.num_heads, S_c, self.qk_head_dim])
        ttnn.deallocate(q)
        q_lat = ttnn.matmul(q_nope, self.w_uk, compute_kernel_config=self.ck_hifi4)  # [1, nh, S, kv_lora]
        ttnn.deallocate(q_nope)
        q_rope = ttnn.experimental.rotary_embedding_llama(q_rope, cos, sin, trans, is_decode_mode=False)
        q_abs = ttnn.concat([q_lat, q_rope], dim=-1)  # [1, nh, S, 576]
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_rope)
        # cos/sin are ttnn.slice views of the persistent rope tables and may
        # alias them when a chunk spans the whole table (max_context == chunk);
        # never explicitly deallocate them — the references drop at return.

        user_pt = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
        attn_lat = ttnn.transformer.chunked_flash_mla_prefill(
            q_abs,
            kv_cache,
            self.kv_lora_rank,
            user_pt,
            chunk_start_idx=chunk_start,
            scale=self.scale,
            compute_kernel_config=self.ck_flash_prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, nh, S, kv_lora]
        ttnn.deallocate(q_abs)
        ttnn.deallocate(kvpe)

        v = ttnn.matmul(attn_lat, self.w_uv_t, compute_kernel_config=self.ck_hifi4)  # [1, nh, S, v_head]
        ttnn.deallocate(attn_lat)
        v = ttnn.permute(v, (0, 2, 1, 3))  # [1, S, nh, v_head]
        v = ttnn.reshape(v, (1, 1, S_c, self.num_heads * self.v_head_dim))
        out = ttnn.linear(v, self.wo, compute_kernel_config=self.ck_hifi4)  # [1,1,S,hidden]
        ttnn.deallocate(v)
        return out

    # ------------------------------------------------------------------ attention: decode

    def _attn_decode(self, x, kv_cache, page_table, cur_pos_tensor, rot_idxs):
        """x: [1, 1, B, hidden] (normed). Returns [1, 1, B, hidden]."""
        B = x.shape[2]
        cos, sin, trans = self.rope.decode_mats(rot_idxs)

        # --- KV projections + rope (kvpe kept in DRAM for now) ---
        kv_a = ttnn.linear(x, self.wkv_a, compute_kernel_config=self.ck_hifi4)  # [1,1,B,576]
        kv_nope = ttnn.slice(kv_a, [0, 0, 0, 0], [1, 1, B, self.kv_lora_rank])
        kv_rope = ttnn.slice(kv_a, [0, 0, 0, self.kv_lora_rank], [1, 1, B, self.kvpe_dim])
        ttnn.deallocate(kv_a)
        kv_nope = self._rms(kv_nope, self.kv_norm_w, self.lora_norm_eps)
        kv_rope = ttnn.transpose(kv_rope, 1, 2, memory_config=self.rope_in_decode_mem)  # [1, B, 1(32), 64]
        kv_rope = ttnn.experimental.rotary_embedding_llama(kv_rope, cos, sin, trans, is_decode_mode=True)
        kv_rope = ttnn.to_memory_config(kv_rope, ttnn.DRAM_MEMORY_CONFIG)
        kv_rope = ttnn.transpose(kv_rope, 1, 2)  # [1, 1, B, 64]
        kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [1, 1, B, 576], DRAM
        ttnn.deallocate(kv_nope)
        ttnn.deallocate(kv_rope)

        # --- Q path ---
        q = ttnn.linear(x, self.wq_a, compute_kernel_config=self.ck_hifi4)
        q = self._rms(q, self.q_norm_w, self.lora_norm_eps)
        q = ttnn.linear(q, self.wq_b, compute_kernel_config=self.ck_hifi4)  # [1,1,B,nh*qk_head]
        q = ttnn.untilize(q)
        q = ttnn.reshape(q, (1, B, self.num_heads, self.qk_head_dim))
        q = ttnn.tilize_with_zero_padding(q, use_multicore=True)  # [1, B, nh(32), qk_head]
        q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, B, self.num_heads, self.qk_nope])
        q_rope = ttnn.slice(q, [0, 0, 0, self.qk_nope], [1, B, self.num_heads, self.qk_head_dim])
        ttnn.deallocate(q)
        q_nope = ttnn.transpose(q_nope, 1, 2)  # [1, nh, B, qk_nope]
        q_lat = ttnn.matmul(q_nope, self.w_uk, compute_kernel_config=self.ck_hifi4)  # [1, nh, B, kv_lora]
        ttnn.deallocate(q_nope)
        q_lat = ttnn.transpose(q_lat, 1, 2)  # [1, B, nh, kv_lora]
        q_rope = ttnn.to_memory_config(q_rope, self.rope_in_decode_mem)
        q_rope = ttnn.experimental.rotary_embedding_llama(q_rope, cos, sin, trans, is_decode_mode=True)
        q_rope = ttnn.to_memory_config(q_rope, ttnn.DRAM_MEMORY_CONFIG)
        q_abs = ttnn.concat([q_lat, q_rope], dim=-1)  # [1, B, nh, 576]
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_rope)
        # Free every L1-resident rope tensor before paged_update_cache: its
        # per-core static CBs scale with B*Wt (output CB is B x 18 tiles at
        # kvpe_dim 576), leaving only ~56 KB of L1 headroom at batch 32 — just
        # enough for the kvpe input shard alone.
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(trans)

        # --- cache update (kvpe sharded one-user-per-core, <= 16 users per call) ---
        for g0, g1, mem in self.kvpe_update_groups:
            kvpe_g = (
                kvpe
                if (g0 == 0 and g1 == B and len(self.kvpe_update_groups) == 1)
                else ttnn.slice(kvpe, [0, 0, g0, 0], [1, 1, g1, self.kvpe_dim])
            )
            pos_g = cur_pos_tensor if len(self.kvpe_update_groups) == 1 else ttnn.slice(cur_pos_tensor, [g0], [g1])
            pt_g = (
                page_table
                if len(self.kvpe_update_groups) == 1
                else ttnn.slice(page_table, [g0, 0], [g1, page_table.shape[1]])
            )
            kvpe_sh = ttnn.transpose(kvpe_g, 1, 2, memory_config=mem)  # [1, g, 1(32), 576]
            ttnn.experimental.paged_update_cache(kv_cache, kvpe_sh, update_idxs_tensor=pos_g, page_table=pt_g)
            ttnn.deallocate(kvpe_sh)
        ttnn.deallocate(kvpe)

        attn_lat = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_abs,
            kv_cache,
            head_dim_v=self.kv_lora_rank,
            page_table_tensor=page_table,
            cur_pos_tensor=cur_pos_tensor,
            scale=self.scale,
            program_config=self.flash_decode_pc,
            compute_kernel_config=self.ck_flash_decode,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, B, nh(padded), kv_lora]
        ttnn.deallocate(q_abs)
        attn_lat = ttnn.slice(attn_lat, [0, 0, 0, 0], [1, B, self.num_heads, self.kv_lora_rank])
        attn_lat = ttnn.transpose(attn_lat, 1, 2)  # [1, nh, B, kv_lora]
        v = ttnn.matmul(attn_lat, self.w_uv_t, compute_kernel_config=self.ck_hifi4)  # [1, nh, B, v_head]
        ttnn.deallocate(attn_lat)
        v = ttnn.transpose(v, 1, 2)  # [1, B, nh, v_head]
        v = ttnn.untilize(v)
        v = ttnn.reshape(v, (1, 1, B, self.num_heads * self.v_head_dim))
        v = ttnn.tilize_with_zero_padding(v, use_multicore=True)
        out = ttnn.linear(v, self.wo, compute_kernel_config=self.ck_hifi4)  # [1,1,B,hidden]
        ttnn.deallocate(v)
        return out

    # ------------------------------------------------------------------ public forwards

    def prefill_forward(self, x, *, kv_cache, page_table, user_id=0, seq_len=None, progress_cb=None):
        S = seq_len if seq_len is not None else x.shape[2]
        block = self.paged_config.block_size
        S_pad = -(-S // block) * block
        if x.shape[2] < S_pad:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, S_pad - x.shape[2]), (0, 0)], 0.0)
        elif x.shape[2] > S_pad:
            raise ValueError(f"input has {x.shape[2]} rows but logical seq_len is {S}")

        chunk = self.prefill_chunk_size
        n_chunks = -(-S_pad // chunk)
        out_acc = None
        for chunk_idx, start in enumerate(range(0, S_pad, chunk)):
            if progress_cb is not None:
                progress_cb(chunk_idx, n_chunks)
            end = min(start + chunk, S_pad)
            x_c = ttnn.slice(x, [0, 0, start, 0], [1, 1, end, self.hidden]) if S_pad > chunk else x

            h = self._rms(x_c, self.input_norm_w, self.rms_eps)
            attn = self._attn_prefill_chunk(h, kv_cache, page_table, user_id, start)
            ttnn.deallocate(h)
            res = ttnn.add(x_c, attn)
            ttnn.deallocate(attn)
            if x_c is not x:
                ttnn.deallocate(x_c)
            h2 = self._rms(res, self.post_norm_w, self.rms_eps)
            mlp = self._mlp(h2, "prefill")
            ttnn.deallocate(h2)
            out_c = ttnn.add(res, mlp)
            ttnn.deallocate(res)
            ttnn.deallocate(mlp)

            if out_acc is None:
                out_acc = out_c
            else:
                new_acc = ttnn.concat([out_acc, out_c], dim=2)
                ttnn.deallocate(out_acc)
                ttnn.deallocate(out_c)
                out_acc = new_acc

        if out_acc.shape[2] != S:
            out = ttnn.slice(out_acc, [0, 0, 0, 0], [1, 1, S, self.hidden])
            ttnn.deallocate(out_acc)
            return out
        return out_acc

    def decode_forward(self, x, *, kv_cache, page_table, cur_pos_tensor, rot_idxs):
        h = self._rms(x, self.input_norm_w, self.rms_eps)
        attn = self._attn_decode(h, kv_cache, page_table, cur_pos_tensor, rot_idxs)
        ttnn.deallocate(h)
        res = ttnn.add(x, attn)
        ttnn.deallocate(attn)
        h2 = self._rms(res, self.post_norm_w, self.rms_eps)
        mlp = self._mlp(h2, "decode")
        ttnn.deallocate(h2)
        out = ttnn.add(res, mlp)
        ttnn.deallocate(res)
        ttnn.deallocate(mlp)
        return out

    forward = decode_forward
