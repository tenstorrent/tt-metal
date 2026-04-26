"""Gemma4 attention (sliding-window + full) — single class, branches on
is_decode at __call__ time and on layer_type at __init__ time.

Phase 1.9.2 absorbs the four legacy attention bodies (sliding/full ×
prefill/decode) directly into this class as private methods. The
weight tensors (`fused_qkv_w`, `q_proj_w`, `k_proj_w`, `q_norm_w`,
`k_norm_w`, `o_proj_w`) are stored on the instance at __init__ time
and referenced as `self.X` throughout the bodies. All other per-call
state (KV caches, RoPE caches, masks, scalars) flows in via __call__
kwargs.

The dispatch in __call__ chooses the right method based on
(layer_type, is_decode) — no more lazy import of the legacy helpers.
"""
import ttnn


def _load_typed(torch_w, mesh_device, *, dim, dtype):
    """Two-step ttnn loader matching consteval's bf16→typecast pipeline.

    `dim=None` means replicate; otherwise shard along the given dim.
    For BFLOAT8_B targets we go bf16 → on-device → typecast, since
    ttnn.as_tensor's direct-to-bf8 path produces a different byte
    pattern than the on-device typecast that consteval uses.
    """
    if dim is None:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    else:
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=dim)
    bf16_t = ttnn.as_tensor(
        torch_w,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=mesh_device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        mesh_mapper=mapper,
    )
    if dtype == ttnn.DataType.BFLOAT16:
        return bf16_t
    out = ttnn.typecast(
        bf16_t, dtype,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(bf16_t, False)
    return out


class Attention:
    """Sliding-window or full attention.

    `layer_type` is fixed at `__init__` time:
      - "sliding": uses fused QKV (single weight tensor) and a V cache.
      - "full":    uses separate Q + K (no V; k_eq_v=True).

    `is_decode` is per-call (`__call__` kwarg):
      - True:  uses paged_update_cache (decode path).
      - False: uses concat (prefill path).
    """

    def __init__(
        self,
        *,
        layer_type,
        # sliding only:
        fused_qkv_w=None,
        # full only:
        q_proj_w=None,
        k_proj_w=None,
        # shared:
        q_norm_w,
        k_norm_w,
        o_proj_w,
    ):
        assert layer_type in ("sliding", "full"), layer_type
        if layer_type == "sliding":
            assert fused_qkv_w is not None
            assert q_proj_w is None and k_proj_w is None
        else:
            assert q_proj_w is not None and k_proj_w is not None
            assert fused_qkv_w is None
        self.layer_type = layer_type
        self.fused_qkv_w = fused_qkv_w
        self.q_proj_w = q_proj_w
        self.k_proj_w = k_proj_w
        self.q_norm_w = q_norm_w
        self.k_norm_w = k_norm_w
        self.o_proj_w = o_proj_w

    def __call__(self, x, *, is_decode, **kwargs):
        if self.layer_type == "sliding":
            if is_decode:
                return self._sliding_decode(x, **kwargs)
            else:
                return self._sliding_prefill(x, **kwargs)
        else:
            if is_decode:
                return self._full_decode(x, **kwargs)
            else:
                return self._full_prefill(x, **kwargs)

    @classmethod
    def from_consteval_sliding(cls, *, fused_qkv_w, q_norm_w, k_norm_w, o_proj_w):
        return cls(
            layer_type="sliding",
            fused_qkv_w=fused_qkv_w,
            q_norm_w=q_norm_w,
            k_norm_w=k_norm_w,
            o_proj_w=o_proj_w,
        )

    @classmethod
    def from_consteval_full(cls, *, q_proj_w, k_proj_w, q_norm_w, k_norm_w, o_proj_w):
        return cls(
            layer_type="full",
            q_proj_w=q_proj_w,
            k_proj_w=k_proj_w,
            q_norm_w=q_norm_w,
            k_norm_w=k_norm_w,
            o_proj_w=o_proj_w,
        )

    @classmethod
    def from_state_dict_sliding(cls, state_dict, layer_idx, mesh_device, *,
                                  proj_dtype=None, norm_dtype=None,
                                  mesh_size=4):
        """Build a sliding-attention layer from HF state_dict.

        Sliding attention uses a fused QKV weight assembled per-shard:
        each device gets q[i].T || k[i].T || v[i].T concatenated along
        the output dim, where q[i], k[i], v[i] are the per-device shards
        of q_proj, k_proj, v_proj split along their respective output dims.

        The naive `cat([q, k, v], dim=0).T → shard dim=1` is WRONG:
        it puts contiguous q-output, then k-output, then v-output, but
        consteval puts a BIT of q + a BIT of k + a BIT of v per device.
        We replicate the per-shard layout by chunking, transposing, and
        concatenating per device, then concatenating per-device shards
        along dim=1 to produce a single tensor for as_tensor + shard_dim=1.
        """
        import torch
        if proj_dtype is None:
            proj_dtype = ttnn.DataType.BFLOAT8_B
        if norm_dtype is None:
            norm_dtype = ttnn.DataType.BFLOAT16

        prefix = f"model.language_model.layers.{layer_idx}.self_attn"
        q = state_dict[f"{prefix}.q_proj.weight"].to(torch.float32)
        k = state_dict[f"{prefix}.k_proj.weight"].to(torch.float32)
        v = state_dict[f"{prefix}.v_proj.weight"].to(torch.float32)
        q_shards = list(q.chunk(mesh_size, dim=0))
        k_shards = list(k.chunk(mesh_size, dim=0))
        v_shards = list(v.chunk(mesh_size, dim=0))
        # Per-device order is K, Q, V (the consteval body fused as
        # `concat([k_t, q_t, v_t], dim=1)`; verified bit-equal before
        # consteval was retired in Phase 4).
        per_device = [
            torch.cat([ki.t(), qi.t(), vi.t()], dim=1)
            for qi, ki, vi in zip(q_shards, k_shards, v_shards)
        ]
        fused_t = torch.cat(per_device, dim=1).contiguous().to(torch.bfloat16)
        fused_qkv_w = _load_typed(fused_t, mesh_device, dim=1, dtype=proj_dtype)

        q_norm_w = _load_typed(
            state_dict[f"{prefix}.q_norm.weight"].to(torch.bfloat16),
            mesh_device, dim=None, dtype=norm_dtype,
        )
        k_norm_w = _load_typed(
            state_dict[f"{prefix}.k_norm.weight"].to(torch.bfloat16),
            mesh_device, dim=None, dtype=norm_dtype,
        )
        o_proj_w = _load_typed(
            state_dict[f"{prefix}.o_proj.weight"].to(torch.bfloat16),
            mesh_device, dim=1, dtype=proj_dtype,
        )

        return cls(
            layer_type="sliding",
            fused_qkv_w=fused_qkv_w,
            q_norm_w=q_norm_w, k_norm_w=k_norm_w, o_proj_w=o_proj_w,
        )

    @classmethod
    def from_state_dict_full(cls, state_dict, layer_idx, mesh_device, *,
                              proj_dtype=None, norm_dtype=None):
        """Build a full-attention layer from HF state_dict.

        Separate q_proj + k_proj (no v — k_eq_v=True). Stored in HF
        orientation [output, hidden]; sharded along dim=0.
        """
        import torch
        if proj_dtype is None:
            proj_dtype = ttnn.DataType.BFLOAT8_B
        if norm_dtype is None:
            norm_dtype = ttnn.DataType.BFLOAT16

        prefix = f"model.language_model.layers.{layer_idx}.self_attn"
        q_proj_w = _load_typed(
            state_dict[f"{prefix}.q_proj.weight"].to(torch.bfloat16),
            mesh_device, dim=0, dtype=proj_dtype,
        )
        k_proj_w = _load_typed(
            state_dict[f"{prefix}.k_proj.weight"].to(torch.bfloat16),
            mesh_device, dim=0, dtype=proj_dtype,
        )
        q_norm_w = _load_typed(
            state_dict[f"{prefix}.q_norm.weight"].to(torch.bfloat16),
            mesh_device, dim=None, dtype=norm_dtype,
        )
        k_norm_w = _load_typed(
            state_dict[f"{prefix}.k_norm.weight"].to(torch.bfloat16),
            mesh_device, dim=None, dtype=norm_dtype,
        )
        o_proj_w = _load_typed(
            state_dict[f"{prefix}.o_proj.weight"].to(torch.bfloat16),
            mesh_device, dim=1, dtype=proj_dtype,
        )

        return cls(
            layer_type="full",
            q_proj_w=q_proj_w, k_proj_w=k_proj_w,
            q_norm_w=q_norm_w, k_norm_w=k_norm_w, o_proj_w=o_proj_w,
        )

    def _sliding_decode(self, x, *, k_cache, v_cache, pos_ids, sliding_cos_cache, sliding_sin_cache, pos_typecast_11, causal_mask_logical_and, causal_mask_logical_not, var_185, var_186, var_190, var_191, var_192, var_193):
        """Sliding-window attention (Gemma3 style): fused-QKV matmul, q/k_norm,
        RoPE rotation, sliding-window masked SDPA, o_proj, distributed
        reduce-scatter. Decode-specific: writes the new K/V state into the
        paged KV cache via `ttnn.experimental.paged_update_cache`.

        Returns a 4-tuple matching the codegen-emitted local names of the
        sub-block's outputs:
          * `ttnn_reshape_46`  -- residual flow that feeds `post_attn_ln`
          * `ttnn_add_15`      -- pos-id increment surfaced to `_main`'s return
          * `ttnn_where_8`     -- sliding-window K-cache write (masked)
          * `ttnn_where_10`    -- sliding-window V-cache write (masked)

        Consumes (deallocates internally): `x`.
        Does NOT deallocate: `k_cache`, `v_cache` (the K/V state is written
        in-place via `paged_update_cache` -- the cache buffers persist for
        the next decode step); `pos_ids` (intentionally kept alive because
        for "before-full" sliding layers (4, 10, 16, 22, 28, 34, 40, 46,
        52, 58) the same input slot doubles as the next full layer's
        `update_idxs_tensor`); weight tensors, RoPE caches, position-id
        helpers, causal-mask helpers, var_*.
        """
        # Aliases that match the verbatim codegen op names below.
        ttnn_typecast_2 = sliding_cos_cache
        ttnn_typecast_3 = sliding_sin_cache
        ttnn_typecast_11 = pos_typecast_11
        ttnn_logical_and_0 = causal_mask_logical_and
        ttnn_logical_not_0 = causal_mask_logical_not
        runtime_a = k_cache
        runtime_b = v_cache
        runtime_c = pos_ids
        ttnn_multiply_21 = x

        ttnn_reshape_30 = ttnn.reshape(
            ttnn_multiply_21,
            [1, 1344],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_21, False)
        ttnn_all_gather_7 = ttnn.all_gather(
            input_tensor=ttnn_reshape_30,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_30, False)
        ttnn_matmul_8 = ttnn.matmul(
            ttnn_all_gather_7,
            self.fused_qkv_w,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_7, False)
        ttnn_slice_7 = ttnn.slice(
            ttnn_matmul_8,
            [0, 0],
            [1, 1024],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_8 = ttnn.slice(
            ttnn_matmul_8,
            [0, 1024],
            [1, 3072],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_9 = ttnn.slice(
            ttnn_matmul_8,
            [0, 3072],
            [1, 4096],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_8, False)
        ttnn_reshape_31 = ttnn.reshape(
            ttnn_slice_7,
            [1, 4, 1, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_7, False)
        ttnn_rms_norm_3 = ttnn.rms_norm(
            ttnn_reshape_31,
            epsilon=9.9999999747524271e-07,
            weight=self.k_norm_w,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_31, False)
        ttnn_reshape_32 = ttnn.reshape(
            ttnn_rms_norm_3,
            [1, 1, 4, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_multiply_22 = ttnn.multiply(
            ttnn_reshape_32,
            ttnn_typecast_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_32, False)
        ttnn_slice_10 = ttnn.slice(
            ttnn_rms_norm_3,
            [0, 0, 0, 128],
            [1, 4, 1, 256],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_neg_2 = ttnn.neg(
            ttnn_slice_10,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_10, False)
        ttnn_slice_11 = ttnn.slice(
            ttnn_rms_norm_3,
            [0, 0, 0, 0],
            [1, 4, 1, 128],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_3, False)
        ttnn_concat_3 = ttnn.concat(
            [ttnn_neg_2, ttnn_slice_11],
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_11, False)
        ttnn.deallocate(ttnn_neg_2, False)
        ttnn_reshape_33 = ttnn.reshape(
            ttnn_concat_3,
            [1, 1, 4, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_3, False)
        ttnn_multiply_23 = ttnn.multiply(
            ttnn_reshape_33,
            ttnn_typecast_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reshape_33, False)
        ttnn_add_14 = ttnn.add(
            ttnn_multiply_22,
            ttnn_multiply_23,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_23, False)
        ttnn.deallocate(ttnn_multiply_22, False)
        ttnn_reshape_34 = ttnn.reshape(
            ttnn_add_14,
            [1, 1024],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_14, False)
        ttnn_to_layout_12 = ttnn.to_layout(
            ttnn_reshape_34, ttnn.Layout.ROW_MAJOR, None, memory_config=None
        )
        ttnn.deallocate(ttnn_reshape_34, False)
        ttnn_embedding_6 = ttnn.embedding(
            var_186,
            ttnn_to_layout_12,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_12, False)
        ttnn_reshape_35 = ttnn.reshape(
            ttnn_embedding_6,
            [256, 1, 4, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_embedding_6, False)
        ttnn_permute_6 = ttnn.permute(
            ttnn_reshape_35,
            [1, 2, 0, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_35, False)
        ttnn_where_7 = ttnn.where(
            ttnn_logical_not_0,
            var_192,
            ttnn_permute_6,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_6, False)
        ttnn_permute_7 = ttnn.permute(
            runtime_a,
            [2, 0, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(runtime_a, False)
        ttnn_reshape_36 = ttnn.reshape(
            ttnn_permute_7,
            [256, 1024],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_7, False)
        ttnn_to_layout_13 = ttnn.to_layout(
            ttnn_reshape_36, ttnn.Layout.ROW_MAJOR, None, memory_config=None
        )
        ttnn.deallocate(ttnn_reshape_36, False)
        ttnn_embedding_7 = ttnn.embedding(
            var_190,
            ttnn_to_layout_13,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_13, False)
        ttnn_reshape_37 = ttnn.reshape(
            ttnn_embedding_7,
            [256, 1, 4, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_embedding_7, False)
        ttnn_permute_8 = ttnn.permute(
            ttnn_reshape_37,
            [1, 2, 0, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_37, False)
        ttnn_where_8 = ttnn.where(
            ttnn_logical_and_0,
            ttnn_where_7,
            ttnn_permute_8,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_8, False)
        ttnn.deallocate(ttnn_where_7, False)
        ttnn_reshape_38 = ttnn.reshape(
            ttnn_slice_9,
            [1, 4, 1, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_9, False)
        ttnn_rms_norm_4 = ttnn.rms_norm(
            ttnn_reshape_38,
            epsilon=9.9999999747524271e-07,
            weight=None,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_38, False)
        ttnn_reshape_39 = ttnn.reshape(
            ttnn_rms_norm_4,
            [1, 1024],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_4, False)
        ttnn_to_layout_14 = ttnn.to_layout(
            ttnn_reshape_39, ttnn.Layout.ROW_MAJOR, None, memory_config=None
        )
        ttnn.deallocate(ttnn_reshape_39, False)
        ttnn_embedding_8 = ttnn.embedding(
            var_186,
            ttnn_to_layout_14,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_14, False)
        ttnn_reshape_40 = ttnn.reshape(
            ttnn_embedding_8,
            [256, 1, 4, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_embedding_8, False)
        ttnn_permute_9 = ttnn.permute(
            ttnn_reshape_40,
            [1, 2, 0, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_40, False)
        ttnn_where_9 = ttnn.where(
            ttnn_logical_not_0,
            var_192,
            ttnn_permute_9,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_9, False)
        ttnn_permute_10 = ttnn.permute(
            runtime_b,
            [2, 0, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(runtime_b, False)
        ttnn_reshape_41 = ttnn.reshape(
            ttnn_permute_10,
            [256, 1024],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_10, False)
        ttnn_to_layout_15 = ttnn.to_layout(
            ttnn_reshape_41, ttnn.Layout.ROW_MAJOR, None, memory_config=None
        )
        ttnn.deallocate(ttnn_reshape_41, False)
        ttnn_embedding_9 = ttnn.embedding(
            var_190,
            ttnn_to_layout_15,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_15, False)
        ttnn_reshape_42 = ttnn.reshape(
            ttnn_embedding_9,
            [256, 1, 4, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_embedding_9, False)
        ttnn_permute_11 = ttnn.permute(
            ttnn_reshape_42,
            [1, 2, 0, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_42, False)
        ttnn_where_10 = ttnn.where(
            ttnn_logical_and_0,
            ttnn_where_9,
            ttnn_permute_11,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_11, False)
        ttnn.deallocate(ttnn_where_9, False)
        ttnn_to_layout_16 = ttnn.to_layout(
            runtime_c, ttnn.Layout.TILE, None, memory_config=None
        )
        # Note: layer 1's original codegen deallocated `var_9` here. For the
        # parameterized helper, we skip this dealloc — for "before-full" sliding
        # layers (4, 10, 16, 22, 28, 34, 40, 46, 52, 58) the runtime_c input
        # slot is also the next full layer's update_idxs_tensor, so freeing it
        # here would leave the full layer with a dangling reference.
        ttnn_add_15 = ttnn.add(
            ttnn_to_layout_16,
            var_185,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_16, False)
        ttnn_reshape_43 = ttnn.reshape(
            ttnn_slice_8,
            [1, 8, 1, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_8, False)
        ttnn_rms_norm_5 = ttnn.rms_norm(
            ttnn_reshape_43,
            epsilon=9.9999999747524271e-07,
            weight=self.q_norm_w,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_43, False)
        ttnn_multiply_24 = ttnn.multiply(
            ttnn_rms_norm_5,
            ttnn_typecast_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_12 = ttnn.slice(
            ttnn_rms_norm_5,
            [0, 0, 0, 128],
            [1, 8, 1, 256],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_neg_3 = ttnn.neg(
            ttnn_slice_12,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_12, False)
        ttnn_slice_13 = ttnn.slice(
            ttnn_rms_norm_5,
            [0, 0, 0, 0],
            [1, 8, 1, 128],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_5, False)
        ttnn_concat_4 = ttnn.concat(
            [ttnn_neg_3, ttnn_slice_13],
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_13, False)
        ttnn.deallocate(ttnn_neg_3, False)
        ttnn_multiply_25 = ttnn.multiply(
            ttnn_concat_4,
            ttnn_typecast_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_4, False)
        ttnn_add_16 = ttnn.add(
            ttnn_multiply_24,
            ttnn_multiply_25,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_25, False)
        ttnn.deallocate(ttnn_multiply_24, False)
        ttnn_repeat_interleave_2 = ttnn.repeat_interleave(
            ttnn_where_8,
            2,
            1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_repeat_interleave_3 = ttnn.repeat_interleave(
            ttnn_where_10,
            2,
            1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_15 = ttnn.typecast(
            ttnn_add_16,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_16, False)
        ttnn_typecast_16 = ttnn.typecast(
            ttnn_repeat_interleave_2,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_repeat_interleave_2, False)
        ttnn_matmul_9 = ttnn.matmul(
            ttnn_typecast_15,
            ttnn_typecast_16,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_typecast_16, False)
        ttnn.deallocate(ttnn_typecast_15, False)
        ttnn_add_17 = ttnn.add(
            ttnn_matmul_9,
            ttnn_typecast_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_9, False)
        ttnn_eq_1 = ttnn.eq(
            ttnn_add_17,
            var_193,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_logical_not_3 = ttnn.logical_not(
            ttnn_eq_1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_eq_1, False)
        ttnn_sum_1 = ttnn.sum(
            ttnn_logical_not_3,
            [3],
            True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_logical_not_3, False)
        ttnn_logical_not_4 = ttnn.logical_not(
            ttnn_sum_1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_sum_1, False)
        ttnn_softmax_1 = ttnn.softmax(
            ttnn_add_17,
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
            numeric_stable=True,
        )
        ttnn.deallocate(ttnn_add_17, False)
        ttnn_typecast_17 = ttnn.typecast(
            ttnn_logical_not_4,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_logical_not_4, False)
        ttnn_where_11 = ttnn.where(
            ttnn_typecast_17,
            var_191,
            ttnn_softmax_1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_17, False)
        ttnn.deallocate(ttnn_softmax_1, False)
        ttnn_typecast_18 = ttnn.typecast(
            ttnn_repeat_interleave_3,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_repeat_interleave_3, False)
        ttnn_matmul_10 = ttnn.matmul(
            ttnn_where_11,
            ttnn_typecast_18,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_typecast_18, False)
        ttnn.deallocate(ttnn_where_11, False)
        ttnn_typecast_19 = ttnn.typecast(
            ttnn_matmul_10,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_10, False)
        ttnn_reshape_44 = ttnn.reshape(
            ttnn_typecast_19,
            [1, 2048],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_19, False)
        ttnn_matmul_11 = ttnn.matmul(
            ttnn_reshape_44,
            self.o_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_reshape_44, False)
        ttnn_reshape_45 = ttnn.reshape(
            ttnn_matmul_11,
            [1, 1, 1, 5376],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_11, False)
        ttnn_reduce_scatter_2 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_45,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_45, False)
        ttnn_reshape_46 = ttnn.reshape(
            ttnn_reduce_scatter_2,
            [1, 1, 1344],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reduce_scatter_2, False)

        return (ttnn_reshape_46, ttnn_add_15, ttnn_where_8, ttnn_where_10)

    def _sliding_prefill(self, x, *, k_cache, v_cache, pos_ids, sliding_cos_cache, sliding_sin_cache, pos_reshape_15, pos_reshape_16, pos_typecast_11, causal_mask_logical_and, causal_mask_logical_not, var_185, var_186, var_187, var_190, var_192, var_193):
        """Sliding-window attention (Gemma3 style): fused-QKV matmul, q/k_norm,
        RoPE rotation, sliding-window masked SDPA, o_proj, distributed
        reduce-scatter.

        Returns a 4-tuple matching the codegen-emitted local names of the
        sub-block's outputs:
          * `ttnn_reshape_44`  — residual flow that feeds `post_attn_ln`
          * `ttnn_add_16`      — concat-form K-cache write
          * `ttnn_where_8`     — sliding-window K-cache write (masked)
          * `ttnn_where_10`    — sliding-window V-cache write (masked)
        The trailing three are pulled up through `_sliding_decoder_layer`'s
        return tuple so they end up in `_main`'s outer return list (the
        static-cache decode step re-reads them).

        Consumes (deallocates internally): `x`, `k_cache`, `v_cache`, `pos_ids`.
        Does NOT deallocate: weight tensors, RoPE caches, position-id helpers,
        causal-mask helpers, var_*.
        """
        # Aliases that match the verbatim codegen op names below.
        ttnn_typecast_2 = sliding_cos_cache
        ttnn_typecast_3 = sliding_sin_cache
        ttnn_reshape_15 = pos_reshape_15
        ttnn_reshape_16 = pos_reshape_16
        ttnn_typecast_11 = pos_typecast_11
        ttnn_logical_and_0 = causal_mask_logical_and
        ttnn_logical_not_0 = causal_mask_logical_not

        ttnn_reshape_30 = ttnn.reshape(
            x,
            [19, 1344],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(x, False)
        ttnn_all_gather_7 = ttnn.all_gather(
            input_tensor=ttnn_reshape_30,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_30, False)
        ttnn_matmul_8 = ttnn.matmul(
            ttnn_all_gather_7,
            self.fused_qkv_w,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_7, False)
        ttnn_slice_8 = ttnn.slice(
            ttnn_matmul_8,
            [0, 0],
            [19, 1024],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_9 = ttnn.slice(
            ttnn_matmul_8,
            [0, 1024],
            [19, 3072],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_10 = ttnn.slice(
            ttnn_matmul_8,
            [0, 3072],
            [19, 4096],
            [1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_8, False)
        ttnn_reshape_31 = ttnn.reshape(
            ttnn_slice_8,
            [1, 19, 4, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_8, False)
        ttnn_rms_norm_3 = ttnn.rms_norm(
            ttnn_reshape_31,
            epsilon=9.9999999747524271e-07,
            weight=self.k_norm_w,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_31, False)
        ttnn_multiply_22 = ttnn.multiply(
            ttnn_rms_norm_3,
            ttnn_typecast_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_11 = ttnn.slice(
            ttnn_rms_norm_3,
            [0, 0, 0, 128],
            [1, 19, 4, 256],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_neg_2 = ttnn.neg(
            ttnn_slice_11,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_11, False)
        ttnn_slice_12 = ttnn.slice(
            ttnn_rms_norm_3,
            [0, 0, 0, 0],
            [1, 19, 4, 128],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_3, False)
        ttnn_concat_6 = ttnn.concat(
            [ttnn_neg_2, ttnn_slice_12],
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_12, False)
        ttnn.deallocate(ttnn_neg_2, False)
        ttnn_multiply_23 = ttnn.multiply(
            ttnn_concat_6,
            ttnn_typecast_3,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_6, False)
        ttnn_add_15 = ttnn.add(
            ttnn_multiply_22,
            ttnn_multiply_23,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_23, False)
        ttnn.deallocate(ttnn_multiply_22, False)
        ttnn_permute_11 = ttnn.permute(
            ttnn_add_15,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_32 = ttnn.reshape(
            ttnn_add_15,
            [19, 1024],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_15, False)
        ttnn_to_layout_12 = ttnn.to_layout(
            ttnn_reshape_32, ttnn.Layout.ROW_MAJOR, None, memory_config=None
        )
        ttnn.deallocate(ttnn_reshape_32, False)
        ttnn_embedding_6 = ttnn.embedding(
            var_186,
            ttnn_to_layout_12,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_12, False)
        ttnn_reshape_33 = ttnn.reshape(
            ttnn_embedding_6,
            [256, 1, 4, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_embedding_6, False)
        ttnn_permute_12 = ttnn.permute(
            ttnn_reshape_33,
            [1, 2, 0, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_33, False)
        ttnn_where_7 = ttnn.where(
            ttnn_logical_not_0,
            var_192,
            ttnn_permute_12,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_12, False)
        ttnn_permute_13 = ttnn.permute(
            k_cache,
            [2, 0, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_34 = ttnn.reshape(
            ttnn_permute_13,
            [256, 1024],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_13, False)
        ttnn_to_layout_13 = ttnn.to_layout(
            ttnn_reshape_34, ttnn.Layout.ROW_MAJOR, None, memory_config=None
        )
        ttnn.deallocate(ttnn_reshape_34, False)
        ttnn_embedding_7 = ttnn.embedding(
            var_190,
            ttnn_to_layout_13,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_13, False)
        ttnn_reshape_35 = ttnn.reshape(
            ttnn_embedding_7,
            [256, 1, 4, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_embedding_7, False)
        ttnn_permute_14 = ttnn.permute(
            ttnn_reshape_35,
            [1, 2, 0, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_35, False)
        ttnn_where_8 = ttnn.where(
            ttnn_logical_and_0,
            ttnn_where_7,
            ttnn_permute_14,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_14, False)
        ttnn.deallocate(ttnn_where_7, False)
        ttnn_reshape_36 = ttnn.reshape(
            ttnn_slice_10,
            [1, 19, 4, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_10, False)
        ttnn_rms_norm_4 = ttnn.rms_norm(
            ttnn_reshape_36,
            epsilon=9.9999999747524271e-07,
            weight=None,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_36, False)
        ttnn_permute_15 = ttnn.permute(
            ttnn_rms_norm_4,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_37 = ttnn.reshape(
            ttnn_rms_norm_4,
            [19, 1024],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_4, False)
        ttnn_to_layout_14 = ttnn.to_layout(
            ttnn_reshape_37, ttnn.Layout.ROW_MAJOR, None, memory_config=None
        )
        ttnn.deallocate(ttnn_reshape_37, False)
        ttnn_embedding_8 = ttnn.embedding(
            var_186,
            ttnn_to_layout_14,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_14, False)
        ttnn_reshape_38 = ttnn.reshape(
            ttnn_embedding_8,
            [256, 1, 4, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_embedding_8, False)
        ttnn_permute_16 = ttnn.permute(
            ttnn_reshape_38,
            [1, 2, 0, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_38, False)
        ttnn_where_9 = ttnn.where(
            ttnn_logical_not_0,
            var_192,
            ttnn_permute_16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_16, False)
        ttnn_permute_17 = ttnn.permute(
            v_cache,
            [2, 0, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_reshape_39 = ttnn.reshape(
            ttnn_permute_17,
            [256, 1024],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_17, False)
        ttnn_to_layout_15 = ttnn.to_layout(
            ttnn_reshape_39, ttnn.Layout.ROW_MAJOR, None, memory_config=None
        )
        ttnn.deallocate(ttnn_reshape_39, False)
        ttnn_embedding_9 = ttnn.embedding(
            var_190,
            ttnn_to_layout_15,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_15, False)
        ttnn_reshape_40 = ttnn.reshape(
            ttnn_embedding_9,
            [256, 1, 4, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_embedding_9, False)
        ttnn_permute_18 = ttnn.permute(
            ttnn_reshape_40,
            [1, 2, 0, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_reshape_40, False)
        ttnn_where_10 = ttnn.where(
            ttnn_logical_and_0,
            ttnn_where_9,
            ttnn_permute_18,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_18, False)
        ttnn.deallocate(ttnn_where_9, False)
        ttnn_to_layout_16 = ttnn.to_layout(
            pos_ids, ttnn.Layout.TILE, None, memory_config=None
        )
        ttnn.deallocate(pos_ids, False)
        ttnn_add_16 = ttnn.add(
            ttnn_to_layout_16,
            var_185,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_16, False)
        ttnn_reshape_41 = ttnn.reshape(
            ttnn_slice_9,
            [1, 19, 8, 256],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_9, False)
        ttnn_rms_norm_5 = ttnn.rms_norm(
            ttnn_reshape_41,
            epsilon=9.9999999747524271e-07,
            weight=self.q_norm_w,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_41, False)
        ttnn_permute_19 = ttnn.permute(
            ttnn_rms_norm_5,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_multiply_24 = ttnn.multiply(
            ttnn_permute_19,
            ttnn_reshape_15,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_19, False)
        ttnn_slice_13 = ttnn.slice(
            ttnn_rms_norm_5,
            [0, 0, 0, 128],
            [1, 19, 8, 256],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_neg_3 = ttnn.neg(
            ttnn_slice_13,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_13, False)
        ttnn_slice_14 = ttnn.slice(
            ttnn_rms_norm_5,
            [0, 0, 0, 0],
            [1, 19, 8, 128],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_5, False)
        ttnn_concat_7 = ttnn.concat(
            [ttnn_neg_3, ttnn_slice_14],
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_14, False)
        ttnn.deallocate(ttnn_neg_3, False)
        ttnn_permute_20 = ttnn.permute(
            ttnn_concat_7,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_concat_7, False)
        ttnn_multiply_25 = ttnn.multiply(
            ttnn_permute_20,
            ttnn_reshape_16,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_20, False)
        ttnn_add_17 = ttnn.add(
            ttnn_multiply_24,
            ttnn_multiply_25,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_25, False)
        ttnn.deallocate(ttnn_multiply_24, False)
        ttnn_concat_8 = ttnn.concat(
            [k_cache, ttnn_permute_11],
            2,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_11, False)
        ttnn.deallocate(k_cache, False)
        ttnn_repeat_interleave_2 = ttnn.repeat_interleave(
            ttnn_concat_8,
            2,
            1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_8, False)
        ttnn_concat_9 = ttnn.concat(
            [v_cache, ttnn_permute_15],
            2,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_15, False)
        ttnn.deallocate(v_cache, False)
        ttnn_repeat_interleave_3 = ttnn.repeat_interleave(
            ttnn_concat_9,
            2,
            1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_9, False)
        ttnn_typecast_15 = ttnn.typecast(
            ttnn_add_17,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_17, False)
        ttnn_typecast_16 = ttnn.typecast(
            ttnn_repeat_interleave_2,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_repeat_interleave_2, False)
        ttnn_matmul_9 = ttnn.matmul(
            ttnn_typecast_15,
            ttnn_typecast_16,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_typecast_16, False)
        ttnn.deallocate(ttnn_typecast_15, False)
        ttnn_add_18 = ttnn.add(
            ttnn_matmul_9,
            ttnn_typecast_11,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_9, False)
        ttnn_eq_1 = ttnn.eq(
            ttnn_add_18,
            var_193,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_logical_not_3 = ttnn.logical_not(
            ttnn_eq_1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_eq_1, False)
        ttnn_sum_1 = ttnn.sum(
            ttnn_logical_not_3,
            [3],
            True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_logical_not_3, False)
        ttnn_logical_not_4 = ttnn.logical_not(
            ttnn_sum_1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_sum_1, False)
        ttnn_softmax_1 = ttnn.softmax(
            ttnn_add_18,
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
            numeric_stable=True,
        )
        ttnn.deallocate(ttnn_add_18, False)
        ttnn_typecast_17 = ttnn.typecast(
            ttnn_logical_not_4,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_logical_not_4, False)
        ttnn_where_11 = ttnn.where(
            ttnn_typecast_17,
            var_187,
            ttnn_softmax_1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_17, False)
        ttnn.deallocate(ttnn_softmax_1, False)
        ttnn_typecast_18 = ttnn.typecast(
            ttnn_repeat_interleave_3,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_repeat_interleave_3, False)
        ttnn_matmul_10 = ttnn.matmul(
            ttnn_where_11,
            ttnn_typecast_18,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_typecast_18, False)
        ttnn.deallocate(ttnn_where_11, False)
        ttnn_typecast_19 = ttnn.typecast(
            ttnn_matmul_10,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_10, False)
        ttnn_transformer_concatenate_heads_1 = ttnn.transformer.concatenate_heads(
            ttnn_typecast_19,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_19, False)
        ttnn_reshape_42 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_1,
            [19, 2048],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_transformer_concatenate_heads_1, False)
        ttnn_matmul_11 = ttnn.matmul(
            ttnn_reshape_42,
            self.o_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_reshape_42, False)
        ttnn_reshape_43 = ttnn.reshape(
            ttnn_matmul_11,
            [1, 1, 19, 5376],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_11, False)
        ttnn_reduce_scatter_2 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_43,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_43, False)
        ttnn_reshape_44 = ttnn.reshape(
            ttnn_reduce_scatter_2,
            [1, 19, 1344],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reduce_scatter_2, False)
        return ttnn_reshape_44, ttnn_add_16, ttnn_where_8, ttnn_where_10

    def _full_decode(self, x, *, k_cache, v_cache, pos_ids, update_idxs, full_cos_cache, full_sin_cache, full_pos_mask, var_185, var_191, var_193):
        """Full attention (Gemma3 style): separate Q + K matmuls (no V -- the
        K cache doubles as V via k_eq_v=True), q/k_norm, RoPE rotation,
        full-attention masked SDPA reading the K cache as V, o_proj, distributed
        reduce-scatter. Decode-specific: writes the new K state into the paged
        K cache via `ttnn.experimental.paged_update_cache`.

        Returns a 2-tuple of the codegen-emitted local names of the sub-block's
        outputs:
          * `ttnn_reshape_226` -- residual flow that feeds `post_attn_ln`
          * `ttnn_add_115`     -- pos-id increment surfaced to `_main`'s return

        Consumes (deallocates internally): `x`.
        Does NOT deallocate: `k_cache`, `v_cache` (the K state is written
        in-place via `paged_update_cache`; the K cache also doubles as V
        via k_eq_v=True, so v_cache is read-only); `pos_ids`, `update_idxs`
        (passed-through references); weight tensors, RoPE caches,
        full_pos_mask, var_*.
        """
        # Aliases that match the verbatim codegen op names below.
        ttnn_typecast_35 = full_cos_cache
        ttnn_typecast_36 = full_sin_cache
        ttnn_typecast_39 = full_pos_mask
        runtime_a = k_cache
        runtime_b = v_cache
        runtime_c = pos_ids
        var_36 = update_idxs
        ttnn_multiply_201 = x

        ttnn_reshape_221 = ttnn.reshape(
            ttnn_multiply_201,
            [1, 1344],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_201, False)
        ttnn_all_gather_67 = ttnn.all_gather(
            input_tensor=ttnn_reshape_221,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_221, False)
        ttnn_matmul_80 = ttnn.matmul(
            ttnn_all_gather_67,
            self.k_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn_reshape_222 = ttnn.reshape(
            ttnn_matmul_80,
            [1, 1, 1, 512],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_80, False)
        ttnn_rms_norm_33 = ttnn.rms_norm(
            ttnn_reshape_222,
            epsilon=9.9999999747524271e-07,
            weight=self.k_norm_w,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn_multiply_202 = ttnn.multiply(
            ttnn_rms_norm_33,
            ttnn_typecast_35,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_74 = ttnn.slice(
            ttnn_rms_norm_33,
            [0, 0, 0, 256],
            [1, 1, 1, 512],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_neg_22 = ttnn.neg(
            ttnn_slice_74,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_74, False)
        ttnn_slice_75 = ttnn.slice(
            ttnn_rms_norm_33,
            [0, 0, 0, 0],
            [1, 1, 1, 256],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_33, False)
        ttnn_concat_24 = ttnn.concat(
            [ttnn_neg_22, ttnn_slice_75],
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_75, False)
        ttnn.deallocate(ttnn_neg_22, False)
        ttnn_multiply_203 = ttnn.multiply(
            ttnn_concat_24,
            ttnn_typecast_36,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_24, False)
        ttnn_add_114 = ttnn.add(
            ttnn_multiply_202,
            ttnn_multiply_203,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_203, False)
        ttnn.deallocate(ttnn_multiply_202, False)
        ttnn_to_memory_config_2 = ttnn.to_memory_config(
            ttnn_add_114,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                    ),
                    [32, 512],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_add_114, False)
        ttnn.experimental.paged_update_cache(
            runtime_a,
            ttnn_to_memory_config_2,
            update_idxs_tensor=update_idxs,
            share_cache=False,
            page_table=None,
        )
        ttnn.deallocate(ttnn_to_memory_config_2, False)
        ttnn_rms_norm_34 = ttnn.rms_norm(
            ttnn_reshape_222,
            epsilon=9.9999999747524271e-07,
            weight=None,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_222, False)
        ttnn_to_memory_config_3 = ttnn.to_memory_config(
            ttnn_rms_norm_34,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                    ),
                    [32, 512],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_34, False)
        ttnn.experimental.paged_update_cache(
            runtime_b,
            ttnn_to_memory_config_3,
            update_idxs_tensor=update_idxs,
            share_cache=False,
            page_table=None,
        )
        ttnn.deallocate(ttnn_to_memory_config_3, False)
        ttnn.deallocate(update_idxs, False)
        ttnn_to_layout_58 = ttnn.to_layout(
            runtime_c, ttnn.Layout.TILE, None, memory_config=None
        )
        ttnn.deallocate(runtime_c, False)
        ttnn_add_115 = ttnn.add(
            ttnn_to_layout_58,
            var_185,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_58, False)
        ttnn_matmul_81 = ttnn.matmul(
            ttnn_all_gather_67,
            self.q_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_67, False)
        ttnn_reshape_223 = ttnn.reshape(
            ttnn_matmul_81,
            [1, 8, 1, 512],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_81, False)
        ttnn_rms_norm_35 = ttnn.rms_norm(
            ttnn_reshape_223,
            epsilon=9.9999999747524271e-07,
            weight=self.q_norm_w,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_223, False)
        ttnn_multiply_204 = ttnn.multiply(
            ttnn_rms_norm_35,
            ttnn_typecast_35,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_76 = ttnn.slice(
            ttnn_rms_norm_35,
            [0, 0, 0, 256],
            [1, 8, 1, 512],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_neg_23 = ttnn.neg(
            ttnn_slice_76,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_76, False)
        ttnn_slice_77 = ttnn.slice(
            ttnn_rms_norm_35,
            [0, 0, 0, 0],
            [1, 8, 1, 256],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_35, False)
        ttnn_concat_25 = ttnn.concat(
            [ttnn_neg_23, ttnn_slice_77],
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_77, False)
        ttnn.deallocate(ttnn_neg_23, False)
        ttnn_multiply_205 = ttnn.multiply(
            ttnn_concat_25,
            ttnn_typecast_36,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_25, False)
        ttnn_add_116 = ttnn.add(
            ttnn_multiply_204,
            ttnn_multiply_205,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_205, False)
        ttnn.deallocate(ttnn_multiply_204, False)
        ttnn_typecast_68 = ttnn.typecast(
            ttnn_add_116,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_116, False)
        ttnn_repeat_interleave_22 = ttnn.repeat_interleave(
            runtime_a,
            8,
            1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_69 = ttnn.typecast(
            ttnn_repeat_interleave_22,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_repeat_interleave_22, False)
        ttnn_matmul_82 = ttnn.matmul(
            ttnn_typecast_68,
            ttnn_typecast_69,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_typecast_69, False)
        ttnn.deallocate(ttnn_typecast_68, False)
        ttnn_add_117 = ttnn.add(
            ttnn_matmul_82,
            ttnn_typecast_39,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_82, False)
        ttnn_eq_11 = ttnn.eq(
            ttnn_add_117,
            var_193,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_logical_not_23 = ttnn.logical_not(
            ttnn_eq_11,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_eq_11, False)
        ttnn_sum_11 = ttnn.sum(
            ttnn_logical_not_23,
            [3],
            True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_logical_not_23, False)
        ttnn_logical_not_24 = ttnn.logical_not(
            ttnn_sum_11,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_sum_11, False)
        ttnn_softmax_11 = ttnn.softmax(
            ttnn_add_117,
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
            numeric_stable=True,
        )
        ttnn.deallocate(ttnn_add_117, False)
        ttnn_typecast_70 = ttnn.typecast(
            ttnn_logical_not_24,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_logical_not_24, False)
        ttnn_where_54 = ttnn.where(
            ttnn_typecast_70,
            var_191,
            ttnn_softmax_11,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_70, False)
        ttnn.deallocate(ttnn_softmax_11, False)
        ttnn_repeat_interleave_23 = ttnn.repeat_interleave(
            runtime_b,
            8,
            1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_71 = ttnn.typecast(
            ttnn_repeat_interleave_23,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_repeat_interleave_23, False)
        ttnn_matmul_83 = ttnn.matmul(
            ttnn_where_54,
            ttnn_typecast_71,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_typecast_71, False)
        ttnn.deallocate(ttnn_where_54, False)
        ttnn_typecast_72 = ttnn.typecast(
            ttnn_matmul_83,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_83, False)
        ttnn_reshape_224 = ttnn.reshape(
            ttnn_typecast_72,
            [1, 4096],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_72, False)
        ttnn_matmul_84 = ttnn.matmul(
            ttnn_reshape_224,
            self.o_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_reshape_224, False)
        ttnn_reshape_225 = ttnn.reshape(
            ttnn_matmul_84,
            [1, 1, 1, 5376],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_84, False)
        ttnn_reduce_scatter_22 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_225,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_225, False)
        ttnn_reshape_226 = ttnn.reshape(
            ttnn_reduce_scatter_22,
            [1, 1, 1344],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reduce_scatter_22, False)

        return (ttnn_reshape_226, ttnn_add_115)

    def _full_prefill(self, x, *, k_cache, v_cache, pos_ids, full_cos_cache, full_sin_cache, full_pos_mask, var_185, var_187, var_193):
        """Full attention (Gemma3 style with k_eq_v=True): separate Q+K
        matmuls (no V — K is reused as V), q/k_norm, RoPE rotation,
        full-attention masked SDPA, o_proj, distributed reduce-scatter.
        Different head_dim (512) and projection shapes from sliding.

        Returns a 2-tuple matching the codegen-emitted local names:
          * `ttnn_reshape_209` — residual flow that feeds `post_attn_ln`
          * `ttnn_add_116`     — `pos_ids + var_185`, the next-step
            position-id increment that flows up through
            `_full_decoder_layer`'s return tuple to `_main`'s outer return
            list. NOTE: this is NOT a KV-cache write — the K and V caches
            are mutated in place by `ttnn.fill_cache(k_cache, …)` and
            `ttnn.fill_cache(v_cache, …)` calls inside this body. The
            postlude reads the updated caches by re-indexing the same
            input slots that were passed in as `k_cache` / `v_cache`.

        Consumes (deallocates internally): `x`, `pos_ids`.
        Does NOT deallocate: weight tensors, RoPE caches, position-id mask,
        var_*, `k_cache`, `v_cache` (latter two are mutated in place but
        stay alive — the postlude re-reads them).
        """
        # Aliases that match the verbatim codegen op names below.
        ttnn_reshape_104 = full_cos_cache
        ttnn_reshape_105 = full_sin_cache
        ttnn_typecast_39 = full_pos_mask

        ttnn_reshape_204 = ttnn.reshape(
            x,
            [19, 1344],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(x, False)
        ttnn_all_gather_67 = ttnn.all_gather(
            input_tensor=ttnn_reshape_204,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
        )
        ttnn.deallocate(ttnn_reshape_204, False)
        ttnn_matmul_80 = ttnn.matmul(
            ttnn_all_gather_67,
            self.k_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn_reshape_205 = ttnn.reshape(
            ttnn_matmul_80,
            [1, 1, 19, 512],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_80, False)
        ttnn_rms_norm_33 = ttnn.rms_norm(
            ttnn_reshape_205,
            epsilon=9.9999999747524271e-07,
            weight=self.k_norm_w,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn_multiply_202 = ttnn.multiply(
            ttnn_rms_norm_33,
            ttnn_reshape_104,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_slice_75 = ttnn.slice(
            ttnn_rms_norm_33,
            [0, 0, 0, 256],
            [1, 1, 19, 512],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_neg_22 = ttnn.neg(
            ttnn_slice_75,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_75, False)
        ttnn_slice_76 = ttnn.slice(
            ttnn_rms_norm_33,
            [0, 0, 0, 0],
            [1, 1, 19, 256],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_33, False)
        ttnn_concat_45 = ttnn.concat(
            [ttnn_neg_22, ttnn_slice_76],
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_76, False)
        ttnn.deallocate(ttnn_neg_22, False)
        ttnn_multiply_203 = ttnn.multiply(
            ttnn_concat_45,
            ttnn_reshape_105,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_concat_45, False)
        ttnn_add_115 = ttnn.add(
            ttnn_multiply_202,
            ttnn_multiply_203,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_203, False)
        ttnn.deallocate(ttnn_multiply_202, False)
        ttnn.fill_cache(k_cache, ttnn_add_115, 0)
        ttnn.deallocate(ttnn_add_115, False)
        ttnn_rms_norm_34 = ttnn.rms_norm(
            ttnn_reshape_205,
            epsilon=9.9999999747524271e-07,
            weight=None,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_205, False)
        ttnn.fill_cache(v_cache, ttnn_rms_norm_34, 0)
        ttnn.deallocate(ttnn_rms_norm_34, False)
        ttnn_to_layout_58 = ttnn.to_layout(
            pos_ids, ttnn.Layout.TILE, None, memory_config=None
        )
        ttnn.deallocate(pos_ids, False)
        ttnn_add_116 = ttnn.add(
            ttnn_to_layout_58,
            var_185,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_to_layout_58, False)
        ttnn_matmul_81 = ttnn.matmul(
            ttnn_all_gather_67,
            self.q_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_all_gather_67, False)
        ttnn_reshape_206 = ttnn.reshape(
            ttnn_matmul_81,
            [1, 19, 8, 512],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_81, False)
        ttnn_rms_norm_35 = ttnn.rms_norm(
            ttnn_reshape_206,
            epsilon=9.9999999747524271e-07,
            weight=self.q_norm_w,
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            program_config=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        ttnn.deallocate(ttnn_reshape_206, False)
        ttnn_permute_104 = ttnn.permute(
            ttnn_rms_norm_35,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn_multiply_204 = ttnn.multiply(
            ttnn_permute_104,
            ttnn_reshape_104,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_104, False)
        ttnn_slice_77 = ttnn.slice(
            ttnn_rms_norm_35,
            [0, 0, 0, 256],
            [1, 19, 8, 512],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_neg_23 = ttnn.neg(
            ttnn_slice_77,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_77, False)
        ttnn_slice_78 = ttnn.slice(
            ttnn_rms_norm_35,
            [0, 0, 0, 0],
            [1, 19, 8, 256],
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_rms_norm_35, False)
        ttnn_concat_46 = ttnn.concat(
            [ttnn_neg_23, ttnn_slice_78],
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_slice_78, False)
        ttnn.deallocate(ttnn_neg_23, False)
        ttnn_permute_105 = ttnn.permute(
            ttnn_concat_46,
            [0, 2, 1, 3],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_concat_46, False)
        ttnn_multiply_205 = ttnn.multiply(
            ttnn_permute_105,
            ttnn_reshape_105,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_permute_105, False)
        ttnn_add_117 = ttnn.add(
            ttnn_multiply_204,
            ttnn_multiply_205,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_205, False)
        ttnn.deallocate(ttnn_multiply_204, False)
        ttnn_typecast_68 = ttnn.typecast(
            ttnn_add_117,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_117, False)
        ttnn_repeat_interleave_22 = ttnn.repeat_interleave(
            k_cache,
            8,
            1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_69 = ttnn.typecast(
            ttnn_repeat_interleave_22,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_repeat_interleave_22, False)
        ttnn_matmul_82 = ttnn.matmul(
            ttnn_typecast_68,
            ttnn_typecast_69,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_typecast_69, False)
        ttnn.deallocate(ttnn_typecast_68, False)
        ttnn_add_118 = ttnn.add(
            ttnn_matmul_82,
            ttnn_typecast_39,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_82, False)
        ttnn_eq_11 = ttnn.eq(
            ttnn_add_118,
            var_193,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_logical_not_23 = ttnn.logical_not(
            ttnn_eq_11,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_eq_11, False)
        ttnn_sum_11 = ttnn.sum(
            ttnn_logical_not_23,
            [3],
            True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_logical_not_23, False)
        ttnn_logical_not_24 = ttnn.logical_not(
            ttnn_sum_11,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_sum_11, False)
        ttnn_softmax_11 = ttnn.softmax(
            ttnn_add_118,
            3,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
            numeric_stable=True,
        )
        ttnn.deallocate(ttnn_add_118, False)
        ttnn_typecast_70 = ttnn.typecast(
            ttnn_logical_not_24,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_logical_not_24, False)
        ttnn_where_54 = ttnn.where(
            ttnn_typecast_70,
            var_187,
            ttnn_softmax_11,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_70, False)
        ttnn.deallocate(ttnn_softmax_11, False)
        ttnn_repeat_interleave_23 = ttnn.repeat_interleave(
            v_cache,
            8,
            1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn_typecast_71 = ttnn.typecast(
            ttnn_repeat_interleave_23,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_repeat_interleave_23, False)
        ttnn_matmul_83 = ttnn.matmul(
            ttnn_where_54,
            ttnn_typecast_71,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_typecast_71, False)
        ttnn.deallocate(ttnn_where_54, False)
        ttnn_typecast_72 = ttnn.typecast(
            ttnn_matmul_83,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_83, False)
        ttnn_transformer_concatenate_heads_11 = ttnn.transformer.concatenate_heads(
            ttnn_typecast_72,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_typecast_72, False)
        ttnn_reshape_207 = ttnn.reshape(
            ttnn_transformer_concatenate_heads_11,
            [19, 4096],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_transformer_concatenate_heads_11, False)
        ttnn_matmul_84 = ttnn.matmul(
            ttnn_reshape_207,
            self.o_proj_w,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn.deallocate(ttnn_reshape_207, False)
        ttnn_reshape_208 = ttnn.reshape(
            ttnn_matmul_84,
            [1, 1, 19, 5376],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_matmul_84, False)
        ttnn_reduce_scatter_22 = ttnn.reduce_scatter(
            input_tensor=ttnn_reshape_208,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(ttnn_reshape_208, False)
        ttnn_reshape_209 = ttnn.reshape(
            ttnn_reduce_scatter_22,
            [1, 19, 1344],
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_reduce_scatter_22, False)
        return ttnn_reshape_209, ttnn_add_116
