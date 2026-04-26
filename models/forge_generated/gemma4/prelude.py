"""Sliding-window and full-attention preludes (RoPE cos/sin caches +
mask helpers). One class per (prefill|decode, sliding|full) combination
because the bodies differ structurally enough that a unified body would
be more confusing than helpful. Phase 2 may unify after HF loading
patterns are clear.

Bodies are verbatim copies of the legacy `_sliding_prelude` and
`_full_prelude` from `gemma4_{prefill,decode}/model.py`, with
`_cached__main` renamed to `cached_main` (threaded as a __call__ arg).
"""
import ttnn


class SlidingPreludeDecode:
    """Sliding-attention prelude for decode.

    Verbatim wrap of the codegen-derived sliding-decode prelude. Returns
    (typecast_2, typecast_3, reshape_3, reshape_18, to_layout_11,
    typecast_11). Six outputs.
    """

    def __init__(self, cached_main):
        self._cached_main = cached_main

    def __call__(self, input, ttnn_to_layout_0, ttnn_add_0):
        cached_main = self._cached_main
        """Compute shared sliding-attention prelude tensors once per call:
        RoPE cos/sin caches, position-id reshapes/typecasts, causal mask
        helper. The op sequence is the verbatim block previously inlined
        inside `_sliding_decoder_layer_0`. Layer 0 used to compute these
        and return them; now they live in this dedicated prelude function.
        """
        var_7 = input[26]
        var_184 = cached_main["main_const_eval_0"][0]
        var_189 = cached_main["main_const_eval_266"][0]

        ttnn_typecast_1 = ttnn.typecast(
            ttnn_to_layout_0,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_3 = ttnn.reshape(
            ttnn_typecast_1,
            [1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_1, False)
        ttnn_matmul_1 = ttnn.matmul(
            cached_main["main_const_eval_626"][0],
            ttnn_reshape_3,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn_reshape_4 = ttnn.reshape(
            ttnn_matmul_1,
            [1, 1, 1, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_1, False)
        ttnn_concat_0 = ttnn.concat(
            [ttnn_reshape_4, ttnn_reshape_4],
            3,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_4, False)
        ttnn_cos_0 = ttnn.cos(
            ttnn_concat_0,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_typecast_2 = ttnn.typecast(
            ttnn_cos_0,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_cos_0, False)
        ttnn_sin_0 = ttnn.sin(
            ttnn_concat_0,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_concat_0, False)
        ttnn_typecast_3 = ttnn.typecast(
            ttnn_sin_0,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sin_0, False)
        ttnn_subtract_0 = ttnn.subtract(
            ttnn_add_0,
            var_189,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_add_5 = ttnn.add(
            ttnn_subtract_0,
            cached_main["main_const_eval_242"][0],
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_subtract_0, False)
        ttnn_ge_0 = ttnn.ge(
            ttnn_add_5,
            var_184,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_17 = ttnn.reshape(
            ttnn_ge_0,
            [1, 1, 1, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_ge_0, False)
        ttnn_reshape_18 = ttnn.reshape(
            ttnn_to_layout_0,
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_19 = ttnn.reshape(
            ttnn_add_5,
            [1, 1, 1, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_ge_1 = ttnn.ge(
            ttnn_reshape_18,
            ttnn_reshape_19,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_logical_and_1 = ttnn.logical_and(
            ttnn_reshape_17,
            ttnn_ge_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_ge_1, False)
        ttnn.deallocate(ttnn_reshape_17, False)
        ttnn_subtract_1 = ttnn.subtract(
            ttnn_to_layout_0,
            var_189,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_20 = ttnn.reshape(
            ttnn_subtract_1,
            [1, 1, 1, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_subtract_1, False)
        ttnn_gt_0 = ttnn.gt(
            ttnn_reshape_19,
            ttnn_reshape_20,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_20, False)
        ttnn.deallocate(ttnn_reshape_19, False)
        ttnn_logical_and_2 = ttnn.logical_and(
            ttnn_logical_and_1,
            ttnn_gt_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_gt_0, False)
        ttnn.deallocate(ttnn_logical_and_1, False)
        ttnn_to_layout_9 = ttnn.to_layout(var_7, ttnn.Layout.TILE, None, memory_config=None)
        ttnn_ne_0 = ttnn.ne(
            ttnn_to_layout_9,
            cached_main["main_const_eval_400"][0],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_to_layout_9, False)
        ttnn_reshape_21 = ttnn.reshape(
            ttnn_ne_0,
            [256, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_ne_0, False)
        ttnn_clamp_0 = ttnn.clamp(
            ttnn_add_5,
            0,
            2147483647,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_5, False)
        ttnn_gt_1 = ttnn.gt(
            var_184,
            ttnn_clamp_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_add_6 = ttnn.add(
            ttnn_clamp_0,
            var_189,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_typecast_4 = ttnn.typecast(
            ttnn_gt_1,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_gt_1, False)
        ttnn_typecast_5 = ttnn.typecast(
            ttnn_add_6,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_6, False)
        ttnn_typecast_6 = ttnn.typecast(
            ttnn_clamp_0,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_clamp_0, False)
        ttnn_where_4 = ttnn.where(
            ttnn_typecast_4,
            ttnn_typecast_5,
            ttnn_typecast_6,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_6, False)
        ttnn.deallocate(ttnn_typecast_5, False)
        ttnn.deallocate(ttnn_typecast_4, False)
        ttnn_typecast_7 = ttnn.typecast(
            ttnn_where_4,
            ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_where_4, False)
        ttnn_reshape_22 = ttnn.reshape(
            ttnn_typecast_7,
            [256, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_7, False)
        ttnn_typecast_8 = ttnn.typecast(
            ttnn_reshape_22,
            ttnn.DataType.UINT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_22, False)
        ttnn_to_layout_10 = ttnn.to_layout(ttnn_typecast_8, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
        ttnn.deallocate(ttnn_typecast_8, False)
        ttnn_to_layout_11 = ttnn.to_layout(ttnn_reshape_21, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
        ttnn.deallocate(ttnn_reshape_21, False)
        ttnn_embedding_5 = ttnn.embedding(
            ttnn_to_layout_10,
            ttnn_to_layout_11,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_to_layout_10, False)
        ttnn_reshape_23 = ttnn.reshape(
            ttnn_embedding_5,
            [1, 1, 1, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_embedding_5, False)
        ttnn_logical_and_3 = ttnn.logical_and(
            ttnn_logical_and_2,
            ttnn_reshape_23,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_23, False)
        ttnn.deallocate(ttnn_logical_and_2, False)
        ttnn_where_5 = ttnn.where(
            ttnn_logical_and_3,
            cached_main["main_const_eval_621"][0],
            cached_main["main_const_eval_543"][0],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_logical_and_3, False)
        ttnn_typecast_11 = ttnn.typecast(
            ttnn_where_5,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_where_5, False)

        ttnn.deallocate(ttnn_to_layout_0, False)
        return (
            ttnn_typecast_2,
            ttnn_typecast_3,
            ttnn_reshape_3,
            ttnn_reshape_18,
            ttnn_to_layout_11,
            ttnn_typecast_11,
        )

    @classmethod
    def from_consteval(cls, cached_main):
        return cls(cached_main)


class FullPreludeDecode:
    """Full-attention prelude for decode.

    Verbatim wrap of the codegen-derived full-decode prelude. Returns
    (typecast_35, typecast_36, typecast_39).
    """

    def __init__(self, cached_main):
        self._cached_main = cached_main

    def __call__(self, input, ttnn_reshape_0, ttnn_reshape_3, ttnn_reshape_18, ttnn_to_layout_11):
        cached_main = self._cached_main
        """Compute the full-attention prelude tensors once per call: full RoPE
        cos/sin caches and a position-id-based mask helper. Used by every full
        decoder layer. The op sequence is the verbatim block previously inlined
        inside layer 5's body. Mirrors prefill commit 828f1b4ef.
        """
        var_192 = cached_main["main_const_eval_486"][0]

        ttnn_matmul_37 = ttnn.matmul(
            cached_main["main_const_eval_75"][0],
            ttnn_reshape_3,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn_reshape_112 = ttnn.reshape(
            ttnn_matmul_37,
            [1, 1, 1, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_matmul_37, False)
        ttnn_concat_11 = ttnn.concat(
            [ttnn_reshape_112, ttnn_reshape_112],
            3,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_112, False)
        ttnn_cos_1 = ttnn.cos(
            ttnn_concat_11,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_typecast_35 = ttnn.typecast(
            ttnn_cos_1,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_cos_1, False)
        ttnn_sin_1 = ttnn.sin(
            ttnn_concat_11,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_concat_11, False)
        ttnn_typecast_36 = ttnn.typecast(
            ttnn_sin_1,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sin_1, False)
        ttnn_ge_2 = ttnn.ge(
            ttnn_reshape_18,
            cached_main["main_const_eval_316"][0],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_logical_and_4 = ttnn.logical_and(
            ttnn_reshape_0,
            ttnn_ge_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_ge_2, False)
        ttnn_embedding_22 = ttnn.embedding(
            cached_main["main_const_eval_509"][0],
            ttnn_to_layout_11,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_114 = ttnn.reshape(
            ttnn_embedding_22,
            [1, 1, 1, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_embedding_22, False)
        ttnn_logical_and_5 = ttnn.logical_and(
            ttnn_logical_and_4,
            ttnn_reshape_114,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_114, False)
        ttnn.deallocate(ttnn_logical_and_4, False)
        ttnn_where_27 = ttnn.where(
            ttnn_logical_and_5,
            var_192,
            cached_main["main_const_eval_123"][0],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_logical_and_5, False)
        ttnn_typecast_39 = ttnn.typecast(
            ttnn_where_27,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_where_27, False)

        ttnn.deallocate(ttnn_reshape_3, False)
        ttnn.deallocate(ttnn_reshape_18, False)
        ttnn.deallocate(ttnn_to_layout_11, False)

        return (
            ttnn_typecast_35,
            ttnn_typecast_36,
            ttnn_typecast_39,
        )

    @classmethod
    def from_consteval(cls, cached_main):
        return cls(cached_main)


class SlidingPreludePrefill:
    """Sliding-attention prelude for prefill (sequence of 19 tokens).

    Verbatim wrap of the codegen-derived sliding-prefill prelude. Returns
    (typecast_2, typecast_3, reshape_3, reshape_15, reshape_16,
    reshape_18, to_layout_11, typecast_11). Eight outputs.
    """

    def __init__(self, cached_main):
        self._cached_main = cached_main

    def __call__(self, input, ttnn_to_layout_0):
        cached_main = self._cached_main
        """Compute shared sliding-attention prelude tensors once per call:
        RoPE cos/sin caches, position-id reshapes, layout/typecast helpers.
        The op sequence is the verbatim block previously inlined inside
        _sliding_decoder_layer_0. Layer 0 used to compute these and return
        them; now they live in this dedicated prelude function.
        """
        var_7 = input[26]
        var_184 = cached_main["main_const_eval_0"][0]
        var_189 = cached_main["main_const_eval_242"][0]

        ttnn_add_2 = ttnn.add(
            ttnn_to_layout_0,
            cached_main["main_const_eval_627"][0],
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_typecast_1 = ttnn.typecast(
            ttnn_add_2,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_3 = ttnn.reshape(
            ttnn_typecast_1,
            [1, 1, 19],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_1, False)
        ttnn_matmul_1 = ttnn.matmul(
            cached_main["main_const_eval_487"][0],
            ttnn_reshape_3,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn_permute_0 = ttnn.permute(
            ttnn_matmul_1,
            [0, 2, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_matmul_1, False)
        ttnn_reshape_4 = ttnn.reshape(
            ttnn_permute_0,
            [1, 19, 1, 128],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_0, False)
        ttnn_concat_0 = ttnn.concat(
            [ttnn_reshape_4, ttnn_reshape_4],
            3,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_4, False)
        ttnn_cos_0 = ttnn.cos(
            ttnn_concat_0,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_typecast_2 = ttnn.typecast(
            ttnn_cos_0,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_cos_0, False)
        ttnn_sin_0 = ttnn.sin(
            ttnn_concat_0,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_concat_0, False)
        ttnn_typecast_3 = ttnn.typecast(
            ttnn_sin_0,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sin_0, False)
        ttnn_reshape_15 = ttnn.reshape(
            ttnn_typecast_2,
            [1, 1, 19, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_16 = ttnn.reshape(
            ttnn_typecast_3,
            [1, 1, 19, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_slice_7 = ttnn.slice(
            ttnn_add_2,
            [0],
            [1],
            [1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_subtract_0 = ttnn.subtract(
            ttnn_slice_7,
            var_189,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_slice_7, False)
        ttnn_add_6 = ttnn.add(
            ttnn_subtract_0,
            cached_main["main_const_eval_401"][0],
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_subtract_0, False)
        ttnn_concat_4 = ttnn.concat(
            [ttnn_add_6, ttnn_add_2],
            0,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_6, False)
        ttnn_ge_0 = ttnn.ge(
            ttnn_concat_4,
            var_184,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_17 = ttnn.reshape(
            ttnn_ge_0,
            [1, 1, 1, 275],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_ge_0, False)
        ttnn_reshape_18 = ttnn.reshape(
            ttnn_add_2,
            [1, 1, 19, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_19 = ttnn.reshape(
            ttnn_concat_4,
            [1, 1, 1, 275],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_ge_1 = ttnn.ge(
            ttnn_reshape_18,
            ttnn_reshape_19,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_logical_and_1 = ttnn.logical_and(
            ttnn_reshape_17,
            ttnn_ge_1,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_ge_1, False)
        ttnn.deallocate(ttnn_reshape_17, False)
        ttnn_subtract_1 = ttnn.subtract(
            ttnn_add_2,
            var_189,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_2, False)
        ttnn_reshape_20 = ttnn.reshape(
            ttnn_subtract_1,
            [1, 1, 19, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_subtract_1, False)
        ttnn_gt_0 = ttnn.gt(
            ttnn_reshape_19,
            ttnn_reshape_20,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_20, False)
        ttnn.deallocate(ttnn_reshape_19, False)
        ttnn_logical_and_2 = ttnn.logical_and(
            ttnn_logical_and_1,
            ttnn_gt_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_gt_0, False)
        ttnn.deallocate(ttnn_logical_and_1, False)
        ttnn_to_layout_9 = ttnn.to_layout(var_7, ttnn.Layout.TILE, None, memory_config=None)
        ttnn_ne_0 = ttnn.ne(
            ttnn_to_layout_9,
            cached_main["main_const_eval_544"][0],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_to_layout_9, False)
        ttnn_reshape_21 = ttnn.reshape(
            ttnn_ne_0,
            [256, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_ne_0, False)
        ttnn_clamp_0 = ttnn.clamp(
            ttnn_concat_4,
            0,
            2147483647,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_concat_4, False)
        ttnn_gt_1 = ttnn.gt(
            var_184,
            ttnn_clamp_0,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_add_7 = ttnn.add(
            ttnn_clamp_0,
            var_189,
            dtype=ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_typecast_4 = ttnn.typecast(
            ttnn_gt_1,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_gt_1, False)
        ttnn_typecast_5 = ttnn.typecast(
            ttnn_add_7,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_add_7, False)
        ttnn_typecast_6 = ttnn.typecast(
            ttnn_clamp_0,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_clamp_0, False)
        ttnn_where_4 = ttnn.where(
            ttnn_typecast_4,
            ttnn_typecast_5,
            ttnn_typecast_6,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_6, False)
        ttnn.deallocate(ttnn_typecast_5, False)
        ttnn.deallocate(ttnn_typecast_4, False)
        ttnn_typecast_7 = ttnn.typecast(
            ttnn_where_4,
            ttnn.DataType.INT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_where_4, False)
        ttnn_reshape_22 = ttnn.reshape(
            ttnn_typecast_7,
            [275, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_7, False)
        ttnn_typecast_8 = ttnn.typecast(
            ttnn_reshape_22,
            ttnn.DataType.UINT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_22, False)
        ttnn_to_layout_10 = ttnn.to_layout(ttnn_typecast_8, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
        ttnn.deallocate(ttnn_typecast_8, False)
        ttnn_to_layout_11 = ttnn.to_layout(ttnn_reshape_21, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
        ttnn.deallocate(ttnn_reshape_21, False)
        ttnn_embedding_5 = ttnn.embedding(
            ttnn_to_layout_10,
            ttnn_to_layout_11,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_to_layout_10, False)
        ttnn_reshape_23 = ttnn.reshape(
            ttnn_embedding_5,
            [1, 1, 1, 275],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_embedding_5, False)
        ttnn_logical_and_3 = ttnn.logical_and(
            ttnn_logical_and_2,
            ttnn_reshape_23,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_23, False)
        ttnn.deallocate(ttnn_logical_and_2, False)
        ttnn_where_5 = ttnn.where(
            ttnn_logical_and_3,
            cached_main["main_const_eval_490"][0],
            cached_main["main_const_eval_622"][0],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_logical_and_3, False)
        ttnn_typecast_11 = ttnn.typecast(
            ttnn_where_5,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_where_5, False)

        return (
            ttnn_typecast_2,
            ttnn_typecast_3,
            ttnn_reshape_3,
            ttnn_reshape_15,
            ttnn_reshape_16,
            ttnn_reshape_18,
            ttnn_to_layout_11,
            ttnn_typecast_11,
        )

    @classmethod
    def from_consteval(cls, cached_main):
        return cls(cached_main)


class FullPreludePrefill:
    """Full-attention prelude for prefill.

    Verbatim wrap of the codegen-derived full-prefill prelude. Returns
    (reshape_104, reshape_105, typecast_39).
    """

    def __init__(self, cached_main):
        self._cached_main = cached_main

    def __call__(self, input, ttnn_reshape_0, ttnn_reshape_3, ttnn_reshape_18, ttnn_to_layout_11):
        cached_main = self._cached_main
        """Compute the full-attention prelude tensors once per call: full RoPE
        cos/sin caches and a position-id-based mask helper. Used by every full
        decoder layer. The op sequence is the verbatim block previously inlined
        inside layer 5's body.
        """
        var_192 = cached_main["main_const_eval_335"][0]

        ttnn_matmul_37 = ttnn.matmul(
            cached_main["main_const_eval_129"][0],
            ttnn_reshape_3,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )
        ttnn_permute_51 = ttnn.permute(
            ttnn_matmul_37,
            [0, 2, 1],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
            pad_value=0.0,
        )
        ttnn.deallocate(ttnn_matmul_37, False)
        ttnn_concat_22 = ttnn.concat(
            [ttnn_permute_51, ttnn_permute_51],
            2,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_permute_51, False)
        ttnn_cos_1 = ttnn.cos(
            ttnn_concat_22,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_typecast_35 = ttnn.typecast(
            ttnn_cos_1,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_cos_1, False)
        ttnn_reshape_104 = ttnn.reshape(
            ttnn_typecast_35,
            [1, 1, 19, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_35, False)
        ttnn_sin_1 = ttnn.sin(
            ttnn_concat_22,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_concat_22, False)
        ttnn_typecast_36 = ttnn.typecast(
            ttnn_sin_1,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_sin_1, False)
        ttnn_reshape_105 = ttnn.reshape(
            ttnn_typecast_36,
            [1, 1, 19, 512],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_typecast_36, False)
        ttnn_ge_2 = ttnn.ge(
            ttnn_reshape_18,
            cached_main["main_const_eval_510"][0],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_logical_and_4 = ttnn.logical_and(
            ttnn_reshape_0,
            ttnn_ge_2,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_ge_2, False)
        ttnn_embedding_22 = ttnn.embedding(
            cached_main["main_const_eval_123"][0],
            ttnn_to_layout_11,
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn_reshape_107 = ttnn.reshape(
            ttnn_embedding_22,
            [1, 1, 1, 256],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_embedding_22, False)
        ttnn_logical_and_5 = ttnn.logical_and(
            ttnn_logical_and_4,
            ttnn_reshape_107,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_reshape_107, False)
        ttnn.deallocate(ttnn_logical_and_4, False)
        ttnn_where_27 = ttnn.where(
            ttnn_logical_and_5,
            var_192,
            cached_main["main_const_eval_217"][0],
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_logical_and_5, False)
        ttnn_typecast_39 = ttnn.typecast(
            ttnn_where_27,
            ttnn.DataType.FLOAT32,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        )
        ttnn.deallocate(ttnn_where_27, False)

        return (
            ttnn_reshape_104,
            ttnn_reshape_105,
            ttnn_typecast_39,
        )

    @classmethod
    def from_consteval(cls, cached_main):
        return cls(cached_main)
