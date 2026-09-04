# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Attention primitive ops for Llama-3.1-8B prefill: projections, GQA head split, RoPE, tails.

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaAttention.forward`, decomposed.
**Template:** `models/demos/gpt_oss_d_p/tt/attention/operations.py:14` onwards, with the biases
removed everywhere (`attention_bias: false`) and two whole functions **not** brought over:
`apply_output_projection_fused_rs` (`:142`) and `is_shape_fused_mm_rs_supported` (`:131`). The
fused matmul+reduce-scatter op they wrap is Ring-only and is gated **off** on Blackhole by its own
comment — "RACES on Blackhole (semaphore/overlap sync bug) -> non-deterministic garbage"
(`models/demos/gpt_oss_d_p/tt/attention/operations.py:132-137`) — so on this box it would be dead
code in a functional-first iteration (`bringup_log/03_OUTLINE.md` §2.7).

**Full rotary, so there is no partial-rotary slice/concat** and no QK-norm
(`bringup_log/00_MODEL_CARD.md` §3).
"""

import ttnn

from .weights import AttentionWeights


def apply_qkv_projection(hidden_states, weights: AttentionWeights, compute_kernel_config):
    """`[1,1,S,4096]` -> `(q, k, v)` = `[1,1,S,512]`, `[1,1,S,128]`, `[1,1,S,128]` at TP=8.

    Three separate matmuls rather than one fused QKV (`DEC-016`): the fused form needs the weights
    pre-concatenated per TP shard at load time, which is a second layout to debug before any PCC
    exists. Each `ttnn.linear` gets an explicit `compute_kernel_config` — recipe §2.4's point is
    that the danger on a matmul is an inherited `fp32_dest_acc_en=False`, not an omitted flag.
    """
    q = ttnn.linear(hidden_states, weights.q_proj, dtype=ttnn.bfloat16, compute_kernel_config=compute_kernel_config)
    k = ttnn.linear(hidden_states, weights.k_proj, dtype=ttnn.bfloat16, compute_kernel_config=compute_kernel_config)
    v = ttnn.linear(hidden_states, weights.v_proj, dtype=ttnn.bfloat16, compute_kernel_config=compute_kernel_config)
    return q, k, v


def split_qkv_heads_prefill(q, k, v, num_heads: int, num_kv_heads: int):
    """`(q, k, v)` -> `Q [1, nq, S, 128]`, `K/V [1, nkv, S, 128]`, head-major.

    `ttnn.experimental.nlp_create_qkv_heads` has a **two-tensor** form — Q on its own plus a fused
    `K|V` second tensor (`ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads.cpp:22-32`
    is that branch) — which is what makes three separate projections compatible with it
    (`DEC-019`). The template passes one fully-fused QKV instead
    (`models/demos/gpt_oss_d_p/tt/attention/operations.py:41-46`).

    `transpose_k_heads=False`: the SDPA op wants K in `[1, nkv, S, head_dim]`, not pre-transposed.
    Note the op's Python keyword is **`num_heads`**, not `num_q_heads`
    (`ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/nlp_create_qkv_heads_nanobind.cpp:30`).
    """
    kv = ttnn.concat([k, v], dim=3)
    heads = ttnn.experimental.nlp_create_qkv_heads(
        q,
        kv,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    kv.deallocate(True)
    return heads


def apply_rope(tensor, rope_mats, transformation_mat, *, kv_actual_global=None, cluster_axis=None):
    """Rotate one tensor (Q or K, never V) in place of the HF `rotate_half`, Meta convention.

    Two inner ops, dispatched exactly as
    `models/demos/gpt_oss_d_p/tt/attention/operations.py:78-89`:

    * **contiguous** (`kv_actual_global is None`) — `ttnn.experimental.rotary_embedding_llama` with
      cos/sin already sliced to this chunk's positions (`tt/rope.py::build_prefill_rope`);
    * **indexed** (`kv_actual_global` set) — `rotary_embedding_indexed`, where `rope_mats` are the
      whole-cache, block-cyclic, SP-sharded cos/sin built once by `tt/rope.py::build_indexed_rope`
      and the op derives this chunk's per-chip start row on device. P7 owns that path; it is wired
      here so P7 does not have to reopen this file.

    The llama3 scaling is baked into the tables at build time, so this is a plain full rotation.
    """
    if kv_actual_global is not None:
        return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            tensor,
            rope_mats[0],
            rope_mats[1],
            transformation_mat,
            kv_actual_global=kv_actual_global,
            cluster_axis=cluster_axis,
        )
    return ttnn.experimental.rotary_embedding_llama(
        tensor, rope_mats[0], rope_mats[1], transformation_mat, is_decode_mode=False
    )


def concat_heads(tensor):
    """`[1, nq, S, 128]` -> `[1, 1, S, nq*128]` (512 at TP=8)."""
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights: AttentionWeights, activation_dtype, compute_kernel_config):
    """`[1,1,S,512]` -> `[1,1,S,4096]`, a **partial sum** per TP device (o_proj is row-parallel).

    No bias (`attention_bias: false`), so unlike
    `models/demos/gpt_oss_d_p/tt/attention/operations.py:105-121` there is no `ttnn.add` tail and no
    first-shard-only bias trick.
    """
    return ttnn.linear(tensor, weights.o_proj, dtype=activation_dtype, compute_kernel_config=compute_kernel_config)


def apply_allreduce(tensor, mesh_config, ccl_manager):
    """The TP all-reduce that turns o_proj's partial sums into the residual (scheme A, `DEC-025`).

    No-op at `tp == 1`, which is every P5-P7 gate. `bringup_log/04_CCL_PLAN.md` §5 row 1.
    There is no padding slice after it: every Llama shard is tile-aligned, so
    `models/demos/gpt_oss_d_p/tt/attention/operations.py:257-269`'s un-pad branch is deleted rather
    than left as a no-op that reads as if padding might exist.
    """
    if mesh_config.tp <= 1:
        return tensor
    # `MeshConfig.allreduce` frees its own input between the reduce-scatter and the all-gather, so
    # the caller must not deallocate `tensor` afterwards (`tt/config.py`).
    return mesh_config.allreduce(tensor, ccl_manager, axis=mesh_config.tp_axis)


def apply_reduce_scatter(tensor, mesh_config, ccl_manager):
    """Residual scheme B's seam: reduce-scatter only, leaving `[1,1,S,4096/TP]`. **Refuses.**

    Wired from day one so switching schemes is a flag rather than a rewrite
    (`BRINGUP_RECIPE.md:1207-1209`), and refusing until P8 for the same reason `tt/mlp.py` does
    (`DEC-038`, `DEC-041`): a reduce-scatter here while the norms and the residual add still expect
    full emb is not scheme B, it is a mixed residual. `bringup_log/04_CCL_PLAN.md` §5 row 3.
    """
    raise NotImplementedError(
        "attention apply_reduce_scatter needs residual scheme B, which is not wired in this "
        "iteration: DEC-025 takes scheme A (replicated full-emb residual) and P8 owns the switch. "
        "bringup_log/04_CCL_PLAN.md section 5 row 3 is the seam."
    )
