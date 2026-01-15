# LLM 常用算子


## all_reduce
Reduce on hidden_states:
[num_tokens, hidden_size]



## add_rms_norm_dynamic_quant
Given **hidden_states** and **residual (optional)**, add **residual** (if exists), rms_norm and dynamic_quant on **hidden_size** dim.

---

- (in) hidden_states
    - [num_tokens, hidden_size]
    - bfloat16
- (in) residual
    - [num_tokens, hidden_size]
    - bfloat16
- (in) norm_weight
    - [hidden_size, ]
    - float32
- (in) smooth_scale
    - [hidden_size, ]
    - float32

---
- (out) quant_tokens
    - [num_tokens, hidden_size]
    - int8
- (out) per_token_scale
    - [num_tokens, ]
    - float32
- (out) after_res
    - [num_tokens, hidden_size]
    - bfloat16
- (out) after_norm
    - [num_tokens, hidden_size]
    - bfloat16



## scale_dynamic_quant
Given **hidden_states**, dynamic quant on **hidden_size** dim.

---
- (in) hidden_states
    - [num_tokens, hidden_size]
    - bfloat16
- (in) smooth_scale
    - [hidden_size, ]
    - float32
---
- (out) quant_tokens
    - [num_tokens, hidden_size]
    - int8
- (out) per_token_scale
    - [num_tokens, ]
    - float32




## moe_quant_matmul
Actually quant_matmul, support:

- M = num_tokens
- K = hidden_size
- N = new_hidden_size

act: [M, K], per token quant, dynamic quant with scale [M]

weight: [K, N], per channel quant, with scale [N]

---

- (in) hidden_states
    - [num_tokens, hidden_size]
    - int8
- (in) per_token_scale
    - [num_tokens]
    - float32
- (in) expert_weight
    - [hidden_size, new_hidden_size]
    - int8
- (in) expert_scale
    - [new_hidden_size]
    - float32
---
- (out) y
    - [num_tokens, new_hidden_size]
    - bfloat16


## head_rms_norm
Given **token_data**, for each token, rms_norm on **head_dim** specified by **norm_head_start** and **norm_head_end**, totally **norm_head_num** heads per token will be normed.

- (in) token_data:
    - [num_tokens, total_head_num, head_dim]
    - bfloat16
- (in) weight:
    - [head_num, head_dim]
    - float32
---
- (out) y:
    - [num_tokens, total_head_num, head_dim]
    - bfloat16

---
- (attr) norm_head_start, depends on model structure
- (attr) norm_head_end, depends on model structure
- (attr) norm_head_num, depends on model structure



## rotary_embedding
Give **packed_qkv**, rotary_embedding on **rope_dim** dim, which is part of **head_dim**.

Three test mode:
1. **prefill**: batch_size = 1, q_len > 0, cache_len > 0
2. **prefill_session_cache**: batch_size >= 1, q_lens/cache_lens are variable
3. **decode**: batch_size >= 1, cache_lens are variable, but q_lens are fixed

---

- (in) packed_qkv
    - [num_tokens, q_head_num + 2 * kv_head_num, head_dim]
    - bfloat16

- (in) q_lens
    - [max_batch_size, ]
    - int32

- (in) accum_q_lens
    - optional
    - [max_batch_size + 1, ]
    - int32

- (in) cache_lens
    - [max_batch_size, ]
    - int32

- (in) sin
    - [max_position_embedding, rope_dim // 2]
    - bfloat16

- (in) cos
    - [max_position_embedding, rope_dim // 2]
    - bfloat16

---

- (out) out
    - [num_tokens, q_head_num + 2 * kv_head_num, head_dim]
    - bfloat16
---

- (attr) nope_dim
- (attr) rope_dim
- (attr) head_dim = nope_dim + rope_dim







## store_kv_cache
1. Give **packed_qkv**,

2. Store to linear k_cache and v_cache:
    **[max_batch_size, kv_head_num, max_seq_len, head_dim]**

3. Or store to paged k_cache and v_cache:
    **[max_block_num, kv_head_num, block_size, head_dim]**

   with **block_table** specifying block_ids for each block of each seq.
   **[max_batch_size, max_block_num_per_seq]**

4. Optionally quantize to int8, currently use **static quant**.

Three test mode:
1. **prefill**: batch_size = 1, q_len > 0, cache_len > 0
2. **prefill_session_cache**: batch_size >= 1, q_lens/cache_lens are variable
3. **decode**: batch_size >= 1, cache_lens are variable, but q_lens are fixed

---
- (in) packed_qkv
    - [num_tokens, q_head_num + 2 * kv_head_num, head_dim]
    - bfloat16

- (in) q_lens
    - [max_batch_size, ]
    - int32

- (in) accum_q_lens
    - optional
    - [max_batch_size + 1, ]
    - int32

- (in) cache_lens
    - [max_batch_size, ]
    - int32

- (in) cache_slot_ids
    - [max_batch_size, ]
    - int32

---
### linear cache
- (in) k_cache
    - [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - float16 / bfloat16 / int8 / float8

- (in) v_cache
    - [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - float16 / bfloat16 / int8 / float8
---
### paged cache
- (in) block_table
    - [max_batch_size, max_block_num_per_seq]
    - int32

- (in) k_cache
    - [max_block_num, kv_head_num, block_size, head_dim]
    - float16 / bfloat16 / int8 / float8

- (in) v_cache
    - [max_block_num, kv_head_num, block_size, head_dim]
    - float16 / bfloat16 / int8 / float8

---


- (in) key_scale
    - optional
    - [kv_head_num, head_dim]
    - float32

- (in) value_scale
    - optional
    - [kv_head_num, head_dim]
    - float32

---







## flash_attention (for benchmark)
Naive flash_attention implementation, used for performance benchmarking, assuming that:
1. **prefill** mode: batch_size = 1, q_len > 0, cache_len > 0
2. **prefill_session_cache** mode: batch_size >= 1, q_lens/cache_lens are variable
3. q from packed_qkv
4. kv from kv_cache, use linear cache for simplicity.

---
- (in) q
    - [num_tokens, q_head_num, head_dim]
    - bfloat16
    - **sliced from packed_qkv**

- (in) q_lens
    - [max_batch_size, ]
    - int32

- (in) accum_q_lens
    - optional
    - [max_batch_size + 1, ]
    - int32

- (in) cache_lens
    - [max_batch_size, ]
    - int32

- (in) cache_slot_ids
    - [max_batch_size, ]
    - int32

- (in) k_cache
    - [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - float16 / bfloat16 / int8 / float8

- (in) v_cache
    - [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - float16 / bfloat16 / int8 / float8

- (in) key_scale
    - optional
    - [kv_head_num, head_dim]
    - float32

- (in) value_scale
    - optional
    - [kv_head_num, head_dim]
    - float32

---

- (out) out
    - [q_seq_len, q_head_num, head_dim]
    - bfloat16

---




## flash_decoding (for benchmark)
Naive flash_decoding implementation, used for performance benchmarking, assuming that:
1. **decode** mode: batch_size >= 1, kv_lens are variable, but q_lens are fixed。
2. q from packed_qkv
3. kv from kv_cache, use linear cache for simplicity.

---

- (in) q
    - [num_tokens, q_head_num, head_dim]
    - bfloat16
    - **sliced from packed_qkv**

- (in) cache_lens
    - [max_batch_size, ]
    - int32

- (in) cache_slot_ids
    - [max_batch_size, ]
    - int32

- (in) k_cache
    - [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - float16 / bfloat16 / int8 / float8

- (in) v_cache
    - [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - float16 / bfloat16 / int8 / float8

- (in) key_scale
    - optional
    - [kv_head_num, head_dim]
    - float32

- (in) value_scale
    - optional
    - [kv_head_num, head_dim]
    - float32

---
- (out) out
    - [num_tokens, q_head_num, head_dim]
    - bfloat16




## moe_gating_gemm
Gemm kernel specialized for moe gating, small N, need to split K.

---
- (in) hidden_states
    - [num_tokens, hidden_size]
    - bfloat16
- (in) gating_weight
    - [hidden_size, num_experts]
    - bfloat16

---

- (out) gating_output
    - [num_tokens, num_experts]
    - float32




## moe_softmax_topk
Select topk experts for each token, expert_weights need to be normalized if softmax first.

---

- (in) gating_output
    - required
    - [num_tokens, num_experts]
    - float32

---

- (out) selected_experts
    - [num_tokens, topk]
    - int32

- (out) moe_weights
    - [num_tokens, topk]
    - float32




## moe_scatter_dynamic_quant
Given **world_size** devices

For N shared experts, use DP (data parallel) and SP (seqence parallel), world_size = DP * SP
- DP: We first split on shared experts, at least 1 shared expert per rank.
- SP: Then split on num_tokens if necessary.

For M experts, use EP (experts parallel), world_size = EP
- each rank will pre-allocated num_experts // world_size weights (up and down gemm).
- When given selected experts, scatter input num_tokens to scatter_tokens, dynamic_quant for each token is required either.

---
- (in) hidden_states
    - [num_tokens, hidden_size]
    - bfloat16
- (in) selected_experts
    - [num_tokens, topk]
    - int32
- (in) moe_weights
    - [num_tokens, topk]
    - float32
- (in) experts_smooth_scale
    - [shared_experts_per_rank + experts_per_rank, hidden_size]
---
- (out) scatter_tokens
    - [max_scatter_tokens, hidden_size]
    - int8
- (out) scatter_per_token_scale
    - [max_scatter_tokens, ]
    - float32
- (out) scatter_tokens_offset
    - [max_scatter_tokens, ]
    - int32
- (out) experts_token_count
    - [shared_experts_per_rank + experts_per_rank, ]
    - int32
- (out) experts_token_start
    - [shared_experts_per_rank + experts_per_rank, ]
    - int32











## moe_quant_group_gemm
Given **world_size** devices

For N shared experts, use DP (data parallel) and SP (seqence parallel), world_size = DP * SP
- DP: We first split on shared experts, at least 1 shared expert per rank.
- SP: Then split on num_tokens if necessary.

For M experts, use EP (experts parallel), world_size = EP
- each rank will pre-allocated num_experts // world_size weights (up and down gemm).

---

- (in) scatter_tokens
    - [real_scatter_tokens, hidden_size]
    - int8
- (in) scatter_per_token_scale
    - [real_scatter_tokens, ]
    - float32
- (in) experts_weight
    - [shared_experts_per_rank + experts_per_rank, hidden_size, new_hidden_size]
    - int8
- (in) experts_scale
    - [shared_experts_per_rank + experts_per_rank, new_hidden_size]
    - float32
- (in) experts_token_count
    - [shared_experts_per_rank + experts_per_rank, ]
    - int32
- (in) experts_token_start
    - [shared_experts_per_rank + experts_per_rank, ]
    - int32

---
- (out) y
    - [allocated_tokens, new_hidden_size]
    - bfloat16



## moe_swiglu_dynamic_quant
Given **world_size** devices

For N shared experts, use DP (data parallel) and SP (seqence parallel), world_size = DP * SP
- DP: We first split on shared experts, at least 1 shared expert per rank.
- SP: Then split on num_tokens if necessary.

For M experts, use EP (experts parallel), world_size = EP
- each rank will pre-allocated num_experts // world_size weights (up and down gemm).

---

- (in) scatter_tokens
    - [real_scatter_tokens, hidden_size * 2]
    - bfloat16
- (in) smooth_scale
    - [shared_experts_per_rank + experts_per_rank, hidden_size]
    - float32
- (in) experts_token_count
    - [shared_experts_per_rank + experts_per_rank, ]
    - int32
- (in) experts_token_start
    - [shared_experts_per_rank + experts_per_rank, ]
    - int32

---

- (out) quant_tokens
    - [real_scatter_tokens, hidden_size]
    - int8
- (out) per_token_scale
    - [real_scatter_tokens]
    - float32






## moe_gather
Given **world_size** devices

For N shared experts, use DP (data parallel) and SP (seqence parallel), world_size = DP * SP
- DP: We first split on shared experts, at least 1 shared expert per rank.
- SP: Then split on num_tokens if necessary.

For M experts, use EP (experts parallel), world_size = EP
- each rank will pre-allocated num_experts // world_size weights (up and down gemm).

---

- (in) scatter_tokens
    - [real_scatter_tokens, hidden_size]
    - bfloat16
- (in) scatter_tokens_offset
    - [real_scatter_tokens]
    - int32
---
- (out) convergent_tokens
    - [num_tokens, hidden_size]
    - bfloat16












## rms_norm
Given hidden_states (**[num_tokens, hidden_size], bfloat16**), add residual first optionally and rms_norm.

---

- (in) hidden_states
    - [num_tokens, hidden_size]
    - bfloat16
- (in) residual
    - optional
    - [num_tokens, hidden_size]
    - bfloat16
- (in) weight
    - [hidden_size]
    - float32
---
- (out) after_res
    - [num_tokens, hidden_size]
    - bfloat16
- (out) y
    - [num_tokens, hidden_size]
    - bfloat16
- (attr) eps
    - float32




## quant_matmul
Given hidden_states (**[num_tokens, hidden_size], int8**), weight (**[hidden_size, new_hidden_size], int8**), matmul and dequant.

---

- (in) hidden_states:
    - [num_tokens, hidden_size]
    - int8
- (in) per_token_scale
    - required
    - [num_tokens]
    - float32
- (in) weight:
    - [hidden_size, new_hidden_size]
    - int8
- (in) weight_scale:
    - [new_hidden_size]
    - bfloat16 or float32
- (in) bias:
    - optional
    - [new_hidden_size]
    - bfloat16

---

- (out) y:
    - [num_tokens, new_hidden_size]
    - bfloat16

---

- (attr) transpose_a
    - bool
- (attr) transpose_b
    - bool
