---
name: graph-fusing
description: Fuse a TTNN op graph into a faster numerically-equivalent one by (1) replacing a primitive op sequence with a dedicated tt-metal op, (2) simplifying/merging structural ops, and (3) folding adjacent ops into an existing op. Use during the operation-topology audit in $optimize, before local program-config tuning, whenever the measured path spells out math that a single op could do, runs redundant data-movement ops, or leaves bias/activation/scale/transpose unfused next to a matmul/conv.
---

# Graph Fusing

This is a correctness-preserving transform. Every rewrite must be proven numerically equivalent to the graph it replaces — see **Verify every rewrite** below. Do not land a rewrite you have not PCC-checked on real shapes and dtypes.

Fusing recurs per stage on the structure that stage introduces: the fused-decoder pass covers the single-device layer, but collectives (fused matmul+reduce-scatter, all-gather+matmul, CCL placement) are only assessable at optimized-multichip, and the full-stack ops (residual→norm across layers, LM-head/sampler) only at optimized-full-model. "No remaining fusing" is always scoped to the current graph, never global.


## Steps to execute graph fusion

### Step 1

Follow instructions from `### Dedicated fused ops` and explore the tt-metal repo for all possible ops that can be used in this graph.

### Step 2

From the perf report + code you are performing fusing on, write out the full op sequence — a short table of each op with its inputs and the data movement (reshard/layout/collective) between them — so repeated ops, redundant movement, and spelled-out-math subgraphs become visible.

### Step 3

Scan table and the implementation for existing patterns in this skill and the new ones you found in step 1. For each candidate subgraph, classify it and choose the most promising candidate to apply, **priority order: Dedicated fused ops first, then Graph rewrites, then Op merging.**

### Step 4

After the graph change, run a PCC equivalence test (unfused vs fused) on device. Keep it only if PCC still meets the threshold (0.99) and the fused path is faster on the measured workload. See **Verify every rewrite** below. Do not give up on a graph rewrite if the first run fails. Try to solve the problem. Read the implementation and validation of the op that you added and figure out if there is something that can be changed in order for the op to be valid for your case. Sometimes even adding an additional input layout change can help.

### Step 5

Once the selected pattern is applied or rejected go back to Step 2 and run the procedure again.
Try every pattern that makes sense, do not stop after you successfully applied one pattern. Only stop when there is nothing else left to try.


## The three kinds of rewrite, in priority order

### Dedicated fused ops — replace a primitive op sequence with a dedicated tt-metal op  (HIGHEST PRIORITY)

The graph spells out the math of an operation (several primitive ops) that a hand-written tt-metal op already implements as one kernel. Collapse the sequence into that op.

The governing principle: **fewer, larger, more specialized ops beat many small primitive ops.** A dedicated kernel does in one dispatch, with fused compute and tuned data movement, what a spelled-out primitive sequence does in many. Most graph wins come from recognizing that a subgraph *is* an op you already have.

**Always explore the tt-metal repo existing ops as potential fusing candidates.** This is the highest-value move and almost always improves performance. Two things to find: whether a dedicated op exists at all, and how a real model already wires it (leading models often assemble the fast op sequence for you, and some ship custom fused kernels you can reuse or imitate).

Find the op and its contract:
- `ttnn/cpp/ttnn/operations/**` — the op library; grep for the operation name and for the math it does.
- `ttnn/ttnn/operations/*.py` — Python bindings and **attached golden functions** (`attach_golden_function`); the golden is the exact math the op computes, so it tells you precisely which primitive sequence the op replaces.
- `tests/ttnn/**` — unit tests give the exact signature, valid shapes, dtypes, and layout constraints.

Find idiomatic usage and existing hand-fused implementations in real models:
- `models/common/modules/` — reusable, pre-tuned building blocks: `attention/`, `mlp/`, `lm_head/`, `rmsnorm/`, `moe/`, `rope/`, `sampling/`. Prefer these before hand-rolling a sequence.
- `models/demos/gpt_oss/tt/` — MoE experts on `ttnn.sparse_matmul`, `topk.py`, attention, and CCL wiring.
- `models/demos/deepseek_v3/tt/` and `models/demos/deepseek_v3_b1/micro_ops/` — MLA attention, MoE gate, LM head, plus a library of custom fused micro-ops (`create_q_heads`, `dram_streaming_matmul`, `deepseek_moe_gate`, fused CCL variants).
- `models/tt_dit/` — DiT / video models (e.g. Wan): model wiring in `models/tt_dit/models/transformers/wan2_2/`, with shared DiT ops in `models/tt_dit/layers/` (`conv2d`, `conv3d`, `linear`, `normalization`, `feedforward`) and `models/tt_dit/blocks/` (`attention`, `transformer_block`).
- Other `models/demos/**` and `models/experimental/**` — when the target resembles a specific model family, read its `tt/` implementation for the idiomatic fused op calls and shape conventions.

Patterns of this kind (light list — see examples for the unfused→fused form):

- Elementwise activation recognition: `relu`, `relu6`, `hardsigmoid`, `silu`, `mish`, `gelu` → the dedicated activation op.
- Softmax: `exp / sum(exp)` → `ttnn.softmax`.
- RMSNorm: `mean(x²)·rsqrt·mul·weight` → `ttnn.rms_norm`.
- Distributed RMSNorm: `all_gather → rms_norm → slice` → `ttnn.rms_norm_pre_all_gather` + small-stats `all_gather` + `ttnn.rms_norm_post_all_gather`.
- SDPA: `matmul → scale → softmax → matmul` → `ttnn.transformer.scaled_dot_product_attention`.
- Split-QKV + split-heads: `slice ×3 → reshape → permute` → `ttnn.transformer.split_query_key_value_and_split_heads`.
- Create QKV heads (decode): the decode slice/reshape/permute → `ttnn.experimental.nlp_create_qkv_heads_decode`.
- Concat heads (decode): decode permute+reshape → `ttnn.experimental.nlp_concat_heads_decode`.
- Concatenate heads (prefill): `permute([0,2,1,3]) → reshape` → `ttnn.transformer.concatenate_heads`.
- RoPE: `x·cos + rotate_half(x)·sin` → `ttnn.experimental.rotary_embedding` (rotate-half) or `rotary_embedding_llama` (interleaved); decode via `token_index`.
- TopK: `sort → slice[:k]` → `ttnn.topk`.
- Fused residual-add + RMSNorm: `add(residual) → rms_norm` → `ttnn.rms_norm(..., residual_input_tensor=…)`.
- Fused matmul + collective (multi-device): local-matmul→all-reduce, matmul→reduce-scatter, or all-gather→matmul → the fused CCL-matmul variant when the op contract allows it.
- MoE experts: routed `ttnn.sparse_matmul` (+ score-weight + reduce), and `ttnn.experimental.moe_compute` where it fits the routing.
- Paged KV-cache update: fold the separate write into the fused cache-update op — but validate it under the watcher NoC sanitizer (a fused update can pass the functional check yet trip a NoC fault).

Fusing caveats: do not fold a bias into a low-precision (e.g. BFP8) linear — keep it a separate higher-precision add (folding can damage Q/K/V PCC); fold an activation into the elementwise op that consumes it (`multiply(..., activations=[SILU])`), not into the upstream matmul.

### Graph rewrites — structural / algebraic rewrite and peer merge  (SECOND PRIORITY)

No new specialized kernel: remove ops or data movement by algebraic identity, or by merging peer ops that share an input. Second priority because it cheaply cuts dispatch and movement without needing a dedicated op.

- RepVGG conv-sum: `conv3x3(x) + conv1x1(x)` → one `conv3x3` with `w3 + pad(w1)` (combine weights host-side).
- Shared-LHS matmul: ≥3 matmuls sharing the same LHS → one matmul over `concat(RHS)` then slice (QKV / gate-up packing).
- Spatial mean: `mean(x, [1,2])` → `reshape → mean(dim=2)` (cheaper reduction geometry).
- Permute-reshape-permute: a `permute → reshape → permute` that composes to identity ordering → a single `reshape`.

### Op merging — fold an adjacent op into an existing op  (THIRD PRIORITY)

An anchor op stays in the graph; a neighbor (bias, activation, scale, transpose, pad, reshape, max-subtract) is folded into it via an op argument or config. Third priority because the win is usually small or free — it is an attribute on an op you were already running — but it removes a dispatch and is worth taking.

- Conv + bias → `ttnn.conv2d(..., bias_tensor=b)`.
- Conv + multiply/scale → fold per-channel scale into conv weights (host-side).
- Conv2d + activation → `Conv2dConfig(activation=...)`.
- Matmul/Linear + activation → `ttnn.matmul(..., activation="silu")`.
- Input-arg activation → eltwise binary → `ttnn.add(x, y, input_tensor_a_activations=[ttnn.UnaryOpType.RELU])`.
- Matmul + bias → `ttnn.linear(a, b, bias=bias)`.
- Permute/transpose + matmul → `ttnn.matmul(a, b, transpose_b=True)`.
- Slice after matmul → push the slice into the matmul operand (narrower matmul).
- BatchNorm (inference) + conv → fold the affine into conv weights/bias (host-side).
- Pad + pooling/conv → fold the pad into the consumer op's padding attribute.
- Numeric-stable softmax: `softmax(x − max(x))` → `ttnn.softmax(x, numeric_stable=True)`.
- Reduction + reshape: `reduce(keepdim=False) → reshape` → `reduce(keepdim=True)`.
- Scaled-sum → mean: `sum(x)·(1/N)` → `ttnn.mean(x)`.
- RoPE decode / decode-reshape: fold the decode layout `permute`/`reshape` into a decode-mode rotary embedding (`token_index=0`).


## Verify every rewrite

A rewrite is not done until it is proven equivalent on device. Once you change the target implementation to a fused op, you must run a PCC equivalence test on the changed path and make sure that accuracy is not degraded and the fused path is faster.

## Worked examples (unfused → fused)

### Dedicated fused ops

```python
# activation (SiLU)
# unfused
s = ttnn.sigmoid(x)
out = ttnn.multiply(x, s)
# fused
out = ttnn.silu(x)

# softmax
# unfused
e = ttnn.exp(x)
denom = ttnn.sum(e, dim=-1, keepdim=True)
out = ttnn.div(e, denom)
# fused
out = ttnn.softmax(x, dim=-1)

# rmsnorm
# unfused
sq = ttnn.multiply(x, x)
var = ttnn.mean(sq, dim=-1, keepdim=True)
inv = ttnn.rsqrt(ttnn.add(var, eps))
normed = ttnn.multiply(x, inv)
out = ttnn.multiply(normed, weight)
# fused
out = ttnn.rms_norm(x, epsilon=eps, weight=weight)

# distributed rmsnorm (mesh; x_shard sharded on dim 3 across the mesh)
# unfused
g = ttnn.all_gather(x_shard, dim=3)
sq = ttnn.multiply(g, g)
var = ttnn.mean(sq, dim=-1, keepdim=True)
inv = ttnn.rsqrt(ttnn.add(var, eps))
normed = ttnn.multiply(ttnn.multiply(g, inv), weight)
out = ttnn.mesh_partition(normed, dim=3, cluster_axis=1)   # re-shard back per device
# fused (only the small stats cross the mesh)
stats = ttnn.rms_norm_pre_all_gather(x_shard)
stats = ttnn.all_gather(stats)
out = ttnn.rms_norm_post_all_gather(x_shard, stats, weight=weight_shard)

# sdpa
# unfused
k_t = ttnn.permute(k, (0, 1, 3, 2))
scores = ttnn.matmul(q, k_t)
scores = ttnn.multiply(scores, scale)
probs = ttnn.softmax(scores, dim=-1)
out = ttnn.matmul(probs, v)
# fused
out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)

# split-qkv + split-heads (qkv: [b, s, 3*h*d])
# unfused
hidden = h * d
q = ttnn.slice(qkv, (0, 0, 0), (b, s, hidden))
k = ttnn.slice(qkv, (0, 0, hidden), (b, s, 2 * hidden))
v = ttnn.slice(qkv, (0, 0, 2 * hidden), (b, s, 3 * hidden))
q = ttnn.permute(ttnn.reshape(q, [b, s, h, d]), (0, 2, 1, 3))
k = ttnn.permute(ttnn.reshape(k, [b, s, h, d]), (0, 2, 1, 3))
v = ttnn.permute(ttnn.reshape(v, [b, s, h, d]), (0, 2, 1, 3))
# fused
q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=h)

# nlp create-qkv-heads (decode; qkv: [b, 1, hidden], seq_len == 1, GQA: h q-heads, kv k/v-heads)
# unfused
q_dim = h * d
kv_dim = kv * d
q = ttnn.slice(qkv, (0, 0, 0), (b, 1, q_dim))
k = ttnn.slice(qkv, (0, 0, q_dim), (b, 1, q_dim + kv_dim))
v = ttnn.slice(qkv, (0, 0, q_dim + kv_dim), (b, 1, q_dim + 2 * kv_dim))
q = ttnn.permute(ttnn.reshape(q, [b, h, 1, d]), (2, 0, 1, 3))    # -> [1, b, h, d] decode layout
k = ttnn.permute(ttnn.reshape(k, [b, kv, 1, d]), (2, 0, 1, 3))   # -> [1, b, kv, d]
v = ttnn.permute(ttnn.reshape(v, [b, kv, 1, d]), (2, 0, 1, 3))
# fused
qkv = ttnn.reshape(qkv, [1, 1, b, hidden])                       # decode op expects [1, 1, batch, hidden]
q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(qkv, num_heads=h, num_kv_heads=kv)

# nlp concat-heads (decode; x: [s=1, b, h, d])
# unfused
x = ttnn.permute(x, (1, 2, 0, 3))
out = ttnn.reshape(x, [b, h * d])
# fused (x height-sharded in L1)
out = ttnn.experimental.nlp_concat_heads_decode(x_sharded, num_heads=h)

# concatenate heads (prefill; x: [b, h, s, d])
# unfused
x = ttnn.permute(x, (0, 2, 1, 3))
out = ttnn.reshape(x, [b, s, h * d])
# fused
out = ttnn.transformer.concatenate_heads(x)

# rope (prefill, rotate-half; x: [b, h, s, d])
# unfused
half = d // 2
x1 = ttnn.slice(x, (0, 0, 0, 0), (b, h, s, half))
x2 = ttnn.slice(x, (0, 0, 0, half), (b, h, s, d))
rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
out = ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rot, sin))
# fused
out = ttnn.experimental.rotary_embedding(x, cos, sin)

# topk
# unfused
vals, idx = ttnn.sort(x, dim=-1, descending=True)
out = ttnn.slice(vals, (0, 0, 0, 0), (n, c, h, k))         # keep first k on the sort dim
# fused
out, idx = ttnn.topk(x, k, dim=-1, largest=True, sorted=True)
```

### Graph rewrites

```python
# repvgg conv-sum
# unfused
y3 = ttnn.conv2d(x, w3, ...)          # 3x3, padding 1
y1 = ttnn.conv2d(x, w1, ...)          # 1x1, padding 0
out = ttnn.add(y3, y1)
# fused (weights combined host-side: w = w3 + w1 padded into the 3x3 center)
out = ttnn.conv2d(x, w3 + pad_to_3x3(w1), ...)   # 3x3, padding 1

# shared-lhs matmul
# unfused
o1 = ttnn.matmul(a, b1)
o2 = ttnn.matmul(a, b2)
o3 = ttnn.matmul(a, b3)
# fused
b = ttnn.concat([b1, b2, b3], dim=-1)
o = ttnn.matmul(a, b)
o1, o2, o3 = <slice o along the last dim by each output width>

# spatial mean (x: [N, H, W, C])
# unfused
out = ttnn.mean(x, dim=[1, 2], keepdim=True)
# fused
x = ttnn.reshape(x, [N, 1, H * W, C])
out = ttnn.mean(x, dim=2, keepdim=True)

# permute-reshape-permute (x: [N, H, W, C])
# unfused
x = ttnn.permute(x, (0, 3, 1, 2))
x = ttnn.reshape(x, [N, 1, C, H * W])
out = ttnn.permute(x, (0, 1, 3, 2))
# fused
out = ttnn.reshape(x, [N, 1, H * W, C])
```

### Op merging

```python
# conv + bias
# unfused
y = ttnn.conv2d(x, w)
out = ttnn.add(y, bias)
# fused
out = ttnn.conv2d(x, w, bias_tensor=bias)

# conv + multiply/scale
# unfused
y = ttnn.conv2d(x, w)
out = ttnn.multiply(y, scale)                  # per-output-channel scale
# fused (scale folded into weights host-side)
out = ttnn.conv2d(x, w * scale_per_out_channel)

# conv2d + activation
# unfused
y = ttnn.conv2d(x, w)
out = ttnn.relu(y)
# fused
out = ttnn.conv2d(x, w, conv_config=Conv2dConfig(activation="relu"))

# matmul/linear + activation
# unfused
y = ttnn.matmul(a, b)
out = ttnn.silu(y)
# fused
out = ttnn.matmul(a, b, activation="silu")

# input-arg activation -> binary
# unfused
a_act = ttnn.relu(x)
out = ttnn.add(a_act, y)
# fused
out = ttnn.add(x, y, input_tensor_a_activations=[ttnn.UnaryOpType.RELU])

# matmul + bias -> linear
# unfused
y = ttnn.matmul(a, b)
out = ttnn.add(y, bias)
# fused
out = ttnn.linear(a, b, bias=bias)

# permute/transpose + matmul
# unfused
b_t = ttnn.permute(b, (0, 1, 3, 2))
out = ttnn.matmul(a, b_t)
# fused
out = ttnn.matmul(a, b, transpose_b=True)

# slice after matmul (keep first n_keep columns)
# unfused
y = ttnn.matmul(a, b)
out = ttnn.slice(y, (0, 0, 0, 0), (bsz, 1, m, n_keep))
# fused (slice pushed into the operand -> narrower matmul)
b = ttnn.slice(b, (0, 0, 0, 0), (bsz, 1, k, n_keep))
out = ttnn.matmul(a, b)

# batchnorm (inference) + conv
# unfused
y = ttnn.conv2d(x, w)
inv = ttnn.rsqrt(ttnn.add(var, eps))
y = ttnn.multiply(ttnn.subtract(y, mean), inv)
out = ttnn.add(ttnn.multiply(y, gamma), beta)
# fused (fold the affine into conv weights/bias host-side)
alpha = gamma / sqrt(var + eps)                # host (torch)
out = ttnn.conv2d(x, w * alpha, bias_tensor=(b0 - mean) * alpha + beta)

# pad + pooling
# unfused
x = ttnn.pad(x, pad, value=-inf)               # max_pool pads with -inf; match it
out = ttnn.max_pool2d(x, padding=[0, 0, 0, 0])
# fused
out = ttnn.max_pool2d(x, padding=pad)

# numeric-stable softmax
# unfused
m = ttnn.max(x, dim=-1, keepdim=True)
shifted = ttnn.subtract(x, m)
out = ttnn.softmax(shifted, dim=-1)
# fused
out = ttnn.softmax(x, dim=-1, numeric_stable=True)

# reduction + reshape
# unfused
r = ttnn.sum(x, dim, keepdim=False)
out = ttnn.reshape(r, <input shape with dim set to 1>)
# fused
out = ttnn.sum(x, dim, keepdim=True)

# scaled-sum -> mean  (N = size of the reduced dim)
# unfused
s = ttnn.sum(x, dim, keepdim=True)
out = ttnn.multiply(s, 1.0 / N)
# fused
out = ttnn.mean(x, dim, keepdim=True)

# rope decode / decode-reshape (x: [b, h, 1, d]; decode)
# unfused
rope_out = <rotate-half rope on x>             # same slice/neg/concat + x*cos + rot*sin as prefill
out = ttnn.permute(rope_out, (2, 0, 1, 3))     # -> [1, b, h, d] decode layout
# fused
x = ttnn.permute(x, (2, 0, 1, 3))
out = ttnn.experimental.rotary_embedding(x, cos, sin, token_index=0)
```
