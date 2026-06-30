# 08 · Decode vs Prefill, KV-Cache, and Multi-Device

Encoder models (BGE-M3, ViT, Swin) run one forward over a full sequence. **Generative
LLMs** (Llama, DeepSeek) run two fundamentally different phases — **prefill** and
**decode** — with opposite bottlenecks. This file covers that split, the KV cache, GQA,
and multi-device fracturing, distilled from the TT-Transformers / Llama-70B / DeepSeek-V3
implementations.

---

## 1. The two phases — opposite bottlenecks

| | Prefill | Decode |
|---|---|---|
| Input shape | `(1, 1, seq_len, dim)`, `bsz=1` | `(1, 1, bsz, dim)`, `seq_len=1` |
| Parallelized over | sequence length | **batch** (one token/user) |
| Compute pattern | large matmuls, **compute-bound** | tiny activations, **DRAM-bandwidth-bound** (weights dominate) |
| Activation location | **DRAM** (seq can be 128k — won't fit L1) | **L1 width-sharded** (small, keep close to compute) |
| Matmul variant | **Matmul 2D** (interleaved DRAM in/out) | **DRAM-sharded matmul** (weights sharded across banks) |
| Trace + 2CQ | not used (compute-bound, host hidden) | **essential** (host-bound, tiny ops) |
| `q/k_chunk` in SDPA | 512 default; `exp_approx_mode=False` for long seq | flash-**decode** (whole Q, chunked K/V) |

**The single most important LLM insight:** prefill and decode need *different program
configs, memory configs, and matmul variants for the same weight.* Build both paths.

---

## 2. Matmul variant by regime — the decision {#regime-matmul-variant}
<!-- route
op_class: matmul
regime: prefill,decode
lever_type: single-shot
-->

| Variant | Parallelizes | Activation / output | Weights | Use for |
|---|---|---|---|---|
| **Matmul 2D** (`MatmulMultiCoreReuseMultiCast`) | M and N | DRAM interleaved | DRAM interleaved or sharded | **all prefill matmuls** (M,N ≥ 256, compute-bound) |
| **DRAM-sharded** (`...MultiCastDRAMSharded`) | N | L1 width-sharded | **DRAM width-sharded across banks** | **all decode matmuls** (DRAM-bandwidth-bound) |
| **Matmul 1D** (`...MultiCast1D`) | N only | L1 width-sharded | DRAM interleaved | decode where DRAM-sharded doesn't apply |

- **Prefill = compute-bound** → 2D mcast, maximize `in0_block_w` and subblock, full grid.
- **Decode = DRAM-bound** → DRAM-sharded matmul reads weights at ~240 GB/s vs ~190 GB/s
  interleaved (Wormhole). Activation + output width-sharded in L1.

```python
# Decode DRAM-sharded (weights pre-sharded across 12 banks):
weights_memcfg = create_dram_sharded_mem_config(k=K, n=N)
pc = dram_matmul_config(m=M, k=K, n=N, num_cores=core_grid.num_cores)
out = ttnn.linear(act, weights, program_config=pc,
                  memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                  compute_kernel_config=compute)
```
**Caution:** DRAM-sharded matmul has **no padding** support — the core grid must evenly
divide both activation and output. (This is also why it OOMs for large-batch *prefill*:
the activation width-shard exceeds L1 — see 01 §6.)

---

## 3. The KV cache {#kv-cache}
<!-- route
op_class: attention,datamove
regime: decode
lever_type: single-shot
-->

Decode reads the whole history from a cache instead of recomputing it.

| Op | Phase | Purpose |
|---|---|---|
| `ttnn.fill_cache(K_cache, K, batch_id)` | prefill | populate cache for one batch |
| `ttnn.experimental.paged_fill_cache(K_cache, K, page_table, batch_idx)` | prefill | paged variant |
| `ttnn.experimental.paged_update_cache(keys, K, update_idxs=cur_pos, page_table=...)` | decode | append one token's K/V |

- **Paged KV cache** (page_table) is how continuous batching / vLLM integration works —
  non-contiguous physical pages mapped to logical positions.
- **Current position as a tensor, not a list.** `cur_pos_tensor` lives in device memory and
  can be updated with `ttnn.add` — this makes it **trace-compatible** (traced vars must be
  statically known at compile time; a Python list can't change between traced iterations).

---

## 4. Decode vs prefill attention ops — they are different ops {#decode-attention-ops}
<!-- route
op_class: attention
regime: decode
lever_type: single-shot
-->

| Step | Prefill op | Decode op |
|---|---|---|
| head split | `nlp_create_qkv_heads` | `nlp_create_qkv_heads_decode` (height-sharded over batch) |
| SDPA | `scaled_dot_product_attention` (flash, `is_causal=True`) | `scaled_dot_product_attention_decode` / `paged_..._decode` (flash-decode) |
| concat | `nlp_concat_heads` | `nlp_concat_heads_decode` |

**Flash-decode** processes the entire (small) Q against chunked K/V, parallelized over
batch then kv-head. When `heads·batch < cores`, multiple cores cooperate on one head —
that's the point of flash-decode. `max_cores_per_head_batch=16` caps it (beyond 16 cores
per head the NoC bandwidth between cores bottlenecks).

`exp_approx_mode`: **True** for small `seqlen/chunk_size`, **False** for long sequences
(>16k) where the approximation error accumulates through chunk accumulation and tanks PCC.

---

## 5. Grouped-Query Attention (GQA) {#gqa}
<!-- route
op_class: attention
regime: decode
lever_type: single-shot
-->

Modern LLMs use fewer KV heads than Q heads (`n_kv_heads < n_q_heads`). The head-split ops
take `num_kv_heads` separately:

```python
Q, K, V = ttnn.experimental.nlp_create_qkv_heads(
    xqkv_fused, num_heads=n_q_heads, num_kv_heads=n_kv_heads, transpose_k_heads=False)
```

The fused QKV weight width is `(n_q_heads + 2·n_kv_heads)·head_dim`, not `3·n_q_heads·head_dim`.
SDPA handles the Q-group → KV-head broadcast internally. This shrinks the KV cache and the
K/V projections proportionally — a real memory + bandwidth win at decode.

---

## 6. RoPE — fused rotary position embeddings {#rope-fused}
<!-- route
op_class: attention,eltwise
rank: count,time
regime: prefill,decode
lever_type: single-shot
-->

LLMs apply rotary embeddings to Q and K before attention. Use the fused op, not a manual
rotate:

```python
q = ttnn.experimental.rotary_embedding_llama(q_pre, cos, sin, trans_mat, is_decode_mode=False)
k = ttnn.experimental.rotary_embedding_llama(k_pre, cos, sin, trans_mat, is_decode_mode=False)
```

| | Prefill (`is_decode_mode=False`) | Decode (`is_decode_mode=True`) |
|---|---|---|
| Input | `[1, n_heads, seq, head_dim]` interleaved L1 | `[1, batch, n_heads, head_dim]` height-sharded L1 |
| cos/sin | computed once at init for the seq length | **updated per token** via `RotarySetup` (positions change) |
| trans matrix | `[1,1,TH,TW]` interleaved | `[1,1,TH·batch,TW]` height-sharded |

Decode RoPE needs fresh cos/sin each iteration because each user's position advances —
generate them on-device via `RotarySetup.get_rot_mats(position_ids)`, don't push from host.

---

## 7. Multi-device weight fracturing {#multidevice-fracturing}
<!-- route
op_class: ccl,matmul
lever_type: single-shot
-->

For models too big for one chip, weights are **fractured** (sharded) across devices:

| System | Scheme |
|---|---|
| n300 / T3000 (1D mesh) | 1D column-fractured weights |
| TG / Galaxy (2D mesh) | 2D fractured weights |

- **MLP**: w1/w3 fractured on the output (up-projection) dim, w2 on the input
  (down-projection) dim, so the cross-device reduction lands at the right place.
- After a fractured matmul you need a **CCL** (collective) op to recombine:
  - `all_gather` — concatenate shards across devices (e.g. after column-fractured proj).
  - `reduce_scatter` — sum partial results and re-shard (e.g. after row-fractured proj).
  - `all_reduce` — sum and replicate.
- **Fuse CCL with matmul** where possible: `ttnn.experimental.all_gather_matmul` overlaps
  the all-gather with the matmul — a real latency win on multi-device.

**CCL cost rules:**
- CCLs are expensive — minimize them. Use **bf8b** inputs to CCLs instead of bf16 to halve
  the bytes moved.
- For 2D weight sharding, verify the **reduction dimension** is set correctly — a wrong
  `cluster_axis` silently produces garbage (a documented accuracy-debug check).

---

## 8. Host-device communication minimization (decode is host-sensitive) {#decode-host-comm}
<!-- route
dispatch: gappy
regime: decode
lever_type: single-shot
-->

Decode runs hundreds of tiny ops; host overhead dominates without care:
- **Embeddings on-device** — token IDs are smaller than embeddings to push.
- **Return only the last token from prefill**, not all positions.
- **Sample on-device** (`ttnn.argmax`) — don't round-trip logits to host.
- **Generate masks / rotation matrices on-device** or reuse across iterations.
- **Never tilize/untilize on host.** `to_torch` untilizes on host by default; instead
  `ttnn.untilize(x, use_multicore=True)` on device, then `to_torch`.

These matter only in the host-bound (decode/small-batch) regime — see 06 §8 and 01 §8.

---

## 9. The DRAM weight-layout trick for norms {#decode-norm-weight-layout}
<!-- route
op_class: reduction
bound: dram
regime: decode
lever_type: single-shot
-->

Norm weights (γ, β) pushed in TILE layout need padding to TILE_HEIGHT, wasting DRAM
bandwidth. Instead wrap them into TILE_WIDTH sticks in ROW_MAJOR — no padding, done once
at init:

```python
gamma = gamma.view(1, 1, embedding_dim // TILE_WIDTH, TILE_WIDTH)
ttnn_gamma_rm = ttnn.as_tensor(gamma, layout=ttnn.ROW_MAJOR_LAYOUT,
                               dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

Zero runtime overhead, smaller DRAM footprint, faster weight reads each forward.

---

## 10. Quick reference

| Question | Answer |
|---|---|
| Prefill matmul | Matmul 2D, DRAM interleaved, compute-bound, full grid |
| Decode matmul | DRAM-sharded, L1 width-sharded act/out, weights across banks |
| Decode activation | L1 width-sharded (smallest core grid keeping tile size) |
| Prefill activation | DRAM interleaved (won't fit L1) |
| KV cache update | `paged_update_cache` (decode), `paged_fill_cache` (prefill) |
| Current position | tensor (trace-compatible), not a Python list |
| Head split | `nlp_create_qkv_heads` (prefill) vs `..._decode` (decode) — different ops |
| SDPA | `scaled_dot_product_attention` vs `..._decode` (flash-decode) |
| GQA | pass `num_kv_heads`; QKV width = `(n_q+2·n_kv)·head_dim` |
| RoPE | `rotary_embedding_llama`, `is_decode_mode` flag; decode regenerates cos/sin |
| `exp_approx_mode` | True for short seq, False for >16k |
| Multi-device weights | 1D fracture (n300/T3K), 2D fracture (TG); recombine with CCL |
| CCL dtype | bf8b not bf16 (halve bytes) |
| CCL + matmul | fuse via `all_gather_matmul` |
| Norm weights | TILE_WIDTH sticks, ROW_MAJOR, no padding |
| On-device | embeddings, sampling, untilize, mask/RoPE generation |

---

## 11. Trace capture — eliminate host dispatch in the generation loop {#gen-trace-capture}
<!-- route
op_class: host_fallback
bound: host
regime: decode
lever_type: structural
-->

**Fires when:** the workload is **host-bound** (the `host_overhead` bucket dominates wall
time — `host_fraction` high). The device kernels are fine; wall time is lost to the **host
re-issuing every op each step**. This is the single biggest generation-loop win, and it does
NOT change the device op graph — so it is only measurable under a **wall/throughput metric**
(`--metric wall_ms`), never `device_ms`.

**The coordinated edit (capture once, replay per step):**
```python
# 1. device must be opened with a trace region (set in device_params / open_device):
#    trace_region_size = <bytes>   (size the captured program needs)
# 2. warm up once (compile kernels) OUTSIDE the trace, then capture:
tid = ttnn.begin_trace_capture(device, cq_id=0)
tt_out = run_decode_step(...)          # the SAME ops you already run, captured
ttnn.end_trace_capture(device, tid, cq_id=0)
# 3. per generated token, REPLAY with no host dispatch:
for _ in range(num_tokens):
    write_new_input(...)               # only the changing input is updated in-place
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
```
Current position / step index must be a **device tensor**, not a Python int, or the trace
captures a stale constant (see §8). Inputs that change per step are written **in place** into
the same buffers the trace reads.

**FORCE-TRY:** apply it and let the **wall-time** gate + PCC gate decide. Do NOT skip as risky.
**SELF-VERIFY:** re-read your edit and confirm it (a) opens/sets a `trace_region_size`,
(b) brackets the step with `begin/end_trace_capture`, and (c) replays via `execute_trace` in
the loop. A loop that still calls the eager forward each step captured nothing — inert.

---

## 12. Two command queues (2-CQ) — overlap I/O with compute {#gen-2cq}
<!-- route
op_class: host_fallback
bound: host
regime: decode
lever_type: structural
-->

**Fires when:** host-bound and input/output movement serializes with compute. Open the device
with `num_command_queues=2`; issue host→device writes (and device→host reads) on **cq 1** while
compute runs on **cq 0**, synchronized with events:
```python
write_event = ttnn.record_event(device, cq_id=1)
ttnn.wait_for_event(0, write_event)     # compute (cq 0) waits for the write (cq 1)
# ... compute on cq 0 ...
read_event = ttnn.record_event(device, cq_id=0)
ttnn.wait_for_event(1, read_event)      # read (cq 1) waits for compute (cq 0)
```
Pairs naturally with trace (§11): trace replays compute on cq 0, cq 1 streams the next input.

**FORCE-TRY + SELF-VERIFY:** confirm the device is opened with `num_command_queues=2` AND that
reads/writes actually use `cq_id=1` with events — a second CQ that nothing uses is inert.
Measured under the wall/throughput metric only.

---

## 13. Bucketed (constant-shape) decode — compile once, reuse {#gen-bucketed-decode}
<!-- route
op_class: host_fallback
bound: host
regime: decode
lever_type: structural
-->

**Fires when:** host-bound because each new token grows the sequence by 1 → a **fresh kernel
compile every step** (JIT cache misses dominate wall time). Fix: **pad the decode length to a
fixed bucket** (e.g. 64/128/256) so the decoder kernels compile **once** and every step reuses
them:
```python
BUCKETS = (64, 128, 256, 512)
bucket_len = next(b for b in BUCKETS if b >= max_new_tokens)   # constant shape
# build decoder inputs/masks at bucket_len; generate up to bucket_len, stop at EOS.
```
Masking must hide the padding so PCC/token output is unchanged. This is a prerequisite for
trace (§11): a trace requires a constant shape, so bucket FIRST, then capture.

**FORCE-TRY + SELF-VERIFY:** confirm the decode loop now runs at a **fixed** sequence length
(constant shape) and that padding is masked so generated tokens match the un-bucketed run.
PCC-gate it (output tokens must be identical); measured under the wall/throughput metric.

---

## 14. Tensor-parallel weight fracture — when the model does not fit on one chip {#tp-fracture}
<!-- route
op_class: matmul
bound: dram
lever_type: structural
-->

**Fires when:** the run is TP-regime (`--allow-tp` AND the model's weights do not fit on one chip)
and a dense projection is still **memory-bound** after every single-chip lever. The optimizer reaches
the `tp-fracture` rung. Fracture the weight across the **TP axis** and insert the matching CCL. Use
the **smallest** TP that makes the model fit (the remaining chips become DP replicas).

Column-fracture (split the output dim) → `all_gather`:
```python
w_shard = ttnn.as_tensor(W, dtype=..., layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ShardTensorToMesh(mesh, dim=-1))
y_local = ttnn.linear(x, w_shard, program_config=...)
y = ttnn.all_gather(y_local, dim=-1, cluster_axis=TP_AXIS, mesh_device=mesh, topology=...)
```
Row-fracture (split the contraction dim `K`) → `reduce_scatter` instead, with
`ShardTensorToMesh(mesh, dim=0)` on the K dimension.

**Constraints:** TP must divide `num_heads` (attention) / the fractured dim (MLP), stay tile-aligned
(dim/TP a multiple of 32), and map to a mesh axis. Keep activations **sharded across consecutive
layers** (see §15) so you do not re-gather every layer.

**FORCE-TRY + SELF-VERIFY:** PCC-gate the distributed forward (the fracture + CCL must be
mathematically identical to the single-device result); record `record_kernel_attempt(...,'tp-fracture',...)`.

## 15. CCL optimization — make the cross-chip communication cheap {#ccl-knobs}
<!-- route
op_class: ccl
lever_type: knob
-->

**Fires when:** TP added `all_gather` / `reduce_scatter` ops (the `ccl` bucket). These are NOT tuned
by grid/dtype/kernel — they need communication-specific levers. In priority order:

1. **Wire dtype** — communicate in `bfloat8_b` instead of `bfloat16` → half the bytes on the link
   (PCC-gate it).
2. **Compute↔comm overlap** — async / persistent CCL so the collective **hides behind** the next op's
   compute instead of stalling. This is the highest-value lever; without it the link time is serial.
3. **Topology / `cluster_axis`** — pick `ring`/`linear` per mesh, and communicate along the shortest
   mesh axis.
4. **Reduce CCL volume/count** — column vs row fracture changes what is gathered; keep activations
   sharded across layers (cross-block handoff) to avoid re-gathering; fuse the gather into its consumer.

**FORCE-TRY + SELF-VERIFY:** score the change against the eth-link roofline (`box_facts` eth bw) — a
TP win requires the per-chip compute saved to beat the CCL time added. PCC-gate; record as a knob.
