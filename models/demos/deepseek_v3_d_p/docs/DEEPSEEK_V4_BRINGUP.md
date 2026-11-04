# DeepSeek-V4: what is done, what is missing, and what it takes to serve

Companion to [`PREFILL_BRINGUP_TUTORIAL.md`](./PREFILL_BRINGUP_TUTORIAL.md). That document
describes the generic path. This one applies it to DeepSeek-V4 specifically: an inventory of what
already exists in the tree, an architecture diff against V3/GLM, and a phased plan to reach the
same level of function as `deepseek_v3_d_p` and `glm_5_2`.

**Definition of "functional", i.e. parity with V3 / GLM-5.2:**

- registered adapter, selectable by `PREFILL_MODEL=deepseek_v4_flash`
- request-mode serving through `common/prefill/runners/prefill_runner.py`
- chunked prefill at 56K context, multi-user slots, multi-rank pipeline parallel
- KV read-back PCC vs a golden trace, via the producer
- TTNN weight cache populated from a real checkpoint
- KV chunk table published for migration

---

## 1. Inventory: what already exists

More than you might expect. The reference and the hardest attention block are done.

### Complete

| Item | Path | Notes |
|---|---|---|
| HF reference model | `reference/deepseek_v4/modeling_deepseek_v4.py` | 1527 lines, architecturally complete for forward/generate. Eager-only (`_supports_sdpa = False`, `head_dim=512` exceeds FlashAttention's 256 cap, sinks are incompatible with SDPA). MTP weights ignored (`_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]`). No TODOs or stubs. |
| HF reference config | `reference/deepseek_v4/configuration_deepseek_v4.py` | Handles the dual rope-type nesting (`main` / `compress`), the `layer_types` and `mlp_layer_types` schedules, and legacy `compress_ratios` / `num_hash_layers` folding. |
| TT dimension constants | `reference/deepseek_v4_flash_config.py`, `reference/deepseek_v4_pro_config.py` | `DeepSeekV4FlashConfig`, `DeepSeekV4ProConfig`. Both already expose `FABRIC_PAYLOAD_SIZE` — but see §3.3, the value is wrong for mHC. |
| **HCA attention block (TTNN prefill)** | `tt/mla/heavily_compressed_attention.py` | `TtHCA`, `TtHCACompressor`, `TtHCAState`. q/kv stems with partial interleaved RoPE, softmax-gated 128:1 compression with position bias, SDPA with per-head learnable sinks, conjugate output RoPE, grouped block-diagonal `o_proj`, chunked state across calls, device-built masks, tail-tile cache write via one-hot matmul. SP/TP mesh aware with the right CCL pattern per projection. Forward allocates no host tensors. |
| HCA PCC tests | `tests/pcc/test_ttnn_hca.py` | Both variants; compressor alone; single-shot at 5 prompt lengths; chunked with 4 scenarios plus two 56K runs. Meshes 2×2, 4×2, 8×4. Floors: flash 0.998 single-shot / 0.99 at 56K; pro 0.997 / 0.98. |
| HCA device-perf tests | `tests/perf/test_ttnn_hca_perf.py` | |
| MoE gate: `sqrtsoftplus` | `ttnn/.../deepseek_prefill/moe_grouped_topk/` | V4's router activation is supported at the op level. |
| MoE gate: hash routing | `ttnn/.../deepseek_prefill/moe_hash_gate/`, `tt/moe/tt_moe_gate_prefill.py` | `GateComputeMode.HASH_HOST` (reuses the reference `DeepseekV4HashRouter`) and `HASH_DEVICE` (the `tid2eid[input_ids]` lookup is fused into the op's reader kernel; the table ships as uint16 rows padded to 16 columns for 32-byte NoC pages, exact for Pro's 384 experts). |
| Gate matmul tuning | `tt/moe/tt_moe_gate_prefill.py` | Program configs keyed on `DeepSeekV4FlashConfig.EMB_SIZE // 4` × `NUM_ROUTED_EXPERTS` already present. |

Two facts worth stating plainly: **the HCA block is production-quality**, and **V4's MoE gate is
already solved at the op level**. The remaining work is not "port V4 to Tenstorrent" — it is the
other attention layer type, the residual structure, and the entire serving/cache layer.

### Absent

| Item | Status |
|---|---|
| CSA (compressed sparse attention) + lightning indexer | Nothing |
| Pure `sliding_attention` layer | Nothing (cheap: `TtHCA` minus the compressor) |
| mHC hyper-connections | Nothing — no `hyper`, `sinkhorn` or `hc_mult` reference anywhere under `tt/` |
| V4 decoder block | Nothing — `TtPrefillBlock` hardcodes `ttMLA` |
| V4 transformer / runtime | Nothing |
| Engine-owned V4 KV cache | Nothing — `TtHCAState` is not it (§3.4) |
| V4 adapter + registry entry + manifest | Nothing |
| Weight cache contract for HCA | Nothing — `TtHCA.from_reference` takes a live torch module |
| Clamped SiLU-GLU expert activation | Nothing — `SwiGluOai` is a different formula (§3.5) |
| `tid2eid` plumbing | Gate accepts it; nothing constructs or passes it |
| V4 golden-trace generator | Nothing — `generate_prompt_trace.py` is dense-MLA only |
| Producer KV read-back branch | Nothing |
| V4 KV chunk table builder | Nothing |
| Checkpoint | Not referenced anywhere; tests use random init |

There is already an explicit marker for the gap, in `tests/pcc/test_ttnn_hca.py`:

> `TODO: move to a V4 prefill adapter once V4 has a block and a runtime, so build_runtime and
> allocate_kv_cache become writable.`

---

## 2. Config: V4 side by side with what we serve today

| | V4-Flash | V4-Pro | DeepSeek-V3 | GLM-5.2 | Kimi-K2.6 |
|---|---|---|---|---|---|
| `hidden_size` | 4096 | 7168 | 7168 | 6144 | 7168 |
| `num_hidden_layers` | 43 | 61 | 61 | 78 | 61 |
| `num_attention_heads` | 64 | 128 | 128 | 64 | 96 |
| `num_key_value_heads` | **1** | **1** | — (MLA) | — | — |
| `head_dim` | **512 unified** | **512** | 128 nope + 64 rope, 128 v | | |
| `kv_lora_rank` | **absent** | **absent** | 512 | 512 | 512 |
| `q_lora_rank` | 1024 | 1536 | 1536 | 2048 | 1536 |
| `qk_rope_head_dim` | 64 (`partial_rotary_factor` 0.125) | 64 | 64 | 64 | 64 |
| output projection | **grouped low-rank**: `o_groups` 8, `o_lora_rank` 1024 | 16 / 1024 | plain `o_proj` | plain | plain |
| `n_routed_experts` | 256 | 384 | 256 | 256 | 384 |
| `num_experts_per_tok` | **6** | 6 | 8 | 8 | 8 |
| `moe_intermediate_size` | 2048 | 3072 | 2048 | 2048 | 2048 |
| `n_group` / `topk_group` | **1 / 1** | 1 / 1 | 8 / 4 | 1 / 1 | 1 / 1 |
| `scoring_func` | **sqrtsoftplus** | sqrtsoftplus | sigmoid | sigmoid | sigmoid |
| `routed_scaling_factor` | 1.5 | 2.5 | 2.5 | 2.5 | 2.827 |
| dense layers | **0** (all MoE) | 0 | 3 | 3 | 1 |
| hash-MoE bootstrap layers | **3** | 3 | — | — | — |
| `swiglu_limit` | **10.0** | 10.0 | — | — | — |
| `sliding_window` | **128** | 128 | — | — | — |
| `index_topk` | 512 | 1024 | — | 2048 | — |
| `index_head_dim` / `index_n_heads` | 128 / 64 | 128 / 64 | — | 128 / 32 | — |
| indexer scores over | **compressed entries** | compressed entries | — | **raw tokens** | — |
| `hc_mult` | **4** | 4 | — | — | — |
| `rope_theta` / `compress_rope_theta` | **10000 / 160000** | 10000 / 160000 | 10000 | 1e6 | 50000 |
| `max_position_embeddings` | 1048576 | 1048576 | 163840 | 202752 | 262144 |
| `vocab_size` | 129280 | 129280 | 129280 | | |

Two per-layer schedules, both in the config:

```python
DEEPSEEK_V4_LAYER_TYPES     = ("sliding_attention", "compressed_sparse_attention",
                               "heavily_compressed_attention")
DEEPSEEK_V4_MLP_LAYER_TYPES = ("hash_moe", "moe")

_COMPRESS_RATIO_TO_LAYER_TYPE = {0: "sliding_attention", 4: "compressed_sparse_attention",
                                 128: "heavily_compressed_attention"}
```

`layer_types` defaults to 2× HCA bootstrap then a CSA/HCA interleave; checkpoints override it via
their own `layer_types` or the legacy per-layer `compress_ratios` ints, which is how pure
`sliding_attention` layers appear. `mlp_layer_types` defaults to 3× `hash_moe` then `moe`.

**The load-bearing consequence:** roughly half the layers of either variant are CSA. There is no
useful subset of V4 you can serve with HCA alone.

---

## 3. Architecture diff: what actually has to be built

### 3.1 Attention is not MLA any more

V3/GLM cache a compressed latent (512) plus a shared rope stem (64) for **every token**, and MLA
absorbs `wkv_b` into Q. V4 throws that out:

- **Shared-KV MQA.** One 512-d KV head, broadcast to all 64/128 query heads by `repeat_kv`.
  `K == V` — the same tensor is passed as both.
- **Partial interleaved RoPE**: only the trailing 64 of 512 dims rotate, and the rotation is applied
  to interleaved pairs (`cos.repeat_interleave(2, dim=-1)`), not half-splits.
- **Conjugate output RoPE**: the attention output's rope slice is rotated back with `-sin`. `TtHCA`
  does this with `ttnn.neg(sin)` reusing the same `cos`.
- **Learnable per-head sinks**, GPT-OSS style: an extra logit column is concatenated before softmax
  and dropped after. In TTNN this goes through SDPA's `attention_sink` argument, with the sink
  pre-divided by `scaling` because the kernel scales both QK and the sink while the reference scales
  only QK.
- **Grouped low-rank output projection** replacing `o_proj`: block-diagonal `o_a_proj` over
  `o_groups` head-groups → `o_lora_rank` each, concatenated, then `o_b_proj`. Groups partition the
  heads, so a TP chip owns whole groups and this needs no collective — one batched matmul with the
  weight sharded on the group axis.
- **Three layer types** rather than one uniform attention.

All of the above is **already implemented** in `TtHCA`. That is the good news: the attention core,
the sinks, the dual-rope handling and the grouped projection are done and PCC-validated to 56K.

### 3.2 CSA is the biggest missing model component

CSA = the same attention core as HCA, but the compressed branch is compressed 4:1 instead of 128:1,
**and** it is filtered by a lightning indexer. Three sub-pieces:

**(a) The CSA compressor with overlapping windows.** HCA pools non-overlapping 128-token windows.
CSA pools with stride 4 and effective width 8 — a two-series `Ca`/`Cb` scheme where each window also
sees the previous window's `Ca` half:

```python
new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]
if n_windows > 1:
    new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :self.head_dim]
if cache_layer is not None:
    prior_kv, prior_gate = cache_layer.update_overlap_state("compressor", chunk_kv, chunk_gate, self.head_dim)
    if prior_kv is not None:
        new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
compressed = self.kv_norm((new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2))
```

The `update_overlap_state` call is chunked-prefill state that has no HCA analogue: window 0 of every
chunk needs the previous chunk's last `Ca` slice. Under SP sharding it also crosses chips, so the
overlap has to be exchanged along the SP axis, not just carried across chunks.

**(b) The lightning indexer over compressed entries.** A second, independent compressor at
`index_head_dim=128` feeding a ReLU scorer:

```python
class DeepseekV4IndexerScorer(nn.Module):
    r"""Lightning-indexer scoring head: `∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)`."""
    def forward(self, q, compressed_kv, hidden_states):
        scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))  # [B,S,H,T]
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling                # [B,S,H]
        return (scores * weights.unsqueeze(-1)).sum(dim=2)                                       # [B,S,T]
```

This is **not** the DSA indexer we already have. Differences that matter:

| | GLM / V3.2 DSA (`tt/mla/indexer.py`) | V4 CSA indexer |
|---|---|---|
| key space | raw token positions | compressed entries (1 per 4 tokens) |
| key cache length | `max_seq_len` | `max_seq_len / 4` |
| scoring | Hadamard + FP8 path, `ring_indexer_score_dsa` | plain ReLU dot product, no Hadamard |
| top-k target | token indices into the KV cache | entry indices into the compressed cache |
| invalid marker | mask | `-1` sentinel |

```python
causal_threshold = (position_ids + 1) // self.compress_rate
future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
index_scores = index_scores.masked_fill(future_mask, float("-inf"))
top_k_indices = index_scores.topk(top_k, dim=-1).indices
invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)
```

`ring_indexer_score_dsa` and `sparse_sdpa` can probably be reused, but only after the key space is
re-pointed at the compressed cache and the `-1` sentinel is handled. Budget this as a new indexer,
not a config change on the existing one.

**(c) Sparse SDPA over compressed entries.** The selected entries become a scattered `block_bias`
into the same `[carry | sliding | compressed | pad]` key layout `TtHCA._attention` already builds.
The mask machinery in `TtHCA._build_masks` is the right starting point.

Note that `sliding_attention` layers are nearly free once CSA is done: `TtHCA` with no compressor,
and a mask with no compressed columns.

### 3.3 mHC changes the residual stream — and the pipeline wire format

This is the structural change that ripples furthest. V4 does not have a residual stream; it has
**four** of them, mixed by a learned doubly-stochastic matrix:

```python
def forward(self, hidden_streams):                       # [B, S, hc_mult, hidden]
    hc = self.hc_mult
    flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
    pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
    pre  = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
    for _ in range(self.hc_sinkhorn_iters - 1):          # 20 iterations, fp32
        comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
    collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2)
    return post, comb, collapsed
```

and the decoder layer applies it around **both** sublayers:

```python
post, comb, collapsed = self.attn_hc(hidden_states)
attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
hidden_states = post.unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), hidden_states)

post, comb, collapsed = self.ffn_hc(hidden_states)
mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
return post.unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), hidden_states)
```

Consequences to plan for:

1. **`TtPrefillBlock`'s `x = ttnn.add(x, sublayer_out)` no longer applies.** The block becomes
   norm → collapse → sublayer → outer-product-and-mix, twice.
2. **The inter-layer activation is `[B, S, 4, hidden]`.** For pipeline parallel, the D2D socket spec
   in `prefill_runner.py` is `[1, 1, chunk_size, hidden_size]`. V4 either sends 4× that, or each
   rank boundary collapses and re-expands the streams (which is not mathematically equivalent —
   `hc_head` only collapses at the very end).
3. **`FABRIC_PAYLOAD_SIZE = EMB_SIZE` in both V4 config classes is wrong** if the streams cross a
   rank boundary. It has to be `hc_mult * EMB_SIZE`, and the migration payload sizing depends on it.
4. **20 Sinkhorn iterations per sublayer per layer, in fp32.** For V4-Pro that is 61 × 2 × 20 = 2440
   normalisation passes over a 4×4 matrix per token. The matrices are tiny but the op count is not;
   expect this to need a fused kernel rather than 40 `ttnn.div` calls per layer.
5. `DeepseekV4UnweightedRMSNorm` (RMSNorm with no learned γ) is used on the mHC input and on `q_b`.
   The existing `TtDistributedRmsNorm` takes a weight; a γ-free path or a ones-weight is needed.

If you want an incremental first milestone, `hc_mult = 1` degenerates mHC to something close to a
plain residual and lets you bring up the rest of the model before this lands. It will not match the
checkpoint, but it unblocks the block/transformer/runtime work.

### 3.4 The KV cache has to be rebuilt from scratch

`TtHCAState` is *not* the cache the engine wants:

```python
class TtHCAState:
    def __init__(self, compressed_kv, sliding_carry, tail, max_seq_len):
        self.compressed_kv = compressed_kv   # [B, 1, compressed_capacity, head_dim]
        self.sliding_carry = sliding_carry   # [B, 1, sliding_window, head_dim]
        self.tail = tail                     # [B, 1, TILE_SIZE, head_dim]
        self.entry_count = 0
        self.kv_actual = 0
```

Five mismatches against §5 of the tutorial:

1. **No user slot dimension.** `alloc_state(..., batch=1)`; `forward` asserts `batch == 1`. The
   engine needs `num_users` independent slots.
2. **Not a `KvCaches` subclass**, and it is created by the layer rather than by
   `adapter.allocate_kv_cache`. Ownership is inverted relative to the engine contract.
3. **Host-side counters.** `entry_count` and `kv_actual` are Python ints advanced in `forward`,
   and `_write_tail_tile` selects a one-hot matrix by `divmod(entry_count, TILE_SIZE)`. That is
   deliberate and clever — it keeps the offset out of the program — but it means state lives on the
   host, which blocks trace and makes multi-user interleaving awkward.
4. **Replicated across SP, not block-cyclic.** The compressor all-gathers `compressed_kv` on the SP
   axis so every chip holds every entry. `update_padded_kv_cache` and the existing chunk-table
   builder both assume the opposite.
5. **Heterogeneous lengths per layer type.** This is the one with no precedent in the codebase. At
   56K context, per user per layer:

   | Layer type | Compressed entries | Row width | Also caches |
   |---|---|---|---|
   | `heavily_compressed_attention` | 56320 / 128 = **440** | 512 | 128-token sliding window |
   | `compressed_sparse_attention` | 56320 / 4 = **14080** | 512 | + 14080 × 128 indexer keys |
   | `sliding_attention` | 0 | — | 128-token sliding window |

   compared with V3 MLA's 56320 rows × 576. V4's cache is one to two orders of magnitude smaller,
   which is a genuine advantage — but the uniform `[num_users * num_layers, 1, seq_local, dim]`
   fold, and the uniform layer stride the migration table depends on, both assume one length for
   every layer.

   The workable shape is **one tensor per layer class, with compacted layer numbering** — exactly
   the pattern `GLM52Adapter` already uses for its index cache:

   ```python
   first_full       = full_indexer_rank(hf_config, params.first_layer_idx)
   num_index_layers = full_indexer_rank(hf_config, params.first_layer_idx + params.num_layers) - first_full
   ```

   Generalise that to a `layer_class_rank(config, layer_idx, cls)` helper and you get:

   ```python
   @dataclass
   class V4KvCaches(KvCaches):
       sliding: ttnn.Tensor          # all layers,  128 tokens  × 512
       hca_compressed: ttnn.Tensor   # HCA layers,  max_seq/128 × 512
       csa_compressed: ttnn.Tensor   # CSA layers,  max_seq/4   × 512
       csa_index: ttnn.Tensor        # CSA layers,  max_seq/4   × 128
   ```

   Four migration configs, four `KvCacheStage`s, and `layer_split_boundaries` probably has to
   constrain rank starts so that a rank's compacted numbering stays contiguous.

**Also needed:** a write op with a slot dimension and a metadata-tensor overload, mirroring
`update_padded_kv_cache`. Without it there is no trace path and no multi-user, and
`fill_cache_for_user_` + the one-hot shift is single-slot by construction.

### 3.5 MoE: mostly there, three concrete gaps

**Gate — done.** `sqrtsoftplus` is in the `moe_grouped_topk` op. Hash routing has both a host path
(reusing the reference `DeepseekV4HashRouter`) and a device path (`moe_hash_gate`, with the
`tid2eid` lookup fused into the reader kernel). Matmul tuning entries for Flash's shape exist.
`n_group = 1` means the grouped gate collapses to plain top-k, which the existing code handles.

**Gap 1 — `tid2eid` plumbing.** `TtMoEGatePrefill.__init__` accepts `hash_table` and validates it,
but nothing constructs it: it has to be extracted from the checkpoint, carried through the adapter
and the block, and only supplied for the first `num_hash_layers` layers. The block also needs to
pick `HASH_*` vs the normal gate mode per layer from `config.mlp_layer_types`.

**Gap 2 — the expert activation is a new kernel variant.** V4:

```python
gate = gate.clamp(max=self.limit)                        # limit = swiglu_limit = 10.0
up   = up.clamp(min=-self.limit, max=self.limit)
return self.act_fn(gate) * up                            # act_fn = silu
```

The nearest existing variant is `RoutedExpertActivation.SwiGluOai`, whose kernel computes

```
(clamp(up, ±L) + 1) * clamp(gate, max=L) * sigmoid(alpha * clamp(gate, max=L))
```

with `alpha = 1.702, L = 7.0` hardcoded (`fused_swiglu.cpp`, `SWIGLU_OAI`). Two differences: the
`+1` on the up branch, and both constants. So V4 needs either a new
`RoutedExpertActivation::SiluGluClamped` or a parameterised `(alpha, limit, up_bias)` form of the
existing one. Either way it is a kernel change plus a `ttnn` enum addition, not a Python change.

**Gap 3 — `NUM_DENSE_LAYERS = 0`.** V4 has no dense FFN layers at all
(`first_k_dense_replace` equivalent is 0). Neither V4 config class defines `NUM_DENSE_LAYERS`, and
`TtPrefillBlock` reads it unconditionally. Add it as `0`. The shared expert width comes from
`intermediate_size`, which V4's `attribute_map` routes to `moe_intermediate_size`.

### 3.6 Everything else in one list

| Item | Note |
|---|---|
| Dual RoPE tables | plain θ=10000 for `main`, YaRN θ=160000 factor 16 for `compress`. `TtHCA` builds its own from the reference's `rotary_emb`; the shared `RotarySetup` has no V4 path. YaRN's `attention_factor` is forced to 1.0 because the V4 reference does not apply mscale. |
| Weight-cache contract | `TtHCA.from_reference(device, reference, config)` takes a live torch module. Needs `check_cache_complete` / `build_ttnn_cache` / `_convert_and_cache_weights`, plus a V4 branch in the transformer's cache walk. Without this every run rebuilds weights from HF. |
| Golden traces | `generate_prompt_trace.py` writes `[seq, kv_lora + rope]` MLA rows and explicitly rejects non-dense models. V4 needs a generator emitting the compressed entries, sliding window and indexer keys per layer class. |
| Producer read-back | `_read_slot_kv_and_check_pcc` branches on `ADAPTER.name`; V4 needs a fourth branch decoding four configs with different lengths. This is the one change in `common/`. |
| Migration chunk table | The MLA builder assumes block-cyclic SP sharding and a uniform layer stride. V4 needs its own `table_builder` passed to `serialize_kv_chunk_table`. |
| MTP | `num_nextn_predict_layers = 1` in config, not instantiated in the reference, weights explicitly ignored. Out of scope for prefill. |
| Checkpoint | Nothing in-tree references V4 weights; the PCC tests use random init. A dequantized checkpoint path and the `tid2eid` table are prerequisites for any end-to-end claim. |

---

## 4. Plan

Ordered so that each phase ends at something you can run and measure. Flash first — 43 layers,
4096 hidden, 64 heads, 256 experts is the smaller target, and its layer schedule exercises all
three attention types.

### Phase 0 — decisions to make before writing code

These are forks in the road, and picking wrong is expensive:

1. **mHC now or later?** Building the block/transformer/runtime against `hc_mult = 1` (plain
   residual) unblocks phases 2–4 and gets you a serving skeleton months earlier. But nothing you
   run then matches the checkpoint, so every PCC number is against a modified reference. The
   alternative is to build mHC first and have no runnable model until it works.
2. **Fork or generalise?** `TtV4PrefillBlock` / `TtV4PrefillTransformer` / `TtV4PrefillRuntime`
   alongside the existing ones, versus generalising `TtPrefillBlock` behind an attention + residual
   factory. Forking is faster and safer now; generalising avoids a second copy of the pipeline,
   layer-ack, trace and migration plumbing. Given that MiniMax-M3 and GPT-OSS already forked into
   their own packages, forking is the precedent.
3. **Where do the streams cross a rank boundary?** Send 4× hidden over D2D (simple, 4× the
   bandwidth), or restructure the pipeline split so rank boundaries land where streams can collapse
   (cheaper, but not equivalent to the reference).
4. **Which cache write op?** Extend `update_padded_kv_cache` with a replicated-SP mode, or write a
   V4-specific op. The former keeps one chunk-table builder; the latter is less risky.

### Phase 1 — the missing model components

| Step | Deliverable | Validated by |
|---|---|---|
| 1.1 | `TtHCA` with `compressor=None` → `sliding_attention` layer | PCC vs reference, extend `test_ttnn_hca.py` |
| 1.2 | `TtCSACompressor`: Ca/Cb overlapping windows, cross-chunk **and** cross-SP-chip overlap state | PCC vs `DeepseekV4CSACompressor`, chunked |
| 1.3 | `TtV4Indexer`: second compressor at 128-d, ReLU scorer, top-k with `-1` sentinel | PCC vs `DeepseekV4Indexer`, top-k index-set match |
| 1.4 | `TtCSA` = attention core + compressor + indexer + scattered `block_bias` | PCC vs `DeepseekV4Attention` on a CSA layer, single-shot then chunked to 56K |
| 1.5 | `TtHyperConnection` + `TtHyperHead`; γ-free RMSNorm; Sinkhorn (likely a fused kernel) | PCC vs `DeepseekV4HyperConnection`, fp32 tolerance |
| 1.6 | Clamped SiLU-GLU `RoutedExpertActivation` variant + kernel | PCC vs `DeepseekV4Experts._apply_gate` |
| 1.7 | `tid2eid` extraction + plumbing; per-layer gate-mode selection from `mlp_layer_types` | PCC vs `DeepseekV4HashRouter`, `HASH_HOST` then `HASH_DEVICE` |

### Phase 2 — assembly

| Step | Deliverable | Validated by |
|---|---|---|
| 2.1 | `V4KvCaches` + allocators, one tensor per layer class, compacted numbering | shape/stride unit test; a `layer_class_rank` helper generalising `full_indexer_rank` |
| 2.2 | Multi-slot cache write op with a metadata-tensor overload | op unit test alongside `test_deepseek_prefill_update_padded_kv_cache.py` |
| 2.3 | `TtV4PrefillBlock`: mHC-wrapped attention + MoE, per-layer type dispatch, `kv_only`, layer ack | block PCC, mirroring `test_prefill_block_chunked.py` |
| 2.4 | `TtV4PrefillTransformer`: embed / blocks / `hc_head` / norm / lm_head, per-rank slicing, dual rope tables | transformer PCC, then chunked |
| 2.5 | Weight-cache contract on every new module + transformer cache walk | `tests/cache/` |
| 2.6 | `TtV4PrefillRuntime`: `compile` / `make_chunk_input` / `prefill_chunk`, layer ack | single-rank chunked run |

### Phase 3 — serving

| Step | Deliverable | Validated by |
|---|---|---|
| 3.1 | `DeepSeekV4FlashAdapter`; registry entry `deepseek_v4_flash`; manifest JSON | adapter unit test, `variant` fixture |
| 3.2 | V4 golden-trace generator | trace dir with all four cache classes |
| 3.3 | Producer read-back branch for the V4 layout | `PREFILL_MOCK_MIGRATION=1` + `PREFILL_PRODUCER_CHECK_PCC=1` |
| 3.4 | Weight cache populated from a real checkpoint | end-to-end request-mode run |
| 3.5 | Multi-rank PP: D2D activation width, `FABRIC_PAYLOAD_SIZE`, `layer_split_boundaries` | 2-rank then 4-rank |
| 3.6 | V4 chunk-table builder + `kv_migration_stages` (4 configs) | `PREFILL_ENABLE_MIGRATION=1` |
| 3.7 | `DeepSeekV4ProAdapter` — should be attributes only if Phase 1–2 stayed config-driven | Pro PCC + serving |

### Phase 4 — performance

Trace capture (needs every V4 op to accept metadata tensors — the same blocker that keeps GLM
untraced today), MLA/SDPA and gate program-config tuning for V4's geometries, shared-expert /
dispatch overlap, and Sinkhorn fusion.

---

## 5. Risk register

| Risk | Why it matters | Mitigation |
|---|---|---|
| **mHC ripples into the engine** | 4× D2D payload, wrong `FABRIC_PAYLOAD_SIZE`, non-additive residual. Touches the one layer we are trying not to touch. | Decide the rank-boundary strategy in Phase 0. Prototype the D2D spec change against a 2-rank run early, before the block is finished. |
| **Heterogeneous per-layer cache lengths** | Breaks the uniform layer stride that the KV chunk table and every existing cache helper assume. | Generalise GLM-5.2's `full_indexer_rank` compaction into a per-layer-class helper *before* writing the caches, so the chunk table and the allocators share one definition. |
| **Sinkhorn op count** | 61 × 2 × 20 fp32 normalisation passes per token for Pro. Death by a thousand small ops. | Profile a single-layer prototype in Phase 1.5. Plan for a fused kernel; do not assume `ttnn.div` in a loop is viable. |
| **CSA indexer is not the DSA indexer** | Easy to under-scope as "reuse `tt/mla/indexer.py` with a different config". Different key space, different cache, different scoring, `-1` sentinels. | Scope it as a new module. Reuse `ring_indexer_score_dsa` / `sparse_sdpa` opportunistically, not by assumption. |
| **PCC decay with depth** | Already visible in HCA: over 56K, flash goes 0.9989 → 0.9925 and pro 0.9984 → 0.9816, because each chunk inherits the previous one's error through the cache and the carry. CSA (4:1) has ~32× more entries and a top-k selection that can flip on small perturbations. | Set per-variant, per-depth floors from measurement, as `test_ttnn_hca.py` already does. Track the top-k index-set overlap separately from output PCC — an index flip is a categorical error that PCC will under-report. |
| **No checkpoint** | Every number so far is random-init. Real weights routinely surface dynamic-range problems that random init hides, especially with bf16 reductions over a 7168-wide hidden. | Get a dequantized checkpoint and the `tid2eid` table staged before Phase 2 ends, not at 3.4. |
| **Two variants diverge** | Flash and Pro differ in more than dimensions: `o_groups` 8 vs 16, `index_topk` 512 vs 1024, `routed_scaling_factor` 1.5 vs 2.5, and different layer schedules. | Keep everything driven by `layer_types` / `mlp_layer_types` / the constants classes. `test_ttnn_hca.py` already parametrises both — extend that discipline rather than special-casing. |

---

## 6. Shortest path to something that runs

If the goal is a demonstrable end-to-end V4 prefill rather than checkpoint-accurate output, the
minimum viable path is:

1. `hc_mult = 1` (plain residual) — defers all of §3.3.
2. Flash only, and only the HCA + sliding layers, by overriding `layer_types` — defers all of §3.2.
3. Single rank — defers the D2D payload question.
4. One user slot — defers the multi-slot write op.
5. Random weights against the same-modified reference — defers the checkpoint.

That reaches a chunked, request-mode, PCC-checked run using code that already exists plus a block,
a transformer, a runtime, a cache and an adapter. It is a real milestone and it de-risks Phases 2–3.
It is also **not** a functional V4: with CSA and mHC stubbed out, the model is a different model.
Every one of those five simplifications has to come back before parity with `deepseek_v3_d_p` or
`glm_5_2` is a defensible claim.
