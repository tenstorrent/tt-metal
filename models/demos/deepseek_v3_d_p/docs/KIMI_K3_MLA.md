# Kimi-K3 MLA: architecture delta vs Kimi-K2.6 and TT op-support audit

Scope: the **full-attention (MLA) layers only**. K3's 69 KDA (linear-attention) layers, the
AttnRes residual scheme, and LatentMoE are out of scope here and are only mentioned where they
change what the MLA layer sees.

Sources: `config.json`, `configuration_kimi_k3.py` and `modeling_kimi_linear.py` from
`huggingface.co/moonshotai/Kimi-K3`, against `huggingface.co/moonshotai/Kimi-K2.6`
(`DeepseekV3ForCausalLM`). TT-side line references are against `models/demos/deepseek_v3_d_p/`
at the time of writing (branched off `5700827bc39`).

## 1. What K2.6 support looks like today

`tt/runners/adapters/kimi_k2_6.py` contains **no MLA-specific code**. It is a 71-line
`MLAPrefillAdapter` subclass carrying identity, paths and PCC thresholds, plus two device knobs
(`l1_small_size = 512`, `routing_use_l1_small_for_semaphores = True`) needed because Kimi has a
single expert group with a device gate. Dims live in `reference/kimi_k2_6_config.py`.
`kimi_k2_7.py` is a 29-line subclass of that (checkpoint only).

The MLA it runs is the shared absorbed-Q path in `tt/mla/mla.py`. The only Kimi-flavoured content
in `tt/mla/` is *tuned shapes*: `mla_config.py` keys program configs on per-device seq_len `640`
with `"num_heads": 64` declared (lines 25, 79, 133, 182, 236, 287).

Residual-stream layout, which drives everything below: hidden state is **TP-fractured on the
feature dim** (7168/tp per device). Both stems consume that shard and all-reduce their small
outputs — `_q_a_latent` does RS+AG on the 1536-wide latent (`mla.py:796-833`), `_kv_stem` does
AG(dim=1)+`fast_reduce_nc` on the 576-wide one. `_o_proj_epilogue` (`mla.py:1005`)
reduce-scatters back to a feature shard.

## 2. MLA architecture delta

| | K2.6 | K3 |
|---|---|---|
| layers / MLA layers | 61 / 61 | 93 / **24** (`full_attn_layers = [4,8,…,88,92,93]`) |
| hidden | 7168 | 7168 |
| attention heads | 64 | **96** |
| q_lora / kv_lora | 1536 / 512 | 1536 / 512 (same) |
| nope / rope / v head dim | 128 / 64 / 128 | 128 / 64 / 128 (same) |
| RoPE | theta 50000 + YaRN factor 64 (4096→262144) | **none** (`mla_use_nope: true`) |
| softmax scale | `192**-0.5 × mscale²` (≈0.1446) | **`192**-0.5` (≈0.0722)** |
| output gate | no | **yes** (`mla_use_output_gate: true`) |
| max context | 262144 | 1048576 |
| MLA params / layer | 101 M | **232 M** |
| MLA params total | 6.17 B | 5.57 B |

### 2.1 NoPE, but the 64 "rope" dims remain

`modeling_kimi_linear.py` sets `self.rotary_emb = None` and hard-`assert`s `use_nope`, yet
`qk_rope_head_dim = 64` is still in the config and still used: `kv_a_proj_with_mqa` emits
`512 + 64 = 576`, q stays `128 + 64 = 192`/head, and `k_rot` is broadcast across all heads. The
64 dims are simply **never rotated**, and they bypass both `kv_a_layernorm` and `kv_b_proj` — a
shared-across-heads, non-positional key channel.

Consequences:
* **Cached latent per token per MLA layer is still 576.** NoPE does *not* shrink it to 512.
  `allocate_mla_kvpe_cache` works unchanged.
* The nope/rope **splits stay** (`mla.py:870-874`, `933-937`): `wkv_b1` absorbs only the 128 nope
  dims, `rms_norm` touches only the 512 latent. Only the rope op itself is removed.
* Absorption / MQA-collapse for decode is unaffected.
* Position information comes exclusively from the KDA layers, so there is no YaRN and no
  rope-interpolation path for context extension.
* Deleted subsystems for these layers: `tt/mla/rope.py` (261 lines), the cos/sin tables, and
  `_apply_rope_padded` + the block-cyclic **indexed-rope** machinery for chunked prefill
  (`mla.py:498-499`, `688-702`). This is the largest simplification K3 buys on the MLA side.

### 2.2 Output gate ("Gated MLA")

```python
if self.use_output_gate:
    g = self.g_proj(hidden_states).sigmoid()   # 7168 -> num_heads*v_head_dim = 12288
    attn_output = attn_output * g
attn_output = self.o_proj(attn_output)
```

`g_proj` is full-rank, +88.1 M params/layer — exactly the size of `o_proj`. It adds
`2 × 88.1 M = 176 MFLOP/token` on top of the layer's ~288, i.e. **~38 % of the MLA layer's GEMM
work**. Across the model, attention GEMMs land roughly even with K2.6 (24 × 464 vs 61 × 202
MFLOP/token) — the gate eats most of what the 3:1 hybrid saves.

### 2.3 Capacity

| | KiB/token/user | 128k | 256k | 1M |
|---|---|---|---|---|
| K2.6 (61 layers × 576 × 2B) | 68.6 | 8.58 GiB | 17.16 GiB | n/a |
| K3 (24 layers × 576 × 2B) | **27.0** | 3.38 GiB | 6.75 GiB | **27.00 GiB** |

(KDA state, for reference: 108.5 M elem/user = 0.40 GiB fp32, ISL-independent.)

## 3. Op-by-op audit

Chunked dense path, SP8×TP4 → `H_loc = 24`, `D_loc = 1792`, `S_loc = 640`.

| # | op | K3 shapes (per device) | verdict |
|---|---|---|---|
| 1 | `linear` q_a_proj | `[1,1,640,1792] × [1792,1536]` | unchanged |
| 2–3 | `reduce_scatter_minimal_async` d3 + `all_gather_async` d3 | 1536 → 384 → 1536 | unchanged |
| 4 | `rms_norm` | 1536 | unchanged |
| 5 | `linear` q_b_proj | `[1536, 4608]` (was 3072) | shape OK, **needs new tuned entry** |
| 6 | `nlp_create_qkv_heads` | 4608, `num_heads=24, num_kv_heads=0` → head_dim 192 | OK: `4608 % 24 == 0` satisfies the head_dim inference check (`nlp_create_qkv_heads.cpp:30`); sharded-input constraints don't apply (interleaved DRAM input) |
| 7 | `slice` ×2 | `[1,24,640,128]`, `[1,24,640,64]` | OK; **rope op deleted** |
| 8 | `linear` wkv_b1 (batched) | in0 batch 24, `[1,24,128,512]` | OK — `_make_batched_mm_kwargs` derives K_t=4, N_t=16, subblock 1×8, in0_block_w=4. Batch 24 vs 16 is irrelevant (`fuse_batch=False`) |
| 9 | `concat` | 512+64 → 576 | unchanged |
| 10–13 | kv stem: `linear` 576, `all_gather` d1 + `fast_reduce_nc`, `slice`, `rms_norm` 512, `kvpe.pack` | 576 / 512 / 64 | **entirely unchanged**; rope op deleted |
| 14 | `transformer.ring_mla` | q `[1,24,640,576]`, `head_dim_v=512`, scale `192**-0.5` | functional; **see finding F2** |
| 15 | `linear` wkv_b2 (batched) | `[1,24,512,128]` | OK — K_t=16, N_t=4 |
| **16** | **gate (new)** | see §4 | **no new ops required** |
| 17 | `nlp_concat_heads` | `[1,24,640,128]` → `[1,1,640,3072]` | OK |
| 18 | `linear` o_proj | `[3072, 7168]` (K was 2048) | shape OK, **needs new tuned entry** |

**Tile alignment**: every width is tile-aligned for tp ∈ {1,2,4,8} — 7168/8=896, 18432/8=2304,
12288/8=1536, and 96 heads/8=12. No padding needed anywhere. `nlp_create_qkv_heads`'s
"1 head per core max" constraint is sharded-input-only and would be satisfied regardless
(24 ≤ 110 cores).

## 4. Gate implementation: zero new ops

Placing the multiply **after** `nlp_concat_heads` means `g` never needs a head split:

```python
g = ttnn.linear(hidden_states, self.g_proj_weight)      # [1,1,640,12288] partial; weight mapper_tp0
g = ttnn.experimental.reduce_scatter_minimal_async(g, dim=3)   # -> [1,1,640,3072], head-sharded
g = ttnn.sigmoid(g)
attn_out = ttnn.experimental.nlp_concat_heads(attn_out)  # [1,1,640,3072] head-major
attn_out = ttnn.multiply(attn_out, g)
out = ttnn.linear(attn_out, self.o_proj_weight)          # -> reduce_scatter
```

`concat_heads` output is head-major over the last dim, and an RS on dim 3 partitions the 12288
into the same contiguous head ranges that `q_b_proj`'s `mapper_tp1` assigns, so the orderings
match with no reshape. `reduce_scatter_minimal_async` is the same op already used at
`mla.py:812`, 8× wider (12288 → 3072 = 96 tiles).

Two layout options for the collective — the residual stream is feature-sharded, so one is
unavoidable:

| | comm per token | sigmoid fusion | notes |
|---|---|---|---|
| **A. K-shard `g_proj` (mapper_tp0) + RS(dim=3)** | 12288-wide RS | **no** — sigmoid must follow the cross-device reduce | smallest code change; mirrors `_q_a_latent` minus the AG. Materializes a 12288-wide activation (15.7 MB at S=640) → must stay in DRAM; at `per_core_M=2, per_core_N=35` the L1 output alone is ~143 KB/core |
| **B. AG hidden to full 7168, shard `g_proj` on output dim (mapper_tp1)** | 7168-wide AG | **yes** | `g` is complete per-device, so `ttnn.linear(activation="sigmoid")` applies — supported via `matmul.cpp:88-94` → `string_to_unary_with_param`; for sharded matmuls it goes in the program config's `fused_activation`, which every `mla_config.py` entry already exposes. Recommended: less traffic, no wide intermediate, one fewer elementwise pass |

Either way the gate is **linear + collective + sigmoid + multiply**, all existing ttnn ops at
supported shapes. Note the gate cannot be moved before `wkv_b2`: it acts in `v_head_dim` space
and `g ⊙ (attn @ W_b2) ≠ (g ⊙ attn) @ W_b2`.

## 5. Findings / action items

**F1 — Immediate hard break: `rope_scaling` is `null`.** `mla.py:330` unconditionally derefs
`config.rope_scaling["factor"]` / `["mscale"]`. K3 raises before anything else runs. The fix is
not just a guard: the correct K3 scale is plain `qk_head_dim**-0.5`, where K2.6 multiplies by
`mscale² ≈ 2.0` (`mla.py:333-336`). Getting this wrong is a silent 2× SDPA-scale error.

**F2 — SDPA k_chunk cliff (biggest perf item).** The tuned 640 SDPA entry carries
`dense_head_cap_non_dsa: 64` (`mla_config.py:368`), and `_get_sdpa_program_config`
(`mla.py:673-675`) discards the config when `num_heads > cap and not _is_dsa_family`. K3 is
96 heads and non-DSA, so **k_chunk drops 640 → 32 and q_chunk → 32**. The cap exists because the
dense path holds full-context K over every head, so L1 footprint scales with head count: K2.6 at
64 heads sat exactly at the ceiling, and 96 heads will genuinely OOM at k=640. K3 needs its own
sweep for the largest L1-safe k_chunk at 24 heads/device; that number will dominate MLA prefill
time.

**F3 — `num_heads = 96` silently disables every tuned matmul config.** `_resolve_mm_cfg`
(`mla.py:579`) drops any config whose declared `num_heads` doesn't match, and all six tuned
640-entries declare 64. K3 therefore falls back to untuned defaults for q_a_proj, q_b_proj,
kv_a_proj, wkv_b1, wkv_b2 and o_proj. Graceful by design, but the whole 640 set needs re-sweeping
— which is required anyway since q_b (4608) and o_proj (K=3072) change width.

**F4 — New weight plumbing.** Add `g_proj` to the prefetch/`as_tensor` block
(`mla.py:131-209`; `mapper_tp0` for option A / `mapper_tp1` for option B, `bfloat8_b`, new
`_cache_name` entry), to `MM_DEFAULT_DTYPES`, and a tuned `mla_config.py` entry. No
`_BATCHED_MM_DIMS` entry — `g_proj` is a plain 2D matmul.

**F5 — KV slot mapping for 24-of-93 layers.** Cache geometry is unchanged (576/token), but the
kvpe cache is a single `num_users × num_layers` user-major slot array indexed by
`cache_layer_idx`. K3 needs a model-layer → kv-slot map (93 → 0..23) instead of passing the layer
index straight through.

**F6 — Adapter shape.** K3 cannot be a thin `MLAPrefillAdapter` subclass the way K2.7 is; the
hybrid layer schedule means it subclasses `PrefillModelAdapter` directly and builds its own
runtime (the case the base docstring already carves out). Out of scope for the MLA work itself.

### Not blockers
* No new device op is required anywhere in the MLA path.
* No dimension is unsupported or needs padding.
* KV cache geometry, `kvpe.pack`, the cache writer, and the whole kv stem are byte-for-byte
  unchanged from K2.6.
