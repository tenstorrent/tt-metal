# Kimi-K3 MLA: architecture delta vs Kimi-K2.6 and TT op-support audit

Scope: the **full-attention (MLA) layers only**. K3's 69 KDA (linear-attention) layers, the
AttnRes residual scheme, and LatentMoE are out of scope here and are only mentioned where they
change what the MLA layer sees.

Sources: `config.json`, `configuration_kimi_k3.py` and `modeling_kimi_linear.py` from
`huggingface.co/moonshotai/Kimi-K3`, against `huggingface.co/moonshotai/Kimi-K2.6`
(`DeepseekV3ForCausalLM`). TT-side line references are against `models/demos/deepseek_v3_d_p/`
at the time of writing (branched off `5700827bc39`).

> **§0 — corrections and additions from implementation.** The audit below is the original analysis;
> these points were established while implementing it and supersede it where they conflict.
>
> 1. **The checkpoint config is a multimodal wrapper.** `architectures` is
>    `["KimiK3ForConditionalGeneration"]` with `text_config` (`model_type: kimi_linear`) +
>    `vision_config`. Every field below lives under `text_config`. `unwrap_multimodal_config`
>    (`tt/runners/adapters/mla.py:39`) and `_unwrap_multimodal_config` (`tests/conftest.py:589`)
>    already handle this, so it costs nothing.
> 2. **F1's failure mode depends on how the config was built, and an `is not None` guard is not
>    enough.** `rope_scaling` is absent from `text_config`'s JSON, and `KimiLinearConfig` defaults it
>    to `None` — but `PretrainedConfig.__init__` on transformers ≥ 5 *synthesizes*
>    `{"rope_theta": 10000.0, "rope_type": "default"}` for any config whose JSON omits it. So a real
>    `AutoConfig`-loaded K3 raises **`KeyError: 'factor'`** at `mla.py:330`, while a hand-built
>    `SimpleNamespace(rope_scaling=None)` raises `TypeError`. The guard must therefore test for the
>    **`"factor"` key**, not for `None` (verified: transformers 5.12.1 does this to
>    `DeepseekV3Config()` too). K2.6/GLM-5.2/V3.2 scales are bit-identical under the key-based guard.
> 3. **The MLA weights are not quantized.** The checkpoint is MXFP4 (`compressed-tensors`, 4-bit,
>    `group_size: 32`), but `quantization_config.ignore` contains `re:.*self_attn.*` — so every MLA
>    weight, `g_proj` included, is plain bf16. No dequant work in the MLA scope. Only the MoE routed
>    experts are quantized.
> 4. **`full_attn_layers` is 1-indexed.** `KimiLinearConfig.is_kda_layer` tests
>    `(layer_idx + 1) in kda_layers`, so the 0-indexed MLA layers are `[3, 7, …, 87, 91, 92]`.
>    **91 and 92 are adjacent** — the 3:1 pattern breaks at the tail, so a stride-4 map is wrong.
>    See `KimiK3Config.mla_layer_ids()` / `mla_kv_slot()`.
> 5. **F2's stated rationale does not hold, and the cap does not apply to K3.** Every CB in
>    `exp_ring_joint_sdpa_program_factory.cpp:358-365` is sized from `Sq_chunk_t`, `Sk_chunk_t` and
>    `DHt` — there is no `num_heads` term. Head count changes the number of work units (runtime), not
>    per-core L1, and K3's K/V is the same MQA-collapsed 1-head 576-wide latent as K2.6's. Measured:
>    every `k_chunk` from 32 to 640 fits at 24 heads/device, and the k=32 fallback the cap forces
>    costs **2.36× device time**. K3's entry carries no cap — see the table under F2. (What the cap
>    empirically protects is V3.1 at 128 heads; the comment at `mla.py:665-672` describing it as an
>    L1-footprint-per-head effect is misleading and has been corrected in place.)
> 6. **`ring_mla` imposes no head-count constraint** — checked every `TT_FATAL` in
>    `exp_ring_joint_sdpa_device_operation.cpp`. 24 heads/device is fine.
> 7. **One tuned config per `(weight, seq_len)`** — a constraint the original audit missed.
>    `MLA_MATMUL_CONFIG[name][seq_len]` and `MLA_SDPA_CONFIG[seq_len]` each held a single dict, and the
>    gating tags only *reject* that one candidate — they cannot choose among several. K2.6 and K3 both
>    want the `640` slot, so those slots now accept a **list of candidates**; `ttMLA._select_cfg` takes
>    the first whose tags match (`_cfg_matches`, shared by the matmul and SDPA resolvers).
> 8. **Upstream `modeling_kimi_linear.py` cannot be vendored whole**: it raises
>    `ImportError("Plese run 'pip install -U fla-core'")` at module import, and `fla` is a triton/GPU
>    library needed only by the KDA layers. `reference/kimi_k3/modeling_kimi_k3_mla.py` is a trimmed
>    MLA-only copy (also dropping the `ALL_ATTENTION_FUNCTIONS` indirection, whose surface moves
>    between transformers majors).
> 9. **Local mesh equivalence.** SP2×TP4 on an 8-chip 2x4 Blackhole box reproduces every per-device
>    shape in §3 (`H_loc=24`, `D_loc=1792`, `S_loc=640`); only `ring_size` changes 8→2. Note the
>    chunked-prefill driver derives `S_loc = chunk_size_global // sp`, so reaching `S_loc=640` at
>    sp=2 needs `chunk_size_global=1280`, not the default 5120.

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

**Measured**, one MLA layer's forward at the same per-device geometry (S_loc=640, tp=4, chunk 1280 on
a 2x4 Blackhole box; between the `MLA_START`/`MLA_END` signposts, tuned configs active):

| | Matmul | CCL | SDPA | Other | **Total** |
|---|---|---|---|---|---|
| Kimi-K2.6 (64 heads, no gate) | 708 µs | 1466 µs | 1369 µs | 645 µs | **4188 µs** |
| Kimi-K3 (96 heads, gated) | 1357 µs | 2002 µs | 1704 µs | 812 µs | **5875 µs** |
| delta | **+92 %** | +37 % | +24 % | +26 % | **+40 %** |

The matmul near-doubling is the 1.5× head count *and* `g_proj` together, consistent with the ~38 %
GEMM estimate above. CCL is the single largest bucket in both, and K3 adds one TP all-gather to it.

Per *model*, though, the hybrid wins outright rather than breaking even: 24 × 5875 µs ≈ 141 ms vs
61 × 4188 µs ≈ 255 ms, i.e. K3's whole MLA stack costs **~55 %** of K2.6's. (Layer-count scaling of a
single-layer measurement — indicative, not a model-level benchmark.)

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

> **Status.** F1–F4 are **implemented and tested**; F5–F6 remain open (they need the hybrid layer
> schedule, which is out of scope for the MLA module). What landed:
>
> | | |
> |---|---|
> | `reference/kimi_k3_config.py` | `KimiK3Config` + `kimi_k3_hf_config()`; `mla_layer_ids()` / `mla_kv_slot()` for F5 |
> | `reference/kimi_k3/` | trimmed upstream `KimiMLAAttention` (unabsorbed truth model, no `fla` dep) |
> | `reference/mla_reference.py` | `MLAReference` now config-driven for NoPE + gate |
> | `tt/mla/mla.py` | scale guard, NoPE rope bind, `g_proj` plumbing, `_output_gate` + gated `_o_proj_epilogue`, multi-candidate config resolution |
> | `tt/mla/mla_config.py` | K3 `640` candidates for all 6 matmuls + `g_proj`; uncapped K3 SDPA entry |
> | `tt/mla/rope.py` | `RotarySetup` short-circuits to `{}` under NoPE |
> | `tt/runners/adapters/kimi_k3.py` | test-only adapter (`build_runtime`/`allocate_kv_cache` raise, pointing at F5/F6) |
> | tests | `tests/torch/test_kimi_k3_mla_reference.py`, `tests/op_unit_tests/test_kimi_k3_mla_matmuls.py`, `test_kimi_k3_gate.py`, `test_mla_config_resolution.py`, `test_mla.py::test_kimi_k3_mla`, `kimi_k3` in `test_mla_chunked_prefill` (+ `chunk1280` scenarios), `tests/cache/test_mla_cache.py` (+ a runnable 2x4 case), `tests/perf/test_mla_perf.py::test_kimi_k3_mla_chunked_perf_loudbox`, `kimi_k3` in `test_ring_joint_sdpa.py` |
>
> Every K3 branch is flag-gated on `mla_use_nope` / `mla_use_output_gate`, and the K2.6 / V3 / GLM
> PCCs are bit-identical before and after.

**F1 — Immediate hard break: `rope_scaling` carries no `"factor"`.** `mla.py:330` unconditionally
derefs `config.rope_scaling["factor"]` / `["mscale"]`. K3 raises before anything else runs. The fix
is not just a guard: the correct K3 scale is plain `qk_head_dim**-0.5`, where K2.6 multiplies by
`mscale² ≈ 2.0` (`mla.py:333-336`). Getting this wrong is a silent 2× SDPA-scale error.
See §0.2 — the guard must key on the `"factor"` key, not on `rope_scaling is None`. **Fixed.**

**F2 — SDPA k_chunk cliff (biggest perf item).** The tuned 640 SDPA entry carries
`dense_head_cap_non_dsa: 64` (`mla_config.py:368`), and `_get_sdpa_program_config`
(`mla.py:673-675`) discards the config when `num_heads > cap and not _is_dsa_family`. K3 is
96 heads and non-DSA, so **k_chunk drops 640 → 32 and q_chunk → 32**. K3 needs its own sweep for the
largest L1-safe k_chunk at 24 heads/device; that number will dominate MLA prefill time.

*Revised (§0.5): the original rationale here — "the dense path holds full-context K over every head,
so L1 footprint scales with head count" — is not supported by the program factory, which sizes every
CB from `Sq_chunk_t`/`Sk_chunk_t`/`DHt` alone.*

**MEASURED, and the cap does not apply to K3.** `test_ring_mla_chunked_accuracy[kimi_k3-q32-k*]`
(24 heads/device, `d_q=d_k=576`, latent `d_v=512`, 11 chunks of 5120 to 56320 on a 2x4 Blackhole box):

| k_chunk | 32 | 128 | 256 | 512 | 640 |
|---|---|---|---|---|---|
| final-chunk PCC | 0.99590 | 0.99919 | 0.99936 | 0.99937 | **0.99938** |
| L1 OOM | no | no | no | no | **no** |
| total device time (11 chunks) | 141.4 ms | 77.6 ms | 67.7 ms | 63.9 ms | **60.0 ms** |
| final-chunk math util | 27.9 % | 50.9 % | 58.5 % | 63.8 % | **65.9 %** |

(perf from `test_ring_mla_create_chunked_perf_table[kimi_k3-q32-k*]`, 100 SDPA cores)

Every value fits, k=640 included — so the `dense_head_cap_non_dsa` fallback is not protecting K3 from
anything, and it is expensive: **k=32 costs 2.36× the device time** of k=640 (141.4 vs 60.0 ms) and
less than half the math utilization, *plus* ~0.0035 PCC. Larger k_chunk wins monotonically on both
axes here. **K3's tuned entry therefore carries no cap and uses k=640**, matching K2.6's tiling.

This makes F2 a one-line config fix rather than the sweep-for-a-safe-smaller-value exercise the
original text implies — and it was indeed the largest single perf item on the MLA side.

**F3 — `num_heads = 96` silently disables every tuned matmul config.** `_resolve_mm_cfg`
(`mla.py:579`) drops any config whose declared `num_heads` doesn't match, and all six tuned
640-entries declare 64. K3 therefore falls back to untuned defaults for q_a_proj, q_b_proj,
kv_a_proj, wkv_b1, wkv_b2 and o_proj. Graceful by design, but the whole 640 set needs re-sweeping
— which is required anyway since q_b (4608) and o_proj (K=3072) change width.

*Revised: "the whole 640 set needs re-sweeping" overstates the work — most of it is re-tagging.*
Measured at the per-device shapes with `test_kimi_k3_mla_matmuls.py` (all PCC 0.9999 on a 2x4 box):

| weight | per-device shape change | outcome |
|---|---|---|
| `q_a_proj` | none (K = hidden/tp = 1792 for both) | K2.6's config **transfers verbatim** |
| `kv_a_proj_with_mqa` | none | K2.6's config **transfers verbatim** |
| `o_proj` | K 2048 → 3072; N is the full 7168 either way | K2.6's config **transfers** (`in0_block_w=8` divides K_t 64 and 96) |
| `q_b_proj` | N 3072 → **4608** (N_t 96 → 144) | new: `per_core_N=14`, subblock 1×7 |
| `wkv_b1` / `wkv_b2` | batch (= `H_loc`) 16 → **24** | new: K2.6's `per_core_M=4` needs `24 × (20/4) = 120` blocks on a 110-core grid → **overflows**; `per_core_M=5` gives 96 |
| `g_proj` | new op, `[7168, 3072]` per device | new: `per_core_N=9`, subblock 2×3, `fused_activation=sigmoid` |

So only `q_b_proj`, `wkv_b1`/`wkv_b2` and `g_proj` needed real work; the batched pair is the one
place the head-count increase is genuinely not free.

**F4 — New weight plumbing.** Add `g_proj` to the prefetch/`as_tensor` block
(`mla.py:131-209`; `mapper_tp0` for option A / `mapper_tp1` for option B, `bfloat8_b`, new
`_cache_name` entry), to `MM_DEFAULT_DTYPES`, and a tuned `mla_config.py` entry. No
`_BATCHED_MM_DIMS` entry — `g_proj` is a plain 2D matmul.
**Done** (option B, `mapper_tp1`), except the tuned `mla_config.py` entry, which is blocked on §0.7.
One site the original list missed: `MLA_WEIGHT_NAMES` drives `check_cache_complete`, so `g_proj`
must be appended *conditionally* (`ttMLA.weight_names(has_output_gate)`) or every existing non-gated
weight cache starts reporting itself incomplete.

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
