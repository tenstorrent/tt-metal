<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 00 — Model card

**Phase:** P0 · **Date (UTC):** 2026-09-03 · **Gate:** `G-CARD`

Every row below has a `Source`. Nothing here is from memory. Rows whose provenance is a *derivation*
say so and show the formula. Rows marked `UNVERIFIED` also appear in `07_RISKS.md`.

---

## 1. Model identity — resolved, with one open assumption

| Question | Answer | How resolved |
|---|---|---|
| Is `HF_MODEL` set? | **No** — empty | `echo $HF_MODEL` → empty string (run 2026-09-03) |
| Is any Llama checkpoint staged on this machine? | **No** | Verified by the orchestrating session: nothing under `/proj_sw`, `/mnt/MLPerf`, `~/.cache/huggingface`; `HF_MODEL` unset |
| Does a public "Llama-3.2 **8B**" exist? | **No.** The Llama-3.2 text family is 1B / 3B; 11B and 90B are Vision. | In-tree evidence: `models/tt_transformers/model_params/` contains `Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`, `Llama-3.2-11B-Vision-Instruct`, `Llama-3.2-90B-Instruct`, `Llama-3.2-90B-Vision-Instruct` — **no `Llama-3.2-8B*`**. The only 8B Llama in the tree is `Llama-3.1-8B-Instruct`. |
| Resolved dims source | `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json` (37 lines, read in full) | Recipe P0 step 2 fallback path; file verified present |

**The single assumption the user must confirm** (`DEC-001`, `07_RISKS.md` `R-001`):
`llama31_8b_d_p` is taken to mean **`meta-llama/Llama-3.1-8B-Instruct`** — the directory name's `32`
is read as a family/label artefact, not as "Llama-3.2 8B", because no such checkpoint exists.

**Blast radius if the intended target is actually a Llama-3.2 text model.** The Llama-3.2 text
configs differ from Llama-3.1-8B in exactly three keys, all of them small and contained:

| Key | Llama-3.1-8B | Llama-3.2-1B | Llama-3.2-3B | Source |
|---|---|---|---|---|
| `rope_scaling.factor` | `8.0` | `32.0` | `32.0` | `Llama-3.1-8B-Instruct/config.json:27`, `Llama-3.2-1B-Instruct/config.json:27`, `Llama-3.2-3B-Instruct/config.json:27` |
| `tie_word_embeddings` | `false` | `true` | `true` | `Llama-3.1-8B-Instruct/config.json:33`, `Llama-3.2-1B-Instruct/config.json:34`, `Llama-3.2-3B-Instruct/config.json:34` |
| `head_dim` present explicitly | absent (derived) | `64` | `128` | `Llama-3.2-1B-Instruct/config.json:13`, `Llama-3.2-3B-Instruct/config.json:13` |

Everything else (`architectures`, `hidden_act`, `mlp_bias`, `attention_bias`, `rms_norm_eps`,
`rope_theta`, `rope_type`, `vocab_size`, `num_key_value_heads`, `max_position_embeddings`) is
**identical across all three**. So the code shape does not change; only three config-read values do.
Read all three from the config, never hard-code them, and the package retargets by swapping
`configs/<Name>/config.json`. This is why P0 does not stall on the ambiguity.

---

## 2. Architecture facts

Source column: `C:<line>` = `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json`
at that line. `HF:<line>` =
`python_env/lib/python3.12/site-packages/transformers/models/llama/configuration_llama.py`.

| Fact | Value | Source |
|---|---|---|
| architecture | `LlamaForCausalLM` | `C:2-4` (`architectures[0]`) |
| model_type | `llama` | `C:19` |
| layers | **32** | `C:21` (`num_hidden_layers`) |
| hidden | **4096** | `C:14` (`hidden_size`) |
| FFN intermediate | **14336** | `C:16` (`intermediate_size`) |
| activation | **`silu`**, used as SwiGLU: `down(silu(gate(x)) * up(x))` | `C:13` (`hidden_act`); SwiGLU form from the HF module `transformers.models.llama.modeling_llama.LlamaMLP` (`python_env/.../transformers/models/llama/modeling_llama.py`) |
| Q heads | **32** | `C:20` (`num_attention_heads`) |
| KV heads | **8** → GQA, group = 32/8 = **4** | `C:22` (`num_key_value_heads`); group derived |
| head_dim | **128** | **Derived**: `hidden_size / num_attention_heads = 4096/32 = 128`. Key is *absent* from `C` (grep: no `head_dim`). HF applies the same derivation: `HF:84` `head_dim: int \| None = None`, `HF:87-88` `if self.head_dim is None: self.head_dim = self.hidden_size // self.num_attention_heads`. Confirmed at runtime: `LlamaConfig.from_pretrained(...).head_dim == 128`. |
| q_proj out | 32·128 = **4096** | derived |
| k_proj / v_proj out | 8·128 = **1024** | derived |
| o_proj in | 4096 | derived |
| norm | **RMSNorm, plain** (`out = rms_norm(x) * weight`, **no `+1` weight fold**) | `transformers.models.llama.modeling_llama.LlamaRMSNorm`. The `+1` fold is Gemma-specific (`Gemma3RMSNorm`), absent from Llama. |
| `rms_norm_eps` | **1e-05** | `C:24` |
| RoPE θ | **500000.0** | `C:32` (`rope_theta`) |
| RoPE coverage | **full rotary** — `rotary_dim == head_dim == 128` | No `partial_rotary_factor` / `rotary_pct` key in `C` (grep: absent); HF `LlamaAttention` applies RoPE over the whole head |
| RoPE scaling | **`llama3`**: `factor=8.0`, `low_freq_factor=1.0`, `high_freq_factor=4.0`, `original_max_position_embeddings=8192` | `C:25-31` (`rope_scaling`) |
| max positions | **131072** | `C:17` (`max_position_embeddings`) |
| vocab | **128256** (tile-friendly: 128256/32 = 4008) | `C:37` (`vocab_size`) |
| attention bias | **false** | `C:5` (`attention_bias`) |
| MLP bias | **false** | `C:18` (`mlp_bias`) |
| tied embeddings | **false** — `lm_head.weight` is a separate tensor | `C:33` (`tie_word_embeddings`) |
| attention dropout | 0.0 (inference: no-op) | `C:6` |
| `pretraining_tp` | 1 (legacy HF field; no effect) | `C:23` |
| bos / eos | 128000 / [128001, 128008, 128009] | `C:7-12` |
| checkpoint dtype | `bfloat16` | `C:34` (`torch_dtype`) |
| config's `transformers_version` | `4.42.3` | `C:35` — see `R-002`: installed transformers is **5.12.1** |
| QK-norm | **none** | absent from `C` (grep for `qk_norm`, `use_qk_norm`: no hits) |
| attention sinks | **none** | absent from `C` (grep `sink`: no hits) |
| sliding window | **none** — all 32 layers full-causal | absent from `C` (grep `sliding`, `layer_types`: no hits) |
| MoE | **none** — dense FFN on every layer | absent from `C` (grep `expert`, `moe`, `num_local_experts`, `router`: no hits) |

### 2.1 Derived per-layer parameter shapes (HF `[out, in]` layout)

| Weight | HF key | Shape |
|---|---|---|
| `input_layernorm.weight` | `model.layers.N.input_layernorm.weight` | `[4096]` |
| `self_attn.q_proj.weight` | `model.layers.N.self_attn.q_proj.weight` | `[4096, 4096]` |
| `self_attn.k_proj.weight` | `model.layers.N.self_attn.k_proj.weight` | `[1024, 4096]` |
| `self_attn.v_proj.weight` | `model.layers.N.self_attn.v_proj.weight` | `[1024, 4096]` |
| `self_attn.o_proj.weight` | `model.layers.N.self_attn.o_proj.weight` | `[4096, 4096]` |
| `post_attention_layernorm.weight` | `model.layers.N.post_attention_layernorm.weight` | `[4096]` |
| `mlp.gate_proj.weight` | `model.layers.N.mlp.gate_proj.weight` | `[14336, 4096]` |
| `mlp.up_proj.weight` | `model.layers.N.mlp.up_proj.weight` | `[14336, 4096]` |
| `mlp.down_proj.weight` | `model.layers.N.mlp.down_proj.weight` | `[4096, 14336]` |
| (model) `model.embed_tokens.weight` | — | `[128256, 4096]` |
| (model) `model.norm.weight` | — | `[4096]` |
| (model) `lm_head.weight` | — | `[128256, 4096]` (untied, `C:33`) |

Key names verified against a randomly-initialised `LlamaForCausalLM` built from this exact config —
see `01_REFERENCE.md` §4 for the enumerated state-dict listing. **No biases anywhere** (`C:5`,
`C:18`), so the per-layer key set is exactly the nine above, ×32, plus the three model-level keys:
9·32 + 3 = **291 tensors**.

---

## 3. What this model does **NOT** have

This section is as load-bearing as §2. The two closest in-repo templates (`gpt_oss_d_p`,
`minimax_m3`) *do* have most of these, and carrying them across is the most likely source of wasted
work. **Llama-3.1-8B is the simplest shape in this family: dense MLP + GQA + full RoPE + plain
RMSNorm + no biases.**

| Absent feature | Where it would have come from | What to delete when adapting |
|---|---|---|
| **MoE / router / experts / expert-parallelism (EP)** | `gpt_oss_d_p` (EP=32), `minimax_m3`, `deepseek_v3_d_p` | the MoE branch in `layer.py`; `moe*.py`, `router*.py`, dispatch/combine; the `MOE_ROUTING_NUM_CORES` chunk-alignment term in `galaxy_prefill_kv_pcc.py:64-68` (Llama needs only `TILE_SIZE*sp`) |
| **Attention sinks** | `gpt_oss_d_p` (`sinks` in `AttentionConfig`, a sink column in the SDPA reference and a per-ring-iteration sink fold) | `sinks` field, the sink column in the torch reference, the sink fold in `dense_sp.py` |
| **Sliding-window / layer-type alternation** | `gpt_oss_d_p` (`layer_types`, `is_sliding`) | `sliding_window`, `layer_types`, `is_sliding` dispatch in `attention/__init__.py` and `layer.py` |
| **QK-norm** | Qwen3 / Gemma-3 style | nothing — never add it |
| **Partial RoPE** (`rotary_dim < head_dim`) | `deepseek_v3_d_p` (MLA), phi-style | nothing — Llama rotates the full 128 |
| **MLA (compressed KV / merged latent cache)** | `deepseek_v3_d_p`, Kimi | the merged-MLA KV reader is the *wrong default* for Llama — see P10 step 5 |
| **Sparse / MSA attention** | `minimax_m3` | nothing |
| **MXFP4 / block-quantised weight loaders** | `gpt_oss_d_p` | the MXFP4 dequant path; Llama ships plain bf16 safetensors (`C:34`) |
| **Any bias tensor** | most templates carry an optional-bias branch | `attention_bias=false` (`C:5`), `mlp_bias=false` (`C:18`) — assert bias keys are absent rather than branching |
| **Tied embeddings** | Llama-3.2-1B/3B *do* tie | `C:33` is `false` for 3.1-8B: `lm_head.weight` is its own tensor. Do not alias it. |
| **Decode / trace / 2CQ** | out of scope this iteration | recipe "Non-goals" (`BRINGUP_RECIPE.md:15-16`) |

---

## 4. Deployment target — `(mesh_shape, TP, SP)` with the arithmetic

### 4.1 The hardware, measured

| Fact | Value | Source |
|---|---|---|
| `ttnn.get_num_devices()` | **32** | run 2026-09-03, this machine |
| arch | **blackhole** | UMD log: `Creating TopologyDiscovery for architecture: blackhole` |
| local chip ids | 0…31, all local (no remote) | UMD log: `Opening local chip ids/PCIe ids: {0..31}`, `remote chip ids {}` |
| physical single-galaxy device topology | **`dims: [8, 4]`** (= 32), both axes `RING` in the torus variant | `tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto:6`; non-torus variant `tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto:6` has the same `[8, 4]` without `dim_types` |
| fabric channels per link | **2** | same file, `:8` `channels { count: 2 policy: STRICT }` — consistent with `num_links = 2` on Blackhole |

### 4.2 Mesh-shape convention: `mesh_shape = (SP, TP)`

Not guessed — read from the engine:

- `models/demos/common/prefill/adapter.py:57` — `mesh_shape: tuple  # (sp, tp)`
- `models/demos/common/prefill/runners/runner_utils.py:78` — `sp_factor, tp_factor = mesh_shape`
- `models/demos/common/prefill/runners/runner_utils.py:22` — `sp = mesh_shape[0]`
- `models/demos/common/prefill/runners/migration.py:295-296` — `rows = mesh_shape[0]`, `cols = mesh_shape[1]`

So **rows = SP, cols = TP**, and `SP · TP = 32`.

### 4.3 TP constraints — derived, one line each

| # | Constraint | Arithmetic | Admissible TP |
|---|---|---|---|
| 1 | TP must divide `num_key_value_heads` or KV heads get replicated | `8 / TP ∈ ℤ` | `{1, 2, 4, 8}` |
| 2 | TP must divide `num_attention_heads` | `32 / TP ∈ ℤ` | `{1, 2, 4, 8, 16, 32}` |
| 3 | `hidden / TP` must be tile-aligned | `4096/TP ≡ 0 (mod 32)` ⇔ `TP \| 128` | `{1,2,4,8,16,32,64,128}` |
| 4 | `intermediate / TP` must be tile-aligned | `14336/TP ≡ 0 (mod 32)` ⇔ `TP \| 448` (since `14336/32 = 448 = 2⁶·7`) | `{1,2,4,7,8,14,16,28,32,56,64,112,224,448}` |
| 5 | `o_proj` row-parallel over `num_heads·head_dim` | `4096/TP ≡ 0 (mod 32)` — same as #3 | as #3 |

**Intersection: TP ∈ {1, 2, 4, 8}.** Constraint #1 is the binding one; TP > 8 would require
replicating KV heads, which the recipe requires a `DEC` for (`BRINGUP_RECIPE.md:270-271`) and which
this iteration does not need.

With `SP · TP = 32`, the four candidate meshes are:

| mesh `(SP, TP)` | Q heads/chip `32/TP` | KV heads/chip `8/TP` | hidden shard `4096/TP` (tiles) | inter shard `14336/TP` (tiles) | `CHUNK_SIZE` must be ≡0 mod `32·SP` | ring-SDPA hops (SP) |
|---|---|---|---|---|---|---|
| `(4, 8)` | 4 | **1** | 512 (16) | 1792 (56) | **128** | 4 |
| `(8, 4)` | 8 | 2 | 1024 (32) | 3584 (112) | **256** | 8 |
| `(16, 2)` | 16 | 4 | 2048 (64) | 7168 (224) | 512 | 16 |
| `(32, 1)` | 32 | 8 | 4096 (128) | 14336 (448) | 1024 | 32 |

All four are *arithmetically legal*. Every shard is tile-aligned; no KV replication in any of them.

### 4.4 SP constraints

Source: `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md` "Shared setup" —
`CHUNK_SIZE % (SP*32) == 0` and `MAX_SEQ_LEN % CHUNK_SIZE == 0`; plus
`ttnn.experimental.deepseek_prefill.update_padded_kv_cache`'s `kv_actual_global % 32 == 0`
(`BRINGUP_RECIPE.md:701-703`).

Worked for the two serious candidates at a conventional `CHUNK_SIZE = 1024`:

- `(4, 8)`: `32·SP = 128`; `1024 % 128 = 0` ✓. `MAX_SEQ_LEN = 131072`; `131072 % 1024 = 0` ✓.
  Per-chip local sequence `S_loc = S/4`; at `S = 1024`, `S_loc = 256` = 8 tiles ✓.
- `(8, 4)`: `32·SP = 256`; `1024 % 256 = 0` ✓. `131072 % 1024 = 0` ✓.
  `S_loc = S/8`; at `S = 1024`, `S_loc = 128` = 4 tiles ✓.

Both hold. `(4,8)` is the more permissive of the two (chunk granularity 128 vs 256), which matters
for short-prompt and pad-tail cases.

### 4.5 The choice

> **Chosen: `mesh_shape = (4, 8)`, TP = 8, SP = 4.** (`DEC-002`)

Arithmetic, restated as the numbers the code will carry:

```
devices        = 32                        (measured)
mesh_shape     = (SP, TP) = (4, 8)         SP·TP = 4·8 = 32                    ✓
TP = 8:  32 Q heads / 8 =  4 Q heads per chip                                  ✓ integral
         8 KV heads / 8 =  1 KV head  per chip  -> no KV replication            ✓ (TP=8 is the max
                                                                                  such TP)
         GQA group      = 4 Q heads share 1 KV head per chip (= global 32/8)    ✓ preserved locally
         4096 hidden    / 8 =  512  =  16 tiles                                 ✓ tile-aligned
         14336 inter    / 8 = 1792  =  56 tiles                                 ✓ tile-aligned
         4096 o_proj in / 8 =  512  =  16 tiles                                 ✓ tile-aligned
SP = 4:  CHUNK_SIZE % (32·4 = 128) == 0    -> CHUNK_SIZE ∈ {128, 256, ..., 1024, ...}
         MAX_SEQ_LEN % CHUNK_SIZE == 0     -> 131072 % 1024 == 0                ✓
         S_loc = S / 4
num_links = 2 on Blackhole                 (galaxy descriptor `channels { count: 2 }`)
```

**Why `(4,8)` and not `(8,4)`:**

1. **It is the shape validated on this exact hardware.** `models/demos/gpt_oss_d_p/README.md:6` —
   "Long-context **prefill** for GPT-OSS-120B on **4×8 Blackhole Galaxy** (TP=8, SP=4, EP=32)"; and
   `models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:44` — `ROWS, COLS = 4, 8  # SP=4 (rows),
   TP=8 (cols), EP=32`, passed to `mesh_shape=(ROWS, COLS)` at `:154`, under
   `TT_MESH_GRAPH_DESC_PATH=.../single_bh_galaxy_torus_xy_graph_descriptor.textproto` (`:26`).
   `gpt_oss_d_p` is the package Llama's attention/KV-cache/CCL code is being adapted from, so taking
   its mesh means the ring-SDPA grid offsets, the block-cyclic KV geometry, and the migration device
   map are all exercised in a configuration that already has a green gate — the P8 failure surface
   shrinks to "is Llama's math right", not "is this mesh viable".
2. **TP = 8 is exactly the largest TP with no KV replication** (constraint #1 is tight at `8/8 = 1`).
   Going to TP = 4 leaves KV parallelism on the table; going past 8 needs a replication `DEC`.
3. **Fewer ring-SDPA hops.** SP = 4 means the P8 ring-attention halo exchange is a 4-step ring, not
   an 8-step one — half the halo iterations, half the semaphore hand-offs, and a proportionally
   smaller race surface for `G-RACE`.
4. **Coarser SP means a larger `S_loc`**, which keeps per-chip matmul shapes further from the
   degenerate small-M regime at short prompts.

**The alternative, stated with its numbers: `(8, 4)`, TP = 4, SP = 8.**
Per chip: 8 Q heads, 2 KV heads, hidden shard 1024 (32 tiles), intermediate shard 3584 (112 tiles),
`CHUNK_SIZE % 256 == 0`, `S_loc = S/8`, 8-step SP ring. It is **also fully legal** and it is in fact
the *engine's coded default* — `models/demos/common/prefill/runners/prefill_producer.py:83-84`:
`SP_AXIS = int(os.environ.get("PREFILL_SP", 8))`, `TP_AXIS = int(os.environ.get("PREFILL_TP", 4))`.
Its genuine advantages: the TP collective spans 4 chips instead of 8 (less all-reduce traffic per
sublayer), and 2 KV heads per chip is a slightly less degenerate SDPA shape than 1. Its costs: an
8-step ring SDPA in P8, coarser chunk quantisation (256), and it is *not* the shape the
`gpt_oss_d_p` scaffolding this package borrows is validated at.

Because the engine reads SP/TP from `PREFILL_SP`/`PREFILL_TP`, both shapes must remain reachable:
`MeshConfig` takes TP as its only knob and derives SP (recipe P4), so `(8,4)` is a parameter change,
not a rewrite. `_VALIDATED_MESH_SHAPE = (4, 8)` / `_VALIDATED_TP = 8` records the *tested* shape;
`(8,4)` stays legal-but-untested until someone runs `G-MESH-KV` on it.

**Single-card phases (P5–P7) run `(1, 1)`, TP = 1, SP = 1, no CCL** — recipe `BRINGUP_RECIPE.md:578`.
`1` satisfies every constraint in §4.3 trivially.

### 4.6 Per-chip tensor shapes at the chosen target

`S` = global sequence length, `S_loc = S/SP = S/4`, `TP = 8`. (Dtype/layout choices are P3/P4
decisions; this table fixes only the *shapes*, which follow from §4.5 alone.)

| tensor | shape (per chip) |
|---|---|
| hidden, replicated residual | `[1, 1, S_loc, 4096]` |
| hidden, TP-sharded residual (scheme B) | `[1, 1, S_loc, 512]` |
| Q | `[1, 4, S_loc, 128]` |
| K, V | `[1, 1, S_loc, 128]` |
| attn out (pre-`o_proj`) | `[1, 4, S_loc, 128]` |
| MLP gate / up | `[1, 1, S_loc, 1792]` |
| `q_proj` weight (ttnn `[in, out]`) | `[1, 1, 4096, 512]` |
| `k_proj` / `v_proj` weight | `[1, 1, 4096, 128]` |
| `o_proj` weight (row-parallel) | `[1, 1, 512, 4096]` |
| `gate_proj` / `up_proj` weight | `[1, 1, 4096, 1792]` |
| `down_proj` weight (row-parallel) | `[1, 1, 1792, 4096]` |

---

## 5. `UNVERIFIED` rows

None of the §2 rows is `UNVERIFIED` — every one is either a literal `config.json` line or a shown
derivation. The open items are *identity and environment* items, not architecture facts, and each
has a `07_RISKS.md` entry:

| Item | Risk id |
|---|---|
| `llama31_8b` ⇒ Llama-3.1-8B-Instruct is an assumption, not a confirmed instruction | `R-001` |
| Config was authored for transformers 4.42.3; installed is 5.12.1, which moved `rope_theta` inside `rope_parameters` (reading `config.rope_theta` yields `None`) | `R-002` |
| Dims come from the bundled config, **not** from a live checkpoint — no `HF_MODEL`, no safetensors on this machine. Real-weight gates are `BLOCKED`. | `R-003` |
| TP = 8 gives 1 KV head per chip; no op in the SDPA / KV-write path has been checked for a `num_kv_heads > 1` assumption | `R-004` |
