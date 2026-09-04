# 00 — Model card — Llama-3.1-8B-Instruct (`models/demos/llama31_8b_d_p`)

Phase P0. Every row carries a **Source**: a `config.json` key, a `path:line`, or a shown derivation.
A row that could not be verified is marked `UNVERIFIED` and copied into `07_RISKS.md`.
Date (UTC): 2026-09-04.

`config.json` below always means the checkpoint config resolved in §1 and bundled verbatim at
`models/demos/llama31_8b_d_p/configs/Llama-3.1-8B-Instruct/config.json`.

## 1. Identity

| Field | Value | Source |
|---|---|---|
| HF repo | `meta-llama/Llama-3.1-8B-Instruct` | the bundled config's dims are byte-identical to the repo's own vendored copy for that id, `models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json:13` (see below) |
| Local checkpoint | `/home/mstojkovic/models/Llama-3.1-8B-Instruct` — 4 safetensors shards + `model.safetensors.index.json` + tokenizer + `config.json` | `echo $HF_MODEL`; `ls $HF_MODEL` |
| Bundled copy | `models/demos/llama31_8b_d_p/configs/Llama-3.1-8B-Instruct/config.json` | copied verbatim from `$HF_MODEL/config.json`; both `md5 = 3cd5831d379b509d53afade0e24c36e9` |
| How the identity was resolved | `llama31_8b` is a directory name, not a HF id. `$HF_MODEL/config.json` was diffed against the repo's vendored `Llama-3.1-8B-Instruct` config: **byte-identical** (same md5), not merely dim-compatible. `architectures[0] == "LlamaForCausalLM"`, `vocab_size 128256` and `rope_scaling.rope_type == "llama3"` fix it as a Llama-3.x 8B instruct checkpoint. | `DEC-001` |
| Ambiguity | none for *this* run. There is no public "Llama-3.2-8B"; the target was named Llama-3.**1**-8B-Instruct by the environment and by the staged checkpoint, and they agree. | `DEC-001` |

The checkpoint dir also contains `P150/` and `ttnn_cache/` (pre-existing tilized-weight caches from
another package). They are **not** inputs to this bring-up; see `07_RISKS.md` R-003.

## 2. Architecture

All values read from `config.json` on 2026-09-04; no row is from memory.

| Fact | Value | Source |
|---|---|---|
| architecture | `LlamaForCausalLM` | `architectures[0]` |
| model_type | `llama` (first-class in `transformers`, no `trust_remote_code`) | `model_type` |
| layers | 32 | `num_hidden_layers` |
| hidden | 4096 | `hidden_size` |
| FFN intermediate | 14336 | `intermediate_size` |
| activation | `silu`; SwiGLU as `down(silu(gate(x)) * up(x))` | `hidden_act` |
| Q heads | 32 | `num_attention_heads` |
| KV heads | 8 → **GQA, group = 32/8 = 4** | `num_key_value_heads`, derived |
| head_dim | 128 | derived: `hidden_size / num_attention_heads = 4096 / 32 = 128`. No `head_dim` key exists in this `config.json` |
| norm | RMSNorm, **plain** (no `+1` weight fold — that is Gemma, not Llama) | `rms_norm_eps`; HF anchor `transformers.models.llama.modeling_llama.LlamaRMSNorm` |
| norm eps | 1e-05 | `rms_norm_eps` |
| RoPE theta | 500000.0 | `rope_theta` **as a key in the JSON file**. On `transformers` 5.12.1 the *config object* has no such attribute — P1 trap 1; read it through `models/tt_transformers/tt/common.py:165` `get_rope_theta`, which takes a **dict** |
| RoPE coverage | **full rotary**: rotary_dim = head_dim = 128 (no `partial_rotary_factor` key) | absent from config → HF default 1.0 |
| RoPE scaling | `rope_type: llama3`, factor 8.0, low_freq_factor 1.0, high_freq_factor 4.0, original_max_position_embeddings 8192 | `rope_scaling`; read through `models/tt_transformers/tt/common.py:183` `get_rope_scaling` |
| max positions | 131072 | `max_position_embeddings` |
| vocab | 128256 | `vocab_size` |
| attention bias | false | `attention_bias` |
| attention dropout | 0.0 (inference: no-op) | `attention_dropout` |
| MLP bias | false | `mlp_bias` |
| tied embeddings | false → a **separate `lm_head.weight`** exists | `tie_word_embeddings` |
| checkpoint dtype | `bfloat16` — the **reference must not inherit it** (§2.1 of the recipe): build the torch oracle in fp32 | `torch_dtype`; policy in `01_REFERENCE.md` |
| bos / eos | 128000 / [128001, 128008, 128009] | `bos_token_id`, `eos_token_id` |
| QK-norm | none | key absent from `config.json` |
| attention sinks | none | key absent (`sinks` is a gpt-oss feature) |
| sliding window | none — every layer is full-causal; there is no `layer_types` schedule | keys `sliding_window` / `layer_types` absent |
| MoE | none — **dense FFN on every one of the 32 layers** | keys `num_local_experts` / `num_experts_per_tok` / `moe_layer_freq` absent |
| `pretraining_tp` | 1 (a checkpoint-provenance field; not a runtime knob here) | `pretraining_tp` |

Derived parameter geometry used later (arithmetic shown, nothing from memory):

| Quantity | Derivation |
|---|---|
| q_proj | `[32*128, 4096] = [4096, 4096]` |
| k_proj / v_proj | `[8*128, 4096] = [1024, 4096]` |
| o_proj | `[4096, 32*128] = [4096, 4096]` |
| gate/up_proj | `[14336, 4096]`; down_proj `[4096, 14336]` |
| embed_tokens / lm_head | `[128256, 4096]` each (untied) |

## 3. What this model does NOT have

The anti-bloat control. The two nearest in-repo templates (`models/demos/gpt_oss_d_p`,
`models/demos/minimax_m3`) have every feature listed here; copying them in is the single most likely
source of wasted work.

- **No MoE** — no router, no experts, no shared expert, no EP dispatch/combine, no
  `unified_routed_expert_ffn`. Every layer is a dense SwiGLU MLP.
- **No attention sinks** — no per-head learned sink logit, no extra softmax column.
- **No sliding-window attention and no per-layer alternation** — no `layer_types`, no window.
- **No QK-norm.**
- **No partial RoPE** — rotary_dim == head_dim (M3 rotates only 64 of 128).
- **No YaRN / no mscale** — the scaling is `llama3` (piecewise `apply_scaling`,
  `models/tt_transformers/tt/common.py:437`), not YaRN.
- **No MLA and no sparse / MSA attention** — plain GQA, dense causal SDPA.
- **No biases anywhere** — `attention_bias` and `mlp_bias` are both false; no `q/k/v/o_proj.bias`,
  no `gate/up/down_proj.bias`. Every bias-handling branch in the templates is dead code here.
- **No MXFP4 / quantised weight loader** — the checkpoint is plain bf16 safetensors.
- **No tied embeddings** — `lm_head.weight` is a real, separate tensor.

Llama-3.1-8B is the **simplest** shape in this family: dense MLP + GQA + full RoPE + plain RMSNorm.

## 4. Deployment target

| Field | Value | Derivation |
|---|---|---|
| mesh shape | `(4, 8)` — 4 rows × 8 columns of the 32-device Blackhole Galaxy | `ttnn.get_num_devices() == 32`, `ttnn.get_arch_name() == 'blackhole'` (measured, `raw/G-CARD_20260904T034303Z.log`); the same target `models/demos/gpt_oss_d_p/tt/config.py:15` pins as validated |
| TP | **8**, on the mesh **columns** (axis 1) | hard equality + divisibility, both shown below |
| SP | **4**, on the mesh **rows** (axis 0) — the other axis, derived not chosen | `SP = 32 / TP = 4`; recipe P4: TP is the only knob |
| chunk / max_seq_len | constrained, value deferred to P7 | `CHUNK_SIZE % (SP*32) == 0` and `MAX_SEQ_LEN % CHUNK_SIZE == 0`, source `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md:62`. At SP=4 that is `CHUNK_SIZE % 128 == 0`. See `DEC-004` |

### 4.1 Why TP is exactly 8 — an equality, not a bound

**Upper and lower bound collapse onto `num_key_value_heads = 8`:**

1. **`TP <= 8` is forced by the packed KV cache.** The per-chip cache is allocated with a hard-coded
   num-heads dimension of 1 — `models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:95` ("Per-chip cache is one
   head") and the `torch.zeros(num_users * num_layers, 1, seq_local, head_dim)` at
   `models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:99`. At TP < 8 the model produces `8/TP > 1` local KV heads and the
   write op aborts:
   `TT_FATAL(cache_shape[1] == input_shape[1], "cache and input num-heads dim must match")` —
   `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp:230`.
   This bites **including on chunk 0**, so on a single card no model-level KV write is possible at all.
2. **`TP >= 8` is forced by not wanting to replicate KV heads.** At TP > 8 (not reachable on a
   32-device mesh with SP ≥ 4 anyway) the 8 KV heads no longer cover the TP group and would have to
   be replicated — its own `DEC`, not taken.
3. **SDPA itself is satisfied either way** and therefore does not constrain TP:
   `TT_FATAL(nqh >= nkv && nqh % nkv == 0, ...)` —
   `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:98`. At TP=8 the local counts are
   `nqh = 32/8 = 4`, `nkv = 8/8 = 1`, giving `4 >= 1 && 4 % 1 == 0`. ✅

### 4.2 TP divisibility arithmetic (tile alignment)

Tiles are 32×32, so every per-chip shard dim must be a multiple of 32.

| Dim | Full | `/TP` at TP=8 | `% 32` | OK |
|---|---|---|---|---|
| hidden | 4096 | 512 | `512 = 16*32` → 0 | ✅ |
| intermediate | 14336 | 1792 | `1792 = 56*32` → 0 | ✅ |
| Q heads | 32 | 4 | — (head count, `32 % 8 == 0`) | ✅ |
| KV heads | 8 | 1 | — (`8 % 8 == 0`, one per chip) | ✅ |
| head_dim | 128 | not sharded | `128 = 4*32` | ✅ |
| vocab | 128256 | 16032 | `16032 = 501*32` → 0 | ✅ |

Sharded-residual scheme B would need `hidden/TP % 32 == 0`: `4096/8 = 512`, satisfied. (The scheme
itself is P4's decision, not P0's.)

### 4.3 Gate-design consequence of the TP=8 equality

A single-card `(1,1)` run has `nkv = TP = 1` — **a head count the model never produces on the
deployment mesh** (at TP=8 the model emits 1 local KV head *per chip*, but the whole-model tensor is
8 heads wide). So a `(1,1)` KV gate (`G-KV`) is a valid test of the cache **primitive** and of the
`head_dim = 128` geometry, and is **not** a test of the model → cache path; P8's `G-KV-TP8` owns
that. Generalised: **a gate that passes on a mesh the deployment never uses can be testing a
configuration the model cannot produce** — every such gate block must say so.

## 5. `UNVERIFIED` rows

None. Every row above resolves to a `config.json` key, a shown derivation, a `path:line`, or a
command run on this box. (`07_RISKS.md` carries the risks that are *not* card rows.)

## 6. Provenance of this card

- `config.json` values: `cat $HF_MODEL/config.json`, 2026-09-04.
- Byte-identity: `md5sum $HF_MODEL/config.json models/tt_transformers/model_params/Llama-3.1-8B-Instruct/config.json`.
- Device facts: `ttnn.get_num_devices()`, `ttnn.get_arch_name()` on this box.
- Every `path:line` above is machine-checked by
  `models/demos/llama31_8b_d_p/scripts/verify_citations.py` (`G-CARD`).
