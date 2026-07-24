# Environment variables for `test_prefill_transformer.py`

This documents every environment variable reachable from
`models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py` — the test file
itself, its `conftest.py`, and the helper modules it imports.

## How env vars get resolved

The test runs the **`DSV3` variant** by default (from `tests/model_variants.py`).
Several env var *names* are indirected through the variant definition — that's why
the DSV3 run uses `DEEPSEEK_V3_HF_MODEL`, `TT_DS_PREFILL_TTNN_CACHE`, and
`TT_DS_PREFILL_HOST_REF_CACHE`. For the Kimi variant the same roles map to
different names (see the Kimi section).

The three reference sources for PCC checks are tried in priority order:

1. **On-disk golden trace** (`DEEPSEEK_V3_TRACE_DIR`)
2. **Reference cache** (`TT_DS_PREFILL_HOST_REF_CACHE`)
3. **Live HF model computation**

---

## Tier 1 — paths/data that matter to run the test

### `DEEPSEEK_V3_HF_MODEL`  (`variant.env_var`)
Path to the HF model dir (weights + `config.json` + `*.index.json` + tokenizer).
Drives the `model_path`, `hf_config`, `state_dict`, and `tokenizer` fixtures.
If unset/invalid it falls back to `default_local_path` → `shared_path`
(`/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528`) → **auto-download from HuggingFace**.

### `TT_DS_PREFILL_TTNN_CACHE`  (`variant.ttnn_cache_env`)
Directory for cached TTNN weight tensors (`.tensorbin`). First run dumps converted
weights; later runs load directly, skipping torch conversion. Subdir is
`{variant}_{bh|wh}_{Ndev}`. Fallback: `<model_path>/tensor_cache_...`.

### `TT_DS_PREFILL_HOST_REF_CACHE`  (`variant.ref_cache_env`)
Directory of cached **PyTorch reference outputs** (`.pt`, keyed by
weight/input/isl/layers/experts/padding). Used as reference source #2 for PCC checks
so it doesn't recompute the HF forward each run.
Fallback: `/tmp/deepseek_v3_d_p_transformer_ref_cache`.

### `DEEPSEEK_V3_TRACE_DIR`
Base dir for pre-computed **golden debug traces** (safetensors) — reference source #1
(highest priority, via `find_trace_dir`/`TRACE_LOOKUP`).
Default `/mnt/MLPerf/deepseek-prefill-cache`. Only used for pretrained + 256-expert
configs; the trace is chosen by `(input_source, padding_side)` and an isl that is either
an exact match or the smallest available trace `>= isl_total` (sliced down — see
"Arbitrary ISL via trace slicing" below).

#### How the trace directory is located (it is a fixed lookup, not a search)

`DEEPSEEK_V3_TRACE_DIR` is only the **base** directory. The code never scans/globs
inside it to "find" a matching trace — the subdirectory names are hardcoded, and a trace
is chosen by dictionary lookup with an isl fallback (exact key match, else the smallest
registered trace whose key length is `>= isl_total`, then sliced).

**Step 1 — the base path**. The env var replaces the leading path; the subdirectory
names are baked in:

```python
TRACE_DIR_BASE = Path(os.getenv("DEEPSEEK_V3_TRACE_DIR", "/mnt/MLPerf/deepseek-prefill-cache")).resolve()
ILLIAD_1024_TRACE     = TRACE_DIR_BASE / "illiad_prefill_fa2"
ILLIAD_25024_TRACE    = TRACE_DIR_BASE / "illiad_prefill_fa2_25024"
ABC_1K_PAD_RIGHT_1024 = TRACE_DIR_BASE / "ABC_1k_prefill_padd_right_1024"
ABC_1K_PAD_LEFT_1024  = TRACE_DIR_BASE / "ABC_1k_prefill_padd_left_1024"
LONGBOOK_QA_ENG_25600 = TRACE_DIR_BASE / "longbook_qa_eng_prefill_25600_nopad"
LONGBOOK_QA_ENG_5120  = TRACE_DIR_BASE / "longbook_qa_eng_prefill_5120_nopad"
LONGBOOK_QA_ENG_56320 = TRACE_DIR_BASE / "longbook_qa_eng_prefill_56320_nopad"
```

**Step 2 — the lookup table**. The key is the tuple
`(input_source, isl_total, padding_side)`, where `isl_total` is the trace's native
(generated) sequence length:

```python
TRACE_LOOKUP: dict[tuple[str, int, str], Path] = {
    ("json_prompts",    1024,  "right"): ILLIAD_1024_TRACE,
    ("json_prompts",    25600, "right"): ILLIAD_25024_TRACE,
    ("abc_1k",          1024,  "right"): ABC_1K_PAD_RIGHT_1024,
    ("abc_1k",          1024,  "left"):  ABC_1K_PAD_LEFT_1024,
    ("longbook_qa_eng", 5120,  "right"): LONGBOOK_QA_ENG_5120,
    ("longbook_qa_eng", 25600, "right"): LONGBOOK_QA_ENG_25600,
    ("longbook_qa_eng", 56320, "right"): LONGBOOK_QA_ENG_56320,
}
```

**Step 3 — `find_trace_dir`**. Requires `use_pretrained` is True **and**
`n_routed_experts == 256` (traces were captured from the full pretrained DeepSeek-R1,
so they are invalid for random-weight or reduced-expert runs), and the directory must be
"ready" (physically exists **and** contains `metadata.json`). It returns
`(trace_dir, lookup_isl)` or `None`, resolving in two stages:

1. **Exact match** — `TRACE_LOOKUP.get((input_source, isl_total, padding_side))`; e.g.
   `("longbook_qa_eng", 5120, "right")`. Normally no slicing.
2. **Arbitrary-ISL fallback** — if there is no exact match, pick the **smallest ready
   trace with the same `(input_source, padding_side)` whose key length is `>= isl_total`**.
   The caller then slices the first `isl_total` positions. So asking for `longbook_qa_eng`
   at `isl=25000` with only a `56320` trace present resolves to the `56320` trace, sliced
   to `25000`.

### Arbitrary ISL via trace slicing

`test_prefill_transformer.py` supports **any requested `isl_total`** as long as a trace
with the same `(input_source, padding_side)` and a **native isl `>= isl_total`** exists.
When the matched trace is longer than requested, `slice_debug_trace()` truncates it to the
first `isl_total` positions.

This is **exact** for causal, `*_nopad` prefill traces: a transformer's per-layer decoder
output and KV-cache entry at position `i` depend only on positions `0..i` (causal
attention + absolute-position RoPE), so they are identical whether the full sequence or
just its first `isl_total` tokens are prefilled. The per-layer hidden-state PCC and the
KVPE PCC checks therefore remain valid on the sliced reference.

**Caveat:** the trace's stored `logits` / `next_token_id` belong to the *full* sequence's
final position, so they are meaningless for a shorter prefill. For a sliced trace,
`slice_debug_trace()` drops `logits` (→ `None`) and the test **skips the logits PCC and
first-token-match checks** (per-layer + KVPE PCC still run). An exact-length match keeps
all checks.

Requesting an `isl_total` **larger** than every available trace still yields no trace
(the test then falls back to reference cache / live HF compute).

**Step 4 — reading the trace** (`load_debug_trace`, line 941+). Once a directory passes,
the loader reads by **fixed filenames/keys**, not by searching:

- `metadata.json` → `token_ids` and `n_layers`.
- Hidden states: per-layer files `hidden_states/layer_{i}.safetensors`
  (key `decoder_output_layer_{i}`) **or** a flat `hidden_states.safetensors` — per-layer
  is used if the `hidden_states/` subdir exists.
- KV cache: `kv_cache/layer_{i}.safetensors` or flat `kv_cache.safetensors`, preferring
  the key `kv_post_transform_layer_{i}` and falling back to `compressed_kv_layer_{i}`
  (with a warning that PCC will be unreliable).
- Optional `logits.safetensors`.

**What this means for you:**

- To point at a custom trace location, set `DEEPSEEK_V3_TRACE_DIR` to a base dir that
  **contains the exact hardcoded subdirectory names** above — e.g.
  `$DEEPSEEK_V3_TRACE_DIR/ABC_1k_prefill_padd_right_1024/metadata.json`.
- If no trace matches your `(input_source, padding_side)` with a native isl `>= isl_total`,
  **no trace is used regardless of the env var** — the test falls back to the reference
  cache / live HF compute.
- This is a different mechanism from the variant's `prefill_trace_default` field
  (`/mnt/models/.../golden/...`), which is consumed by the *chunked* transformer test and
  the standalone runners via `PREFILL_TRACE_DIR` — not by this `find_trace_dir` path.

### `HF_HOME`
HuggingFace cache dir used for the auto-download fallback (weights, config, tokenizer).
Default `~/.cache/huggingface`.

---

## Tier 2 — optional behavior toggles

### `TT_DS_PREFILL_INFINITEBENCH_CACHE`
Where InfiniteBench prompt subsets are cached (only relevant when `input_source` is
`passkey`/`kv_retrieval`/`longdialogue_qa_eng`/`longbook_qa_eng`). Falls back under `HF_HOME`.

### `TT_DS_PREFILL_DEBUG_TOKEN_COUNT`
`1/true/yes` enables a token-count debug path inside the MoE block (`tt_moe.py`).

### `MESH_DEVICE`
If `TG`/`GALAXY`, the code treats the device as a Galaxy (`is_tg` in `perf_utils.py`).

### `PCC_SUMMARY_DIR`
Where PCC summary files/plots are written. Default `/tmp/pcc_summaries_<user>`.

### `<NAME>_OUTPUT_DIR`
Per-stage dump dir for `save_intermediate_output` — e.g. `NORM_OUTPUT_DIR`,
`LM_HEAD_OUTPUT_DIR`. Default `/tmp/{name}_outputs`.

### `DEEPSEEK_V3_MLA_REF_CACHE`  (`variant.mla_ref_cache_env`)
MLA-layer reference cache dir. Primarily exercised by `test_mla.py`, defined on the variant.

---

## Tier 3 — CI-only (no effect on a normal local run)

### `GITHUB_ACTIONS`
When **unset** and a trace dir exists, the test does an extra local-only branch
(line 826). Set by CI.

### `EXPECT_NUM_TESTS`
Optional guardrail: warns (never fails) if collected test count ≠ this. Inert unless set.

### `GITHUB_STEP_SUMMARY`
Path where the above warning is also appended (CI job summary).

---

## Kimi variant equivalents (only if you run the `kimi_k2_6` variant)

Same roles as the DSV3 names above:

- `DEEPSEEK_V3_HF_MODEL` → `KIMI_K2_6_HF_MODEL`
- `TT_DS_PREFILL_TTNN_CACHE` → `TT_KIMI_PREFILL_TTNN_CACHE`
- `TT_DS_PREFILL_HOST_REF_CACHE` → `TT_KIMI_PREFILL_HOST_REF_CACHE`
- `DEEPSEEK_V3_MLA_REF_CACHE` → `KIMI_MLA_REF_CACHE`

---

## NOT used by this test

A large `PREFILL_*` family exists in this module
(`PREFILL_HF_MODEL`, `PREFILL_TTNN_CACHE`, `PREFILL_NUM_LAYERS`, `PREFILL_SP`/`PREFILL_TP`,
`PREFILL_STANDALONE*`, `PREFILL_CHUNK_SIZE`, `PREFILL_MIGRATE_*`, `PREFILL_STRESS_*`,
`PREFILL_H2D_*`, `MIGRATION_DONE_FILE`, `DEEPSEEK_PREFILL_TRACE_DIR/PT`, etc.).

These belong to the standalone multi-process prefill runner (`tt/runners/*`) and the
chunked/migration/stress tests — they are **not** read by `test_prefill_transformer.py`,
so setting them here has no effect. Note the runner uses `PREFILL_TTNN_CACHE` /
`PREFILL_HF_MODEL` while this test uses the `TT_DS_PREFILL_*` / `DEEPSEEK_V3_HF_MODEL`
names — don't confuse the two families.

---

## Minimal working example (DSV3)

```bash
TT_DS_PREFILL_TTNN_CACHE=/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure \
DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528 \
TT_DS_PREFILL_HOST_REF_CACHE=/mnt/models/deepseek-prefill-cache/golden/ \
pytest models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py
```

The effective working set for this test is: `DEEPSEEK_V3_HF_MODEL`,
`TT_DS_PREFILL_TTNN_CACHE`, `TT_DS_PREFILL_HOST_REF_CACHE` (+ optionally
`DEEPSEEK_V3_TRACE_DIR` and `HF_HOME`).
