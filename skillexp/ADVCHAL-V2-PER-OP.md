# advchal-v2 — per-op detail, grouped by model and layer kind

Every op the advisor placed differently from the shipped graph, laid side by side across the cells of each
model: **what was tried, what was kept, what was rejected as slower, what was never measured, and what
could have been tried and was not.** Generated from each cell's `reconciliation_<kind>.json` and
`final.json`.

Cells of one model start from **different optimized decoders** — different arms, sometimes different
machines — so the *shipped* column varies. The *advised* column is the interesting one: the advisor is
deterministic and re-derives from DRAM-interleaved inputs, never seeing the shipped layout.

## Reading the cells

`shipped→advised cores, op µs, verdict`, where **KEPT** shipped, `rej` measured slower than the incumbent,
`below` was under the screening threshold, and `agree` means the row is `agrees_with_shipped`.

> **A caution that caught me out.** An `agrees_with_shipped` row can still show a *different* advised core
> count. `reconcile.py` treats a DRAM-sharded matmul as agreement when the **program-config family** matches,
> even if the grids differ — so `12→99, agree` means *both are DS*, not that 87 cores were left on the table.
> Only rows with a chain verdict are genuine disagreements. I initially misread these as un-screened headroom;
> they are not. The tool should record `agreed_on: grid | ds_family` and currently does not.


## phi-3.5

### `dense`

| op | phi B | phi FN | phi exp17 | phi onA |
|---|---|---|---|---|
| `paged_scaled_dot_product_attention_decode` | 110→DRAM, 192µs, DRAM-advice | 110→DRAM, 50µs, DRAM-advice | 32→DRAM, 186µs, DRAM-advice | 64→DRAM, 59µs, DRAM-advice |
| `linear` | 12→99, 93µs ×4, agree | 103→99, 104µs ×4, rej | 96→88, 183µs ×4, below | 12→88, 55µs ×5, agree |
| `nlp_create_qkv_heads_decode` | 32→22, 22µs, below | 32→22, 57µs, rej | 32→22, 32µs, rej | 32→22, 16µs, below |
| `rms_norm` | 8→11, 7µs ×2, below | 1→11, 45µs ×2, rej | 32→11, 8µs ×2, below | 16→11, 7µs ×2, below |
| `concat` | 110→22, 12µs ×2, **KEPT** | 110→22, 12µs ×2, **KEPT** | 110→22, 15µs ×2, rej | 110→22, 13µs ×2, **KEPT** |
| `multiply` | 96→22, 8µs ×5, **KEPT** | 110→77, 8µs ×5, rej | 110→77, 8µs ×5, below | 110→77, 10µs ×5, below |
| `slice_static` | 110→DRAM, 7µs ×6, **KEPT** | 110→DRAM, 7µs ×6, **KEPT** | 110→DRAM, 7µs ×6, rej | 110→DRAM, 7µs ×4, **KEPT** |
| `silu` | 110→77, 5µs, below | — | — | — |
| `nlp_concat_heads_decode` | 32→DRAM, 5µs, DRAM-advice | 32→DRAM, 5µs, DRAM-advice | 32→DRAM, 5µs, DRAM-advice | 32→DRAM, 5µs, DRAM-advice |
| `add` | 110→1, 3µs ×4, **KEPT** | 110→22, 3µs ×4, **KEPT** | 110→1, 3µs ×4, rej | 110→22, 4µs ×4, **KEPT** |
| `embedding` | 1→DRAM, 2µs ×2, below | 1→DRAM, 2µs ×2, rej | 1→DRAM, 2µs ×2, rej | 1→DRAM, 2µs ×2, **KEPT** |
| `neg` | 110→22, 2µs ×2, **KEPT** | 110→22, 2µs ×2, **KEPT** | 110→22, 2µs ×2, rej | 110→22, 2µs ×2, **KEPT** |
| `typecast` | 1→DRAM, 1µs, DRAM-advice | 1→1, 1µs, agree | — | 1→DRAM, 1µs, DRAM-advice |

**What each cell shipped**

- **phi B** → `improved` — {'advisor_rope_l1_chain': True, 'decode_core_grid': [8, 1], 'down_in0_block_w': 16}
- **phi FN** → `improved` — {'advisor_rope_l1': 'query_key', 'advisor_norm_cores': 0, 'advisor_sdpa_concat_l1': False}
    - kept and confirmed
- **phi exp17** → `no_change` — {'advisor_changes': [], 'decode_batch': 32, 'policy': 'frozen incumbent'}
    - no candidate satisfies strict non-overlap
- **phi onA** → `shipped` — {'use_advisor_decode_rope_l1': True, 'rope_storage': 'L1 interleaved', 'restore_geometry': '32-core exact rectangular height shard'}


## gemma-4-26B

### `full_attention`

| op | g26 B | g26 onA |
|---|---|---|
| `linear` | 107→99, 90µs ×4, below | 88→88, 172µs ×4, agree |
| `paged_scaled_dot_product_attention_decode` | 32→DRAM, 54µs, DRAM-advice | 32→DRAM, 51µs, DRAM-advice |
| `rms_norm` | 1→88, 44µs ×7, below | 1→88, 44µs ×7, below |
| `rotary_embedding` | 16→11, 31µs ×2, below | 16→11, 33µs ×2, below |
| `nlp_create_qkv_heads_decode` | 1→1, 26µs, agree | 1→1, 26µs, agree |
| `multiply` | 110→66, 2µs, below | 110→66, 3µs, below |
| `nlp_concat_heads_decode` | 16→DRAM, 2µs, DRAM-advice | 16→DRAM, 2µs, DRAM-advice |
| `add` | 110→88, 2µs, below | 110→88, 2µs, below |
| `gelu` | 110→66, 2µs, below | 110→66, 2µs, below |
| `slice_static` | 66→DRAM, 1µs ×2, below | 66→DRAM, 2µs ×2, below |

### `sliding_attention`

| op | g26 B | g26 onA |
|---|---|---|
| `linear` | 86→77, 123µs ×4, below | 86→77, 122µs ×4, below |
| `paged_scaled_dot_product_attention_decode` | 32→DRAM, 45µs, DRAM-advice | 32→DRAM, 44µs, DRAM-advice |
| `rms_norm` | 1→88, 44µs ×7, below | 1→88, 45µs ×7, below |
| `nlp_create_qkv_heads_decode` | 1→1, 28µs, agree | 1→1, 28µs, agree |
| `multiply` | 110→66, 3µs, below | 110→66, 3µs, below |
| `rotary_embedding` | 1→DRAM, 3µs ×2, DRAM-advice | 1→DRAM, 3µs ×2, DRAM-advice |
| `add` | 110→88, 2µs, below | 110→88, 2µs, below |
| `gelu` | 110→66, 2µs, below | 110→66, 2µs, below |
| `slice_static` | 66→DRAM, 1µs ×2, below | 66→DRAM, 2µs ×2, below |
| `nlp_concat_heads_decode` | 16→DRAM, 1µs, DRAM-advice | 16→DRAM, 1µs, DRAM-advice |

**What each cell shipped**

- **g26 B** → `SHIPPED` — {'layer_kind': 'sliding_attention', 'dram_sharded_roles': ['o_proj', 'packed_mlp_gate_up', 'mlp_down'], 'o_proj_in0_block_w': 1}
- **g26 onA** → `improved` — 88-core width-sharded hidden-width RMSNorm in decode; result converted back to the incumbent DRAM interface
    - boundary chains have zero advisor-attributable value, but reconciliation exposed material one-core RMSNorm operations advised onto 88 cores
    - 88-core RMSNorm beat every control repeat in both layer kinds, passed fresh confirmation and real-weight oracle, and was shipped


## north-mini

### `dense_full_attention`

| op | nm FN | nm onA |
|---|---|---|
| `linear` | 12→77, 28µs ×3, agree | 64→55, 55µs ×4, - |
| `rms_norm` | 8→22, 6µs, below | 1→22, 26µs, - |
| `nlp_create_qkv_heads_decode` | 1→1, 5µs, agree | 1→1, 25µs, agree |
| `paged_scaled_dot_product_attention_decode` | — | 64→DRAM, 9µs, DRAM-advice |
| `multiply` | 96→88, 2µs, below | 110→88, 3µs, - |
| `silu` | — | 110→88, 3µs, - |
| `add` | — | 110→55, 2µs ×2, - |
| `slice_static` | 64→DRAM, 2µs ×2, below | 96→DRAM, 2µs ×3, - |
| `rotary_embedding` | 1→DRAM, 2µs ×2, DRAM-advice | 1→DRAM, 2µs ×2, DRAM-advice |
| `nlp_concat_heads_decode` | — | 32→DRAM, 1µs, DRAM-advice |

### `dense_full_forced_rope`

| op | nm B |
|---|---|
| `linear` | 12→77, 31µs ×5, agree |
| `nlp_create_qkv_heads_decode` | 1→1, 14µs, agree |
| `paged_scaled_dot_product_attention_decode` | 110→DRAM, 9µs, DRAM-advice |
| `rms_norm` | 32→22, 6µs, below |
| `multiply` | 110→88, 3µs, rej |
| `silu` | 110→88, 3µs, rej |
| `rotary_embedding` | 1→DRAM, 2µs ×2, DRAM-advice |
| `add` | 110→55, 1µs ×2, rej |
| `nlp_concat_heads_decode` | 32→DRAM, 1µs, DRAM-advice |
| `slice_static` | 64→DRAM, 1µs, below |

### `full_attention_moe`

| op | nm FN |
|---|---|
| `linear` | 80→77, 44µs, - |
| `rms_norm` | 1→22, 26µs, - |
| `nlp_create_qkv_heads_decode` | 1→1, 25µs, agree |

### `full_attention_sparse_moe`

| op | nm onA |
|---|---|
| `linear` | 80→77, 44µs ×2, - |
| `rms_norm` | 1→22, 26µs, - |
| `nlp_create_qkv_heads_decode` | 1→1, 26µs, agree |
| `paged_scaled_dot_product_attention_decode` | 64→DRAM, 9µs, DRAM-advice |
| `slice_static` | 64→DRAM, 2µs, - |
| `nlp_concat_heads_decode` | 32→DRAM, 1µs, DRAM-advice |

### `full_no_rope_moe`

| op | nm B |
|---|---|
| `linear` | 12→77, 31µs ×2, agree |
| `nlp_create_qkv_heads_decode` | 1→1, 14µs, agree |
| `paged_scaled_dot_product_attention_decode` | 110→DRAM, 9µs, DRAM-advice |
| `rms_norm` | 32→22, 6µs, below |
| `add` | 110→55, 2µs, below |
| `slice_static` | 64→DRAM, 1µs, below |
| `nlp_concat_heads_decode` | 32→DRAM, 1µs, DRAM-advice |

### `sliding_attention_moe`

| op | nm FN |
|---|---|
| `linear` | 80→77, 44µs, - |
| `rms_norm` | 1→22, 26µs, - |
| `nlp_create_qkv_heads_decode` | 1→1, 26µs, agree |
| `rotary_embedding` | 1→DRAM, 2µs ×2, DRAM-advice |

### `sliding_attention_sparse_moe`

| op | nm onA |
|---|---|
| `linear` | 80→77, 44µs ×2, - |
| `rms_norm` | 1→22, 26µs, - |
| `nlp_create_qkv_heads_decode` | 1→1, 26µs, agree |
| `paged_scaled_dot_product_attention_decode` | 64→DRAM, 10µs, DRAM-advice |
| `rotary_embedding` | 1→DRAM, 2µs ×2, DRAM-advice |
| `slice_static` | 64→DRAM, 2µs, - |
| `nlp_concat_heads_decode` | 32→DRAM, 1µs, DRAM-advice |

### `sliding_rope_moe`

| op | nm B |
|---|---|
| `linear` | 12→77, 31µs ×2, agree |
| `nlp_create_qkv_heads_decode` | 1→1, 14µs, agree |
| `paged_scaled_dot_product_attention_decode` | 110→DRAM, 9µs, DRAM-advice |
| `rms_norm` | 32→22, 6µs, below |
| `add` | 110→55, 2µs, below |
| `rotary_embedding` | 1→DRAM, 2µs ×2, DRAM-advice |
| `nlp_concat_heads_decode` | 32→DRAM, 1µs, DRAM-advice |
| `slice_static` | 64→DRAM, 1µs, below |

**What each cell shipped**

- **nm B** → `no_change` — {'candidate': 'default'}
- **nm FN** → `improved` — 32-core L1-width-sharded RMSNorm for both sparse-MoE layer kinds
    - 32-core MoE RMSNorm won both layer kinds; advised 22 and above-advice 64 were slower
- **nm onA** → `no_change` — frozen incumbent candidate=default; no advisor candidate was measurable
    - all three layer kinds not_measurable; screening prohibited


## qwen-27B

### `full_attention`

| op | qwen B | qwen FN |
|---|---|---|
| `linear` | 12→88, 295µs ×5, rej | 12→77, 185µs ×5, agree |
| `multiply` | 110→99, 42µs ×6, below | 110→99, 23µs ×6, below |
| `paged_scaled_dot_product_attention_decode` | 64→DRAM, 20µs, DRAM-advice | 110→DRAM, 22µs, DRAM-advice |
| `nlp_create_qkv_heads_decode` | 32→22, 16µs, rej | 32→22, 16µs, below |
| `rms_norm` | 8→11, 10µs ×4, below | 8→11, 9µs ×4, below |
| `nlp_concat_heads_decode` | 24→DRAM, 9µs, DRAM-advice | 24→DRAM, 9µs, DRAM-advice |
| `sigmoid` | 110→88, 6µs, below | 110→88, 4µs, below |
| `slice_static` | 110→DRAM, 5µs ×10, rej | 110→DRAM, 3µs ×12, below |
| `concat` | 110→DRAM, 3µs ×4, DRAM-advice | 110→DRAM, 3µs ×5, DRAM-advice |
| `repeat` | 110→22, 2µs ×4, rej | 110→22, 3µs ×4, rej |
| `embedding` | 1→DRAM, 2µs ×4, DRAM-advice | 1→DRAM, 2µs ×4, DRAM-advice |
| `add` | 110→77, 2µs ×2, below | 110→77, 2µs ×4, below |
| `neg` | 110→22, 1µs ×2, below | — |
| `typecast` | 1→DRAM, 1µs, DRAM-advice | 1→DRAM, 1µs, DRAM-advice |

### `linear_attention`

| op | qwen B |
|---|---|
| `linear` | 12→99, 250µs ×3, agree |
| `multiply` | 12→99, 182µs, below |
| `add` | 12→77, 156µs ×2, below |
| `rms_norm` | 110→11, 79µs ×2, below |

**What each cell shipped**

- **qwen B** → `no_change` — None
- **qwen FN** → `improved` — {'full_attention_packed_qkv_output': 'one L1-interleaved conversion before four slices', 'linear_attention': 'unchanged'}


## gemma-4-12B

### `full_attention`

| op | gemma-4-12B |
|---|---|
| `linear` | 12→55, 213µs ×5, agree |
| `concatenate_heads` | 1→DRAM, 103µs, **KEPT** |
| `paged_scaled_dot_product_attention_decode` | 32→DRAM, 34µs, DRAM-advice |
| `rotary_embedding` | 32→11, 25µs ×2, **KEPT** |
| `nlp_create_qkv_heads_decode` | 32→22, 13µs, **KEPT** |
| `multiply` | 110→88, 11µs, **KEPT** |
| `rms_norm` | 16→64, 10µs ×7, below |
| `embedding` | 1→DRAM, 4µs ×2, DRAM-advice |
| `add` | 110→55, 1µs ×2, below |

### `sliding_attention`

| op | gemma-4-12B |
|---|---|
| `linear` | 12→55, 212µs ×5, agree |
| `paged_scaled_dot_product_attention_decode` | 110→DRAM, 97µs, DRAM-advice |
| `concatenate_heads` | 1→DRAM, 51µs, rej |
| `nlp_create_qkv_heads_decode` | 32→22, 15µs, **KEPT** |
| `rotary_embedding` | 32→22, 12µs ×2, **KEPT** |
| `multiply` | 110→88, 11µs, **KEPT** |
| `rms_norm` | 16→64, 6µs ×7, below |
| `embedding` | 1→DRAM, 3µs ×2, DRAM-advice |
| `add` | 110→55, 1µs ×2, below |

**What each cell shipped**

- **gemma-4-12B** → `improved` — {'keep_q_l1_extended_to_sharded_sdpa': True, 'keep_k_l1_across_per_head_norm': True, 'keep_v_l1_across_per_head_norm': True, 'mlp_direct_down_input_layout': True, 'o_chai


## llama-8B

### `dense`

| op | llama-8B |
|---|---|
| `linear` | 12→55, 115µs ×5, agree |
| `paged_scaled_dot_product_attention_decode` | 64→DRAM, 62µs, DRAM-advice |
| `nlp_create_qkv_heads_decode` | 32→22, 18µs, below |
| `multiply` | 110→88, 11µs, rej |
| `rms_norm` | 32→22, 7µs ×2, rej |
| `nlp_concat_heads_decode` | 32→DRAM, 6µs, DRAM-advice |
| `rotary_embedding_llama` | 32→22, 3µs ×2, below |
| `add` | 110→55, 2µs ×2, rej |

**What each cell shipped**

- **llama-8B** → `contribution_zero` — frozen incumbent; no advisor candidate passed non-overlap


## llama-1B

### `dense`

| op | llama-1B |
|---|---|
| `linear` | 12→77, 68µs ×5, agree |
| `paged_scaled_dot_product_attention_decode` | 64→DRAM, 19µs, DRAM-advice |
| `nlp_create_qkv_heads_decode` | 32→22, 15µs, below |
| `multiply` | 110→77, 8µs, rej |
| `rms_norm` | 32→22, 6µs ×2, rej |
| `nlp_concat_heads_decode` | 32→DRAM, 4µs, DRAM-advice |
| `rotary_embedding_llama` | 32→22, 3µs ×2, below |
| `add` | 110→55, 2µs ×2, rej |

**What each cell shipped**

- **llama-1B** → `no_change` — incumbent
    - hard_error: paged GQA SDPA explicitly rejects sharded output, which concat requires
    - rejected; slower than incumbent


## Rejected because it measured slower

Chains screened against the frozen incumbent and rejected on measurement — the bad-advice evidence.

| model | cell | kind | op | shipped→advised | op µs | chain measured ms | incumbent ms |
|---|---|---|---|---|---|---|---|
| qwen-27B | qwen B | `full_attention` | `linear` | 12→88 | 295.3 | 1.450132 | 1.449416 |
| phi-3.5 | phi exp17 | `dense` | `nlp_create_qkv_heads_decode` | 32→22 | 31.8 | 1.101395 | 1.100939 |
| qwen-27B | qwen B | `full_attention` | `nlp_create_qkv_heads_decode` | 32→22 | 16.1 | 1.450132 | 1.449416 |
| phi-3.5 | phi exp17 | `dense` | `concat` | 110→22 | 14.9 | 1.101395 | 1.100939 |
| phi-3.5 | phi exp17 | `dense` | `concat` | 110→22 | 12.4 | 1.101395 | 1.100939 |
| llama-8B | llama-8B | `dense` | `multiply` | 110→88 | 10.7 | 0.667582 | 0.665046 |
| llama-1B | llama-1B | `dense` | `multiply` | 110→77 | 8.2 | 0.420341 | 0.373080 |
| phi-3.5 | phi exp17 | `dense` | `multiply` | 96→1 | 7.8 | 1.101395 | 1.100939 |
| phi-3.5 | phi exp17 | `dense` | `multiply` | 96→22 | 7.5 | 1.101395 | 1.100939 |
| llama-8B | llama-8B | `dense` | `rms_norm` | 32→22 | 7.1 | 0.667582 | 0.665046 |
| phi-3.5 | phi exp17 | `dense` | `slice_static` | 110→None | 7.0 | 1.101395 | 1.100939 |
| phi-3.5 | phi exp17 | `dense` | `slice_static` | 110→None | 6.6 | 1.101395 | 1.100939 |
| llama-1B | llama-1B | `dense` | `rms_norm` | 32→22 | 6.0 | 0.420341 | 0.373080 |
| qwen-27B | qwen B | `full_attention` | `slice_static` | 110→None | 5.1 | 1.450132 | 1.449416 |
| qwen-27B | qwen B | `full_attention` | `slice_static` | 110→None | 4.5 | 1.450132 | 1.449416 |
| phi-3.5 | phi exp17 | `dense` | `multiply` | 110→22 | 3.1 | 1.101395 | 1.100939 |
| phi-3.5 | phi exp17 | `dense` | `multiply` | 110→22 | 3.1 | 1.101395 | 1.100939 |
| phi-3.5 | phi exp17 | `dense` | `add` | 110→1 | 3.0 | 1.101395 | 1.100939 |
| phi-3.5 | phi exp17 | `dense` | `add` | 110→22 | 3.0 | 1.101395 | 1.100939 |
| qwen-27B | qwen B | `full_attention` | `slice_static` | 110→None | 2.9 | 1.453619 | 1.449416 |
| qwen-27B | qwen FN | `full_attention` | `repeat` | 110→22 | 2.6 | 1.222463 | 1.208257 |
| phi-3.5 | phi exp17 | `dense` | `embedding` | 1→None | 2.5 | 1.101395 | 1.100939 |
| phi-3.5 | phi exp17 | `dense` | `embedding` | 1→None | 2.5 | 1.101395 | 1.100939 |
| phi-3.5 | phi exp17 | `dense` | `neg` | 110→22 | 2.2 | 1.101395 | 1.100939 |
| llama-1B | llama-1B | `dense` | `add` | 110→55 | 2.1 | 0.420341 | 0.373080 |
| qwen-27B | qwen B | `full_attention` | `slice_static` | 64→None | 1.9 | 1.453619 | 1.449416 |
| qwen-27B | qwen B | `full_attention` | `repeat` | 110→22 | 1.7 | 1.454418 | 1.449416 |
| qwen-27B | qwen B | `full_attention` | `repeat` | 110→22 | 1.7 | 1.454418 | 1.449416 |
| qwen-27B | qwen FN | `full_attention` | `repeat` | 110→22 | 1.7 | 1.222463 | 1.208257 |
| phi-3.5 | phi exp17 | `dense` | `slice_static` | 64→None | 1.6 | 1.101395 | 1.100939 |
| llama-8B | llama-8B | `dense` | `add` | 110→55 | 1.5 | 0.667582 | 0.665046 |
| llama-8B | llama-8B | `dense` | `add` | 110→55 | 1.5 | 0.667680 | 0.665046 |
| phi-3.5 | phi exp17 | `dense` | `neg` | 110→22 | 1.5 | 1.101395 | 1.100939 |
| llama-1B | llama-1B | `dense` | `add` | 110→55 | 1.4 | 0.420341 | 0.373080 |
| phi-3.5 | phi exp17 | `dense` | `slice_static` | 64→None | 1.3 | 1.101395 | 1.100939 |
| qwen-27B | qwen B | `full_attention` | `repeat` | 110→22 | 1.1 | 1.453619 | 1.449416 |
| qwen-27B | qwen FN | `full_attention` | `repeat` | 110→22 | 1.1 | 1.222652 | 1.208257 |
| qwen-27B | qwen B | `full_attention` | `repeat` | 110→22 | 1.1 | 1.453619 | 1.449416 |
| qwen-27B | qwen FN | `full_attention` | `repeat` | 110→22 | 1.1 | 1.222652 | 1.208257 |

## Tried but under the threshold — measured against a bar and dropped

| model | cell | kind | op | shipped→advised | op µs | % of window |
|---|---|---|---|---|---|---|
| phi-3.5 | phi exp17 | `dense` | `linear` | 96→88 | 183.0 | 18.014 % |
| qwen-27B | qwen B | `linear_attention` | `multiply` | 12→99 | 182.0 | 1.149 % |
| phi-3.5 | phi exp17 | `dense` | `linear` | 32→99 | 171.7 | 16.895 % |
| gemma-4-26B | g26 onA | `full_attention` | `linear` | 107→99 | 158.8 | 8.014 % |
| qwen-27B | qwen B | `linear_attention` | `add` | 12→77 | 156.2 | 0.986 % |
| qwen-27B | qwen B | `full_attention` | `linear` | 12→77 | 127.9 | 10.092 % |
| gemma-4-26B | g26 B | `sliding_attention` | `linear` | 86→77 | 123.3 | 10.218 % |
| gemma-4-26B | g26 onA | `sliding_attention` | `linear` | 86→77 | 121.9 | 6.816 % |
| gemma-4-26B | g26 B | `full_attention` | `linear` | 107→99 | 89.6 | 7.401 % |
| qwen-27B | qwen B | `linear_attention` | `rms_norm` | 110→11 | 79.4 | 0.502 % |
| gemma-4-26B | g26 onA | `sliding_attention` | `rms_norm` | 1→88 | 44.7 | 2.496 % |
| gemma-4-26B | g26 B | `sliding_attention` | `rms_norm` | 1→88 | 44.5 | 3.687 % |
| gemma-4-26B | g26 onA | `sliding_attention` | `rms_norm` | 1→88 | 44.4 | 2.48 % |
| gemma-4-26B | g26 B | `full_attention` | `rms_norm` | 1→88 | 44.3 | 3.656 % |
| gemma-4-26B | g26 onA | `full_attention` | `rms_norm` | 1→88 | 44.2 | 2.231 % |
| gemma-4-26B | g26 onA | `full_attention` | `rms_norm` | 1→88 | 44.1 | 2.227 % |
| gemma-4-26B | g26 B | `sliding_attention` | `rms_norm` | 1→88 | 44.1 | 3.655 % |
| gemma-4-26B | g26 onA | `full_attention` | `rms_norm` | 1→88 | 44.1 | 2.225 % |
| gemma-4-26B | g26 B | `full_attention` | `rms_norm` | 1→88 | 44.1 | 3.638 % |
| gemma-4-26B | g26 B | `sliding_attention` | `rms_norm` | 1→88 | 44.0 | 3.646 % |

## Could have been tried and was not: a starved op where the sweep stopped at the advised value

The skill is explicit — *never sweep only at or below an advised core count; always measure at least one
exactly-dividing grid*. These are ops on **≤2 cores** whose only screened candidate was the advised value:

| model | cell | kind | op | shipped | advised | op µs | % of window | verdict at the advised value |
|---|---|---|---|---|---|---|---|---|
| gemma-4-26B | g26 onA | `sliding_attention` | `rms_norm` | **1** | 88 | **44.7** | 2.496 % | below_threshold |
| phi-3.5 | phi FN | `dense` | `rms_norm` | **1** | 11 | **44.5** | 6.138 % | rejected |
| gemma-4-26B | g26 B | `sliding_attention` | `rms_norm` | **1** | 88 | **44.5** | 3.687 % | below_threshold |
| gemma-4-26B | g26 onA | `sliding_attention` | `rms_norm` | **1** | 88 | **44.4** | 2.48 % | below_threshold |
| phi-3.5 | phi FN | `dense` | `rms_norm` | **1** | 11 | **44.3** | 6.107 % | rejected |
| gemma-4-26B | g26 B | `full_attention` | `rms_norm` | **1** | 88 | **44.3** | 3.656 % | below_threshold |
| gemma-4-26B | g26 onA | `full_attention` | `rms_norm` | **1** | 88 | **44.2** | 2.231 % | below_threshold |
| gemma-4-26B | g26 onA | `full_attention` | `rms_norm` | **1** | 88 | **44.1** | 2.227 % | below_threshold |
| gemma-4-26B | g26 B | `sliding_attention` | `rms_norm` | **1** | 88 | **44.1** | 3.655 % | below_threshold |
| gemma-4-26B | g26 onA | `full_attention` | `rms_norm` | **1** | 88 | **44.1** | 2.225 % | below_threshold |
| gemma-4-26B | g26 B | `full_attention` | `rms_norm` | **1** | 88 | **44.1** | 3.638 % | below_threshold |
| gemma-4-26B | g26 B | `sliding_attention` | `rms_norm` | **1** | 88 | **44.0** | 3.646 % | below_threshold |
| gemma-4-26B | g26 B | `sliding_attention` | `rms_norm` | **1** | 88 | **44.0** | 3.646 % | below_threshold |
| gemma-4-26B | g26 onA | `sliding_attention` | `rms_norm` | **1** | 88 | **44.0** | 2.458 % | below_threshold |
| gemma-4-26B | g26 onA | `full_attention` | `rms_norm` | **1** | 88 | **44.0** | 2.22 % | below_threshold |
| gemma-4-26B | g26 B | `full_attention` | `rms_norm` | **1** | 88 | **43.9** | 3.628 % | below_threshold |
| gemma-4-26B | g26 B | `full_attention` | `rms_norm` | **1** | 88 | **43.9** | 3.627 % | below_threshold |
| gemma-4-26B | g26 onA | `sliding_attention` | `rms_norm` | **1** | 88 | **43.9** | 2.452 % | below_threshold |
| north-mini | nm FN | `sliding_attention_moe` | `rms_norm` | **1** | 22 | **26.1** | 4.998 % | not_measurable |
| north-mini | nm onA | `full_attention_sparse_moe` | `rms_norm` | **1** | 22 | **26.1** | 3.168 % | not_measurable |
| north-mini | nm FN | `full_attention_moe` | `rms_norm` | **1** | 22 | **26.1** | 5.239 % | not_measurable |
| north-mini | nm onA | `sliding_attention_sparse_moe` | `rms_norm` | **1** | 22 | **26.0** | 3.145 % | not_measurable |
| north-mini | nm onA | `dense_full_attention` | `rms_norm` | **1** | 22 | **26.0** | 9.478 % | not_measurable |

**The 1-core RMSNorm is the corpus's highest-yield op class, and it recurs across models.** Three cells met
one; two swept above the advised value and won big, one stopped at the advised value and abandoned it:

| cell | shipped | advised | what it did | result |
|---|---|---|---|---|
| north-mini FN | 1 core, 26.1 µs | 22 | swept 22 / **32** / 64 — *"advised 22 and above-advice 64 were slower"* | **−10.23 %** |
| gemma-4-26B onA | 1 core, ~44 µs ×2 | 88 | took **88** — *"reconciliation exposed material one-core RMSNorm operations advised onto 88 cores"* | **−12.98 %** |
| phi arm FN | 1 core, 44.3 + 44.5 µs = **12.25 % of window** | 11 | screened **11 only**, measured slower, shipped `advisor_norm_cores: 0` | −4.91 % from RoPE instead |

phi arm FN is the clearest missed opportunity in the experiment: its largest single disagreement, on the one op
class that paid twice elsewhere, dropped after a sweep bounded above by the advised value — the exact failure
the skill warns about. Its sibling `phi arm onA` even shipped `restore_geometry: 32-core exact rect`, so the
grid that works on this model was known within the same corpus.

**Why the cells differ, and where they agree.** The advisor's advice is near-identical across a model's cells —
for phi it advised `rms_norm`→11, `nlp_create_qkv_heads_decode`→22, `neg`/`concat`/`add`→22 in all four,
regardless of incumbent. What varies is the *shipped* side: the same `rms_norm` arrives on 1, 8, 16 or 32 cores
depending on which stage-02 arm produced the decoder. So the cells diverge because their **arms** diverge, not
because the advice did — and that is why a cell's outcome tracks how much its arm left behind, not how good the
advice was.

