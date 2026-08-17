# Functional Decoder — `Qwen/Qwen3.6-35B-A3B`

Stage 1/11 of the repo-local TTNN autoport pipeline. Deliverable:
`models/autoports/qwen_qwen3_6_35b_a3b/tt/functional_decoder.py` — a functionally complete
TTNN implementation of the HF `Qwen3_5MoeDecoderLayer`, covering **both** decoder layer
kinds, validated against HF on a single 1x1 Blackhole mesh.

* Hardware: 1 x Blackhole `p300c` chip (grid 11x10, 31.5 GiB usable DRAM), 1x1 mesh.
* HF reference: `transformers` 5.10.2 `Qwen3_5MoeDecoderLayer`, **float32**.
* Acceptance bar: **PCC >= 0.995** (skill default; no model-specific exception needed).
* Worst PCC in the main suite: **0.9999450** (§3.1). Worst anywhere including the 262144-token
  advertised-context cases: **0.9985674** (§3.8 root-causes that one).
* Context: full HF-advertised **262144**, no capability reduction (§4).
* Warmed prefill and **traced** warmed decode measured with `tt-perf-report` (§5).
* Watcher-clean, determinism and runtime-fallback evidence: §3.6, §3.7, §3.9.

---

## 1. What the layer is

`config.json -> text_config` gives 40 decoder layers whose `layer_types` repeat
`[linear, linear, linear, full]` (`full_attention_interval = 4`):

| kind | count | layer indices | token mixer | per-sequence state |
|---|---|---|---|---|
| `linear_attention` | 30 | `(i+1) % 4 != 0` | `Qwen3_5MoeGatedDeltaNet` | conv state (3 taps x 16384) + recurrent state `[32,128,128]` |
| `full_attention` | 10 | 3, 7, ... 39 | `Qwen3_5MoeAttention` | paged K/V cache |

Everything after the mixer is shared and identical in both kinds: RMSNorm (`1 + w`) around a
256-expert / top-8 MoE with a sigmoid-gated shared expert. `tests/test_functional_decoder.py::test_layer_kinds_cover_the_whole_model`
pins that layer 0 and layer 3 are the only two distinct kinds, so testing those two indices
is complete coverage.

Full architecture notes (projection layouts, the head-interleaved `q_proj`/gate split, partial
RoPE, gate/decay math, MoE weight layout) are in `work_log.md` §2.

## 2. Public contract

```python
FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, **cfg)
    # state_dict keys = HF layer names with "model.language_model.layers.<i>." stripped.
    # All torch work (transposes, q/gate de-interleave, q/k head duplication, conv-tap split,
    # `1 + w` norm folding, RoPE tables, cache allocation) happens here.

prefill_forward(x, *, user_id=0, page_table=None, start_pos=0) -> ttnn.Tensor
    x           [1, 1, seq_len, 2048] TILE/DRAM. ANY seq_len >= 1 with
                start_pos + seq_len <= supported_context. Non-aligned lengths are padded,
                masked and sliced back internally.
    user_id     sequence slot: page_table row (full attention) / state slot (linear).
    page_table  int32 [max_batch, supported_context // 64] ROW_MAJOR. Required for
                full_attention, ignored for linear_attention.
    start_pos   absolute position of x[..., 0, :]; multiple of PREFILL_ALIGN (128). Lets a
                caller stream a long prompt across several calls (KV cache, conv state and
                recurrent state all carry over).
    returns     [1, 1, seq_len, 2048]

decode_forward(x, *, current_pos=None, page_table=None) -> ttnn.Tensor
    x            [1, 1, max_batch, 2048] TILE/DRAM, one token per slot.
    current_pos  int32 [max_batch] DEVICE tensor. Paged-cache write index, SDPA cur_pos and
                 (via an on-device typecast) the RoPE table lookup. -1 marks a slot inactive:
                 its attention is skipped and its paged K/V is left untouched. Required for
                 full_attention; linear_attention ignores it (its recurrence carries no
                 position), so it may be omitted for those layers.
    page_table   same tensor as prefill.
    returns      [1, 1, max_batch, 2048]
```

**One sequence per prefill call.** That matches the per-request prefill vLLM and the
downstream full-model stage issue. Batch >1 prefill = one call per `user_id`; each lands in
its own cache slot / page-table row and stays independent
(`test_prefill_per_user_slots`, 32 slots). Decode is fully batched.

**Decode batch is fixed at `max_batch_size`.** The linear-attention conv and recurrent state
buffers are updated whole-tensor and in place, which is what lets one captured trace be
replayed at any position. Partial batches use `current_pos = -1` on unused full-attention
slots.

### Runtime knobs (`DecoderConfig`)

| field | default | note |
|---|---|---|
| `supported_context` | 262144 | HF-advertised; see `doc/context_contract.json` |
| `max_batch_size` | 1 | decode batch, and number of cache/state slots |
| `block_size` | 64 | paged KV block |
| `prefill_chunk_size` | 2048 | internal prefill chunking (multiple of 128) |
| `delta_chunk_size` | 64 | gated-delta-rule chunk (matches HF's default) |
| `moe_prefill_chunk_tokens` | 512 | bounds the dense-over-experts MoE intermediates |
| `activation_dtype` / `weight_dtype` / `kv_cache_dtype` | `bfloat16` | |
| `delta_dtype` | `float32` | HF pins the SSM state to fp32 (`mamba_ssm_dtype`) |

## 3. Correctness evidence

Commands (all from `$TT_METAL_HOME`):

```bash
# CPU-only algebra unit tests (no device, no checkpoint)
pytest models/autoports/qwen_qwen3_6_35b_a3b/tests/test_reference_math.py

# main correctness suite (device, synthetic weights from real-checkpoint stats)
pytest models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py

# advertised-context evidence (slow)
pytest models/autoports/qwen_qwen3_6_35b_a3b/tests/test_long_context.py -m slow
```

Logs: `logs/test_suite_main.log` (the CPU algebra tests + the whole device suite, including the
real-weight cases: **94 passed**), `logs/long_*.log` (one per advertised-context case),
`logs/diag_*.txt` (the §3.8 diagnostics), `watcher/pytest.log`.
Machine-readable PCC rows: `pcc.jsonl` (271), `pcc_real_weights.jsonl` (6),
`long_context.jsonl` (7). Every number quoted below is re-derivable from those files.

### 3.1 HF-vs-TTNN PCC, synthetic weights (real shapes)

Worst case per test family, re-derived from `pcc.jsonl` (271 rows):

| family | n | worst PCC | worst case |
|---|---|---|---|
| `prefill[linear]` | 13 | 0.9999737 | seq=4096 |
| `prefill[full]` | 13 | 0.9999792 | seq=1 |
| `prefill-cont[linear]` | 2 | 0.9999812 | start=512 |
| `prefill-cont[full]` | 2 | 0.9999903 | start=0 |
| `prefill-slot[linear]` (32 slots) | 32 | 0.9999901 | user=26 |
| `prefill-slot[full]` (32 slots) | 32 | 0.9999907 | user=10 |
| `decode[linear]` | 81 | 0.9999857 | pos=129, batch=1 |
| `decode[full]` | 81 | 0.9999870 | pos=130, batch=1 |
| `decode-ragged` (per-slot positions) | 4 | 0.9999930 | user=0, pos=128 |
| `decode-active-slot` (with `current_pos=-1` peers) | 2 | 0.9999927 | user=0 |
| `decode-seeded-state` (random DeltaNet state) | 1 | 0.9999720 | |
| `traced-decode[linear]` | 2 | 0.9999891 | pos=257, batch=8 |
| `traced-decode[full]` | 2 | 0.9999936 | pos=256, batch=8 |
| `paged-kv` cache contents | 2 | 0.9999835 | keys |
| `linear` conv/recurrent state | 2 | 0.9999450 | recurrent_state |

**Overall minimum: 0.9999450** (`linear recurrent_state`) vs a 0.995 bar — a margin of about
3 orders of magnitude in `1 - PCC`.

### 3.2 Real-weight PCC (`pcc_real_weights.jsonl`)

| case | PCC | rel-RMS |
|---|---|---|
| `prefill[linear] seq=1024` | 0.9999435 | 1.06% |
| `decode[linear] pos=512/513, batch=2` | 0.9999834 / 0.9999946 | 0.58% / 0.34% |
| `prefill[full] seq=1024` | 0.9999796 | 0.64% |
| `decode[full] pos=512/513, batch=2` | 0.9999888 / 0.9999902 | 0.48% / 0.45% |

Context for the rel-RMS: **HF's own bf16-vs-fp32 divergence on the same layers and input is
0.32% rel-RMS (PCC 0.999995)**, measured with real weights for both layer kinds
(`work_log.md` §5.2). The TTNN bf16 path is 2-3.3x that, which is the expected cost of bf16
activations plus bf16 sparse matmuls; PCC stays ~2 orders of magnitude inside the bar. Both
layer kinds load the **real** HF layer state dict through `from_state_dict`, so these runs also
pin the state-dict key and shape contract.

### 3.3 Sequence-length coverage

`test_prefill_pcc_sequence_lengths` covers, for **both** kinds:
1, 32 (one tile), 33 (past a tile), 64 (one paged block / one delta chunk), 65 (past a
block), 128 (`PREFILL_ALIGN`), 129 (past it), 1024, 1025, 2048 (internal prefill chunk),
2049 (forces chunk continuation), 3000 (divides nothing in play), 4096.

`test_decode_after_non_aligned_prefill` covers decode continuing from prefill lengths
65 / 129 / 320 / 1000 — the case that caught a real bug (see §7, item 2).

`test_long_context.py` runs the largest feasible length, `262143` — deliberately
non-aligned, so the longest test also exercises the pad/mask/slice path.

### 3.4 Paged cache and state behaviour

* Page tables are always a **random permutation** of the physical block pool
  (`make_page_table`), so an implementation that assumed identity mapping fails.
* `test_paged_kv_cache_contents_match_hf` compares the K/V actually written through the page
  table against HF's cache, and asserts the other three slots are untouched.
* `test_page_table_permutation_is_respected` re-runs decode with a permuted page table and
  asserts the output changes — a guard against silently treating the cache as contiguous.
* `test_linear_state_matches_hf` compares the conv and recurrent state against HF's, in HF's
  own layout (via the documented `tt_conv_state_to_hf` translation).
* `test_decode_from_seeded_random_linear_state` decodes from a random, non-prefill-derived
  state so every term of the recurrence contributes.
* `test_decode_ragged_current_positions` prefills four slots to 128/256/384/640 and decodes
  all four in one batched call, proving per-slot `current_pos` (KV write index, SDPA context
  length and RoPE lookup) is honoured.
* `test_decode_skips_inactive_slots_with_negative_position` decodes a batch of four with
  `current_pos = -1` on two slots: the active slots still match HF, and the inactive slots'
  paged K/V is bit-unchanged, so the cache update does not scribble through a negative index.

### 3.5 Traced decode

`test_traced_decode_pcc` measures PCC **from trace replay output**, not from an uncaptured
forward: prefill → snapshot state → capture (which runs a warmup forward and perturbs the
state) → rewind → replay → compare. Two positions are replayed through the **same** captured
trace, proving `current_pos` is read from device memory rather than baked in.

`test_traced_decode_matches_eager` asserts trace replay is **bit-identical** to eager decode
from the same state (max abs diff exactly 0.0) for both layer kinds.

Two trace-safety properties this establishes that are not obvious:

* The MoE runs **inside** the trace with `nnz=None`, i.e. `sparse_matmul` resolves the
  non-zero count from the sparsity tensor at *replay* time. The two replayed steps use
  different tokens (different seeds), so they route to different experts and therefore
  produce different sparsity patterns — and both match HF. A capture-time-baked count would
  have deadlocked or mis-computed on the second replay.
* The linear-attention conv taps are shifted with in-place `ttnn.copy` and the recurrent state
  is updated with `output_tensor=state`, so all persistent buffers keep their addresses across
  replays. That is why `decode_forward` requires `batch == max_batch_size`.

### 3.6 Determinism

`test_prefill_determinism` / `test_decode_determinism`: three repeats of the same input from
the same state produce **bit-identical** output (max abs diff exactly 0.0), both kinds.

### 3.7 Runtime fallback audit

`test_no_runtime_host_fallback` is a static audit of the **whole runtime call graph** (26
methods reachable from `prefill_forward` / `decode_forward`, plus the module-level helpers)
for `torch`, `ttnn.from_torch`, `ttnn.to_torch`, `.cpu()`, `.item()`, `.tolist()`,
`ttnn.to_device`, `ttnn.from_device`, `copy_host_to_device_tensor`. Docstrings and comments
are stripped first so prose about torch does not trip it. The test also asserts that the
audited method list actually covers every `self._*` helper the entry points call — that
self-check caught two newly added helpers during development.

`test_no_host_ops_during_forward` is the dynamic counterpart: it monkeypatches
`ttnn.from_torch` / `to_torch` / `copy_host_to_device_tensor` / `to_device` / `from_device`
to raise, then runs a real prefill and a real decode for both kinds.

Setup-time conversion (`from_state_dict`, `_prepare_weights`, `_build_rope_tables`,
`_init_state`, `_to_device`) is exempt: that *is* the weight-loading boundary. Test-harness
input construction and PCC comparison are the other explicit boundaries.

### 3.8 Investigated anomaly: full-attention decode at 262144-token context

`test_longest_decode_context[full]` reports PCC **0.9986** at position 262143 — passing, but the
only number in the stage materially below the ~0.99999 everywhere else. Root-caused, not
annotated. Two diagnostics, both kept in `tests/` with their output in `logs/`:

**Step 1 — localise it (`diag_long_decode.py` -> `logs/diag_long_decode.txt`).** Sweep the decode
position over one cache and isolate the attention branch (the only position-dependent part of the
layer) by driving the TTNN and HF mixers directly:

| context | layer PCC | TTNN attn vs fp32 HF | TTNN attn vs **bf16-operand HF control** | **control vs fp32 HF** | attn RMS |
|---|---|---|---|---|---|
| 1024 | 0.9999955 | 0.9996975 | 0.9996994 | 0.9999980 | 0.03327 |
| 8192 | 0.9999954 | 0.9990401 | 0.9990331 | 0.9999977 | 0.01292 |
| 32768 | 0.9999790 | 0.9924129 | 0.9924141 | 0.9999973 | 0.00639 |
| 131072 | 0.9995946 | 0.9187724 | 0.9187617 | 0.9999973 | 0.00306 |
| 262143 | 0.9975827 | 0.7686928 | 0.7686691 | 0.9999976 | 0.00221 |

The last column is the decisive control: HF's own attention math with q/k/v rounded to bf16 and
exact accumulation matches fp32 to 2.4e-6 at **every** context. So **operand precision is not the
cause** — the device diverges from an exact bf16 reference just as much as from fp32.

**Step 2 — reproduce it with no model code (`diag_sdpa_decode.py` -> `logs/diag_sdpa_decode.txt`).**
Drive `ttnn.transformer.paged_scaled_dot_product_attention_decode` alone (no projections, RoPE,
gate, o_proj or MoE) with random K/V over a paged cache, sweeping context x grid x `k_chunk_size`:

```
 grid  kchunk        1       32       64      128      257     1024     4096    32768   131072   262143
 8x8  dynamic   1.0000   0.9998   0.9998   0.9998   0.9998   0.9998   0.9995   0.9875   0.9170   0.7664
 1x1      128   1.0000   0.9998   0.9998   0.9998   0.9998   0.9998   0.9998   0.9980   0.9839   0.9704
 2x1      128   1.0000   0.9998   0.9998   0.9998   0.9998   0.9998   0.9998   0.9980   0.9839   0.9704
 4x1      128   1.0000   0.9998   0.9998   0.9998   0.1528   0.9998   0.9998   0.9995   0.9942   0.9891
 8x1       64   1.0000   0.9998   0.9998   0.7251   0.2392   0.9998   0.9998   0.9996   0.9963   0.9910
 8x1      128   1.0000   0.9998   0.9998   0.9998   0.1621   0.9998   0.9998   0.9997   0.9988   0.9973
 8x8      128   1.0000   0.9998   0.9998   0.9998   0.1621   0.3721   0.9998   0.9998   0.9997   0.9996
 8x8      512   1.0000   0.9998   0.9998   0.9998   0.9998   0.6918   0.2514   0.9998   0.9998   0.9998
```

**Root cause.** The op reproduces the exact same degradation (0.7664 at 262143) with zero model
code involved, so it is not a page-table, `cur_pos`, cache-write or layer-composition defect. With
`k_chunk_size` left unset the op takes its **dynamic** chunk path — `get_dynamic_Sk_chunk_t` in
`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp`, which picks
`nearest_pow_of_2_up_to_8(seq_len_in_tiles)` from `cur_pos` and carries an in-source
*"Technically, should not be an issue but seeing PCC issues"* caveat. Its accuracy falls as the
per-core sequential chunk count grows with context.

**A second independent control** points the same way: the *same* 262144-key attention computed by
the **prefill** op (`chunked_scaled_dot_product_attention`, which the layer calls with an explicit
`q=k=128` program config) scores **0.9999891** at the layer level. Same cache, same page table,
same context — accurate when an explicit config is used, degraded only on the decode op's dynamic
path.

**Why the functional decoder keeps the dynamic path.** Every explicit config that improves
long-context accuracy is *structurally wrong* at some shorter context — `8x8/128` collapses to
0.3721 at context 1024, `8x1/128` to 0.1621 at 257, `8x8/512` to 0.2514 at 4096 — because too few
k-chunks are produced to feed the cross-core reduction. The only explicit configs correct at every
context (`1x1`, `2x1`) run the whole op on one or two cores. And `cur_pos` is a runtime *device*
tensor, so the layer cannot choose a config per call without a host read, which the fallback audit
(§3.7) forbids on the runtime path. The dynamic path is therefore the only setting that is
correct at every position, which is what a functional decoder must guarantee.

**Impact and handoff.** At the layer level the effect is bounded because the attention branch is
~0.2% of the layer output (`attn RMS` 0.00221 against a residual RMS of ~1.0): the measured
long-context decode PCC is **0.9986** (`1 - PCC` = 1.4e-3 against the bar's 5e-3, i.e. ~3.5x
inside it), and the layer stays >= 0.99959 at every context up to and including 131072.

Choosing an SDPA program config is an `optimize`-stage lever, and the sweep above is the data
that stage needs: it must pick per-position-safe configs (or bucket traces by context) rather
than inherit this default. Reproduce with
`python models/autoports/qwen_qwen3_6_35b_a3b/tests/diag_sdpa_decode.py`.

### 3.9 Watcher-clean run

Command (`tests/run_watcher.sh`, exact env in `work_log.md` §7):

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_LOGS_PATH=.../doc/functional_decoder/watcher \
  pytest .../test_functional_decoder.py -k "<selector>"
```

Selection deliberately spans both layer kinds and every device-facing mechanism the stage owns:
traced decode capture+replay, chunked prefill continuation, ragged per-slot `current_pos`,
`current_pos = -1` slot skipping, the paged-cache write path and the linear-attention state.

Result: **8 passed, 0 failed** (`watcher/pytest.log`), watcher log 17262 lines
(`watcher/generated/watcher/watcher.log.gz` — gzipped because the raw log is ~1.1 MB, over this
repo's 500 KB committed-file limit; inspect with `zless`/`zgrep`), and **no fatal, sanitize, out-of-bounds, stack/L1
overflow, invalid-NOC, CB-overrun or watcher-assert findings** — the automated grep in
`run_watcher.sh` writes any hit to `watcher/watcher_hits.txt`, which is empty.

The log's content is the expected benign mix: attach/detach lines, periodic core-status dump
tables and `k_ids:` kernel-id rows (8537 table/legend lines plus ~8700 `k_ids` rows), with no
diagnostic classes. Watcher was **not** combined with the device profiler — `tests/run_perf.sh`
is the separate profiling run and `test_perf.py` fails fast if `TT_METAL_WATCHER` is set.

## 4. Capability contract

Measured at the full advertised context (`long_context.jsonl`, one pytest process per case so
each of these — the largest allocations in the stage — starts from empty device DRAM):

| case | value | PCC vs HF | device wall |
|---|---|---|---|
| `longest-prefill[linear]` | seq_len **262143** (non-aligned), tail-128 compared | 0.9999742 | 43.9 s |
| `longest-prefill[full]` | seq_len **262143** (non-aligned), tail-128 compared | 0.9999891 | 48.8 s |
| `longest-prefill[linear]` carried conv state | after 262143 tokens | 0.9999698 | |
| `longest-prefill[linear]` carried recurrent state | after 262143 tokens | 0.9998960 | |
| `longest-decode[linear]` | position **262143** after a 262143-token prefill | 0.9999860 | 0.04 s |
| `longest-decode[full]` | position **262143** after a 262143-token prefill | 0.9985674 | 0.05 s |
| batch-32 full-context paged KV | 131072 blocks, 2 x 8 GiB, **allocated on device** | n/a | |

`longest-decode[full]` is the one number materially below the rest. §3.8 root-causes it to the
decode SDPA op's **dynamic `k_chunk_size` path** — reproduced with zero model code in
`tests/diag_sdpa_decode.py`, and contrasted against the *prefill* op computing the same
262144-key attention at 0.9999891 with an explicit program config. It is not a paging,
`cur_pos`, cache-write or layer-composition defect, and the dynamic path is the only setting
correct at every position. Recorded as an `optimize`-stage handoff, not a waiver.

See `../context_contract.json` for the machine-readable version (regenerate with
`python models/autoports/qwen_qwen3_6_35b_a3b/tests/write_context_contract.py`; the contract is
derived from `long_context.jsonl`, and `test_context_contract_file_is_consistent` re-checks the
relationship so it cannot go stale). Summary:

| claim | evidence | remaining risk |
|---|---|---|
| context 262144 (= HF `max_position_embeddings`), **no reduction** | `test_longest_prefill` prefills 262143 tokens in one call; `test_longest_decode_context` decodes position 262143 | none for a single layer; the 40-layer model is stage 5's problem |
| batch 32 at full context fits DRAM | `test_max_batch_full_context_capacity` allocates the real 2 x 8 GiB (16 GiB) paged cache on device | leaves ~15 GiB for weights + activations on this 31.5 GiB part |
| both layer kinds | `test_layer_kinds_cover_the_whole_model` + every test parameterised over `["linear", "full"]` | none |
| any logical seq_len | 13 lengths per kind incl. 1/33/65/129/1025/2049/3000/262143 | none |
| paged prefill + paged decode, permuted page tables | §3.4 | none |
| per-slot `current_pos` | `test_decode_ragged_current_positions` | none |
| carried state across prefill calls | `test_prefill_chunk_continuation` | none |
| traced decode | §3.5 | none |

There is **no mode switch** in this model (no sliding-window / full-attention split within a
kind, no windowed masking) — `layer_types` selects the mixer and nothing else, so there is no
mode-switch axis to test beyond the two kinds.

## 5. Performance

Command: `./models/autoports/qwen_qwen3_6_35b_a3b/tests/run_perf.sh` (one Tracy run per
mode/kind — see `work_log.md` §7). Artifacts per case in `tracy/<kind>_<mode>/`:

| file | what it is |
|---|---|
| `<mode>_perf_report.txt` | **human-readable `tt-perf-report` table** (per-op device time, cores, DRAM %, FLOPs %, math fidelity) |
| `<mode>_perf_report.csv` | the same rows as CSV, signpost-filtered (`--csv`) |
| `<mode>_perf_report.console.log` | `--csv`-run stdout, kept for provenance |
| `<mode>_perf_report_stacked.csv` / `.png` | tt-perf-report's stacked breakdown |
| `<mode>_ops.csv.gz` | the raw post-processed Tracy ops CSV the report was built from (6-21 MB raw; `gunzip -k` before re-running `tt-perf-report`). The two decode CSVs exceed this repo's 500 KB committed-file limit even gzipped and are therefore not in git — see `tracy/README.md` for the rule and the one-command regeneration |
| `tracy_run.log.gz` | full Tracy + pytest transcript (gzipped) |

Plus `perf_host_summary.jsonl` (host wall-clock rows) and `perf_summary.json` (reduced table,
produced by `tests/summarize_perf.py`).

Measured windows are bounded by signposts (`PERF_PREFILL`/`PERF_PREFILL_END`,
`PERF_DECODE`/`PERF_DECODE_END`) after a compile+warmup pass and a device synchronize. **Decode
is measured from trace replay only** (`ttnn.execute_trace` in a loop). Watcher is never enabled
in a profiling run — `test_perf.py` fails fast if `TT_METAL_WATCHER` is set.

**Units:** this `tt-perf-report` (1.2.8) emits `Device Time` and `Op-to-Op Gap` in
**microseconds** (not the raw Tracy `DEVICE KERNEL DURATION [ns]`); `summarize_perf.py` sums
those columns and divides by the iteration count.

| case | shape | ops in window | device kernel / iter | op-to-op gap / iter | host wall / iter |
|---|---|---|---|---|---|
| prefill `linear` | seq 2048, batch 1 | 1746 | **342.06 ms** | 0.813 ms | 343.32 ms |
| prefill `full` | seq 2048, batch 1 | 388 | **281.56 ms** | 0.133 ms | 282.05 ms |
| decode `linear` (traced) | batch 32, `cur_pos` 4095 | 968 | **57.99 ms** | 0.089 ms | 58.11 ms |
| decode `full` (traced) | batch 32, `cur_pos` 4095 | 880 | **50.16 ms** | 0.228 ms | 50.42 ms |

Two things worth carrying into the `optimize` stage:

* **Device-bound, not dispatch-bound.** Op-to-op gap is 0.03-0.24% of device time in every
  case, and host wall-clock matches device kernel time to within 0.4%. The traced decode has
  essentially no dispatch overhead left to remove.
* **The MoE dominates.** For prefill `full`, the two expert sparse matmuls are 147.1 ms
  (gate/up) + 79.6 ms (down) = **227 of 282 ms (80%)**; the whole attention path is the
  remainder. For decode they are 9.5 + 9.1 = 18.6 of 50 ms (37%). That is the direct consequence
  of limitation 1 below (dense-per-tile-group expert selection), so expert routing — not
  attention — is where the optimisation budget belongs. The prefill `linear` case additionally
  pays 19.2 ms in `1024 x 64 x 64 x 64` batched matmuls, which is the delta-rule UT transform and
  chunk scan.

`linear` prefill issues 4.5x more ops than `full` (1746 vs 388) for the same token count: the
gated delta rule contributes a 32-step Python-driven chunk scan plus the UT transform per
2048-token chunk. It is still device-bound, so the op count is a latency risk only at much
shorter sequences.

## 6. Known limitations / optimisation opportunities

These are *functional-stage* compromises, all deliberate and all correctness-neutral. They
belong to the `optimize` stage, not here.

1. **MoE prefill is dense-per-tile-group.** Prefill groups 32 tokens per sparse-matmul tile;
   the union of their top-8 selections is ~163 of 256 experts, so ~20x more expert-matmul
   work than the ideal gather-by-expert. Measured 9.5 TFLOP/s on the fused gate/up
   (N=1024, 32 of 64 cores busy because `sparse_matmul` parallelises over N tiles only).
2. **`nnz` is left inferred** for every `sparse_matmul`. A static count that disagrees with
   the actual non-zeros deadlocks the mcast receivers (tt-metal #45943 / #45052), and the
   count is data-dependent, so robustness wins here.
3. **q/k head duplication is folded into the weights**, growing the DeltaNet input projection
   from 8192 to 16384 columns (`+4096x2048` MACs/token) to avoid a runtime
   `repeat_interleave`. A 16-head-with-256-wide-value state layout would remove that cost.
4. **Everything is DRAM-interleaved bf16** with no L1 sharding or program-config tuning
   outside the sparse matmuls and SDPA.
5. **`delta_dtype = float32`** for the whole chunked delta rule, not just the state.
6. `chunked_scaled_dot_product_attention` is called with an integer `chunk_start_idx`, so one
   program is compiled per distinct prefill-chunk offset. The device-tensor form would avoid
   that but needs a host write to update the offset, which the fallback audit forbids on the
   runtime path.
7. **Decode SDPA runs on the op's dynamic `k_chunk_size` path** (no explicit
   `SDPAProgramConfig`). That is the only setting correct at every `cur_pos`, but it costs
   accuracy at very long context — see §3.8 for the full grid x `k_chunk` x context sweep. The
   `optimize` stage should pick per-position-safe explicit configs (or bucket traces by context)
   using that sweep; doing so would also lift the 262143-position decode PCC from 0.9986 toward
   the ~0.9996 the best explicit config reaches at that length.

## 7. Bugs found and fixed during bringup

1. **`ttnn.slice`, `ttnn.reshape` and `ttnn.to_memory_config` can return aliasing views.**
   `ttnn.slice` hands back the *input* (same buffer address) when the slice covers the whole
   tensor, `ttnn.reshape` is always a view, and `to_memory_config` returns its input when
   nothing has to move. Deallocating those results freed the caller's page table and the
   persistent conv/recurrent state buffers, and in the MoE it freed the buffer an earlier
   chunk's output still pointed at. Fixed with three explicit ownership helpers — `_subview`
   (returns an `owned` flag from a buffer-address comparison), `_owned_slice` (clones when the
   slice aliases) and `_move` — plus a `_view` helper that documents "the input stays the
   owner". **Measured effect of the MoE part of this fix:** `prefill[full] seq=2049` went from
   PCC 0.9998672 to 0.9999925, i.e. the aliasing was silently corrupting multi-chunk MoE
   prefill output, not just leaking.
2. **Padding tokens contaminated the linear-attention state.** Prefill pads up to
   `PREFILL_ALIGN`; the padded rows were advancing the conv and recurrent state past the end
   of the real sequence, so the next decode step continued from the wrong state. Output PCC
   could not see it (padded outputs are sliced off). Fixed by taking the conv context at the
   *logical* end and by zeroing `beta` and `g` on padded positions — which makes the
   recurrence step exactly the identity there. Regression test:
   `test_decode_after_non_aligned_prefill`.
3. **`conv_dim` was 3 blocks wide, not 4.** The z block rides the depthwise conv with an
   identity tap (so one conv+silu emits post-conv q/k/v *and* `silu(z)`), which the derived
   width did not account for.
4. **Repo-level:** the root `conftest.py` still imported the deleted
   `models/tt_transformers`, breaking pytest collection for the whole repository. Guarded the
   import with a no-op fallback.
5. **Not a bug in this stage, but found by it:** the decode SDPA op's dynamic `k_chunk_size`
   path loses accuracy as context grows (PCC 0.9998 -> 0.7664 from 1024 to 262143 keys, measured
   on the op alone with no model code). Root-caused, bounded and handed to `optimize` — §3.8.
   The in-source comment at
   `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp:104`
   already flags the same path as PCC-sensitive.

## 8. Repository files this stage owns

| path | role |
|---|---|
| `tt/functional_decoder.py` | **the deliverable** — `FunctionalDecoder`, `DecoderConfig`, ownership helpers |
| `tt/reference.py` | HF reference / weight-loading boundary (torch); single-layer state-dict extraction, weight stats, synthetic weights, prefill/decode goldens, O(seq) tail references |
| `tests/harness.py` | matched (HF, TTNN) pair construction, page tables, state snapshot/restore, `TracedDecode`, PCC logging |
| `tests/conftest.py` | session device + layer-pair cache |
| `tests/test_reference_math.py` | 26 CPU-only tests pinning the algebraic rewrites (no device, no checkpoint) |
| `tests/test_functional_decoder.py` | the correctness suite |
| `tests/test_long_context.py` | 262144-token advertised-context evidence |
| `tests/test_perf.py` | signposted warmed prefill / traced warmed decode |
| `tests/probe_ttnn_ops.py` | 21 device op-behaviour probes the design was built on |
| `tests/diag_long_decode.py`, `tests/diag_sdpa_decode.py` | the two diagnostics behind §3.8 |
| `tests/run_perf.sh`, `tests/run_watcher.sh`, `tests/run_long_context.sh` | reproducible runners |
| `tests/write_context_contract.py`, `tests/summarize_perf.py` | derive `context_contract.json` / `perf_summary.json` from evidence |
| `tests/dev_smoke.py` | single-command dev driver (build a pair, prefill, decode, print PCC) |

One repo file outside the autoport directory was touched: `conftest.py` (§7 item 4).
