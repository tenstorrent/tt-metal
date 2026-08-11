# Functional decoder — meta-models/Muse-Glimmer-30B (text decoder)

Stage 01 of the repo-local TTNN autoport pipeline: a functionally complete TTNN
implementation of the Muse-Glimmer **text** decoder layer, validated against the
HuggingFace reference on one Blackhole chip.

* implementation: [`../../tt/functional_decoder.py`](../../tt/functional_decoder.py)
* host reference: [`../../reference/hf_reference.py`](../../reference/hf_reference.py)
* tests: [`../../tests/test_functional_decoder.py`](../../tests/test_functional_decoder.py),
  [`../../tests/test_functional_decoder_perf.py`](../../tests/test_functional_decoder_perf.py)
* measured numbers: [`evidence_tables.md`](evidence_tables.md) (generated from the raw artifacts)
* chronology, bugs, exact commands: [`work_log.md`](work_log.md)
* context contract: [`../context_contract.json`](../context_contract.json)

## Scope

`meta-models/Muse-Glimmer-30B` is a vision-language checkpoint
(`MuseGlimmerForConditionalGeneration`). This stage implements **only** the text decoder
layer (`text_config`, `model.language_model.layers.*`). The vision tower, the multimodal
projector and image/video inputs belong to a later dedicated stage and are deliberately
absent here — that is the stage split, not a capability reduction.

`final_logit_softcapping = 20.0` and `output_multiplier = 0.19611613513818404` are applied
by `MuseGlimmerForConditionalGeneration.forward` to the **logits**
(`modeling_muse_glimmer.py:1255-1260`), i.e. at the LM head, not inside the decoder layer.
They are recorded here for the full-model stage (a working TTNN softcap reference is
`models/demos/gemma4/tt/model.py`) but are out of scope for this file.

## Layer kinds

`text_config.layer_types` repeats `[sliding, sliding, sliding, full]` over 52 layers, and
`text_config.layer_rope_theta` is `500000.0` on the sliding layers and `0` on the
full-attention ones. In `MuseGlimmerTextModel.forward` a `layer_rope_theta[i]` of 0 means
`position_embeddings=None` is passed to that layer — those layers are **NoPE**: no rotary
embedding at all. There is exactly one rotary embedding module in the HF model, built from
`rope_parameters["rope_theta"] = 500000.0`; `layer_rope_theta` acts purely as this on/off
marker. So the model has exactly two meaningful decoder-layer kinds, and every test below
runs against both:

| kind id | representative layer | attention | RoPE | window |
|---|---|---|---|---|
| `sliding_rope` | 0 (also 1, 2, 4, 5, 6, …) | sliding | yes, theta 500000.0 | 2048 |
| `full_nope` | 3 (also 7, 11, …, 51) | full causal | **none (NoPE)** | — |

Everything else is shared: GQA 32 Q heads / 2 KV heads, `head_dim` 128, hidden 6656,
SwiGLU intermediate 19968, no projection biases, `qk_scale_factor` 3.87, weight-less QK
RMSNorm over `head_dim`, gated attention (`attn * sigmoid(gate_proj(xn))`), and sandwich
centered RMS norms (`rms_norm_eps` 1e-5 on the pre-norms, `post_norm_eps` 1e-8 on the
post-norms, each computing `normed * (1 + w)`).

## Public contract

```python
FunctionalDecoder.from_state_dict(
    state_dict,                # HF names, layer-local or checkpoint-prefixed
    *, hf_config, layer_idx, mesh_device,
    state_dict_prefix=None, weight_dtype=ttnn.bfloat16, cache_dtype=ttnn.bfloat16,
    rope_dtype=ttnn.bfloat16, block_size=64, prefill_chunk_size=8192,
    rope_max_seq_len=None, sdpa_core_grid=None,
    prefill_sdpa_q_chunk=256, prefill_sdpa_k_chunk=256,
    decode_sdpa_core_grid=None, decode_sdpa_k_chunk=None,
) -> FunctionalDecoder

decoder.allocate_kv_cache(*, batch_size, max_seq_len, num_blocks=None)
    -> (k_cache, v_cache)      # [num_blocks, num_key_value_heads, block_size, head_dim]

decoder.prefill_forward(
    hidden_states,             # [batch, 1, seq_len, hidden_size]  TILE, bf16
    *, kv_cache, page_table,   # page_table int32 [rows, blocks_per_seq], rows >= batch
    user_ids=None,             # host list[int]: page-table row (cache slot) per input row
    seq_len=None,              # logical length; defaults to hidden_states.shape[-2]
    start_pos=0,               # continued prefill: full_attention layers only, and a
                               # multiple of 32 and of block_size (see Limitations)
) -> ttnn.Tensor               # [batch, 1, seq_len, hidden_size]

decoder.decode_forward(
    hidden_states,             # [1, 1, batch, hidden_size]  TILE, bf16
    *, kv_cache,
    page_table,                # int32 [batch, blocks_per_seq] — one row per user, in batch order
    current_pos,               # int32 device tensor [batch]: absolute positions
    rope_idxs=None,            # uint32 device tensor [1, batch]: same positions (RoPE layers)
) -> ttnn.Tensor               # [1, 1, batch, hidden_size]

decoder.forward(*args, mode="prefill"|"decode", **kwargs)   # dispatch helper
```

Invariants the contract guarantees:

* A single `prefill_forward` call handles **any** logical `seq_len` in `[1, 131072]`. The layer owns tile padding,
  sequence chunking, page-table slicing and output slicing; there is no
  `seq_len % N == 0` requirement on the public path. Padding rows carry K/V derived from
  zero activations, which causal masking and `current_pos`-bounded decode reads never
  observe.
* Prefill is chunked at `prefill_chunk_size` (8192) so activation memory is bounded at any
  context length and every SDPA call stays within the 32768-token Q bound beyond which the
  non-chunked prefill SDPA is known to return wrong results.
* `current_pos` and `rope_idxs` are plain device tensors, so decode is trace-safe: capture
  once, then only their contents change per step.
* After `from_state_dict`, prefill and decode are pure device paths (no `torch`, no
  `ttnn.from_torch` / `ttnn.to_torch`, no host fallback) — enforced by two tests.
* **Continued prefill (`start_pos > 0`) is supported on `full_attention` layers only.** A
  `sliding_attention` layer raises: its windowed SDPA takes the window prefix from the input
  tensor, and the chunked SDPA op that could read that prefix out of the paged cache has no
  sliding-window mode, so a continued segment's first 2048 positions would silently attend to
  a truncated window. A whole prompt of any supported length in one call is unaffected — the
  layer chunks internally. Recorded in
  [`../context_contract.json`](../context_contract.json) under `continued_prefill`.

## Capability-contract evidence

| claim | evidence | remaining risk |
|---|---|---|
| Context length is the full HF-advertised 131072 (`text_config.max_position_embeddings`); no reduction | `test_full_context_prefill_and_decode[both kinds]`: prefill at 131072 **and** 131071 with streaming PCC over every query position, plus decode at position 131071. `../context_contract.json` | none identified; a 131072-token single-layer prefill peaks well under the 34.18 GB device DRAM |
| Both decoder-layer kinds are implemented and exercised | every device test is parameterised over `sliding_rope` and `full_nope`; `test_layer_config_matches_hf` asserts the config mapping incl. the NoPE marker | none |
| Sliding window is really 2048 tokens, inclusive of the current position | HF `sliding_window_overlay` (`kv_idx > q_idx - window`), TTNN prefill (`left = window - 1` behind the diagonal) and TTNN decode (`window_start = cur_pos + 1 - window`) agree; `test_reference_matches_hf[3000]` and `test_sliding_window_is_enforced` confirm empirically | none |
| Paged KV cache addressing is correct | every test uses a shuffled page table drawn from a pool larger than needed (so logical block != physical block), asserts no aliasing, and compares the un-paged cache contents against the reference; block sizes 32/64/128 | none |
| Cache slots / `batch_idx` handling is correct | `test_ragged_slots_and_current_positions`: four users with different non-aligned prompt lengths prefilled into a permuted slot order, then decoded together | none |
| Per-user `current_pos` handling is correct | ragged positions `[37, 1000, 2049, 3111]` in one batched decode; multi-step decode; decode at the maximum position 131071 | none |
| Non-aligned lengths work | prefill/decode at 32, 100, 1000, 2048, 2080, 3000, 8192, 8256, 12345, 131071, 131072 — covering tile, page, window and chunk boundaries and one length just past each | none |
| Decode runs under traced execution | `test_traced_decode_pcc` measures PCC **from the replay output**; `test_decode_perf` measures the traced window | none |
| Batch up to 32 | `test_batched_prefill_and_decode[4, 32]` for both kinds, per-user PCC asserted | batch > 32 not attempted. The hard ceiling is `110 / num_key_value_heads = 55` users: the decode SDPA needs one core per (user, KV head) pair on the 11x10 grid (see bug 8 in `work_log.md`), and `nlp_concat_heads_decode` needs one core per user |

## Test matrix

| test | what it pins down |
|---|---|
| `test_reference_matches_hf` | the chunked host reference **is** `MuseGlimmerTextDecoderLayer.forward` with HF's own mask builders (PCC 1.0 eager) |
| `test_reference_decode_matches_hf` | the host reference's *decode* step is the untouched HF layer driven through a real `DynamicCache` and HF's mask builder |
| `test_streaming_pcc_matches_comp_pcc` | the streaming PCC used for long context matches `models.common.utility_functions.comp_pcc` |
| `test_layer_config_matches_hf` | config mapping: layer type, window, RoPE/NoPE, eps values, `qk_scale_factor`, 131072 |
| `test_sdpa_chunk_divides_length` | SDPA chunk sizes always divide the sequence length (a non-dividing `q_chunk_size` *hangs* the op) |
| `test_chunked_sdpa_chunk_sizes_respect_the_configured_cap` | the full-attention prefill SDPA runs the configured, swept geometry — not "as large as divides" |
| `test_chunked_sdpa_k_chunk_stays_inside_the_page_table` | the k-chunk-padded K extent never overruns the page table, for every chunk of 5 lengths x 3 block sizes (the op does not check this itself) |
| `test_decode_sdpa_k_chunk_stays_inside_the_page_table` | the same guard on the decode side, for every position the page table can address |
| `test_decode_sdpa_grid_covers_kv_heads` | the decode SDPA grid always has one core per (user, KV head), and fails loudly past the grid's capacity |
| `test_paged_prefill_decode_pcc` | paged prefill + 2 decode steps + K/V cache contents at 9 sequence lengths x 2 kinds |
| `test_page_block_sizes` | page block sizes 32 / 64 / 128 |
| `test_batched_prefill_and_decode` | batched prefill and batched decode at batch 4 and 32, per-user PCC |
| `test_ragged_slots_and_current_positions` | permuted cache slots, ragged prompt lengths, ragged decode positions |
| `test_short_prefill_lengths` | sub-tile prompts (1, 7, 31 tokens) — the bottom of the advertised range |
| `test_batched_multichunk_prefill_shared_pool` | batch 2 x 8256 tokens at `block_size=128` from a 4-row shared pool with non-identity slots: multi-chunk fill at `first_block > 0`, the multi-row page-table gather, and the sliding overlap trim, all with batch > 1 |
| `test_continued_prefill_contract` | `start_pos > 0`: exact on full-attention layers (vs a single-shot reference), refused on sliding layers |
| `test_sliding_window_is_enforced` | sliding layers ignore out-of-window tokens; full layers do not |
| `test_determinism_repeated_inputs` | three identical prefills and decodes are **bit-identical** |
| `test_traced_decode_pcc` | trace capture + replay, PCC from the replayed output, 3 steps |
| `test_real_weights_prefill_decode` | real checkpoint weights and real activations (real embeddings run through the real preceding layers) |
| `test_full_context_prefill_and_decode` | the full 131072 context (and 131071), decode at position 131071 |
| `test_no_runtime_host_fallback` | a prefill and a decode pass with every torch op and every ttnn host-transfer entry point trip-wired |
| `test_source_has_no_runtime_torch` | AST audit: `torch` / host transfers appear only in `from_state_dict` and `allocate_kv_cache` |

Run them:

```bash
# fast suite (host reference + device), ~5.5 min
python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py \
  -q -k "not long_context"

# full-context suite (the host reference at 131072 dominates the runtime)
python -m pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py \
  -q -k "long_context" -s
```

## Correctness results

Acceptance bar: **PCC >= 0.995**, the `$functional-decoder` default. No model-specific
exception was needed or claimed.

See [`evidence_tables.md`](evidence_tables.md) for the per-test tables, generated from
[`pcc/pcc_results.json`](pcc/pcc_results.json) — every PCC the suite measured, including
the real-weight and full-context runs.

## Performance

Warmed prefill and traced warmed decode are measured between Tracy signposts
(`PERF_PREFILL`/`PERF_PREFILL_END`, `PERF_DECODE`/`PERF_DECODE_END`) by
`tests/test_functional_decoder_perf.py`, collected by
[`../../scripts/collect_perf.sh`](../../scripts/collect_perf.sh):

* human-readable `tt-perf-report` tables: `tracy/<kind>/<mode>_<size>_perf_report.txt`
* filtered per-op CSV: `tracy/<kind>/<mode>_<size>_perf_report.csv`
* summary + advice: `tracy/<kind>/<mode>_<size>_perf_report.summary.txt`
* raw Tracy ops CSV (gzipped) + its original path:
  `tracy/<kind>/<mode>_<size>_ops.csv.gz` and `...csv.provenance`
* the pytest/Tracy session log: `tracy/<kind>/<mode>_<size>_pytest.log`
* wall clock per measured window: [`perf/perf_summary.json`](perf/perf_summary.json)

Aggregated numbers (device time from the signposted window and wall clock per iteration)
are in [`evidence_tables.md`](evidence_tables.md). Headline figures, one decoder layer on one
Blackhole chip:

| measurement | sliding_rope | full_nope |
|---|---|---|
| warmed prefill, 4096 tokens | 59.1 ms (69.3 k tok/s) | 59.6 ms (68.7 k tok/s) |
| warmed prefill, 8192 tokens | 98.1 ms (83.5 k tok/s) | 98.1 ms (83.5 k tok/s) |
| traced warmed decode, batch 1, context 4096 | 3.24 ms | 3.22 ms |
| traced warmed decode, batch 32, context 4096 | 3.40 ms (9.4 k tok/s) | 3.57 ms (9.0 k tok/s) |

(unprofiled wall clock, 5 prefill / 32 decode measured iterations)

Wall clock and summed device time agree for decode (sliding batch 1: 3.24 ms wall clock vs
3.21 ms of summed device time per replay — pure device time) and are ~20% apart for prefill at
8192 (~79 ms device vs ~98 ms wall clock), i.e. prefill still carries per-chunk host
dispatch. Every committed capture is complete: op-instance counts in each report are exact
multiples of the measured iteration count, which is why the profiled decode window is
shortened to `PROFILED_DECODE_ITERS = 12` (32 replays overflowed the per-core profiler buffers
and silently dropped rows).

This stage optimises for correctness, not for speed — dtype/layout/program-config tuning is
stage 02. The profiler reports say exactly where that work is:

* **prefill** is dominated by matmul device time, and `tt-perf-report` marks every dominant
  matmul `SLOW` at 8-15% DRAM utilisation and ~43% of peak FLOPs with `in0_block_w=1` and
  inputs in DRAM interleaved — i.e. program-config and layout work, not an algorithmic
  problem.
* **decode** is dominated by matmul device time at 4-7% of peak FLOPs: the classic
  DRAM-bandwidth-bound decode matmul with unsharded DRAM weights, with the norms second and
  the decode SDPA itself a small fraction.

The exact per-op-family percentages for every artifact are generated into
[`evidence_tables.md`](evidence_tables.md) ("Device-time share by op") straight from the
committed reports, so they cannot drift from the data.

Deliberate correctness-first choices that stage 02 should revisit are listed under
*Limitations*.

Blackhole core grids were swept rather than inherited from Wormhole examples:
[`perf/core_grid_sweep.md`](perf/core_grid_sweep.md) measures 8 legal grids — the 11x10 full
grid, 11x8, 10x10, 8x10, 11x5 and the Wormhole-shaped 8x8 / 8x4, plus 4x4 as a floor — across
chunk-size combinations for the three SDPA call sites, at batch 1 *and* batch 32 for decode,
under the layer's own compute-kernel config. Raw data:
[`perf/core_grid_sweep.csv`](perf/core_grid_sweep.csv). It is a representative set of legal
grids, not an exhaustive enumeration of all 110.

## Runtime fallback audit

Two independent checks, both in the fast suite:

1. `test_no_runtime_host_fallback` runs one prefill and one decode pass inside a
   `torch.overrides.TorchFunctionMode` that raises on **any** torch op, with
   `ttnn.from_torch`, `ttnn.to_torch` and `ttnn.as_tensor` replaced by tripwires. This
   covers helpers the layer calls, not just the layer file.
2. `test_source_has_no_runtime_torch` walks the module AST and fails if `torch`, a torch
   attribute, or a ttnn host-transfer entry point appears outside `from_state_dict` and
   `allocate_kv_cache`.

Setup-time host work is intentional and confined to those two methods: weight transposes,
QKV fusion, the `1 + w` norm folding, the RoPE cos/sin caches, and the zero-filled cache
allocation.

## Determinism

`test_determinism_repeated_inputs` runs three identical prefills and three identical decode
steps for both layer kinds and asserts `torch.equal` on the outputs — bit-identical, not
just high PCC.

## Watcher

A watcher-enabled run over a representative subset — chunked prefill (8256), non-aligned
prefill (100), sub-tile prefill (1/7/31), traced decode, batch-32 prefill+decode, ragged
slots/positions, batched multi-chunk prefill from a shared pool at `block_size=128`, the
continued-prefill contract, page block sizes 32/64/128 and the real-weight tests, both layer
kinds: **26 of the 103
collected tests, all passed** — is in [`watcher/`](watcher):
[`WATCHER_SUMMARY.md`](watcher/WATCHER_SUMMARY.md), the 19416-line raw log at
`watcher/generated/watcher/watcher.log.gz` (gzipped for the repo's 500 KB file limit), and the test log at `watcher/pytest_watcher.log`.
Zero watcher-detected errors, kernel asserts, NOC sanitization failures, CB/L1 overflows or
tripped waypoints; minimum stack headroom 1312 bytes free. The summary records the code
fingerprint it certifies, which is the same fingerprint every PCC record carries — the run is
not from an earlier revision. `TT_METAL_LOGS_PATH` isolates it, and watcher and profiler runs
are kept separate as `$tt-device-usage` requires. The full-context (131072) and real-weight
paths are outside the watcher selection, for runtime reasons.

## Limitations and follow-ups

* **Single device by design.** Functional bringup runs on a `(1, 1)` mesh; the target
  `(1, 4)` P300x2 mesh is stage 03 (`$multichip`). Nothing in the layer hard-codes one
  device: weights are uploaded with `ReplicateTensorToMesh` and no CCL is needed yet.
* **Correctness-first numerics.** bf16 weights, bf16 KV cache, `HiFi2` + `fp32_dest_acc_en`
  for the big matmuls and `HiFi4` for norms/SDPA, everything DRAM-interleaved. There is
  precision headroom for stage 02 / the dtype sweep — see `evidence_tables.md` for the
  measured worst case.
* **Batched prefill fills the cache per user.** `paged_fill_cache` is called once per user
  slot with a one-row page-table slice, because the batched `batch_idx_tensor` path would
  need a host-built tensor inside the runtime path. Matmuls and SDPA still run batched.
  Stage 02 can hoist a persistent `batch_idx_tensor` into setup.
* **`chunked_scaled_dot_product_attention` cannot take a Python `scale`.** Its nanobind
  binding marks `scale` `.noconvert()` on a `std::optional<float>`
  (`ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp:381`), so any Python float
  raises "incompatible function arguments". Worked around exactly (not approximately) by
  applying `qk_scale_factor` to Q — where HF applies it — and letting the op use its default
  `1/sqrt(head_dim)`. Worth an upstream fix (drop `.noconvert()` or accept `double`).
* **Blackhole decode sharding is op-dictated.** `nlp_create_qkv_heads_decode` picks a
  non-rectangular core set for batch 32 on an 11x10 grid, while `nlp_concat_heads_decode`
  requires a rectangle. The layer therefore derives the RoPE cos/sin shard layout from Q's
  own `shard_spec` and builds a rectangle only for the concat reshard. See bug 6 in
  [`work_log.md`](work_log.md) — this one costs ~0.09 PCC when it is wrong and is invisible
  at batch 1.
* **The chunked (paged) prefill SDPA can read past the page table.**
  `chunked_scaled_dot_product_attention` validates only
  `kv_length >= q_len + chunk_start_idx`, but its program factory rounds the K extent up to
  `k_chunk_size` and the reader consumes one page-table entry per `block_size` of that
  *rounded* extent with no bound check, so it can read the page-table stick's alignment
  padding as physical block ids and then read K/V from outside the cache buffer. Symptoms
  measured: layer PCC 0.7345 instead of 0.9998, and separately a device hang.
  `_chunked_sdpa_chunk_sizes` therefore requires
  `round_up(chunk_start + chunk_len, k_chunk) <= page-table capacity`, which costs a smaller
  K chunk on some geometries. Standalone reproducer:
  [`../../scripts/repro_chunked_sdpa_page_table_overrun.py`](../../scripts/repro_chunked_sdpa_page_table_overrun.py)
  (its out-of-bounds calls are opt-in because they can hang the device). Needs an upstream
  bound check; see `work_log.md` section 9.
* **The paged decode SDPA has the same unvalidated rounding**, on a different axis:
  `rt_args_common.hpp` rounds its K extent to `nearest_n(cur_pos + 1, k_chunk_size)` and the
  reader resolves that rounded extent through the same unbounded page-table lookup, with
  `cur_pos` a device tensor that nothing host-side bounds. `_decode_sdpa_k_chunk` therefore
  picks a K chunk that **divides the page-table capacity**, which makes the rounding safe for
  every addressable position. Also needs an upstream bound check.
* **SDPA chunk sizes must divide the sequence length.** `scaled_dot_product_attention`
  *hangs* (not errors) when `q_chunk_size` does not divide the Q length — reproduced with
  Q `[1, 32, 2080, 128]` and `q_chunk_size=512`, captured in
  [`triage/tt-triage.txt`](triage/tt-triage.txt). `_sdpa_chunk_for_length` therefore picks
  the largest candidate that divides the length, which costs some prefill throughput at
  awkward lengths (2080 falls back to `q_chunk=32`). Worth an upstream validation check.
* **Decode SDPA needs one core per (user, KV head).**
  `sdpa_decode_program_factory.cpp` validates only `cores >= batch`; with
  `cores < batch * num_kv_heads` it silently folds both KV heads onto one core and returns
  wrong results (batch 32 on an 8x4 grid: PCC 0.7176). The layer only uses the
  sweep-optimal small grid while it has enough cores. Also worth an upstream validation
  check.
* **Continued prefill (`start_pos > 0`) is unavailable on sliding layers** — 39 of the 52
  layers. The blocker is an op contract: the windowed prefill SDPA has no paged-cache mode
  and the chunked (paged) SDPA has no window, so a continued sliding segment cannot see the
  part of its window that lives in the cache. It raises rather than returning a
  truncated-window answer (`test_continued_prefill_contract`). Full-attention layers do
  support it, exactly (same test). A whole prompt of any supported length in a single call is
  unaffected. Any later prefix-caching or chunked-prefill serving path needs this extended.
* **`start_pos` must also be a multiple of 32 and of `block_size`**: the paged fill maps input
  tile 0 to the first block of the page-table slice it is given, so a sub-block start offset
  cannot be expressed.
* Batch is capped at 55 users by the decode SDPA's one-core-per-(user, KV head) requirement
  on the 110-core Blackhole grid; 32 is tested, which is the target serving batch.
