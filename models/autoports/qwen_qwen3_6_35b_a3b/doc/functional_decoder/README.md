# Qwen3.6-35B-A3B functional decoder

Status: all functional hard gates pass at the full advertised context of 262144 tokens and PCC >= 0.995. Warmed performance, watcher, runtime audit, real/synthetic weights, trace reuse and determinism evidence are complete. Independent [stage review](stage_review_round1.md) returned **clean-pass** in round1, with no Required Work.

Target: `Qwen/Qwen3.6-35B-A3B@345d467b117d1fd5eda9b879a5988de9ae5f47a7`, tt-metal starting revision `822d04806f292e2cf7e92de7f7be79fa8d83e5aa`. Execution uses Blackhole physical chip 0 in a 1x1 mesh. The target config is unchanged: hidden width 2048; 40 layers with three linear-attention layers followed by one full-attention layer; 256 routed experts, top 8, intermediate 512 and a gated shared expert. Advertised context: 262144.

## API and state contract

```python
FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device)
decoder.new_recurrent_state()
decoder.new_kv_state(*, context_length=262144, page_order=None,
                     num_physical_pages=None, kv_cache=None)
decoder.prepare_positions(*, start_pos, seq_len)
decoder.prepare_prefill(*, seq_len, start_pos=0, chunk_size=256)
decoder.prefill_forward(hidden_states, *, state, plan)
decoder.prefill_batch(hidden_states, *, states, plans)
decoder.decode_forward(hidden_states, *, state, positions=None)
decoder.decode_batch(hidden_states, *, states, positions)
```

`state_dict` contains the canonical HF decoder-local keys, as returned by a layer's `state_dict()`. The test loader selects only the needed safetensor shards and strips `model.language_model.layers.<idx>.`. Config dimensions are validated against the target; no miniature config is substituted. Layer 0 represents `linear_attention`, layer 3 `full_attention`; every layer of each kind has identical computation and geometry with distinct weights.

A row input/output is a device BF16 tensor `[1, S, 2048]`; decode uses `[1, 1, 2048]`. Batch methods take equally sized sequences of row tensors and row states. Logical row lengths and starts may differ. The functional batch baseline loops over rows with shared weights. All dense work, router decisions and tensor reductions remain on the device. MoE computes a dense expert bank and weights only the actual top-8 selected experts, plus the sigmoid-gated shared expert. This is a correctness baseline; dense expert execution and serial rows are performance opportunities for a later stage, not changed semantics.

Linear-attention state holds the last three raw convolution projection rows `[1,3,8192]` and the FP32 recurrence `[1,32,128,128]`. The next input continues exactly from this state. A fresh request requires zero state; a prefix continuation requires the matching convolution/recurrent snapshot. Absolute position alone cannot reconstruct recurrent history. The state is updated in place to preserve trace addresses.

Full-attention state holds BF16 K/V `[physical_pages,2,32,256]`, an INT32 page table, and the requested logical capacity. Supply `ceil(context_length/32)` distinct owned pages. The internal table pads to `8*ceil(context_length/256)` entries by repeating the final owned page; those aliases are finite, causal read-only padding. The default physical pool rounds up to that table width; an explicitly supplied shared pool must be at least as wide because native cache-update validation requires it. This adds at most 458752 bytes across K/V and does not change logical capacity. Requests may share a larger pool with disjoint owned pages. The caller owns page assignment and must keep current positions inside the requested capacity. When refreshing a live trace's page table, refresh the same padded shape and repeat the final owned page in padding entries.

Full attention applies Q/K normalization, 64-dimensional partial RoPE, causal paged attention, the output sigmoid gate, residuals and MoE. Text positions have identical HF mRoPE axes. Callers with precomputed position embeddings can supply the corresponding `cos`/`sin` tensors directly.

`prepare_prefill` creates a host plan outside the measured pass. It distinguishes absolute position from input offset. Full-attention interiors start at 256-aligned absolute positions and use chunks of at most 256 tokens; linear attention uses 32-aligned interiors. Unaligned starts and tails use exact one-token kernels until alignment. Every logical output is returned. Padding never advances recurrent state or overwrites live KV entries. Every intermediate chunk fills the cache. Interior chunks use paged fill and chunked SDPA; boundary tokens and decode use paged update and paged decode SDPA. Public lengths are unrestricted within the HF context and requested capacity.

Both full-attention paths use 256-token key chunks to limit repeated BF16 output accumulation at long contexts. Decode additionally distributes context work over an 8x8 program grid; its sole request's Q owner remains core (0,0). These settings repair measured long-context accuracy failures. The page-table padding makes rounded key reads legal even for a one-token capacity; the causal mask excludes all padded logical positions.

Preparation and state allocation must precede trace capture. Warm every physical input/plan path that will execute while a trace is live, including eager prefill paths. Decode reads stable device input, state, page-table, current-position and RoPE buffers; refresh their contents before replay. `decode_forward` itself has no host conversion. Caller orchestration owns capture/replay and request state restoration. The trace output is consumed before the next replay; only that exact output buffer is acknowledged to the allocation tracker. No broad allocation exemptions are used.

## Validation and artifacts

The acceptance threshold is PCC >= 0.995, unchanged. Real weights are used for both layer kinds. Ordinary pytest generates deterministic synthetic weights from `weight_stats.json` with the exact target config in `hf/config.json`. The CPU reference is the installed canonical HF decoder with DynamicCache; its optional FLA warning concerns only that reference.

| Layer kind | Prefill 262143 PCC | Prefill 262144 PCC | Traced decode at 262143 PCC |
|---|---:|---:|---:|
| Linear attention, layer 0 | 0.999876320 | 0.999876326 | 0.999923123 |
| Full attention, layer 3 | 0.999887458 | 0.999887432 | 0.999947275 |

Both long traces replay three times with bit-identical output. `full_long.json` additionally checks every final 64 row at both lengths: minimum 0.998946022 and 0.998649944. All long runs include real weights and device-generated state/cache. `context_contract.json` records the complete 262144-token contract with no capability reduction. Earlier full-attention failures and intermediate repairs are historical diagnostics, identified in `AUTOFIX.md`; final evidence is `full_long.json` with its source-hash sidecar. `source_audit.json` confirms unchanged linear arithmetic and equivalent linear planning against the checkpoint that contains its long-context run.

| Contract | Evidence | Scope / remaining risk |
|---|---|---|
| Both kinds, real target shapes and weights | `linear_long.json`, `full_long.json`, `tracy/layer_{0,3}/correctness.json` | Representative layer 0/3; other layers of each kind share geometry but have different weights |
| Arbitrary logical lengths and chunk boundaries | `partition_{0,3}.json`: 1,31,32,33,255,256,257,513, then 7 on one instance; chunk32/256 parity | Finite tested boundary set, plus full-context nonaligned length262143 |
| Prefix continuation | `linear_battery.json`, `full_resume_fixed.json`: starts 32,33,256,512,528 | Every resumed row uses contribution PCC after subtracting residual; dropped-history controls fail the same gate |
| Mixed batch 32 and ownership | `linear_mixed_batch.json`, `full_shared_pool.json` | Per-row functional loop; full kind checks 64 K/V views and guard pages |
| Stable traced decode across changed requests | `trace_reuse_{0,3}.json`: one capture, nine requests each | Changes inputs, lengths, positions, pages and continuation; positive allocation-hazard control rejects |
| Padded page-table capacity and masks | `page_extent_regression.json`, `padded_trace_regression.json` | Capacities 1/31/32/33/255/256/257/511/512/513; nonzero future rows, alias refresh and guard ownership |
| Synthetic target-shape CI | `synthetic_pytest.xml`, `pytest/` | Two tests, one per kind; no miniature configs |
| Runtime without host fallback | `fallback_audit.json`, `tests/runtime_audit.py` | AST/native-helper inspection and Torch-dispatch/conversion guards around measured forwards |
| Determinism and watcher | `trace_reuse_{0,3}.json`, `watcher_audit.json` | Repeated identical request is bit-equal; watcher 10 separately from profiler |
| Warmed device performance | `functional_decoder.md`, `perf_summary.json`, `tracy/layer_{0,3}/` | One warmed 256-token prefill and 16 traced decode replays; device time, not model throughput |

`hard_gates.json` records the completed gate check. `profiler_completeness.json` verifies identical operation sequences across all 20 warmup/correctness/measured replay sessions per kind, with 16 measured sessions and no dropped markers. Independent [stage review](stage_review_round1.md) returned **clean-pass** in round1, with no Required Work.

Run commands from this checkout with `source python_env/bin/activate` and `source ../lane-env.sh`. Device jobs must remain serialized and bounded. Exact run commands, failures and evidence paths are in `work_log.md`; the main reproduction commands are:

```bash
timeout 14400 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.long_tail_regression
bash models/autoports/qwen_qwen3_6_35b_a3b/tests/finish_validation.sh
bash models/autoports/qwen_qwen3_6_35b_a3b/tests/collect_perf.sh
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.audit_source
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.audit_watcher
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.check_evidence
```

For linear full-context reproduction, use `tests.long_context --layer 0 --output <path>` under the same 14400-second timeout and allocation-tracking environment. `tests.finish_validation.sh` runs both kinds by default; `FUNCTIONAL_FULL_ONLY=1` refreshes only affected full-attention checks after a full-only change, while still running both synthetic tests.

This stage exposes text decoder layers, with no optimized decoder, multi-chip execution, full model, generation or serving implementation. The largest context runs are correctness tests; their compilation/readback-inclusive wall times are not performance claims. Finite cache contents, disjoint owned pages and valid current positions are caller obligations. The dense expert bank and serial batch loop are functional performance limitations. Preparation must precede live traces as documented above.

Exact closed logs/CSVs are retained as deterministic `.gz` archives with hashes in `archive_manifest.json`; use `gzip -cd <path>.gz`. The largest raw CSV uses ordered `.gz.part000`/`.gz.part001` pieces: `cat tracy/layer_0/ops.csv.gz.part* | gzip -cd` restores it. Original text files and large reproducible oracle/Tracy binaries remain local. `log_manifest.json` describes an older checkpoint snapshot and is superseded by the final archive manifest.

## Host-only reference preparation

`tests.long_context --oracle-only --oracle-cache <path> --layer 3 --output <json>` runs the canonical full-context HF reference without importing TTNN. The cache is accepted only when the exact weights, config, Torch/Transformers versions and reference source hashes match. The local `oracle_cache/` tensor file is reproducible and excluded from Git; its compact provenance is retained. Omit `--oracle-only` and reuse the same cache path for the TTNN validation after recovery. Host oracle evidence is explicitly not a TTNN PCC result.

`tests/collect_perf.sh` runs `tests.summarize_perf` only after both real profiler runs produce their reports. The installed tt-perf-report 1.5 CSV uses `Device Time [μs]`, contains operation rows without the terminal total, and separates `Op-to-Op Gap [μs]`. Profiler buffers are drained between complete replays to prevent dropped markers; the gap column therefore includes measurement overhead and is not a throughput measurement. Raw replay-session counts must agree across all 16 passes. The summary reports per-pass device duration and gaps separately, rejects overlaps requiring separate timeline analysis, and divides decode totals by the measured 16 replays. No estimated performance artifact is produced.
