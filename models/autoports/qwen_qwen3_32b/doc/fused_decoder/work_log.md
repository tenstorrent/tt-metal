# Fused-decoder work log

## Scope and starting point

- Repository: `/home/mvasiljevic/tt-metal`
- Branch: `mvasiljevic/models/v2/qwen-qwen3-32b`
- Starting HEAD: `885f2ca1833`
- Functional stage checkpoint: `606ab816a33692f0b1f0c22105dadc235b37ba12`
- Model: `Qwen/Qwen3-32B`
- Stage ownership: `tt/fused_decoder.py`, its tests, and
  `doc/fused_decoder/`; `doc/context_contract.json` was read but not changed

No optimized-decoder, multichip-decoder, full-model, generator, or vLLM work
was started.

## Device procedure

`timeout 60 tt-smi -ls --local` listed four Blackhole p300c devices. A 1x1
mesh open/close smoke passed with the required paired visibility
`TT_VISIBLE_DEVICES=2,3`. All hardware commands were serialized. Watcher and
Tracy/device-profiler runs were separate, and no reset was needed.
The final bounded `tt-smi -ls --local` again listed all four p300c boards.

Pytest shutdown emits known nanobind reference-leak diagnostics after a clean
device close. Every bounded command exited, subsequent mesh opens passed, and
no model hang or device-health failure occurred.

## Baseline and implementation sequence

The Stage 01 suite was rerun first:

```bash
TT_LOGGER_LEVEL=fatal TT_VISIBLE_DEVICES=2,3 \
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
timeout 300 pytest -q -s \
  models/autoports/qwen_qwen3_32b/tests/test_functional_decoder.py
```

Result: `3 passed in 16.06s`. Representative real layer-32 PCC was 0.998887
prefill and 0.998576 decode.

The functional source and profiler topology were scanned against every
`graph-fusing` category. Candidate classes in
`tests/fused_decoder_candidates.py` isolate cache update, SiLU folding,
packed gate/up, warmed metadata, decode input/output view sequences, and
sharded Q/K norm/RoPE. Each candidate uses real layer-32 weights and the same
BF16 shapes.

The sharded Q/K candidate initially produced decode PCC 0.600446. A focused
intermediate probe showed Q/K norm and Q/K RoPE PCC of 0.999994, 0.999994,
0.999993, and 0.999993. An explicit SDPA program config still produced
0.613319, localizing the contract boundary to sharded-Q decode SDPA. Moving Q
back to interleaved DRAM restored 0.998694 PCC; the corrected candidate was
then measured and rejected as slower.

The winning post-concat sequence was confirmed with 200 replays. Cache and
metadata interactions were then checked with 500 replays:

```bash
TT_LOGGER_LEVEL=fatal TT_VISIBLE_DEVICES=2,3 \
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
QWEN3_32B_FUSED_PREFILL_ITERATIONS=11 \
QWEN3_32B_FUSED_DECODE_ITERATIONS=500 \
timeout 1200 python \
  models/autoports/qwen_qwen3_32b/tests/fused_decoder_candidates.py \
  --candidates legacy_concat_heads,legacy_concat_separate_cache,legacy_concat_uncached_metadata \
  --output models/autoports/qwen_qwen3_32b/doc/fused_decoder/results/candidates_final_interactions.json
```

The fused-cache/cached-metadata winner measured 81.7700 ms; separate cache
updates measured 81.7902 ms and uncached metadata measured 81.7928 ms.

The final MLP comparison used 200 replays. Final fused was 81.7677 ms,
standalone SiLU 82.0975 ms, packed gate/up 82.5056 ms, and functional
82.0990 ms. All decode PCC values were 0.998694.

After the first independent review, the formerly inlined direct concat rewrite
was restored as the checked-in `direct_concat_view` candidate. A fresh
500-replay same-process run measured 81.76745 ms final versus 81.82055 ms
direct, with identical 0.998694 PCC. The exact samples are in
`results/candidates_concat_final.json`.

## Correctness, determinism, context, and watcher gates

Final suite command:

```bash
TT_LOGGER_LEVEL=fatal TT_VISIBLE_DEVICES=2,3 \
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
pytest -q -s models/autoports/qwen_qwen3_32b/tests/test_fused_decoder.py
```

Result after review remediation: `7 passed, 2 skipped in 39.90s`; the two skips are the explicit
capacity and performance gates run separately. The suite covers non-aligned
S=3/17/33, paged cache append, two-run determinism, first/middle/last real
weights, advancing repeated decode, complete initialized cache prefixes, and
ten bitwise-equal trace replays. It also warms every decode position 0-127 and
repeated prefill tile classes; DRAM stays exactly flat at 980,276,736 and
976,015,360 bytes respectively. `logs/final_suite.log` is the raw output.

Layer 63 produced below-0.995 Hugging Face PCC in both functional and fused:
0.989861 prefill and 0.993443 decode. The test measured functional and fused
in the same process and got fused-to-functional PCC 1.0 for every real layer;
this is an inherited functional limitation, not a fusion delta.

Context preservation command:

```bash
TT_LOGGER_LEVEL=fatal TT_VISIBLE_DEVICES=2,3 \
QWEN3_32B_FUSED_CONTEXT_PROBE_LEN=4096 timeout 900 pytest -q -s \
  models/autoports/qwen_qwen3_32b/tests/test_fused_decoder.py::test_fused_preserves_functional_context_capacity
```

Result: pass, output `[1, 32, 4096, 5120]`. No layout/dtype/capacity contract
changed, so `doc/context_contract.json` remains the authoritative 4,096-pass,
4,097-DRAM-OOM boundary. The post-remediation raw output is retained in
`logs/context_4096.log`.

Watcher command:

```bash
TT_LOGGER_LEVEL=fatal TT_VISIBLE_DEVICES=2,3 \
TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
pytest -q -s \
  models/autoports/qwen_qwen3_32b/tests/test_fused_decoder.py::test_repeated_and_traced_decode_preserve_full_cache
```

Result: pass in 22.47s. `logs/watcher_device.log.gz` (lossless gzip) ends in clean device detach
and contains no NoC error, assert, stuck-kernel, exception, or fatal
signature. `logs/watcher_correctness.log` preserves the test output.

## Final performance and profiler evidence

Final same-process gate:

```bash
TT_LOGGER_LEVEL=fatal TT_VISIBLE_DEVICES=2,3 \
QWEN3_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
QWEN3_32B_RUN_FUSED_PERF=1 \
QWEN3_32B_FUSED_PREFILL_ITERATIONS=11 \
QWEN3_32B_FUSED_DECODE_ITERATIONS=500 timeout 1200 pytest -q -s \
  models/autoports/qwen_qwen3_32b/tests/test_fused_decoder.py::test_fused_beats_functional_warmed_prefill_and_traced_decode
```

Result: pass. Functional/fused warmed prefill medians were
83.3613/83.0103 ms; traced decode means were 82.1000/81.7882 ms. PCC was
unchanged at 0.998839 prefill and 0.998694 decode. The exact samples are in
`results/before_after.json`, and raw output is in `logs/performance_gate.log`.

Profiler commands, run without watcher:

```bash
TT_VISIBLE_DEVICES=2,3 QWEN3_32B_REAL_WEIGHT_DIR=<snapshot> \
python -m tracy -r -p -v -o <prefill-report-dir> \
  models/autoports/qwen_qwen3_32b/tests/fused_decoder_profile.py --path prefill

TT_VISIBLE_DEVICES=2,3 QWEN3_32B_REAL_WEIGHT_DIR=<snapshot> \
python -m tracy -r -p -v -o <decode-report-dir> \
  models/autoports/qwen_qwen3_32b/tests/fused_decoder_profile.py --path decode
```

The raw device CSVs were delimited with `FUSED_PREFILL`/`_END` and
`FUSED_TRACED_DECODE`/`_END`, then analyzed with `tt-perf-report --no-host-ops
--raw-op-codes`. Prefill has 147 operations and 82.850 ms including gaps;
decode has 26 operations and 81.776 ms including gaps. Exact raw and reduced
CSVs plus summary PNGs are under `perf/`.

## Artifact index

- `results/before_after.json`
- `results/candidates_concat_final.json`
- `results/candidates_mlp_final.json`
- `results/candidates_decode_confirm.json`
- `results/candidates_final_interactions.json`
- `results/sharded_qk_norm_probe.json`
- `results/sharded_qk_norm_sdpa.json`
- `results/sharded_qk_norm_adapted.json`
- `perf/prefill_ops_perf_results.csv`
- `perf/decode_ops_perf_results.csv`
- `perf/prefill_tt_perf_report.csv`
- `perf/decode_tt_perf_report.csv`
- `perf/prefill_tt_perf_summary.csv` and PNG
- `perf/decode_tt_perf_summary.csv` and PNG
- `logs/final_suite.log`
- `logs/context_4096.log`
- `logs/performance_gate.log`
- `logs/watcher_correctness.log`
- `logs/watcher_device.log.gz` (lossless gzip of the raw device log)

## Review and checkpoints

The first independent reviewer returned `more-work-needed` with three
findings: unbounded DRAM view dictionaries, a no-longer-reproducible direct
concat candidate, and no retained fused 4,096 log. Remediation replaced both
dictionaries with explicit single-entry caches that deallocate replaced
tensors, added the allocation-stability stress, restored and reran the direct
candidate, retained `logs/context_4096.log`, and reran correctness,
performance, capacity, and watcher gates. A fresh rereview follows. Nothing
is pushed.

The different fresh reviewer `/root/stage_rereview_qwen3_32b_fused` returned
`clean-pass` with no required work. It independently reconciled the bounded
allocator evidence, current-source direct candidate, post-fix context log,
performance JSON/log, final suite, watcher/device log, raw/reduced profiler
CSVs, graph-pattern audit, and strict stage scope. The full review ledger is
in `stage_review.md`.

Stage implementation and evidence checkpoint: `4bceff06006` (`Add fused
Qwen3-32B decoder stage`). This is a local commit; it was not pushed.
