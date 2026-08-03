# Phi-3.5 Mini functional decoder

This stage implements one dense `Phi3DecoderLayer` in
`tt/functional_decoder.py`. Prefill accepts `[1, batch, S, 3072]`; decode
accepts `[1, 1, batch, 3072]`. Both use a 32-token paged KV cache and an int32
page table. Decode also takes an on-device int32 `current_positions[batch]`;
short- and long-RoPE traces are captured separately at the 4096-token
transition. Runtime forwards contain TTNN operations only.

## Correctness and coverage

The functional acceptance threshold is PCC >= 0.995. The pre-review complete
test log is `functional_decoder_tests_final.log`; review-remediation and final
integrated logs are recorded in `work_log.md`.

- Synthetic prefill PCC is approximately 0.99997 at page/tile boundary lengths
  31/32/33 and 63/64/65.
- Synthetic decode at position 33: PCC 0.999973.
- Real layer-0 weights: prefill-33 PCC 0.999991 and decode-33 PCC 0.999995.
- Real-weight decode at logical context 131072: PCC 0.999988.
- Fully on-device prefill executes at non-aligned lengths 32769 and 131071 and
  exactly 131072. The target-stat-derived nonzero 32769 last-token oracle
  passes PCC 0.999865.
- Traced decode replay is bitwise deterministic and equals a fresh-cache eager
  control at batch 1 and serving batch 32; replay/reference PCC is measured at
  position 33 at batch 1 and distinct positions 1..32 at batch 32; a long-RoPE
  trace is checked at position 4096.
- The static runtime audit rejects Torch conversion, collectives, and host
  fallback in the measured forwards.

For sequences above 32768, the implementation uses ordinary causal SDPA for
the first bounded prefix and 128-query chunks with a TTNN-generated
absolute-position mask thereafter. This avoids the non-chunked SDPA correctness
cliff without runtime program or grid tuning. A tail-SDPA-only
HiFi4/exact-math/FP32-accumulation override is retained because isolated
AutoFix evidence showed default PCC 0.992089 and the minimal override passes at
0.995360. The only L1 layouts are workload-derived requirements of decode
cache update, decode SDPA, and decode head concatenation.

## Performance

On one Blackhole p300c device, the Tracy run recorded:

| Path | Workload | Warmed host latency |
| --- | --- | ---: |
| Prefill | batch 1, sequence 128 | 2.110 ms |
| Traced decode | batch 1, context 128 | 1.098 ms mean / 1.093 ms min |
| Traced decode | batch 32, context 128 | 1.225 ms mean / 1.223 ms min |

Final-runtime operation provenance is in `tracy_final/ops.csv`. Human-readable `tt-perf-report`
tables and derived CSVs are `tracy_final/prefill.{txt,csv}`,
`tracy_final/decode_b1.{txt,csv}`, and
`tracy_final/decode_b32.{txt,csv}`. The exact Tracy
command output is `tracy_final/profile_console.log`.

## Watcher and commands

The watcher-enabled real-weight and traced batch-1/batch-32 run passed three
tests with no watcher error:

```bash
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=$PWD/models/autoports/microsoft_phi_3_5_mini_instruct/doc/functional_decoder/watcher_final \
pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py \
  -k 'real_weight_paged_prefill_and_decode or decode_trace_replay_is_deterministic'
```

See `watcher_final/watcher_console.log` and
`watcher_final/generated/watcher/watcher.log`. The real weight source was the
official cached Hugging Face safetensors snapshot
`2fe192450127e6a83f7441aef6e3ca586c338b77`.

AutoDebug and AutoFix evidence, including the trace capture-state analysis,
batch-32 rectangular head-concat layout fix, and long-prefill repair, is in
`AUTODEBUG.md` and `AUTOFIX.md`.

## Capability matrix

| Claim | Evidence | Remaining risk |
| --- | --- | --- |
| Paged prefill, batch 1 and 2 | PCC tests with distinct inputs and permuted page rows | Functional, not optimized |
| Advertised 131072 prefill | Exact and 131071 non-aligned device tests | Full-output long PCC is sampled at the critical 32769 tail |
| Decode context 131072 | Real-weight PCC at position 131071 | Batch-32 full-context cache is outside the tested serving workload |
| Traced decode batch 1/32 | Replay/reference PCC, exact repeatability, and separate passing short/long traces | None in the functional contract |
| No runtime host fallback | Static audit plus code inspection | Setup and test oracles intentionally use Torch |
