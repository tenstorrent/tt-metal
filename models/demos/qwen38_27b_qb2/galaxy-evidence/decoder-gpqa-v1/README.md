# Full GPQA with BFP8 decoder weights

Started **Oct 9, 09:28:02 UTC**, after the completed decoder reference controls.
This experiment is running, not qualified. The previous head-only control
finished at 166/198; native BFP4 at the same output budget finished at 170/198.

The candidate changes all decoder projections to BFP8/HiFi2, retaining the
BFP8/HiFi2 head, BFP8 KV, FP32 recurrent state, native recurrence and accurate
full-tile attention. Eight-replica G0 precedes the serving evaluation. The
GPQA protocol remains all 198 questions, temperature 1, top-p 0.95, top-k 20,
seed 42, thinking enabled, 65,536 output-token budget, 262,144 model context,
128 client concurrency. The gate remains 177/198. No questions are removed.

- Unit: `qwen38-accuracy-decoder-v1-20261009.service`
- Observed PID: `1449553`
- Invocation: `ecb6408f79414ed0974c6352f006bce5`
- Source: `/home/ttuser/qwen38-artifacts-20261007/accuracy-decoder-source-v1`
- Controller/results: `accuracy-decoder-control-v1` / `accuracy-decoder-v1`
  under the same artifact root.
- Source-manifest SHA256:
  `73ee67ce619698028df06677f0b52fd9dd2d6f89d4523aafe3320ad58ede80e3`
- Precision config: `precision_accurate_decode_bfp8_all.json`.

Preflight passed 423 tests and 40 subtests, with one expected hardware skip.
The source is frozen separately from the worktree. The controller requires
both eight-step numerical controls to complete and clean up, checks the exact
predecessor identity, then uses the shared device lock. Eight-hour systemd
bound, 256 GiB host RAM, sixteen CPU quota; survives disconnect, not reboot.
No running model source is patched. This policy has not been promoted.

The completion auditor is queued separately as
`qwen38-decoder-gpqa-audit-v2-20261009.service`. It waits for this exact
invocation to finish and release workers, checks all raw-response hashes, and
retains all 198 questions in its strict completion-qualified denominator.
It has a nine-hour bound and passed 22 dedicated unit tests. The first audit
staging attempt assumed the helper existed in the frozen model snapshot; it
stopped before creating any directory or service. The corrected staging uses
a separate, tested frozen audit bundle; it does not alter the live runtime.

BFP8 adds an estimated 5.81 GiB of resident weights per chip in the serving
layout. Full serving admission and throughput remain to be measured. The
short numerical comparison is a reason to run this test, not a predicted score.
