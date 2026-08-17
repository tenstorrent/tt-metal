# Triage artifacts

`tt-triage.txt` is **empty on purpose**, and that emptiness is the evidence.

At 2026-08-17 15:37 the 4th Tracy perf case aborted inside device-profiler post-processing
(`TT_FATAL: Start and end marker IDs do not match.`, `profiler.cpp:2104`) and left
`tracy-capture` hung. `$tt-device-usage` says to collect triage before killing a hung device
job, so this was run first:

```bash
timeout 180 tools/tt-triage.py --llm-output \
  --llm-output-path models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/triage/tt-triage.txt \
  --triage-summary-path models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder/triage/triage-summary.txt
```

It wrote zero bytes and made no progress inside its bound — it blocks on the device lock the
aborted process still held — so it was killed, which is the escape the skill allows ("unless
... triage itself hangs"). The file is kept rather than deleted so the order of operations is
auditable: triage attempted, triage hung, then the stale pids were killed by explicit pid,
then `tt-smi -ls --local` (4/4 chips) and a 1x1 mesh open/close smoke (`MESH_SMOKE_OK`)
confirmed health with **no reset**.

Full incident write-up, including the cause and the mitigation: `../work_log.md` §6, entry
"2026-08-17 ~15:34-15:40 — device-profiler marker mismatch aborted one Tracy case".

**Second occurrence (2026-08-17 ~21:24), same outcome.** `tt-triage-perf-hang.txt` is also 0 bytes,
captured while a `tracy-capture` left over from the second device-profiler abort still held the
device (`timeout 180` -> rc 124). Same explanation as above and the same conclusion: while a hung
profiler process owns the device lock, `tt-triage` cannot get far enough to write a report, so this
stage has no triage capture for its one recurring hardware-adjacent failure mode. Both incidents,
with the exact signatures and the recovery steps, are in `work_log.md` section 6.
