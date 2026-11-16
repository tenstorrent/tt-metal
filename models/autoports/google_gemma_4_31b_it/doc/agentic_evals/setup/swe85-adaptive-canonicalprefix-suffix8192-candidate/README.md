# Canonical-prefix suffix8192 candidate (not admitted)

This host-only package prepares six diagnostic jobs: the exact last 8192 IDs from the original canonical-prefix `call91` source, served and full modes, seeds 42/43/44, chosen 64-token teacher forcing, and one 512-token free run per job. The prompt IDs, chosen continuation, sampling, and seed are held fixed across each matched mode pair. The candidate is intentionally separate from precision panel P and is not admitted or run.

The source native full run showed the duplicated-ve contraction motif. V's 4096-ID full suffix produced 39/148/264 free-run tokens for seeds 42/43/44 and did not show the adjacent duplicate-ve motif; the 8192 test extends the same original prompt without padding or re-rendering. This is diagnostic context exploration, not native replay or a task-score test.

`forecast.json` gives three planning scenarios from V actual per-case results. The low scenario reuses V measured component times and stop lengths, the point doubles chosen-TF and free prefill/first-token residual while leaving traced decode time unchanged, and the upper scenario also doubles decode time per replay and fills each 512-token cap. These are workload scenarios, not confidence limits; the prefix may change sampled outputs and decode behavior.

`command.json` contains timeout-bounded serial commands for use only after root admission. No device commands were run while preparing this package. CPU validation is recorded separately.

## Admission and P predecessor gate

`run_after_p.py` is inert until root explicitly updates `admission.json` to `status: admitted` with the pinned proposal and driver hashes. It waits for P at `setup/swe85-precision-bisect/head-bfp8-hifi2-first-panel-after-w/` to reach queue `completed`/exit 0 and completion exit 0 under admission SHA `91dc764e6a3a6b1e8a1be51e6eee5fc958c19d1dfcfa3d057b989aa325618cdd`. It rechecks P's admission, proposal, driver, full source inventory and terminal queue/completion/launch hashes before each mode.

Before each timeout-1800 mode, the driver also requires no `precision_repro_hf.py` process and zero device owners. It stops after an error and does not retry. The P completion's `elapsed_seconds` must fall inside the frozen forecast range in P's launch record. If it misses, the driver refuses both suffix8192 modes unless root creates `P/forecast-miss-accepted.json` with exactly these fields: `status: root_accepted_forecast_miss`, `reviewer: root`, `accepted: true`, `completion_sha256` (the exact terminal P completion file hash), `measured_seconds` (exact `elapsed_seconds`), `forecast_range_seconds` (the exact two values from the P launch), and `admission_sha256` (the pinned P admission hash). The record is not created by this candidate.

The wrapper self-test (`run_after_p.py --self-test`) checks range-hit, missing-miss-record, accepted-miss, and mismatched-record behavior without importing TTNN or touching predecessor files. No run or admission was performed during preparation.
