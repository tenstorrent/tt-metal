# dev/ — sparse ring SDPA dev tooling

Lives only on the `-dev` branch; never merged into the clean branch.

- `run.sh` — timeout-wrapped test steps (`smoke`, `check`, `prod`, `dense`, `perf`, `profile`; no arg = correctness steps); logs to `log_val_<step>.txt`.
- `profile_summary.py` — turns `profile_log_device.csv` into a short text summary (zones per RISC, per-core tables, perf counters).
- `capture_masks.py` — runs the Sol pre-pass on a `sosborne/sol-attn-cleanup` checkout and saves the mask + pooled K/V `.pt` the perf benchmark loads.

Branch workflow:
- Kernel / op fixes are committed on the clean branch.
- The dev branch is rebased onto the clean branch and carries `dev/` plus temporary instrumentation.
- Temporary instrumentation goes in commits titled `DEBUG: ...`, dropped before anything moves to the clean branch.
  Example: a `DEBUG:` commit that adds `defines["SPARSE_STEP_PROFILE"] = "1"` to the reader/writer/compute kernels in
  `sparse_ring_sdpa_program_factory.cpp`, which turns on the profile zones that `run.sh profile` summarizes.
