# AutoFix: linear-prefill slice CB overflow

## Starting Evidence

- Diagnosis: `LINEAR_PREFILL_STRESS_AUTOTRIAGE.md`.
- Original failure: TP4 linear-attention prefill S128 under Watcher requested a 130-byte
  row-major slice read into a 128-byte CB page.

## Hypothesis Experiment

- Hypothesis: `compute_cb_size()` omitted the reader's temporary leading misalignment
  from CB page capacity.
- Source proof: the NCRISC reader writes `unpadded_stick_size + misalignment`, while
  the host allocated `round_up(unpadded_row_size_bytes, alignment)`.
- Focused regression: BF16 row-major input width 66, slice `[1:65]`, producing an exact
  128-byte useful row with a 2-byte prefix.
- Verdict: verified.
- Fix: allocate
  `round_up(unpadded_row_size_bytes + misalignment, alignment)`. The semantic merge
  stride and reader/writer useful-row sizes are unchanged.

## Verification

- Build: `cmake --build build --target ttnn -j 12`; install into the runtime used by
  Python: `cmake --install build`.
- Focused Watcher/no-ETH regression: PASS, exact values, clean Watcher.
  Log: `logs/autofix_slice_cb_exact_boundary_installed_watcher.log`.
- Existing 5-D misaligned slice regression: PASS, PCC threshold met, clean Watcher.
  Log: `logs/autofix_slice_existing_misaligned_watcher.log`.
- Original TP4 command (`linear`, S128, warmup 4, iterations 16, default): PASS,
  PCC 0.999994331685145, replicas equal, fallback audit true, median 671.418645 ms,
  clean Watcher. Log: `logs/autofix_linear_prefill_s128_16_watcher_installed.log`;
  JSON: `artifacts/final/autofix_linear_prefill_s128_watcher.json`.
- Full-attention control (`full`, S128, warmup 4, iterations 16, default): PASS,
  PCC 0.9999945998163279, replicas equal, fallback audit true, median 19.562482 ms,
  clean Watcher. Log: `logs/autofix_full_prefill_s128_16_watcher.log`; JSON:
  `artifacts/final/autofix_full_prefill_s128_watcher.json`.

The first model rerun after compiling still loaded the stale installed
`build_Release/lib/_ttnncpp.so` and reproduced the original error. After bounded abort,
`tt-smi -r && tt-smi -s` recovered all four devices, and `cmake --install build`
installed the rebuilt library. All final results above use that installed runtime.

## Final Status

Fixed with focused, original-command, nearby-slice, and full-prefill Watcher evidence.
No unrelated implementation changes and no commit were made by this AutoFix task.
