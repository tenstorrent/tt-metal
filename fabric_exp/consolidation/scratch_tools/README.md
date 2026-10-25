# Scratch tools (Phase 1, kept for the record)

Scripts the consolidation agents and I wrote in the session scratchpad
(`/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/<dir>/`), copied here so
they survive the scratchpad. One directory per scratch directory (`top` = the scratchpad root). They are not
maintained: most have that scratch path, its build directories and the Phase 0 kernel cache hard-coded.

Most useful:
- `v3t/`: the E1 insulation work (V3E1_NOTES §11).
  - `stability.sh`, `stab.sh`, `derive_edits.py`: the six-edit stability test.
  - `fncmp.py`: compares the main-loop region of two ELFs.
  - `mults.py`: lists constant multiplies in `run_main_loop`'s optimized GIMPLE.
  - `rtlfirst.py`, `rtlcmp.py`: find the first RTL pass where two builds differ.
  - `rxscen.py`, `walk.py`, `tab.sh`: per-path instruction counts.
- `v3l/loadcmp.py`: PT_LOAD comparison of two build trees (exact-composition checks).
- `ka13/`: the keepalive host-setting checks (RESULTS §95). `mock_ka.py` runs CPU-only mock runs without the study
  override.
- `shtest/`: local tests of `vc1_capture_lib.sh`, with a fake ttrun.
