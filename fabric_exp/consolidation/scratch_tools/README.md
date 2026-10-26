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
- `e0audit/`: the E0 audit of 2026-10-26 (`E0_AUDIT.md`).
  - `emu2.py` (v3l's `emu.py` plus first-level-ack counters, `--seq` and the credit-frame-to-send lag), `paths.py` (per-pass
    table over the five hot E0 configs), `e0hot.sh` / `e0cmp.sh` (hot E0 builds; `sender_erisc` function comparison),
    `rml.py` (compact listing of `run_main_loop`), `full.sh` / `fullv.sh` (all-config builds), `cfgs.py` (E0 config
    classes from the named CT args).
  - `mk_v3y.py` (`--api-only` for the API-only control), `mk_cpp.py`, `mk_v3yb.py` (the worker room-first change, nested
    form), `mk_v3yb_allch.py` and `mk_v3yb_formC.py` (the rejected all-channel forms), `probes.py` (the two per-round
    edits).
