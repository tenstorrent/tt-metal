# #134 notes: SDPA chain cherry-picks + LTX_SDPA_MM_LOFI (off-device only)

Branch `ttp/t134-cherry-pick-sdpa-chain-57979-58032-58223`, base t48 c4409b1fa24.
(The root NOTES.md is t48's; this file holds #134's notes.)

## Done
- a0954fcb8fe #57979, 619a15b1bf5 #58032, 0604cf84205 #58223: no textual conflicts. compute_common.hpp
  (t48's `mask_subblock_stride` param on matmul_blocks vs upstream's sdpa_max_sfpi/max_block_sfpi/zero_block
  helpers), transformer_nanobind.cpp (t48's neighborhood SDPA binding vs the extended SDPAProgramConfig
  binding) and sources.cmake merged side by side; checked by hand, no name clashes.
- e2d55644e28: `LTX_SDPA_MM_LOFI=1` sets `matmul_math_fidelity=LoFi` on all five SDPA configs in
  attention_ltx.py. Only ring_joint SDPA reads the field (local SDPA ignores it). Unset passes no kwarg:
  field nullopt, no `SDPA_MATMUL_FIDELITY` define, kernels as before; the new merge helpers only run with
  max_k_splits>1 or segmented accumulation (both off by default).
- Local Release build rc=0 (build.log). CPU: test_transformer_ltx.py -k sdpa 10 passed (3 new knob cases;
  the LoFi case fails against the pre-knob attention_ltx.py); tests/unit/test_ltx_*.py 92 passed,
  2 skipped; VAE ref + conv3d halo cpu + denoise trims 79 passed; test_fold_gate_layout.py 10 passed.
- e146b42f0e5: blx03 2x4 A/B harness (test_sdpa_mm_lofi_ab.py, run134.sh, driver.sh, setup134.sh,
  READY.md). Collect-only and helper dry tests pass. NOT run.

- Branch pushed at e146b42f0e (push checks: 92 passed + 2 skipped, 79 passed).
- Static check of the shared kernel headers: the cherry-picks only add helpers, add trailing defaulted
  params (sdpa_ring_v2 k_split_begin/end) or swap internal matmul calls for the sdpa_mm_* wrappers, so
  local sdpa.cpp, exp_ring_joint_sdpa.cpp, sparse_sdpa_msa_compute.cpp and neighborhood SDPA keep compiling
  as far as reading the code shows.

## blx03 build (off-device), launched 2026-10-05 05:14 UTC
`~/fasth3/t134-setup.sh` (copy of setup134.sh) ran as pid 3188016: worktree ~/fasth3/t134 (of ~/fasth3/tt-metal)
at e146b42f0e, lean Release build. `~/fasth3/t134-setup.log` ends with `SETUP134_DONE rc=<n>`:
0 = ready to launch; 11 fetch, 12/13 checkout, 14 submodules, 16 build, 17 no _ttnn.so.
The REF tree ~/fasth3/t48 is at 64571a953b2 (base's parent; no diff in ttnn, tt_metal or the LTX model
paths), built Oct 3.

## Not verified (needs device)
JIT compile of the cherry-picked kernels, knob-off bit-identity on device, LoFi speed and quality.

## Next
1. Check `ssh g14blx03 'grep SETUP134_DONE ~/fasth3/t134-setup.log'`: rc=0 means ready; else fix, re-run.
2. Launch per READY.md only after the user lifts the device STOP-ALL.
