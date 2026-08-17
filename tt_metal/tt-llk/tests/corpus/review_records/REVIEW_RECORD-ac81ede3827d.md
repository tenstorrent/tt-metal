# REVIEW_RECORD-ac81ede3827d — silicon authorization for the pin-cycle-5 build

Pin: cc1plus sha256 `ac81ede3827d9309f47d3fa6e53a4f530fbdeeee67a48b1d32af74646c66684c`
Built from: sfpi-gcc `69a885b9f67` via sfpi `b755eb3` (`scripts/build.sh`; log `~/sfpi-uplift/toolchain-rebuild-pin9.log`; stage2 stamp removed before the build and the installed cc1plus sha VERIFIED CHANGED from pin-8 `7e9480e7371b…` — D6 discipline)
Date: `2026-08-17`
Reviewer: orchestrator session (Claude, session a64aef93, operated by nkapre@tenstorrent.com)
Independence: NOT independent — the reviewer is the same orchestrator that spawned lanes AT/AX/AS and merged their branches. Each lane ran adversarially-designed gates (renamed/varied/near-miss twins, corpus byte-identity, frozen-FAIL DejaGnu comparison), and the orchestrator re-verified evidence manifests, charter greps, and diff scopes at merge time, but no human or independent session has re-reviewed the union. An independent re-review should supersede this record.

## Reviewed commits/branches
- sfpi-gcc `69a885b9f67` = merge of, on base d8aaa2caa61 (AR LaneConfig audit):
  - AT `b798bca1b19` agent/where-prefix-elision (WP11 cross-tile config-epoch prefix elision)
  - AX `8df85f7798d` agent/mop-outward-ownership (mop-caller-template-live-unproven refusal)
  - AS `19f8248e129` agent/capture-rotation-fill (seam fill + prologue rotation, default-off)
- sfpi superproject `b755eb3` (gcc submodule pin advance)
- tt-metal sweep wiring: ON set adds capture-rotation and the M3 fire pair
  (-mtt-tensix-optimize-prgm-const + -DLLK_ENABLE_TTREGION_MARKERS); OFF set pins
  both off; knobs added for both (prgm-const knob carries the marker define).

## Gates checked
- Union DejaGnu (this build tree at 69a885b9f67, full rvtt.exp, fresh xg++/xgcc):
  2612 PASS; unexpected-FAIL set == the frozen environmental 15 MINUS
  41863-consteval.C (now passes) — zero new failures
  (`~/sfpi-uplift/laneAA-union-dejagnu2.log`, g++.sum in gcc-build-laneAA).
- Per-lane gates at merge review (each on its own base, all green):
  AT — 48/48 kernel dumps, CRAQ 96/96 (corrected sims), byte-identity =
  exactly the intended where impl-1 elision, corpus 3154/3154, +5 dg tests;
  AX — force-leg byte-identical to normal-ON on the hang TU, corpus 3153/3153
  flags-off identity, CRAQ 4/4, +3 dg tests;
  AS — dg 22/22, corpus 58-row byte-identity on 6 legs incl. NEWFLAG, CRAQ
  PASS both legs (exp/sdpa/lerp), frozen-15 FAIL set byte-identical.
- Evidence manifests: laneAT/laneAX/laneAS evidence dirs SHA256SUMS all verify (0 mismatches).
- Charter greps on the three new/changed passes: no op names, calendar words,
  or coefficient fingerprints in decision logic.
- Installed-toolchain sanity: new flags accepted; removed WP8 exact-calendar
  flags still error by name.
- NOT checked: no pin-9 silicon exists yet (this record authorizes the first
  run); the M3 fire's first engagement is CRAQ-gated inside that run; AS's
  prologue-rotation first real-kernel fire requires multi-tile CRAQ before any
  ON-set reliance (currently fires nowhere in shipped kernels).
