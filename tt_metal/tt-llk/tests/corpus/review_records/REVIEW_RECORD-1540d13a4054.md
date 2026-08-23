# REVIEW_RECORD — pin 25

cc1plus sha256: 1540d13a4054a9263e2d7dc5f8afc58a6a090b161fb3ce3bb30efb70be8100f3
driver (g++) sha256: 774e83d7a3d53d2e000730c47080b60a96cc8993bbabd2bbe62aa7fbd110e31e (unchanged from pin 24)
source: sfpi-gcc nkapre/sfpi 452167b53e7 (merge of lane FV
agent/crosslane-x6 6300d0708b0 off pin-24 tip 92629b12c64; clean merge).
Companion repos: sfpi nkapre/sfpi fb2465c (X6 surface in
sfpi_crosslane.h), tt-metal nkapre/sfpi 1e14d0bdbc (oracle + probe +
sim battery + vehicle pairs). Built in gcc-build-laneFR at the merged
tip (build-pin25.log rc=0); X6 builtin parse smoke OK on the union
driver. Installed via pin-install-fast.sh with loud --expect-cc1plus
verification; no live sweeps at install. sfpi_crosslane.h STAGED into
the install include/ by rm-then-cp from the merged sfpi repo (shas
verified identical); sfpi.h + sfpi_crosslane.h compile-verified on the
installed binary.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

- Lane FV (X6 face-transpose, closing report 2026-08-22): research
  verdict FEASIBLE — Matrix-Unit choreography MOVD2B -> TRNSPSRCB ->
  MOVB2A/MOVB2D -> MOVA2D in the hand-topk_xl three-pass form
  (lo16-park / hi16 / lo16-writeback), SFPU-invariant addr-mod 7, cfg
  block (DISABLE_IMPLIED_SRCA_FMT=1, zero-flag disabled, STALLWAIT
  entry), caller-owned cross-thread bank grants; bit-exact for ALL
  32-bit patterns via host composition theorem from the doc functional
  models. BH doc gap confirmed (BlackholeA0 tree has no pages for the
  family; WormholeB0 pages carry the BH arms; pinned sim = BH oracle).
  Full citations in laneFV-evidence X6-RESEARCH.md.
- sfpi-gcc side: 7 immediate-only VOL builtins (ttmovd2b, ttmovb2a,
  ttmovb2d, ttmova2d, tttrnspsrcb, ttstallwait, ttrmwcib);
  gas-mnemonic emission word-identity proven vs TT_OP encodings; QSR
  expand-time refusals; deliberately effect-UNAUDITED (opaque => every
  optimization layer refuses; xtt_replay barrier). Fail-open gimple
  defaults CLOSED as part of the lane: reassoc
  reassoc-fpu-choreography-boundary, crosscall
  drain-init-ownership-unproven, prgm-const transparency adjudicated
  in-line. rvtt-cost.md entry added.
- sfpi surface: face_transpose_cfg_enter/leave,
  face_transpose_dst_32b<FaceRow>, batch<N,Base>, release_banks; named
  refusals crosslane-facetranspose-unsupported-target / -row-unaligned
  / -toolchain-missing-builtins; BH constants named and
  probe-static_asserted against production headers; __has_builtin
  degradation guard is LOAD-BEARING (pre-X6 compilers parse sfpi.h
  cleanly; X6 use refuses by name).
- Findings filed: X6-F2 (pinned sim gates the MOV-family
  implied-format override on DISABLE_IMPLIED_SRCB where MOVD2B.md says
  SRCA — end-to-end invisible, stage-visible, sim-owner item);
  unconsumed dummy bank grants wedge the next test's unpacker (FS
  persistence class, bank-valid edition — probe modes drain); SrcB
  format contract edge documented in-surface (exact for
  8b-exponent-class SrcBFmt, FP16-class corrupts). FU-F1 adjudicated
  NOT an X6 shape.

## Gates

- Merge gate: clean merges all three repos (no conflicts).
- Union full rvtt.exp (dejagnu-pin25, SFPI env laneFP-sfpi-env, srcdir
  at merged tip): 5734 PASS (= pin-24's 5655 + FV's 79 exactly); FAIL
  set 16 rows LINE-IDENTICAL to the frozen baseline (diff vs
  dejagnu-pin24/fail-set.txt empty).
- Lane FV gates (closing report): dg 79 new PASS; full rvtt.exp 5734
  frozen-identical on its build; corpus byte-gates TRUE-DEFAULT /
  ON-25 / OFF all .text byte-identical 3249/3249 (controls
  non-vacuous: def-vs-ON 407 rows differ); crosslane arsenal 56/56 +
  sortnet sim gate 64/64 on the lane hybrid; X6 sim battery 14/14 on
  the pinned sim. SILICON (BH p150): typed vehicle 2/2 PASS BIT-EXACT
  (Float32 + Int32, 16 tiles); perf TILE_LOOP 289.3 vs hand 293.9
  cyc/tile (-1.55% win); hand controls PASS.
- Install: pin-install-fast sha-verified c944257c78f3... ->
  1540d13a4054...; driver unchanged; include/ staged (rm-then-cp,
  sha-verified); header compile smoke rc=0 on installed binary; no
  live sweeps at install.
- Evidence: ~/sfpi-uplift/laneFV-evidence-20260822/ (SHA256SUMS
  verified), ~/sfpi-uplift/dejagnu-pin25/.
