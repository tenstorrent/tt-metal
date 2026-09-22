# Stage Review

Verdict: clean-pass

Scope: **independent source and build-readiness review only**, within Stage 11 Qwen/Qwen3.8-27B QB2. The isolated candidate is ready for the root-owned build/install procedure. This verdict does not accept the optimization, native correctness, performance, FP8/full64 integration, serving, or Stage 11 completion. The candidate was unbuilt, unapplied and unexecuted throughout this review.

Reviewed live branch: `mvasiljevic/qwen38-full-bringup`, HEAD `a24ad15f9104e1ded223ce9be182fb50084ef087`, with unrelated dirty shared files. All seven affected production native files still matched their archived baseline hashes. All 28 entries in `artifact_manifest.json` matched their recorded hashes. `git apply --check candidate.patch` passed without changing the checkout.

## Required Work

- None before attempting the build within this bounded scope. Native build, execution and promotion gates remain outstanding as described below; their absence is expected for this prebuild review.

## Source Findings

- **Synchronization is coherent by inspection.** In candidate `device/kernels/compute/chunk_gdn_prep.cpp:207`, UNPACK reads after `WAIT`, then writes exactly one decision to MATH and one to PACK. The receiving threads use blocking mailbox reads, following `tt_metal/hw/inc/api/compute/cb_api.h:202`. `CircularBuffer::wait_front/pop_front` operate on UNPACK, while pushes/reservations operate on PACK. The beta/g short circuit at line 433 is safe by this protocol: every thread receives the beta result before deciding whether to call the g helper. The final short circuit at line 540 similarly agrees on all threads. No branch-specific CB/tile work precedes the shared decision.
- **The predicate examines the intended storage.** The prep factory allocates g, beta and scratch CBs as full Float32 tiles; the reader loads g/beta tiles following their head-major reshape. A left-column element at row `r` has face-major word offset `(r / 16) * 512 + (r % 16) * 16`; the tail predicate covers rows 1 through 31, including the lower-left face. The inverse-input predicate covers all 1024 words of tile zero. Masking only bit 31 admits both zero signs and refuses every nonzero bit pattern, including subnormals and nonfinite values.
- **The optimization boundary is narrow.** Existing Q/K normalization, beta multiplication, decay and complete `negN` construction are retained. The new read follows the existing `WAIT(cb_scr3, cc)`. Only zero beta/g tails plus an actually zero computed negN select the eye copy; otherwise the unchanged `invert_block` executes. Both branches push the same Tinv tile and retain the subsequent wait/pop sequence. Neither scan code nor recurrent-state write code changes. The public and primitive flags default false; public validation requires phased flat Q/K/V and exactly T=C=32, and primitive validation requires one normalized flat 32-row chunk.
- **Program selection carries the flag.** The parameter enters `ChunkGdnPrepParams`, the default operation-attribute hash, and compute argument index 6. The primitive defines no custom hash omitting this field. The public declaration, definition, nanobind argument, primitive declaration/definition and factory agree on the added final argument. Existing CMake unity sources include the changed host implementation and binding units; no new source registration is needed.
- **The numerical probe uses the real call path.** `check_identity_inverse.py:364` calls the unchanged `layer._delta`, intercepting only `ttnn.transformer.chunk_gated_delta_rule`. The wrapper opts in for the region arm with logical input shape `(8,1,5120)`, including the method's native batch partitions. B4 leaves the keyword absent. The reconstructed split AST is checked but is not substituted for the measured method. Eleven direct native cases use actual flat input shapes, native constants and the operation itself, and compare complete output/state byte hashes on four ranks. The full-method controls compare BF16 projected output, BF16 convolution history and FP32 recurrent history; traces use independent persistent states and copy each arm's result to host before the other trace is captured.
- **Build/runtime evidence cannot be supplied by host checks alone.** Source validation requires the candidate to be installed, a successful native build receipt tied to `candidate_sources.json`, and hashes for both installed libraries. Runtime additionally checks the Python extension location and the actually mapped `_ttnncpp.so`. The external validator requires the genuine child exit, pinned command/program, complete result shapes and four empty external owner files before timing can begin.

## Other Concerns

- Canonical eye/tril/ones/quadrant constants are an explicit opt-in API precondition, not kernel-validated values. The direct probe checks the layer's canonical constants on all ranks before enabled calls, and the default path remains available for arbitrary constants. Do not broaden opt-in callers without preserving this precondition.
- The public flag is batch-independent; B4 fallback is selected by the probe/model caller, not encoded in the native flag. Any eventual model wiring must preserve the intended B8/t1 selection explicitly.
- The prepared component uses the pinned base checkpoint and selected `head_bfp4_lofi` policy. Its result cannot establish equivalence or benefit for the separately composed full64 FP8 serving path.

## Hard-Check Gaps

These are later execution/promotion gates, not reasons to reject this prebuild package:

- Run and retain the actual mandated build-wrapper outcome. If that fails because Docker is unavailable, the inspected existing Release configuration supports the proposed narrow `ttnn` build and `ttnn-runtime`/`tt_pybinds` install components. Record real build/install commands, exits and installed hashes. No build was attempted by this reviewer. Host compilation will not itself prove the device JIT kernel compiles.
- Run the native finite/nonfinite comparisons, B4 control, all-rank state/output checks and trace comparisons with actual successful external exit/cleanup. Mailbox behavior, pack/unpack behavior and exact signed-zero propagation remain hardware questions despite coherent source.
- The native matrix does not explicitly inject negative-zero tails, subnormal g tails, or change from identity-eligible to fallback inputs within one captured trace. The host word oracle covers the bit predicate, while the prepared traces repeatedly use one frozen input. These are useful focused additions if exact signed-zero or replay branch-transition coverage is needed before broadening the candidate's use; no source defect was inferred solely from their absence.
- `verify_execution.py --mode timing` validates measurement integrity and equal final state, not speedup. Promotion still requires root review of the six paired measurements and a repeatable complete-method benefit. The unmeasured cost of reading up to 1024 L1 words must be included.
- Retain full64 downstream/logit and serving gates after any component promotion. This report provides no waiver for them.

## Anomaly Ledger

- Observed anomaly: No contradictory source or artifact result found within the bounded review.
  Evidence: All 28 hashes matched; seven live sources matched baseline; patch applicability and the host-only preparation controls passed.
  Affected path: Candidate preparation only.
  Control or comparison: Archived baseline versus candidate plus existing compute/mailbox APIs, factory/reader code, actual `_delta` method and validator source.
  Likely subsystem: No defect assigned.
  Investigation performed: Read-only inspection, stdlib artifact checks and the audited host preparation script.
  Resolution: Controlled for build readiness; native behavior remains unverified.

## Scope Inspected

- Goal/skill: Parent's Stage 11 QB2 bounded candidate contract; `.agents/skills/stage-review/SKILL.md`; supplied repository AGENTS.md build requirements.
- Artifacts: Candidate README, patch, source/manifest/bridge files, build preparation, host controls, correctness/timing commands, probe, external validator and AutoFix note.
- Code: All seven changed candidate native files; unchanged prep reader, compute CB/mailbox APIs, operation hashing, model `_delta`, generated unity build and installation rules.
- Commands: Read-only `cat`, `sed`, `rg`, Git status/HEAD and `git apply --check`; stdlib manifest/source hash checks; `/usr/bin/python3 -B .../gdn-identity-candidate/check_host.py` (exit 0, `PASS_HOST_PREPARATION_ONLY`, 6,144 word refusals, 31 tail refusals, 27 external-record refusals).
- No device or owner-file access, target imports, HTTP, compilation, library installation, implementation edits, server/process control, or runtime experiments occurred. This report is the only review-authored file.

## Residual Risk

The device compiler, hardware synchronization, native bit equivalence and net runtime benefit remain untested. Source inspection supports attempting the isolated build; only later real controls can support retaining and integrating this optimization.
