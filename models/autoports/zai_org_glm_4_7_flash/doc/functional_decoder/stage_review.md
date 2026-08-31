# Stage review record — functional-decoder, zai-org/GLM-4.7-Flash

Reviewer: fresh independent subagent (xhigh) in $stage-review reviewer mode,
read-only, agent id aa8a6068a12305827.

## Round 1 verdict: more-work-needed

Required work (both P2):
1. Watcher evidence was stale relative to the final implementation (predated
   the chunked-prefill fp32-accumulator fix; the fixed prefill flash kernel
   had no watcher coverage).
2. README numbers (bf4 arm, dense prefill perf) came from pre-fix runs and
   disagreed with the final regenerated logs it cites.

Other concerns raised: tie rows exempted without flip-reconstruction proof in
the 202k windows; "4 ulp" terminology (actually 2 conventional bf16 ULPs);
latent slice-aliasing deallocs (prefill cos/sin table slices, sparsity_e);
aligned S=202752 never run; fp32-acc A/B repro not preserved in-tree; stale
docstring JSON names; traced decode only at batch 1.

## Remediation (work_log.md "Stage review round 1" entry)

Watcher rerun on final code (TT_METAL_WATCHER=2, now covering the traced
decode test; 5 passed, 20 dumps, 0 suspicious lines); README refreshed from
final artifacts; tie-bypass removed from the 202k window gate (every below-bar
moe row must pass alternate-top-4 reconstruction); ulp docs corrected;
aliasing deallocs removed; test_full_context_aligned_202752 added and passed;
repro preserved as tests/probe_fp32acc_drift.py; suite + 202k reruns all green
on the final code.

## Round 2 verdict: clean-pass

No required work. Reviewer independently re-derived: fresh watcher tree
postdating all code with zero error patterns; README numbers match logs/JSON
to full precision; dealloc-only change proven numerically inert (bit-identical
PCCs); the strengthened routing-flip proof machinery "strictly stronger than
at first review". Non-gating notes (both fixed post-review, no evidence
regeneration needed since neither was exercised by the recorded runs):
probe_fp32acc_drift.py knob renamed to ck_flash_prefill to stay a live repro;
decode-at-max-position branch aligned with the window rule (tie annotation,
not bypass).

Full round-1 and round-2 reports are in the runner transcript for this stage.
