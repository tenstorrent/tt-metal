# Blaze SFPLOADMACRO transplant experiment (2026-08-17, p150a) — NEGATIVE RESULT

Owner-directed: "try the blaze SFPLOADMACRO locally first before considering any uplift."

Method: applied our tt-llk commit 725748766a6 (SFPLOADMACRO-scheduled unfused
merge/rebuild, 427 lines) to blaze's forked ckernel_sfpu_topk_xl.h at
~/tt-blaze (11/13 hunks fuzzy-clean; 2 hand-applied: topk_mop_config body_len
gate + the merge preamble the fuzzy patch scrambled). A/B on their own GLM
indexer bench (the comp4 blaze arm, k=2048 W=65536, sweep --with-blaze):

  base  24,355 ns
  macro 24,361 ns   -> +6 ns = ZERO

Compile-path verification (sabotage probe): an #error inside the macro branch
of merge<fused=false> KILLED their kernel JIT at the transplanted line -> the
header IS their JIT source and the branch IS instantiated. The transplant ran;
it just doesn't matter: their indexer_local_topk tree is FUSED merge/rebuild
(op.hpp:355-401); the unfused region exists only past the bank boundary
(op.hpp:564-633) over 8,192 valid positions — share ~0 of the 24.4 us.

Verdict: Front D's transplant estimate (topk phase 14.4 -> ~11-12 us) is
REFUTED for the flagship decode cell. Do NOT pitch the macro to blaze on perf
grounds for this program. Residual possibilities (unmeasured): their
distributed_topk / cross_device_allgather_tree_merge programs call
merge/rebuild<fused=false> in anger and might benefit; the #1971 LLK
unification case stands on maintenance grounds alone, not speed.

Also observed: their bench pytest swallows kernel JIT failures (rc=0 with no
device rows) — a blaze harness bug worth mentioning in #1971.

Their checkout restored pristine (git checkout of the header + .orig removed).
