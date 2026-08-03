# Stage review round 3

Verdict: `more-work-needed`.

The fresh reviewer found four remaining blockers:

1. Composite GeGLU beat explicit lowering by 0.0043 ms in one serving case,
   and the aggregate gate did not control measurement noise.
2. Current composite-prefill coverage was missing because the candidate used
   explicit lowering for rank-4 tensors.
3. Dense activation-fold selection was bound to a pre-selection source and
   could not be reproduced against the current mode-specific final path.
4. Fused prefill had no direct physical capacity evidence at the advertised
   262,144-token context.

Resolution:

- Added paired 95% intervals to 101-replay decode selection. On the final rerun
  explicit lowering won 3/4 raw medians and the aggregate by 0.120 ms; the
  7.2-us inversion's interval crossed zero and no interval significantly
  favored composite.
- Made the composite candidate exercise both rank-3 decode and rank-4 prefill.
  Current-source, PCC-1.0 prefill lost by 0.949 ms sliding and 0.980 ms full.
- Reverted the unproven dense prefill fold. A complementary current-source
  always-fold control regressed decode and offered no material prefill gain
  under a predeclared 0.1% threshold.
- Added real-weight fused capacity probes. Sliding and full attention passed
  with finite last-token readback at both 262,143 and 262,144 logical tokens.

All shorter correctness, candidate, profiler, watcher, and full-suite evidence
was regenerated after the final source/test edit. Completion remains held for
a fresh independent rereview.
