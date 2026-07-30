# Advisor-challenger work log

## 2026-07-30 independent-check remediation

The runner-side check log
`02-02b-advisor-challenger.check-1.log` contained only:

```text
.agents/prompts/model_bringup_multigoal/02b-advisor-challenger.check.sh: line 13: 1: model_dir
```

This reproduced when the check script was invoked with no positional argument.
It is an invocation/interface failure, not evidence against the measured
decoder result: invoking the same checker with the model directory completed
all incumbent, capture, reconciliation, and final-invariant checks with exit 0.

The checker now supports omitted `model_dir` when exactly one
`models/autoports/*/doc/advisor_challenger` result directory is changed by the
active stage, or when only one such result exists in the repository. Ambiguous
or missing results still fail and require an explicit target, preventing a
runner from accidentally validating the wrong model.

The underlying measured conclusion remains unchanged:

- frozen incumbent: 1.275091270065826 ms;
- advisor attention chain: 1.3292236051157764 ms, slower by 0.0541323350499504 ms;
- advisor norm/residual R11 chain: 1.1771252662267373 ms, but failed the
  incumbent PCC bar (0.9947948362 < 0.995);
- fastest oracle-passing result: incumbent, 1.275091270065826 ms.

Thus the advisor did not produce a legal measured winner, and shipping no
decoder change is required by the challenger invariant.

Post-fix verification:

- runner form:
  `.agents/prompts/model_bringup_multigoal/02b-advisor-challenger.check.sh`
  exited 0;
- explicit form:
  `.agents/prompts/model_bringup_multigoal/02b-advisor-challenger.check.sh models/autoports/google_gemma_4_26b_a4b_it`
  exited 0;
- both runs reported all four gate sections `ok` and ended with
  `02b-advisor-challenger gate PASSED for google_gemma_4_26b_a4b_it`.
