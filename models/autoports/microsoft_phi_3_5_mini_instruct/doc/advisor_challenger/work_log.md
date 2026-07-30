# Advisor-challenger remediation work log

## 2026-07-30 runner-side advisory failure

The independent runner reported:

```text
.agents/prompts/model_bringup_multigoal/02b-advisor-challenger.check.sh: line 13: 1: model_dir
```

This reproduces when the gate is invoked without its required positional
`<model_dir>` argument. Line 13 is:

```bash
MD=${1:?model_dir}
```

In Bash, an unset positional parameter produces the diagnostic
`1: model_dir`; it does not indicate that `1` was executed as a command. The
gate's usage comment and original stage prompt both require:

```bash
.agents/prompts/model_bringup_multigoal/02b-advisor-challenger.check.sh \
    models/autoports/microsoft_phi_3_5_mini_instruct
```

Running that exact command from `/home/mvasiljevic/tt-metal` exited 0 and
reported:

```text
ok: incumbent.json: >=3 repeats, incumbent_ms = best repeat, policy sourced from execution
ok: captures: per layer kind, parse, contain matmul, dtypes match shipped, DS-zero classified
ok: reconciliation.json: every disagreement has a number or an explicit below_threshold
ok: final.json: invariant holds, ties go to the incumbent, oracle passed, iterations bounded
02b-advisor-challenger gate PASSED for microsoft_phi_3_5_mini_instruct
```

The measured no-change conclusion remains valid. At decode batch 32, the
frozen incumbent's best repeat was `0.656791 ms` with a `0.000966 ms` noise
floor. The best legal advisor-derived material-chain candidate was
`0.667588 ms`, or `0.010797 ms` (`1.64%`) slower. The 11-core literal-advice
candidate was `0.667707 ms`; the 12-core bracket was illegal because it placed
a reshard kernel on a dispatch core. The incumbent therefore wins by
measurement, and `tt/optimized_decoder.py` must remain unchanged.

The shipped real-weight batch-32 trace-replay oracle passed at PCC
`0.9999923310282319` against the required `0.995` threshold.
