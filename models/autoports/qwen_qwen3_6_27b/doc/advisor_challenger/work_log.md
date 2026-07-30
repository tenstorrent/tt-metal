# Advisor-challenger work log

## 2026-07-30 runner-side advisory failure

The independent check log at
`/home/mvasiljevic/skillexp-logs/p-challenger-qwen/02-02b-advisor-challenger.check-1.log`
contains only:

```text
.agents/prompts/model_bringup_multigoal/02b-advisor-challenger.check.sh: line 13: 1: model_dir
```

This fails before the checker reads any stage artifact. The checker contract is
`02b-advisor-challenger.check.sh <model_dir>` and line 13 deliberately requires
that positional argument (`MD=${1:?model_dir}`). The reported exit 1 therefore
does not refute the stage measurements; its underlying cause is that the runner
invoked the check script without its required model argument.

Reproduction:

```bash
bash .agents/prompts/model_bringup_multigoal/02b-advisor-challenger.check.sh
# exit 1: model_dir
```

Correct verification:

```bash
bash .agents/prompts/model_bringup_multigoal/02b-advisor-challenger.check.sh \
  models/autoports/qwen_qwen3_6_27b
```

Result: exit 0, with all four gate sections passing:

```text
ok: incumbent.json: >=3 repeats, incumbent_ms = best repeat, policy sourced from execution
ok: captures: per layer kind, parse, contain matmul, dtypes match shipped, DS-zero classified
ok: reconciliation.json: every disagreement has a number or an explicit below_threshold
ok: final.json: invariant holds, ties go to the incumbent, oracle passed, iterations bounded
02b-advisor-challenger gate PASSED for qwen_qwen3_6_27b
```

## Measurement conclusion

The batch-32 incumbent's best weighted repeat is 937.128544 ms. The material
advisor-derived Q/K per-head RMSNorm chain's best weighted repeat is
937.367808 ms, 0.239264 ms slower. That difference is inside the frozen
0.697760 ms same-configuration spread, so it is a tie and the
advisor-challenger rule awards ties to the incumbent. The optimized decoder
therefore remains unchanged apart from its documentation note.

