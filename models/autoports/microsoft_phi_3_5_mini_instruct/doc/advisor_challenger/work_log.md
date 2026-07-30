# Advisor challenger work log

## 2026-07-30 runner-side verification remediation

The reported runner failure is an invocation failure, not an artifact, model,
correctness, or performance failure. The complete reported output was:

```text
.agents/prompts/model_bringup_multigoal/02b-advisor-challenger.check.sh: line 13: 1: model_dir
```

Line 13 is `MD=${1:?model_dir}`. The runner invoked a gate whose documented
usage is `02b-advisor-challenger.check.sh <model_dir>` without `$1`, so the
shell exited before any stage artifact was read.

Reproduction:

```bash
.agents/prompts/model_bringup_multigoal/02b-advisor-challenger.check.sh
# exit 1: line 13: 1: model_dir
```

Required verification:

```bash
.agents/prompts/model_bringup_multigoal/02b-advisor-challenger.check.sh \
  models/autoports/microsoft_phi_3_5_mini_instruct
# exit 0: 02b-advisor-challenger gate PASSED
```

The successful gate independently checked the frozen three-repeat incumbent,
batch/order and shipped-precision capture, DRAM-sharded advice accounting,
chain-level reconciliation, measured combinations, correctness oracle, and
the final latency invariant.

The measured result remains a real win: incumbent best 0.806270 ms, incumbent
spread/noise floor 0.000501 ms, shipped best 0.796420 ms, improvement
0.009850 ms (1.2217%). The shipped batch-32 real-weight oracle passed at PCC
0.998931492826291 against a required 0.995. The best measured combination was
qkv + o_proj + gate_up DRAM-sharded decode weights; all component, pairwise,
and cumulative measurements are recorded in `final.json`.

Additional integrity checks in this remediation:

```bash
python3 -m py_compile \
  models/autoports/microsoft_phi_3_5_mini_instruct/tt/optimized_decoder.py \
  models/autoports/microsoft_phi_3_5_mini_instruct/doc/advisor_challenger/capture_phi.py
# exit 0

git diff --check
# exit 0
```

No model change was necessary for this runner-side defect. The underlying
model change and measurement evidence are retained unchanged; the fix is to
invoke the self-contained gate with its required model argument.
