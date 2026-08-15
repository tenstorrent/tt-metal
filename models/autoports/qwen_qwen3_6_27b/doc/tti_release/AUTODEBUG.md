# AutoDebug: Qwen3.6-27B release runs zero standard evals

## Scope and verdict

Inspection-only diagnosis against `/home/mvasiljevic/tt-inference-server` at
`e26e723bf0266cde85f674e381fbee10068ae0ec`. No TT device, live endpoint, or
implementation/harness/config file was changed or exercised.

**Headline finding:** this is deterministic eval-catalog under-onboarding, not a
runtime-model-spec lookup or external-server dispatch failure. The
`Qwen/Qwen3.6-27B` `EvalConfig` contains only the agentic
`terminal_bench_2` task. The standard release child accepts only
`EVALS_COMMON`, `EVALS_META`, or `EVALS_VISION`, therefore selects zero tasks;
the workflow explicitly records that empty selection as a successful no-op.
Adding the mandatory `meta_ifeval` and `meta_gpqa_cot` tasks to this existing
model config is the smallest intervention that fixes the reported boundary.

## Direct observations

1. The supplied runtime spec has the correct catalog keys and context contract:
   `hf_model_repo` is `Qwen/Qwen3.6-27B`, `model_name` is `Qwen3.6-27B`, the
   device `max_context` and vLLM `max_model_len` are both `262144`, and the CLI
   metadata selects external-server coordinates plus `ci-nightly`
   (`autoport_release_spec.json:9-26,65-73`). The observed command reaches the
   v2 release workflow with that model and external URL
   (`tti_release_ci_nightly.log:286-290`).

2. The matching eval catalog entry begins at
   `reference_config/evals/eval_config.py:1262-1264`. Its only active task is
   `terminal_bench_2`, marked `WorkflowVenvType.EVALS_AGENTIC`
   (`:1265-1268`); the entry closes at `:1372`. The SWE-bench task is commented
   out (`:1318-1371`). There is no active `meta_ifeval` or `meta_gpqa_cot` in
   this model entry.

3. `llm_module/eval_configs.py:18-26` defines standard backends as common,
   Meta, and vision only. `get_llm_eval_tasks` looks up by
   `model_spec.model_name` (`:70-83`), filters tasks by that set (`:85-87`), and
   deliberately returns `[]` when none remain (`:88-96`). This exactly predicts
   the log at `tti_release_ci_nightly.log:291-293`.

4. The empty result is masked as release success: `run_llm_eval` returns no
   blocks when task selection is empty
   (`test_module/llm_tests/llm_eval_tests.py:360-374`), and
   `EvalsWorkflow._run_llm_eval_task` maps no blocks to exit code zero
   (`workflow_module/workflows.py:135-145`). Thus the harness is behaving as
   authored for an agentic-only model, but that authored configuration violates
   this release's mandatory standard-eval contract.

5. Release dispatch itself already supports the desired combination. Standard
   evals are an ordinary release child (`workflow_module/workflows.py:497-518`),
   while a model with an agentic task also receives the separate agentic child
   (`:581-601`). Provisioning discovers every selected standard task venv from
   `EVAL_CONFIGS` (`workflows/workflow_dispatch.py:685-705`) and adds the
   agentic venv independently (`:708-735`). No release-dispatch rewrite is
   needed.

## Smallest correct intervention

Extend the existing `Qwen/Qwen3.6-27B` task list in
`reference_config/evals/eval_config.py` with two active `EvalTask` entries:

- `meta_ifeval`, `workflow_venv_type=EVALS_META`, `include_path="work_dir"`;
  use `score_task_keys_mean` with the four established IFEval result keys shown
  by the canonical sibling recipe at `eval_config.py:2965-2984`.
- `meta_gpqa_cot`, `workflow_venv_type=EVALS_META`,
  `include_path="work_dir"`; use `score_task_single_key` with
  `exact_match,strict-match`, as at `:2987-3003`.

For both tasks, preserve the Meta recipe's **preformatted-prompt contract**:
`apply_chat_template=False`. Meta setup writes a model-specific work directory
and supplies the actual HF model identity to dataset preparation
(`workflows/workflow_venvs.py:414-445`), and command construction stages that
directory because the Meta YAMLs hard-code `./work_dir`
(`llm_module/eval_command.py:364-389`). Setting `apply_chat_template=True`, or
switching these established Meta tasks blindly to the chat-completions API,
would risk applying Qwen's chat format twice rather than making the prompts
chat-correct. The task's default completions route is selected at
`llm_module/eval_command.py:199-203`.

Each new task must also declare a finite `EvalLimitMode.CI_NIGHTLY` value in
`limit_samples_map` (and retain a smoke limit if desired). The command builder
only emits `--limit` for CI-nightly when that map contains the mode
(`llm_module/eval_command.py:77-82,120-126,402-408`); omitting it silently runs
the full dataset. The exact sample count/fraction and the Qwen-specific GPU or
mode reference scores are release-policy data and are not derivable from this
source tree. They must be supplied from an approved Qwen GPU baseline; copying
Llama scores would make acceptance invalid. This is the only unresolved input
to the configuration patch.

No runtime spec change is required for context or server ownership. Command
construction takes `device_model_spec.max_context` as authoritative and injects
it as `max_length` for text API evals (`llm_module/eval_command.py:229-257,
294-299`), so these tasks inherit `262144`. `run_llm_eval` selects
`RemoteOpenAIController` when the media context is remote
(`test_module/llm_tests/llm_eval_tests.py:376-385`), matching the observed
external endpoint.

## Focused non-hardware verification

After the two config entries and their approved limits/references are added,
run only source/unit checks (do not invoke the release command or contact port
8000):

```bash
cd /home/mvasiljevic/tt-inference-server

python - <<'PY'
from types import SimpleNamespace
from llm_module.eval_configs import get_llm_eval_tasks
from reference_config.evals.eval_config import EVAL_CONFIGS
from workflows.workflow_types import EvalLimitMode, WorkflowVenvType

spec = SimpleNamespace(model_name="Qwen3.6-27B")
rc = SimpleNamespace(eval_samples=None, limit_samples_mode="ci-nightly")
tasks = get_llm_eval_tasks(spec, rc)
assert [t.task_name for t in tasks] == ["meta_ifeval", "meta_gpqa_cot"]
assert all(t.workflow_venv_type == WorkflowVenvType.EVALS_META for t in tasks)
assert all(EvalLimitMode.CI_NIGHTLY in t.limit_samples_map for t in tasks)
assert all(not t.apply_chat_template and t.include_path == "work_dir" for t in tasks)
assert any(t.workflow_venv_type == WorkflowVenvType.EVALS_AGENTIC
           for t in EVAL_CONFIGS["Qwen3.6-27B"].tasks)
PY

pytest -q \
  tests/llm_module/test_eval_command.py \
  tests/workflows/test_workflow_dispatch_routing.py \
  tests/workflow_module/test_release_workflow.py
```

Add a focused regression test that constructs the Qwen runtime spec (or a
minimal equivalent), asserts the two standard tasks survive selection in
CI-nightly mode, asserts `_llm_eval_venv_types` includes `EVALS_META`, and
inspects `build_eval_command` without executing it for all of the following:
`max_length=262144`, external base URL
`http://127.0.0.1:8000/v1/completions`, no `--apply_chat_template`, and a
non-null CI-nightly `--limit` for each task.

## Other potential issue / guardrail

The generic no-standard-task path returning success is intentional for truly
agentic-only models, so changing it globally is broader than the reported fix.
A separate release-contract validation could fail early when a release declares
mandatory standard tasks but selection is empty; however, no such declaration
exists in the supplied runtime spec, so that would require a new policy/schema
boundary and is not the smallest repair.
