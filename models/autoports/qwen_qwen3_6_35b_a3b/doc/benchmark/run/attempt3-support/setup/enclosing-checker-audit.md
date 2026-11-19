# Enclosing checker routing audit

The active enclosing multigoal runner selects the cached 0.1.24 benchmark gate.
That gate cannot interpret the explicitly owner-approved 28800-second client
budget. This is a checker-version incompatibility, independent of whether the
third attempt eventually satisfies the measurement checks.

## Observations

- The inspected runner (PID 2687900) and `multigoal_logs_s11/manifest.txt` select
  the cached 0.1.24 `prompts/model_bringup_multigoal/11-benchmark.txt`.
- `lane-env.sh` also selects cached 0.1.24 and says not to edit it during a run.
- `scripts/multigoal::find_check_script()` selects the `.check.sh` beside the
  positional prompt. The cached `11-benchmark.check.sh` derives
  `TT_MODEL_BRINGUP_ROOT` from its own location (line 6), prepends that package's
  runtime to `PYTHONPATH` (line 16), and invokes `benchmark_stage.check` (line 23).
  An external `PYTHONPATH` or root-variable setting does not override that gate.
- A read-only CPU call to each version's `batch_plan` using the actual
  `run/attempt3/run_config.json` produced:

```text
budget_seconds field: 28800
cached0.1.24 ValueError budget_seconds (client clock) must be positive and at most 7200
editable0.1.26 28800.0
```

The editable workflow is commit
`f2873162ea1b8b4971aa9b41c83a551e7c1d53e5`, including the earlier timeout repair.
It validates the approval fields and the same measurement completeness checks.
The active enclosing runner has no supported live checker-path override.

## Standalone repaired gate

After final evidence promotion, run this exact command from the tt-metal
checkout, using the existing lane Python environment:

```sh
MODEL_DIR=models/autoports/qwen_qwen3_6_35b_a3b \
HF_MODEL=Qwen/Qwen3.6-35B-A3B \
bash /home/ttuser/qwen36-35b-a3b/skills-v24/plugins/tt-model-bringup/prompts/model_bringup_multigoal/11-benchmark.check.sh
```

This check script selects its adjacent editable runtime and runs both the
benchmark evidence gate and context-contract check. Preserve its complete output
and exit status separately from the enclosing cached gate. A repaired checker
failure due to incomplete measurements remains a real failure; approval of a
larger clock does not waive those checks.

This audit did not modify the active runner, installed cache or lane environment,
and did not run inference or access hardware. It does not request or authorize
another inference attempt. The parent will retain the explicit repaired check
result and report the enclosing cached gate's incompatibility.
