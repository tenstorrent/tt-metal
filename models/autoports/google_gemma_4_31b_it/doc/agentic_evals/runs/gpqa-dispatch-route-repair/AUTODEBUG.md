# GPQA candidates dispatched to the wrong workflow

**High-confidence diagnosis:** `workflow=agentic` excludes GPQA by design. With both agentic suites disabled for CI nightly, this yields no result blocks. Correct only the dispatch input to `workflow=evals`; no cohort, configuration, or source change is needed to fix this routing failure.

## Evidence and revision scope

Inspected clean worktrees (HEAD verified):

- **C4:** `/home/mvasiljevic/gemma4-31b-pipeline-v3/tti-gpqa-c4-integrated`, `82da396ef63bbe95cedbe254bb5433d905cd0491`.
- **C2:** `/home/mvasiljevic/gemma4-31b-pipeline-v3/tti-gpqa-larger20-c2`, `82569bf17497585555a5710929a8e2f78ea47f1b`.

Source references below are relative to C2. All routing/runner files cited are unchanged between these commits. Their complete diff affects only `reference_config/evals/eval_config.py` and two tests: GPQA concurrency 4→2, frozen IDs 10→20, and nightly limit 10→20.

Direct observations:

- Private log `/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/stage12-gpqa-c4-live-1424.log`: lines 51–53 identify C4 SHA and `agentic`; line 1225 executes `run.py --workflow agentic --ci-mode`; line 1257 reports `ci-nightly`; lines 1704–1711 show GPQA excluded, both agentic tasks skipped at zero, and `rc=1 error=no_blocks`. Line 1990 identifies run **42232104462**. Missing report artifacts at lines 2089–2100 follow the empty evaluation.
- `/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/stage12-gpqa20-route-failure-run.json:1` records run **42233126698**, `agentic`, and the C2 SHA in its display title. This snapshot says `in_progress`; it does not establish C2's final outcome.

## Causal chain checked against source

1. `reference_config/evals/eval_config.py:11935–11957` registers this model's `r1_gpqa_diamond` as **EVALS_COMMON**, not EVALS_AGENTIC. C2's twenty IDs/nightly limit are at 11972–11996; C4's ten IDs/nightly limit are at 11972–11975 in C4. Both revisions retain nightly zero for SWE and Terminal (C2: 12051–12054 and 12110–12113).
2. `workflow_module/workflows.py:166–184` sends `agentic` to `run_llm_agentic_eval`. `test_module/llm_tests/agentic_eval_tests.py:95–126` selects only EVALS_AGENTIC tasks and emits the exact observed non-agentic skip message, explicitly directing standard tasks to `evals`.
3. `llm_module/drivers/agentic.py:299–322` resolves the nightly limit; lines 127–133 return success with `raw=None` for zero tasks. The actual caller, `test_module/llm_tests/agentic_eval_tests.py:234–265`, skips these empty results and returns an empty block list. `workflow_module/workflows.py:190–193` emits the observed empty-task error; `workflow_module/execution.py:297–305` emits `no_blocks` when no report schema exists. This fully explains C4 without invoking model accuracy, sampling, or hardware faults.
4. The alternative route is explicit: `workflows/workflow_dispatch.py:138–149,195–196` recognizes LLM `evals`; `workflow_module/workflows.py:102–127` calls `run_llm_eval`. `llm_module/eval_configs.py:22–27,241–251` includes EVALS_COMMON and excludes EVALS_AGENTIC. GPQA therefore survives this selection while SWE/Terminal do not.
5. Frozen cohorts are already supported: `llm_module/eval_command.py:197–208,231–246` slices configured document IDs by the active limit; lines 764–785 pass them via `--samples` instead of `--limit`. Keeping CI nightly selects the existing ten or twenty IDs for the respective candidate. No cohort rewrite is implied.

## Minimal correction and remaining uncertainty

For a new dispatch of the intended candidate, replace **only** `-f workflow=agentic` with **`-f workflow=evals`**. Preserve its immutable tt-metal and inference-server SHAs, model/implementation, device, CI nightly selection (`run-full-evals=false`), disabled summary/comment flags, and all candidate settings. Do not enable SWE/Terminal merely to manufacture blocks, relabel GPQA as agentic, or suppress `no_blocks`. The generic submission template in `bringup/agentic_evals/inputs.json` uses `agentic`; it is unsuitable unchanged for these GPQA-only candidates.

Reviewed the draft against both revisions and the actual agentic caller: the empty driver results are discarded before parsing, matching the observed sequence. C2 has the same deterministic exclusion if it reaches this route, but its final execution is not proven by the inspected snapshot. No corrected-route run was performed. Dataset access, common-venv provisioning, completion within timeout, generated trajectories, scores, and CI artifact collection on `evals` remain runtime checks; routing correction alone does not prove them. No additional headline source defect is established.

Inspection only: no workload execution, hardware commands, network operations, publication, delegation, or source/config edits. Only this evidence-directory report was written.
