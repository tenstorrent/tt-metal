# Running QB2 CI for Qwen3.8-27B on the autoport

What it takes to get `tt-agentic-bringup-qb2` to run benchmarks and evals
against *our* implementation, and why each piece is needed. Everything below is
measured or read out of the code that consumes it, not inferred.

## The dispatch was never wrong

`tt-shield-dispatch.yml` is the pre-rename name of
`manual-tt-shield-dispatch.yml` (commit `79921d5`, "rename to fit naming
convention"); it is the only workflow in the repo (id `342177897`). The callee
maps the inputs correctly:

```yaml
# tt-shield/.github/workflows/on-dispatch.yml
model: ${{ inputs.custom-model || inputs.model }}
inference-server-sha: ${{ inputs.inference-server-git-ref }}
```

Five runs failed with `No model spec matches model='Qwen3.8-27B'` for one
reason: **`inference-server-git-ref` defaults to `main`, and tt-inference-server
main has no Qwen3.8-27B spec.** The onboarding lives on branches.

Runs: 34113034709, 34113046572, 34113175558, 34113186560, 34113429291.

## Which tt-inference-server ref

Qwen3.8-27B exists on exactly two branches, both descended from
`vvukoman/add-8-models-to-release-flow`:

| | `vvukoman/add-8-models-to-release-flow` | `mvasiljevic/qwen38-autoport-qb2` |
| --- | --- | --- |
| `impl` | `qwen36_blackhole` — the **demo** | `qwen36_autoport` |
| bundle | none | `EXTRA_MODELS_DIR` + `QWEN_AUTOPORT_MODEL_ID/REVISION` |
| `trace_region_size` | `1073741824` | `200000000` |

vvukoman's spec serves the demo, so a green run there measures nothing of this
branch. Its 1 GB trace region is also the value that OOMs at startup (below).

## The main merge is not optional

`mvasiljevic/qwen38-autoport-qb2` (vvukoman + 3 commits) still fails, one stage
later:

```
ERROR: Failed to determine server type: resolve_model_spec not found in model_spec.py
```

`resolve_model_spec` was added to tt-inference-server main by #5027 ("Harden
release resolution against ambiguous catalog entries"). Neither Qwen3.8 branch
has it — both fork 63 commits back.

The subtlety: qb2 calls
`on-dispatch.yml@vvukoman/enable-on-dispatch-cross-repo-trigger`, and the script
at *that* SHA (`ba2f0331`) does **not** use `resolve_model_spec` — nothing under
`.github/scripts/` there does. But `workflow_determine-server-type.yml` checks
tt-shield out at `${{ github.job_workflow_sha }}`, which resolves to **tt-shield
main**:

```
git fetch ... +refs/heads/main:refs/remotes/origin/main
git checkout --force -B main refs/remotes/origin/main
```

So the dispatch always runs *main's* `determine_server_type.py`, whatever ref
`uses:` names — and main's version requires `resolve_model_spec`. Pinning a
tt-shield ref cannot avoid it; the tt-inference-server ref must carry it.

Measured A/B, same tt-shield SHA both times:

| `inference-server-git-ref` | `determine-server-type` |
| --- | --- |
| `mvasiljevic/qwen38-autoport-qb2` (no main) | failed — `resolve_model_spec not found` |
| `mvasiljevic/qwen38-autoport-ci` (+ main) | **success** |

Main also re-keyed the benchmark catalog by full HF repo id (#4722); the lookup
is `model_performance_reference.get(hf_model_repo)`, which no longer matches a
bare basename, so the branch's six models are re-keyed on merge.

## Working branch

`tt-inference-server` `mvasiljevic/qwen38-autoport-ci` =
`vvukoman/add-8-models-to-release-flow` + merge of `main` + one commit.

```bash
gh workflow run manual-tt-shield-dispatch.yml \
  --repo tenstorrent/tt-agentic-bringup-qb2 --ref main \
  -f model=Qwen/Qwen3.8-27B \
  -f runner-label=bh-qb-ge -f device-type=p300x2 \
  -f workflow=benchmarks \
  -f tt-metal-git-ref=mvasiljevic/qwen38-deltanet-kda \
  -f inference-server-git-ref=mvasiljevic/qwen38-autoport-ci \
  -f vllm-git-ref=main -f impl-of-model=default
```

`resolve_model_spec` accepts either `Qwen/Qwen3.8-27B` (matches
`hf_model_repo`) or the bare `Qwen3.8-27B` (matches `model_name`, via
`Path(model).name`). Both resolve; the full id is unambiguous.

## The four changes that are actually needed

Each was checked against its consumer. This is the complete set.

### Required to serve our implementation, not the demo

- **`EXTRA_MODELS_DIR`** — `Qwen3_5ForConditionalGeneration` resolves in the
  vLLM plugin's built-in map to `models.demos.blackhole.qwen36`, so without this
  the server silently serves the **demo**. The plugin registers every bundle
  under this directory *ahead* of its built-in map, so the autoport claims the
  architecture first. Per `models/autoports/vllm_bundles/README.md`, serving the
  demo "silently invalidates any autoport release report".
- **`QWEN_AUTOPORT_MODEL_ID` / `QWEN_AUTOPORT_MODEL_REVISION`** —
  `tt/functional_decoder.py` defaults these to `Qwen/Qwen3.6-27B`:
  ```python
  MODEL_ID = os.environ.get("QWEN_AUTOPORT_MODEL_ID", "Qwen/Qwen3.6-27B")
  ```
  Without them the autoport loads **3.6 weights and reports them as 3.8**.

### Required to start at all

- **`trace_region_size` 1073741824 -> 200000000** — the value is subtracted
  *per bank* (8.6 GB/device) and the engine dies during startup:
  `Out of Memory: ... 476544000 B DRAM buffer across 8 banks ... bank size is
  3198599552 B`. 200 MB is what the autoport derives its serving capacity
  against. Measured on p300x2.
- **`llm_module/runner.py`: 1200 s timeouts -> `None`** —
  `wait_for_healthy()` / `capture_traces()` treat `None` as
  `DEFAULT_WAIT_HEALTHY_TIMEOUT_S` (3600 s), verified on current main:
  ```python
  effective_timeout = DEFAULT_WAIT_HEALTHY_TIMEOUT_S if timeout is None else float(timeout)
  ```
  The runner defeated that with a hardcoded 1200 s no caller could raise.
  Qwen3.8-27B on p300x2 spends ~7 min staging weights plus ~15 min loading, so
  1200 s killed the server mid-load (bring-up run 33405666767) for reasons
  unrelated to the model. This is the one change not scoped to Qwen3.8; it only
  relaxes a cap, and no caller passes these arguments. Unaffected by the prefill
  work — this is model *load* time.

### Not required to run: attribution only

- **`impl: qwen36_blackhole -> qwen36_autoport`** (plus its `ImplSpec`).
  `ImplSpec.code_path` feeds the generated docs/report link, not the runner —
  execution is byte-identical either way. Without it the release report credits
  autoport numbers to `models/demos/blackhole/qwen36`. Drop this hunk if the new
  `impl_id` is unwelcome; nothing else depends on it.

### Deliberately unchanged

`TT_QWEN35_TEXT_VER` stays `qwen36_blackhole`. The upstream selector whitelists
only that value and raises on anything else; the bundle README says "leave unset
(or `qwen36_blackhole`)". Setting it to `qwen36_autoport` hard-fails at
registration.

## Merge fallout that is not ours

Main and vvukoman's branch each added **DiffusionGemma** and
**gemma-4-26B-A4B-it** independently. The catalog rejects the collision
outright, and this breaks *every* model, not just those two:

```
ValueError: Duplicate model spec leaf identity:
  ('google/diffusiongemma-26B-A4B-it', 'P300X2', 'vLLM', 'diffusion_gemma')
```

So a dedup is forced before Qwen3.8 can resolve at all. Which copy to keep:

- **gemma-4-26B-A4B-it — vvukoman's copy.** No test distinguishes them.
- **DiffusionGemma — main's copy**, in `llm.yaml` and `eval_config.py` alike.
  Not a preference: main carries eight tests that validate main's version
  (`test_diffusiongemma_dev_spec_matches_validated_256k_contract`,
  `TestDiffusionGemmaEvalContract`, `test_run_vllm_api_server`, ...) and
  vvukoman's earlier draft fails all eight.
- **`test_suites/llm.json`** — main's DiffusionGemma suite kept, and the model
  dropped from vvukoman's generic conformance suite. Listing it in both collides
  on the generated suite id `diffusiongemma-26b-a4b-it-p300x2` and silently
  shadows the model-specific suite.
- **Benchmark targets and eval published scores stay vvukoman's**, including its
  gemma-4-31B-it figures over main's.

## Verification

- `resolve_model_spec("Qwen/Qwen3.8-27B", "p300x2")` returns
  `impl=qwen36_autoport`, `code_path=models/autoports/qwen_qwen3_6_27b`, all
  four env vars set, `trace_region_size=200000000`.
- Eval config resolves `r1_gpqa_diamond`, `terminal_bench_2_1`,
  `swe_bench_verified`; benchmark targets resolve for `p300x2`.
- `tests/`: **5 failed / 1751 passed** on the merged branch and **5 failed /
  1751 passed** on clean `origin/main` — the same five, all pre-existing
  (`test_logging_utils`, `test_requirements_*`). No regression from the merge.
  (`tests/test_helm_generator` and `test_promote_dev_spec_to_prod` cannot be
  collected in this environment: `ModuleNotFoundError: ruamel`, on main too.)
- `determine-server-type` succeeds; runs 34116873055 (benchmarks) and
  34116898537 (evals) proceed to building the inference server.

## Long-ISL evals that used to time out

The two points the recorded sweep could not finish, rerun on this branch
(batch 1, single cold iteration):

| ISL | recorded | now | TTFT |
| --- | --- | --- | --- |
| 65536 | 1837 s | **740.1 s** (2.48x) | 573.9 s |
| 131072 | **TIMEOUT** (rc124) | **1341.8 s** | 1151.2 s |

ISL 131072 now completes rather than timing out. Decode is unchanged as
expected: 52.1 ms/token at 65536, 53.7 ms/token at 131072.
