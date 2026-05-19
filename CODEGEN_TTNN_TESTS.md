# Codegen TTNN Tests

End-to-end workflow for regenerating the per-op test corpus under `tests/ttnn/code_gen/` from real model traces and the existing unit/nightly suites.

## 1. Download model traces

Run `download_master_jsons.py` to pull every master JSON in the trace registry that matches the hardware you care about. Arguments are positional: architecture, card count, and (optional) device series.

```bash
python3 download_master_jsons.py <arch> <card_count> [<device_series>] \
    --repo-root . \
    --db "<TTNN_OPS_DATABASE_URL>"
```

Examples:

```bash
# Every Wormhole 1-card trace, all device series
python3 download_master_jsons.py Wormhole 1 --repo-root . --db "$TTNN_OPS_DATABASE_URL"

# Only n300 (Wormhole, 1 card, n300)
python3 download_master_jsons.py Wormhole 1 n300 --repo-root . --db "$TTNN_OPS_DATABASE_URL"

# Blackhole p150b
python3 download_master_jsons.py Blackhole 1 p150b --repo-root . --db "$TTNN_OPS_DATABASE_URL"
```

Output lands in `generated/model_traces/<model1>_<model2>_..._trace<id>.json`. These are the inputs Step 7 of the skill reads.

## 2. Run the extract-op-tests skill

The skill lives in the `tt_ops_code_gen` submodule (`tt_metal/third_party/tt_ops_code_gen/skills/extract-op-tests/`). Invoke it from Claude with the fully-qualified op name:

```
/extract-op-tests ttnn.matmul
/extract-op-tests ttnn.layer_norm
```

### Stage scoping (optional 2nd argument)

By default the skill runs the full pipeline. Pass a mode token after the op name to run just one stage:

| Mode | What runs | Output area |
|---|---|---|
| `sanity` | Discover/copy/trace/validate for `tests/ttnn/unit_tests` only | `tests/ttnn/code_gen/unit_tests/operations/<op>/` |
| `nightly` | Same pipeline, scoped to `tests/ttnn/nightly/unit_tests` | `tests/ttnn/code_gen/nightly/unit_tests/operations/<op>/` |
| `model-traced` | Skip extraction/trace/validate entirely; only merge configs from `generated/model_traces/` | `tests/ttnn/code_gen/model_traced/operations/<op>/<arch>/` |
| `all` (default) | Sanity + nightly + model-traced | All three above |

```
/extract-op-tests ttnn.matmul sanity
/extract-op-tests ttnn.layer_norm nightly
/extract-op-tests ttnn.matmul model-traced
```

`model-traced` is the fast path when `generated/model_traces/` already contains fresh traces and you only need to re-merge configs — it doesn't touch the unit/nightly buckets.

### Other flags

Flag: `--no-sweep-infra-fixes` — run trace+validate+prune once and stop. The agent will **not** patch the sweep module, `master_config_loader_v2.py`, `op_kwargs_utils.py`, or any other framework file even when the validator surfaces fixable diffs. Whatever gets pruned stays pruned. Use this when you want a quick baseline and are willing to defer the infra work for later.

```
/extract-op-tests ttnn.matmul --no-sweep-infra-fixes
```

What the skill does, in short:

- **Discovers** every `test_*.py` under `tests/ttnn/unit_tests` and `tests/ttnn/nightly/unit_tests` that calls `ttnn.<op>` and AST-classifies each test function as keep / drop / inject-skip / needs-Claude.
- **Copies** the kept files (plus injected `pytest.skip` for foreign call sites) into `tests/ttnn/code_gen/{unit_tests,nightly/unit_tests}/operations/<op>/`.
- **Adjudicates** the ambiguous (`needs_claude`) files — string-valued op selection in parametrize axes, multi-axis gating, dynamic dispatch through helpers.
- **Traces + validates** the extracted tests: runs `model_tracer` to produce a master JSON, generates sweep vectors via `<op>_model_traced.py`, replays the sweep wrapped in the tracer, and diffs the reconstructed JSON against the master. Artifacts land in `tests/ttnn/code_gen/<bucket>/operations/<op>/traced/<arch>/`.
- **Iterates** on diffs — patching the sweep module, `master_config_loader_v2.py`, or `op_kwargs_utils.py` as needed, until either `validate.passed: true` or every remaining diff is a documented known limit.
- **Extracts model-sweep configs** from `generated/model_traces/` (the JSONs from Step 1) into `tests/ttnn/code_gen/model_traced/operations/<op>/<arch>/ttnn_<op>_all.json`, tagged with `source_trace_id` / `source_models`.
- **Reports** file counts, validator coverage %, and every framework edit it made.

### Traced artifacts

After the trace+validate step the per-bucket output dir `tests/ttnn/code_gen/<bucket>/operations/<op>/traced/<arch>/` contains three files:

| File | What it is |
|---|---|
| `ttnn_operations_master.json` | Full trace produced by `model_tracer/generic_ops_tracer.py` against the extracted tests. Every `ttnn.<op>` invocation the tests made, plus any auxiliary ops they called (e.g. `linear`, `sharded_to_interleaved`, `ones`), captured as kwargs + a `config_hash`. |
| `ttnn_operations_master_pruned.json` | Replayable subset of the master. After running the sweep module against the master, every config whose `config_hash` did **not** show up as a `sweep_source_hash` is dropped. So this file = configs the sweep module successfully reconstructed and matched (target op kept; auxiliary ops the sweep doesn't replay are pruned out). This is the input you feed back into the sweep runner to re-run the tests without retracing. |
| `validation_summary.md` | Per-op match/diff/missing counts, coverage %, and the list of pruned configs with their hashes. |

The **difference** between master and master_pruned, in one line: master is what was *recorded*; master_pruned is what the sweep module can *replay*. The gap is everything the sweep module isn't wired to reconstruct (typed-dict kwargs without a parser, sub-device setups not mirrored in the fixture, sibling ops outside the target op's sweep module, …).

### Running the generated tests

- **unit / nightly buckets** — standard pytest, e.g. `scripts/run_safe_pytest.sh tests/ttnn/code_gen/unit_tests/operations/<op>/`.
- **Replaying traced configs without re-tracing** — point the sweep generator at `ttnn_operations_master_pruned.json` and run the sweep runner:
  ```bash
  python tests/sweep_framework/sweeps_parameter_generator.py \
      --module-name model_traced.<op>_model_traced \
      --suite-name model_traced \
      --master-trace tests/ttnn/code_gen/<bucket>/operations/<op>/traced/<arch>/ttnn_operations_master_pruned.json
  MESH_DEVICE_SHAPE=1x1 python tests/sweep_framework/sweeps_runner.py \
      --module-name model_traced.<op>_model_traced \
      --suite-name model_traced \
      --vector-source vectors_export \
      --result-dest results_export \
      --main-proc-verbose
  ```
- **model_traced bucket** — fed through the sweep runner:
  ```bash
  python tests/sweep_framework/sweeps_runner.py \
      --module-name model_traced.<op>_model_traced \
      --file-path tests/ttnn/code_gen/model_traced/operations/<op>/<arch>/ttnn_<op>_all.json
  ```

## What's missing / open questions

### Compute ops only
The skill targets compute ops. TM (tensor-manipulation) ops — reshape, transpose, slice, pad, concat, etc. — are not currently in scope. The discovery/whitelist logic and the `<op>_model_traced.py` sweep pattern both assume a compute-style call signature, and TM op extraction needs its own design pass.

### Framework patches need upstreaming
After a successful skill run we are often left with edits in:
- `tests/sweep_framework/master_config_loader_v2.py`
- `tests/sweep_framework/sweep_utils/op_kwargs_utils.py`
- `tests/sweep_framework/sweeps/model_traced/<op>_model_traced.py`

These are diffs against the model-trace infra owned by another team. They should be reviewed for whether they generalize correctly (i.e. they're not just hacks specific to one op's reconstruction), and the ones that hold up should be split into a PR to the model-trace folks.

### Baseline tt-metal commit for the test corpus
We need to nail down which tt-metal commit the generated tests are pinned to:
- **Sanity + nightly buckets** — straightforward: they pick up from the same tt-metal commit we run the skill on. Whatever HEAD is at the time of extraction is the baseline.
- **Model-traced bucket** — open. Two options:
  1. **Re-run model traces against the chosen baseline commit** so every config in `generated/model_traces/` reflects the exact code path at that commit.
  2. **Take a database snapshot** at some point and freeze the traces as-is, accepting that the traces were captured on older commits.
