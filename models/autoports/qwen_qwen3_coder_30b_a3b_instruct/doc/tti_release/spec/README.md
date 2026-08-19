# What is in `spec/`, and which state each file records

Three files, and they deliberately do **not** all describe the same moment. Read
this before reconstructing the configuration from them.

| File | What it is | State it records |
|---|---|---|
| `tti_catalog_edits.patch` | `git diff` of the uncommitted registration edits in `/home/raahem/tt-inference-server`, branch `raahem/qwen3-coder-30b-a3b-tti` | **Current tree.** Regenerated after review. Includes the post-Findings-1-and-3 eval settings that the release run actually used, and the `known_issues` waiver added after the run. |
| `resolved_model_spec.json` | `ModelSpec.get_serialized_dict()` for `id_qwen3-coder-30b-a3b-autoport_…_p300x2`, resolved from the dev catalog | **Current tree**, regenerated alongside the diff, so it agrees with it. Carries the two `EVALS` `known_issues` entries. |
| `runtime_model_spec_smoke.json` | the runtime spec TTI wrote for the step-1 smoke `--workflow benchmarks` run | **2026-08-18 19:40**, unedited. |

The record of **what the release run itself loaded** is not here — it is
`../run_specs/runtime_model_spec_release.json`, byte-identical to what TTI wrote
at 2026-08-18 22:35:11. That file still shows `known_issues: []`, correctly: the
waiver postdates the run. Its `cli_args` is the 63-key argparse namespace dump,
whose `service_port` is the untouched `8000` default rather than the `8100` the
run used — see the `cli_args` bullet under "TTI friction" in `../RUN_NOTES.md`.

**Two settings in the diff are the subject of findings, not defaults.**
`max_gen_toks` is `2048` (coding, `ifeval`) / `4096` (GPQA) rather than the `256`
inherited from `Qwen/Qwen2.5-Coder-32B-Instruct`, and `batch_size` is left at the
`EvalTask` default of `1` rather than the inherited `16`. Reconstructing a run
from an earlier copy of this diff — which documented the inherited values —
reproduces the 27.6 % `mbpp_instruct` result that Finding 1 exists to explain.
See Findings 1 and 3 in `../RUN_NOTES.md`.
