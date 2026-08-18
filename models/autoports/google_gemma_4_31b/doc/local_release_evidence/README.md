# Evidence: the fully green local release run

Run: 2026-08-18, `python run.py --model gemma-4-31B --workflow release
--tt-device p300x2 --impl gemma4-31b-autoport --local-server --ci-mode`.
Outcome `rc=0` -- evals 2 blocks, benchmarks 17 blocks.

Code under test: tt-metal `c54dca6b8bf`, tt-inference-server `80415992`
(branch `mvasiljevic/fast-models-fast/gemma4-31b-minimal`).

| File | Contents |
| --- | --- |
| `release_rc0_workflow.txt` | The workflow log filtered to task/block/return-code lines, all 17 sweep points and their metrics, and the `No perf targets ... NA (ungraded)` warnings |
| `results_2026-08-18T14-09-55.548355.json` | Raw lm-eval output for `mmlu_generative` (2127 samples): `exact_match,get_response = 0.7851` |
| `results_2026-08-18T14-12-11.378115.json` | Raw lm-eval output for `gpqa_diamond_generative_n_shot` (40 samples): `flexible-extract = 0.4000`, `strict-match = 0.0` |
| `release_rc0_server_config.txt` | Server startup lines proving the data-only wiring: `EXTRA_MODELS_DIR` from the spec, the bundle registration, the greedy `override_generation_config`, the watchdog raised to 120 while detection stays on, and the KV pool / concurrency figures |

Read `../local_release_flow_p300x2.md` for the interpretation, including why the
`max_concurrency=1` points are warmup-dominated and why a green result asserts
"everything executed" rather than "performance and accuracy were acceptable".
