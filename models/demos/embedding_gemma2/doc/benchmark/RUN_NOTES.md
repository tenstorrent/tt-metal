# E8 benchmark: run notes

Logs: `/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e8/logs/` (each with `__start` / `__end` /
`wall_s` / `__tt_exit:<rc>` lines). Launcher `e8/scripts/run.sh` (card-free check, env dump, tee).

## Environment

```bash
source /local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e1/env.sh
#  cd tt-metal; TT_METAL_HOME=$PWD PYTHONPATH=/local/ttuser/gtobar/venvs/tf_overlay:$PWD TT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16
export HF_HUB_OFFLINE=1 HF_MODEL=/local/ttuser/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b
export EMBEDDING_GEMMA2_RESULTS=/local/ttuser/gtobar/tt-ops/runs/embeddinggemma-2/e8/results.jsonl
# EMBEDDING_GEMMA2_PRECISION unset: the selected config file is used
```

## Commits

| commit | content |
|---|---|
| `119d0ba4c7` | stage base |
| `e01ed3a09b` | E6b review doc fixes (docs only) |
| `1bcb04d2f0` | perf.py full benchmark harness; the full-sweep run below ran on it |
| `81d8ef58ed` | perf.py quick default, every extra opt-in (change request after the full run) |
| `4c2e82e044` | perf.py docstring (measured costs); final code commit, gates below |

## Runs (all on card 0, one P150a; times are host local)

| when | commit | command | result | wall |
|---|---|---|---|---|
| 06:33:05-06:33:41 | e01ed3a09b + uncommitted perf.py | `pytest perf.py -k TestAudio -s -q` (ITERS=5, SUSTAINED_S=5, output to scratch) | 7 passed, 3 skipped, rc 0 | 36 s |
| 06:33:49-06:36:42 | same | `pytest perf.py -s -q` (ITERS=5, SUSTAINED_S=5, scratch) | 44 passed, 18 skipped, rc 0 | 173 s |
| 06:36:54-06:38:58 | same, probe list extended to 8192 | `pytest perf.py -k "TestTextBatchProbe and (b2048 or b4096 or b8192)" -s -q` | 2048 / 4096 fit, 8192 DRAM OOM; rc 0 | 124 s |
| **06:39:28-06:44:46** | **1bcb04d2f0** | **`pytest models/demos/embedding_gemma2/tests/perf/perf.py -s`** (defaults: N 20, warm-up 3, sustained 30 s) -> `doc/benchmark/run/` | **46 passed, 19 skipped, rc 0**, run_id 20261028-063930 | **318 s** |
| 06:45-06:47 | 1bcb04d2f0 | gates (test_full_model / _trace / demo / test_precision_config) | 30 / 11 / 2 / 7 passed, rc 0 | superseded by the final-commit gates |
| 06:50:32-06:53:09 | 81d8ef58ed | default perf run, opt-in runs, gates | all passed, rc 0 | superseded (docstring commit followed) |
| 06:53:37-06:53:52 | **4c2e82e044** | `pytest perf.py` (default, nothing opted in; output to e8/run_extra) | 4 passed, rc 0 | **15 s** (pytest 12.1 s) |
| 06:53:52-06:54:04 | 4c2e82e044 | `EMBEDDING_GEMMA2_PERF_HF_CHECK=1 pytest perf.py -k TestVision` | 1 passed (soft PCC 0.996577), rc 0 | 12 s |
| 06:54:04-06:54:13 | 4c2e82e044 | `EMBEDDING_GEMMA2_PERF_SHAPES=1 EMBEDDING_GEMMA2_PERF_MODES=1 pytest perf.py -k "TestAudio and b4x256"` | 2 passed, 1 skipped (2cq), rc 0 | 9 s |
| 06:54:13-06:55:03 | 4c2e82e044 | `pytest .../tests/pcc/test_full_model.py -q` | 30 passed, rc 0 | 50 s |
| 06:55:04-06:55:33 | 4c2e82e044 | `pytest .../tests/pcc/test_full_model_trace.py -q` (trace == eager) | 11 passed, rc 0 | 29 s |
| 06:55:34-06:56:02 | 4c2e82e044 | `pytest .../demo/demo.py -s -q` | 2 passed, rc 0 | 28 s |
| 06:56:02-06:56:13 | 4c2e82e044 | `pytest .../tests/pcc/test_precision_config.py -q` | 7 passed, rc 0 | 11 s |

Gate logs on the final commit: `e8/logs/gate_*.log`; the earlier gate rounds are kept under
`e8/logs/gates_1bcb04d2f0/` and `e8/logs/gates_81d8ef58ed/`.

## Reruns and why

- Smoke runs (rows 1-3) checked the harness at low iteration counts before the code commit; their
  records went to the scratchpad, not to `run/`. The batch probe was extended from 1024 to 8192 after
  1024 still fit, to reach the OOM boundary.
- After the full-sweep run the person asked for a quick default with every extra as an opt-in. The
  full-sweep results stay the evidence for REPORT.md: `EMBEDDING_GEMMA2_PERF_ALL=1 pytest perf.py
  --collect-only` on the final code collects exactly the 65 test ids the full-sweep run executed
  (`e8/logs/collect_check.log`, diff empty; `full_run_ids.txt` vs `collect_all.txt`), and the default
  collects only the 4 headline ids. Per-option collection counts: SHAPES 15, MODES 13, LONG 5,
  PROBE 13, SUSTAINED 7, HF_CHECK 4.
- The gates ran three times because the code commit moved twice (opt-in restructure, then a
  docstring fix); only the last round, on `4c2e82e044`, is the gate evidence.
- No hang, no reset, no tt-smi -r in this stage. `tt-smi -s` snapshot taken before the full run
  (`e8/logs/tt_smi_snapshot.json`) for identity.json.
