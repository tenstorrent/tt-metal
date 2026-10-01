# Qwen3.5-2B on one Blackhole P150: reproduce TTFT, TPOT and E2E

This page explains how to reproduce the TTFT, TPOT and E2E benchmark on one Blackhole P150.
Branch: `atupe/qwen35-2b-p150-prefill-perf`. Commit: `7684b1f0b66`.
The runner is `demo/run_bench_e2e_p150.sh`.

## What you get

Measured on 2026-10-01 on one P150. Each value is the median of 3 processes.

| Prompt | OSL | Weights | TTFT | TPOT | E2E |
|---|---|---|---|---|---|
| Frankenstein corpus, exactly 4096 tokens | 8 | bfp8 (default) | 42.7 ms | 8.4 ms | 101.1 ms |
| Frankenstein corpus, exactly 4096 tokens | 8 | bfp4 (`QWEN36_W_BF4=1`) | 42.0 ms | 7.0-7.2 ms | 91.2-92.4 ms |
| Demo AI-history prompt, 2642 tokens | 100 | bfp8 (default) | 81.6 ms | 8.4 ms | 0.909 s |

- TTFT stays within about 1% between processes.
- TPOT varies up to about 5% between processes.
- The demo row uses the default bfp8 weights.

## Setup

1. Check out the branch: `git checkout atupe/qwen35-2b-p150-prefill-perf`.
2. Build: `./build_metal.sh --release`.
3. Create the environment: `./create_venv.sh`.
4. Activate it: `source python_env/bin/activate`.
5. Set `export HF_MODEL=Qwen/Qwen3.5-2B`. The weights must be in the HF cache.
6. The runner sets `HF_HUB_OFFLINE=1` for you.
7. Make sure no other process holds `/dev/tenstorrent/0`. Only one process can use the device.

## The runs

Run all commands from the repository root.

The first command uses a real 4096-token passage from the demo source document (Frankenstein).
It adds "Summarize the text above in a few bullet points."
It generates 8 tokens over 5 timed runs.

```bash
bash models/demos/blackhole/qwen36/demo/run_bench_e2e_p150.sh 4096 8 5
```

The second command runs the same case with bfp4 weights.

```bash
QWEN36_W_BF4=1 bash models/demos/blackhole/qwen36/demo/run_bench_e2e_p150.sh 4096 8 5
```

The third command uses the demo AI-history prompt of 2642 tokens.
It generates 100 tokens over 3 timed runs.

```bash
bash models/demos/blackhole/qwen36/demo/run_bench_e2e_p150.sh demo 100 3
```

The runner sets every `QWEN*` flag itself. You do not need to set the environment by hand.
Each run writes a timestamped JSON file to `models/demos/blackhole/qwen36/demo/bench_results/`.

## What the numbers mean

| Metric | Meaning |
|---|---|
| TTFT | Wall time from prompt submission to the first generated token on the host. It includes traced prefill, LM head and argmax. |
| TPOT | Mean wall time per decode step over the remaining generated tokens. |
| E2E | Wall time from submission to the last generated token. E2E is about TTFT + (OSL - 1) x TPOT. |

## Output check

The script fails if the generated token ids differ between runs.
At the default (bfp8), the first 8 tokens for the 4096-token prompt are these ids.

| Ids | Text |
|---|---|
| `[8160, 369, 264, 11782, 314, 279, 1414, 3766]` | "Here is a summary of the text provided" |

The first corpus-mode run downloads the source text.
It can land on a slightly shifted 4096-token window.
The prompt sha256 in the output JSON identifies the window of a given run.

## Notes

- bfp8 weights are the default.
- `QWEN36_W_BF4=1` is a speed toggle. It fails the accuracy gates (needle 41/50, 128k retrieval 4/6). Do not use it in production.
- `QWEN36_LM_BF4FAST` has the runner default 1. It acts only when `QWEN36_W_BF4=1`.
- `QWEN36_M3_ZB=0` is the runner default.
- `QWEN36_M3_ZB=1` restores the numerics that are bit-exact with `minimal_matmul`.
