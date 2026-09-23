# Qwen3.5-2B on one Blackhole P150 — reproduce TTFT / TPOT / E2E

Reproducible TTFT/TPOT/E2E benchmark for the experimental fused FLA GDN-prefill prim on this
branch (`fla-fused-eval`), on a single Blackhole P150.

## What you get

| Prompt | OSL | TTFT | TPOT | E2E |
|---|---|---|---|---|
| Frankenstein corpus, exactly 4096 tok | 8 | 132 ms | 20.5 ms | 276 ms |
| Demo AI-history prompt, 2642 tok | 100 | 140 ms | 20.3 ms | 2.15 s |

Run-to-run: TTFT stays within 1%; TPOT varies up to ~5% between processes.

## Setup

1. `git checkout fla-fused-eval`
2. `./build_metal.sh --release`
3. `./create_venv.sh`
4. `source python_env/bin/activate`
5. `export HF_MODEL=Qwen/Qwen3.5-2B` — the weights must already be in the HF cache;
   `HF_HUB_OFFLINE=1` (the runner sets this for you).
6. Make sure nothing else holds `/dev/tenstorrent/0` — only one process may use the device at a
   time.

## Run

```bash
models/demos/blackhole/qwen36/demo/run_bench_e2e_p150.sh f13 4096 8 5
models/demos/blackhole/qwen36/demo/run_bench_e2e_p150.sh f13 demo 100 3
```

The runner sets every `QWEN*` flag itself, including `QWEN_GDN_PATH=fused QWEN_GDN_NP=6` for the
fused FLA prim — no manual env setup needed. Each run writes a timestamped JSON to
`models/demos/blackhole/qwen36/demo/bench_results/`.

## What the numbers mean

- **TTFT** — wall time from submitting the prompt to the first generated token on host (traced
  prefill + LM head + argmax).
- **TPOT** — mean wall time per decode step, over the remaining generated tokens.
- **E2E** — wall time from submission to the last generated token; E2E ~ TTFT + (OSL-1) x TPOT.

## Output check

The script fails if generated token ids differ between runs. The expected first 8 tokens for the
4096-token prompt decode to "Here is a summary of the text provided". The first corpus-mode run
downloads the source text and may land on a slightly shifted 4096-token window; the prompt sha256
recorded in the output JSON identifies which window a given run used.

## Notes

The fused FLA prim on this branch comes from PR #53961 (evaluation only, not for merge). Measured
2026-09-23 on one Blackhole P150.
