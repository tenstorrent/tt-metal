# Perf Harness — mistralai/Ministral-8B-Instruct-2410

This is a minimal, additive harness to collect reproducible metrics for the Ministral-8B bring-up.

## What it emits
CSV/JSON with columns:
- timestamp
- model_id
- backend
- hw
- dtype
- batch_size
- max_new_tokens
- ttft_ms
- tok_per_s_user
- latency_ms

## Backends
- --backend dry — simulated numbers (runs anywhere)
- --backend cpu — HuggingFace CPU baseline (if `transformers` is available)
- --backend tt — Tenstorrent path (stub; to be wired once HW access is available)

## Quick start (dry run)

    python tools/llm_perf_ministral8b.py       --backend dry       --model-id mistralai/Ministral-8B-Instruct-2410       --hw N150 --dtype bf16 --bs 1 --max_new_tokens 64       --out perf_ministral8b.csv --json perf_ministral8b.json

Example output line:

    [dry] mistralai/Ministral-8B-Instruct-2410 hw=N150 bs=1 ttft=262ms tok/s/u=7.32 latency=8743ms -> csv=perf_ministral8b.csv json=perf_ministral8b.json

## Notes
- Intentionally small, no code duplication; intended to live under tools/ and reuse existing TT-Metal/tt-transformers components.
- Once N150/N300 access is granted, implement --backend tt, target the ≥6 / ≥12 / ≥16 tokens/s/user tiers, and add the CPU-parity accuracy check.
