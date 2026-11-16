# Llama-3.3-70B baselines, WH Galaxy (tt-metal main f0a85c864bf, no code changes)

Short prompts (~128-145 tokens in), decode speed measured at the 128th output token.

## TP32 (models/demos/llama3_70b_galaxy)

| Users | ms/token | Tokens/s/user | Total tokens/s |
|---|---|---|---|
| 32 | 16.0 | 62.6 | 2,004 |

| Prompt | TTFT (1 request, no load) | Tokens/s/user |
|---|---|---|
| 16K | 2.1 s | 42.0 |
| 32K | 4.1 s | 33.4 |

## TP8/DP4 (models/ttt_compat), greedy, all users checked

| Users | ms/token | Tokens/s/user | Total tokens/s | Log |
|---|---|---|---|---|
| 32 (4 x 8) | 53.70 | 18.62 | 596 | runs/tp8dp4_step_baseline.log |
| 128 (4 x 32) | 54.47 | 18.36 | 2,350 | runs/tp8dp4_step_b128_baseline.log |

| Prompt | TTFT (1 request per submesh, no load) | Tokens/s/user |
|---|---|---|
| 16K | 7.1 s | 17.1 |
| 32K | 21.2 s | 16.3 |

Greedy reference outputs: runs/outputs_baseline.json (32), runs/outputs_b128_baseline.json (128).

## Acceptance rule (every optimization step)

Run: `[tuning env vars] bash ~/teja/tp8_gate.sh <tag>`; results in `runs/tp8_gates.jsonl`.

Hard gate (all must pass):
1. Token accuracy, `ci-token-matching`: 500 decode tokens, teacher-forced, vs BF16 reference.
   - Top-1 >= 96.4% (baseline 96.8%, at most 2 more misses).
   - Top-5 >= 99.8%.
2. 128-user greedy run: all 128 users produce output.
3. No new degenerate output (empty, or a 12-char string repeated 3+ times).

Reported, not gated: greedy outputs identical to baseline (x/128), speed.

Before shipping: MMLU within 1 point of BF16 (customer SLA), via vLLM.

## Gate results (128 users)

| Step | Gate | Top-1 / Top-5 | Greedy identical | ms/token | Tokens/s/user | Total tokens/s |
|---|---|---|---|---|---|---|
| Baseline (stock) | PASS | 96.8 / 100 | 128/128 | 54.47 | 18.36 | 2,350 |
| + Reduce-scatter 4 links | PASS | 96.8 / 100 | 128/128 | 51.77 | 19.32 | 2,472 |
| + Matmul configs (wo, w1w3, qkv) | PASS | 97.0 / 100 | 12/128 | 50.91 | 19.64 | 2,514 |
| + All-gather before wo on 4 links (path not used without fused AG+MM; noise) | PASS | 97.0 / 100 | 12/128 | 50.66 | 19.74 | 2,527 |
| + 8B Galaxy CCL config (norm AG 4 links, MLP RS L1) | PASS | 97.0 / 100 | 12/128 | 50.65 | 19.74 | 2,527 |
| Fused AG+MM, 1 link (instead of wo AG 4 links) | PASS, rejected: slower | 96.6 / 99.8 | 4/128 | 51.77 | 19.32 | 2,472 |
| 8B CCL config + fused AG+MM, 4 links | PASS, rejected: no gain, lower accuracy | 96.6 / 99.8 | 4/128 | 50.70 | 19.73 | 2,525 |

Best config env: `RS_LINKS=4 RS_WORKERS=2 MM_CFG='{"1024x8192":[1,4],"8192x3584":[16,0],"8192x1280":[32,0]}'`
