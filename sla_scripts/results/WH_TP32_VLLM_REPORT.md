# Llama-3.3-70B, TP32 on Wormhole Galaxy via vLLM: load test vs customer SLA

Server image: tt-inference-server main 0ad0f4be9 + tt-metal f0a85c864bf + vllm-tt-plugin 042a948 (`../create_docker.txt`).
Model: `models/demos/llama3_70b_galaxy`, DP4 x 8 seqs (32 users max per Galaxy), dev spec (top_k 32), `--no-async-scheduling`.
Client: `vllm bench serve`, random dataset, `--ignore-eos`, top_k 32 / top_p 0.9 / temperature 0.6 (`../tis/bench_tis.sh`).
Mixed traffic: 2K-6K in (mean 4K), 250-750 out (mean 500), max 32 in flight. Tok/s/user = 1000 / TPOT.
Results: `bench_tis_new/` (this image), `bench_tis/` (shipped v0.9.0), `bench/` (own stack, async on).

## Verdict

- One Galaxy: fails. Max 32 users; under SLA traffic each new prompt stalls decode for all users.
- Quad (4 Galaxies): estimate only, not run. Near pass at mean load, fails at peak.
- Chunked prefill for TP32: code on this branch, off by default, not run on hardware.
- Accuracy: MMLU 82.3%, zero garbage answers with async scheduling off.

## SLA targets vs measured (one Galaxy)

| SLA line | Target | Measured | Result |
|---|---|---|---|
| Concurrent users | 128 | 32 max | Fail |
| Single 4K / 500, decode | 25 tok/s/user | 49.7 | Pass |
| 60 RPM: TTFT mean / P99 | 1 / 5 s | 1.69 / 7.02 s | Fail |
| 60 RPM: decode mean / P99 | 25 / 15 tok/s/user | 13.5 / 10.1 | Fail |
| 60 RPM: end-to-end mean / P99 | 20 / 140 s | 39.5 / 63.3 s | Fail (P99 ok) |
| 60 RPM + 40% prefix: TTFT mean / P99 | 0.5 / 1.5 s (warm) | 1.00 / 3.03 s | Fail |
| 60 RPM + 40% prefix: hit rate | 40% | 38.7% | Pass |
| 240 RPM: TTFT P99 / decode P99 | 5 s / 15 | 23.7 s / 9.6 | Fail |
| Cold 16K / 32K, single request | 5 / 10 s | 2.14 / 4.20 s | Pass |
| Cold 16K at P99 concurrency (32 at once) | 5 s | 45.9 s mean / 66.6 s P99 (own stack) | Fail |
| MMLU | >= 80%, within 1 pt of BF16 | 82.3% (TP8/DP4: 83.15%) | Pass; BF16 not run |

## Quad estimate (one Galaxy at 1/4 of the SLA rate)

Assumes an even load balancer. Prefix hits may drop if same-prefix requests go to different Galaxies.

| One Galaxy at | Same as on a Quad | TTFT mean / P99 | Decode mean / P99 | Result |
|---|---|---|---|---|
| 15 RPM | SLA mean (60 RPM) | 1.18 / 3.12 s | 35.0 / 22.7 | Near (TTFT mean 1.18 s) |
| 60 RPM | SLA peak (240 RPM) | 1.69 / 7.02 s | 13.5 / 10.1 | Fail |

## Deployment findings

- top_k must be <= 32, or sampling runs on the host.
    - Host: 333 ms/step at 32 users (3 tok/s/user). Device: 17.5 ms/step (57 tok/s/user).
    - tt-inference-server dev spec sets top_k 32; prod spec does not.
- Async scheduling must be off for TP32.
    - vLLM 0.31 turns it on by default.
    - On: 1,945 garbage MMLU answers ("\\\\..."), mostly prompts over 1.5K tokens.
    - Off: zero garbage in 13,702 answers. Decode about 10-15% slower.
- Requests stall at 32 concurrent clients: engine idle while clients time out. 8 clients is fine. Not filed.
- Shipped image v0.9.0 has prefix caching disabled: 0% hit rate.

## MMLU

lm-eval 0.4.9, task `mmlu_llama` (5-shot, chat template, greedy, 10 tokens), through the vLLM server (`../eval/run_mmlu.sh`).

| Setup | Score | Answered | Garbage |
|---|---|---|---|
| TP32, async on (own stack) | 71.9% | 14,042 | 1,945 |
| TP32, async off (this image) | 82.3% | 13,702 (340 timed out) | 0 |
| TP32, async off, unanswered counted wrong | 80.44% | 14,042 | 0 |
| TP8/DP4 | 83.15% +- 0.30 | 14,042 | 0 |

Results: `mmlu_*/`. BF16 GPU reference not run.

## Other measurements (own stack, async on; speed only)

| Case | TTFT mean / P99 | Tok/s/user mean / P99 |
|---|---|---|
| Pure decode, 32 users, 128 in / 500 out | 1.68 / 2.24 s | 57.3 / 53.9 |
| Burst: 32 x 4K arriving together | 13.6 / 20.7 s | 31.6 / 26.7 |
| Burst: 32 x 4K with 2K shared prefix | 11.8 / 13.3 s | 42.1 / 36.1 |
| 8 x 32K at once | 25.2 / 32.9 s | 14.1 / 10.6 |
| agentx traces, 64K context, 8 in flight | 4.6 / 37.5 s | 20.4 mean |

## Shipped image v0.9.0 vs this image (same tests)

| Case | v0.9.0 TTFT mean / P99 | This image TTFT mean / P99 |
|---|---|---|
| Single 4K | 14.48 s mean (first request 52 s) | 0.69 s |
| 15 RPM | 1.51 / 3.86 s | 1.18 / 3.12 s |
| 60 RPM | 5.56 / 9.15 s | 1.69 / 7.02 s |
| 60 RPM, 40% prefix | 10.5 / 15.35 s | 1.00 / 3.03 s |
| 240 RPM | 5.83 / 22.6 s | 3.12 / 23.7 s |

## What would close the gaps

- Run the TP32 chunked prefill test: `../vllm/test_chunked_prefill.sh 2048` (~2 hours, one WH Galaxy).
- Fix the request stall bug; file the async-scheduling garbage bug.
- BF16 MMLU reference on GPU.
- Confirm the Quad estimate on 4 WH Galaxies with a load balancer.
- Agree the "cold 16K at P99 concurrency" wording with the customer.
