# Customer SLA scripts (Llama-3.3-70B, Galaxy)

Scripts assume the layout `~/teja/{tt-metal,vllm-tt-plugin,weights,cache,runs}`.
HF token is read from `~/.cache/huggingface/token`.

| Folder | What |
|---|---|
| `setup/` | Build tt-metal, download weights |
| `demo/` | TP32 and TP8/DP4 demo runs, TP8 tuning steps, accuracy gate (`tp8_gate.sh`) |
| `sweep/` | Single-chip DRAM matmul config sweep |
| `vllm/` | TP32 vLLM server, SLA bench, prefix-hit and garbage probes, chunked prefill test |
| `tis/` | tt-inference-server image: launch, SLA bench, full suite |
| `eval/` | MMLU (lm-eval) and agentx runs |
| `bh_vllm/` | TP8/DP4 vLLM server + bench, used on BH and WH |
| `patches/` | vllm-tt-plugin DP discovery timeout (TP8/DP4 vLLM server only) |
| `create_docker.txt` | Build, run and test the server image |
| `results/` | Reports, load test and MMLU result JSONs |

TP8/DP4 tuning knobs (`ttt_compat`, all off unless set):

| Env var | Best value | Effect |
|---|---|---|
| `RS_LINKS` | `4` | Reduce-scatter links; 102 -> 64 us per layer |
| `RS_WORKERS` | `2` | Reduce-scatter workers per link |
| `MM_CFG` | `'{"1024x8192":[1,4],"8192x3584":[16,0],"8192x1280":[32,0]}'` | Decode matmul configs, `KxN: [in0_block_w, cores_per_bank]` |
| `RS_CHUNKS`, `RS_BUFFERS` | unset | No gain |
| `ATTN_AG_LINKS`, `CCL_CFG_FROM`, `AGMM`, `SILU_FUSE` | unset | No gain or slower |

Best values give 18.36 -> 19.64 tok/s/user at 128 users (accuracy gate pass). See `demo/run_4k_b128.sh`.

Key commands:

| Task | Command |
|---|---|
| Chunked prefill test (WH Galaxy) | `vllm/test_chunked_prefill.sh 2048` |
| Prefill levers SLA suite (WH Galaxy, best: `C`) | `vllm/run_prefill_levers_suite.sh C`; report `results/WH_TP32_PREFILL_LEVERS_REPORT.md` |
| Own TIS image, full suite | `tis/tis_new_suite.sh` |
| SLA bench against a running server | `OUT=<dir> tis/bench_tis.sh` |
| MMLU against a running server | `MODEL=<id> NC=8 TMO=600 eval/run_mmlu.sh <tag>` |
| Reset Galaxy | `tt-smi -glx_reset_auto` |

TP32 must run with `--no-async-scheduling`; async on gives garbage output on long prompts.
