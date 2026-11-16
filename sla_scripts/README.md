# customer SLA scripts (Llama-3.3-70B, Galaxy)

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
| `patches/` | TP8/DP4 tuning knobs (tt-metal), DP discovery timeout (vllm-tt-plugin) |

Key commands:

| Task | Command |
|---|---|
| Chunked prefill test (WH Galaxy) | `vllm/test_chunked_prefill.sh 2048` |
| Own TIS image, full suite | `tis/tis_new_suite.sh` |
| SLA bench against a running server | `OUT=<dir> tis/bench_tis.sh` |
| MMLU against a running server | `MODEL=<id> NC=8 TMO=600 eval/run_mmlu.sh <tag>` |
| Reset Galaxy | `tt-smi -glx_reset_auto` |

TP32 must run with `--no-async-scheduling`; async on gives garbage output on long prompts.
