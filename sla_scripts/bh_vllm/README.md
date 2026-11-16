# BH Galaxy TP8/DP4 vLLM traffic test (Llama-3.3-70B)

1. Install vLLM into the tt-metal python_env: `vllm-tt-plugin/docs/install-vllm-tt.sh` (plugin 042a948 was used on WH).
2. Set paths:
   ```bash
   export TT_METAL_DIR=/mnt/nvme0/teja/tt-metal PLUGIN_DIR=/mnt/nvme0/teja/vllm-tt-plugin
   export HF_MODEL=/path/to/Llama-3.3-70B-Instruct TT_CACHE_PATH=/path/to/cache OUT=/mnt/nvme0/teja/logs/vllm_bh
   ```
3. Use the same `MESH_DEVICE` as the BH demo runs (default `TG`).
4. `tmux new -d -s bhsrv bash run_server.sh` then `tmux new -d -s bhbench bash bench.sh`.
5. Result: `$OUT/summary.txt`; `BENCH FINISHED` at the end of the bench tmux.

Notes:
- top_k <= 32 keeps sampling on device; without it decode is ~10x slower.
- 128 users total (DP4 x 32).
- If the server fails, check `$OUT/server.log` and reset with `tt-smi -glx_reset_auto`.
- Prefill budget is 8192 tokens per step; 2048 (vLLM default) made TTFT ~2x worse on WH.
- On WH, DP engine 2 hung twice under load (KV ~96% full). If a lane stops, note which one.
- vLLM's "Prefix cache hit rate" log line is a rolling mix of recent runs; use hit_*.json.
