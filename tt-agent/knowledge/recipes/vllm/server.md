# vLLM Server Lifecycle

## Start server

```bash
python examples/server_example_tt.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  > server.log 2>&1 &
echo $! > server.pid
```

With TT config overrides (for multi-chip, tracing, etc.):

```bash
VLLM_RPC_TIMEOUT=900000 python examples/server_example_tt.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --override_tt_config '{"fabric_config":"FABRIC_1D_RING","sample_on_device_mode":"all"}' \
  > server.log 2>&1 &
echo $! > server.pid
```

## Common server args

- `--model` — HuggingFace model ID (required)
- `--data_parallel_size N` — DP replicas across chips
- `--max_num_seqs N` — max concurrent sequences
- `--max_model_len N` — max context length
- `--override_tt_config '{...}'` — TT-specific config JSON

## Health check

```bash
curl -sf http://localhost:8000/health
```

Returns HTTP 200 when ready. Large models (70B+) can take 30-45 minutes to load.

## Health check with retry loop

```bash
timeout=2700  # 45 minutes
elapsed=0
while [ $elapsed -lt $timeout ]; do
  curl -sf http://localhost:8000/health && echo "Ready" && break
  sleep 20
  elapsed=$((elapsed + 20))
done
```

## Stop server

```bash
kill $(cat server.pid)
```

## OpenAI-compatible API

Once running, the server exposes standard OpenAI endpoints:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","prompt":"Hello","max_tokens":64}'
```
