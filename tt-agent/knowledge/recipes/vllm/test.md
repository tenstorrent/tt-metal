# Run vLLM Sampling Tests

## Prerequisites

Server must be running and healthy (see server.md).

## Run sampling tests

```bash
pytest vllm/tests/tt -v \
  --tt-server-url=http://localhost:8000 \
  --tt-model-name=meta-llama/Llama-3.1-8B-Instruct
```

## Full CI lifecycle

```bash
# 1. Start server (see server.md)
# 2. Wait for health check
# 3. Run benchmarks (see benchmark.md)
# 4. Run sampling tests
pytest vllm/tests/tt -v --tt-server-url=http://localhost:8000 --tt-model-name=<model>
# 5. Kill server
kill $(cat server.pid)
```
