# Run vLLM Benchmarks

## Prerequisites

Server must be running and healthy (see server.md).

## Standard benchmark

```bash
vllm bench serve \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 100 \
  --random-output-len 100 \
  --num-prompts 32 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result \
  --result-filename output/vllm_result.json
```

## Key metrics

- **TTFT** — time to first token
- **TPOT** — time per output token
- **ITL** — inter-token latency
- **E2EL** — end-to-end latency

## Structured output benchmark (optional)

```bash
python vllm/benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 32 \
  --dataset json-unique \
  --output-len 1024
```

## Varying parameters

Adjust `--num-prompts`, `--random-input-len`, `--random-output-len` to test
different load patterns. Use `--request-rate` to control throughput testing.
