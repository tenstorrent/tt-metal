# Molmo2 Video Verification Test Results (Docker)
Date: 2026-03-28

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | 105 |
| Successful | 105 (100.0%) |
| Errors | 0 |
| Timeouts | 0 |
| Avg Latency | 10266ms (10.3s) |

## Configuration

- Model: allenai/Molmo2-8B
- Device: T3K (8 Wormhole devices)
- Deployment: Docker container (dev image)
- Image: ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.11.0-3035237ebd-ba84dbf0
- Server: vLLM with TT backend
- Traces: ENABLED for text prefill, DISABLED for vision prefill

## Docker Command

```bash
cd /home/ttuser/ssinghal/PR-fix/tt-metal/tt-inference-server
source python_env/bin/activate
python run.py --model Molmo2-8B --workflow server --tt-device t3k \
  --docker-server --dev-mode --no-auth --skip-system-sw-validation --host-hf-cache
```

## Test Results

All 105 video tests from `test.jsonl` completed successfully with coherent responses.

### Comparison: Local vs Docker

| Metric | Local Server | Docker Server |
|--------|-------------|---------------|
| Success Rate | 100% | 100% |
| Avg Latency | 10265ms | 10266ms |
| Total Time | 1077.8s | 1078.0s |

Both deployments produce identical performance and quality.
