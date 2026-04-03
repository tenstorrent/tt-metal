# Install vLLM with TT backend

## Prerequisites

- tt-metal built and importable (`import ttnn` works)
- `VLLM_TARGET_DEVICE=tt` set **before** install

## Install (development)

```bash
export VLLM_TARGET_DEVICE=tt
cd vllm
pip install -e .
```

## Install (CI-style, with uv)

```bash
export VLLM_TARGET_DEVICE=tt
uv pip install vllm/ --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

## Verify

```bash
python -c "import vllm; print(vllm.__version__)"
```

## Note

The vLLM fork is at `tenstorrent/vllm` (branch: `dev`). It adds a TT hardware
backend to standard vLLM. The `VLLM_TARGET_DEVICE=tt` flag is critical — without
it, vLLM builds for CUDA.
