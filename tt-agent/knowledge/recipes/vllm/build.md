# Install vLLM with TT backend

## Prerequisites

- tt-metal built and importable (`import ttnn` works)
- `uv` installed (see `recipes/developer-setup.md`)

## Install

From the vllm repo root:

```bash
bash tt_metal/install-vllm-tt.sh
```

This sets `VLLM_TARGET_DEVICE=tt` and runs `uv pip install -e .` with the
correct index flags.

## Verify

```bash
python -c "import vllm; print(vllm.__version__)"
```

## Note

The vLLM fork is at `tenstorrent/vllm` (branch: `dev`). The TT backend is
selected by `VLLM_TARGET_DEVICE=tt` — without it, vLLM builds for CUDA.
