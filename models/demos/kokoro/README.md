# Kokoro-82M (reference bring-up)

This directory is a **reference bring-up** for the upstream **Kokoro-82M** open-weight TTS model (PyTorch).
It is intended as the first step before any TT hardware port, matching the “reference first” pattern used in
other `tt-metal/models/experimental/*` model bring-ups.

## Upstream references

- https://huggingface.co/hexgrad/Kokoro-82M
- https://github.com/hexgrad/kokoro

## What is the “main” runnable Python entry?

Kokoro’s primary inference entrypoint is the `KPipeline` API in the `kokoro` Python package.
In this repo, the checked-in demo is:

- `demo/cpu_demo.py` — upstream `KModel` + pipeline (downloads weights via upstream).

The **repo-owned torch** stack (`reference/kokoro_full_model.py` and per-block modules loaded from Hugging Face) is validated by **`tests/test_reference_vs_official.py`** (and other tests under `tests/`), similar to how `speecht5_tts` exercises its `reference/` code via PCC tests rather than a separate “torch-only” demo script.

## Reference implementation (tt-metal style)

To mirror the `models/experimental/speecht5_tts/reference/` layout, this bring-up includes
`models/demos/kokoro/reference/` wrappers that expose a stable reference interface:

- `reference/kokoro_model.py`: wraps upstream `kokoro.model.KModel` and exposes submodules
- `reference/kokoro_pipeline.py`: wraps upstream `kokoro.pipeline.KPipeline` (G2P/chunking/voices)

## CPU / CUDA support

Kokoro runs on **CPU or CUDA via PyTorch** (device selection follows `torch.cuda.is_available()` by default).
Some text processing relies on `espeak-ng` being available on the system.

## Run the reference bring-up

Install dependencies:

```bash
cd tt-metal
source python_env/bin/activate
pip install "kokoro>=0.9.2"
sudo apt-get update && sudo apt-get install -y espeak-ng
```

Upstream `KModel` demo:

```bash
cd tt-metal
export PYTHONPATH=$(pwd)
python models/demos/kokoro/demo/cpu_demo.py \
  --text "Kokoro is an open-weight text to speech model." \
  --voice af_heart \
  --output kokoro_reference.wav
```

## Tests

`tests/test_reference_vs_official.py` compares the repo-owned HF reference modules against the official `kokoro` `KModel` (same pattern as Qwen3 TTS `test_reference_vs_official`). Run:

```bash
cd tt-metal
export PYTHONPATH=$(pwd)
pytest models/demos/kokoro/tests/test_reference_vs_official.py
```

Or run all sections with prints: `python models/demos/kokoro/tests/test_reference_vs_official.py`.
