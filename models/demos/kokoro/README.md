# Kokoro-82M (reference bring-up)

This directory is a **reference bring-up** for the upstream **Kokoro-82M** open-weight TTS model (PyTorch).
It is intended as the first step before any TT hardware port, matching the “reference first” pattern used in
other `tt-metal/models/experimental/*` model bring-ups.

## Upstream references

- https://huggingface.co/hexgrad/Kokoro-82M
- https://github.com/hexgrad/kokoro

## What is the “main” runnable Python entry?

Kokoro’s primary inference entrypoint is the `KPipeline` API in the `kokoro` Python package.
In this repo, the equivalent “main file” for reference execution is:

- `demo/cpu_demo.py`

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

Run:

```bash
python models/experimental/kokoro/demo/cpu_demo.py \
  --text "Kokoro is an open-weight text to speech model." \
  --voice af_heart \
  --output kokoro_reference.wav
```
