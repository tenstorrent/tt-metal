# Voxtral TTS Reference

Reference-only Voxtral TTS helpers live under `reference/`. This keeps the Mistral/vLLM-Omni request path isolated from TT-Transformers while TTNN bring-up can consume the model metadata, tokenizer flow, standalone functional blocks, and golden-output generation.

Voxtral uses Mistral-format assets rather than a standard Hugging Face Transformers checkpoint:

- `params.json` contains the text, audio model, and tokenizer dimensions.
- `tekken.json` is loaded through `mistral-common`.
- The default reference generation path is direct PyTorch on host CPU for TTNN bring-up.
- vLLM-Omni remains available as an optional CUDA-only comparison backend.

## Run

Generate deterministic golden tensors for TTNN unit tests:

```bash
python -m models.experimental.voxtraltts.reference.generate_golden
```

Run the host CPU end-to-end reference:

```bash
python -m models.experimental.voxtraltts.reference.demo_reference \
  --model mistralai/Voxtral-4B-TTS-2603 \
  --text "Paris is a beautiful city!" \
  --voice casual_male \
  --write-audio
```

The CPU backend is intentionally slow and currently supports preset voices. It loads the Mistral text backbone, acoustic transformer, and audio tokenizer directly from `consolidated.safetensors`. For lower-level TTNN bring-up, use `reference/functional.py`, whose functions take explicit tensors, weight dictionaries, and configs.

For the upstream vLLM-Omni backend on an NVIDIA/CUDA machine:

```bash
python -m models.experimental.voxtraltts.reference.demo_reference \
  --backend vllm \
  --model mistralai/Voxtral-4B-TTS-2603 \
  --text "Paris is a beautiful city!" \
  --voice casual_male \
  --streaming \
  --write-audio
```

The CPU reference expects `torch`, `transformers`, `mistral-common >= 1.10.0`, `safetensors`, `huggingface-hub`, and `soundfile` when writing WAV files. The optional vLLM backend expects `vllm-omni >= 0.18.0`, matching `vllm`, and CUDA.
