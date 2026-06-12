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

## Audio tokenizer decode optimizations

`VoxtralTTAudioTokenizer` uses `voxtral_audio_tokenizer_default_optimizations()` unless you pass a custom `AudioTokenizerOptimizations` preset.

### SDPA: dense ALiBi (default **on**)

By default, decoder transformer SDPA uses a dense **ALiBi + causal + sliding-window** ``attn_mask``
(production accuracy; clean waveform vs CPU golden). Matmul Tier-1 program configs stay enabled.

**Optional native SDPA** (``is_causal=True`` + ``sliding_window_size``, no ALiBi mask) is faster but
adds audible hiss / ~15% PCC drop — opt in only for perf bring-up.

| Goal | Setting |
|------|---------|
| **Production decode (default)** | *(no env var)* — dense ALiBi |
| **Fast native SDPA (perf)** | `export VOXTRAL_AUDIO_TOKENIZER_SDPA_NATIVE_WINDOW=1` |
| **Python (accuracy tests)** | `voxtral_audio_tokenizer_dense_mask_sdpa_optimizations()` |

```bash
# Production path (dense ALiBi)
pytest models/experimental/voxtraltts/tests/test_audio_tokenizer_decoder_transformer_block.py -sv

# Native SDPA perf path (no ALiBi)
pytest models/experimental/voxtraltts/tests/test_audio_tokenizer_native_sdpa_pcc.py -sv
```

### Matmul Tier 1 program configs (default **on**)

Explicit 2D multicast matmul configs per decoder forward. Disabled when `T > 6400` (block 7) to avoid L1 OOM.

```bash
export VOXTRAL_AUDIO_TOKENIZER_MATMUL_PROGCFG_OFF=1   # opt out
```

### Perf smoke test

```bash
pytest models/experimental/voxtraltts/tests/test_audio_tokenizer_opt.py -sv --timeout=0
```

Uses default optimizations (dense ALiBi SDPA + Tier 1 matmul; set ``VOXTRAL_AUDIO_TOKENIZER_SDPA_NATIVE_WINDOW=1`` for native SDPA perf). For Tracy capture, see `tests/perf/README.md`.
