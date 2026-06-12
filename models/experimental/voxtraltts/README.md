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

### SDPA: native sliding window (default **on**)

By default, decoder transformer SDPA uses TTNN **native causal + sliding-window** attention:

- `is_causal=True` + `sliding_window_size` (per stage: 2 / 4 / 8 / 16 from config)
- Chunked `SDPAProgramConfig` (8×8 grid)
- **No** dense `[1, H, T, T]` DRAM mask — faster decode, especially at large `T`

**Trade-off:** ALiBi is omitted (TTNN does not allow `is_causal` and `attn_mask` together). Native SDPA matches structural causal+window attention only, not the full vLLM/Mistral ALiBi golden. Use dense-mask mode when you need production-accuracy PCC against the trained reference.

| Goal | Setting |
|------|---------|
| **Fast decode (default)** | *(no env var)* or ensure `VOXTRAL_AUDIO_TOKENIZER_SDPA_NATIVE_WINDOW_OFF` is unset |
| **Production ALiBi accuracy** | `export VOXTRAL_AUDIO_TOKENIZER_SDPA_NATIVE_WINDOW_OFF=1` |
| **Python (accuracy tests)** | `voxtral_audio_tokenizer_dense_mask_sdpa_optimizations()` |

```bash
# Default: native SDPA (fast)
pytest models/experimental/voxtraltts/tests/test_audio_tokenizer_native_sdpa_pcc.py -sv

# Dense ALiBi mask (production golden, slower)
VOXTRAL_AUDIO_TOKENIZER_SDPA_NATIVE_WINDOW_OFF=1 \
  pytest models/experimental/voxtraltts/tests/test_audio_tokenizer_decoder_transformer_block.py -sv
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

Uses default optimizations (native SDPA + Tier 1 matmul). For Tracy capture, see `tests/perf/README.md`.
