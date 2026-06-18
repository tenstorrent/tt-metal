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

## Device mesh (P150 and BH QB2 1×4)

Voxtral selects the runtime device in ``open_voxtral_runtime_mesh()`` (demo and tests):

The compute mesh is **hardware-aware**: when ``VOXTRAL_COMPUTE_MESH_SHAPE`` is unset the natural
mesh for the detected host is used; when set, it is honored.

| Host | Default compute (env unset) | Behavior |
|------|-----------------|----------|
| **P150** (1 card) | 1×1 | ``CreateDevice(0)`` — unchanged single-card path |
| **BH QB2** (4 cards) | 1×4 | ``open_mesh_device(1×4)`` host fabric; tensor-parallel text on the full mesh, acoustic and audio tokenizer replicate weights |
| **BH QB2** (4 cards) | 1×1 submesh (opt-in) | Set ``VOXTRAL_COMPUTE_MESH_SHAPE=1,1`` to pin compute to a single rank |

Trace replay is **ON by default** on every topology (it removes the host-dispatch gaps that
dominate decode). 2CQ overlap and the acoustic-FM trace are auto-tuned per topology and fall back
to single-CQ / untraced FM only on the BH QB2 1×1 submesh (where they diverge). Override with
``VOXTRAL_DECODE_TRACE`` / ``VOXTRAL_DECODE_TRACE_2CQ`` / ``VOXTRAL_ACOUSTIC_FM_TRACE``.

```bash
# Demo (auto-detects mesh: 1×1 on P150, full 1×4 on QB2)
python models/experimental/voxtraltts/demo/demo.py --text "Hello" --output-dir out

# Pin QB2 compute to a single rank (1×1 submesh)
export VOXTRAL_COMPUTE_MESH_SHAPE=1,1
python models/experimental/voxtraltts/demo/demo.py --text "Hello" --output-dir out

# E2E PCC (uses voxtral ``device`` fixture — P150 or QB2)
pytest models/experimental/voxtraltts/tests/pcc/test_voxtral_e2e_pcc.py -sv --timeout=3600
```

Optional: ``VOXTRAL_DEVICE_ID`` selects the PCIe rank for single-card compute on QB2 (default ``0``).
