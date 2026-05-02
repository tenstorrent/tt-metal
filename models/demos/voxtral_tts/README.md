# Voxtral-4B-TTS-2603 on Tenstorrent N150

Text-to-speech model by Mistral AI, running on Tenstorrent N150 (single Wormhole B0).

## Status

**Phase 3 (TTNN) COMPLETE** — all 11 TTNN tests + 20 reference tests pass.
End-to-end TTS pipeline produces audio: "Hello." → 0.40s waveform at 24kHz.

See [BRINGUP_LOG.md](BRINGUP_LOG.md) for detailed test results.

## Architecture

Three-component TTS pipeline:

| Component | Params | TTNN Status |
|-----------|--------|-------------|
| Text Decoder Backbone | 3.4B | On-device (N150, BF16 attention + BF8 MLP) |
| Acoustic Flow-Matching Transformer | 390M | On-device (N150, BF16) |
| Voxtral Codec Decoder | 300M | CPU Phase 1 (see Known Limitations) |

**Inference flow:**
1. Voice embedding (pre-computed, loaded from `.pt` file) + text tokens
2. Text decoder prefill → hidden states per position
3. ODE solve (8 Euler steps × 2 CFG passes) → acoustic tokens [N, 36]
4. Codec decoder: tokens → 24kHz waveform

## Requirements

- Hardware: Tenstorrent N150 (12GB Wormhole B0)
- Model: `mistralai/Voxtral-4B-TTS-2603` (~8GB BF16)
- Max sequence length: 4096 (voice + text positions)
- Python: 3.10, PyTorch 2.x, TTNN

## Setup

```bash
cd tt-metal
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd):$(pwd)/models
export ARCH_NAME=wormhole_b0
source python_env/bin/activate

# Download model weights (once)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('mistralai/Voxtral-4B-TTS-2603', local_dir='/tmp/voxtral_tts')
"
export VOXTRAL_MODEL_DIR=/tmp/voxtral_tts
```

## Demo

```bash
python3 models/demos/voxtral_tts/demo/demo.py \
    --text "Hello, this is a test of the Voxtral TTS model." \
    --voice casual_male \
    --output output.wav
```

## Test Commands

### Reference tests (no device needed):
```bash
pytest models/demos/voxtral_tts/reference/test_functional.py -v -s
```

### TTNN block PCC tests (N150):
```bash
pytest models/demos/voxtral_tts/tests/test_tt_text_decoder.py -v -s
pytest models/demos/voxtral_tts/tests/test_tt_acoustic_transformer.py -v -s
pytest models/demos/voxtral_tts/tests/test_tt_codec_decoder.py -v -s
```

### Integration test (N150):
```bash
pytest models/demos/voxtral_tts/tests/test_integration.py -v -s
```

### Full suite:
```bash
pytest models/demos/voxtral_tts/ -v --timeout=600
```

## Profiling Commands

```bash
# Tracy profiling of text decoder attention
python -m tracy -p -v -r models/demos/voxtral_tts/demo/profile_attention.py

# Run block profiles
bash models/demos/voxtral_tts/demo/run_block_profiles.sh
```

## Server Commands

```bash
# (Phase 5 — not yet implemented)
# vllm-omni TTS server will be added in Phase 5
```

## PCC Results (Phase 3, N150)

| Block | PCC |
|-------|-----|
| text_attention_layer0 | 0.998867 |
| text_mlp_layer0 | 0.999927 |
| decoder_block_layer0 | 0.999996 |
| acoustic_transformer_velocity | 0.999669 |
| codec_decoder_waveform | 1.000000 (CPU ref) |

## Known Limitations

1. **Codec decoder on CPU (Phase 1)**: The 300M-parameter codec uses causal Conv1D,
   ALiBi+sliding-window attention, and ConvTranspose1D — non-standard TTNN ops.
   Phase 2 will move transformer blocks to device for improved throughput.

2. **Simplified TTS inference**: The semantic token prediction uses the acoustic
   transformer's semantic head on the final ODE x_continuous rather than proper
   autoregressive decoding with the text decoder. Full autoregressive decode is
   in Phase 4 scope.

3. **max_seq_len=4096**: KV cache is sized for TTS sequences (voice ~500 + text ~1000).
   Increase in config for longer inputs (will require more DRAM).

4. **License**: CC BY-NC 4.0 — non-commercial use only without Mistral agreement.

5. **SDPA prefill grid**: N150 has 7×8=56 cores. SDPA uses (8,4) grid = 32 cores.

## File Structure

```
models/demos/voxtral_tts/
├── ARCHITECTURE.md          # Block inventory, N150 layout, non-standard op strategies
├── BRINGUP_LOG.md           # Phase-by-phase results and debug history
├── README.md                # This file
├── reference/
│   ├── functional.py        # Pure PyTorch reference (CPU)
│   ├── test_functional.py   # 20 reference tests
│   ├── save_goldens.py      # Golden tensor generation
│   └── golden/              # 26 .pt golden tensor files
├── tt/
│   ├── model_config.py      # VoxtralTTSConfig (N150 settings)
│   ├── load_checkpoint.py   # Weight loading + weight_norm fusing
│   ├── attention.py         # TtVoxtralTextAttention (BF16, GQA, RoPE)
│   ├── mlp.py               # TtVoxtralTextMLP (SwiGLU)
│   ├── acoustic_transformer.py  # TtVoxtralAcousticTransformer + ODE solver
│   ├── codec_decoder.py     # TtVoxtralCodecDecoder (Phase 1: CPU)
│   └── model.py             # VoxtralTTSModel (full pipeline)
└── tests/
    ├── test_tt_text_decoder.py       # 4 tests: attention, MLP, block, full
    ├── test_tt_acoustic_transformer.py  # 2 tests: velocity PCC, ODE range
    ├── test_tt_codec_decoder.py      # 3 tests: shape, waveform PCC, block0
    └── test_integration.py           # 2 tests: end-to-end, reference PCC
```
