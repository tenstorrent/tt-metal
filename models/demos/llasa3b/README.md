# Llasa-3B

## Platforms:
    N300 (WH)

## Introduction
[Llasa-3B](https://huggingface.co/HKUSTAudio/Llasa-3B) is a text-to-speech (TTS) model that treats speech synthesis as a language modeling task. Built on the LLaMA-3.2-3B architecture, it extends the vocabulary with 65,536 [XCodec2](https://huggingface.co/HKUSTAudio/xcodec2) speech tokens, enabling autoregressive generation of speech from text input.

This implementation runs the Llasa-3B model on Tenstorrent N300 hardware using the existing `tt_transformers` infrastructure. The model is architecturally identical to LLaMA-3.2-3B — no custom TTNN modules were needed.

### Key Features
- **Zero-shot TTS**: Generate natural speech from text alone (TTNN + PyTorch reference)
- **Prompted TTS (Voice Cloning)**: Clone a voice from a short audio sample — no training required (TTNN + PyTorch reference)
- **Full audio pipeline**: Text → TTNN inference → speech tokens → XCodec2 decode → 16kHz WAV file
- **Large vocabulary support**: Handles the extended 193k token vocabulary (128k text + 65k speech)
- **Supports English and Chinese** text input

### Model Specifications
| Parameter | Value |
|-----------|-------|
| Base Architecture | LLaMA-3.2-3B (`LlamaForCausalLM`) |
| Hidden Dim | 3072 |
| Layers | 28 |
| Attention Heads | 24 (8 KV heads, GQA) |
| Total Vocab Size | ~193,800 |
| Max Sequence Length | 2048 (training limit) |
| Audio Output | 16kHz mono WAV |
| Audio Token Rate | 50 speech tokens per second of audio |

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- N300 Wormhole device

## Setup

### 1. Install Dependencies
```bash
source python_env/bin/activate
uv pip install xcodec2==0.1.5 soundfile
```

### 2. Fix XCodec2 for CPU-only Environments
The `xcodec2==0.1.5` package has two issues that must be patched:

**Fix 1: Dead `torchaudio` import** — An unused `import torchaudio` in `bs_roformer5.py` causes an `OSError` in CPU-only environments:
```bash
# File: python_env/lib/python3.10/site-packages/xcodec2/vq/bs_roformer5.py
# Comment out line 5:
#   import torchaudio  →  # import torchaudio
```

**Fix 2: SnakeBeta weight name mismatch** — Required for prompted TTS (voice cloning). The package renamed the `SnakeBeta` activation parameter from `beta` to `bias`, but the HuggingFace checkpoint still stores it as `beta`. Without this fix, the XCodec2 encoder loads random weights and produces garbage VQ codes:
```bash
# File: python_env/lib/python3.10/site-packages/xcodec2/vq/activations.py
# In the SnakeBeta class (around line 90), rename self.bias → self.beta:
#   Line 97:  self.bias = Parameter(...)  →  self.beta = Parameter(...)
#   Line 100: self.bias = Parameter(...)  →  self.beta = Parameter(...)
#   Line 103: self.bias.requires_grad    →  self.beta.requires_grad
#   Line 114: beta = self.bias.unsqueeze →  beta = self.beta.unsqueeze
```

**Clear bytecode cache** after applying fixes:
```bash
find python_env/lib/python3.10/site-packages/xcodec2 -name "*.pyc" -delete
```

### 3. Download Model Weights
The model weights are automatically downloaded from HuggingFace on first run. To pre-download:
```bash
export HF_MODEL=HKUSTAudio/Llasa-3B
```

TTNN weight caches will be generated on first run and stored at `model_cache/HKUSTAudio/Llasa-3B/N300/`.

## How to Run

### Zero-Shot TTS (TTNN Demo)
Generate speech from text on Tenstorrent hardware:
```bash
export HF_MODEL=HKUSTAudio/Llasa-3B
pytest models/demos/llasa3b/demo/llasa_demo.py -s -k "test_llasa_tts"
```

**Process:**
1. Tokenize input text (from `models/demos/llasa3b/demo/input_data.json`)
2. Run prefill and autoregressive decode on TTNN
3. Extract speech token IDs from generated output
4. Decode speech tokens to waveform using XCodec2 (on CPU)
5. Save audio to `models/demos/llasa3b/demo/output/llasa_output.wav`

### Prompted TTS / Voice Cloning (TTNN Demo)
Clone the voice from `Anna.wav` and generate new speech:
```bash
export HF_MODEL=HKUSTAudio/Llasa-3B
# Modify ANNA_PROMPT_TEXT and ANNA_TARGET_TEXT in llasa_demo.py to customize
pytest models/demos/llasa3b/demo/llasa_demo.py -s -k "test_llasa_tts_prompted"
```
Output: `models/demos/llasa3b/demo/output/llasa_output_prompted.wav` (contains only the generated speech).

### PyTorch Reference

#### Zero-Shot TTS (text only)
```bash
python models/demos/llasa3b/reference/llasa_reference.py \
    --text "Hello, this is a test of the Llasa speech synthesis model." \
    --output_dir reference_output \
    --device cpu
```

#### Prompted TTS / Voice Cloning
Clone a voice from a short audio sample. No training required — just provide:
- A 16kHz WAV recording of the target voice (2–3 seconds ideal)
- The transcript of what's spoken in that recording
- The new text you want spoken in that voice

```bash
# Download the official example prompt
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('HKUSTAudio/Llasa-3B', 'Anna.wav', local_dir='.')"

# Run voice cloning
python models/demos/llasa3b/reference/llasa_reference.py \
    --text "Dealing with family secrets is never easy. Yet, sometimes, omission is a form of protection." \
    --prompt_text "A chance to leave him alone, but... No. She just wanted to see him again. Anna, you don't know how it feels to lose a sister." \
    --prompt_wav Anna.wav \
    --output_dir reference_output \
    --device cpu
```

> **Note on prompt length:** The model was trained with a total sequence limit of 2048 tokens. Prompt audio is automatically truncated to ~20 seconds (1000 speech tokens) to leave room for generation. Use clear audio clips for best results.

## Performance

Measured on N300 (2x Wormhole):

| Metric | Value |
|--------|-------|
| Prefill throughput | ~1,360 tokens/sec |
| Decode throughput | ~16.4 tokens/sec/user |
| Typical speech tokens generated | 300–500 |
| Audio output sample rate | 16kHz |
| vs. PyTorch CPU | ~10x faster |

> **Note:** Current performance is not yet optimized. Trace mode and 2CQ optimizations are planned for future improvements.

## Details

### How It Works

Llasa-3B treats TTS as a next-token prediction task:

```
Input text ──→ Tokenizer ──→ [text tokens] ──→ LLaMA-3.2-3B (TTNN) ──→ [speech tokens] ──→ XCodec2 ──→ WAV
                              with chat template    autoregressive decode    <|s_xxxxx|>         CPU decode
```

1. The input text is wrapped in a chat template with special markers (`<|TEXT_UNDERSTANDING_START|>`, `<|SPEECH_GENERATION_START|>`, etc.)
2. The model autoregressively generates speech tokens (e.g., `<|s_12345|>`) until it produces `<|SPEECH_GENERATION_END|>`
3. Speech token IDs are extracted and decoded by XCodec2 into a 16kHz audio waveform

For **prompted TTS** (voice cloning), an additional step encodes the prompt audio into speech tokens using XCodec2's encoder, and those tokens are prepended to the assistant's message so the model continues generating in the same voice.

### Token Budget
The model was trained with `max_length=2048`. This budget is shared across all input and output tokens:
- ~30 tokens for chat template overhead
- ~N tokens for input text (~1 token per word)
- ~M tokens for prompt speech tokens (50 tokens per second of prompt audio)
- ~R tokens for generated speech tokens (50 tokens per second of output audio)

The prompt audio is automatically truncated to 1000 tokens (~20 seconds) to ensure it fits within the context window while leaving room for generation.

### Implementation Notes

- **No custom TTNN code**: Llasa-3B reuses the existing `tt_transformers` infrastructure entirely (model, attention, MLP, embedding, LM head, generator)
- **Large vocab handling**: The extended vocabulary (193k vs 128k for standard LLaMA) required a fix to the LM head concatenation step in `models/tt_transformers/tt/lm_head.py` to use DRAM instead of L1 memory for the intermediate buffer
- **XCodec2 runs on CPU**: Both the encoder (for prompt audio) and decoder (for output audio) are lightweight and run on the host CPU

### Folder Contents
```
models/demos/llasa3b/
├── README.md                       # This file
├── PLAN.md                         # Development notes and bring-up plan
├── model_params/
│   └── config.json                 # Model configuration (vocab_size=193800)
├── tt/
│   ├── __init__.py
│   └── llasa_utils.py              # Speech token utils, chat template, XCodec2 decode
├── demo/
│   ├── llasa_demo.py               # Main TTNN demo (pytest entry point)
│   ├── input_data.json             # Sample text prompts
│   └── output/                     # Generated audio files
├── reference/
│   └── llasa_reference.py          # PyTorch reference (zero-shot + prompted TTS)
└── tests/                          # (future) PCC and accuracy tests
```
