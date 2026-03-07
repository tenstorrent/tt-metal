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

### 2. Download Model Weights
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
pytest models/demos/llasa3b/demo/llasa_demo.py -s -k "test_llasa_zero_shot"
```

**Process:**
1. Tokenize input text (from `models/demos/llasa3b/demo/input_data.json`)
2. Run prefill and autoregressive decode on TTNN
3. Extract speech token IDs from generated output
4. Decode speech tokens to waveform using XCodec2 (on CPU)
5. Save audio to `models/demos/llasa3b/demo/output/llasa_output_{i}.wav`

### Prompted TTS / Voice Cloning (TTNN Demo)
Clone the voice from `Anna.wav` and generate new speech:
```bash
export HF_MODEL=HKUSTAudio/Llasa-3B
# Modify ANNA_PROMPT_TEXT and ANNA_TARGET_TEXT in llasa_demo.py to customize
pytest models/demos/llasa3b/demo/llasa_demo.py -s -k "test_llasa_voice_cloning"
```
Output: `models/demos/llasa3b/demo/output/llasa_output_prompted.wav` (contains only the generated speech).


> **Note on prompt length:** The model was trained with a total sequence limit of 2048 tokens. Prompt audio is automatically truncated to ~20 seconds (1000 speech tokens) to leave room for generation. Use clear audio clips for best results.


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


The prompt audio is automatically truncated to 1000 tokens (~20 seconds) to ensure it fits within the context window while leaving room for generation.

### Implementation Notes

- **No custom TTNN code**: Llasa-3B reuses the existing `tt_transformers` infrastructure entirely (model, attention, MLP, embedding, LM head, generator)
- **Large vocab handling**: The extended vocabulary (193k vs 128k for standard LLaMA) required a fix to the LM head concatenation step in `models/tt_transformers/tt/lm_head.py` to use DRAM instead of L1 memory for the intermediate buffer


### Folder Contents
```
models/demos/llasa3b/
├── README.md                       # This file
├── model_params/
│   └── config.json                 # Model configuration (vocab_size=193800)
├── tt/
│   ├── __init__.py
│   ├── llasa_pipeline.py           # Core decoding loop and text token generation
│   └── llasa_utils.py              # Speech token utils, chat template, XCodec2 encode/decode
├── demo/
│   ├── llasa_demo.py               # Main TTNN demo (pytest entry point)
│   ├── input_data.json             # Sample text prompts
│   ├── prompts/                    # Downloaded prompt audio for voice cloning
│   └── output/                     # Generated audio files
├── reference/
│   └── llasa_reference.py          # PyTorch reference (zero-shot + prompted TTS)
└── tests/
    ├── test_llasa_lm_head.py       # Isolated tests for LM head specifically
    └── test_llasa_model.py         # Transformer building tests
```
