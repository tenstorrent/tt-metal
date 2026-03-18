# N300 Agentic Workflow Plan — TTNN Tool-Calling LLM

## Overview

An agentic system where an LLM orchestrator runs on a Tenstorrent N300 device
(24 GB DRAM, 2× Wormhole B0 chips) and dynamically calls TTNN-accelerated
specialist models as tools. All models are loaded once at startup and kept
resident in device DRAM. Traces are captured once and reused for all
subsequent inference calls.

---

## Hardware: N300

| Property | Value |
|----------|-------|
| Chips | 2× Wormhole B0 |
| DRAM | 2 × 12 GB = 24 GB total |
| Tensix cores | 2 × 80 = 160 |
| Connectivity | N300 internal Ethernet link |

---

## Model Selection

| Role | Model | Params | DRAM (BF16) | Hardware |
|------|-------|--------|-------------|----------|
| **LLM Orchestrator** | Llama 3.2 3B Instruct | 3B | ~6 GB | N300 |
| **STT** | Whisper distil-large-v3 | 756M | ~1.5 GB | N150+ |
| **TTS** | SpeechT5 TTS | 270M | ~0.6 GB | N300 |
| **Object Detection** | RetinaNet (N150) or OWL-ViT (N300) | ~50M / ~94M | ~0.2 / ~0.3 GB | N150/N300 |
| **VQA / Image captioning** | Qwen2.5-VL-3B-Instruct | 3B | ~6 GB* | N300 |
| **Extractive QA** | BERT Large | 340M | ~0.7 GB | N150+ |
| **Semantic Search** | SentenceBERT / BGE-Large | 110M / 330M | ~0.2 / ~0.7 GB | N150+ |
| **Image Classification** | ViT-base-patch16-224 | 86M | ~0.2 GB | N150+ |

> *VQA model is optional — swap Qwen2.5-VL for Gemma3-4B if preferred.
> Load VQA only if image-understanding tool is required; otherwise omit to save 6 GB.

### Recommended N300 Baseline (no VQA)

```
Llama 3B        6.0 GB
Whisper         1.5 GB
SpeechT5        0.6 GB
OWL-ViT         0.3 GB
BERT Large      0.7 GB
SentenceBERT    0.2 GB
ViT             0.2 GB
KV cache        0.5 GB
Traces          0.8 GB
───────────────────────
Total          10.8 GB / 24 GB   (13 GB headroom)
```

---

## Tool Definitions

Each tool exposed to the LLM via `apply_chat_template(tools=[...])`:

### 1. `transcribe_audio`
```python
{
    "name": "transcribe_audio",
    "description": "Converts an audio file (wav/mp3/flac) to text using Whisper STT. "
                   "Use when the user attaches an audio file or asks about spoken content.",
    "parameters": {
        "path": {"type": "string", "description": "Path to the audio file"}
    }
}
```
- **Backend:** `WhisperGenerator` — `models/demos/audio/whisper/tt/whisper_generator.py`
- **Returns:** `str` — transcript text
- **Trace:** Persistent decoder trace, captured on first call, reused for all subsequent audio files

### 2. `translate_audio`
```python
{
    "name": "translate_audio",
    "description": "Transcribes and translates audio to English. "
                   "Use when the user attaches non-English audio.",
    "parameters": {
        "path": {"type": "string", "description": "Path to the audio file"},
        "source_language": {"type": "string", "description": "ISO 639-1 language code, e.g. 'fr', 'de'"}
    }
}
```
- **Backend:** Same `WhisperGenerator` with `task='translate'`

### 3. `text_to_speech`
```python
{
    "name": "text_to_speech",
    "description": "Converts text to speech and saves as a wav file. "
                   "Use when the user explicitly wants an audio response.",
    "parameters": {
        "text": {"type": "string", "description": "Text to synthesize"},
        "output_path": {"type": "string", "description": "Output .wav file path"}
    }
}
```
- **Backend:** `models/experimental/speecht5_tts/demo_ttnn.py` — `generate_speech_long_text()`
- **Returns:** `str` — path to generated `.wav` file

### 4. `detect_objects`
```python
{
    "name": "detect_objects",
    "description": "Detects objects in an image based on a text query. "
                   "Use when the user attaches an image and asks what is in it.",
    "parameters": {
        "image_path": {"type": "string", "description": "Path to the image file"},
        "query": {"type": "string", "description": "What to look for, e.g. 'a person', 'cars'"}
    }
}
```
- **Backend:** `models/demos/wormhole/owl_vit/` — OWL-ViT zero-shot detection
- **Returns:** `list[dict]` — `[{label, score, bbox: [x1,y1,x2,y2]}, ...]`

### 5. `classify_image`
```python
{
    "name": "classify_image",
    "description": "Classifies what is shown in an image (ImageNet 21k classes). "
                   "Use for general image classification when detect_objects is too specific.",
    "parameters": {
        "image_path": {"type": "string", "description": "Path to the image file"}
    }
}
```
- **Backend:** `models/demos/vision/classification/vit/` — ViT-base
- **Returns:** `str` — top predicted class label with confidence

### 6. `answer_from_context`
```python
{
    "name": "answer_from_context",
    "description": "Extracts an answer from a provided text passage. "
                   "Use when the user asks a specific question about a document or long text.",
    "parameters": {
        "question": {"type": "string"},
        "context": {"type": "string", "description": "The passage to search for the answer"}
    }
}
```
- **Backend:** `models/demos/bert/` — BERT Large SQuAD2
- **Returns:** `str` — extracted answer span

### 7. `semantic_search`
```python
{
    "name": "semantic_search",
    "description": "Finds the most semantically similar documents to a query. "
                   "Use for retrieval-augmented generation (RAG) over a document corpus.",
    "parameters": {
        "query": {"type": "string"},
        "documents": {"type": "array", "items": {"type": "string"}},
        "top_k": {"type": "integer", "default": 3}
    }
}
```
- **Backend:** `models/demos/sentence_bert/` — SentenceBERT embeddings + cosine similarity
- **Returns:** `list[str]` — top-k most relevant documents

---

## Agentic Loop Implementation

### Architecture

```
                    ┌──────────────────────────┐
                    │    ORCHESTRATOR (Python)  │
                    │                          │
    User input ────►│  messages = [...]        │
    (text/audio/    │                          │
     image)         │  1. detect input type    │
                    │  2. inject [ATTACHMENT]  │
                    │     tag into user msg    │
                    └──────────┬───────────────┘
                               │
                     ┌─────────▼─────────┐
                     │   LLM (Llama 3B)  │  ◄── tool schema via apply_chat_template
                     │   on N300 DRAM    │
                     └─────────┬─────────┘
                               │ emits tool_call JSON or final text
                               │
              ┌────────────────▼────────────────────┐
              │         TOOL DISPATCHER              │
              │                                      │
              │  transcribe_audio → WhisperGenerator │
              │  detect_objects   → OWL-ViT          │
              │  classify_image   → ViT              │
              │  answer_from_context → BERT Large    │
              │  semantic_search  → SentenceBERT     │
              │  text_to_speech   → SpeechT5         │
              └────────────────┬────────────────────┘
                               │ tool result (str/list)
                               │
                    ┌──────────▼───────────────┐
                    │  messages.append(result)  │
                    │  loop back to LLM         │
                    └───────────────────────────┘
```

### Startup Sequence

```python
import ttnn
from models.demos.minimax_m2.agentic.loader import load_all_models

# Open device once — all models share it
device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))  # N300: 1 row, 2 cols

# Load all models into DRAM (one-time, ~30-90s)
models = load_all_models(device)
# models.llm      — Llama 3B, with prefill+decode traces
# models.whisper  — WhisperGenerator, decoder trace captured lazily on first call
# models.speecht5 — SpeechT5, weights resident
# models.owlvit   — OWL-ViT, weights resident
# models.bert     — BERT Large, weights resident
# models.sbert    — SentenceBERT, weights resident
# models.vit      — ViT, weights resident

# Run agent — no more loading after this
run_agentic_loop(models, device)
```

### Conversation / Tool Call Loop

```python
def run_agentic_loop(models, device):
    tokenizer = models.llm.tokenizer
    messages = []

    while True:
        user_input, attachments = get_user_input()

        # Build user message with attachment tags
        content = user_input
        for path in attachments:
            ext = Path(path).suffix.lower()
            if ext in (".wav", ".mp3", ".flac", ".m4a"):
                content += f"\n[AUDIO_ATTACHMENT: {path}]"
            elif ext in (".jpg", ".jpeg", ".png", ".webp"):
                content += f"\n[IMAGE_ATTACHMENT: {path}]"

        messages.append({"role": "user", "content": content})

        # Agentic loop — keep calling LLM until it gives a final answer
        while True:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tools=TOOL_SCHEMAS,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt"
            )

            # LLM forward — reuses pre-captured traces
            output_text = models.llm.generate(input_ids)

            if is_tool_call(output_text):
                tool_name, args = parse_tool_call(output_text)
                result = dispatch_tool(tool_name, args, models)

                messages.append({"role": "assistant", "content": output_text})
                messages.append({"role": "tool", "name": tool_name, "content": str(result)})
                # Loop: re-prefill with tool result appended
            else:
                # Final answer
                messages.append({"role": "assistant", "content": output_text})
                deliver_response(output_text)
                break


def dispatch_tool(name: str, args: dict, models) -> str:
    if name == "transcribe_audio":
        return models.whisper.generate(args["path"])
    elif name == "translate_audio":
        return models.whisper.generate(args["path"], task="translate")
    elif name == "text_to_speech":
        return models.speecht5.synthesize(args["text"], args.get("output_path", "/tmp/response.wav"))
    elif name == "detect_objects":
        detections = models.owlvit.run(args["image_path"], args["query"])
        return format_detections(detections)
    elif name == "classify_image":
        return models.vit.classify(args["image_path"])
    elif name == "answer_from_context":
        return models.bert.qa(args["question"], args["context"])
    elif name == "semantic_search":
        return models.sbert.search(args["query"], args["documents"], args.get("top_k", 3))
    else:
        return f"Unknown tool: {name}"
```

---

## System Prompt

```
You are a helpful AI assistant running on Tenstorrent N300 hardware.
You have access to specialist AI tools for audio, vision, and text processing.

ATTACHMENT HANDLING RULES:
- If you see [AUDIO_ATTACHMENT: path], you MUST call transcribe_audio(path) before answering.
- If you see [IMAGE_ATTACHMENT: path] and the user asks what's in it, call detect_objects or classify_image.
- If the user wants an audio response, call text_to_speech with your final answer text.
- If the user provides a long document and asks a specific question, call answer_from_context.

Always explain what you are doing before calling a tool. After receiving tool results,
use them to construct your final answer — do not call the same tool twice for the same input.
```

---

## Trace / Warmup Strategy

| Model | When traced | Trace type | Reuse |
|-------|------------|-----------|-------|
| Llama 3B prefill | Startup warmup | Per supported seq length (128, 256, 512, 1024, 2048) | Every agent turn |
| Llama 3B decode | Startup warmup | Single token forward | Every decode step |
| Whisper decoder | First audio call | Persistent cross-attn trace | Every subsequent audio file |
| OWL-ViT / ViT / BERT / SentenceBERT | No trace needed | Single-pass encoder | Each call |
| SpeechT5 | No trace needed | Autoregressive TTNN forward | Each synthesis |

Warmup cost: ~60-120 seconds at startup. Zero overhead between agentic turns.

---

## File Structure (To Be Created)

```
models/demos/minimax_m2/
├── ARCHITECTURE.md          ✅ done
├── BRINGUP_LOG.md           ✅ done
├── AGENTIC_PLAN.md          ✅ this file
├── reference/               ✅ done — PyTorch reference + goldens
├── tests/                   ✅ done — TTNN block tests
├── tt/                      ✅ done — TTNN implementation (MoE PCC in progress)
└── agentic/                 ← to be created
    ├── __init__.py
    ├── loader.py            ← load_all_models() — opens device, loads all tools
    ├── orchestrator.py      ← run_agentic_loop() — main conversation loop
    ├── tools.py             ← TOOL_SCHEMAS + dispatch_tool()
    ├── tool_wrappers/
    │   ├── whisper_tool.py  ← wraps WhisperGenerator
    │   ├── speecht5_tool.py ← wraps SpeechT5 demo
    │   ├── owlvit_tool.py   ← wraps OWL-ViT demo
    │   ├── vit_tool.py      ← wraps ViT classification
    │   ├── bert_tool.py     ← wraps BERT Large QA
    │   └── sbert_tool.py    ← wraps SentenceBERT
    └── demo.py              ← CLI entry point
```

---

## Example Agentic Interaction

```
User: "What did I say in this recording?" + [voice_memo.wav]

LLM sees:  "What did I say in this recording?\n[AUDIO_ATTACHMENT: voice_memo.wav]"
LLM emits: {"tool": "transcribe_audio", "args": {"path": "voice_memo.wav"}}

Whisper:   "We need to finalize the Q3 roadmap by Friday."

LLM emits: "You said: 'We need to finalize the Q3 roadmap by Friday.'"
```

```
User: "What objects are in this photo?" + [office.jpg]

LLM sees:  "What objects are in this photo?\n[IMAGE_ATTACHMENT: office.jpg]"
LLM emits: {"tool": "detect_objects", "args": {"image_path": "office.jpg", "query": "objects in the scene"}}

OWL-ViT:   [{"label": "laptop", "score": 0.91, "bbox": [...]},
             {"label": "chair",  "score": 0.87, "bbox": [...]},
             {"label": "desk",   "score": 0.83, "bbox": [...]}]

LLM emits: "The photo contains a laptop (91% confidence), a chair (87%), and a desk (83%)."
```

```
User: "Summarize this document and read it back to me." + [long_document.txt]

LLM emits: <summarizes document as text>
LLM emits: {"tool": "text_to_speech", "args": {"text": "<summary>", "output_path": "/tmp/summary.wav"}}

SpeechT5:  "/tmp/summary.wav"
LLM emits: "Here is the summary. Audio saved to /tmp/summary.wav."
```

---

## Current Status & Next Steps

### MiniMax-M2 Bringup (Galaxy — separate from N300 agentic work)

| Block | Status | PCC |
|-------|--------|-----|
| RMSNorm | ✅ Passing | 0.9999 |
| Partial RoPE | ✅ Passing | ~0.9999 |
| Attention (TP=4) | ✅ Passing | 0.9946 |
| MoE (EP=8, TP=4) | ❌ Failing | 0.940 |
| Full decoder layer | 🔄 Blocked on MoE | — |
| Full model | 🔄 Blocked | — |

### N300 Agentic Workflow (this plan)

- [ ] Create `models/demos/minimax_m2/agentic/` directory structure
- [ ] Implement `loader.py` — open N300 device, load all 7 models
- [ ] Implement `tool_wrappers/whisper_tool.py` — wrap `WhisperGenerator`
- [ ] Implement `tool_wrappers/owlvit_tool.py` — wrap OWL-ViT demo
- [ ] Implement `tool_wrappers/speecht5_tool.py` — wrap SpeechT5 `generate_speech_long_text`
- [ ] Implement `tool_wrappers/bert_tool.py` — wrap BERT Large QA
- [ ] Implement `tool_wrappers/sbert_tool.py` — wrap SentenceBERT
- [ ] Implement `tool_wrappers/vit_tool.py` — wrap ViT classification
- [ ] Implement `tools.py` — TOOL_SCHEMAS + dispatch_tool()
- [ ] Implement `orchestrator.py` — conversation loop with `apply_chat_template`
- [ ] Implement `demo.py` — CLI entry point
- [ ] Write pytest tests for each tool wrapper (PCC not applicable; test output format)
- [ ] Integration test: audio → STT → LLM → TTS end-to-end
- [ ] Integration test: image → detect → LLM multi-step
