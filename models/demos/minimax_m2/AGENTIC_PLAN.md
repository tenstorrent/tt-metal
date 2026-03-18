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

### Overview

| Model | When traced | Trace type | Reuse across turns |
|-------|------------|-----------|-------------------|
| Llama 3B prefill | Startup `warmup_model_prefill()` | One trace per seq length bucket | Every re-prefill after tool result |
| Llama 3B decode | Startup `warmup_model_prefill()` | Single 1-token forward | Every decode step, every turn |
| Whisper decoder | Lazily on first audio call | Persistent cross-attn trace | Every subsequent audio file |
| OWL-ViT / ViT / BERT / SentenceBERT | No trace — single-pass encoder | N/A | Each call runs normally |
| SpeechT5 | No trace — chunked autoregressive | N/A | Each synthesis runs normally |

Warmup cost: ~60–120 seconds at startup. After that, zero re-warmup cost between agentic turns.

### LLM (Llama 3B via `tt_transformers` Generator)

The `Generator` class in `models/tt_transformers/tt/generator.py` handles warmup
and trace capture automatically via `warmup_model_prefill()`.

```python
from models.tt_transformers.tt.generator import Generator

generator = Generator(model, model_args, mesh_device)

# Called ONCE at startup — captures all prefill + decode traces
kv_cache = generator.initialize_kv_cache()
generator.warmup_model_prefill(
    kv_cache=kv_cache,
    enable_trace=True,           # capture Metal traces for decode
    can_sample_on_device=True,
    non_greedy_decoding_on_device=False,
)
```

What `warmup_model_prefill()` does internally:
1. Iterates over `model_args.get_warmup_prefill_supported_seq_lens()` — typically
   `[128, 256, 512, 1024, 2048]` for Llama 3B
2. For each length, runs a dummy prefill with mock tokens to JIT-compile ops
3. Captures a Metal trace for the decode forward pass (`_capture_trace_decode`)
4. Captures Metal traces for each prefill length (`_capture_trace_prefill`)

**After warmup, each agentic turn does:**
- `prefill(tokens)` → replays the trace for the padded sequence length bucket
- `decode()` × N → replays the decode trace N times until stop token

**Critical: context is NOT on-device between turns.** The KV cache holds the
decoded state, but after a tool call the full `messages` list is re-tokenized
and re-prefilled (KV cache reset). This is correct and expected — the prefill
trace handles it.

**Batch size must stay fixed at 1** for the agentic use case. Trace is captured
at a specific batch size; changing batch size requires a new trace.

### Whisper (WhisperGenerator)

```python
from models.demos.audio.whisper.tt.whisper_generator import WhisperGenerator, GenerationParams

params = GenerationParams(language="en", task="transcribe", use_trace=True)
whisper = WhisperGenerator(mesh_device, model_args, params)

# No explicit warmup call — trace is captured lazily on the first real audio call:
# _capture_decoder_trace() is called internally after the first non-traced
# decoder iteration populates the cross-attention KV cache.
# All subsequent calls reuse the same trace.
```

Key properties of the Whisper trace (from `whisper_generator.py` docstring):
1. Encoder output and cross-attention KV cache are pre-allocated at fixed addresses
2. First decoder iteration runs un-traced to populate the cross-attention cache
3. From iteration 2 onward, the trace is captured and replayed
4. The trace references the stable pre-allocated buffer addresses — it stays valid
   across ALL audio files without re-capture

**The Whisper trace is NOT invalidated between agentic turns**, as long as
`global_batch_size` stays constant. Each new audio file reuses the same trace.

### Single-Pass Models (OWL-ViT, ViT, BERT, SentenceBERT)

These are encoder-only or single-forward-pass models. They do not use Metal
traces because:
- Each call may have a different input shape (different image size, text length)
- The forward pass completes in a single dispatch — trace overhead not worthwhile
- TTNN ops are already JIT-compiled after the first call (warm cache)

The first call to each model will be slower (~2–5×) due to op compilation.
Subsequent calls are fast. **To avoid slow first-call latency in production,
run a dummy warmup call during startup:**

```python
# Dummy warmup to pre-compile TTNN ops (no trace captured)
owlvit.run("/dev/null/dummy.jpg", "a person")   # will fail gracefully; ops compiled
vit.classify("/dev/null/dummy.jpg")
bert.qa("warmup", "warmup context")
sbert.embed(["warmup"])
```

### SpeechT5

SpeechT5 uses chunked autoregressive decoding in TTNN. There is no Metal trace.
Long text is split into chunks (default 100 chars) and each chunk is synthesized
independently. The TTNN encoder and decoder ops are compiled on first call.

Run a dummy call at startup to pre-compile:
```python
speecht5.synthesize("warmup", output_path="/tmp/warmup.wav")
```

### What Breaks the Trace

| Event | Effect | Recovery |
|-------|--------|----------|
| Batch size change | Decode trace invalid | Re-run `warmup_model_prefill()` |
| Process restart | All traces lost | Re-run startup warmup |
| Device reset | All traces lost | Re-run startup warmup |
| Different audio duration (Whisper) | No effect | Trace handles variable length |
| Different sequence length (LLM) | Uses different prefill trace bucket | No action needed |
| Context grows beyond max seq len | Prefill fails | Truncate or evict KV cache pages |

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

## Coding Agent Extension

A coding agent uses the same agentic loop but with a different LLM and a different
tool set. Most coding tools require **no model** — they are pure Python subprocess
calls. The LLM reasons about what to do; the tools execute it.

### LLM Swap for Coding

Use **Qwen2.5-Coder-7B-Instruct** instead of Llama 3B. It runs on N300 and is
specifically fine-tuned on code generation, debugging, and explanation.

| Mode | LLM | DRAM | Best for |
|------|-----|------|---------|
| General agent | Llama 3.2 3B Instruct | ~6 GB BF16 | Audio, vision, QA |
| Coding agent | Qwen2.5-Coder-7B-Instruct | ~7 GB BF8 | Code gen, debug, refactor |

Both fit on N300 with the tool set. Select at startup via `--mode general|coding`.

### Coding Tool Definitions

#### Pure Subprocess Tools (no model required)

**`execute_python(code, timeout=30)`**
```python
{
    "name": "execute_python",
    "description": "Executes Python code in an isolated subprocess and returns stdout/stderr. "
                   "Use to test code, run computations, or verify fixes.",
    "parameters": {
        "code": {"type": "string", "description": "Python code to execute"},
        "timeout": {"type": "integer", "default": 30, "description": "Max seconds to wait"}
    }
}
# Implementation: subprocess.run(["python", "-c", code], capture_output=True, timeout=timeout)
# Returns: {"stdout": str, "stderr": str, "exit_code": int}
```

**`execute_shell(command, timeout=30)`**
```python
{
    "name": "execute_shell",
    "description": "Runs a shell command and returns output. "
                   "Use for pip install, git commands, file operations, running scripts.",
    "parameters": {
        "command": {"type": "string", "description": "Shell command to run"},
        "timeout": {"type": "integer", "default": 30}
    }
}
# Implementation: subprocess.run(command, shell=True, capture_output=True, timeout=timeout)
# Safety: run in a restricted working directory, block rm -rf / and similar
```

**`read_file(path)`**
```python
{
    "name": "read_file",
    "description": "Reads the contents of a file. Use before editing to see current content.",
    "parameters": {
        "path": {"type": "string", "description": "Absolute or relative file path"}
    }
}
# Returns: {"content": str, "lines": int} or {"error": str} if not found
```

**`write_file(path, content)`**
```python
{
    "name": "write_file",
    "description": "Writes content to a file, creating it if needed. "
                   "Use to save generated code or apply fixes.",
    "parameters": {
        "path": {"type": "string"},
        "content": {"type": "string", "description": "Full file content to write"}
    }
}
```

**`patch_file(path, old_string, new_string)`**
```python
{
    "name": "patch_file",
    "description": "Replaces an exact string in a file. Safer than write_file for small edits.",
    "parameters": {
        "path": {"type": "string"},
        "old_string": {"type": "string", "description": "Exact text to replace (must be unique in file)"},
        "new_string": {"type": "string", "description": "Replacement text"}
    }
}
```

**`list_directory(path, recursive=False)`**
```python
{
    "name": "list_directory",
    "description": "Lists files and directories at a path.",
    "parameters": {
        "path": {"type": "string"},
        "recursive": {"type": "boolean", "default": False}
    }
}
```

**`run_tests(path, flags="")`**
```python
{
    "name": "run_tests",
    "description": "Runs pytest on a file or directory and returns pass/fail/error output. "
                   "Use to verify a fix or check for regressions.",
    "parameters": {
        "path": {"type": "string", "description": "Test file or directory"},
        "flags": {"type": "string", "default": "-x", "description": "Extra pytest flags, e.g. '-k test_name'"}
    }
}
# Implementation: subprocess.run(["pytest", path, flags, "--tb=short"], ...)
```

#### Model-Backed Coding Tools

**`search_codebase(query, directory=".")`**
```python
{
    "name": "search_codebase",
    "description": "Semantically searches a codebase for code related to a query. "
                   "Use to find where a function is defined, how a feature is implemented, "
                   "or which files are relevant to a bug.",
    "parameters": {
        "query": {"type": "string", "description": "What to search for, e.g. 'attention forward pass'"},
        "directory": {"type": "string", "default": ".", "description": "Directory to search"}
    }
}
```
- **Backend:** BGE-Large-EN (N150/N300) — embed all `.py` files at startup into an
  in-memory vector index; at query time embed the query and return top-k cosine matches
- **Returns:** `list[{file, line, snippet, score}]`
- **Index built once at startup** over the project directory; refresh on `write_file`

**`explain_error(error, code)`**
```python
{
    "name": "explain_error",
    "description": "Explains what an error message means and suggests a fix. "
                   "Use when execute_python or run_tests returns an error you cannot interpret.",
    "parameters": {
        "error": {"type": "string", "description": "The full error/traceback"},
        "code": {"type": "string", "description": "The code that produced the error"}
    }
}
```
- **Backend:** The LLM itself (Qwen2.5-Coder) — implemented as a nested LLM call
  with a focused system prompt, not a separate model
- **Returns:** `str` — plain-language explanation + suggested fix

### Coding Agent Loop (ReAct pattern)

The coding agent follows the **ReAct** (Reason + Act) pattern:
Think → Act (tool call) → Observe (result) → repeat until done.

```
User: "Write a function to compute fibonacci and add tests for it"

Turn 1:
  LLM thinks: "I need to write the function first."
  LLM emits:  write_file("fib.py", "def fib(n): ...")
  Result:     "Written successfully"

Turn 2:
  LLM thinks: "Now write the tests."
  LLM emits:  write_file("test_fib.py", "def test_fib(): ...")
  Result:     "Written successfully"

Turn 3:
  LLM thinks: "Run the tests to verify."
  LLM emits:  run_tests("test_fib.py")
  Result:     "FAILED: AssertionError: fib(0) expected 0 got 1"

Turn 4:
  LLM thinks: "fib(0) is wrong. Read the file and fix it."
  LLM emits:  read_file("fib.py")
  Result:     "def fib(n): ..."

Turn 5:
  LLM emits:  patch_file("fib.py", "if n <= 1: return 1", "if n <= 0: return 0\n    if n == 1: return 1")
  Result:     "Patched"

Turn 6:
  LLM emits:  run_tests("test_fib.py")
  Result:     "PASSED 5 tests"

LLM final answer: "Done. fib.py and test_fib.py are written. All 5 tests pass."
```

### Codebase Semantic Search Flow

BGE-Large builds a vector index over source files at startup:

```python
# loader.py — at startup
from models.demos.bge_large_en.runner.performant_runner import BGERunner

bge = BGERunner(device)

# Index all Python files in project (one-time, ~30s for large codebases)
code_index = build_code_index(
    bge,
    directory=".",
    glob="**/*.py",
    chunk_size=50,      # lines per chunk
    overlap=10,         # overlap between chunks
)

# At query time (fast — embedding is TTNN-accelerated):
# query_embedding = bge.embed([query])
# scores = cosine_similarity(query_embedding, code_index.embeddings)
# return top_k chunks with file + line range
```

### DRAM Budget for Coding Agent on N300

```
Qwen2.5-Coder-7B (BF8)    ~7.0 GB
BGE-Large (code search)    ~0.7 GB
BERT Large (QA)            ~0.7 GB
SpeechT5 (optional TTS)    ~0.6 GB
KV cache + traces          ~1.0 GB
────────────────────────────────
Total                     ~10.0 GB / 24 GB   (14 GB headroom)
```

Coding agent does not need Whisper, OWL-ViT, or ViT — audio/vision tools
are optional and can be added if needed.

### Safety Constraints for Code Execution

All subprocess tools run with:
- **Working directory:** a sandboxed temp directory, not the repo root
- **Blocked commands:** `rm -rf /`, `sudo`, `curl | bash`, `dd`, device resets
- **Timeout:** hard kill after N seconds (default 30)
- **No network access:** `subprocess.run(..., env={**os.environ, "no_proxy": "*"})`
- **Output truncation:** cap stdout/stderr at 4096 chars to stay within context window

```python
BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"sudo\s+",
    r"tt-smi\s+-r",      # NEVER reset device from agent
    r">\s*/dev/sd",
]

def safe_execute(command: str) -> dict:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command):
            return {"error": f"Blocked: command matches unsafe pattern '{pattern}'"}
    return subprocess.run(command, shell=True, capture_output=True, timeout=30, cwd=SANDBOX_DIR)
```

### File Structure Addition

```
models/demos/minimax_m2/agentic/
├── tool_wrappers/
│   ├── ...                     (existing tools)
│   ├── code_executor.py        ← execute_python, execute_shell, run_tests
│   ├── file_tools.py           ← read_file, write_file, patch_file, list_directory
│   └── code_search.py          ← BGE-Large vector index + semantic search
├── coding_agent_tools.py       ← CODING_TOOL_SCHEMAS + dispatch
└── coding_demo.py              ← CLI: python coding_demo.py "fix the bug in fib.py"
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
