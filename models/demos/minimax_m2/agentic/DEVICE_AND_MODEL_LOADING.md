# N300 Device Opening and Model Loading

This document describes how the Tenstorrent N300 device is opened and how models are loaded for the multi-modal agentic workflow.

## Table of Contents

1. [Device Architecture](#device-architecture)
2. [Device Opening](#device-opening)
3. [Model Loading](#model-loading)
4. [Memory Budget](#memory-budget)
5. [Cleanup and Shutdown](#cleanup-and-shutdown)
6. [Troubleshooting](#troubleshooting)

---

## Device Architecture

### N300 Hardware

The N300 board contains **2 × Wormhole B0 chips** connected via high-speed fabric interconnect:

```
┌─────────────────────────────────────────────────┐
│                    N300 Board                    │
│  ┌──────────────┐         ┌──────────────┐      │
│  │   Chip 0     │◄───────►│   Chip 1     │      │
│  │  (12 GB DRAM)│ Fabric  │  (12 GB DRAM)│      │
│  │   PCIe Host  │         │   Remote     │      │
│  └──────────────┘         └──────────────┘      │
│                                                  │
│         Total: 24 GB DRAM across mesh           │
└─────────────────────────────────────────────────┘
```

### Mesh Configuration

- **Mesh Shape**: `(1, 2)` = 1 row × 2 columns
- **Chip 0**: Local chip (connected to PCIe host)
- **Chip 1**: Remote chip (accessed via fabric)
- **Fabric**: FABRIC_1D topology for tensor sharding across chips

---

## Device Opening

### Function: `open_n300_device()`

Located in `loader.py`, this function opens the N300 mesh device with parameters optimized for multi-model operation.

```python
def open_n300_device(enable_fabric: bool = True) -> ttnn.MeshDevice:
```

### Step 1: Enable Fabric Configuration

Before opening the device, fabric must be configured for multi-chip tensor operations:

```python
ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_1D,           # 1D topology for 2-chip mesh
    ttnn.FabricReliabilityMode.STRICT_INIT, # Strict initialization mode
    None,                                    # tensix_config placeholder
    ttnn.FabricTensixConfig.DISABLED,       # Tensix fabric disabled
)
```

**Why FABRIC_1D?**
- Required for LLM tensor parallelism across 2 chips
- Enables `mesh_composer` for automatic weight sharding
- Allows CCL (Collective Communication Library) operations

### Step 2: Open Mesh Device

```python
mesh_device = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 2),           # 1 row × 2 cols = 2 chips
    l1_small_size=79_104,           # L1 small buffer allocation
    trace_region_size=100_000_000,  # 100 MB for trace capture
    num_command_queues=2,           # Dual command queues
)
```

### Device Parameters Explained

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `l1_small_size` | 79,104 bytes | Sentence BERT/BGE embeddings require this size. Must be below Whisper's L1 buffer offset (1,018,400). |
| `trace_region_size` | 100 MB | Space for captured execution traces (Whisper decoder trace is largest). |
| `num_command_queues` | 2 | SpeechT5 uses CQ1 for async decode with `ttnn.wait_for_event`. Whisper only uses CQ0. |

### Step 3: Enable Program Cache

```python
mesh_device.enable_program_cache()
```

This caches compiled kernels to avoid recompilation across inference calls.

### Device Opening Sequence Diagram

```
Application
    │
    ▼
┌───────────────────────────┐
│ ttnn.set_fabric_config()  │  ← Configure fabric BEFORE opening device
│   - FABRIC_1D             │
│   - STRICT_INIT           │
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│ ttnn.open_mesh_device()   │  ← Open 2-chip mesh
│   - MeshShape(1, 2)       │
│   - l1_small_size=79104   │
│   - trace_region=100MB    │
│   - num_cqs=2             │
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│ mesh_device.              │  ← Enable kernel caching
│   enable_program_cache()  │
└───────────────────────────┘
    │
    ▼
  Device Ready
```

---

## Model Loading

### Function: `load_all_models()`

Located in `loader.py`, this function loads all specialist models into device DRAM.

```python
def load_all_models(
    mesh_device,
    load_llm: bool = True,
    load_whisper: bool = True,
    load_speecht5: bool = False,
    load_owlvit: bool = True,
    load_bert: bool = True,
    load_sbert: bool = False,
    # ... more flags
) -> ModelBundle:
```

### Model Loading Order

The loading order is designed to handle memory constraints and trace capture conflicts:

```
1. SBERT (if enabled)     ← First: sets up RAG embeddings
   └── RAG System         ← Depends on SBERT
2. LLM (Llama 3.1 8B)     ← Second: largest model, needs most memory
3. Whisper                ← Third: captures decoder trace
4. SpeechT5               ← Fourth: TTS with 2CQ async
5. OWL-ViT                ← Fifth: object detection
6. BERT                   ← Sixth: question answering
7. Other models...        ← Remaining models
```

### Model Initialization Details

#### 1. LLM (Llama 3.1 8B Instruct)

```python
# llm_tool.py
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs

model_args = ModelArgs(
    mesh_device,
    instruct=True,
    max_batch_size=1,
    max_seq_len=4096,
)

state_dict = model_args.load_state_dict()  # Load weights

tt_model = Transformer(
    args=model_args,
    mesh_device=mesh_device,
    dtype=ttnn.bfloat8_b,      # 8-bit weights for memory efficiency
    state_dict=state_dict,
    weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
)

generator = Generator([tt_model], [model_args], mesh_device, tokenizer=tokenizer)

# Warmup: compile prefill kernels
generator.warmup_model_prefill(
    kv_cache=None,              # Uses internal KV cache
    enable_trace=True,
    can_sample_on_device=False,
)
```

**Key Points:**
- Uses **internal KV cache** (`kv_cache=None`) - no external paged attention needed
- Weights in **BF8** format for memory efficiency
- Prefill trace compiled during warmup
- Decode trace captured lazily on first decode

#### 2. Whisper (STT)

```python
# whisper_tool.py
from models.demos.audio.whisper.demo.demo import (
    create_functional_whisper_for_conditional_generation_inference_pipeline
)

pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
    mesh_device=mesh_device,
    model_repo="distil-whisper/distil-large-v3",
    generation_params=GenerationParams(),
    language="en",
    task="transcribe",
    use_trace=True,
    batch_size_per_device=1,
)
```

**Key Points:**
- Uses **distil-whisper/distil-large-v3** (English-only, distilled)
- Decoder trace captured **lazily** on first transcription
- Trace persists across agentic turns for reuse

#### 3. Sentence BERT (RAG Embeddings)

```python
# sbert_tool.py
from models.demos.sentence_bert.runner.performant_runner import SentenceBERTPerformantRunner

runner = SentenceBERTPerformantRunner(
    device=mesh_device,
    device_batch_size=8,
    sequence_length=384,
    model_location_generator=model_location_generator,
    input_ids=dummy_input_ids,
    extended_mask=dummy_extended_mask,
    attention_mask=dummy_attention_mask,
    token_type_ids=dummy_token_type_ids,
    position_ids=dummy_position_ids,
)

if use_trace:
    # Fast traced mode (standalone)
    runner._capture_sentencebert_trace_2cqs()
else:
    # Non-traced mode (LLM compatible)
    # Setup L1 sharded inputs manually
    (ttnn_input_ids, input_mem_config, ...) = runner.runner_infra.setup_l1_sharded_input()
    runner.runner_infra.ttnn_input_ids = ttnn.to_memory_config(
        ttnn_input_ids.to(mesh_device), input_mem_config
    )
    # ... setup other tensors
    runner.runner_infra.run()  # Warmup JIT compilation
```

**Key Points:**
- **Two modes**: Traced (fast, standalone) vs Non-traced (slower, LLM-compatible)
- Auto-detection: `use_trace = not load_llm`
- 768-dimensional embeddings from BERT base
- ~45ms per batch in non-traced mode

### Model Bundle Structure

All loaded models are stored in a `ModelBundle` dataclass:

```python
@dataclass
class ModelBundle:
    llm: LLMTool = None           # Orchestrator
    whisper: WhisperTool = None   # Speech-to-text
    speecht5: SpeechT5Tool = None # Text-to-speech
    owlvit: OWLViTTool = None     # Object detection
    bert: BERTTool = None         # Question answering
    sbert: SBERTTool = None       # Embeddings (TTNN)
    rag: RAGTool = None           # Knowledge base search
    # ... more models
```

---

## Memory Budget

### DRAM Allocation (BF8/BF16 mix across 2 chips)

| Model | Size | Notes |
|-------|------|-------|
| Llama 3.1 8B | ~8.0 GB | Sharded across 2 chips, BF8 weights |
| Whisper distil-v3 | ~1.5 GB | Encoder + decoder |
| SpeechT5 | ~0.3 GB | Smaller TTS model |
| OWL-ViT | ~0.3 GB | Vision transformer |
| BERT Large | ~0.7 GB | QA model |
| Sentence BERT | ~0.3 GB | Embedding model |
| YUNet | ~0.01 GB | Tiny face detector |
| T5-small | ~0.06 GB | Translation |
| KV cache + traces | ~1.0 GB | Runtime allocations |
| **Total** | **~12.2 GB** | of 24 GB available |

### L1 Memory Layout

```
L1 Address Space (per chip)
────────────────────────────────────────────
0x00000000 ┬─────────────────────────────┬
           │ Small buffer region          │
           │ (l1_small_size = 79,104)     │
0x00013500 ├─────────────────────────────┤
           │ General L1 allocations       │
           │                              │
0x000F8600 ├─────────────────────────────┤ ~1,018,400 (Whisper offset)
           │ Whisper L1 buffers           │
           │                              │
0x00186A00 ├─────────────────────────────┤ ~1,600,000
           │ Trace region                 │
           │ (100 MB)                     │
           │                              │
────────────────────────────────────────────
```

---

## Cleanup and Shutdown

### Function: `cleanup_models()`

**Critical**: Must be called BEFORE `ttnn.close_mesh_device()` to prevent segfaults.

```python
def cleanup_models(bundle: ModelBundle) -> None:
    """
    Release all model traces before closing the device.

    Python's GC runs __del__ after device close, causing trace
    release to fail. Explicit cleanup prevents this.
    """
    # Release each model's traces
    if bundle.llm is not None:
        bundle.llm.close()
    if bundle.whisper is not None:
        bundle.whisper.close()
    # ... other models
```

### Proper Shutdown Sequence

```python
# CORRECT shutdown sequence
cleanup_models(bundle)           # 1. Release all traces
ttnn.close_mesh_device(mesh_device)  # 2. Close device

# WRONG - causes segfault
ttnn.close_mesh_device(mesh_device)  # Device closed first
# Python GC later calls __del__ → trace release fails → SEGFAULT
```

---

## Troubleshooting

### Device Initialization Failures

**Error**: `Device 0 init: failed to initialize FW! Try resetting the board.`

**Cause**: Previous process crashed without proper cleanup, leaving device in bad state.

**Solution**: Reset the device:
```bash
tt-smi -r
```

### Trace Capture Conflicts

**Error**: `RuntimeError: Buffer must be allocated on device!`

**Cause**: SBERT trace conflicts with LLM memory.

**Solution**: Use non-traced mode when loading with LLM:
```python
bundle.sbert = SBERTTool(mesh_device=mesh_device, use_trace=False)
```

### L1 Memory Exhaustion

**Error**: `TT_FATAL: Not enough L1 memory`

**Cause**: `l1_small_size` too large, overlapping with model buffers.

**Solution**: Use `l1_small_size=79104` (validated to work with all models).

### Device Lock Contention

**Error**: `Waiting for lock 'CHIP_IN_USE_0_PCIe'`

**Cause**: Another process holding the device lock.

**Solution**:
```bash
# Kill stuck processes
pkill -9 -f "python.*minimax"
# Clear lock files
sudo rm -f /tmp/CHIP_IN_USE_*
```

---

## Usage Example

```python
import ttnn
from models.demos.minimax_m2.agentic.loader import (
    open_n300_device,
    load_all_models,
    cleanup_models,
)

# 1. Open device
mesh_device = open_n300_device(enable_fabric=True)

try:
    # 2. Load models
    models = load_all_models(
        mesh_device,
        load_llm=True,
        load_whisper=True,
        load_sbert=True,  # Auto-detects non-traced mode
        load_owlvit=True,
        load_bert=True,
    )

    # 3. Use models
    text = models.whisper.transcribe("audio.wav")
    embeddings = models.sbert.embed(["Hello world"])
    response = models.llm.generate_response([{"role": "user", "content": "Hi"}])

finally:
    # 4. Cleanup (CRITICAL - must be before device close)
    cleanup_models(models)
    ttnn.close_mesh_device(mesh_device)
```

---

## Quick Reference

| Operation | Function | Notes |
|-----------|----------|-------|
| Open device | `open_n300_device()` | Call once at startup |
| Load models | `load_all_models()` | Flags control which models to load |
| Cleanup | `cleanup_models()` | MUST call before device close |
| Close device | `ttnn.close_mesh_device()` | Call after cleanup |

| Device Parameter | Value | Why |
|-----------------|-------|-----|
| Mesh shape | (1, 2) | 2 Wormhole chips |
| l1_small_size | 79,104 | SBERT compatible, below Whisper |
| trace_region_size | 100 MB | Whisper decoder trace |
| num_command_queues | 2 | SpeechT5 async decode |
| enable_fabric | True | LLM tensor parallelism |
