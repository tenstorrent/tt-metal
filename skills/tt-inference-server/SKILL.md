---
name: tt-inference-server
description: Expose a completed TTNN model through the tt-inference-server OpenAI-compatible HTTP API. Use after TTNN blocks pass PCC tests and end-to-end generation is verified. Covers generator_vllm.py, plugin registration, model_spec.py, VLM multimodal processor, video backend, decode trace warnings, and accuracy debugging.
---

# SKILL: tt-inference-server Integration

## Purpose
Expose a completed TTNN model through the tt-inference-server OpenAI-compatible HTTP API.
This skill starts after the TTNN blocks are working (PCC > 0.99, end-to-end verified).

## Prerequisites

1. All TTNN sub-block PCC tests pass (> 0.99)
2. End-to-end generation verified against HF reference (correct output tokens)
3. Demo script (`demo/demo.py`) produces correct answers locally
4. Know the HF architecture class name (e.g. `LlamaForCausalLM`) — used for registration

---

## Integration Overview

```
python run.py --model <MODEL_ID> --workflow server --local-server --tt-device <DEVICE>
    │
    └─ tt-inference-server (run.py)
         ├─ workflows/model_spec.py        ← DeviceModelSpec: vllm_args, max_context
         └─ vllm (API server)
              ├─ tt-vllm-plugin/__init__.py ← ModelRegistry.register_model("TT<Arch>", ...)
              └─ generator_vllm.py          ← TT model vLLM plugin class
                   ├─ initialize_vllm_model()
                   ├─ allocate_kv_cache()
                   ├─ prefill_forward()
                   └─ decode_forward()
```

---

## What Changes by Model Type

### Text-only LLM (Llama, Qwen, DeepSeek, Mistral)

**Minimal new code — reuse existing tt_transformers infrastructure.**

```python
# generator_vllm.py — thin wrapper
from models.tt_transformers.tt.generator_vllm import (
    allocate_vllm_kv_cache, initialize_vllm_text_transformer,
)
from models.tt_transformers.tt.generator import Generator

class MyModelForCausalLM(WarmupForwardMixin, Generator):
    model_capabilities = {"supports_prefix_caching": False, "supports_async_decode": False}

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size,
                               max_seq_len, tt_data_parallel=1, optimizations=None):
        return initialize_vllm_text_transformer(
            hf_config, tt_data_parallel, mesh_device, max_batch_size, max_seq_len
        )

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return allocate_vllm_kv_cache(kv_cache_shape, dtype, num_layers,
                                       self.tt_model, self.model_args.tt_cache_path)
    # prefill_forward / decode_forward inherited from Generator
```

**Files to create/modify:**
| File | Action |
|------|--------|
| `models/demos/{model}/tt/generator_vllm.py` | CREATE (small wrapper) |
| `tt-vllm-plugin/__init__.py` | MODIFY: add register_model |
| `workflows/model_spec.py` | MODIFY: add ModelSpecTemplate |

---

### Image VLM (LLaVA, Mistral-Vision, Gemma3)

**Needs multimodal processor + pixel_values handling.**

```python
# generator_vllm.py
class ModelForConditionalGeneration(WarmupForwardMixin, SupportsMultiModal):
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens,
                        enable_trace=False, **kwargs):
        def unwrap(v):
            if isinstance(v, list): v = v[0] if v else None
            if isinstance(v, list): v = v[0] if v else None
            return v

        pixel_values = unwrap(kwargs.get("pixel_values"))
        seq_len = int(prompt_lens[0].item())
        input_ids = tokens[:1, :seq_len]

        # Reconstruct token_type_ids for image-bidirectional attention
        if pixel_values is not None:
            IMAGE_PATCH_ID = <model_specific>
            token_type_ids = (input_ids == IMAGE_PATCH_ID).long()
        else:
            token_type_ids = None

        self.model.reset_kv_cache(user_id=0)
        logits = self.model.forward_prefill(
            input_ids=input_ids, pixel_values=pixel_values.float().unsqueeze(0),
            token_type_ids=token_type_ids, user_id=0,
        )
        return logits, None

    def decode_forward(self, tokens, start_pos, **kwargs):
        # No trace — S varies per request
        logits = self.model.forward_decode_step(int(tokens[0,0].item()),
                                                 int(start_pos[0].item()))
        return logits.squeeze(0).unsqueeze(0)
```

**Files to create/modify:**
| File | Action | Notes |
|------|--------|-------|
| `models/demos/{model}/tt/generator_vllm.py` | CREATE | SupportsMultiModal class |
| `vllm/vllm/model_executor/models/{model}.py` | CREATE | Processor + stub class |
| `tt-vllm-plugin/__init__.py` | MODIFY | Register |
| `workflows/model_spec.py` | MODIFY | VLM spec, `"image": 1` limit |

---

### Video VLM (Molmo2, Qwen2.5-VL, Qwen3-VL)

**Needs video backend + frame marker handling + video_input_ids pattern.**

Key differences from image VLM:

#### 1. token_type_ids must include frame markers

```python
# WRONG — patches only:
token_type_ids = (input_ids == IMAGE_PATCH_ID).long()

# CORRECT — patches AND frame boundary markers:
IMAGE_PATCH_ID = <model_specific>   # e.g. 151938
IM_START       = <model_specific>   # e.g. 151936  (<im_start>)
IM_END         = <model_specific>   # e.g. 151937  (<im_end>)
token_type_ids = (
    (input_ids == IMAGE_PATCH_ID) |
    (input_ids == IM_START) |
    (input_ids == IM_END)
).long()
# Omitting frame markers: ~30 pp accuracy drop (causal-only attention on frame boundaries)
```

Find model-specific IDs: `proc(text="<|video|>", videos=[frames])["token_type_ids"]`

#### 2. video_input_ids: inject frame markers into PromptReplacement

In `vllm/vllm/model_executor/models/{model}.py`, `_call_hf_processor`:

```python
# After calling HF processor with full text+video:
combined_ids = result.get("input_ids")
if combined_ids is not None:
    ids_1d = combined_ids.squeeze(0)
    result["video_input_ids"] = ids_1d.long()           # full token seq WITH frame markers
    result["video_num_input_tokens"] = torch.tensor([ids_1d.shape[0]])
```

In `_get_prompt_updates.get_replacement()`, prefer `video_input_ids` over
`[IMAGE_PATCH_ID] * N_pooled`:

```python
vid_ids = item_data.get("video_input_ids")
if vid_ids is not None:
    t = vid_ids.data if hasattr(vid_ids, "data") else vid_ids
    if isinstance(t, torch.Tensor) and t.numel() > 0:
        return PromptUpdateDetails.select_token_id(t.long().tolist(),
                                                    embed_token_id=IMAGE_PATCH_ID)
# fallback to N_pooled × IMAGE_PATCH_ID if video_input_ids not available
```

This gives S with frame markers (e.g. 2701) vs patches-only (e.g. 2481), matching demo.
Impact: 68% → 98% accuracy.

#### 3. Model-specific video backend

```python
# In vllm/vllm/multimodal/video.py:
@VIDEO_LOADER_REGISTRY.register("{model}")
class {Model}VideoBackend(VideoBackend):
    @staticmethod
    def load_bytes(data: bytes, num_frames: int = 384,
                   frame_sample_mode: str = "uniform_last_frame",
                   max_fps: float = 2.0, sampling_fps: float = 2.0, **kwargs):
        # Replicate HF VideoProcessor sampling exactly
        # Must produce same frames + frames_indices as proc(videos=str(path))
        ...
        return frames, metadata_dict  # metadata includes fps, duration, frames_indices
```

In `model_spec.py` vllm_args:
```python
"media_io_kwargs": json.dumps({"video": {
    "video_backend": "{model}",
    "frame_sample_mode": "uniform_last_frame",
    "max_fps": 2,
    "num_frames": 384,
}}),
```

#### 4. KV cache: bfloat16 (not bfloat8_b)

bfloat8_b causes logit flips for S > 2500. See TTNN skill.

**Files to create/modify:**
| File | Action | Notes |
|------|--------|-------|
| `models/demos/{model}/tt/generator_vllm.py` | CREATE | Video kwargs, frame marker tti |
| `vllm/vllm/model_executor/models/{model}.py` | CREATE | Processor + stub + video_input_ids |
| `vllm/vllm/multimodal/video.py` | MODIFY | Register VideoBackend subclass |
| `tt-vllm-plugin/__init__.py` | MODIFY | Register |
| `workflows/model_spec.py` | MODIFY | media_io_kwargs with video backend |

---

### Embedding Model (BGE, Qwen3-Embedding)

**encode() interface, no prefill/decode, no KV cache.**

```python
class EmbeddingModel(BiEncoderModel):
    def encode(self, input_ids, positions, **kwargs):
        return self.model.forward(input_ids)
```

**Files to create/modify:**
| File | Action |
|------|--------|
| `models/demos/{model}/demo/generator_vllm.py` | CREATE |
| `tt-vllm-plugin/__init__.py` | MODIFY |
| `workflows/model_spec.py` | MODIFY (ModelType.EMBEDDING) |

---

## Steps

### Step 1: Create generator_vllm.py

Location: `models/demos/{model}/tt/generator_vllm.py`

```python
from pathlib import Path
from typing import Mapping, Optional
import torch
from loguru import logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
import ttnn
from models.common.warmup import WarmupForwardMixin
from models.demos.{model}.tt.model import Tt{Model}Model
from models.demos.{model}.tt.model_config import {Model}Config
from models.tt_transformers.tt.ccl import TT_CCL

WEIGHT_CACHE_PATH = Path("/tmp/{model}_weight_cache")

def allocate_{model}_kv_cache(kv_cache_shape, dtype, num_layers, model, cfg):
    cache_shape = (cfg.max_batch_size, cfg.n_local_kv_heads, cfg.max_seq_len, cfg.head_dim)
    for layer_idx in range(num_layers):
        cache_kv = torch.zeros(cache_shape, dtype=torch.bfloat16)
        model.layers[layer_idx].attention.layer_past = [
            ttnn.as_tensor(cache_kv, device=model.mesh_device, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
                           cache_file_name=None)
            for _ in range(2)
        ]
    return [layer.attention.layer_past for layer in model.layers]

@MULTIMODAL_REGISTRY.register_processor(...)  # VLMs only
class {Model}ForConditionalGeneration(WarmupForwardMixin, SupportsMultiModal):
    model_capabilities = {"supports_prefix_caching": False, "supports_async_decode": False}

    def __init__(self, model, cfg, mesh_device, processor):
        self.model = model; self.cfg = cfg
        self.mesh_device = mesh_device; self.processor = processor

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len,
                               tt_data_parallel=1, optimizations=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        hf_id = getattr(hf_config, "_name_or_path", "") or hf_config.name_or_path
        hf = AutoModelForImageTextToText.from_pretrained(
            hf_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu")
        sd = hf.state_dict(); del hf
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        cfg = {Model}Config(mesh_device=mesh_device)
        cfg.max_batch_size = max_batch_size; cfg.max_seq_len = max_seq_len
        WEIGHT_CACHE_PATH.mkdir(parents=True, exist_ok=True)
        model = Tt{Model}Model(mesh_device=mesh_device, tt_ccl=TT_CCL(mesh_device),
                                state_dict=sd, weight_cache_path=WEIGHT_CACHE_PATH,
                                dtype=ttnn.bfloat16, configuration=cfg)
        del sd
        return cls(model=model, cfg=cfg, mesh_device=mesh_device, processor=processor)

    @property
    def cache_path(self): return WEIGHT_CACHE_PATH

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_{model}_kv_cache(*args, **kwargs, model=self.model, cfg=self.cfg)

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens,
                        enable_trace=False, **kwargs):
        def unwrap(v):
            if isinstance(v, list): v = v[0] if v else None
            if isinstance(v, list): v = v[0] if v else None
            return v
        # Extract mm inputs
        pixel_values_videos = unwrap(kwargs.get("pixel_values_videos"))
        video_token_pooling  = unwrap(kwargs.get("video_token_pooling"))
        pv, pool_idx = None, None
        if pixel_values_videos is not None:
            pv = pixel_values_videos.float().unsqueeze(0)
            if video_token_pooling is not None:
                pool_idx = video_token_pooling.unsqueeze(0)
        seq_len = int(prompt_lens[0].item() if hasattr(prompt_lens[0], "item") else prompt_lens[0])
        input_ids = tokens[:1, :seq_len]
        # Reconstruct token_type_ids — include ALL image tokens (patches + frame markers)
        if pv is not None:
            token_type_ids = (
                (input_ids == IMAGE_PATCH_ID) |
                (input_ids == IM_START) |
                (input_ids == IM_END)
            ).long()
        else:
            token_type_ids = None
        self.model.reset_kv_cache(user_id=0)
        logits = self.model.forward_prefill(
            input_ids=input_ids, pixel_values=pv, pooled_patches_idx=pool_idx,
            token_type_ids=token_type_ids, user_id=0,
        )
        return logits, None

    def decode_forward(self, tokens, start_pos, page_table, kv_cache,
                       enable_trace=False, read_from_device=True,
                       sampling_params=None, **kwargs):
        # No trace — S varies per request; trace captured at S₁ gives wrong output at S₂≠S₁
        logits = self.model.forward_decode_step(
            int(tokens[0, 0].item()),
            int(start_pos[0].item() if hasattr(start_pos[0], "item") else start_pos[0])
        )
        return logits.squeeze(0).unsqueeze(0)
```

### Step 2: Register in __init__.py

```python
# In register_models():
try:
    ModelRegistry.register_model(
        "TT{HFArchName}",   # e.g. "TTLlamaForCausalLM", "TTMolmo2ForConditionalGeneration"
        "models.demos.{model}.tt.generator_vllm:{Model}ForCausalLM",
    )
except Exception as e:
    logger.warning(f"Failed to register TT{HFArchName}: {e}")
```

Naming convention: `"TT" + HF architecture class name` (from `model.config.architectures[0]`).

### Step 3: Add to model_spec.py

```python
ModelSpecTemplate(
    weights=["org/ModelName-7B"],
    impl=tt_transformers_impl,
    inference_engine=InferenceEngine.VLLM.value,
    model_type=ModelType.VLM,           # or LLM / EMBEDDING
    version="1.0.0",
    tt_metal_commit="<commit>",
    device_model_specs=[
        DeviceModelSpec(
            device=DeviceTypes.T3K,
            max_concurrency=1,
            max_context=32768,
            default_impl=True,
            vllm_args={
                "trust_remote_code": True,
                "limit-mm-per-prompt": json.dumps({"video": 1}),  # VLM only
                "media_io_kwargs": json.dumps({"video": {           # video VLM only
                    "video_backend": "{model}",
                    "frame_sample_mode": "uniform_last_frame",
                    "max_fps": 2, "num_frames": 384,
                }}),
            },
            has_builtin_warmup=True,
        ),
    ],
    status=ModelStatusTypes.EXPERIMENTAL,
    supported_modalities=["text", "video"],  # or ["text"] / ["text", "image", "video"]
),
```

### Step 4: Start Server

```bash
cd tt-inference-server
export TT_METAL_HOME=/path/to/tt-metal
python run.py --model ModelName --workflow server --local-server --tt-device t3k \
    --skip-system-sw-validation
```

### Step 5: Validate Accuracy

```bash
# Quick smoke test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"org/Model","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'

# Full test suite (if available)
python verification/run_video_tests.py \
    --server-url http://localhost:8000 \
    --output results.jsonl
```

Compare against reference:
```python
def first_char(s): return (s or "").strip()[:1]
match = sum(1 for idx in range(N) if first_char(server[idx]) == first_char(ref[idx]))
print(f"Accuracy: {match}/{N} = {100*match/N:.1f}%")
```

---

## Debugging Accuracy (see also debug/SKILL.md)

If server accuracy < demo accuracy:

1. **Check S values** — `logger.info(f"Prefill: S={seq_len}")` in prefill_forward
2. **Frame markers missing** — add `video_input_ids` to `_call_hf_processor`
3. **token_type_ids incomplete** — add IM_START/IM_END to reconstruction
4. **Wrong video backend** — register model-specific VideoBackend
5. **Decode trace wrong S** — disable trace, use `forward_decode_step()`
6. **bfloat8_b KV cache** — switch to bfloat16

---

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImportError: platform` | `platform.py` conflicts with stdlib | Rename to `tt_platform.py` everywhere |
| `sample_tokens() NotImplementedError` | vLLM V1 two-phase model | Cache `execute_model` output; return in `sample_tokens()` |
| `spec_token_ids unexpected kwarg` | New `ModelRunnerOutput` field | Check `dataclasses.fields()` before adding |
| `mm_features not found` | vLLM changed `mm_inputs` → `mm_features` | Detect field names via `dataclasses.fields()` |
| `validate_request missing args` | New vLLM signature | Use `*args, **kwargs` |
| `Expected 1 placeholder, found 0` | mm_processor_cache_gb=0 with wrong target | Keep mm_processor_cache_gb default (4); use `video_input_ids` pattern instead |
| Server accuracy << demo (20–30 pp) | Frame markers missing from input_ids | `video_input_ids` in `_call_hf_processor`; add IM_START/IM_END to tti |
| Decode trace wrong results | S varies; trace baked at S_first | Disable trace; use `forward_decode_step()` |
| First test correct, rest wrong | Decode trace reused at wrong S | Same fix as above |
| `AscendScheduler` errors | Scheduler compat changes | Check `max_model_len`, `block_size` param |
| `num_links=2` hang | T3K doesn't support 2 links | Use `num_links=1` only |

---

## Verification Checklist

- [ ] Server starts without errors
- [ ] First request returns sensible output
- [ ] Full test suite run via HTTP client
- [ ] Server accuracy ≥ demo accuracy (within 2–3 pp)
- [ ] `tt-smi -r` not needed between requests
- [ ] Latency vs GPU reference measured and documented
- [ ] BRINGUP_LOG.md updated with server results
