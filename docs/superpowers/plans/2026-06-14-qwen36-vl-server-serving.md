# Qwen3.6-27B VL (image + video) serving via tt-inference-server + vLLM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve Qwen3.6-27B text + multi-image + video through tt-inference-server using vLLM on Blackhole Galaxy, reusing the already-validated on-device vision/M-RoPE/decode internals.

**Architecture:** Add a `SupportsMultiModal` vLLM generator class that builds the on-device `Qwen36VisionEncoder` alongside the text `TtTransformer`; in `prefill_forward` it runs vision on `pixel_values`/`pixel_values_videos`, splices features into text embeddings, builds M-RoPE, and calls the existing `prefill_forward_text_embeds`. A multimodal `qwen3_5`/`qwen3_6` config keeps `vision_config` so vLLM's native Qwen3VL processor expands image/video tokens. Decode reuses the text path via rope-delta offset.

**Tech Stack:** tt-metal (TTNN, Blackhole Galaxy mesh), vLLM (`MULTIMODAL_REGISTRY`, `SupportsMultiModal`, Qwen3VL processor), tt-inference-server (tt-vllm-plugin, model spec yaml), HF `Qwen3VLProcessor`/`Qwen3VLVideoProcessor`.

**Spec:** `docs/superpowers/specs/2026-06-14-qwen36-vl-server-serving-design.md`

---

## IMPORTANT scope note (read first)

**CRITICAL — TRACE-SAFETY INVARIANT (perf).** A device trace cannot capture host/CPU ops, so any
CPU op inside a traced region prevents trace capture — and trace is the perf mechanism. Therefore:
the per-step **decode loop must be 100% on-device** (NO `to_torch`/host-argmax/per-step cos-sin
rebuild/`get_rope_index`/splice inside it); use **on-device sampling**, advance rope on-device
(`ttnn.plus_one`), set the MM rope offset ONCE via `set_decode_rope_offset()`. All host work
(HF processor, `get_rope_index`, cos/sin build, splice) lives in the **one-time, untraced prefill
setup** — prefill tracing is disabled for qwen3.6 (GDN/DeltaNet CB clash), so prefill host ops are
trace-safe. Reference: `mm_perf_qwen36.py` (prefill_forward_text_embeds + decode_forward(enable_trace=True)
+ on-device sampler + async one-deep `process_decode_output_host` readback outside the trace).
See memory `qwen36-cpu-ops-break-trace`. (Task A1's CPU code is prefill-only → trace-safe.)

**CRITICAL — this model REQUIRES sampling, never greedy/argmax.** Confirmed 2026-06-14: with greedy
argmax the image demo emits pad/`<|im_end|>` → "empty generation". With `QWEN36_SAMPLE=1`
(top_k=20, top_p=0.95, temp=1.0) it produces coherent output ("This is an AI-generated image…"),
PASSED. The top logit is fragile and flips run-to-run. ALL coherence validation (demo + server
parity) MUST use sampling — the same on-device sampling the server uses. See memory
`qwen36-requires-sampling-not-argmax`.

The spec assumed "the demo already does image **and** video." Verification on 2026-06-14 found:
- **Image: wired end-to-end in the demo AND confirmed coherent on HW with sampling** (`mm_demo_qwen36.py` + `QWEN36_SAMPLE=1`). This is the image parity baseline.
- **Video: encoder-level only.** `test_vision_encoder_seqp_pcc.py` validates the vision encoder on a synthetic 2-frame `grid_thw` vs HF (PCC), but **no end-to-end path accepts a video**: `Qwen36MMPreprocessor` only calls `self.processor(text, images=...)` — there is no `videos=` kwarg, no frame sampling, no video M-RoPE/splice exercised.

Therefore **Phase A extends the demo to a video end-to-end path and establishes a parity baseline** before the server work. Without it there is no demo ground-truth to validate server video against. Image serving (Phases B–E) does not depend on Phase A and can proceed in parallel.

### ON-DEVICE-FROM-THE-START directive (user, 2026-06-14)

The forward path must be **100% on-device for BOTH prefill and decode** — because prefill will be
traced later, and any CPU op in the forward would force a rework when that happens (CLAUDE.md: no
shortcuts that need reverting). Do **NOT** port the demo's host `get_rope_index` + host splice +
host cos/sin into the server forward.

**molmo2 precedent** (`models/demos/molmo2/tt/generator_vllm.py` @ `fa9e266d40`, read 2026-06-14):
its `prefill_forward` passes `input_ids` + `pixel_values`/`pixel_values_videos` + a cheap host-built
`token_type_ids` mask into ONE on-device `model.forward_prefill(...)` that does vision encode + splice
on-device. molmo2 uses **1D RoPE** (`arange`) so it has no 3D-position computation. qwen3.6's M-RoPE
is the only extra: its 3D positions (from `grid_thw`) must be produced on-device too.

**Consequence for this plan:**
- Task A1's host `get_rope_index` (committed `09c32e344c0`) is **repurposed as the golden CPU
  reference** for tests only — NOT used in the server forward.
- New on-device units (replace the demo-host approach in the server path):
  (i) on-device M-RoPE 3D positions from `grid_thw` + vision-token mask, PCC-validated vs the host
      `get_rope_index` golden;
  (ii) on-device vision/text splice (`merge_vision_tokens_ttnn`-style), not host splice;
  (iii) one on-device `forward_prefill(input_ids, pixel_values/videos, grid_thw, …)` wiring encode +
      splice + M-RoPE so prefill traces cleanly later.
- Host is reduced to vLLM's pixel decode/patchify (+ at most a position/type index input tensor).

Run env (every HW command). `tt-smi`/`ARCH_NAME` per box; this model is **Blackhole Galaxy, `MESH_DEVICE=BH_GLX`**:
```bash
export TT_METAL_HOME="$(pwd)" PYTHONPATH="$(pwd)"; source python_env/bin/activate
export HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=BH_GLX
export QWEN36_FORCE_SWITCH_DECODE=1 QWEN36_DECODE_L1_RESIDUAL=1 QWEN36_RESIDUAL_BUF_BF16=1 QWEN36_LM_HEAD_PLAIN_DECODE=1
export QWEN36_SEQ_CORES_PER_HEAD=4 QWEN36_FULLATTN_WO_TUNED=1 QWEN36_DELTA_OP_TUNED=1 QWEN36_CCL_NUM_LINKS_DELTA=2
export QWEN36_SAMPLE=1 QWEN36_TOP_K=20 QWEN36_TOP_P=0.95 QWEN36_TEMP=1.0   # MANDATORY: never validate with greedy/argmax
```

## File structure (what each file owns)

**tt-metal (model + demo):**
- `models/demos/qwen3_6_galaxy_v2/tt/qwen36_mm_preprocessor.py` — MOD: accept `videos=`, call HF processor with videos, expose `pixel_values_videos`/`video_grid_thw`.
- `models/demos/qwen3_6_galaxy_v2/tt/qwen36_mm_pipeline.py` — MOD: thread videos through vision encoder + splice.
- `models/demos/qwen3_6_galaxy_v2/tt/qwen36_mm_generator.py` — MOD: `prepare_inputs(..., videos=)`.
- `models/demos/qwen3_6_galaxy_v2/demo/mm_demo_qwen36.py` — MOD: add a video e2e demo case.
- `models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py` — MOD: add `Qwen3_6VLForConditionalGeneration(Generator, SupportsMultiModal)` + processing-info subclass + registration.
- `models/demos/qwen3_6_galaxy_v2/tests/test_server_mm_parity.py` — CREATE: token-expansion + prefill parity (server-path vs demo-path).

**tt-inference-server (plugin + spec):**
- `tt-inference-server/tt-vllm-plugin/tt_vllm_plugin/qwen3_5_config.py` — MOD: multimodal-capable config (keep `vision_config`), new VL arch name.
- `tt-inference-server/tt-vllm-plugin/tt_vllm_plugin/__init__.py` — MOD: register the VL model class.
- `tt-inference-server/workflows/model_specs/dev/llm.yaml` — MOD: VLM spec entry (model_type, modalities, limit-mm-per-prompt).
- `tt-inference-server/QWEN36_SERVING.md` / `models/demos/qwen3_6_galaxy_v2/README.md` — MOD: VL serving docs.

---

## Phase A — Demo video end-to-end + parity baseline

### Task A1: Preprocessor accepts videos

**Files:**
- Modify: `models/demos/qwen3_6_galaxy_v2/tt/qwen36_mm_preprocessor.py`
- Test: `models/demos/qwen3_6_galaxy_v2/tests/test_mm_preprocessor_video.py` (create)

- [ ] **Step 1: Write the failing test** — preprocessor returns video tensors for a synthetic clip.

```python
# tests/test_mm_preprocessor_video.py
import numpy as np, torch
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mm_preprocessor import Qwen36MMPreprocessor

def test_preprocessor_video_produces_video_grid():
    pre = Qwen36MMPreprocessor()  # loads HF Qwen3VLProcessor from checkpoint
    # 4-frame fake video, 224x224 RGB, uint8 [T,H,W,C]
    video = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    prompt = "<|vision_start|><|video_pad|><|vision_end|>Describe the video."
    out = pre(prompt, videos=[video])
    assert out.pixel_values_videos is not None
    assert out.video_grid_thw is not None and out.video_grid_thw.shape[-1] == 3
    # video token (248057) must appear in expanded ids
    assert (out.input_ids == 248057).any()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest models/demos/qwen3_6_galaxy_v2/tests/test_mm_preprocessor_video.py -v`
Expected: FAIL — `Qwen36MMPreprocessor.__call__` has no `videos` kwarg / `Qwen36MMInputs` has no `pixel_values_videos`.

- [ ] **Step 3: Implement** — add `videos` to `__call__` and the dataclass.

In `qwen36_mm_preprocessor.py`, extend the `Qwen36MMInputs` dataclass with:
```python
    pixel_values_videos: torch.Tensor | None = None  # [N_video_patches, 1536]
    video_grid_thw: torch.Tensor | None = None        # [num_videos, 3]
```
and change `__call__`:
```python
    def __call__(self, prompt: str, images=None, videos=None):
        proc_out = self.processor(
            text=prompt, images=images, videos=videos,
            return_tensors="pt", padding=True,
        )
        # ... existing input_ids/attention_mask/pixel_values/image_grid_thw extraction ...
        pixel_values_videos = proc_out.get("pixel_values_videos", None)
        video_grid_thw = proc_out.get("video_grid_thw", None)
        # position_ids_3d must use get_rope_index with BOTH image_grid_thw and
        # video_grid_thw so temporal positions for video frames are correct.
```
Update the `get_rope_index(...)` call in this module to pass `video_grid_thw` and the video token id (248057) in addition to images. (See `qwen36_mrope.get_rope_index` — extend it in Task A2 if it is image-only.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest models/demos/qwen3_6_galaxy_v2/tests/test_mm_preprocessor_video.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add models/demos/qwen3_6_galaxy_v2/tt/qwen36_mm_preprocessor.py models/demos/qwen3_6_galaxy_v2/tests/test_mm_preprocessor_video.py
git commit -m "Qwen3.6 VLM: preprocessor accepts videos= (pixel_values_videos/video_grid_thw)"
```

### Task A2: M-RoPE + splice handle video segments

**Files:**
- Modify: `models/demos/qwen3_6_galaxy_v2/tt/qwen36_mrope.py` (`get_rope_index`)
- Modify: `models/demos/qwen3_6_galaxy_v2/tt/qwen36_mm_pipeline.py` (`prepare_decoder_inputs`, `splice_vision_into_embeddings`)

- [ ] **Step 1: Write the failing test** — CPU-only position-id parity vs HF for a video.

```python
# tests/test_video_rope_index.py
import torch
from transformers import Qwen3VLProcessor
# Build input_ids+grids via HF processor for a 4-frame clip, then compare
# our get_rope_index(position_ids_3d) to HF model.get_rope_index reference.
def test_video_position_ids_match_hf():
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mrope import get_rope_index
    # ... construct input_ids with video tokens + video_grid_thw=[[4, 16, 16]] ...
    pos_ours, _ = get_rope_index(input_ids, image_grid_thw=None,
                                 video_grid_thw=torch.tensor([[4,16,16]]),
                                 image_token_id=248056, video_token_id=248057,
                                 spatial_merge_size=2)
    # compare pos_ours[:, 0, :] to HF reference position_ids for same input
    assert torch.equal(pos_ours, pos_hf)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest models/demos/qwen3_6_galaxy_v2/tests/test_video_rope_index.py -v`
Expected: FAIL — `get_rope_index` signature has no `video_grid_thw`/`video_token_id`.

- [ ] **Step 3: Implement** — extend `get_rope_index` to interleave video segments (temporal T axis advances per frame, H/W per patch) exactly as HF `Qwen3VL` does, and extend the splice to place video features at `video_token_id` positions. The vision encoder already produces correct per-frame features (block-diagonal cu_seqlens); splice mirrors the image path keyed on token id 248057.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest models/demos/qwen3_6_galaxy_v2/tests/test_video_rope_index.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add models/demos/qwen3_6_galaxy_v2/tt/qwen36_mrope.py models/demos/qwen3_6_galaxy_v2/tt/qwen36_mm_pipeline.py models/demos/qwen3_6_galaxy_v2/tests/test_video_rope_index.py
git commit -m "Qwen3.6 VLM: video M-RoPE position ids + video-token splice (CPU parity vs HF)"
```

### Task A3: Video end-to-end demo + decode

**Files:**
- Modify: `models/demos/qwen3_6_galaxy_v2/tt/qwen36_mm_generator.py` (`prepare_inputs(..., videos=)`, `generate(..., videos=)`)
- Modify: `models/demos/qwen3_6_galaxy_v2/demo/mm_demo_qwen36.py` (add `QWEN36_MM_VIDEO` case)

- [ ] **Step 1: Add the video demo case** — read a short video via `decord`/`torchvision`, sample frames at 2 fps (stock Qwen3VL default), run prefill + `_DECODE_STEPS` decode. Gate on `QWEN36_MM_VIDEO` env so the default test stays image.

```python
# in mm_demo_qwen36.py, after the image case:
_VIDEO_PATH = os.environ.get("QWEN36_MM_VIDEO", "")
if _VIDEO_PATH:
    frames = _load_video_frames(_VIDEO_PATH, fps=2)  # list[np.ndarray HWC]
    inputs, fused_unpadded = gen.prepare_inputs(_VIDEO_PROMPT, videos=[frames])
    # ... identical prefill+decode flow as the image path ...
```

- [ ] **Step 2: Run the video demo on HW**

Run:
```bash
QWEN36_MM_VIDEO=/path/to/short.mp4 QWEN36_MM_DECODE_STEPS=24 \
python -m pytest models/demos/qwen3_6_galaxy_v2/demo/mm_demo_qwen36.py -v -s 2>&1 | tee /tmp/qwen36_mm_video_demo.log
```
Expected: coherent text describing the video; no hang. **This is the video parity baseline.** Save the generated text.

- [ ] **Step 3: Commit**

```bash
git add models/demos/qwen3_6_galaxy_v2/tt/qwen36_mm_generator.py models/demos/qwen3_6_galaxy_v2/demo/mm_demo_qwen36.py
git commit -m "Qwen3.6 VLM: video end-to-end demo (frame sampling + prefill + decode)"
```

---

## Phase B — Multimodal server config

### Task B1: VL config keeps vision_config

**Files:**
- Modify: `tt-inference-server/tt-vllm-plugin/tt_vllm_plugin/qwen3_5_config.py`
- Test: `tt-inference-server/tt-vllm-plugin/tests/test_qwen3_6_vl_config.py` (create)

- [ ] **Step 1: Write the failing test** — config keeps vision_config + multimodal arch.

```python
def test_vl_config_keeps_vision_and_is_multimodal():
    from tt_vllm_plugin.qwen3_5_config import Qwen3_6VLConfig
    import json, pathlib
    cfg_dict = json.load(open(pathlib.Path(HF_LOCAL)/"config.json"))
    cfg = Qwen3_6VLConfig(**cfg_dict)
    assert getattr(cfg, "vision_config", None) is not None
    assert cfg.architectures == ["Qwen3_6VLForConditionalGeneration"]
    # text fields still promoted for vLLM scheduling/KV sizing
    assert cfg.num_hidden_layers and cfg.hidden_size
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tt-inference-server/tt-vllm-plugin/tests/test_qwen3_6_vl_config.py -v`
Expected: FAIL — `Qwen3_6VLConfig` does not exist.

- [ ] **Step 3: Implement** — add `Qwen3_6VLConfig(Qwen3_5Config)` that promotes the same text fields **but keeps `vision_config`** (do not `pop` it), keeps `text_config` removal, and sets `architectures = ["Qwen3_6VLForConditionalGeneration"]`. Register it under a distinct `model_type` (e.g. `qwen3_5_vl`) OR reuse `qwen3_5` and select VL vs text by spec — the simplest is a separate config class + model_type so the text-only path is untouched.

- [ ] **Step 4: Run to verify it passes**

Run: same as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd tt-inference-server && git add tt-vllm-plugin/tt_vllm_plugin/qwen3_5_config.py tt-vllm-plugin/tests/test_qwen3_6_vl_config.py
git commit -m "tt-vllm-plugin: Qwen3_6VLConfig keeps vision_config (multimodal)"
```

---

## Phase C — VL generator class

### Task C1: Registration skeleton + ProcessingInfo

**Files:**
- Modify: `models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py`

- [ ] **Step 1: Add the class + registration** (mirrors `models/demos/qwen3_vl/tt/generator_vllm.py:104-167`).

```python
from typing import Mapping, Optional
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor, Qwen3VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

class TT_Qwen36VLProcessingInfo(Qwen3VLProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": _MAX_IMAGES, "video": 1}   # _MAX_IMAGES set in Task D2 spec

@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=TT_Qwen36VLProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen3_6VLForConditionalGeneration(Generator, SupportsMultiModal):
    pass  # methods added in C2-C4
```

- [ ] **Step 2: Smoke-import** the module so the decorator resolves the vLLM Qwen3VL symbols.

Run: `python -c "import models.demos.qwen3_6_galaxy_v2.tt.generator_vllm as g; print(g.Qwen3_6VLForConditionalGeneration)"`
Expected: prints the class; no ImportError on the vLLM Qwen3VL processor symbols.

- [ ] **Step 3: Commit**

```bash
git add models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py
git commit -m "Qwen3.6 VLM serving: register SupportsMultiModal generator skeleton"
```

### Task C2: initialize_vllm_model builds vision encoder

**Files:** Modify: `models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py`

- [ ] **Step 1: Implement** `initialize_vllm_model` to build the text transformer (reuse `initialize_vllm_text_transformer_qwen36`) **plus** the on-device `Qwen36VisionEncoder` + `CCLManager` + the reference `embed_tokens` (mirror `Qwen36MMGenerator.__init__` / `Qwen36MMPipeline.__init__`). Return `cls(model, args, mesh_device, tokenizer=..., vision_encoder=..., ccl_manager=..., embed_weight=...)`.

- [ ] **Step 2: Verify** the class constructs against a real mesh (this is HW; covered by the C3 prefill test). No standalone unit test — proceed.

- [ ] **Step 3: Commit**

```bash
git add models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py
git commit -m "Qwen3.6 VLM serving: initialize_vllm_model builds on-device vision encoder"
```

### Task C3: prefill_forward (image + video)

**Files:** Modify: `models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py`

- [ ] **Step 1: Implement** `prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, enable_trace, **kwargs)` following `qwen3_vl/tt/generator_vllm.py:178-303`, adapted to qwen3.6 helpers:
  1. Fix padding to `pad_token_id`; build per-user `attention_mask`.
  2. Collect `pixel_values` (image) and `pixel_values_videos` (video) + `image_grid_thw`/`video_grid_thw` from `kwargs` (per-user lists, concat).
  3. `vision_features = self.vision_encoder.forward(pixel_values, grid_thw, seq_parallel=True)` for images and/or videos.
  4. Text embed lookup via `self.embed_tokens(input_ids)`; splice vision features at image (248056) / video (248057) positions (reuse the Phase-A splice helper).
  5. Build M-RoPE `(cos, sin)` + `rope_deltas` via `get_rope_index` + `build_mrope_tt_tensors` (both grids).
  6. `logits = self.prefill_forward_text_embeds(inputs_embeds=fused, rot_mats=(cos,sin), page_table=page_table, kv_cache=kv_cache, prompt_lens=decoding_pos)`.
  7. `return logits, rope_deltas`.
  Keep `enable_trace=False` (GDN/DeltaNet prefill-trace clash).

- [ ] **Step 2: Test (HW) — server-path image prefill == demo image prefill** (Task E1 test). Run after E1 is written; expected first-token match vs demo.

- [ ] **Step 3: Commit**

```bash
git add models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py
git commit -m "Qwen3.6 VLM serving: prefill_forward for image+video via prefill_forward_text_embeds"
```

### Task C4: decode_forward rope-delta

**Files:** Modify: `models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py`

- [ ] **Step 1: Implement** `decode_forward(self, *args, **kwargs)` that pops `rope_deltas_all_users`, applies it via `set_decode_rope_offset` (or `update_rope_deltas` if present in this Generator), then calls `super().decode_forward(*args, **kwargs)`. Mirror `qwen3_vl/tt/generator_vllm.py:305-312`.

- [ ] **Step 2: Commit**

```bash
git add models/demos/qwen3_6_galaxy_v2/tt/generator_vllm.py
git commit -m "Qwen3.6 VLM serving: decode_forward applies multimodal rope-delta offset"
```

---

## Phase D — Plugin registration + model spec

### Task D1: Register the VL model in the plugin

**Files:** Modify: `tt-inference-server/tt-vllm-plugin/tt_vllm_plugin/__init__.py`

- [ ] **Step 1: Add** in `register_models()`:
```python
ModelRegistry.register_model(
    "TTQwen3_6VLForConditionalGeneration",
    "models.demos.qwen3_6_galaxy_v2.tt.generator_vllm:Qwen3_6VLForConditionalGeneration",
)
```
and register `Qwen3_6VLConfig` (Task B1) at import time next to `register_qwen3_5_config()`.

- [ ] **Step 2: Verify** plugin imports without error:
Run: `python -c "import tt_vllm_plugin; tt_vllm_plugin.register_models()"` (inside the server venv/container).
Expected: logs "Registered Qwen3.6 VL model"; no exception.

- [ ] **Step 3: Commit**

```bash
cd tt-inference-server && git add tt-vllm-plugin/tt_vllm_plugin/__init__.py
git commit -m "tt-vllm-plugin: register TTQwen3_6VLForConditionalGeneration + VL config"
```

### Task D2: VLM model spec entry

**Files:** Modify: `tt-inference-server/workflows/model_specs/dev/llm.yaml`

- [ ] **Step 1: Add/adjust** the Qwen3.6 dev entry to a VLM spec: `model_type: VLM`, `supported_modalities: [text, image, video]`, `hf_overrides` architecture → `Qwen3_6VLForConditionalGeneration`, and `vllm_args`:
```yaml
      trust_remote_code: true
      limit-mm-per-prompt: '{"image": 8, "video": 1}'   # set _MAX_IMAGES=8 to match (Task C1)
      # default 2 fps matches the demo; only add media_io_kwargs if non-default tuning is needed
```
Keep the existing `QWEN36_*` decode/trace flags and 256k context sizing. Confirm the `_MAX_IMAGES` constant in C1 equals the `image` limit here.

- [ ] **Step 2: Verify** `run.py` resolves the spec (dev mode):
Run (host): `python3 run.py --workflow server --model Qwen3.6-27B --tt-device blackhole_galaxy --dev-mode --no-auth --dry-run` (or inspect spec resolution).
Expected: spec resolves with VLM modalities + mm limits; no schema error.

- [ ] **Step 3: Commit**

```bash
cd tt-inference-server && git add workflows/model_specs/dev/llm.yaml
git commit -m "Qwen3.6-27B: VLM model spec (image+video, limit-mm-per-prompt)"
```

---

## Phase E — Validation

### Task E1: Server-path vs demo-path parity (token expansion + prefill)

**Files:** Create: `models/demos/qwen3_6_galaxy_v2/tests/test_server_mm_parity.py`

- [ ] **Step 1: Token-expansion parity (CPU)** — vLLM's Qwen3VL processor output (`input_ids`, `image_grid_thw`, `video_grid_thw`, pixel tensors) equals `Qwen36MMPreprocessor` output for the same prompt + dog.jpg and for a short video. This catches the primary risk (config/processor compat) without HW.

```python
def test_token_expansion_matches_demo_preprocessor():
    # demo path
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mm_preprocessor import Qwen36MMPreprocessor
    demo = Qwen36MMPreprocessor()(PROMPT, images=[dog])
    # vLLM path
    from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
    # build via TT_Qwen36VLProcessingInfo on the same checkpoint/config
    vllm_out = run_vllm_processor(PROMPT, images=[dog])
    assert torch.equal(demo.input_ids, vllm_out["input_ids"])
    assert torch.equal(demo.image_grid_thw, vllm_out["image_grid_thw"])
```

- [ ] **Step 2: Run** `python -m pytest models/demos/qwen3_6_galaxy_v2/tests/test_server_mm_parity.py -v`. Expected: PASS. If FAIL on config compat → fix `TT_Qwen36VLProcessingInfo` token-id/grid overrides (the spec's primary risk).

- [ ] **Step 3: Prefill parity (HW)** — compare server `Qwen3_6VLForConditionalGeneration.prefill_forward` logits vs demo `prefill_multimodal` for dog.jpg and the Phase-A video via **logit/top-k distribution PCC at the last prefill position** (NOT first-token argmax — this model's top logit is fragile and flips run-to-run; greedy argmax is meaningless here, see memory `qwen36-requires-sampling-not-argmax`). End-to-end coherence is checked with **sampling** (`QWEN36_SAMPLE=1`, top_k=20/top_p=0.95/temp=1.0), the same on-device sampling the server uses. (Marked `@pytest.mark.hardware`.)

- [ ] **Step 4: Commit**

```bash
git add models/demos/qwen3_6_galaxy_v2/tests/test_server_mm_parity.py
git commit -m "Qwen3.6 VLM serving: server-vs-demo MM parity tests (token expansion + prefill)"
```

### Task E2: Live server smoke (image + video + text)

- [ ] **Step 1: Launch** the server (Case A warm start) per `QWEN36_SERVING.md`, with the VL spec.
- [ ] **Step 2: Image request** — `curl /v1/chat/completions` with an image_url; expect coherent description, no hang.
- [ ] **Step 3: Video request** — `curl /v1/chat/completions` with a video_url; expect coherent description.
- [ ] **Step 4: Text-only request** still returns " Paris." (regression guard).
- [ ] **Step 5: Confirm** `tt-smi -r` not needed between requests. Document curl commands + outputs in `QWEN36_SERVING.md`.

### Task E3: Eval suite + docs

- [ ] **Step 1: Run** the VLM/multi-video eval suite (mirror the molmo2 `evals/eval_config.py` image/video entries) and record accuracy.
- [ ] **Step 2: Update** `BRINGUP_LOG.md` (PCC/parity values, server results), `models/demos/qwen3_6_galaxy_v2/README.md` (VL demo + server + eval commands), `QWEN36_SERVING.md` (VL section + known limitations).
- [ ] **Step 3: Commit** docs.

---

## Self-review notes
- **Spec coverage:** Components 1–4 map to Phases B/C/D/E; the spec's "reuse demo internals" assumption is corrected by Phase A (video demo gap). Image path needs no Phase A.
- **Primary risk** (vLLM Qwen3VL processor accepting the `qwen3_5` config) is isolated to Task E1 Step 1 (CPU, runs early) and Task B1 — fail fast there before HW.
- **Open numbers to lock during execution:** `_MAX_IMAGES` (C1) == `limit-mm-per-prompt.image` (D2); the concrete supported multi-image count; whether VL needs its own `model_type` or a flag on `qwen3_5`.
