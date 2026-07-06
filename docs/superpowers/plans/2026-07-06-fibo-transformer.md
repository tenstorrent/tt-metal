# FIBO Transformer (denoiser) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tt_dit implementation of the FIBO MMDiT denoiser (`BriaFiboTransformer2DModel`) that reproduces the HF reference numerically (PCC ≥ 0.99) on Blackhole, built by adapting tt_dit's Flux transformer and adding FIBO's per-block "concat-halves" text injection.

**Architecture:** FIBO ≈ Flux MMDiT with 8 dual + 38 single blocks (inner_dim 3072), in_channels 48, timestep-only modulation, axial RoPE θ=10000, and a per-block text injection that replaces the second half of the running context with a per-block projection of `text_encoder_layers[block_id]`. tt_dit's Flux dual block and **single block (which already keeps the spatial/prompt streams separate)** are reused as-is; the injection is applied in the forward loop before each block.

**Tech Stack:** Python, PyTorch + `diffusers` (reference), `ttnn`, tt_dit (`models/tt_dit`), Tenstorrent Blackhole mesh.

**Spec:** `docs/superpowers/specs/2026-07-06-fibo-transformer-design.md`
**Branch:** `fibo-transformer` (stacked on `fibo-smollm3-encoder`).

## Global Constraints

- **SPDX header** on every new `.py`: `# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.` / blank / `# SPDX-License-Identifier: Apache-2.0` / blank.
- **Location:** `models/tt_dit/models/transformers/transformer_bria_fibo.py`; tests `models/tt_dit/tests/models/bria_fibo/`.
- **FIBO transformer config (verified from `briaai/FIBO/transformer/config.json`):** `num_layers=8` (dual), `num_single_layers=38`, total 46; `num_attention_heads=24`, `attention_head_dim=128` → `inner_dim=3072`; `in_channels=48`; `joint_attention_dim=4096`; `text_encoder_dim=2048`; `pooled_projection_dim=None`; `guidance_embeds=False`; `patch_size=1`; `axes_dims_rope=[16,56,56]`; `rope_theta=10000`; `time_theta=10000`. out_channels = in_channels = 48.
- **Per-block injection:** `caption_projection` = ModuleList of 46 `BriaFiboTextProjection(in=2048, out=inner_dim//2=1536)`. `context_embedder = Linear(4096 → 3072)`. Before each block (single `block_id` counter across both loops, 0..45): `enc = cat([enc[..., :1536], caption_projection[block_id](text_encoder_layers[block_id])], dim=-1)`.
- **Reference:** `from diffusers import BriaFiboTransformer2DModel` (or `from diffusers.models.transformers.transformer_bria_fibo import BriaFiboTransformer2DModel`); `.from_pretrained("briaai/FIBO", subfolder="transformer", torch_dtype=torch.bfloat16)`. If the class name differs at import, fall back to `diffusers.AutoModel`/the pipeline's import; confirm during Task 1.
- **Weights:** gated `briaai/FIBO/transformer/*` (multi-GB). Pre-download once with a valid token, then run OFFLINE. Test invocation:
  `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest <path>::<test> -v`  (interpreter: `python_env/bin/python`; there is no `python` on PATH).
- **Precision/PCC:** bf16; PCC ≥ 0.99 vs reference (the Flux transformer test uses 0.997 full-model / 0.9996 single-block — aim similarly, report the measured floor if depth drift lowers it).
- **Reuse base:** `models/tt_dit/models/transformers/transformer_flux1.py` (`Flux1Transformer`, `Flux1SingleTransformerBlock`, `Flux1Checkpoint`), `blocks/transformer_block.py` (`TransformerBlock`), `blocks/attention.py`, `layers/embeddings.py` (`Timesteps`, `TimestepEmbedding`, `CombinedTimestepGuidanceTextProjEmbeddings`), `layers/linear.py`, `utils/{cache,padding,substate,tensor}.py`. Test harness pattern: `tests/models/flux1/test_transformer_flux1.py`.
- **Parallelism:** `DiTParallelConfig(cfg, sp, tp)`. Validate tp=1 first (injection = plain `ttnn.concat`), then the 2×2 mesh (`sp=2,tp=2`). At tp=2 the injection's 1536 half-boundary equals the per-device feature-shard boundary, so the concat-halves is a per-device shard replace (see Task 5).

---

## File Structure
- `models/tt_dit/models/transformers/transformer_bria_fibo.py` — `BriaFiboTextProjection`, `BriaFiboTimestepEmbed`, `BriaFiboTransformer`, `BriaFiboCheckpoint`, injection helper. (Reuses Flux dual/single blocks + attention.)
- `models/tt_dit/tests/models/bria_fibo/{__init__.py, test_transformer.py}` — unit + full-model PCC tests.
- `models/tt_dit/models/BriaFibo.md` — extend the existing doc (sub-project 2 section).

---

### Task 1: Weights pre-download + reference/config smoke + package skeleton

**Files:**
- Create: `models/tt_dit/models/transformers/transformer_bria_fibo.py` (header + imports only for now)
- Create: `models/tt_dit/tests/models/bria_fibo/__init__.py`, `models/tt_dit/tests/models/bria_fibo/test_transformer.py`

**Interfaces:**
- Produces: nothing importable yet; establishes the reference-loading + offline-weights path used by all later tasks.

- [ ] **Step 1: Pre-download the transformer weights** (one-time, with a valid token read from ~/.bashrc; multi-GB):
```bash
cd /localdev/mstojkovic/tt-metal
export HF_TOKEN=$(grep -E '^[[:space:]]*export[[:space:]]+HF_TOKEN=' ~/.bashrc | tail -1 | sed -E 's/.*HF_TOKEN=//; s/^["'"'"']//; s/["'"'"']$//')
python_env/bin/python -c "from huggingface_hub import snapshot_download; import os; print(snapshot_download('briaai/FIBO', allow_patterns=['transformer/*'], token=os.environ['HF_TOKEN']))"
```

- [ ] **Step 2: Write a reference/config smoke test** (host-only) in `test_transformer.py`:
```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")


def _load_ref_transformer(dtype=torch.bfloat16):
    try:
        from diffusers import BriaFiboTransformer2DModel
    except Exception:
        from diffusers.models.transformers.transformer_bria_fibo import BriaFiboTransformer2DModel
    try:
        return BriaFiboTransformer2DModel.from_pretrained(
            FIBO_PATH, subfolder="transformer", torch_dtype=dtype
        ).eval()
    except Exception as e:
        pytest.skip(f"FIBO transformer unavailable: {e}")


def test_fibo_transformer_reference_config():
    m = _load_ref_transformer()
    c = m.config
    assert c.num_layers == 8 and c.num_single_layers == 38
    assert c.num_attention_heads == 24 and c.attention_head_dim == 128
    assert c.in_channels == 48 and c.joint_attention_dim == 4096
    assert c.axes_dims_rope == [16, 56, 56]
    assert len(m.caption_projection) == c.num_layers + c.num_single_layers  # 46
```

- [ ] **Step 3: Run it** — `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest models/tt_dit/tests/models/bria_fibo/test_transformer.py::test_fibo_transformer_reference_config -v`. Expected: PASS (confirms the reference class name/import, offline load, and the 46-block count). If it FAILS on the import name, record the working import and update `_load_ref_transformer` + the Global Constraints note.

- [ ] **Step 4: Create `transformer_bria_fibo.py`** with the SPDX header and the import block (mirror `transformer_flux1.py:1-30` imports; add `from ...blocks.transformer_block import TransformerBlock, _chunk_time3d`, `from .transformer_flux1 import Flux1SingleTransformerBlock, _re_fuse_proj_out_weight`, and the layers/linear/embeddings imports). No classes yet.

- [ ] **Step 5: Commit** — `git add models/tt_dit/models/transformers/transformer_bria_fibo.py models/tt_dit/tests/models/bria_fibo/ && git commit -m "feat(fibo-transformer): reference/config smoke test + skeleton"`

---

### Task 2: `BriaFiboTextProjection` + timestep-only embedding

**Files:** Modify `transformer_bria_fibo.py`; Test `test_transformer.py`.

**Interfaces:**
- Produces: `BriaFiboTextProjection(in_features=2048, hidden_size=1536, mesh_device, ...)` with `forward(x)->[...,1536]`; `BriaFiboTimestepEmbed(inner_dim, mesh_device)` with `forward(timestep)->[batch,inner_dim]`. Child names match HF (`caption_projection.N.*`, `time_text_embed.*`).

- [ ] **Step 1: Write failing tests** (tp=1, device) comparing each to the HF reference submodule:
```python
import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor as tt_tensor
from models.tt_dit.utils.check import assert_quality


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_fibo_text_projection(*, mesh_device):
    from models.tt_dit.models.transformers.transformer_bria_fibo import BriaFiboTextProjection
    m = _load_ref_transformer()
    ref = m.caption_projection[0]                      # HF BriaFiboTextProjection
    torch.manual_seed(0)
    x = torch.randn(1, 64, 2048)
    with torch.no_grad():
        r = ref(x)
    tt = BriaFiboTextProjection(in_features=2048, hidden_size=1536, mesh_device=mesh_device)
    tt.load_torch_state_dict(ref.state_dict())
    out = tt.forward(tt_tensor.from_torch(x, device=mesh_device))
    assert tuple(tt_tensor.to_torch(out).shape)[-1] == 1536
    assert_quality(r, tt_tensor.to_torch(out), pcc=0.99)
```
(Add an analogous `test_fibo_timestep_embed` comparing `BriaFiboTimestepEmbed` against the reference's timestep path: the reference exposes a `time_text_embed`/timestep embedder producing `[batch, inner_dim]` from `timestep`; inspect `m` to find the exact submodule and mirror the Flux `test` timestep handling — timestep passed as `timestep/1000` per the Flux test.)

- [ ] **Step 2: Run → fail** (ImportError).

- [ ] **Step 3: Implement.** `BriaFiboTextProjection` — inspect the HF module (`ref`): it is a small MLP (typically `linear_1` → activation(SiLU/GELU) → `linear_2`) mapping 2048→1536. Reproduce with tt_dit `Linear`/`ColParallelLinear` + the matching activation, child names matching the HF keys (verify via `ref.state_dict().keys()`). `BriaFiboTimestepEmbed` — reuse `Timesteps` + `TimestepEmbedding` from `layers/embeddings.py` (the timestep-only subset of `CombinedTimestepGuidanceTextProjEmbeddings`; no guidance, no pooled). Map weights from the reference's timestep submodule.

- [ ] **Step 4: Run → pass** (PCC ≥ 0.99 for both).
- [ ] **Step 5: Commit** — `feat(fibo-transformer): text projection + timestep embedding`.

---

### Task 3: dual block with injection (tp=1)

**Files:** Modify `transformer_bria_fibo.py`; Test `test_transformer.py`.

**Interfaces:**
- Produces: `inject_text(encoder_hidden_states, projected) -> ttnn.Tensor` (concat-halves; tp=1 = `ttnn.concat([enc[..., :1536], projected], dim=-1)`); confirms tt_dit's `TransformerBlock` reproduces FIBO's dual block given injected context.

- [ ] **Step 1: Write failing test** — build a 1-dual-block reduced reference and compare. Load `ref = _load_ref_transformer()`, take `ref_block = ref.transformer_blocks[0]` and `ref_proj = ref.caption_projection[0]`. Drive the reference block exactly as FIBO's forward does (inject then block) for one block, and compare against a tt `TransformerBlock` (from `blocks/transformer_block.py`, `context_pre_only=False`) fed the tt-injected context. Use the Flux single-block test's input-prep pattern (`bf16_tensor`, rope tensors) at tp=1. Assert PCC ≥ 0.99 on both returned streams.
  (Because reproducing the reference block's exact call signature is fiddly, an acceptable alternative gate: build a **1-dual-block, 0-single-block** `BriaFiboTransformer` (Task 4 class, `num_layers=1, num_single_layers=0`) and compare its output to a monkeypatched reference with `transformer_blocks=transformer_blocks[:1]`, `single_transformer_blocks=[]`. Prefer whichever the implementer can make faithful; document the choice.)

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** `inject_text` (tp=1 concat) and confirm `TransformerBlock` reuse. The dual block itself is reused unchanged from `blocks/transformer_block.py`; the only new code is `inject_text` and the loop wiring (fully realized in Task 4).
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(fibo-transformer): dual block + concat-halves injection (tp=1)`.

---

### Task 4: full `BriaFiboTransformer` + `BriaFiboCheckpoint` (tp=1)

**Files:** Modify `transformer_bria_fibo.py`; Test `test_transformer.py`.

**Interfaces:**
- Produces: `BriaFiboTransformer(*, patch_size, in_channels, num_layers, num_single_layers, num_attention_heads, attention_head_dim, joint_attention_dim, text_encoder_dim, out_channels, axes_dims_rope, mesh_device, ccl_manager, parallel_config, padding_config)`; `forward(spatial, prompt, timestep, text_encoder_layers, spatial_rope, prompt_rope, spatial_sequence_length, prompt_sequence_length) -> ttnn.Tensor`. `BriaFiboCheckpoint(name)` with `.build(ccl_manager, parallel_config)` and `.pos_embed` (the reference's RoPE helper).

- [ ] **Step 1: Write failing full-model test (tp=1, reduced depth via env, then full).** Adapt `tests/models/flux1/test_transformer_flux1.py::test_transformer` (lines 223-326) for FIBO:
  - Reference call: `ref.forward(hidden_states=spatial, encoder_hidden_states=prompt, text_encoder_layers=prompt_layers, timestep=timestep/1000, img_ids=image_ids, txt_ids=text_ids, ...).sample` (no pooled/guidance). `spatial` has **48** channels; `prompt` is 4096-dim; `prompt_layers` = a 46-entry list of `[batch, prompt_seq, 2048]` tensors (build random seeded, or slice/pad from a 37-length stand-in per the pipeline rule — see Global Constraints).
  - RoPE: `rope_cos, rope_sin = ref.pos_embed.forward(cat([text_ids, image_ids]))` (θ=10000 baked into the reference pos_embed).
  - tt: `BriaFiboCheckpoint(model_name).build(...)`, then `forward(...)` with tp=1 sharding (`bf16_tensor` inputs, sp=1 tp=1). Compose output, `assert_quality(ref_out, tt_out, pcc=0.99)`.
  - Reduced depth: gate `num_layers`/`num_single_layers` by an env knob (e.g. `FIBO_DUAL`, `FIBO_SINGLE`, defaulting to a small count) by truncating the reference's block ModuleLists and building the tt model with matching counts — mirrors the encoder's `N_LAYERS`.

- [ ] **Step 2: Run → fail** (class undefined).

- [ ] **Step 3: Implement `BriaFiboTransformer`.** Copy `Flux1Transformer` (`transformer_flux1.py:230-425`) and apply these changes:
  - `x_embedder = ColParallelLinear(in_channels=48, inner_dim, ...)`.
  - `context_embedder = ColParallelLinear(joint_attention_dim=4096, inner_dim, ...)` (unchanged from Flux structurally).
  - Replace `time_text_embed` (CombinedTimestepGuidanceTextProjEmbeddings) with `BriaFiboTimestepEmbed` (timestep-only). Drop `pooled`/`guidance` from the forward signature.
  - Add `self.caption_projection = ModuleList(BriaFiboTextProjection(2048, inner_dim//2, ...) for _ in range(num_layers + num_single_layers))`.
  - `transformer_blocks` = `num_layers` `TransformerBlock`s (as Flux). `single_transformer_blocks` = `num_single_layers` `Flux1SingleTransformerBlock`s (as Flux — already stream-separate).
  - forward: after `x_embedder`/`context_embedder`, thread a `block_id=0` counter; before each dual block AND each single block do `prompt = inject_text(prompt, self.caption_projection[block_id](text_encoder_layers[block_id])); block_id += 1`, then call the (unchanged) block. Keep Flux's `norm_out`/`time_embed_out`/`proj_out` tail.
  - `BriaFiboCheckpoint`: copy `Flux1Checkpoint` (`:428-495`) but load the reference `BriaFiboTransformer2DModel`, keep `.pos_embed`, set `out_channels=in_channels=48`, and map the FIBO state dict (the extra `caption_projection.N.*` and `context_embedder.*` load into the new modules; `time_text_embed`→`BriaFiboTimestepEmbed` name mapping via `_prepare_torch_state`). Inspect `ref.state_dict().keys()` to pin exact names.

- [ ] **Step 4: Run → pass** at reduced depth, then once at full 8+38 (tp=1). Record PCC. If full-depth tp=1 OOMs on one chip, note it and rely on the mesh (Task 5) for the full-depth gate; keep the reduced-depth tp=1 gate green.
- [ ] **Step 5: Commit** — `feat(fibo-transformer): full BriaFiboTransformer + checkpoint (tp=1)`.

---

### Task 5: tensor-parallel injection + full model on the 2×2 mesh

**Files:** Modify `transformer_bria_fibo.py` (`inject_text` for tp>1); Test `test_transformer.py`.

**Interfaces:**
- Produces: `inject_text` handling tp>1 (the concat-halves under feature-dim sharding).

- [ ] **Step 1: Write failing full-model mesh test.** Same as Task 4's full test but `mesh_device=(2,2)`, `sp_axis=0, tp_axis=1` (sp=2, tp=2), `device_params={"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 8192}`, `num_links=1`, inputs 2D-sharded (`bf16_tensor` with `mesh_axis=sp_axis, shard_dim=1` for spatial; rope sharded on sp; prompt replicated) exactly as `test_transformer_flux1.py::test_transformer`. Full 8+38 blocks. `assert_quality(ref_out, tt_out, pcc=0.99)`.

- [ ] **Step 2: Run → fail** (injection wrong under sharding, or a shape/CCL error).

- [ ] **Step 3: Implement tp>1 injection.** In tt_dit, `prompt` after `context_embedder` is feature-sharded on `tp_axis` (`inner_dim/tp` per device). The concat-halves keeps the first 1536 features and replaces the last 1536. For **tp=2** the 1536 boundary equals the shard boundary (device 0 holds features 0-1535, device 1 holds 1536-3071), so the operation is: keep device 0's shard, set device 1's shard = `caption_projection[block_id](text_layer)`. Implement `inject_text` for tp>1 as this shard-aligned replace — options to try, validating against the reference: (a) make `caption_projection` a `ColParallelLinear(2048 → 1536)` sharded so its output lands as the "second-half" shard, and assemble via the mesh layout; (b) all-gather `prompt` on `tp_axis`, do the plain concat on the full 3072, re-shard; (c) a ttnn slice/concat on the sharded dim if supported. Prefer (a) if achievable (no gather); fall back to (b) for correctness first. Document which was used and why.
  **If none of these yields PCC ≥ 0.99 without excessive complexity, STOP and report** — this is the plan's known research task; the controller may re-scope (e.g., land tp=1 correctness + a documented gather-based tp=2 path) rather than have you thrash.

- [ ] **Step 4: Run → pass** on the 2×2 mesh at full depth. Record seq lengths + PCC. If bf16 depth drift dips below 0.99, report the measured floor (do not silently lower).
- [ ] **Step 5: Commit** — `feat(fibo-transformer): tensor-parallel injection + full-mesh validation`.

---

### Task 6: Model doc

**Files:** Modify `models/tt_dit/models/BriaFibo.md`.

- [ ] **Step 1** Update the doc: mark sub-project 2 done; record the transformer architecture (8+38/48/3072/θ=10000), the concat-halves injection + the tp=2 shard-aligned handling, the reuse of Flux dual/single blocks, how to run the tests (offline command, `FIBO_PATH`, the `FIBO_DUAL`/`FIBO_SINGLE` reduced-depth knobs, weights pre-download), and the measured PCCs (tp=1 reduced + full, 2×2 full).
- [ ] **Step 2: Commit** — `docs(fibo-transformer): BriaFibo.md sub-project 2 section`.

---

## Self-Review

**Spec coverage:** transformer forward incl. 8 dual + 38 single, concat-halves injection, context_embedder, timestep-only temb, in_channels 48, RoPE θ=10000 (Tasks 2-5) ✓; weight loading via BriaFiboCheckpoint (Task 4) ✓; tp=1 + 2×2 mesh validation vs reference at PCC≥0.99 (Tasks 4-5) ✓; the 46-entry text_encoder_layers consumed by index, list-construction deferred to the pipeline (test builds it) ✓; doc (Task 6) ✓. Open spec items (single-block re-split, reference class name, in_channels packing) are resolved (tt_dit single block already stream-separate → no re-split; class name confirmed in Task 1; packing deferred, test uses 48-ch latents).

**Placeholder scan:** implementation steps that require reading the reference (exact `BriaFiboTextProjection` internal structure, exact state_dict key names, the timestep submodule) are marked as "inspect `ref`/`ref.state_dict().keys()`" — legitimate adaptation instructions, not placeholders. The tp>1 injection (Task 5, Step 3) is an explicit research task with concrete candidate approaches + an escalation gate rather than pre-written code, because it depends on ttnn sharded-op behavior that must be validated on device.

**Type consistency:** `BriaFiboTextProjection`, `BriaFiboTimestepEmbed`, `inject_text`, `BriaFiboTransformer(...)`, `BriaFiboCheckpoint` names and the forward signature are used consistently across tasks; block/child names (`transformer_blocks`, `single_transformer_blocks`, `caption_projection`, `context_embedder`, `x_embedder`, `norm_out`, `proj_out`) match the HF reference for clean loading.

**Note on fidelity:** this component is more mesh-intensive and diverges more from its template than the encoder; Tasks 3-4 give copy-from-Flux deltas + complete tests, and Task 5 (the tensor-parallel injection) is the genuine research task, deliberately ordered last and gated by the working tp=1 model.
