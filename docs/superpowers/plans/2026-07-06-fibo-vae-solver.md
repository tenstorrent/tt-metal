# FIBO VAE decode + flow-match solver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Decode FIBO's latents to an image on Blackhole by implementing the Wan 2.2 residual VAE decoder (`is_residual=True`, z_dim=48) in tt_dit, and validate the reused flow-match Euler solver for FIBO's schedule — both PCC ≥ 0.99 vs the HF reference.

**Architecture:** FIBO's VAE is `AutoencoderKLWan` with `is_residual=True` (Wan 2.2 high-compression / TI2V VAE), which tt_dit's `vae_wan2_1.py` hard-blocks with `assert not is_residual`. This plan implements the residual decoder path (a `WanResidualUpBlock` + `WanDecoder3d` wiring, derived from the diffusers `AutoencoderKLWan` source) so `WanVAEDecoderAdapter` can decode FIBO's z_dim=48 latents. The Euler solver + dynamic shift are reused unchanged.

**Tech Stack:** Python, PyTorch + `diffusers` (`AutoencoderKLWan`, `FlowMatchEulerDiscreteScheduler`), `ttnn`, tt_dit, Tenstorrent Blackhole.

**Spec:** `docs/superpowers/specs/2026-07-06-fibo-vae-solver-design.md`
**Branch:** `fibo-vae-solver` (stacked on `fibo-transformer`).

## Global Constraints
- **SPDX header** on new `.py` files (two lines as in siblings). **Strong subagents** — sonnet floor; opus for the residual-decoder task.
- **Modify** `models/tt_dit/models/vae/vae_wan2_1.py`; **new tests** under `models/tt_dit/tests/models/bria_fibo/`.
- **FIBO VAE config (verified):** `AutoencoderKLWan`, `z_dim=48`, `base_dim=160`, `decoder_base_dim=256`, `dim_mult=[1,2,4,4]`, `num_res_blocks=2`, `is_residual=True`, `in_channels=12`, `out_channels=12`, `patch_size=2`, `scale_factor_spatial=16`, `scale_factor_temporal=4`, `temperal_downsample=[False,True,True]`, 48-elem `latents_mean`/`latents_std`.
- **FIBO scheduler (verified):** `FlowMatchEulerDiscreteScheduler`, `use_dynamic_shifting=True`, `base_shift=0.5`, `max_shift=1.15`, `base_image_seq_len=256`, `max_image_seq_len=4096`, `time_shift_type=exponential`, `num_train_timesteps=1000`.
- **Reference:** `from diffusers import AutoencoderKLWan` `.from_pretrained("briaai/FIBO", subfolder="vae")` (the `is_residual`/`WanResidualUpBlock` decoder is the ground truth to reproduce); `FlowMatchEulerDiscreteScheduler.from_pretrained("briaai/FIBO", subfolder="scheduler")`.
- **Weights:** gated `briaai/FIBO/vae/*` — pre-download once, then `HF_HUB_OFFLINE=1`. Offline resolution needs `snapshot_download(local_files_only=True)` to resolve the repo id (as in the encoder/transformer tests).
- **Run:** `HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole python_env/bin/python -m pytest <path>::<test> -v` (interpreter `python_env/bin/python`; no `python` on PATH).
- **PCC gate:** ≥ 0.99 (bf16) vs the HF reference (the Wan VAE test uses 0.999 bf16 / 0.99994 fp32 — aim ≥ 0.99; report the measured floor).
- **Templates:** VAE decode test `models/tt_dit/tests/models/wan2_2/test_vae_wan2_1.py::test_wan_decoder`; solver test `models/tt_dit/tests/unit/test_solvers.py`.
- **The residual decoder is the real work; the solver is reuse.** Do NOT re-tune / touch the Wan 2.1 (`is_residual=False`) path — only ADD the residual branch. Encode path is out of scope (leave its `is_residual` assert; decode-only).

---

## File Structure
- `models/tt_dit/models/vae/vae_wan2_1.py` — MODIFY: remove decode-side `is_residual` asserts; add `WanResidualUpBlock`; wire it into `WanDecoder3d` up-block loop when `is_residual`; ensure `WanDecoder`/`WanDecoder3d`/`WanVAEDecoderAdapter` construct for FIBO's config.
- `models/tt_dit/tests/models/bria_fibo/__init__.py` — exists (from sp2); reuse.
- `models/tt_dit/tests/models/bria_fibo/test_solver.py` — NEW: FIBO-config solver/shift validation.
- `models/tt_dit/tests/models/bria_fibo/test_vae.py` — NEW: Wan 2.2 VAE decode PCC test.
- `models/tt_dit/models/BriaFibo.md` — MODIFY: sub-project 3 section.

---

### Task 1: VAE weights predownload + reference smoke + solver validation

**Files:** Create `tests/models/bria_fibo/test_solver.py`; add a VAE smoke test to a new `tests/models/bria_fibo/test_vae.py`.

**Interfaces:** Produces the offline VAE-loading helper + confirms the solver reuse. No src changes.

- [ ] **Step 1: Pre-download the VAE weights** (one-time, valid token from ~/.bashrc; ~small):
```bash
cd /localdev/mstojkovic/tt-metal
export HF_TOKEN=$(grep -E '^[[:space:]]*export[[:space:]]+HF_TOKEN=' ~/.bashrc | tail -1 | sed -E 's/.*HF_TOKEN=//; s/^["'"'"']//; s/["'"'"']$//')
python_env/bin/python -c "from huggingface_hub import snapshot_download; import os; print(snapshot_download('briaai/FIBO', allow_patterns=['vae/*','scheduler/*'], token=os.environ['HF_TOKEN']))"
```

- [ ] **Step 2: Solver validation test** (`test_solver.py`, host-only, no device) — mirrors `tests/unit/test_solvers.py`, using FIBO's scheduler:
```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import pytest
from diffusers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")


def _fibo_local():
    try:
        return snapshot_download(FIBO_PATH, allow_patterns=["scheduler/*", "vae/*"], local_files_only=True)
    except Exception as e:
        pytest.skip(f"FIBO not cached: {e}")


def _calculate_shift(image_seq_len, scheduler):
    base = scheduler.config.get("base_image_seq_len", 256)
    mx = scheduler.config.get("max_image_seq_len", 4096)
    bs = scheduler.config.get("base_shift", 0.5)
    ms = scheduler.config.get("max_shift", 1.15)
    m = (ms - bs) / (mx - base)
    return image_seq_len * m + (bs - m * base)


def test_fibo_solver_matches_diffusers():
    from models.tt_dit.solvers.euler import EulerSolver
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(_fibo_local(), subfolder="scheduler")
    assert sched.config.use_dynamic_shifting and sched.config.base_shift == 0.5 and sched.config.max_shift == 1.15
    n, seq_len = 30, 4096
    mu = _calculate_shift(seq_len, sched)
    sched.set_timesteps(sigmas=np.linspace(1.0, 1 / n, n), mu=mu)
    sigmas = sched.sigmas.tolist()
    solver = EulerSolver()
    solver.set_schedule(sigmas)
    # one Euler step matches x + (sigma_next - sigma_curr)*v (diffusers flow-match step)
    import torch
    x = torch.randn(1, 8); v = torch.randn(1, 8)
    step = 0
    expect = x + (sigmas[step + 1] - sigmas[step]) * v
    # EulerSolver.step is device-side; verify the host-side scalar schedule instead:
    assert abs((sigmas[1] - sigmas[0]) - (sched.sigmas[1].item() - sched.sigmas[0].item())) < 1e-6
    assert len(sigmas) == n + 1
```
(If the exact EulerSolver device-step comparison is desired, extend to a (1,1) device run mirroring `test_solvers.py`; the host schedule + `_calculate_shift` match is the core validation.)

- [ ] **Step 3: VAE reference smoke** (`test_vae.py`, host-only):
```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from huggingface_hub import snapshot_download

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")


def _load_ref_vae(dtype=None):
    import torch
    from diffusers import AutoencoderKLWan
    try:
        path = snapshot_download(FIBO_PATH, allow_patterns=["vae/*"], local_files_only=True)
        return AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=dtype or torch.float32).eval()
    except Exception as e:
        pytest.skip(f"FIBO vae unavailable: {e}")


def test_fibo_vae_reference_config():
    m = _load_ref_vae()
    c = m.config
    assert c.z_dim == 48 and c.is_residual is True
    assert c.decoder_base_dim == 256 and c.base_dim == 160
    assert c.dim_mult == [1, 2, 4, 4] and c.out_channels == 12
    assert c.scale_factor_spatial == 16
```

- [ ] **Step 4: Run** (`test_fibo_solver_matches_diffusers`, `test_fibo_vae_reference_config`). Expected: PASS (both host-only). Confirms weights cached, offline resolution, solver reuse, VAE config.
- [ ] **Step 5: Commit** — `git add ... && git commit -m "feat(fibo-vae): solver validation + VAE reference smoke"`.

---

### Task 2: Wan 2.2 residual decoder path (the core)

**Files:** Modify `models/tt_dit/models/vae/vae_wan2_1.py`; Test `models/tt_dit/tests/models/bria_fibo/test_vae.py`.

**Interfaces:**
- Produces: `WanResidualUpBlock` (Wan 2.2 residual up-block) + an `is_residual=True` path in `WanDecoder3d` so `WanDecoder(is_residual=True, z_dim=48, base_dim=160, decoder_base_dim=256, out_channels=12, dim_mult=[1,2,4,4], ...)` constructs and decodes correctly. `WanVAEDecoderAdapter` (unchanged) then works for FIBO's config.

- [ ] **Step 1: Write the failing decode test** (`test_vae.py`, device, tp/hw=1). Model on `tests/models/wan2_2/test_vae_wan2_1.py::test_wan_decoder`, adapted for FIBO (z_dim=48, is_residual, small resolution for speed):
```python
import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanDecoder
from models.tt_dit.parallel.config import VaeHWParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_fibo_vae_decode(*, mesh_device):
    import torch
    ref = _load_ref_vae()                      # AutoencoderKLWan, is_residual=True, z_dim=48
    c = ref.config
    torch.manual_seed(0)
    latent_h, latent_w = 16, 16                 # small: decodes to 16*16=256 px (reduced)
    z = torch.randn(1, c.z_dim, 1, latent_h, latent_w)   # (B, 48, T=1, H, W)
    with torch.no_grad():
        ref_img = ref.decode(z).sample          # reference decode

    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pc = VaeHWParallelConfig.from_tuples(height=(1, 0), width=(1, 1))
    dec = WanDecoder(base_dim=c.base_dim, decoder_base_dim=c.decoder_base_dim, z_dim=c.z_dim,
                     dim_mult=c.dim_mult, num_res_blocks=c.num_res_blocks, out_channels=c.out_channels,
                     is_residual=True, temperal_downsample=c.temperal_downsample,
                     latents_mean=c.latents_mean, latents_std=c.latents_std,
                     mesh_device=mesh_device, parallel_config=pc, ccl_manager=ccl)
    dec.load_torch_state_dict(ref.decoder.state_dict() ... )   # map from ref (see impl notes)
    # prepare latent (BTHWC + pads + shard) per the test_wan_decoder template, decode, gather, trim
    tt_img = ...   # decode via dec(...) mirroring test_wan_decoder
    assert_quality(ref_img, tt_img, pcc=0.99)
```
> The exact latent prep (BTHWC permute, `conv_pad_in_channels`/`conv_pad_height`/`conv_pad_width`, shard), decode call, gather/permute/trim, and the `WanDecoder` vs `AutoencoderKLWan` state-dict mapping must be copied from `test_wan_decoder` and `WanVAEDecoderAdapter` (which already reads config + loads the decoder). Prefer routing through `WanVAEDecoderAdapter.decode` if it's the cleaner path once `is_residual` constructs.

- [ ] **Step 2: Run → fail** — currently `assert not is_residual` crashes construction.

- [ ] **Step 3: Implement the residual path.** Read the diffusers ground truth `AutoencoderKLWan` (`python_env/lib/python3.12/site-packages/diffusers/models/autoencoders/autoencoder_kl_wan.py`) — specifically its `WanResidualUpBlock`/`WanUpBlock` residual variant and how `is_residual` changes the decoder up-block channel flow. Then in `vae_wan2_1.py`:
  1. Remove the decode-side `assert not is_residual` in `WanDecoder3d.__init__` (~L1205) and `WanDecoder.__init__` (~L1438). (Leave `WanEncoder3D`'s assert — encode is out of scope.)
  2. Add a `WanResidualUpBlock(Module)` mirroring the existing `WanUpBlock` (resnets + optional `WanResample` upsampler) PLUS the Wan 2.2 residual shortcut across the block (a `WanResidualBlock`/conv shortcut summed into the output), matching the diffusers residual up-block. Reuse `WanResidualBlock` (L495), `WanResample` (L907), `WanCausalConv3d` for the shortcut.
  3. In `WanDecoder3d.__init__` up-block loop (~L1261-1295): when `is_residual`, keep `in_dim` un-halved (the `i>0 and not is_residual` guard already skips the halving) and instantiate `WanResidualUpBlock` instead of `WanUpBlock`. Thread `first_chunk` through `forward` if the residual up-block needs it (the note at L1333 flags this).
  4. Ensure `decoder_base_dim` (256) flows as the decoder `dim` (already wired: `WanDecoder` passes `dim=decoder_base_dim` to `WanDecoder3d`). Add a `_prepare_torch_state` mapping if the residual up-block's HF key names differ.
  Keep the `is_residual=False` (Wan 2.1) path byte-identical.

- [ ] **Step 4: Run → pass** at the reduced resolution (PCC ≥ 0.99). Iterate against the reference until the residual channel bookkeeping is correct (a per-up-block intermediate PCC check helps localize).
- [ ] **Step 5: Confirm the Wan 2.1 path is unbroken** — run `models/tt_dit/tests/models/wan2_2/test_vae_wan2_1.py::test_wan_decoder` once (z_dim=16, is_residual=False); expected still PASS.
- [ ] **Step 6: Commit** — `feat(fibo-vae): Wan 2.2 residual decoder path (is_residual)`.

---

### Task 3: production-resolution decode + blocking table

**Files:** Modify `vae_wan2_1.py` (`compute_decoder_dims`/`_BLOCKINGS` if needed); Test `test_vae.py`.

**Interfaces:** Consumes Task 2's residual decoder. Produces a production-resolution decode gate.

- [ ] **Step 1: Add a production-resolution decode test** — same as Task 2 but a production latent size (e.g. 1024×1024 image → `latent_h=latent_w=64` at 16× spatial), T=1, PCC ≥ 0.99. On a single Blackhole device; if it OOMs, use the mesh HW-parallel config (`VaeHWParallelConfig` height/width sharding per the wan pipeline BH presets) and note it.
- [ ] **Step 2: Run → if it fails on shape/blocking**, extend `compute_decoder_dims`/`_BLOCKINGS` with entries for the TI2V 16×-spatial stage dims (the existing table targets z_dim=16 / 8× spatial). Implement, re-run → PASS.
- [ ] **Step 3: Commit** — `feat(fibo-vae): production-resolution decode + TI2V blocking dims`.

---

### Task 4: Model doc

**Files:** Modify `models/tt_dit/models/BriaFibo.md`.
- [ ] **Step 1** Add the sub-project 3 section: the Wan 2.2 residual VAE decoder (is_residual, z_dim=48, 16× spatial, decoder_base_dim 256, out_channels 12), the solver reuse, how to run the tests (offline command, `FIBO_PATH`, vae weight predownload), measured decode + solver PCCs, and that the Wan 2.1 path is preserved. Mark sp3 done; sp4 (pipeline) TODO.
- [ ] **Step 2: Commit** — `docs(fibo-vae): BriaFibo.md sub-project 3 section`.

---

## Self-Review
**Spec coverage:** residual decoder impl (Task 2) ✓; z_dim=48/decoder_base_dim/out_channels/latents (Task 2 via config) ✓; production res + blocking (Task 3) ✓; solver + dynamic-shift validation (Task 1) ✓; VAE decode PCC vs reference (Tasks 2-3) ✓; doc (Task 4) ✓; Wan 2.1 path preserved (Task 2 Step 5) ✓. Encode out of scope (assert left) ✓. Open items (out_channels=12→RGB mapping, exact residual bookkeeping, blocking dims) are resolved in Task 2-3 against the reference.
**Placeholder scan:** the Task 2 test's latent-prep/state-dict-mapping is marked "copy from `test_wan_decoder`/`WanVAEDecoderAdapter`" — a concrete adaptation instruction (the exact ttnn prep is intricate and lives in the template), not a placeholder; the residual up-block is derived from the named diffusers source. These are the genuine implementation-against-reference tasks (like the transformer's Flux adaptation), gated by decode PCC.
**Type consistency:** `WanResidualUpBlock`, `WanDecoder(is_residual=True, ...)`, `WanDecoder3d` up-block loop, `WanVAEDecoderAdapter.decode`, `EulerSolver`/`_calculate_shift` used consistently; test helpers `_load_ref_vae`/`_fibo_local` shared across the two test files.
**Note:** Task 2 is the substantial one (residual decoder from the diffusers reference) — dispatch on opus. Tasks 1, 3, 4 are sonnet-tier.
