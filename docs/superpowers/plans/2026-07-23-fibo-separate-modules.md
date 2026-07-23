# FIBO — Separate FIBO Changes Into FIBO-Owned Modules — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every FIBO-specific change out of shared tt_dit library files into FIBO-owned files, with byte-identical FIBO behavior and performance.

**Architecture:** A pure re-organization. The VAE residual code is forked verbatim into `vae_bria_fibo.py` and the shared Wan VAE is restored to `main`. FIBO conv-blocking data moves to a FIBO-owned file and is injected via the shared registration API (extended generically to cover the exact-key table). The 12×10 Galaxy grid stays shared (reframed as a general capability). Low-value/backward-compatible touchpoints (encoder parallel config, sweep dev-tool configs) are optional trailing tasks.

**Tech Stack:** Python, ttnn, pytest, Tenstorrent Blackhole (2×2 and 4×8 Galaxy meshes).

## Global Constraints

- **Behavior identical.** FIBO image output PCC vs. HF reference must match baseline exactly.
- **Performance identical.** No FIBO op may regress; all conv-blocking and matmul configs remain in effect for FIBO.
- **Copying whole files is acceptable** where surgical extraction would require editing shared internals (user-approved).
- **Other models unaffected.** Reverting shared files must leave QwenImage / Wan / SD3.5 / Flux / Motif / LTX working (they equal `main`, which is known-good).
- **Merge-base for reverts:** `2eb991a4100117464d67d5331bd6968470bf9daa` (branch `fibo-pipeline` vs `main`). Referred to below as `$MB`.
- Every touched Python file must pass the repo pre-commit hooks (black, isort, autoflake, trailing-whitespace) — they run automatically on `git commit`.

## File Structure

**New FIBO-owned files:**
- `models/tt_dit/models/vae/vae_bria_fibo.py` — full fork of the Wan VAE incl. residual decoder.
- `models/tt_dit/pipelines/bria_fibo/fibo_conv3d_configs.py` — FIBO conv-blocking data + a `register()` call.
- `models/tt_dit/pipelines/bria_fibo/fibo_parallel_config.py` — FIBO encoder parallel config *(optional, Task 6)*.
- `models/tt_dit/utils/sweep_mm_block_sizes_fibo.py` — FIBO sweep configs *(optional, Task 7)*.

**Shared files restored to `$MB` (pristine `main`):**
- `models/tt_dit/models/vae/vae_wan2_1.py`
- `models/tt_dit/parallel/config.py` *(only if Task 6 is done)*

**Shared files kept with a generic (non-FIBO) enhancement:**
- `models/tt_dit/utils/conv3d.py` — `register_conv3d_configs` extended to route by key arity.
- `models/tt_dit/utils/matmul.py` — 12×10 grid retained, comment reframed.

---

## Task 0: Capture behavior + performance baseline

**Files:** none (records reference numbers).

**Interfaces:**
- Produces: baseline PCC and perf numbers stored in `/tmp/claude-4165/-home-mstojkovic-tt-metal/539fe323-7316-4ce2-a265-6797db6d4539/scratchpad/fibo_baseline.txt` for comparison in Task 8.

- [ ] **Step 1: Record current HEAD and clean working state note**

Run: `git -C /home/mstojkovic/tt-metal rev-parse HEAD`
Note the working tree already has uncommitted edits to `pipeline_bria_fibo.py` and `test_performance_bria_fibo.py`; leave them as-is (they are not part of this re-org unless a task edits them).

- [ ] **Step 2: Run the FIBO VAE decode PCC test and save output**

Run (on a machine with the target mesh):
```bash
cd /home/mstojkovic/tt-metal
pytest models/tt_dit/tests/models/bria_fibo/test_vae.py -v 2>&1 | tee /tmp/claude-4165/-home-mstojkovic-tt-metal/539fe323-7316-4ce2-a265-6797db6d4539/scratchpad/fibo_baseline_vae.txt
```
Expected: PASS. Record the reported PCC value(s).

- [ ] **Step 3: Run the FIBO performance harness and save output**

Run:
```bash
cd /home/mstojkovic/tt-metal
pytest models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py -v 2>&1 | tee /tmp/claude-4165/-home-mstojkovic-tt-metal/539fe323-7316-4ce2-a265-6797db6d4539/scratchpad/fibo_baseline_perf.txt
```
Expected: PASS. Record end-to-end and any per-stage timings.

- [ ] **Step 4: No commit** (baseline capture only).

---

## Task 1: Fork the Wan VAE into `vae_bria_fibo.py`

**Files:**
- Create: `models/tt_dit/models/vae/vae_bria_fibo.py` (copied from current HEAD `vae_wan2_1.py`)
- Modify: `models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py:44`
- Modify: `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py:713`
- Modify: `models/tt_dit/tests/models/bria_fibo/test_vae.py:53,147`

**Interfaces:**
- Produces: module `models.tt_dit.models.vae.vae_bria_fibo` exporting the same public names as `vae_wan2_1` today, notably `WanVAEDecoderAdapter` and `WanDecoder` (identical signatures — it is a verbatim copy).

- [ ] **Step 1: Create the fork from the current HEAD blob (guarantees identical logic)**

Run:
```bash
cd /home/mstojkovic/tt-metal
git show HEAD:models/tt_dit/models/vae/vae_wan2_1.py > models/tt_dit/models/vae/vae_bria_fibo.py
```

- [ ] **Step 2: Add a fork-provenance header to the new file**

Insert immediately after the SPDX header lines at the top of `models/tt_dit/models/vae/vae_bria_fibo.py`:
```python
# NOTE: This is a FIBO-owned fork of ``vae_wan2_1.py`` taken at the commit that added Wan 2.2
# residual-decoder support (WanDupUp3D / WanResidualUpBlock / is_residual / first_chunk /
# decoder_base_dim, and the SDPA k_chunk_size autoscale). It is intentionally self-contained so
# FIBO changes do not live in the shared Wan 2.1 VAE. Shared fixes to vae_wan2_1.py do NOT
# auto-propagate here — port them manually if relevant.
```

- [ ] **Step 3: Point the FIBO pipeline import at the fork**

In `models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py`, change line 44 from:
```python
from models.tt_dit.models.vae.vae_wan2_1 import WanVAEDecoderAdapter
```
to:
```python
from models.tt_dit.models.vae.vae_bria_fibo import WanVAEDecoderAdapter
```

- [ ] **Step 4: Point the FIBO tests at the fork**

In `models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py:713` change
`from models.tt_dit.models.vae.vae_wan2_1 import WanVAEDecoderAdapter` →
`from models.tt_dit.models.vae.vae_bria_fibo import WanVAEDecoderAdapter`.

In `models/tt_dit/tests/models/bria_fibo/test_vae.py` change both occurrences (lines 53 and 147)
`from models.tt_dit.models.vae.vae_wan2_1 import WanDecoder` →
`from models.tt_dit.models.vae.vae_bria_fibo import WanDecoder`.

- [ ] **Step 5: Restore the shared Wan VAE to `$MB` (removes residual/FIBO additions)**

Run:
```bash
cd /home/mstojkovic/tt-metal
git checkout 2eb991a4100117464d67d5331bd6968470bf9daa -- models/tt_dit/models/vae/vae_wan2_1.py
```

- [ ] **Step 6: Verify the shared file is now pristine**

Run:
```bash
cd /home/mstojkovic/tt-metal
git diff 2eb991a4100117464d67d5331bd6968470bf9daa -- models/tt_dit/models/vae/vae_wan2_1.py
```
Expected: empty output (no diff).

- [ ] **Step 7: Import smoke test for both modules**

Run:
```bash
cd /home/mstojkovic/tt-metal
python -c "import models.tt_dit.models.vae.vae_bria_fibo as f; import models.tt_dit.models.vae.vae_wan2_1 as s; print('fibo has WanVAEDecoderAdapter:', hasattr(f,'WanVAEDecoderAdapter')); print('fibo has WanResidualUpBlock:', hasattr(f,'WanResidualUpBlock')); print('shared has WanResidualUpBlock:', hasattr(s,'WanResidualUpBlock'))"
```
Expected: `fibo has WanVAEDecoderAdapter: True`, `fibo has WanResidualUpBlock: True`, `shared has WanResidualUpBlock: False`.

- [ ] **Step 8: Verify QwenImage still imports the shared VAE**

Run:
```bash
cd /home/mstojkovic/tt-metal
python -c "import models.tt_dit.models.vae.vae_qwenimage; print('qwenimage vae import OK')"
```
Expected: `qwenimage vae import OK`.

- [ ] **Step 9: Run the FIBO VAE test — must match baseline PCC**

Run:
```bash
cd /home/mstojkovic/tt-metal
pytest models/tt_dit/tests/models/bria_fibo/test_vae.py -v
```
Expected: PASS with PCC equal to the Task 0 baseline.

- [ ] **Step 10: Commit**

```bash
cd /home/mstojkovic/tt-metal
git add models/tt_dit/models/vae/vae_bria_fibo.py models/tt_dit/models/vae/vae_wan2_1.py \
  models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py \
  models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py \
  models/tt_dit/tests/models/bria_fibo/test_vae.py
git commit -m "refactor(fibo-pipeline): fork Wan 2.2 residual VAE into vae_bria_fibo.py; revert shared vae_wan2_1.py to main"
```

---

## Task 2: Move FIBO conv-blocking data to a FIBO-owned file

**Files:**
- Modify: `models/tt_dit/utils/conv3d.py` (extend `register_conv3d_configs`, then revert the FIBO data additions)
- Create: `models/tt_dit/pipelines/bria_fibo/fibo_conv3d_configs.py`
- Modify: `models/tt_dit/models/vae/vae_bria_fibo.py` (register FIBO configs on import)

**Interfaces:**
- Consumes: `register_conv3d_configs` from `models.tt_dit.utils.conv3d`.
- Produces: `models.tt_dit.pipelines.bria_fibo.fibo_conv3d_configs.register_fibo_conv3d_configs()` — idempotent function that injects all FIBO blocking entries into the shared conv3d tables.

- [ ] **Step 1: Restore the shared conv3d file to `$MB` (removes ALL FIBO data + resets the API)**

Run:
```bash
cd /home/mstojkovic/tt-metal
git checkout 2eb991a4100117464d67d5331bd6968470bf9daa -- models/tt_dit/utils/conv3d.py
```
This wipes the inline FIBO `_BLOCKINGS`/`_DEFAULT_BLOCKINGS` entries and restores the original
`register_conv3d_configs` (which only handles `_DEFAULT_BLOCKINGS`). The next step re-adds the
generic API enhancement on top of the pristine file.

- [ ] **Step 2: Extend `register_conv3d_configs` to also populate the exact-key table**

In the now-restored `models/tt_dit/utils/conv3d.py`, replace the body of `register_conv3d_configs` (currently only updates `_DEFAULT_BLOCKINGS`) with a generic router that dispatches by key length. New implementation:
```python
def register_conv3d_configs(configs: dict) -> None:
    """Register additional conv3d blocking configs from external models.

    Keys with 3 elements ``(in_channels, out_channels, kernel_size)`` are added to the fallback
    table (``_DEFAULT_BLOCKINGS``). Keys with 8 elements
    ``(h_factor, w_factor, in_channels, out_channels, kernel_size, T, H, W)`` are added to the
    exact-match table (``_BLOCKINGS``). Values are
    ``(C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)``.

    Example::

        register_conv3d_configs({
            (32, 96, (3, 3, 3)): (32, 96, 1, 8, 16),                 # fallback (channel) key
            (2, 2, 1024, 1024, (3, 3, 3), 3, 64, 64): (64, 256, 1, 4, 8),  # exact key
        })
    """
    for key, value in configs.items():
        if len(key) == 3:
            c_in, c_out, ks = key
            _DEFAULT_BLOCKINGS[(c_in, c_out, _ntuple(ks, 3))] = tuple(value)
        elif len(key) == 8:
            h, w, c_in, c_out, ks, T, H, W = key
            _BLOCKINGS[(h, w, c_in, c_out, _ntuple(ks, 3), T, H, W)] = tuple(value)
        else:
            raise ValueError(f"register_conv3d_configs: key must have 3 or 8 elements, got {len(key)}: {key}")
```

- [ ] **Step 3: Create the FIBO conv-blocking file with the exact entries**

Create `models/tt_dit/pipelines/bria_fibo/fibo_conv3d_configs.py`:
```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FIBO-owned conv3d blocking configs.

These were previously inlined into the shared ``utils/conv3d.py`` ``_BLOCKINGS`` /
``_DEFAULT_BLOCKINGS`` tables. They are registered into the shared tables at import of the FIBO
VAE via :func:`register_fibo_conv3d_configs`, so the shared file carries no FIBO data.
"""
from ...utils.conv3d import register_conv3d_configs

# Exact-match entries: (h_factor, w_factor, C_in, C_out, kernel, T, H, W) -> blocking.
_FIBO_EXACT_BLOCKINGS = {
    # BH 2x2, FIBO image decode (1024x1024, latent T=1, full-T uncached). h_factor=2, w_factor=2.
    (2, 2, 1024, 1024, (3, 3, 3), 3, 64, 64): (64, 256, 1, 4, 8),
    (2, 2, 1024, 1024, (3, 3, 3), 3, 128, 128): (64, 256, 1, 2, 16),
    (2, 2, 512, 512, (3, 3, 3), 3, 256, 256): (64, 256, 1, 8, 4),
    (2, 2, 256, 256, (3, 3, 3), 3, 512, 512): (64, 256, 1, 8, 4),
    (2, 2, 64, 1024, (3, 3, 3), 3, 64, 64): (32, 512, 1, 8, 4),
    (2, 2, 1024, 512, (3, 3, 3), 3, 256, 256): (64, 256, 1, 8, 4),
    (2, 2, 512, 256, (3, 3, 3), 3, 512, 512): (64, 256, 1, 8, 4),
    (2, 2, 256, 12, (3, 3, 3), 3, 512, 512): (128, 32, 1, 16, 2),
    (2, 2, 1024, 1024, (1, 3, 3), 1, 128, 128): (256, 128, 1, 4, 8),
    (2, 2, 1024, 1024, (1, 3, 3), 1, 256, 256): (256, 128, 1, 4, 8),
    (2, 2, 512, 512, (1, 3, 3), 1, 512, 512): (256, 128, 1, 8, 4),
    (2, 2, 1024, 2048, (3, 1, 1), 2, 64, 64): (256, 512, 1, 2, 16),
    (2, 2, 1024, 2048, (3, 1, 1), 2, 128, 128): (256, 512, 1, 4, 8),
    # BH Galaxy 4x8, FIBO image decode. h_factor=8, w_factor=4.
    (8, 4, 1024, 1024, (3, 3, 3), 3, 16, 32): (64, 256, 1, 8, 4),
    (8, 4, 1024, 1024, (3, 3, 3), 3, 32, 64): (64, 256, 1, 8, 4),
    (8, 4, 512, 512, (3, 3, 3), 3, 64, 128): (64, 256, 1, 4, 8),
    (8, 4, 256, 256, (3, 3, 3), 3, 128, 256): (64, 256, 1, 16, 2),
    (8, 4, 64, 1024, (3, 3, 3), 3, 16, 32): (64, 256, 1, 8, 4),
    (8, 4, 1024, 512, (3, 3, 3), 3, 64, 128): (64, 256, 1, 8, 4),
    (8, 4, 512, 256, (3, 3, 3), 3, 128, 256): (64, 256, 1, 16, 2),
    (8, 4, 256, 12, (3, 3, 3), 3, 128, 256): (128, 32, 1, 16, 2),
    (8, 4, 1024, 1024, (1, 3, 3), 1, 32, 64): (256, 128, 1, 4, 8),
    (8, 4, 1024, 1024, (1, 3, 3), 1, 64, 128): (256, 128, 1, 2, 16),
    (8, 4, 512, 512, (1, 3, 3), 1, 128, 256): (128, 256, 1, 2, 16),
    (8, 4, 1024, 2048, (3, 1, 1), 2, 16, 32): (256, 256, 1, 2, 16),
    (8, 4, 1024, 2048, (3, 1, 1), 2, 32, 64): (256, 512, 1, 16, 2),
}

# Fallback channel-keyed defaults: (C_in, C_out, kernel) -> blocking.
_FIBO_DEFAULT_BLOCKINGS = {
    (64, 1024, (3, 3, 3)): (64, 32, 1, 1, 1),
    (1024, 1024, (3, 3, 3)): (256, 32, 1, 1, 1),
    (1024, 1024, (1, 3, 3)): (256, 32, 1, 1, 1),
    (1024, 2048, (3, 1, 1)): (256, 32, 1, 1, 1),
    (1024, 512, (3, 3, 3)): (256, 32, 1, 1, 1),
    (512, 512, (3, 3, 3)): (256, 32, 1, 1, 1),
    (512, 512, (1, 3, 3)): (256, 32, 1, 1, 1),
    (512, 256, (3, 3, 3)): (256, 32, 1, 1, 1),
    (256, 256, (3, 3, 3)): (256, 32, 1, 8, 8),
    (256, 12, (3, 3, 3)): (256, 32, 1, 8, 8),
}

_registered = False


def register_fibo_conv3d_configs() -> None:
    """Idempotently inject FIBO conv3d blockings into the shared conv3d tables."""
    global _registered
    if _registered:
        return
    register_conv3d_configs(_FIBO_EXACT_BLOCKINGS)
    register_conv3d_configs(_FIBO_DEFAULT_BLOCKINGS)
    _registered = True
```

- [ ] **Step 4: Register FIBO configs at import of the FIBO VAE**

At the end of the import block in `models/tt_dit/models/vae/vae_bria_fibo.py` (after the existing
`from ...utils.conv3d import (...)` import), add:
```python
from ...pipelines.bria_fibo.fibo_conv3d_configs import register_fibo_conv3d_configs

register_fibo_conv3d_configs()
```

- [ ] **Step 5: Verify the shared conv3d data tables are pristine (only the function body differs)**

Run:
```bash
cd /home/mstojkovic/tt-metal
git diff 2eb991a4100117464d67d5331bd6968470bf9daa -- models/tt_dit/utils/conv3d.py
```
Expected: the ONLY hunk is inside `register_conv3d_configs`; no changes to `_BLOCKINGS` or `_DEFAULT_BLOCKINGS` literals.

- [ ] **Step 6: Verify registration injects the exact keys**

Run:
```bash
cd /home/mstojkovic/tt-metal
python -c "
import models.tt_dit.models.vae.vae_bria_fibo  # triggers registration
from models.tt_dit.utils import conv3d
assert (2, 2, 1024, 1024, (3, 3, 3), 3, 64, 64) in conv3d._BLOCKINGS, 'exact key missing'
assert (64, 1024, (3, 3, 3)) in conv3d._DEFAULT_BLOCKINGS, 'default key missing'
print('FIBO conv blockings registered OK')
"
```
Expected: `FIBO conv blockings registered OK`.

- [ ] **Step 7: Re-run the FIBO VAE test — PCC and per-op timings must match baseline**

Run:
```bash
cd /home/mstojkovic/tt-metal
pytest models/tt_dit/tests/models/bria_fibo/test_vae.py -v
```
Expected: PASS, PCC unchanged from Task 0 baseline, no `[fallback]`/`[NONE]` conv3d warnings for FIBO decode shapes.

- [ ] **Step 8: Commit**

```bash
cd /home/mstojkovic/tt-metal
git add models/tt_dit/utils/conv3d.py \
  models/tt_dit/pipelines/bria_fibo/fibo_conv3d_configs.py \
  models/tt_dit/models/vae/vae_bria_fibo.py
git commit -m "refactor(fibo-pipeline): move FIBO conv3d blockings to fibo_conv3d_configs; register via generic conv3d API"
```

---

## Task 3: Reframe the 12×10 Galaxy grid comment (keep shared)

**Files:**
- Modify: `models/tt_dit/utils/matmul.py` (comment only, above `_BH_GALAXY_MAX_CORE_GRID`)

**Interfaces:**
- Consumes: nothing.
- Produces: no code change; `_BH_GALAXY_MAX_CORE_GRID` stays `(12, 10)`.

- [ ] **Step 1: Rewrite the comment to describe a general capability**

In `models/tt_dit/utils/matmul.py`, replace the comment block currently above
`_BH_GALAXY_MAX_CORE_GRID = (12, 10)` (the one that frames it as a FIBO-specific verification) with:
```python
# Full 12x10 compute grid on the 4x8 Blackhole Galaxy. Previously clamped to 11x10 for a power
# constraint; the full 12x10 grid has since been verified to run cleanly on all 32 devices (no
# fabric/power fault) and is faster than 11x10 for the matmul shapes exercised on this platform.
# 11x10 configs are retained as a fallback.
```
(No change to the constant value.)

- [ ] **Step 2: Confirm the constant is unchanged and value is 12×10**

Run:
```bash
cd /home/mstojkovic/tt-metal
grep -n "_BH_GALAXY_MAX_CORE_GRID = " models/tt_dit/utils/matmul.py
```
Expected: `_BH_GALAXY_MAX_CORE_GRID = (12, 10)`.

- [ ] **Step 3: Import smoke**

Run:
```bash
cd /home/mstojkovic/tt-metal
python -c "import models.tt_dit.utils.matmul; print('matmul import OK')"
```
Expected: `matmul import OK`.

- [ ] **Step 4: Commit**

```bash
cd /home/mstojkovic/tt-metal
git add models/tt_dit/utils/matmul.py
git commit -m "docs(fibo-pipeline): reframe 12x10 Galaxy grid as a general verified capability"
```

---

## Task 4: Verify matmul config registration is fully FIBO-owned (no-op check)

**Files:** none expected (verification only).

**Interfaces:**
- Consumes: `register_matmul_configs` from `models.tt_dit.utils.matmul`, called from `transformer_bria_fibo.py`.

- [ ] **Step 1: Confirm no FIBO-specific config data remains inline in matmul.py**

Run:
```bash
cd /home/mstojkovic/tt-metal
git diff 2eb991a4100117464d67d5331bd6968470bf9daa -- models/tt_dit/utils/matmul.py
```
Expected: the only diff is the comment reframe from Task 3 (no config-dict data added for FIBO).

- [ ] **Step 2: Confirm the FIBO DiT registers its own matmul configs**

Run:
```bash
cd /home/mstojkovic/tt-metal
grep -n "register_matmul_configs" models/tt_dit/models/transformers/transformer_bria_fibo.py
```
Expected: at least one call site (the FIBO config data lives here, not in shared matmul.py).

- [ ] **Step 3: No commit** (verification only; if inline FIBO config data is unexpectedly found in matmul.py, stop and extend the plan).

---

## Task 5: Full-pipeline behavior + perf verification (required tasks)

**Files:** none (verification).

**Interfaces:**
- Consumes: baseline from Task 0.

- [ ] **Step 1: Run the FIBO end-to-end pipeline test**

Run:
```bash
cd /home/mstojkovic/tt-metal
pytest models/tt_dit/tests/models/bria_fibo/test_pipeline.py -v
```
Expected: PASS; generated image matches baseline (same PCC / same visual result).

- [ ] **Step 2: Run the FIBO performance harness and diff against baseline**

Run:
```bash
cd /home/mstojkovic/tt-metal
pytest models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py -v 2>&1 | tee /tmp/claude-4165/-home-mstojkovic-tt-metal/539fe323-7316-4ce2-a265-6797db6d4539/scratchpad/fibo_after_perf.txt
diff /tmp/claude-4165/-home-mstojkovic-tt-metal/539fe323-7316-4ce2-a265-6797db6d4539/scratchpad/fibo_baseline_perf.txt /tmp/claude-4165/-home-mstojkovic-tt-metal/539fe323-7316-4ce2-a265-6797db6d4539/scratchpad/fibo_after_perf.txt || true
```
Expected: PASS; timings within run-to-run noise of the Task 0 baseline (no systematic regression).

- [ ] **Step 3: Confirm shared files are pristine / minimally changed**

Run:
```bash
cd /home/mstojkovic/tt-metal
MB=2eb991a4100117464d67d5331bd6968470bf9daa
echo "vae_wan2_1 (want empty):"; git diff $MB -- models/tt_dit/models/vae/vae_wan2_1.py
echo "conv3d (want only register_conv3d_configs body):"; git diff $MB -- models/tt_dit/utils/conv3d.py
echo "matmul (want only comment):"; git diff $MB -- models/tt_dit/utils/matmul.py
```
Expected: vae empty; conv3d only the function body; matmul only the comment.

- [ ] **Step 4: No commit** (verification only).

---

## Task 6 (OPTIONAL — low value): FIBO-owned encoder parallel config

> **Recommendation:** consider skipping. The shared `EncoderParallelConfig` additions (`cfg_parallel` optional field + `from_tuples`) are additive and backward-compatible; no other caller is affected. Separating them duplicates a library-wide NamedTuple purely for ownership hygiene and risks coupling the general SmolLM3 encoder to FIBO. Do this only if strict "zero FIBO edits in shared config.py" is required.

**Files:**
- Create: `models/tt_dit/pipelines/bria_fibo/fibo_parallel_config.py`
- Modify: `models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py:45,65,134`
- Modify: `models/tt_dit/parallel/config.py` (revert to `$MB`)

**Interfaces:**
- Produces: `FiboEncoderParallelConfig` NamedTuple with `tensor_parallel`, `sequence_parallel`, `cfg_parallel` and `from_tuples(*, tp, sp=None, cfg=None)`.

- [ ] **Step 1: Create the FIBO encoder config**

Create `models/tt_dit/pipelines/bria_fibo/fibo_parallel_config.py`:
```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FIBO-owned encoder parallel config (previously the cfg_parallel/from_tuples additions on the
shared EncoderParallelConfig)."""
from __future__ import annotations

from typing import NamedTuple

from ...parallel.config import ParallelFactor


class FiboEncoderParallelConfig(NamedTuple):
    tensor_parallel: ParallelFactor
    sequence_parallel: ParallelFactor | None = None
    cfg_parallel: ParallelFactor | None = None

    @classmethod
    def from_tuples(
        cls,
        *,
        tp: tuple[int, int],
        sp: tuple[int, int] | None = None,
        cfg: tuple[int, int] | None = None,
    ) -> "FiboEncoderParallelConfig":
        return cls(
            tensor_parallel=ParallelFactor(*tp),
            sequence_parallel=ParallelFactor(*sp) if sp is not None else None,
            cfg_parallel=ParallelFactor(*cfg) if cfg is not None else None,
        )
```

- [ ] **Step 2: Use the FIBO config in the pipeline**

In `models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py`:
- Line 45: drop `EncoderParallelConfig` from the `from ...parallel.config import ...` line, and add
  `from models.tt_dit.pipelines.bria_fibo.fibo_parallel_config import FiboEncoderParallelConfig`.
- Line 65: change the dataclass/field annotation `encoder_parallel_config: EncoderParallelConfig` →
  `encoder_parallel_config: FiboEncoderParallelConfig`.
- Line 134: change `EncoderParallelConfig.from_tuples(` → `FiboEncoderParallelConfig.from_tuples(`.

- [ ] **Step 3: Revert the shared parallel config**

Run:
```bash
cd /home/mstojkovic/tt-metal
git checkout 2eb991a4100117464d67d5331bd6968470bf9daa -- models/tt_dit/parallel/config.py
git diff 2eb991a4100117464d67d5331bd6968470bf9daa -- models/tt_dit/parallel/config.py
```
Expected: empty diff.

- [ ] **Step 4: Import + build smoke (duck-typing check)**

Run:
```bash
cd /home/mstojkovic/tt-metal
python -c "
from models.tt_dit.pipelines.bria_fibo.fibo_parallel_config import FiboEncoderParallelConfig as C
c = C.from_tuples(tp=(8,1), sp=(1,0), cfg=(1,0))
print('tp', c.tensor_parallel, 'sp', c.sequence_parallel, 'cfg', c.cfg_parallel)
import models.tt_dit.pipelines.bria_fibo.pipeline_bria_fibo as p; print('pipeline import OK')
"
```
Expected: prints the factors and `pipeline import OK`.

- [ ] **Step 5: Run FIBO pipeline + encoder tests**

Run:
```bash
cd /home/mstojkovic/tt-metal
pytest models/tt_dit/tests/models/bria_fibo/test_pipeline.py models/tt_dit/tests/encoders/smollm3/test_smollm3.py -v
```
Expected: PASS, output unchanged from baseline.

- [ ] **Step 6: Commit**

```bash
cd /home/mstojkovic/tt-metal
git add models/tt_dit/pipelines/bria_fibo/fibo_parallel_config.py \
  models/tt_dit/pipelines/bria_fibo/pipeline_bria_fibo.py \
  models/tt_dit/parallel/config.py
git commit -m "refactor(fibo-pipeline): FIBO-owned encoder parallel config; revert shared parallel/config.py"
```

---

## Task 7 (OPTIONAL — low value): FIBO-owned sweep configs

> **Recommendation:** consider deferring. `sweep_mm_block_sizes.py` is a dev/profiling tool (no runtime, perf, or output impact) and the FIBO configs are woven through its `DEVICE_CONFIGS`, `SHAPES`, and inline registrations. Cleanly extracting them is disproportionate effort for a tool. Do this only if the sweep file must also be pristine.

**Files:**
- Create: `models/tt_dit/utils/sweep_mm_block_sizes_fibo.py`
- Modify: `models/tt_dit/utils/sweep_mm_block_sizes.py` (move FIBO entries out; import them back if the tool needs them)

**Interfaces:**
- Produces: FIBO device/shape configs importable by the sweep tool.

- [ ] **Step 1: Inventory the FIBO entries in the sweep file**

Run:
```bash
cd /home/mstojkovic/tt-metal
grep -niE 'fibo' models/tt_dit/utils/sweep_mm_block_sizes.py
```
Record the FIBO `DEVICE_CONFIGS` key(s) (e.g. `bh_4x8_fibo`), FIBO `SHAPES` blocks, and FIBO comment/registration sections.

- [ ] **Step 2: Move the FIBO entries to the companion file**

Create `models/tt_dit/utils/sweep_mm_block_sizes_fibo.py` containing the FIBO `DEVICE_CONFIGS`
fragment and FIBO `SHAPES` list, exported as `FIBO_DEVICE_CONFIGS` (dict) and `FIBO_SHAPES` (list).
Copy the exact entries identified in Step 1 verbatim.

- [ ] **Step 3: Wire the companion file back into the tool**

In `models/tt_dit/utils/sweep_mm_block_sizes.py`, after the `DEVICE_CONFIGS = {...}` and
`SHAPES = [...]` definitions, merge in the FIBO entries:
```python
from models.tt_dit.utils.sweep_mm_block_sizes_fibo import FIBO_DEVICE_CONFIGS, FIBO_SHAPES

DEVICE_CONFIGS.update(FIBO_DEVICE_CONFIGS)
SHAPES.extend(FIBO_SHAPES)
```
Remove the original inline FIBO `DEVICE_CONFIGS`/`SHAPES` entries so the base file has no FIBO data.

- [ ] **Step 4: Verify the tool still resolves the FIBO device config**

Run:
```bash
cd /home/mstojkovic/tt-metal
python -c "
from models.tt_dit.utils import sweep_mm_block_sizes as s
assert 'bh_4x8_fibo' in s.DEVICE_CONFIGS, 'FIBO device config missing after merge'
print('sweep FIBO configs merged OK; total device configs:', len(s.DEVICE_CONFIGS))
"
```
Expected: `sweep FIBO configs merged OK` (adjust the asserted key name to match Step 1).

- [ ] **Step 5: Commit**

```bash
cd /home/mstojkovic/tt-metal
git add models/tt_dit/utils/sweep_mm_block_sizes.py models/tt_dit/utils/sweep_mm_block_sizes_fibo.py
git commit -m "refactor(fibo-pipeline): move FIBO sweep configs to sweep_mm_block_sizes_fibo.py"
```

---

## Task 8: Final audit

**Files:** none.

- [ ] **Step 1: Full shared-file diff audit**

Run:
```bash
cd /home/mstojkovic/tt-metal
MB=2eb991a4100117464d67d5331bd6968470bf9daa
for f in models/tt_dit/models/vae/vae_wan2_1.py models/tt_dit/utils/conv3d.py \
         models/tt_dit/utils/matmul.py models/tt_dit/parallel/config.py \
         models/tt_dit/utils/sweep_mm_block_sizes.py; do
  echo "=== $f ==="; git diff $MB -- "$f"
done
```
Expected:
- `vae_wan2_1.py`: empty
- `conv3d.py`: only the `register_conv3d_configs` body
- `matmul.py`: only the comment reframe
- `parallel/config.py`: empty (if Task 6 done) or the additive fields (if skipped)
- `sweep_mm_block_sizes.py`: empty/merge-only (if Task 7 done) or FIBO entries (if skipped)

- [ ] **Step 2: Grep for stray FIBO references in shared files**

Run:
```bash
cd /home/mstojkovic/tt-metal/models/tt_dit
grep -rilE 'fibo' --include='*.py' . | grep -vE 'fibo|bria|smollm3'
```
Expected: only files intentionally kept shared (`utils/matmul.py`, `utils/conv3d.py` comments/examples, and — if Tasks 6/7 skipped — `parallel/config.py`, `utils/sweep_mm_block_sizes.py`). Anything unexpected → investigate.

- [ ] **Step 3: Final behavior confirmation**

Run:
```bash
cd /home/mstojkovic/tt-metal
pytest models/tt_dit/tests/models/bria_fibo/ -v
```
Expected: all FIBO tests PASS, output identical to Task 0 baseline.

- [ ] **Step 4: No commit** (audit only).
