# Merge Notes: `samt/standalone_mediajan26`

**Branch**: `samt/standalone_mediajan26` (1 commit ahead of `main`)
**Source**: Cherry-picked from `samt/standalone_fork_jan2026` (13 commits, Dec 2025)
**Base**: `main` at `74b1f48c3d` (Feb 19, 2026)
**Commit**: `bbdaa64e2c` — "Add standalone SDXL HTTP server with worker/runner architecture"

---

## What this branch is

A clean branch off current `main` with a single squashed commit that brings in the standalone SDXL HTTP server from the old fork (`samt/standalone_fork_jan2026`). The old fork was based on `main` from Dec 10, 2025 — roughly 1,869 commits behind. Instead of merging that stale branch, we cherry-picked only the standalone server changes and adapted them to current `main`.

---

## What was brought in

### New files (12)

| File | What it does |
|------|-------------|
| `sdxl_server.py` | FastAPI HTTP server — `/image/generations`, `/health`, `/metrics` |
| `sdxl_worker.py` | Multi-process device worker with overlapped init |
| `sdxl_runner.py` | Device init, model loading, warmup, inference loop |
| `sdxl_config.py` | Dataclass config for server/device/pipeline |
| `launch_sdxl_server.sh` | Start script |
| `kill_sdxl_server.sh` | Stop script (graceful + forced) |
| `utils/cache_utils.py` | Cache validation and management |
| `utils/image_utils.py` | PIL/base64/tensor image conversions |
| `utils/logger.py` | Structured logger setup |
| `utils/validation_utils.py` | MSE/SSIM image comparison |
| `image_test.py` | CLI test client for hitting the server |
| `SDXL_SERVER_README.md` | Server docs |

### Modified files (4)

| File | Changes |
|------|---------|
| `tt_sdxl_pipeline.py` | `release_traces()` public method (safe double-release), `start_latents` param for img2img, `torch.stack` → `torch.cat` for embeddings, `from_torch` with mesh_mapper for in-place tensor updates, safe `__del__` |
| `tt_unet.py` | Added `batch_size=1` param to `forward()`, used in `ttnn.reshape` |
| `test_common.py` | `tqdm` progress bar, `batch_size=B` to unet calls, text_embeds slicing for non-CFG-parallel, `ttnn.synchronize_device` after trace execution, robust None handling for `prompt_2`/`negative_prompt_2` |
| `test_module_tt_unet.py` | `batch_size=B` to 2 unet forward calls |

---

## Merge conflicts / adaptations from main

Main had ~1,869 commits since the fork. These are the conflicts that mattered:

### `tt_unet.py` — Structural conflict

**Main changed**: `fbd10589f3` (SDXL model config refactor) rewrote groupnorm handling to use `model_config.get_groupnorm_params()`. The fork's version had inline groupnorm config.

**Resolution**: Kept main's `get_groupnorm_params` infrastructure. Only layered in the `batch_size` parameter addition, which is a clean 2-line change.

### `tt_sdxl_pipeline.py` — Structural conflict

**Main changed**: `48b9819ded` (TP=2 prompt batch hotfix) and `fbd10589f3` (config refactor) added `VAEModelOptimisations`, `determinate_min_batch_size`, `cpu_device="cpu"`, and restructured tensor allocation paths.

**Resolution**: Kept all of main's infrastructure intact. The fork's changes were layered on top:
- `release_traces()` replaces `__release_trace()` — now public, with double-release guard
- `start_latents` param added to `__generate_input_tensors` alongside main's existing params
- Embedding concat changed from `torch.stack` to `torch.cat` (fixes batch dimension handling)
- In-place tensor update path uses `from_torch` with `mesh_mapper` instead of raw allocation
- `all_prompt_embeds_torch` added to `__allocate_device_tensors` signature for subsequent-call updates

### `test_common.py` — Semantic conflict

**Main changed**: `8e47ec59db` (fix clip encoders), `48b9819ded` (TP=2 batch hotfix), `6901c4f80c` (auto slicing VAE), `614f49d208` (tt_dit move) added `normalize_prompt_for_text_encoder`, `determine_data_parallel`, updated CLIP encoder call signatures, and added significant parallelism infrastructure.

**Resolution**: Fork's changes were adapted to work with main's new infrastructure rather than the old code:
- `prompt_2` / `negative_prompt_2` None-handling works with main's `normalize_prompt_for_text_encoder` flow
- `text_embeds` slicing and reshaping added for non-CFG-parallel mode
- `ttnn.synchronize_device` after `ttnn.execute_trace` — this was a real bug fix from the fork (prevents the "jump to step 26" issue where the loop runs ahead of trace execution)
- `tqdm` progress bar integrated cleanly

### `sdxl_config.py` — Trivial conflict

**Resolution**: `l1_small_size` bumped from 30500 → 30800 to match main's current value.

---

## What main gained while the fork was stale

The fork was based on Dec 10, 2025 main. Here's what landed on main in the ~1,869 commits since.

### Nov 1 – Dec 15

**SDXL** — Major performance push. L1 attention, transformer sharding, bfp8 matmul, 40-core conv. New combined base+refiner pipeline, inpainting, img2img via refiner. UNet optimizations and fused bias in linear.

**Wan** — Initial bringup. CI tests added, first optimizations applied via TT-DiT, image-to-video first implementation, BH config.

**SD3/DiT** — Motif + TT-DiT refactor. Old SD3.5 deleted. Distributed RMSNorm hang fixed. Dashboard metrics added.

**Flux** — Test fixes only.

### Dec 15 – Present

**SDXL** — TP=2 on Galaxy, model config refactor, refiner matmul optimizations across 40 cores, VAE auto-slicing, clip encoder fixes, img2img accuracy and seed fixes, bfloat16 rounding error fix.

**Wan** — Image-to-video fixed and encoder implemented, pipeline aligned to reference, VAE latency reduced, step-independent computation caching, distributed LayerNorm, ring attention optimization.

**SD3/DiT** — Moved out of experimental. Module base class migration. Distributed LayerNorm fixes (Welford reciprocals, batch gamma/beta). Module cache size reduction.

**Flux** — BH load-balancing config, ring attention optimization. Still riding shared DiT improvements.

---

## What was NOT brought in

- **Merge commit** `32429351f8` — "Merge main into samt/standalone_sdxl" — skipped (it's a merge commit, would cause cherry-pick issues, and brings in 60+ unrelated files)
- **Core tt-metal changes** — `ttnn/cpp/.../compute_common.hpp` change from `bd0fe2ddc4` was unrelated to the server
- **All model files changed only by the merge commit** — these were main-into-fork changes, not server work

---

## Known risks / things to watch

1. **`torch.stack` → `torch.cat` in embeddings**: The fork changed this in `tt_sdxl_pipeline.py` to fix batch dimension issues. If main's tests rely on the old stacking behavior, this could break things. The change affects `__encode_prompts` — verify that prompt encoding still produces correct shapes for all modes (single prompt, batch, CFG parallel, non-CFG parallel).

2. **In-place tensor updates**: The `__allocate_device_tensors` method now does `copy_host_to_device_tensor` for prompt_embeds and text_embeds on subsequent calls (not just time_ids). This is needed for the server's multi-inference loop but changes behavior for any code that calls `__allocate_device_tensors` more than once.

3. **`ttnn.synchronize_device` after trace execute**: Added inside the denoising loop in `test_common.py`. This is a correctness fix but adds a sync point per step — may affect throughput benchmarks.

4. **`batch_size` param in unet forward**: Callers that don't pass `batch_size` get the default `1`. This should be fine but any new call sites on main need to be checked.

5. **The shell scripts** (`launch_sdxl_server.sh`, `kill_sdxl_server.sh`) reference paths and env vars that may need updating if the deployment environment changed since Dec 2025.

---

## How to validate

```bash
# Start the server
./launch_sdxl_server.sh

# Test it
python image_test.py --prompt "a cat sitting on a windowsill"

# Stop it
./kill_sdxl_server.sh
```

If pipeline tests exist on main, also run:
```bash
pytest models/experimental/stable_diffusion_xl_base/tests/ -v
```

---

## Merge from main (March 21, 2026)

Merged 1,003 commits from `main` into `samt/standalone_mediajan26` (commit `3aa01d431e1`).

### Conflicts resolved

**1. `tt_metal/python_env/requirements-dev.txt` (trivial)**
- Branch added `fastapi`, `uvicorn`; main added `peft`. Kept all three.

**2. `models/demos/stable_diffusion_xl_base/tt/tt_sdxl_pipeline.py` (significant)**
- Main renamed `models/experimental/` → `models/demos/` and added: LoRA support, `image_resolution` config, `get_latents_shape()`, 512x512 resolution support, `torch.manual_seed()` seeding, docstrings, throttle control.
- Resolution: took main's version as base, layered branch additions:
  - `start_latents` param for img2img (branch) + main's `torch.manual_seed()` and `self.pipeline_config.image_resolution`
  - Branch's `__allocate_device_tensors` with mesh_mapper and in-place update path
  - Branch's `release_traces()` (public, with safety tracking)
  - Branch's safe `__del__`
  - Branch's trace re-capture workaround in `generate_images`

### Known divergence: `torch.cat` vs `torch.stack` in `encode_prompts`

The branch uses `torch.cat` (dim=0) instead of main's `torch.stack` (dim=0) + cat (dim=1) for combining negative/positive prompt embeddings. This is intentional for standalone server batching but diverges from main's pattern. **Must be reconciled before PR to main.**

Affected lines in `models/demos/stable_diffusion_xl_base/tt/tt_sdxl_pipeline.py`:
```python
# Branch (current):
all_prompt_embeds_torch = torch.cat(
    [torch.cat(negative_prompt_embeds, dim=0), torch.cat(prompt_embeds, dim=0)], dim=0
)
# Main (upstream):
all_prompt_embeds_torch = torch.cat(
    [torch.stack(negative_prompt_embeds, dim=0), torch.stack(prompt_embeds, dim=0)], dim=1
)
```

### Additional fix: `scripts/convert_notebooks_to_python.py`

Added `tmp_file.exists()` check to handle race condition where pre-commit's stash mechanism causes nbconvert's output file to be missing. This is a pre-existing bug in main's hook script exposed by merges with many staged files.

---

## Branch lineage

```
main (74b1f48c3d, Feb 19 2026)
└── samt/standalone_mediajan26
    ├── bbdaa64e2c (+1 commit, cherry-picked from fork)
    ├── ... (+6 more commits: SD3.5, WAN, ComfyUI bridge, device detection)
    └── 3aa01d431e1 (merge main, +1003 commits from main, March 21 2026)
        ↑ cherry-picked from
    samt/standalone_fork_jan2026 (65928f5c24, 13 commits ahead of Dec 10 2025 main)
```
