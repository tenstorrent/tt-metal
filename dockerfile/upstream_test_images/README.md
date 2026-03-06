# Upstream Testing

Please refer to https://tenstorrent.atlassian.net/wiki/spaces/MI6/pages/1112015146/External+How+to+run+Metalium+upstream+Docker+tests

for background, instructions, further reading, and policies about these upstream tests.

---

## Container Restructuring Changelog

This section documents the changes made to consolidate and optimize the upstream test container images.

### Summary

The upstream test images were restructured to reduce count (10 -> 2), shrink size (~8-10 GB -> ~2-3 GB per image), and make them independently consumable on any system with Tenstorrent hardware. The images require no network access at runtime -- only `/dev/tenstorrent` device access and hugepages.

### Running the images

```bash
# Pull the consolidated upstream test image
docker pull ghcr.io/tenstorrent/tt-metal/upstream-tests:latest

# Run for a specific hardware topology (passed as argument, not baked in)
docker run --network none \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  --device /dev/tenstorrent \
  ghcr.io/tenstorrent/tt-metal/upstream-tests:latest blackhole

# Run with model weights mounted (for topologies that run Llama demos)
docker run --network none \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  --device /dev/tenstorrent \
  -v /path/to/Llama-3.1-8B-Instruct:/path/to/Llama-3.1-8B-Instruct:ro \
  -e HF_MODEL=/path/to/Llama-3.1-8B-Instruct \
  -e TT_CACHE_PATH=/path/to/Llama-3.1-8B-Instruct \
  ghcr.io/tenstorrent/tt-metal/upstream-tests:latest blackhole_llmbox

# Run a specific test suite only
docker run --network none \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  --device /dev/tenstorrent \
  ghcr.io/tenstorrent/tt-metal/upstream-tests:latest blackhole \
  --test-suite test_suite_bh_single_pcie_metal_unit_tests

# Run profiler tests
docker run --network none \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  --device /dev/tenstorrent \
  ghcr.io/tenstorrent/tt-metal/upstream-profiler-tests:latest blackhole
```

### Supported hardware topologies

| Topology | Description |
|---|---|
| `blackhole` | Single BH PCIe card (P100, P150) |
| `blackhole_llmbox` | BH LLMBox (4-card) |
| `blackhole_deskbox` | BH DeskBox (2-card) |
| `blackhole_loudbox` | BH LoudBox (8-card) |
| `blackhole_p300` | BH P300 (2-card) |
| `blackhole_qb_ge` | BH QB-GE (4-card) |
| `blackhole_glx` | BH Galaxy (32-card 6U) |
| `wh_6u` | WH Galaxy 6U |

---

## File-by-file change list

### `dockerfile/Dockerfile`

- **New `runtime` stage** added between `dev` and `release`. Branches from `base` (not `dev`), containing only: system runtime deps, Python, uv, a relocatable venv, pip.conf with PyTorch CPU index, openssh-server, and sudo. This is the base image for upstream test containers.
- **Layer reordering in `ci-build`**: moved pinned binary downloads (ccache, mold) before source builds (doxygen, ClangBuildAnalyzer), before apt packages, before pip packages. Optimized for Docker layer cache hit rates.
- **BuildKit cache mounts** added to all `apt-get` and `uv pip install` RUN instructions across `ci-build`, `ci-test`, `dev`, `runtime`, and `release-models` stages. Uses `--mount=type=cache,target=/var/cache/apt` and `--mount=type=cache,target=/root/.cache/uv`.
- **Single-pass pip install** in `ci-test`: removed the separate `requirements-docs.txt` install since `requirements-dev.txt` now composes all tiers via `-r` includes. Eliminated redundant dependency resolution.
- **Added COPY instructions** for the new tiered requirement files (`requirements-core.txt`, `requirements-dev-tools.txt`, `requirements-models.txt`, `requirements-notebooks.txt`).

### `dockerfile/upstream_test_images/Dockerfile.template`

- **Base image changed** from `ubuntu-22.04-dev-amd64` (~8-10 GB, full build+dev toolchain) to `ubuntu-22.04-runtime-amd64` (~2-3 GB, minimal runtime only).
- **Removed `git clone`** of the entire tt-metal repository. Previously cloned the full repo with all submodules inside each image.
- **Replaced with selective COPY** of only what tests actually need: build artifacts (`_tt-metal/build/`, `_tt-metal/runtime/`), the wheel, test scripts (`_test_scripts/`, `_test_scripts_extra/`), test directories (`_tests/`), model test directories (`_model_tests/`), and cabling descriptor configs (`_extra_sources/`).
- **User creation preserved** since the `runtime` base (unlike `dev`) does not include the user.
- **HW_TOPOLOGY is now a runtime parameter** via CMD, not baked into the image at build time via envsubst. The same image serves all hardware topologies.

### `.github/workflows/upstream-tests.yaml`

- **Collapsed 10 image env vars** (`WH_6U_IMAGE_NAME`, `BLACKHOLE_IMAGE_NAME`, `BLACKHOLE_LLMBOX_IMAGE_NAME`, etc.) into 2: `UPSTREAM_IMAGE_NAME` and `UPSTREAM_PROFILER_IMAGE_NAME`.
- **Build matrix reduced** from 10 entries to 2 (vanilla + profiler).
- **Added "Prepare test scripts for build context" step** that selectively copies test scripts, test directories, model directories, and cabling descriptor configs from the checkout into staging directories for the Docker build context.
- **All test jobs updated** to pass the hw-topology as a docker run argument instead of using a per-topology image tag. For example: `docker run ... upstream-tests:latest blackhole_llmbox`.
- **`test-bh-multicard-image` simplified** -- the matrix now carries `hw-topology` directly instead of per-topology image names.
- **`calculate-to-publish` simplified** -- since there are only 2 images, the logic publishes both or neither based on aggregate test results.
- **All original test functionality preserved**: same runner labels, same timeout minutes, same environment variables, same docker run flags (`--network none`, `--device /dev/tenstorrent`, hugepages mount, model weight mounts).

### `.github/workflows/job_configs/upstream-tests.json`

- All topology entries updated to reference `upstream-image-tag` and `upstream-profiler-image-tag` instead of per-topology image tags (`bh-image-tag`, `bh-llmbox-image-tag`, etc.).

### `.github/workflows/build-docker-artifact.yaml`

- **Added `runtime-tag` output** with description, hash computation, existence check, build step (targeting the `runtime` stage of `dockerfile/Dockerfile`), and latest-tag push.
- **Updated build-ubuntu-images `if` condition** to also trigger when `runtime-exists != 'true'`.
- All original outputs preserved: `ci-build-tag`, `ci-test-tag`, `dev-tag`, `basic-dev-tag`, `basic-ttnn-runtime-tag`, `manylinux-tag`.

### `.github/workflows/build-all-docker-images.yaml`

- **Added `ubuntu-2204-runtime-tag` and `ubuntu-2404-runtime-tag` outputs** with corresponding tag computation and build steps for both Ubuntu 22.04 and 24.04.
- All original outputs preserved.

### `dockerfile/Dockerfile.evaluation`

- **Refactored into multi-stage build**: `builder` stage compiles tt-metal from source and produces a wheel; `runtime` stage starts from a clean `ubuntu:24.04`, installs only Python deps and system runtime deps, then copies built artifacts and installs the wheel.
- **BuildKit cache mounts** added to all apt-get and uv pip install instructions in both stages.
- **Result**: the final image no longer contains source code, build tools (ccache), or intermediate compilation artifacts.

### `dockerfile/Dockerfile.basic-dev`

- **BuildKit cache mounts** added to both apt-get instructions (base stage and basic-ttnn-runtime stage).

### `dockerfile/Dockerfile.manylinux`

- **BuildKit cache mount** added to the dnf install instruction.

### `tt_metal/python_env/requirements-dev.txt`

- **Rewritten as a composition file** that includes all tiers via `-r`:
  - `-r requirements-core.txt`
  - `-r requirements-dev-tools.txt`
  - `-r requirements-models.txt`
  - `-r requirements-notebooks.txt`
  - `-r ../../docs/requirements-docs.txt`
  - `-r ../../tests/sweep_framework/requirements-sweeps.txt`
- Installing `requirements-dev.txt` still gives you everything. The split only lets lighter images install just the tiers they need.

### `tt_metal/python_env/requirements-core.txt` (new)

- Minimal test dependencies: pytest and plugins, numpy, networkx, pyyaml, click, tqdm, tabulate, pydantic, loguru, tt-smi, torch (CPU), torchvision.
- This is what the upstream `runtime` image would install if it needed Python test packages.

### `tt_metal/python_env/requirements-dev-tools.txt` (new)

- Linters, formatters, build tools: pre-commit, black, clang-format, build, twine, yamllint, mypy, platformdirs.

### `tt_metal/python_env/requirements-models.txt` (new)

- Model-specific dependencies: transformers, huggingface-hub, diffusers, accelerate, timm, librosa, fiftyone, medpy, mmcv, datasets, evaluate, bert-score, and other packages needed only when running model demos/benchmarks.

### `tt_metal/python_env/requirements-notebooks.txt` (new)

- Notebook and visualization: jupyterlab, ipywidgets, bokeh, dash, plotly, pandas, seaborn.
