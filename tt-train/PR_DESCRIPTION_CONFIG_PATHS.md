### Ticket
N/A (Infrastructure fix)

### Problem description
Training binaries only worked when run from `tt-metal/tt-train/` — running from elsewhere failed because `main.cpp` resolved config paths relative to the working directory via `std::filesystem::current_path()`. Additionally, all CMakeLists used `CMAKE_SOURCE_DIR` for `CONFIGS_FOLDER`, which resolves to `tt-metal/` instead of `tt-metal/tt-train/` when built from the tt-metal root (the default `build_metal --build-tt-train` flow). This was silently wrong because `main.cpp` never used `CONFIGS_FOLDER`, but the 3-tier workers and mnist_mlp did.

### What's changed
- **19 YAML configs**: `model_config` now uses `${TT_METAL_RUNTIME_ROOT}/tt-train/...` instead of relative paths
- **3 CMakeLists.txt**: replaced `CMAKE_SOURCE_DIR` with `TT_TRAIN_SOURCE_DIR` for `CONFIGS_FOLDER` definitions; added missing `CONFIGS_FOLDER` for the `llama_inference` target
- **5 C++ files**: added shared `expand_config_path()` utility to `nano_gpt` and `mnist_mlp` that expands `${TT_METAL_RUNTIME_ROOT}` at runtime, with a compile-time fallback inferred from `CONFIGS_FOLDER` when the env var is not set

### Testing
- [ ] Binary runs correctly from any directory
- [ ] Works with and without `TT_METAL_RUNTIME_ROOT` set

### Checklist
- [ ] [![All post-commit tests](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml/badge.svg?branch=mdragula/config-fixed-paths)](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml?query=branch:mdragula/config-fixed-paths)
- [ ] [![Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml/badge.svg?branch=mdragula/config-fixed-paths)](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml?query=branch:mdragula/config-fixed-paths)
- [ ] [![cpp-unit-tests](https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml/badge.svg?branch=mdragula/config-fixed-paths)](https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml?query=branch:mdragula/config-fixed-paths)
- [ ] New/Existing tests provide coverage for changes
