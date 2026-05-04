# GPT-OSS: Mixture of Experts Language Model

Inference implementation for GPT-OSS models on Tenstorrent **Wormhole** and **Blackhole** hardware.

**Model Source**: [GPT-OSS on HuggingFace](https://huggingface.co/gpt-oss) (custom MoE architecture)

**Target Hardware**:
- **LoudBox**: Single Wormhole device (1×8 configuration)
- **Galaxy**: **Wormhole-only** multi-device mesh (4×8, 32 devices). This is **not** BH T3K (Blackhole); do not conflate the names.
- **BH T3K** (Blackhole multi-chip): supported with **4** or **8** devices. Default mesh shape is **1×4** when `get_num_devices()==4` and **1×8** when `get_num_devices()==8` (override with `GPT_OSS_MESH_SHAPE`, e.g. `2,2` on 4-chip or `2,4` on 8-chip, so that rows×cols equals device count). Device names from weights/trace config include **P150x4** and **P150x8**. From the repo root with `python_env` active, inspect the host mesh with:
  `python3 -c "import ttnn; print('num_devices', ttnn.get_num_devices(), 'arch', ttnn.get_arch_name()); d=ttnn._ttnn.multi_device.SystemMeshDescriptor(); print('system_mesh', tuple(d.shape()))"`

**Current Status**: This model is under active development.
- ✅ Supported: Prefill up to sequence length 128, batch size 1, total sequence length 4096
- 🚧 In Progress: Extended sequence lengths, larger batch sizes

## Quick Start

```bash
# Activate the repo venv that was created by the build (so pytest and python share the same stack).
# If pip says "Defaulting to user installation because normal site-packages is not writeable",
# the venv has no pip; use uv pip instead (uv ships in python_env/bin):
#   source python_env/bin/activate
#   uv pip install -r models/demos/gpt_oss/requirements.txt

source python_env/bin/activate
uv pip install -r models/demos/gpt_oss/requirements.txt

# Set model path using HF_MODEL environment variable
export HF_MODEL="/mnt/MLPerf/tt_dnn-models/openai/gpt-oss-20b"

# Run text generation demo on Galaxy (4×8 mesh) from the repo root
# (paths inside text_demo.py are repo-root relative)
pytest models/demos/gpt_oss/demo/text_demo.py -k "4x8 and prefill_128"
```

### BH T3K (Blackhole 4- or 8-chip) + GPT-OSS-120B

1. Weights on disk; directory basename must be `gpt-oss-120b` (or `gpt-oss-20b`), e.g.
   `export HF_MODEL=/path/to/gpt-oss-120b`
2. Optional: set `GPT_OSS_MESH_SHAPE` if your fabric layout is not the default **1×4** (4 devices) or **1×8** (8 devices). Inspect the host with the one-liner under **Target Hardware** above.
3. Smoke the prefill-128 demo **from the repo root** (paths in `text_demo.py` are repo-relative):

**4-chip BH T3K**

```bash
source python_env/bin/activate
export HF_MODEL=/path/to/gpt-oss-120b
timeout 2400 pytest models/demos/gpt_oss/demo/text_demo.py \
    -k "mesh_bh_t3k_4 and prefill_128" -x
```

**8-chip BH T3K** (uses the same `(1,8)` mesh row as Wormhole LoudBox, but on Blackhole with 8 devices)

```bash
source python_env/bin/activate
export HF_MODEL=/path/to/gpt-oss-120b
timeout 2400 pytest models/demos/gpt_oss/demo/text_demo.py \
    -k "mesh_1x8 and prefill_128" -x
```

The demo gate `run_for_gpt_oss_text_demo_hw` allows **Wormhole B0 with 8 or 32 devices** or **Blackhole BH T3K with 4 or 8 devices**.

**pytest and `GPT_OSS_PARALLEL_LAYER_LOAD`:** Under pytest, parallel layer load is **forced off** by default so a `pytest-timeout` interrupt does not leave worker threads still packing weights (you would see log lines after `Timeout`). To allow threads inside pytest, set `GPT_OSS_PARALLEL_LAYER_LOAD_IN_PYTEST=1`. The `text_demo` test also uses a long per-test timeout with `func_only=True` so cold HF + cache work does not hit the repo-wide 300s limit.

## Configuration

### Model Selection
```bash
# GPT-OSS-20B (faster, recommended for development)
export HF_MODEL="/mnt/MLPerf/tt_dnn-models/openai/gpt-oss-20b"

# GPT-OSS-120B (higher quality, requires more memory)
export HF_MODEL="/mnt/MLPerf/tt_dnn-models/openai/gpt-oss-120b"
```

## Testing

```bash
# Run all tests
pytest models/demos/gpt_oss/tests/unit/ -v

# Run specific test files
pytest models/demos/gpt_oss/tests/unit/test_modules.py -v     # Core components
pytest models/demos/gpt_oss/tests/unit/test_model.py -v       # Full model accuracy
```

### Test Files Overview

| File | Purpose | Tests |
|------|---------|-------|
| **`test_modules.py`** | Core MoE components | • Attention component<br>• RMSNorm<br>• TopK router<br>• Experts<br>• Full MLP pipeline<br>• Complete decoder layer |
| **`test_model.py`** | Full model integration | • End-to-end accuracy<br>• Teacher forcing<br>• Reference model comparison |
