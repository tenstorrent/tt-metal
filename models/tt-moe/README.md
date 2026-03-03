# Unified MoE Block Implementation

This directory contains a unified, configurable Mixture of Experts (MoE) implementation that can support multiple architectures through JSON configuration files.

## Directory Structure

```
tt-moe/
├── configs/
│   ├── deepseek_v3.json   # DeepSeek-V3 configuration
│   ├── gpt_oss.json       # GPT-OSS configuration
│   └── glm4.json          # GLM-4 MoE configuration
├── utils/
│   ├── __init__.py
│   └── load_config.py     # load_moe_config(), get_moe_block_config()
├── __init__.py
└── README.md
```

## How to use the JSON config

### 1. Load by file path (from repo root)

```python
import json
from pathlib import Path

config_path = Path("models/tt-moe/configs/glm4.json")
with open(config_path) as f:
    config = json.load(f)

moe_block = config["moe_block"]
hidden_size = moe_block["model_params"]["hidden_size"]
num_experts = moe_block["model_params"]["num_experts"]
```

### 2. Use the tt-moe loader

Add the `utils` folder to your path and use the loader (preset name `"glm4"` resolves to `configs/glm4.json` next to the loader):

```python
import sys
sys.path.insert(0, "models/tt-moe/utils")
from load_config import load_moe_config, get_moe_block_config

# By path (from repo root)
full = load_moe_config("models/tt-moe/configs/glm4.json")
moe_block = get_moe_block_config("models/tt-moe/configs/glm4.json")

# By preset name (loads tt-moe/configs/glm4.json relative to the loader)
moe_block = get_moe_block_config("glm4")
params = moe_block["model_params"]  # hidden_size, num_experts, etc.
```

### 3. Use in tests (e.g. GLM Flash)

You can validate the loaded model against the JSON or build cache/config from it:

```python
# In test_glm_flash.py or a similar test
from pathlib import Path
import json

def get_glm4_moe_config():
    path = Path(__file__).resolve().parents[2] / "tt-moe" / "configs" / "glm4.json"
    with open(path) as f:
        return json.load(f)["moe_block"]

# Then use get_glm4_moe_config()["model_params"] for hidden_size, num_experts, etc.
# or to assert model.config matches (e.g. model.config.hidden_size == cfg["model_params"]["hidden_size"])
```

### 4. When the full TT-MoE stack is available (PR #37920)

After the unified MoEBlock lands, you will use the JSON to construct the block:

```python
from tt_moe import MoEBlock

moe = MoEBlock("configs/glm4.json", mesh_device, ccl)
moe.load_weights(state_dict)
output = moe.forward(x, mode="prefill")
```

---

## GLM-4 Implementation

The unified MoE block can be configured for GLM-4 (e.g. THUDM/GLM-4-100B-A10B, zai-org/GLM-4.7-Flash) via `configs/glm4.json`.

### Configuration Mapping

| GLM-4 (HuggingFace / tt_symbiote) | TT-MoE config |
|-----------------------------------|----------------|
| `n_routed_experts` 128            | `model_params.num_experts` |
| `num_experts_per_tok` 8           | `model_params.num_experts_per_tok` |
| `hidden_size` 4096                | `model_params.hidden_size` |
| `moe_intermediate_size` 1408      | `model_params.moe_intermediate_size` |
| `n_shared_experts` 1              | `model_params.n_shared_experts` |
| `n_group` / `topk_group`           | `model_params.n_group`, `topk_group` |
| `norm_topk_prob`                  | `router.norm_topk_prob` |
| `routed_scaling_factor`           | `router.routed_scaling_factor` |
| Router `e_score_correction_bias`   | `router.score_correction_bias` |
| Shared expert (SiLU MLP)           | `experts.shared`, `activation: swiglu` |

### Architecture Summary

- **Router**: Grouped top-k (same style as DeepSeek): score correction bias, optional top-k probability normalization, routed scaling factor.
- **Routed experts**: 128 experts, 8 per token; SiLU (SwiGLU) activation; configurable group/topk_group for group-based selection.
- **Shared expert**: Single shared MLP with intermediate size `moe_intermediate_size * n_shared_experts` (1408 for default), added to MoE output.
- **Tensor parallelism**: Supported on configurable axis (e.g. axis 1).
- **Expert parallelism**: Supported on configurable axis (e.g. axis 0).

### Usage (when MoEBlock is available)

```python
from tt_moe import MoEBlock

# Initialize with GLM-4 configuration
moe = MoEBlock("configs/glm4.json", mesh_device, ccl)
moe.load_weights(state_dict)
output = moe.forward(x, mode="prefill")
```

## References

- TT-MoE PR: [TT-MoE: Unified Configurable Mixture of Experts Infrastructure #37920](https://github.com/tenstorrent/tt-metal/pull/37920)
- GLM-4 MoE implementation in tt_symbiote: `models/experimental/tt_symbiote/modules/moe.py` (`Glm4MoeConfig`, `Glm4MoeMoE`, `TTNNGlm4MoeMoE`, `TTNNMoE`).
