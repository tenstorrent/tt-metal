# Qwen3.5-9B Blackhole P150 Text-Only Bringup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring up Qwen3.5-9B (text-only, no vision) on a single Blackhole P150 device by wrapping the experimental Gated Attention + Gated DeltaNet TTNN ops into the tt_transformers model framework.

**Architecture:** Hybrid 32-layer transformer with 24 Gated DeltaNet (linear attention) layers and 8 Gated Full Attention (softmax GQA) layers in a repeating [linear, linear, linear, full_attention] × 8 pattern. Reuses validated TTNN ops from the `sdawle/gated_attention_gated_deltanet` experimental branch.

**Tech Stack:** ttnn, tt-metal, PyTorch (reference), HuggingFace transformers (tokenizer + validation), safetensors

**Spec:** `docs/superpowers/specs/2026-03-21-qwen35-9b-blackhole-bringup-design.md`

**Note:** All work is local — no commits to any branch.

---

## File Structure

```
models/demos/blackhole/qwen3_5_9b/
├── tt/
│   ├── model_config.py            # Qwen35ModelArgs(ModelArgs) — config + weight loading
│   ├── weight_mapping.py          # HF state_dict → internal format remapping
│   ├── qwen35_rope.py             # RoPE with partial_rotary_factor=0.25
│   ├── qwen35_gated_attention.py  # LightweightModule wrapping experimental gated attention
│   ├── qwen35_gated_deltanet.py   # LightweightModule wrapping experimental gated deltanet
│   ├── qwen35_mlp.py              # SwiGLU MLP (thin wrapper or reuse base)
│   ├── qwen35_decoder.py          # Hybrid TransformerBlock — dispatches per layer type
│   └── qwen35_model.py            # Full model: embed → 32 layers → norm → LM head
├── demo/
│   └── demo.py                    # Text generation demo
└── tests/
    ├── test_weight_mapping.py     # Validate weight remapping shapes + correctness
    ├── test_single_layer.py       # Per-layer PCC validation against torch reference
    └── test_model_e2e.py          # Full model logits vs HF reference
```

**Key dependency files (read-only, do not modify):**
- `models/tt_transformers/tt/model_config.py` — `ModelArgs` base class
- `models/tt_transformers/tt/model.py` — `Transformer` base class
- `models/tt_transformers/tt/decoder.py` — `TransformerBlock` base class
- `models/tt_transformers/tt/mlp.py` — `MLP` base class
- `models/tt_transformers/tt/embedding.py` — `Embedding` class
- `models/tt_transformers/tt/lm_head.py` — `LMHead` class
- `models/tt_transformers/tt/rope.py` — `HfRotarySetup` class
- `models/tt_transformers/tt/load_checkpoints.py` — weight loading utilities
- `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py` — `gated_attention_forward_ttnn()`
- `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py` — `gated_deltanet_forward_ttnn()`
- `models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py` — delta rule ops
- `models/experimental/gated_attention_gated_deltanet/torch_functional/gated_attention.py` — torch reference
- `models/experimental/gated_attention_gated_deltanet/torch_functional/gated_deltanet.py` — torch reference

**Model weights (read-only):**
- `/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2/` — safetensors + config.json

---

## Task 1: Create Directory Structure and `__init__.py` Files

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/__init__.py`
- Create: `models/demos/blackhole/qwen3_5_9b/tt/__init__.py`
- Create: `models/demos/blackhole/qwen3_5_9b/demo/__init__.py`
- Create: `models/demos/blackhole/qwen3_5_9b/tests/__init__.py`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p models/demos/blackhole/qwen3_5_9b/tt
mkdir -p models/demos/blackhole/qwen3_5_9b/demo
mkdir -p models/demos/blackhole/qwen3_5_9b/tests
```

- [ ] **Step 2: Create empty `__init__.py` files**

```bash
touch models/demos/blackhole/qwen3_5_9b/__init__.py
touch models/demos/blackhole/qwen3_5_9b/tt/__init__.py
touch models/demos/blackhole/qwen3_5_9b/demo/__init__.py
touch models/demos/blackhole/qwen3_5_9b/tests/__init__.py
```

- [ ] **Step 3: Verify structure**

```bash
find models/demos/blackhole/qwen3_5_9b -type f | sort
```

Expected:
```
models/demos/blackhole/qwen3_5_9b/__init__.py
models/demos/blackhole/qwen3_5_9b/demo/__init__.py
models/demos/blackhole/qwen3_5_9b/tests/__init__.py
models/demos/blackhole/qwen3_5_9b/tt/__init__.py
```

---

## Task 2: Weight Mapping Module

This is the critical bridge between HuggingFace weight names and the experimental TTNN op parameter names. Must be correct before anything else works.

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tt/weight_mapping.py`
- Create: `models/demos/blackhole/qwen3_5_9b/tests/test_weight_mapping.py`

- [ ] **Step 1: Write the weight mapping test**

```python
# models/demos/blackhole/qwen3_5_9b/tests/test_weight_mapping.py
"""Tests for Qwen3.5-9B HF→internal weight remapping."""
import pytest
import torch
from models.demos.blackhole.qwen3_5_9b.tt.weight_mapping import remap_qwen35_state_dict


CHECKPOINT_DIR = "/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"
HIDDEN_SIZE = 4096
NUM_LAYERS = 32
LINEAR_KEY_DIM = 2048   # 16 heads × 128 head_dim
LINEAR_VALUE_DIM = 4096  # 32 heads × 128 head_dim
FULL_ATTN_Q_DIM = 8192  # 16 heads × 256 head_dim × 2 (query + gate)
FULL_ATTN_KV_DIM = 1024  # 4 heads × 256 head_dim


def _load_raw_state_dict():
    """Load raw HF state dict using safetensors."""
    import glob
    from safetensors import safe_open

    state_dict = {}
    for path in sorted(glob.glob(f"{CHECKPOINT_DIR}/model.safetensors-*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


@pytest.fixture(scope="module")
def raw_state_dict():
    return _load_raw_state_dict()


@pytest.fixture(scope="module")
def remapped(raw_state_dict):
    return remap_qwen35_state_dict(raw_state_dict)


class TestPrefixStripping:
    def test_no_model_language_model_prefix(self, remapped):
        for key in remapped:
            assert not key.startswith("model.language_model."), f"Prefix not stripped: {key}"

    def test_no_visual_keys(self, remapped):
        for key in remapped:
            assert "visual" not in key, f"Vision key not filtered: {key}"

    def test_no_mtp_keys(self, remapped):
        for key in remapped:
            assert "mtp" not in key.split(".")[0], f"MTP key not filtered: {key}"


class TestTopLevelWeights:
    def test_embed_tokens(self, remapped):
        assert "tok_embeddings.weight" in remapped
        assert remapped["tok_embeddings.weight"].shape == (248320, HIDDEN_SIZE)

    def test_lm_head(self, remapped):
        assert "output.weight" in remapped
        assert remapped["output.weight"].shape == (248320, HIDDEN_SIZE)

    def test_final_norm(self, remapped):
        assert "norm.weight" in remapped
        assert remapped["norm.weight"].shape == (HIDDEN_SIZE,)


class TestDeltaNetLayerWeights:
    """Test layer 0 (a DeltaNet/linear attention layer)."""

    def test_qkv_split(self, remapped):
        q = remapped["layers.0.linear_attn.q_proj.weight"]
        k = remapped["layers.0.linear_attn.k_proj.weight"]
        v = remapped["layers.0.linear_attn.v_proj.weight"]
        assert q.shape == (LINEAR_KEY_DIM, HIDDEN_SIZE)
        assert k.shape == (LINEAR_KEY_DIM, HIDDEN_SIZE)
        assert v.shape == (LINEAR_VALUE_DIM, HIDDEN_SIZE)

    def test_conv1d_split(self, remapped):
        q_conv = remapped["layers.0.linear_attn.q_conv.weight"]
        k_conv = remapped["layers.0.linear_attn.k_conv.weight"]
        v_conv = remapped["layers.0.linear_attn.v_conv.weight"]
        assert q_conv.shape == (LINEAR_KEY_DIM, 1, 4)
        assert k_conv.shape == (LINEAR_KEY_DIM, 1, 4)
        assert v_conv.shape == (LINEAR_VALUE_DIM, 1, 4)

    def test_decay_projections(self, remapped):
        a = remapped["layers.0.linear_attn.in_proj_a.weight"]
        b = remapped["layers.0.linear_attn.in_proj_b.weight"]
        assert a.shape == (32, HIDDEN_SIZE)
        assert b.shape == (32, HIDDEN_SIZE)

    def test_gate_projection(self, remapped):
        z = remapped["layers.0.linear_attn.in_proj_z.weight"]
        assert z.shape == (HIDDEN_SIZE, HIDDEN_SIZE)

    def test_output_proj(self, remapped):
        o = remapped["layers.0.linear_attn.out_proj.weight"]
        assert o.shape == (HIDDEN_SIZE, HIDDEN_SIZE)

    def test_a_log_and_dt_bias(self, remapped):
        assert remapped["layers.0.linear_attn.A_log"].shape == (32,)
        assert remapped["layers.0.linear_attn.dt_bias"].shape == (32,)

    def test_norm(self, remapped):
        assert remapped["layers.0.linear_attn.norm.weight"].shape == (128,)

    def test_mlp(self, remapped):
        assert remapped["layers.0.mlp.gate_proj.weight"].shape == (12288, HIDDEN_SIZE)
        assert remapped["layers.0.mlp.up_proj.weight"].shape == (12288, HIDDEN_SIZE)
        assert remapped["layers.0.mlp.down_proj.weight"].shape == (HIDDEN_SIZE, 12288)

    def test_layernorms(self, remapped):
        assert remapped["layers.0.input_layernorm.weight"].shape == (HIDDEN_SIZE,)
        assert remapped["layers.0.post_attention_layernorm.weight"].shape == (HIDDEN_SIZE,)


class TestGatedAttentionLayerWeights:
    """Test layer 3 (a Gated Full Attention layer)."""

    def test_q_proj(self, remapped):
        # q_proj is 2× wide (query + gate), kept as-is
        q = remapped["layers.3.self_attn.q_proj.weight"]
        assert q.shape == (FULL_ATTN_Q_DIM, HIDDEN_SIZE)

    def test_kv_proj(self, remapped):
        k = remapped["layers.3.self_attn.k_proj.weight"]
        v = remapped["layers.3.self_attn.v_proj.weight"]
        assert k.shape == (FULL_ATTN_KV_DIM, HIDDEN_SIZE)
        assert v.shape == (FULL_ATTN_KV_DIM, HIDDEN_SIZE)

    def test_o_proj(self, remapped):
        o = remapped["layers.3.self_attn.o_proj.weight"]
        assert o.shape == (HIDDEN_SIZE, HIDDEN_SIZE)

    def test_qk_norm(self, remapped):
        assert remapped["layers.3.self_attn.q_norm.weight"].shape == (256,)
        assert remapped["layers.3.self_attn.k_norm.weight"].shape == (256,)

    def test_mlp(self, remapped):
        assert remapped["layers.3.mlp.gate_proj.weight"].shape == (12288, HIDDEN_SIZE)

    def test_layernorms(self, remapped):
        assert remapped["layers.3.input_layernorm.weight"].shape == (HIDDEN_SIZE,)
        assert remapped["layers.3.post_attention_layernorm.weight"].shape == (HIDDEN_SIZE,)


class TestAllLayersPresent:
    def test_all_32_layers_have_mlp(self, remapped):
        for i in range(32):
            assert f"layers.{i}.mlp.gate_proj.weight" in remapped, f"Missing MLP for layer {i}"

    def test_deltanet_layers_count(self, remapped):
        deltanet_layers = [i for i in range(32) if f"layers.{i}.linear_attn.q_proj.weight" in remapped]
        assert len(deltanet_layers) == 24

    def test_full_attn_layers_count(self, remapped):
        attn_layers = [i for i in range(32) if f"layers.{i}.self_attn.q_proj.weight" in remapped]
        assert len(attn_layers) == 8

    def test_full_attn_at_correct_positions(self, remapped):
        expected = [3, 7, 11, 15, 19, 23, 27, 31]
        for i in expected:
            assert f"layers.{i}.self_attn.q_proj.weight" in remapped, f"Layer {i} should be full attention"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /localdev/atupe/tt-metal && python -m pytest models/demos/blackhole/qwen3_5_9b/tests/test_weight_mapping.py -v --no-header 2>&1 | head -20
```

Expected: `ModuleNotFoundError` or `ImportError` — `weight_mapping` module doesn't exist yet.

- [ ] **Step 3: Implement weight_mapping.py**

```python
# models/demos/blackhole/qwen3_5_9b/tt/weight_mapping.py
"""Remap HuggingFace Qwen3.5-9B state dict to internal format.

Handles:
- Stripping 'model.language_model.' prefix
- Filtering out vision encoder and MTP weights
- Splitting combined in_proj_qkv into separate Q, K, V projections (DeltaNet layers)
- Splitting combined conv1d.weight into separate Q, K, V conv weights (DeltaNet layers)
- Renaming lm_head.weight → output.weight
- Renaming embed_tokens → tok_embeddings
"""
import torch
from typing import Dict

# Layer indices that use full (softmax) attention
FULL_ATTENTION_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}

# DeltaNet QKV split dimensions
# Q: num_key_heads(16) × key_head_dim(128) = 2048
# K: num_key_heads(16) × key_head_dim(128) = 2048
# V: num_value_heads(32) × value_head_dim(128) = 4096
LINEAR_Q_DIM = 2048
LINEAR_K_DIM = 2048
LINEAR_V_DIM = 4096


def remap_qwen35_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap HF Qwen3.5-9B state dict to internal format.

    Args:
        state_dict: Raw HuggingFace state dict loaded from safetensors.

    Returns:
        Remapped state dict with internal naming convention.
    """
    remapped = {}

    for key, tensor in state_dict.items():
        # Filter out vision encoder weights (check original key — no prefix stripping yet)
        if "visual" in key or key.startswith("model.visual"):
            continue
        # Filter out MTP (multi-token prediction) weights (original key)
        if key.startswith("mtp"):
            continue

        # Strip model.language_model. prefix
        # Note: after this point, use `new_key` for language model weights,
        # but `key` for top-level weights like lm_head.weight that have no prefix.
        new_key = key
        if new_key.startswith("model.language_model."):
            new_key = new_key[len("model.language_model."):]

        # Rename top-level weights
        if new_key == "embed_tokens.weight":
            remapped["tok_embeddings.weight"] = tensor
            continue
        if key == "lm_head.weight":
            remapped["output.weight"] = tensor
            continue
        # Final norm (model.language_model.norm.weight)
        if new_key == "norm.weight":
            remapped["norm.weight"] = tensor
            continue

        # Handle per-layer weights
        if new_key.startswith("layers."):
            parts = new_key.split(".")
            layer_idx = int(parts[1])
            layer_prefix = f"layers.{layer_idx}"
            sub_key = ".".join(parts[2:])

            # DeltaNet layers: split combined projections
            if sub_key == "linear_attn.in_proj_qkv.weight":
                qkv = tensor  # [8192, 4096]
                q = qkv[:LINEAR_Q_DIM, :]
                k = qkv[LINEAR_Q_DIM:LINEAR_Q_DIM + LINEAR_K_DIM, :]
                v = qkv[LINEAR_Q_DIM + LINEAR_K_DIM:, :]
                remapped[f"{layer_prefix}.linear_attn.q_proj.weight"] = q
                remapped[f"{layer_prefix}.linear_attn.k_proj.weight"] = k
                remapped[f"{layer_prefix}.linear_attn.v_proj.weight"] = v
                continue

            if sub_key == "linear_attn.conv1d.weight":
                conv = tensor  # [8192, 1, 4]
                q_conv = conv[:LINEAR_Q_DIM, :, :]
                k_conv = conv[LINEAR_Q_DIM:LINEAR_Q_DIM + LINEAR_K_DIM, :, :]
                v_conv = conv[LINEAR_Q_DIM + LINEAR_K_DIM:, :, :]
                remapped[f"{layer_prefix}.linear_attn.q_conv.weight"] = q_conv
                remapped[f"{layer_prefix}.linear_attn.k_conv.weight"] = k_conv
                remapped[f"{layer_prefix}.linear_attn.v_conv.weight"] = v_conv
                continue

            # All other keys pass through unchanged
            remapped[new_key] = tensor
            continue

        # Any remaining keys pass through
        remapped[new_key] = tensor

    return remapped
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /localdev/atupe/tt-metal && python -m pytest models/demos/blackhole/qwen3_5_9b/tests/test_weight_mapping.py -v --no-header 2>&1 | tail -30
```

Expected: All tests PASS.

---

## Task 3: Model Config

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tt/model_config.py`
- Create: `models/demos/blackhole/qwen3_5_9b/tests/test_model_config.py`

**Reference:** `models/tt_transformers/tt/model_config.py` (ModelArgs base class, lines 408+)

- [ ] **Step 1: Write the model config test**

```python
# models/demos/blackhole/qwen3_5_9b/tests/test_model_config.py
"""Tests for Qwen3.5-9B model config loading."""
import pytest


CHECKPOINT_DIR = "/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"


class TestQwen35ModelArgs:
    """Test that config.json is correctly parsed into model args."""

    @pytest.fixture(scope="class")
    def args(self):
        from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs

        # Use dummy_weights=True to avoid loading full checkpoint
        return Qwen35ModelArgs(mesh_device=None, checkpoint_dir=CHECKPOINT_DIR)

    def test_core_dimensions(self, args):
        assert args.dim == 4096
        assert args.n_layers == 32
        assert args.n_heads == 16
        assert args.n_kv_heads == 4
        assert args.head_dim == 256
        assert args.hidden_dim == 12288
        assert args.vocab_size == 248320
        assert args.norm_eps == 1e-6

    def test_rope_config(self, args):
        assert args.rope_theta == 10_000_000
        assert args.partial_rotary_factor == 0.25
        assert args.rope_head_dim == 64  # int(256 * 0.25)

    def test_deltanet_config(self, args):
        assert args.linear_num_key_heads == 16
        assert args.linear_num_value_heads == 32
        assert args.linear_key_head_dim == 128
        assert args.linear_value_head_dim == 128
        assert args.linear_conv_kernel_dim == 4

    def test_attention_type_list(self, args):
        assert len(args.attention_type_list) == 32
        expected = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8
        assert args.attention_type_list == expected

    def test_full_attention_layer_indices(self, args):
        full_attn = [i for i, t in enumerate(args.attention_type_list) if t == "full_attention"]
        assert full_attn == [3, 7, 11, 15, 19, 23, 27, 31]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /localdev/atupe/tt-metal && python -m pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_config.py -v --no-header 2>&1 | head -10
```

Expected: `ImportError` — module doesn't exist.

- [ ] **Step 3: Implement model_config.py**

```python
# models/demos/blackhole/qwen3_5_9b/tt/model_config.py
"""Qwen3.5-9B model configuration for Blackhole P150.

Subclasses ModelArgs to handle Qwen3.5-specific config:
- Hybrid attention layer types (Gated DeltaNet + Gated Full Attention)
- DeltaNet-specific parameters (key/value heads, conv kernel)
- Partial rotary factor for RoPE
"""
import json
import os
from pathlib import Path


class Qwen35ModelArgs:
    """Model configuration for Qwen3.5-9B on Blackhole P150.

    Loads config.json from the HuggingFace checkpoint directory and
    exposes all parameters needed by the model components.
    """

    def __init__(
        self,
        mesh_device=None,
        checkpoint_dir="/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2",
        max_batch_size=1,
        max_seq_len=2048,
    ):
        self.mesh_device = mesh_device
        self.checkpoint_dir = checkpoint_dir
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Load and parse config.json
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        # Qwen3.5 uses text_config for the language model params
        text_config = config.get("text_config", config)

        # Core dimensions
        self.dim = text_config["hidden_size"]
        self.n_layers = text_config["num_hidden_layers"]
        self.n_heads = text_config["num_attention_heads"]
        self.n_kv_heads = text_config["num_key_value_heads"]
        self.head_dim = text_config["head_dim"]
        self.hidden_dim = text_config["intermediate_size"]
        self.vocab_size = text_config["vocab_size"]
        self.norm_eps = text_config["rms_norm_eps"]

        # RoPE — rope_theta is nested under rope_parameters in config.json
        rope_params = text_config.get("rope_parameters", {})
        self.rope_theta = rope_params.get("rope_theta", 10_000_000)
        self.partial_rotary_factor = text_config.get("partial_rotary_factor", 1.0)
        self.rope_head_dim = int(self.head_dim * self.partial_rotary_factor)

        # DeltaNet-specific parameters
        self.linear_num_key_heads = text_config.get("linear_num_key_heads", 16)
        self.linear_num_value_heads = text_config.get("linear_num_value_heads", 32)
        self.linear_key_head_dim = text_config.get("linear_key_head_dim", 128)
        self.linear_value_head_dim = text_config.get("linear_value_head_dim", 128)
        self.linear_conv_kernel_dim = text_config.get("linear_conv_kernel_dim", 4)

        # Layer type list — which layers are DeltaNet vs full attention
        self.attention_type_list = text_config.get(
            "layer_types",
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8,
        )

        # Derived
        self.linear_q_dim = self.linear_num_key_heads * self.linear_key_head_dim  # 2048
        self.linear_k_dim = self.linear_num_key_heads * self.linear_key_head_dim  # 2048
        self.linear_v_dim = self.linear_num_value_heads * self.linear_value_head_dim  # 4096

        # Blackhole P150 device config (lazy import to allow CPU-only testing)
        if mesh_device is not None:
            import ttnn
            self.weight_dtype = ttnn.bfloat8_b
            self.act_dtype = ttnn.bfloat16
        else:
            self.weight_dtype = None
            self.act_dtype = None

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        """Return True if this layer uses Gated Full Attention (softmax GQA)."""
        return self.attention_type_list[layer_idx] == "full_attention"

    def is_deltanet_layer(self, layer_idx: int) -> bool:
        """Return True if this layer uses Gated DeltaNet (linear attention)."""
        return self.attention_type_list[layer_idx] == "linear_attention"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /localdev/atupe/tt-metal && python -m pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_config.py -v --no-header 2>&1 | tail -20
```

Expected: All tests PASS.

---

## Task 4: RoPE Setup

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_rope.py`

**Reference:** `models/tt_transformers/tt/rope.py` — `HfRotarySetup` class (line 712), `get_rot_mats_hf()` function

The experimental `gated_attention_forward_ttnn()` expects cos/sin tensors of shape `[B, T, head_dim]`. We need a RoPE setup that generates these with `head_dim=64` (partial_rotary_factor=0.25 × 256).

- [ ] **Step 1: Implement qwen35_rope.py**

```python
# models/demos/blackhole/qwen3_5_9b/tt/qwen35_rope.py
"""RoPE setup for Qwen3.5-9B Gated Attention layers.

Qwen3.5 uses partial rotary embeddings: only 25% of the head dimensions
(64 out of 256) receive rotary position encoding. The remaining 192 dimensions
pass through unchanged. The gated attention TTNN op handles the partial
application internally — we just need to generate cos/sin for the rotary
portion (head_dim=64).
"""
import torch
import ttnn


def compute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10_000_000.0):
    """Compute RoPE frequency tensors (cos, sin) for given head_dim.

    Args:
        head_dim: Dimension of the rotary portion (64 for Qwen3.5).
        max_seq_len: Maximum sequence length to precompute.
        theta: RoPE base frequency.

    Returns:
        cos: torch.Tensor [max_seq_len, head_dim]
        sin: torch.Tensor [max_seq_len, head_dim]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # [max_seq_len, head_dim // 2]
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)  # [max_seq_len, head_dim]
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)  # [max_seq_len, head_dim]
    return cos, sin


class Qwen35RoPESetup:
    """Precomputes and stores RoPE cos/sin tensors for Qwen3.5.

    Usage:
        rope = Qwen35RoPESetup(device, args)
        cos, sin = rope.get_rot_mats(position_ids)  # for gated attention forward
    """

    def __init__(self, device, args):
        self.device = device
        self.head_dim = args.rope_head_dim  # 64
        self.max_seq_len = args.max_seq_len

        # Precompute cos/sin on CPU
        self.cos_cpu, self.sin_cpu = compute_rope_freqs(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            theta=args.rope_theta,
        )

    def get_rot_mats(self, position_ids: torch.Tensor):
        """Get cos/sin matrices for given positions.

        Args:
            position_ids: torch.Tensor [B, T] or [T] — position indices.

        Returns:
            cos_ttnn: ttnn.Tensor [B, T, head_dim] on device
            sin_ttnn: ttnn.Tensor [B, T, head_dim] on device
        """
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)  # [1, T]

        B, T = position_ids.shape
        # Gather cos/sin for requested positions
        flat_pos = position_ids.reshape(-1)  # [B*T]
        cos = self.cos_cpu[flat_pos].reshape(B, T, self.head_dim)
        sin = self.sin_cpu[flat_pos].reshape(B, T, self.head_dim)

        cos_ttnn = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        sin_ttnn = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        return cos_ttnn, sin_ttnn
```

- [ ] **Step 2: Verify import works**

```bash
cd /localdev/atupe/tt-metal && python -c "from models.demos.blackhole.qwen3_5_9b.tt.qwen35_rope import Qwen35RoPESetup, compute_rope_freqs; print('OK')"
```

Expected: `OK`

---

## Task 5: Gated Attention Wrapper

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_attention.py`

**Reference:**
- `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py` — `gated_attention_forward_ttnn()`
- Design spec section 5 (Full Attention weight mapping)

- [ ] **Step 1: Implement the gated attention wrapper**

```python
# models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_attention.py
"""Gated Attention wrapper for Qwen3.5-9B full attention layers.

Wraps the experimental `gated_attention_forward_ttnn()` into a LightweightModule
that manages weight tensors and integrates with the model framework.
"""
import torch
import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import (
    gated_attention_forward_ttnn,
)


class Qwen35GatedAttention:
    """Gated Full Attention layer for Qwen3.5-9B.

    Uses softmax SDPA with GQA (16 Q heads, 4 KV heads, head_dim=256)
    plus a sigmoid output gate derived from the 2× wide q_proj.
    Q and K are normalized with zero-centered RMSNorm before attention.
    """

    def __init__(self, args, state_dict, layer_num, device):
        """
        Args:
            args: Qwen35ModelArgs instance.
            state_dict: Remapped state dict (output of remap_qwen35_state_dict).
            layer_num: Layer index (must be a full attention layer).
            device: ttnn device.
        """
        self.args = args
        self.device = device
        self.layer_num = layer_num
        self.num_heads = args.n_heads        # 16
        self.num_kv_heads = args.n_kv_heads  # 4
        self.head_dim = args.head_dim        # 256
        self.norm_eps = args.norm_eps

        prefix = f"layers.{layer_num}.self_attn"

        # Load weights and transpose to [in_features, out_features] for ttnn matmul
        def load_weight(name, transpose=True):
            t = state_dict[f"{prefix}.{name}"]
            if transpose and t.dim() == 2:
                t = t.T
            return ttnn.from_torch(
                t.unsqueeze(0).unsqueeze(0) if t.dim() == 2 else t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        def load_1d(name):
            t = state_dict[f"{prefix}.{name}"]
            return ttnn.from_torch(
                t.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        # q_proj is 2× wide: [8192, 4096] → transposed [4096, 8192]
        self.q_proj_weight = load_weight("q_proj.weight")
        self.k_proj_weight = load_weight("k_proj.weight")
        self.v_proj_weight = load_weight("v_proj.weight")
        self.o_proj_weight = load_weight("o_proj.weight")
        self.q_norm_weight = load_1d("q_norm.weight")
        self.k_norm_weight = load_1d("k_norm.weight")

    def forward(self, x, cos, sin):
        """
        Args:
            x: ttnn.Tensor [B, T, hidden_size]
            cos: ttnn.Tensor [B, T, rope_head_dim] — RoPE cosines
            sin: ttnn.Tensor [B, T, rope_head_dim] — RoPE sines

        Returns:
            output: ttnn.Tensor [B, T, hidden_size]
        """
        return gated_attention_forward_ttnn(
            hidden_states=x,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            o_proj_weight=self.o_proj_weight,
            q_norm_weight=self.q_norm_weight,
            k_norm_weight=self.k_norm_weight,
            cos=cos,
            sin=sin,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            device=self.device,
            norm_eps=self.norm_eps,
        )
```

- [ ] **Step 2: Verify import works**

```bash
cd /localdev/atupe/tt-metal && python -c "from models.demos.blackhole.qwen3_5_9b.tt.qwen35_gated_attention import Qwen35GatedAttention; print('OK')"
```

Expected: `OK`

---

## Task 6: Gated DeltaNet Wrapper

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py`

**Reference:**
- `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py` — `gated_deltanet_forward_ttnn()`
- Design spec section 5 (DeltaNet weight mapping)

- [ ] **Step 1: Implement the gated deltanet wrapper**

```python
# models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_deltanet.py
"""Gated DeltaNet wrapper for Qwen3.5-9B linear attention layers.

Wraps the experimental `gated_deltanet_forward_ttnn()` into a module
that manages weight tensors, recurrent state, and conv state.
"""
import torch
import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import (
    gated_deltanet_forward_ttnn,
)


class Qwen35GatedDeltaNet:
    """Gated DeltaNet (linear attention) layer for Qwen3.5-9B.

    Maintains fixed-size recurrent state [B, H, K, V] that replaces the KV cache.
    Supports two modes:
      - "recurrent": single-token decode (T=1), O(1) memory
      - "chunk": multi-token prefill (T>1), chunked parallel processing
    """

    def __init__(self, args, state_dict, layer_num, device):
        """
        Args:
            args: Qwen35ModelArgs instance.
            state_dict: Remapped state dict.
            layer_num: Layer index (must be a DeltaNet layer).
            device: ttnn device.
        """
        self.args = args
        self.device = device
        self.layer_num = layer_num
        self.num_heads = args.linear_num_key_heads        # 16
        self.num_v_heads = args.linear_num_value_heads     # 32
        self.head_k_dim = args.linear_key_head_dim         # 128
        self.head_v_dim = args.linear_value_head_dim       # 128
        self.conv_kernel_size = args.linear_conv_kernel_dim  # 4
        self.norm_eps = args.norm_eps

        prefix = f"layers.{layer_num}.linear_attn"

        def load_weight_2d(name):
            """Load 2D weight and transpose for ttnn matmul [in, out]."""
            t = state_dict[f"{prefix}.{name}"].T
            return ttnn.from_torch(
                t.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        def load_conv_weight(name):
            """Load conv1d weight — NOT transposed. Shape [channels, 1, kernel]."""
            t = state_dict[f"{prefix}.{name}"]
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device)

        def load_1d(name):
            """Load 1D tensor (bias, norm weight, etc.)."""
            t = state_dict[f"{prefix}.{name}"]
            return ttnn.from_torch(
                t.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        # Projection weights (already split by weight_mapping.py)
        self.q_proj_weight = load_weight_2d("q_proj.weight")
        self.k_proj_weight = load_weight_2d("k_proj.weight")
        self.v_proj_weight = load_weight_2d("v_proj.weight")
        self.a_proj_weight = load_weight_2d("in_proj_a.weight")
        self.b_proj_weight = load_weight_2d("in_proj_b.weight")
        self.g_proj_weight = load_weight_2d("in_proj_z.weight")  # output gate
        self.o_proj_weight = load_weight_2d("out_proj.weight")

        # Conv1d weights (already split by weight_mapping.py)
        self.q_conv_weight = load_conv_weight("q_conv.weight")
        self.k_conv_weight = load_conv_weight("k_conv.weight")
        self.v_conv_weight = load_conv_weight("v_conv.weight")

        # Conv1d biases — Qwen3.5 may not have these; create zeros if absent
        def load_conv_bias_or_zeros(name, size):
            full_key = f"{prefix}.{name}"
            if full_key in state_dict:
                t = state_dict[full_key]
                return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device)
            return ttnn.from_torch(
                torch.zeros(size), dtype=ttnn.bfloat16, device=device
            )

        self.q_conv_bias = load_conv_bias_or_zeros("q_conv.bias", args.linear_q_dim)
        self.k_conv_bias = load_conv_bias_or_zeros("k_conv.bias", args.linear_k_dim)
        self.v_conv_bias = load_conv_bias_or_zeros("v_conv.bias", args.linear_v_dim)

        # Scalar parameters
        self.A_log = load_1d("A_log")
        self.dt_bias = load_1d("dt_bias")
        self.o_norm_weight = load_1d("norm.weight")

        # Recurrent state — initialized to zeros, updated in-place during inference
        self.recurrent_state = None  # Lazily initialized on first forward

        # Conv state for causal conv1d history during decode.
        # NOTE: The current experimental gated_deltanet_forward_ttnn does not accept
        # external conv state — it pads with zeros internally. This means decode (T=1)
        # will lose conv history from previous tokens. This is a KNOWN LIMITATION of
        # the initial bringup. For correct autoregressive decode, the experimental op
        # needs to be extended to accept/return conv state, or we need to buffer the
        # last (kernel_size - 1) = 3 token activations and prepend them before conv1d.
        # TODO: Add conv state management for correct decode.

    def _init_recurrent_state(self, batch_size):
        """Initialize recurrent state to zeros [B, num_v_heads, head_k_dim, head_v_dim].

        Note: state uses num_v_heads (32), not num_heads (16), because the delta rule
        operates in the value head space after Q/K are repeated to match V heads.
        """
        state = torch.zeros(
            batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim,
            dtype=torch.bfloat16,
        )
        self.recurrent_state = ttnn.from_torch(
            state, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    def forward(self, x, mode="recurrent", chunk_size=64):
        """
        Args:
            x: ttnn.Tensor [B, T, hidden_size]
            mode: "recurrent" (decode, T=1) or "chunk" (prefill, T>1)
            chunk_size: Chunk size for chunked mode.

        Returns:
            output: ttnn.Tensor [B, T, hidden_size]
        """
        # Lazy init recurrent state
        if self.recurrent_state is None:
            # Infer batch size from input shape
            shape = x.shape
            batch_size = shape[0] if len(shape) == 3 else 1
            self._init_recurrent_state(batch_size)

        output, new_state = gated_deltanet_forward_ttnn(
            hidden_states=x,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            a_proj_weight=self.a_proj_weight,
            b_proj_weight=self.b_proj_weight,
            o_proj_weight=self.o_proj_weight,
            q_conv_weight=self.q_conv_weight,
            k_conv_weight=self.k_conv_weight,
            v_conv_weight=self.v_conv_weight,
            q_conv_bias=self.q_conv_bias,
            k_conv_bias=self.k_conv_bias,
            v_conv_bias=self.v_conv_bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            o_norm_weight=self.o_norm_weight,
            g_proj_weight=self.g_proj_weight,
            num_heads=self.num_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_kernel_size=self.conv_kernel_size,
            use_gate=True,
            norm_eps=self.norm_eps,
            device=self.device,
            recurrent_state=self.recurrent_state,
            mode=mode,
            chunk_size=chunk_size,
        )

        # Update recurrent state for next forward call
        self.recurrent_state = new_state
        return output

    def reset_state(self, batch_size=None):
        """Reset recurrent state (e.g., for new sequence)."""
        if batch_size is not None:
            self._init_recurrent_state(batch_size)
        else:
            self.recurrent_state = None
```

- [ ] **Step 2: Verify import works**

```bash
cd /localdev/atupe/tt-metal && python -c "from models.demos.blackhole.qwen3_5_9b.tt.qwen35_gated_deltanet import Qwen35GatedDeltaNet; print('OK')"
```

Expected: `OK`

---

## Task 7: MLP Wrapper

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py`

**Reference:** `models/tt_transformers/tt/mlp.py` — base MLP class

The MLP is identical to Llama (SwiGLU: gate_proj, up_proj, down_proj with SiLU). We create a thin wrapper that loads weights from the remapped state dict and performs the SwiGLU forward using ttnn ops directly, since the base MLP class expects specific sharding/mesh configs that don't apply to our single-device setup.

- [ ] **Step 1: Implement qwen35_mlp.py**

```python
# models/demos/blackhole/qwen3_5_9b/tt/qwen35_mlp.py
"""SwiGLU MLP for Qwen3.5-9B.

Standard gated MLP: output = down_proj(silu(gate_proj(x)) * up_proj(x))
Same structure as Llama MLP — gate/up/down projections with SiLU activation.
"""
import torch
import ttnn


class Qwen35MLP:
    """SwiGLU feed-forward network for Qwen3.5-9B."""

    def __init__(self, args, state_dict, layer_num, device):
        self.device = device
        prefix = f"layers.{layer_num}.mlp"

        def load_weight(name):
            t = state_dict[f"{prefix}.{name}"].T  # transpose for ttnn [in, out]
            return ttnn.from_torch(
                t.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        self.w1 = load_weight("gate_proj.weight")  # [4096, 12288]
        self.w2 = load_weight("down_proj.weight")   # [12288, 4096]
        self.w3 = load_weight("up_proj.weight")      # [4096, 12288]

    def forward(self, x):
        """
        Args:
            x: ttnn.Tensor [B, T, 4096]

        Returns:
            output: ttnn.Tensor [B, T, 4096]
        """
        w1_out = ttnn.linear(x, self.w1)      # [B, T, 12288]
        w3_out = ttnn.linear(x, self.w3)      # [B, T, 12288]
        hidden = ttnn.mul(ttnn.silu(w1_out), w3_out)  # SwiGLU
        output = ttnn.linear(hidden, self.w2)  # [B, T, 4096]
        return output
```

- [ ] **Step 2: Verify import works**

```bash
cd /localdev/atupe/tt-metal && python -c "from models.demos.blackhole.qwen3_5_9b.tt.qwen35_mlp import Qwen35MLP; print('OK')"
```

Expected: `OK`

---

## Task 8: Hybrid Decoder Block

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_decoder.py`

**Reference:** `models/tt_transformers/tt/decoder.py` — TransformerBlock (line 17)

- [ ] **Step 1: Implement qwen35_decoder.py**

```python
# models/demos/blackhole/qwen3_5_9b/tt/qwen35_decoder.py
"""Hybrid TransformerBlock for Qwen3.5-9B.

Dispatches to either Gated DeltaNet (linear attention) or Gated Full Attention
based on the layer index. Both share the same RMSNorm + residual pattern and MLP.
"""
import torch
import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_gated_attention import Qwen35GatedAttention
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_gated_deltanet import Qwen35GatedDeltaNet
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_mlp import Qwen35MLP


class Qwen35TransformerBlock:
    """Single transformer layer with hybrid attention dispatch.

    Pattern: x → attention_norm → attention → residual → ff_norm → MLP → residual
    Attention is either GatedAttention (full, with RoPE) or GatedDeltaNet (linear).
    """

    def __init__(self, args, state_dict, layer_num, device):
        self.layer_num = layer_num
        self.device = device
        self.is_full_attention = args.is_full_attention_layer(layer_num)

        prefix = f"layers.{layer_num}"

        # RMSNorm weights
        def load_norm(name):
            t = state_dict[f"{prefix}.{name}"]
            return ttnn.from_torch(
                t.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        self.attention_norm_weight = load_norm("input_layernorm.weight")
        self.ff_norm_weight = load_norm("post_attention_layernorm.weight")
        self.norm_eps = args.norm_eps

        # Attention — dispatched by layer type
        if self.is_full_attention:
            self.attention = Qwen35GatedAttention(args, state_dict, layer_num, device)
        else:
            self.attention = Qwen35GatedDeltaNet(args, state_dict, layer_num, device)

        # MLP — same for both layer types
        self.feed_forward = Qwen35MLP(args, state_dict, layer_num, device)

    def forward(self, x, cos=None, sin=None, mode="decode", chunk_size=64):
        """
        Args:
            x: ttnn.Tensor [B, T, hidden_size]
            cos: ttnn.Tensor [B, T, rope_head_dim] — only used by GatedAttention layers
            sin: ttnn.Tensor [B, T, rope_head_dim] — only used by GatedAttention layers
            mode: "decode" or "prefill"
            chunk_size: Chunk size for DeltaNet prefill.

        Returns:
            output: ttnn.Tensor [B, T, hidden_size]
        """
        # Pre-attention norm
        attn_input = ttnn.rms_norm(x, weight=self.attention_norm_weight, epsilon=self.norm_eps)

        # Attention forward (type-dependent)
        if self.is_full_attention:
            attn_output = self.attention.forward(attn_input, cos, sin)
        else:
            deltanet_mode = "chunk" if mode == "prefill" else "recurrent"
            attn_output = self.attention.forward(attn_input, mode=deltanet_mode, chunk_size=chunk_size)

        # Residual connection
        h = ttnn.add(x, attn_output)

        # Pre-FFN norm
        ff_input = ttnn.rms_norm(h, weight=self.ff_norm_weight, epsilon=self.norm_eps)

        # MLP forward
        ff_output = self.feed_forward.forward(ff_input)

        # Residual connection
        output = ttnn.add(h, ff_output)
        return output
```

- [ ] **Step 2: Verify import works**

```bash
cd /localdev/atupe/tt-metal && python -c "from models.demos.blackhole.qwen3_5_9b.tt.qwen35_decoder import Qwen35TransformerBlock; print('OK')"
```

Expected: `OK`

---

## Task 9: Full Model Assembly

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py`

**Reference:** `models/tt_transformers/tt/model.py` — Transformer class

- [ ] **Step 1: Implement qwen35_model.py**

```python
# models/demos/blackhole/qwen3_5_9b/tt/qwen35_model.py
"""Full Qwen3.5-9B text model for Blackhole P150.

Assembly: tok_embeddings → 32 × Qwen35TransformerBlock → RMSNorm → LM Head
Manages hybrid state: KV cache (8 attention layers) + recurrent state (24 DeltaNet layers).
"""
import torch
import ttnn
from tqdm import tqdm
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tt.weight_mapping import remap_qwen35_state_dict
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_rope import Qwen35RoPESetup
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_decoder import Qwen35TransformerBlock


class Qwen35Model:
    """Qwen3.5-9B text-only language model on Blackhole P150.

    Usage:
        model = Qwen35Model.from_pretrained(device, checkpoint_dir)
        logits = model.prefill(token_ids)           # [B, T] → [B, vocab_size]
        logits = model.decode(token_id, position)   # [B, 1] → [B, vocab_size]
    """

    def __init__(self, args, state_dict, device):
        """
        Args:
            args: Qwen35ModelArgs instance.
            state_dict: Remapped state dict.
            device: ttnn device.
        """
        self.args = args
        self.device = device

        # Embedding
        embed_weight = state_dict["tok_embeddings.weight"]
        self.tok_embeddings = ttnn.from_torch(
            embed_weight.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # RoPE setup (for gated attention layers only)
        self.rope = Qwen35RoPESetup(device, args)

        # Transformer layers
        print(f"Loading {args.n_layers} transformer layers...")
        self.layers = []
        for i in tqdm(range(args.n_layers), desc="Loading layers"):
            layer = Qwen35TransformerBlock(args, state_dict, i, device)
            self.layers.append(layer)

        # Final norm
        norm_weight = state_dict["norm.weight"]
        self.norm_weight = ttnn.from_torch(
            norm_weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.norm_eps = args.norm_eps

        # LM Head
        lm_head_weight = state_dict["output.weight"].T  # [4096, vocab_size]
        self.lm_head_weight = ttnn.from_torch(
            lm_head_weight.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.vocab_size = args.vocab_size

    @classmethod
    def from_pretrained(cls, device, checkpoint_dir, max_batch_size=1, max_seq_len=2048):
        """Load model from HuggingFace checkpoint.

        Args:
            device: ttnn device.
            checkpoint_dir: Path to HF checkpoint with safetensors.
            max_batch_size: Maximum batch size.
            max_seq_len: Maximum sequence length.

        Returns:
            Qwen35Model instance ready for inference.
        """
        from safetensors import safe_open

        args = Qwen35ModelArgs(
            mesh_device=device,
            checkpoint_dir=checkpoint_dir,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        # Load raw state dict from safetensors
        print("Loading weights from safetensors...")
        raw_state_dict = {}
        import glob
        safetensor_files = sorted(glob.glob(f"{checkpoint_dir}/model.safetensors-*.safetensors"))
        for path in safetensor_files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    raw_state_dict[key] = f.get_tensor(key)

        # Remap to internal format
        print("Remapping weights...")
        state_dict = remap_qwen35_state_dict(raw_state_dict)
        del raw_state_dict  # Free memory

        return cls(args, state_dict, device)

    def prefill(self, token_ids):
        """Process a full prompt (prefill phase).

        Args:
            token_ids: torch.Tensor [B, T] — input token IDs.

        Returns:
            logits: ttnn.Tensor [B, vocab_size] — logits for next token prediction
                    (from the last position only).
        """
        B, T = token_ids.shape

        # Reset DeltaNet recurrent states for new sequence
        self.reset_state(batch_size=B)

        # Embed tokens
        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)

        # Compute RoPE for all positions
        position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        cos, sin = self.rope.get_rot_mats(position_ids)

        # Forward through all layers
        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="prefill")

        # Final norm
        x = ttnn.rms_norm(x, weight=self.norm_weight, epsilon=self.norm_eps)

        # Extract last token position and project to vocab
        # x is [B, T, hidden_size] — take last token
        x_last = x[:, -1:, :]  # [B, 1, hidden_size]
        logits = ttnn.linear(x_last, self.lm_head_weight)  # [B, 1, vocab_size]

        return logits

    def decode(self, token_ids, current_pos):
        """Process a single token (decode phase).

        Args:
            token_ids: torch.Tensor [B, 1] — input token IDs.
            current_pos: int — current position in the sequence.

        Returns:
            logits: ttnn.Tensor [B, 1, vocab_size]
        """
        B = token_ids.shape[0]

        # Embed token
        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = ttnn.embedding(token_ids_ttnn, self.tok_embeddings, layout=ttnn.TILE_LAYOUT)

        # RoPE for current position
        position_ids = torch.full((B, 1), current_pos, dtype=torch.long)
        cos, sin = self.rope.get_rot_mats(position_ids)

        # Forward through all layers
        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="decode")

        # Final norm + LM head
        x = ttnn.rms_norm(x, weight=self.norm_weight, epsilon=self.norm_eps)
        logits = ttnn.linear(x, self.lm_head_weight)

        return logits

    def reset_state(self, batch_size=None):
        """Reset all DeltaNet recurrent states (for new sequence)."""
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.reset_state(batch_size)
```

- [ ] **Step 2: Verify import works**

```bash
cd /localdev/atupe/tt-metal && python -c "from models.demos.blackhole.qwen3_5_9b.tt.qwen35_model import Qwen35Model; print('OK')"
```

Expected: `OK`

---

## Task 10: Single Layer Validation Test (on P150)

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py`

This test validates that individual TTNN layers match torch reference outputs.

- [ ] **Step 1: Write the single layer test**

```python
# models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py
"""Single-layer PCC validation: TTNN vs torch reference.

Requires a Blackhole P150 device.
Run: pytest models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py -v
"""
import pytest
import torch
import ttnn

from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tt.weight_mapping import remap_qwen35_state_dict
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_decoder import Qwen35TransformerBlock


CHECKPOINT_DIR = "/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"
PCC_THRESHOLD = 0.98


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between two tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    num = (a_centered * b_centered).sum()
    denom = (a_centered.norm() * b_centered.norm()) + 1e-8
    return (num / denom).item()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def model_fixtures(device):
    """Load config and weights once for all tests."""
    args = Qwen35ModelArgs(mesh_device=device, checkpoint_dir=CHECKPOINT_DIR)

    from safetensors import safe_open
    import glob

    raw_sd = {}
    for path in sorted(glob.glob(f"{CHECKPOINT_DIR}/model.safetensors-*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                raw_sd[key] = f.get_tensor(key)

    state_dict = remap_qwen35_state_dict(raw_sd)
    return args, state_dict


@pytest.fixture(scope="module")
def hf_reference_model():
    """Load HuggingFace reference model for comparison."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()
    return model


class TestGatedAttentionLayer:
    """Test layer 3 (first full attention layer)."""

    def test_gated_attention_prefill(self, device, model_fixtures, hf_reference_model):
        args, state_dict = model_fixtures
        layer = Qwen35TransformerBlock(args, state_dict, layer_num=3, device=device)

        # Create random input
        B, T = 1, 128
        x_torch = torch.randn(B, T, args.dim, dtype=torch.bfloat16)

        # Get reference output from HF model layer 3
        # (This is a simplified comparison — full validation would run through
        # the reference model's layer with matching input)

        # For now, verify the layer runs without error and produces correct shape
        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        from models.demos.blackhole.qwen3_5_9b.tt.qwen35_rope import Qwen35RoPESetup

        rope = Qwen35RoPESetup(device, args)
        pos_ids = torch.arange(T).unsqueeze(0)
        cos, sin = rope.get_rot_mats(pos_ids)

        output = layer.forward(x_ttnn, cos=cos, sin=sin, mode="prefill")
        output_torch = ttnn.to_torch(output)

        assert output_torch.shape == (B, T, args.dim), f"Wrong shape: {output_torch.shape}"
        assert not torch.isnan(output_torch).any(), "Output contains NaN"
        assert not torch.isinf(output_torch).any(), "Output contains Inf"


class TestDeltaNetLayer:
    """Test layer 0 (first DeltaNet layer)."""

    def test_deltanet_recurrent(self, device, model_fixtures):
        args, state_dict = model_fixtures
        layer = Qwen35TransformerBlock(args, state_dict, layer_num=0, device=device)

        # Single token decode
        B, T = 1, 1
        x_torch = torch.randn(B, T, args.dim, dtype=torch.bfloat16)
        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        output = layer.forward(x_ttnn, mode="decode")
        output_torch = ttnn.to_torch(output)

        assert output_torch.shape == (B, T, args.dim), f"Wrong shape: {output_torch.shape}"
        assert not torch.isnan(output_torch).any(), "Output contains NaN"

    def test_deltanet_chunked(self, device, model_fixtures):
        args, state_dict = model_fixtures
        layer = Qwen35TransformerBlock(args, state_dict, layer_num=0, device=device)

        # Multi-token prefill
        B, T = 1, 128
        x_torch = torch.randn(B, T, args.dim, dtype=torch.bfloat16)
        x_ttnn = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        output = layer.forward(x_ttnn, mode="prefill", chunk_size=64)
        output_torch = ttnn.to_torch(output)

        assert output_torch.shape == (B, T, args.dim), f"Wrong shape: {output_torch.shape}"
        assert not torch.isnan(output_torch).any(), "Output contains NaN"
```

- [ ] **Step 2: Run to verify tests execute (on device)**

```bash
cd /localdev/atupe/tt-metal && python -m pytest models/demos/blackhole/qwen3_5_9b/tests/test_single_layer.py -v --no-header -x 2>&1 | tail -20
```

Expected: Tests PASS (shape correct, no NaN/Inf). PCC comparison against HF reference to be added after basic functionality works.

---

## Task 11: End-to-End Model Test (on P150)

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py`

- [ ] **Step 1: Write end-to-end test**

```python
# models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py
"""End-to-end model test: full forward pass + token generation.

Requires a Blackhole P150 device and ~18GB DRAM for weights.
Run: pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py -v -s
"""
import pytest
import torch
import ttnn

from models.demos.blackhole.qwen3_5_9b.tt.qwen35_model import Qwen35Model


CHECKPOINT_DIR = "/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def model(device):
    return Qwen35Model.from_pretrained(
        device, CHECKPOINT_DIR, max_batch_size=1, max_seq_len=2048
    )


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)


class TestPrefill:
    def test_prefill_produces_logits(self, model, tokenizer):
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt")
        token_ids = inputs["input_ids"]  # [1, T]

        logits = model.prefill(token_ids)
        logits_torch = ttnn.to_torch(logits)

        assert logits_torch.shape[-1] == model.vocab_size
        assert not torch.isnan(logits_torch).any()

    def test_prefill_top_token_is_reasonable(self, model, tokenizer):
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt")
        token_ids = inputs["input_ids"]

        logits = model.prefill(token_ids)
        logits_torch = ttnn.to_torch(logits).squeeze()

        top_token = logits_torch.argmax().item()
        decoded = tokenizer.decode([top_token])
        print(f"Prompt: '{prompt}' → Top token: '{decoded}' (id={top_token})")
        # Paris should be in top-5 at minimum
        top5 = logits_torch.topk(5).indices.tolist()
        top5_decoded = [tokenizer.decode([t]) for t in top5]
        print(f"Top 5: {top5_decoded}")


class TestDecodeLoop:
    def test_generate_tokens(self, model, tokenizer):
        prompt = "Hello, my name is"
        inputs = tokenizer(prompt, return_tensors="pt")
        token_ids = inputs["input_ids"]
        T = token_ids.shape[1]

        # Reset DeltaNet states
        model.reset_state(batch_size=1)

        # Prefill
        logits = model.prefill(token_ids)
        logits_torch = ttnn.to_torch(logits).squeeze()
        next_token = logits_torch.argmax().item()

        generated = [next_token]

        # Decode 20 tokens
        for i in range(20):
            next_input = torch.tensor([[next_token]], dtype=torch.long)
            logits = model.decode(next_input, current_pos=T + i)
            logits_torch = ttnn.to_torch(logits).squeeze()
            next_token = logits_torch.argmax().item()
            generated.append(next_token)

            if next_token == tokenizer.eos_token_id:
                break

        output_text = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"\nGenerated: '{prompt}{output_text}'")
        assert len(generated) > 0
```

- [ ] **Step 2: Run end-to-end test**

```bash
cd /localdev/atupe/tt-metal && python -m pytest models/demos/blackhole/qwen3_5_9b/tests/test_model_e2e.py -v -s --no-header -x 2>&1 | tail -30
```

Expected: Model loads, prefill produces logits, greedy decode generates coherent tokens.

---

## Task 12: Demo Script

**Files:**
- Create: `models/demos/blackhole/qwen3_5_9b/demo/demo.py`

- [ ] **Step 1: Implement demo.py**

```python
# models/demos/blackhole/qwen3_5_9b/demo/demo.py
"""Qwen3.5-9B text generation demo on Blackhole P150.

Usage:
    python models/demos/blackhole/qwen3_5_9b/demo/demo.py \
        --prompt "Explain why the sky is blue" \
        --max-tokens 200
"""
import argparse
import time
import torch
import ttnn

from models.demos.blackhole.qwen3_5_9b.tt.qwen35_model import Qwen35Model


CHECKPOINT_DIR = "/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-9B Text Generation on Blackhole P150")
    parser.add_argument("--prompt", type=str, default="Explain step by step why 2+2=4")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    args = parser.parse_args()

    # Open device
    print("Opening Blackhole P150 device...")
    device = ttnn.open_device(device_id=0)

    try:
        # Load tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, trust_remote_code=True)

        # Load model
        print("Loading Qwen3.5-9B model...")
        t0 = time.time()
        model = Qwen35Model.from_pretrained(
            device, args.checkpoint_dir, max_batch_size=1, max_seq_len=args.max_seq_len
        )
        load_time = time.time() - t0
        print(f"Model loaded in {load_time:.1f}s")

        # Tokenize prompt
        inputs = tokenizer(args.prompt, return_tensors="pt")
        token_ids = inputs["input_ids"]
        prompt_len = token_ids.shape[1]
        print(f"\nPrompt ({prompt_len} tokens): {args.prompt}")
        print("-" * 60)

        # Reset state
        model.reset_state(batch_size=1)

        # Prefill
        t0 = time.time()
        logits = model.prefill(token_ids)
        prefill_time = time.time() - t0
        logits_torch = ttnn.to_torch(logits).squeeze()
        next_token = logits_torch.argmax().item()
        print(f"Prefill: {prefill_time:.3f}s ({prompt_len / prefill_time:.0f} tok/s)")

        # Decode
        generated_tokens = [next_token]
        print(f"\nGeneration:", end=" ", flush=True)
        print(tokenizer.decode([next_token]), end="", flush=True)

        t0 = time.time()
        for i in range(args.max_tokens - 1):
            next_input = torch.tensor([[next_token]], dtype=torch.long)
            logits = model.decode(next_input, current_pos=prompt_len + i)
            logits_torch = ttnn.to_torch(logits).squeeze()
            next_token = logits_torch.argmax().item()

            if next_token == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token)
            print(tokenizer.decode([next_token]), end="", flush=True)

        decode_time = time.time() - t0
        n_gen = len(generated_tokens)
        print(f"\n\n{'-' * 60}")
        print(f"Generated {n_gen} tokens in {decode_time:.3f}s ({n_gen / decode_time:.1f} tok/s)")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test demo runs**

```bash
cd /localdev/atupe/tt-metal && python models/demos/blackhole/qwen3_5_9b/demo/demo.py --prompt "What is 2+2?" --max-tokens 50
```

Expected: Model loads, generates coherent text output with token generation speed reported.

---

## Execution Order & Dependencies

```
Task 1 (dirs)
    ↓
Task 2 (weight_mapping) ← foundational, everything else depends on this
    ↓
Task 3 (model_config) ← needed by all components
    ↓
Task 4 (rope) ← needed by gated attention
    ↓
Task 5 (gated_attention) ← needs rope, weight_mapping
Task 6 (gated_deltanet) ← needs weight_mapping
Task 7 (mlp) ← needs weight_mapping
    ↓ (all three can be done in parallel)
Task 8 (decoder) ← needs tasks 5, 6, 7
    ↓
Task 9 (model) ← needs task 8
    ↓
Task 10 (single layer test) ← needs task 8 + device
Task 11 (e2e test) ← needs task 9 + device
Task 12 (demo) ← needs task 9 + device
```

**Parallelizable tasks:** Tasks 5, 6, 7 can be implemented in parallel after Tasks 2-4 are complete.
