# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests Qwen3-Coder-Next MoE (Qwen3NextSparseMoeBlock) with TTNN acceleration."""

import json
import os
from pathlib import Path

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.modules.moe import TTNNQwen3MoE
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

QWEN3_CODER_NEXT_MODEL_PATH = "Qwen/Qwen3-Coder-Next"
QWEN3_REAL_WEIGHTS_LAYER_INDEX = 3

_MESH_DEVICE_ENV = "MESH_DEVICE"
if _MESH_DEVICE_ENV not in os.environ:
    os.environ[_MESH_DEVICE_ENV] = "T3K"
MESH_DEVICE = os.environ.get(_MESH_DEVICE_ENV, "T3K")

MESH_SHAPE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}


def _load_qwen3_moe_from_cache(
    model_path=QWEN3_CODER_NEXT_MODEL_PATH,
    layer_index=QWEN3_REAL_WEIGHTS_LAYER_INDEX,
):
    """
    Load a Qwen3NextSparseMoeBlock with real weights from cached HF safetensor shards.

    Only loads the shards needed for the requested layer — does not require the full
    model download. Falls back to AutoModelForCausalLM.from_pretrained if cache is
    unavailable. Skips the test on network / download errors.
    """
    from safetensors.torch import load_file
    from transformers import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock

    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_path.replace('/', '--')}"
    snap = _find_snapshot(cache_root)
    if snap is None:
        return _load_qwen3_moe_via_pretrained(model_path, layer_index)

    config_path = snap / "config.json"
    index_path = snap / "model.safetensors.index.json"
    if not config_path.exists() or not index_path.exists():
        return _load_qwen3_moe_via_pretrained(model_path, layer_index)

    with open(config_path) as f:
        config = Qwen3NextConfig(**json.load(f))
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    prefix = f"model.layers.{layer_index}.mlp."
    layer_keys = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
    if not layer_keys:
        pytest.skip(f"No MoE weights for layer {layer_index} in weight map")

    needed_shards = sorted(set(layer_keys.values()))
    for shard in needed_shards:
        if not (snap / shard).exists():
            return _load_qwen3_moe_via_pretrained(model_path, layer_index)

    raw = {}
    for shard in needed_shards:
        sd = load_file(str(snap / shard))
        for k in layer_keys:
            if layer_keys[k] == shard and k in sd:
                raw[k[len(prefix) :]] = sd[k]
        del sd

    fused = {}
    gate_up, down = [], []
    for i in range(config.num_experts):
        gate_up.append(torch.cat([raw[f"experts.{i}.gate_proj.weight"], raw[f"experts.{i}.up_proj.weight"]], dim=0))
        down.append(raw[f"experts.{i}.down_proj.weight"])
    fused["experts.gate_up_proj"] = torch.stack(gate_up, dim=0)
    fused["experts.down_proj"] = torch.stack(down, dim=0)
    for k, v in raw.items():
        if not k.startswith("experts.") or not any(p in k for p in ("gate_proj", "up_proj", "down_proj")):
            fused[k] = v

    moe = Qwen3NextSparseMoeBlock(config)
    missing, _ = moe.load_state_dict(fused, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys loading MoE layer {layer_index}: {missing}")

    return moe.to(torch.bfloat16), config.hidden_size


def _find_snapshot(cache_root):
    snapshots = cache_root / "snapshots"
    if not snapshots.exists():
        return None
    dirs = list(snapshots.iterdir())
    return dirs[0] if dirs else None


def _load_qwen3_moe_via_pretrained(model_path, layer_index):
    """Fallback: load full model via from_pretrained. Skips on network errors."""
    from transformers import AutoModelForCausalLM

    try:
        full_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
    except OSError as e:
        errno_val = getattr(e, "errno", None)
        msg = str(e).lower()
        if errno_val in (-3, 101) or any(
            kw in msg
            for kw in ("name resolution", "temporary failure", "network is unreachable", "does not appear to have")
        ):
            pytest.skip(f"Hugging Face model unavailable: {e}")
        raise
    except Exception as e:
        msg = str(e).lower()
        if any(
            kw in msg
            for kw in (
                "network is unreachable",
                "name resolution",
                "does not appear to have",
                "disconnected",
                "remoteprotocolerror",
                "connection",
            )
        ):
            pytest.skip(f"Hugging Face model unavailable: {e}")
        raise

    moe_block = full_model.model.layers[layer_index].mlp
    hidden_size = full_model.config.hidden_size
    del full_model
    return moe_block.to(torch.bfloat16), hidden_size


def _make_qwen3_moe_random(
    hidden_size=2048,
    intermediate_size=4096,
    moe_intermediate_size=1024,
    shared_expert_intermediate_size=2048,
    num_experts=64,
    num_experts_per_tok=4,
):
    """Create a synthetic Qwen3NextSparseMoeBlock with random weights."""
    from transformers import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock

    config = Qwen3NextConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        norm_topk_prob=True,
        decoder_sparse_step=1,
    )
    return Qwen3NextSparseMoeBlock(config).to(torch.bfloat16), hidden_size


def _compute_pcc(outputs_torch, outputs_ttnn):
    t = outputs_torch.to(torch.float32)
    n = outputs_ttnn if isinstance(outputs_ttnn, TorchTTNNTensor) else TorchTTNNTensor(outputs_ttnn)
    n.elem = None
    n = n.to_torch.to(torch.float32)
    pcc = torch.corrcoef(torch.stack([t.flatten(), n.flatten()]))[0, 1]
    diff = torch.abs(t - n)
    return pcc.item(), torch.max(diff).item(), torch.mean(diff).item()


@pytest.mark.parametrize(
    "mesh_device",
    [MESH_SHAPE_MAP.get(MESH_DEVICE, (1, 8))],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 245760, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
class TestQwen3MoE:
    PCC_THRESHOLD = 0.99

    def test_random_weights(self, mesh_device):
        """Qwen3-Coder-Next MoE with synthetic random weights (no download)."""
        model, hidden_size = _make_qwen3_moe_random()
        model.eval()
        torch.set_grad_enabled(False)

        inputs = torch.randn((1, 115, hidden_size), dtype=torch.bfloat16)
        outputs_torch = model(inputs)

        ttnn_model = TTNNQwen3MoE.from_torch(model)
        set_device(ttnn_model, mesh_device)
        outputs_ttnn = ttnn_model.forward_validate(inputs)

        pcc, max_diff, mean_diff = _compute_pcc(outputs_torch, outputs_ttnn)
        print(f"Qwen3NextSparseMoeBlock (random) PCC: {pcc:.6f}, Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")
        assert pcc >= self.PCC_THRESHOLD, f"Qwen3NextSparseMoeBlock PCC {pcc:.6f} below {self.PCC_THRESHOLD}"

    def test_real_weights(self, mesh_device):
        """Qwen3-Coder-Next MoE with real HF weights. Skips if unavailable."""
        model, hidden_size = _load_qwen3_moe_from_cache()
        model.eval()
        torch.set_grad_enabled(False)

        inputs = torch.randn((1, 115, hidden_size), dtype=torch.bfloat16)
        outputs_torch = model(inputs)

        ttnn_model = TTNNQwen3MoE.from_torch(model)
        set_device(ttnn_model, mesh_device)
        outputs_ttnn = ttnn_model.forward_validate(inputs)

        pcc, max_diff, mean_diff = _compute_pcc(outputs_torch, outputs_ttnn)
        print(f"Qwen3NextSparseMoeBlock (real) PCC: {pcc:.6f}, Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")
        assert pcc >= self.PCC_THRESHOLD, f"Qwen3NextSparseMoeBlock PCC {pcc:.6f} below {self.PCC_THRESHOLD}"
