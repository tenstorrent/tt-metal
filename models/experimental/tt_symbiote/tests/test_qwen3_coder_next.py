# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Qwen3-Coder-Next with TTNN backend (MoE + Gated Attention)."""

import os
from pathlib import Path

import pytest
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextAttention, Qwen3NextSparseMoeBlock

from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.attention import TTNNQwen3NextGatedAttention
from models.experimental.tt_symbiote.modules.moe import TTNNQwen3MoE
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

QWEN3_MODEL_ID = "Qwen/Qwen3-Coder-Next-FP8"


def _get_cached_model_path():
    """Resolve HF cache snapshot path for Qwen3-Coder-Next-FP8 (avoids network)."""
    if hub := os.environ.get("HF_HUB_CACHE"):
        cache_root = Path(hub) / f"models--{QWEN3_MODEL_ID.replace('/', '--')}"
    else:
        base = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        cache_root = Path(base) / "hub" / f"models--{QWEN3_MODEL_ID.replace('/', '--')}"
    snapshots = cache_root / "snapshots"
    if not snapshots.exists():
        return None
    dirs = sorted(snapshots.iterdir())
    return str(dirs[0]) if dirs else None


def _load_qwen3_model():
    """Load tokenizer and model. Uses HF cache if available to avoid network."""
    try:
        cached = _get_cached_model_path()
        if cached:
            tokenizer = AutoTokenizer.from_pretrained(cached, trust_remote_code=True, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(cached, trust_remote_code=True, local_files_only=True).to(
                torch.bfloat16
            )
            return tokenizer, model
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(QWEN3_MODEL_ID, trust_remote_code=True).to(torch.bfloat16)
        return tokenizer, model
    except ModuleNotFoundError as e:
        if "triton" in str(e).lower():
            pytest.skip("Qwen3-Coder-Next-FP8 requires triton for loading")
        raise
    except Exception as e:
        msg = str(e).lower()
        if any(
            kw in msg
            for kw in (
                "network is unreachable",
                "name resolution",
                "connection",
                "protobuf",
                "does not appear to have",
            )
        ):
            pytest.skip(f"Model unavailable: {e}")
        raise


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
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
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_qwen3_coder_next(mesh_device):
    """Test Qwen3-Coder-Next model with TTNN acceleration (MoE + Gated Attention)."""
    tokenizer, model = _load_qwen3_model()

    nn_to_ttnn = {
        Qwen3NextSparseMoeBlock: TTNNQwen3MoE,
        Qwen3NextAttention: TTNNQwen3NextGatedAttention,
    }
    nn_to_ttnn2 = {
        # nn.Linear: TTNNLinearIColShardedWRowSharded,
    }

    # All layers use TTNN (no torch fallback). MoE uses lazy weight loading to avoid OOM.
    exclude_replacement = set()

    messages = [
        {
            "role": "user",
            "content": "Write a Python function to calculate fibonacci numbers using dynamic programming.",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    def _to_device(v):
        if not isinstance(v, torch.Tensor):
            return v
        v = v.to(model.device)
        # Keep input_ids (and other index tensors) as long for embedding; only float tensors get bfloat16
        if v.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            return v
        return v.to(torch.bfloat16)

    inputs = {k: _to_device(v) for k, v in inputs.items()}

    modules1 = register_module_replacement_dict(
        model, nn_to_ttnn, model_config=None, exclude_replacement=exclude_replacement
    )
    modules2 = register_module_replacement_dict(
        model, nn_to_ttnn2, model_config=None, exclude_replacement=exclude_replacement
    )
    set_device(model, mesh_device)
    all_modules = {**modules1, **modules2}

    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    print("Running inference...")
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead

    # Warmup
    outputs = model.generate(**inputs, max_new_tokens=2, use_cache=True)

    DispatchManager.clear_timings()
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)

    prompt_len = inputs["input_ids"].shape[-1]
    output_ids = outputs[0][prompt_len:].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True)

    print(f"Qwen3-Coder-Next output: {content}")
    DispatchManager.save_stats_to_file("qwen3_coder_next_timing_stats.csv")
