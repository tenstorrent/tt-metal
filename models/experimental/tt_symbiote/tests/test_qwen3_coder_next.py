# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Qwen3-Coder-Next with TTNN backend (MoE + Gated Attention)."""

import os

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

# Prefer FP8 model when triton is available; otherwise use BF16
QWEN3_MODEL_ID = "Qwen/Qwen3-Coder-Next-FP8"
QWEN3_FALLBACK_MODEL_ID = "Qwen/Qwen3-Coder-Next"


def _load_qwen3_model():
    """Load tokenizer and model, with fallbacks for triton/network/deps."""
    for model_id in (QWEN3_MODEL_ID, QWEN3_FALLBACK_MODEL_ID):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(torch.bfloat16)
            return tokenizer, model
        except ModuleNotFoundError as e:
            if "triton" in str(e).lower():
                continue
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
    pytest.skip("Could not load Qwen3 model (triton required for FP8)")


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

    messages = [
        {
            "role": "user",
            "content": "Write a Python function to calculate fibonacci numbers using dynamic programming.",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device).to(torch.bfloat16)

    modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)
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

    output_ids = outputs[0][len(inputs["input_ids"][0]) :].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True)

    print(f"Qwen3-Coder-Next output: {content}")
    DispatchManager.save_stats_to_file("qwen3_coder_next_timing_stats.csv")
