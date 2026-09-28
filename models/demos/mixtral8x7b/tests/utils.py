# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import torch
from transformers import AutoConfig


def load_hf_mixtral_config():
    hf_model = os.getenv("HF_MODEL")
    assert hf_model is not None, "Please set HF_MODEL to a HuggingFace name e.g. meta-llama/Llama-3.1-8B-Instruct"
    config = AutoConfig.from_pretrained(hf_model, local_files_only=os.getenv("CI") == "true")
    return config


def fuse_mixtral_experts(state_dict, num_experts, src_prefix, dst_prefix):
    """Convert per-expert w1/w2/w3 weights to the transformers 5.x fused MixtralExperts format.

    transformers 5.x replaced the per-expert MixtralBlockSparseTop2MLP modules with a single fused
    ``MixtralExperts`` module whose weights are applied via ``F.linear`` (same orientation as the old
    per-expert ``nn.Linear`` weights):
      * ``{dst_prefix}.gate_up_proj``: (num_experts, 2*intermediate, hidden) = cat([w1, w3], dim=0)
        per expert (gate first, up second; split via ``chunk(2, dim=-1)`` in the forward).
      * ``{dst_prefix}.down_proj``:    (num_experts, hidden, intermediate)   = w2 per expert.

    ``src_prefix`` is where the per-expert weights live (e.g. ``experts`` for a standalone
    MixtralSparseMoeBlock, ``block_sparse_moe.experts`` inside a decoder layer). The matching
    per-expert source keys are consumed; all other keys pass through unchanged.
    """
    consumed = set()
    gate_up, down = [], []
    for e in range(num_experts):
        w1 = state_dict[f"{src_prefix}.{e}.w1.weight"]
        w2 = state_dict[f"{src_prefix}.{e}.w2.weight"]
        w3 = state_dict[f"{src_prefix}.{e}.w3.weight"]
        gate_up.append(torch.cat([w1, w3], dim=0))
        down.append(w2)
        consumed.update(f"{src_prefix}.{e}.w{i}.weight" for i in (1, 2, 3))
    out = {k: v for k, v in state_dict.items() if k not in consumed}
    out[f"{dst_prefix}.gate_up_proj"] = torch.stack(gate_up, dim=0)
    out[f"{dst_prefix}.down_proj"] = torch.stack(down, dim=0)
    return out
