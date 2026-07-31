# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Model hooks for the fixed advisor-challenger timing protocol."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch

_TEMPLATE = Path(__file__).parents[5] / ".agents/skills/advisor-challenger/scripts/harness_template.py"
_SPEC = importlib.util.spec_from_file_location("advisor_challenger_harness_template", _TEMPLATE)
template = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(template)
_PROFILE_DEVICE = None


def _dtype(ttnn, value: str):
    return getattr(ttnn, value)


def build(device, policy: dict):
    global _PROFILE_DEVICE
    import ttnn
    from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_functional_decoder import (
        _decode_rot_mats,
        _hf_rotary,
        _page_table,
        _rope_setup,
        _synthetic_state_dict,
        _tt_tensor,
    )
    from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import OptimizedDecoder
    from transformers import LlamaConfig

    batch = template.BATCH
    _PROFILE_DEVICE = device
    cfg = LlamaConfig(
        hidden_size=4096, intermediate_size=14336, num_hidden_layers=32,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128,
        max_position_embeddings=131072, rms_norm_eps=1e-5, rope_theta=500000.0,
        rope_scaling={"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                      "original_max_position_embeddings": 8192, "rope_type": "llama3"},
    )
    cfg._attn_implementation = "eager"
    max_context = int(os.environ.get("CHALLENGER_MAX_CONTEXT", "128"))
    max_num_blocks = batch * 2
    kwargs = {name: _dtype(ttnn, value) for name, value in policy.items()}
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state_dict(), hf_config=cfg, layer_idx=0, mesh_device=device,
        max_batch_size=batch, max_seq_len=max_context + 64, page_block_size=64,
        max_num_blocks=max_num_blocks, **kwargs,
    )
    _, page_table = _page_table(device, batch=batch, max_num_blocks=max_num_blocks)
    current_pos_host = torch.full((batch,), max_context, dtype=torch.int32)
    current_pos = ttnn.from_torch(
        current_pos_host, device=device, dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    rope = _rope_setup(device, cfg, _hf_rotary(cfg), max_context + 65, batch)
    rot_mats = _decode_rot_mats(rope, current_pos_host.to(torch.long))
    hidden = torch.randn(1, 1, batch, cfg.hidden_size, dtype=torch.bfloat16) * 0.05
    decode_input = ttnn.to_memory_config(_tt_tensor(device, hidden), decoder.decode_residual_memcfg)
    return decoder, decode_input, current_pos, rot_mats, page_table


def decode(state):
    decoder, hidden, current_pos, rot_mats, page_table = state
    return decoder.decode_forward(hidden, current_pos=current_pos, rot_mats=rot_mats, page_table=page_table)


template.build = build
template.decode = decode

if __name__ == "__main__":
    if os.environ.get("CHALLENGER_CLEAR_PROFILER_BEFORE_SIGNPOST") == "1":
        import tracy
        import ttnn

        original_signpost = tracy.signpost

        def clearing_signpost(*args, **kwargs):
            header = kwargs.get("header") or (args[0] if args else None)
            if header == "PERF_DECODE":
                ttnn.ReadDeviceProfiler(_PROFILE_DEVICE)
            return original_signpost(*args, **kwargs)

        tracy.signpost = clearing_signpost
    args = template.argparse.ArgumentParser()
    args.add_argument("--label", default="incumbent")
    args.add_argument("--out", required=True)
    args.add_argument("--policy", default=None)
    parsed = args.parse_args()
    default_policy = f"models/autoports/{template.MODEL_DIR}/doc/advisor_challenger/incumbent.json"
    if parsed.label == "incumbent" and not parsed.policy:
        raise SystemExit("--policy is required for the incumbent run")
    template.measure(parsed.label, parsed.out, parsed.policy or default_policy)
