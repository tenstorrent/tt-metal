"""Model hooks for the unmodified advisor-challenger timing template."""
from __future__ import annotations

import argparse
import runpy

import torch
import ttnn
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from models.autoports.meta_llama_llama_3_2_1b_instruct.tests.test_functional_decoder import (
    DecodeRotaryHelper,
    MODEL_ID,
    _make_page_table,
    _synthetic_layer_state_dict,
    _to_tt_decode,
)
from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.optimized_decoder import (
    OptimizedDecoder,
    precision_policy_from_config,
)


def build(device, policy: dict):
    batch = 32
    max_seq_len = 64
    cfg = AutoConfig.from_pretrained(
        "models/tt_transformers/model_params/Llama-3.2-1B-Instruct", local_files_only=True
    )
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_layer_state_dict(cfg), hf_config=cfg, layer_idx=0, mesh_device=device,
        page_block_size=64, max_seq_len=max_seq_len, max_batch_size=batch,
        precision_policy=precision_policy_from_config(policy),
    )
    _, page_table = _make_page_table(device, batch=batch, max_seq_len=max_seq_len, block_size=64, seed=17)
    positions = torch.zeros(batch, dtype=torch.int32)
    current_pos = ttnn.from_torch(
        positions, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    rope = DecodeRotaryHelper(LlamaRotaryEmbedding(cfg), max_seq_len, cfg.head_dim, device)
    rot_mats = rope.get_rot_mats(positions)
    hidden = torch.randn(1, batch, cfg.hidden_size, dtype=torch.bfloat16) * 0.15
    decode_input = _to_tt_decode(hidden, decoder, device)
    return decoder, decode_input, current_pos, rot_mats, page_table


def decode(state):
    decoder, hidden, current_pos, rot_mats, page_table = state
    return decoder.decode_forward(hidden, current_pos=current_pos, rot_mats=rot_mats, page_table=page_table)


if __name__ == "__main__":
    template = runpy.run_path(".agents/skills/advisor-challenger/scripts/harness_template.py", run_name="advisor_challenger_template")
    template["measure"].__globals__["build"] = build
    template["measure"].__globals__["decode"] = decode
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="incumbent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy")
    args = ap.parse_args()
    default_policy = f"models/autoports/{template['MODEL_DIR']}/doc/advisor_challenger/incumbent.json"
    template["measure"](args.label, args.out, args.policy or default_policy)
