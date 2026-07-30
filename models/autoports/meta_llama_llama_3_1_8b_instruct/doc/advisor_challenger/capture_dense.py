# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Advisor capture target for the single dense Llama 3.1 8B layer kind."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn

TT_METAL_ROOT = Path(os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal"))
if str(TT_METAL_ROOT) not in sys.path:
    sys.path.append(str(TT_METAL_ROOT))

OUT_DIR = Path(os.environ["CHALLENGER_OUT_DIR"])
INCUMBENT = TT_METAL_ROOT / (
    "models/autoports/meta_llama_llama_3_1_8b_instruct/"
    "doc/advisor_challenger/incumbent.json"
)
BATCH = 32
MAX_SEQ_LEN = 128
PAGE_BLOCK_SIZE = 64
MAX_NUM_BLOCKS = 64

_DECODER = None
_KWARGS = None


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="llama",
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        rms_norm_eps=1e-5,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        max_position_embeddings=131072,
        rope_theta=500000.0,
    )


def _synthetic_state_dict():
    cfg = _config()
    generator = torch.Generator().manual_seed(20260730)

    def randn(*shape):
        return (torch.randn(*shape, generator=generator) * 0.02).to(torch.bfloat16)

    prefix = "model.layers.0."
    return {
        prefix + "input_layernorm.weight": torch.ones(cfg.hidden_size, dtype=torch.bfloat16),
        prefix + "post_attention_layernorm.weight": torch.ones(cfg.hidden_size, dtype=torch.bfloat16),
        prefix + "self_attn.q_proj.weight": randn(cfg.hidden_size, cfg.hidden_size),
        prefix + "self_attn.k_proj.weight": randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
        prefix + "self_attn.v_proj.weight": randn(cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
        prefix + "self_attn.o_proj.weight": randn(cfg.hidden_size, cfg.hidden_size),
        prefix + "mlp.gate_proj.weight": randn(cfg.intermediate_size, cfg.hidden_size),
        prefix + "mlp.up_proj.weight": randn(cfg.intermediate_size, cfg.hidden_size),
        prefix + "mlp.down_proj.weight": randn(cfg.hidden_size, cfg.intermediate_size),
    }


def _policy():
    from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (
        OptimizedDecoderPolicy,
    )

    incumbent = json.loads(INCUMBENT.read_text())
    executed = incumbent["shipped_policy"]["policy"]
    dtype = {
        "BFLOAT16": ttnn.bfloat16,
        "BFLOAT8_B": ttnn.bfloat8_b,
        "BFLOAT4_B": ttnn.bfloat4_b,
    }
    fidelity = {
        "LoFi": ttnn.MathFidelity.LoFi,
        "HiFi2": ttnn.MathFidelity.HiFi2,
        "HiFi4": ttnn.MathFidelity.HiFi4,
    }
    return OptimizedDecoderPolicy(
        name=executed["name"],
        activation_dtype=dtype[executed["activation_dtype"]],
        attention_weight_dtype=dtype[executed["attention_weight_dtype"]],
        mlp_gate_up_dtype=dtype[executed["mlp_gate_up_dtype"]],
        mlp_down_dtype=dtype[executed["mlp_down_dtype"]],
        kv_cache_dtype=dtype[executed["kv_cache_dtype"]],
        mlp_mul_dtype=dtype[executed["mlp_mul_dtype"]],
        mlp_math_fidelity=fidelity[executed["mlp_math_fidelity"]],
    )


def _build(device):
    from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (
        OptimizedDecoder,
    )
    from models.common.modules.lazy_weight import LazyWeight
    from models.common.modules.rope.rope_1d import Rope1DConfig, RotarySetup1D, prepare_rot_idxs

    cfg = _config()
    policy = _policy()
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state_dict(),
        hf_config=cfg,
        layer_idx=0,
        mesh_device=device,
        max_batch_size=BATCH,
        max_seq_len=MAX_SEQ_LEN,
        page_block_size=PAGE_BLOCK_SIZE,
        max_num_blocks=MAX_NUM_BLOCKS,
        policy=policy,
    )
    page_table_host = torch.arange(MAX_NUM_BLOCKS, dtype=torch.int32).reshape(
        BATCH, MAX_NUM_BLOCKS // BATCH
    )
    page_table = ttnn.from_torch(
        page_table_host,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    current_pos_host = torch.zeros((BATCH,), dtype=torch.int32)
    current_pos = ttnn.from_torch(
        current_pos_host,
        device=device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    positions = torch.arange(MAX_SEQ_LEN + 1, dtype=torch.float32)
    inv_freq = 1.0 / (
        cfg.rope_theta
        ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim)
    )
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
    sin = emb.sin().to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
    rope = RotarySetup1D.from_config(
        Rope1DConfig(
            cos_matrix=LazyWeight(
                source=cos,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            sin_matrix=LazyWeight(
                source=sin,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            max_batch_size=BATCH,
            head_dim=cfg.head_dim,
            device=device,
            use_qk_fused=False,
            datatype=ttnn.bfloat16,
        )
    )
    rot_idxs = prepare_rot_idxs(rope.config, current_pos_host.to(torch.long), on_host=False)
    rot_mats = tuple(rope.decode_forward(rot_idxs))
    hidden = torch.randn(1, 1, BATCH, cfg.hidden_size, dtype=torch.bfloat16)
    tt_interleaved = ttnn.from_torch(
        hidden,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    tt_hidden = ttnn.to_memory_config(
        tt_interleaved, decoder.decode_residual_memcfg
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "traced_dtypes.json").write_text(
        json.dumps(
            {
                "layer_kind": "dense",
                "layer_idx": 0,
                "batch": BATCH,
                "traced_weight_dtypes": {
                    "attention": "BFLOAT8_B",
                    "mlp_gate_up": "BFLOAT4_B",
                    "mlp_down": "BFLOAT4_B",
                    "norm": "BFLOAT16",
                },
                "policy_source": incumbent_source(),
            },
            indent=2,
        )
        + "\n"
    )
    kwargs = {
        "current_pos": current_pos,
        "rot_mats": rot_mats,
        "page_table": page_table,
    }
    return decoder, kwargs, tt_hidden


def incumbent_source() -> str:
    return json.loads(INCUMBENT.read_text())["shipped_policy_source"]


def decode(hidden):
    return _DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, _KWARGS, hidden = _build(device)
    return (hidden,)
