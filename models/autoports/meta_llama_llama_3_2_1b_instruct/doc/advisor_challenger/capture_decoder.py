# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shipped-precision batch-32 capture target for the dense Llama layer kind."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MethodType

import torch
import ttnn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

TT_METAL_ROOT = "/home/mvasiljevic/tt-metal"
if TT_METAL_ROOT not in sys.path:
    sys.path.insert(0, TT_METAL_ROOT)

from models.autoports.meta_llama_llama_3_2_1b_instruct.tests.test_functional_decoder import (
    DecodeRotaryHelper,
    LAYER_IDX,
    _make_page_table,
    _synthetic_layer_state_dict,
    _to_tt_decode,
)
from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.optimized_decoder import (
    OptimizedDecoder,
    precision_policy_from_config,
)

BATCH = 32
MAX_SEQ_LEN = 256
PAGE_BLOCK = 64
INCUMBENT_PATH = Path(__file__).with_name("incumbent.json")
_DECODER = None
_KWARGS = None


def _config() -> LlamaConfig:
    return LlamaConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        attention_bias=False,
        mlp_bias=False,
    )


def _capture_safe_mlp_decode(self, hidden_states):
    """Trace-equivalent MLP path using declared configs, never tensor queries.

    The shipped implementation conditionally avoids two redundant conversions
    by querying ``tensor.memory_config()``. Layout is intentionally unknown to
    the advisor while it is solving the graph, so the capture target spells out
    the already-guaranteed phase contracts instead. This is the exact op path
    used for the successful capture; it is not a candidate decoder change.
    """
    self.load_device_weights()
    cfg = self.config
    gate = ttnn.linear(
        hidden_states,
        self.gate_weight,
        dtype=cfg.linear_dtype,
        compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
        program_config=cfg.decode_w1_w3_prg_config,
        memory_config=cfg.decode_w1_w3_output_memcfg,
    )
    up = ttnn.linear(
        hidden_states,
        self.up_weight,
        dtype=cfg.linear_dtype,
        compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
        program_config=cfg.decode_w1_w3_prg_config,
        memory_config=cfg.decode_w1_w3_output_memcfg,
    )
    ttnn.deallocate(hidden_states)
    fused = ttnn.mul(
        gate,
        up,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=cfg.mul_dtype,
        memory_config=cfg.decode_w1_w3_output_memcfg,
    )
    ttnn.deallocate(gate)
    ttnn.deallocate(up)
    out = ttnn.linear(
        fused,
        self.down_weight,
        dtype=cfg.linear_dtype,
        compute_kernel_config=cfg.ff2_compute_kernel_cfg,
        program_config=cfg.decode_w2_prg_config,
        memory_config=cfg.decode_residual_memcfg,
    )
    ttnn.deallocate(fused)
    return out


def _build(device):
    cfg = _config()
    incumbent = json.loads(INCUMBENT_PATH.read_text())
    if int(incumbent["decode_batch"]) != BATCH or int(incumbent["requested_decode_batch"]) != BATCH:
        raise RuntimeError("capture batch must match the frozen and requested incumbent batch")
    if "constructor_default" in incumbent["shipped_policy_source"].lower():
        raise RuntimeError("refusing to capture a constructor-default policy")
    policy = precision_policy_from_config(incumbent["shipped_policy"])
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_layer_state_dict(cfg),
        hf_config=cfg,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        page_block_size=PAGE_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=BATCH,
        precision_policy=policy,
    )
    decoder.mlp.decode_forward = MethodType(_capture_safe_mlp_decode, decoder.mlp)
    host_hidden = torch.randn(1, BATCH, cfg.hidden_size, dtype=torch.bfloat16)
    hidden = _to_tt_decode(host_hidden, decoder, device)
    host_pos = torch.full((BATCH,), 128, dtype=torch.int32)
    current_pos = ttnn.from_torch(
        host_pos,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    _, page_table = _make_page_table(
        device, batch=BATCH, max_seq_len=MAX_SEQ_LEN, block_size=PAGE_BLOCK, seed=3202
    )
    rot_mats = DecodeRotaryHelper(LlamaRotaryEmbedding(cfg), MAX_SEQ_LEN, cfg.head_dim, device).get_rot_mats(host_pos)
    return decoder, dict(current_pos=current_pos, rot_mats=rot_mats, page_table=page_table), hidden


def decode(hidden):
    return _DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, _KWARGS, hidden = _build(device)
    return (hidden,)
