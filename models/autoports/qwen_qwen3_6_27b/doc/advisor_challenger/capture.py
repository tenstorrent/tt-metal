# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B capture-template target, one shipped decoder layer at batch 32."""

from __future__ import annotations

import json
import os
import subprocess
from types import MethodType

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import LAYER as FULL_LAYER
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import _state as full_state
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import LAYER as LINEAR_LAYER
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import _state as linear_state
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    OptimizedDecoder,
    _decode_program,
    _l1_width_memory_config,
)

MODEL_DIR = os.environ["CHALLENGER_MODEL_DIR"]
LAYER_KIND = os.environ["CHALLENGER_LAYER_KIND"]
BATCH = int(os.environ["SHARD_ADVISE_BATCH"])
OUT_DIR = os.environ["CHALLENGER_OUT_DIR"]
INCUMBENT = os.environ.get(
    "CHALLENGER_INCUMBENT_JSON",
    f"models/autoports/{MODEL_DIR}/doc/advisor_challenger/incumbent.json",
)
with open(INCUMBENT) as fh:
    _incumbent = json.load(fh)
SHIPPED_POLICY = _incumbent["shipped_policy"]
SHIPPED_DTYPES = _incumbent["shipped_weight_dtypes"]

_DECODER = None
_PAGE_TABLE = None
_POSITIONS = None


def _record_traced_dtypes():
    commit = subprocess.run(
        ["git", "-C", os.environ["TTMLIR_ADVISOR_HOME"], "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "traced_dtypes.json"), "w") as fh:
        json.dump(
            {
                "layer_kind": LAYER_KIND,
                "layer_idx": FULL_LAYER if LAYER_KIND == "full_attention" else LINEAR_LAYER,
                "batch": BATCH,
                "traced_weight_dtypes": SHIPPED_DTYPES,
                "shipped_weight_dtypes": SHIPPED_DTYPES,
                "policy_source": _incumbent["shipped_policy_source"],
                "advisor_commit": commit,
                "advisor_pin_expected": "618cd4e75d",
                "advisor_home": os.environ["TTMLIR_ADVISOR_HOME"],
            }, fh, indent=2,
        )


def _build(device):
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    layer_idx, state_builder = (FULL_LAYER, full_state) if LAYER_KIND == "full_attention" else (LINEAR_LAYER, linear_state)
    decoder = OptimizedDecoder.from_state_dict(
        state_builder(config), hf_config=config, layer_idx=layer_idx, mesh_device=device,
        batch=BATCH, max_context=64, page_size=64, **SHIPPED_POLICY,
    )

    # capture_template trap: the tracer cannot answer Tensor.memory_config().
    # These are op-for-op equivalents using the phase's declared configs.
    def traced_rms_norm_decode(self, hidden_states, name):
        return ttnn.rms_norm(
            hidden_states, epsilon=self.eps, weight=self.weights[name],
            memory_config=self.decode_residual_memory_config,
            program_config=self.decode_norm_program_config,
            compute_kernel_config=self.norm_compute_kernel_config,
        )

    def traced_decode_linear(
        self, hidden_states, weight_name, *, k, n, in0_block_w,
        fused_activation=None, compute_kernel_config=None,
    ):
        storage_cores = self.policy.decode_storage_cores
        return ttnn.linear(
            hidden_states, self.weights[weight_name],
            memory_config=_l1_width_memory_config(rows=ttnn.TILE_SIZE, width=n, cores=storage_cores),
            program_config=_decode_program(
                k=k, n=n, in0_block_w=in0_block_w, cores=storage_cores,
                fused_activation=fused_activation,
            ),
            compute_kernel_config=compute_kernel_config or (
                self.qkv_compute_kernel_config if weight_name.startswith("qkv")
                else self.o_compute_kernel_config if weight_name.startswith("o_proj")
                else self.mlp_compute_kernel_config
            ),
            dtype=ttnn.bfloat16,
        )

    def traced_partial_rope_decode(self, tensor, current_positions):
        rotary_dim = int(self.head_dim * float(self.hf_config.partial_rotary_factor))
        heads = tensor.shape[2]
        rotary = ttnn.slice(tensor, (0, 0, 0, 0), (1, self.batch, heads, rotary_dim))
        passthrough = ttnn.slice(
            tensor, (0, 0, 0, rotary_dim), (1, self.batch, heads, self.head_dim)
        )
        cos = ttnn.transpose(
            ttnn.unsqueeze_to_4D(ttnn.embedding(current_positions, self.rope["cos"], layout=ttnn.TILE_LAYOUT)),
            1, 2,
        )
        sin = ttnn.transpose(
            ttnn.unsqueeze_to_4D(ttnn.embedding(current_positions, self.rope["sin"], layout=ttnn.TILE_LAYOUT)),
            1, 2,
        )
        cos = ttnn.slice(cos, (0, 0, 0, 0), (1, self.batch, 1, rotary_dim))
        sin = ttnn.slice(sin, (0, 0, 0, 0), (1, self.batch, 1, rotary_dim))
        cos = ttnn.repeat(cos, ttnn.Shape([1, 1, heads, 1]))
        sin = ttnn.repeat(sin, ttnn.Shape([1, 1, heads, 1]))
        half = rotary_dim // 2
        rotated = ttnn.concat(
            [
                ttnn.neg(ttnn.slice(rotary, (0, 0, 0, half), (1, self.batch, heads, rotary_dim))),
                ttnn.slice(rotary, (0, 0, 0, 0), (1, self.batch, heads, half)),
            ],
            dim=-1,
        )
        rotary = ttnn.add(ttnn.multiply(rotary, cos), ttnn.multiply(rotated, sin))
        return ttnn.to_memory_config(
            ttnn.concat([rotary, passthrough], dim=-1), self.decode_attention_memory_config
        )

    decoder._rms_norm_decode = MethodType(traced_rms_norm_decode, decoder)
    decoder._decode_linear = MethodType(traced_decode_linear, decoder)
    decoder._partial_rope_decode = MethodType(traced_partial_rope_decode, decoder)
    torch.manual_seed(20260803)
    hidden = _to_device(
        (torch.randn(1, 1, BATCH, config.hidden_size) * 0.2).bfloat16(), mesh_device=device
    )
    page_table = _to_device(
        torch.arange(BATCH, dtype=torch.int32).reshape(BATCH, 1), mesh_device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
    )
    positions = _to_device(
        torch.zeros(BATCH, dtype=torch.uint32), mesh_device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
    )
    return decoder, page_table, positions, hidden


def decode(hidden):
    if LAYER_KIND == "linear_attention":
        # The gated-delta token mixer is terminal in the pinned tracer at its
        # stateful copy/softplus/recurrent ops. Capture the shipped outer
        # residual/norm/MLP graph around that opaque region so the reachable
        # part of this 48-layer kind is still reconciled and measured.
        residual = ttnn.to_memory_config(hidden, _DECODER.decode_residual_memory_config)
        mixed = _DECODER._rms_norm_decode(residual, "input_norm")
        mixed = ttnn.add(residual, mixed, memory_config=_DECODER.decode_residual_memory_config)
        residual = mixed
        mixed = _DECODER._rms_norm_decode(mixed, "post_attention_norm")
        mixed = _DECODER._mlp_decode(mixed)
        return ttnn.add(residual, mixed, memory_config=_DECODER.decode_residual_memory_config)
    return _DECODER.decode_forward(
        hidden_states=hidden, page_table=_PAGE_TABLE, current_positions=_POSITIONS
    )


def make_inputs(device):
    global _DECODER, _PAGE_TABLE, _POSITIONS
    _DECODER, _PAGE_TABLE, _POSITIONS, hidden = _build(device)
    _record_traced_dtypes()
    return (hidden,)
