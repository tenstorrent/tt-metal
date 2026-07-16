# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shard-advisor target for the rewritten dense GPT-OSS decoder block.

The runtime optimized path will replace dense experts with sparse matmuls after
this mandatory dense-path layout seed.  The advisor intentionally traces the
rewritten packed-attention plus dense MoE graph because tt-mlir has no TTIR op
for ``ttnn.sparse_matmul``.  It therefore owns the attention/router/residual
layout seed and provides dense expert configs to measure as the required first
candidate; sparse-expert geometry remains a separate local search.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn

TT_METAL_ROOT = Path("/home/mvasiljevic/tt-metal")
LAYER_IDX = 12
BATCH = 1
MAX_CACHE_LEN = 128
CURRENT_POS = 17


class _CaptureDevice:
    """Shape-only device for interception capture; no model op executes."""

    @staticmethod
    def get_num_devices():
        return 1

    @staticmethod
    def compute_with_storage_grid_size():
        # The reserved Blackhole reports 11x10.
        return ttnn.CoreCoord(11, 10)

    @staticmethod
    def dram_grid_size():
        return ttnn.CoreCoord(8, 1)

    @staticmethod
    def arch():
        return ttnn.Arch.BLACKHOLE


_CAPTURE_DEVICE = _CaptureDevice()
_HOST_FROM_TORCH = ttnn.from_torch


def _from_torch_without_silicon(tensor, *args, device=None, mesh_mapper=None, memory_config=None, **kwargs):
    del device, mesh_mapper, memory_config
    return _HOST_FROM_TORCH(tensor, *args, **kwargs)


ttnn.open_mesh_device = lambda *args, **kwargs: _CAPTURE_DEVICE
ttnn.close_mesh_device = lambda *args, **kwargs: None
ttnn.from_torch = _from_torch_without_silicon


def _install_host_advisor_runner():
    from ttnn_jit._src.shard_advisor import ShardAdvisor

    def run(self, *args, **kwargs):
        del kwargs
        system_desc_path = os.environ["SYSTEM_DESC_PATH"]
        if self.tracer == "ttnn":
            from ttnn_jit._src.ttnn_emit_tracer import trace_ttnn

            ir, _output_type = trace_ttnn(self.func, *args)
        elif self.tracer == "interception":
            from ttnn_jit._src.interception_tracer import trace_intercepted

            ir, _output_type = trace_intercepted(self.func, *args)
        else:
            raise ValueError("host capture supports ttnn/interception tracers")
        return self._advise_ir(ir, self.func.__name__, system_desc_path)

    ShardAdvisor.run = run


_install_host_advisor_runner()

for path in (TT_METAL_ROOT.parent, TT_METAL_ROOT):
    text = str(path)
    if text not in sys.path:
        # SETUP.md Part B requires append, not prepend, so tt-metal's source
        # directory cannot shadow the advisor environment's ttnn package.
        sys.path.append(text)


def _install_capture_handlers():
    """Bridge generic RoPE and decode-SDPA handlers in the advisor branch."""

    from ttnn_jit._src import interception_tracer as tracer
    from ttnn_jit._src.jit_functions import PermuteOpHandler

    original_permute = PermuteOpHandler.create_operation

    def permute_handler(self, *args, **kwargs):
        # The runtime API accepts ``ttnn.permute(x, dims)`` while this advisor
        # branch's generic handler only accepts the keyword spelling.
        if len(args) >= 2 and "permutation" not in kwargs:
            kwargs["permutation"] = args[1]
            args = args[:1]
        return original_permute(self, *args, **kwargs)

    PermuteOpHandler.create_operation = permute_handler

    def rotary_handler(jit_ctx, input, cos_cache, sin_cache, token_index=None, **kwargs):
        del cos_cache, sin_cache, token_index, kwargs
        tensor_type = input.mlir_value.type
        shape = [int(dimension) for dimension in tensor_type.shape]
        with tracer.InsertionPoint(jit_ctx.func_bb), tracer.Location.unknown(jit_ctx.ctx):
            result_type = tracer.RankedTensorType.get(shape, tensor_type.element_type)
            return tracer.ttir.reshape(result=result_type, input=input.mlir_value, shape=shape)

    tracer._EXPERIMENTAL_VALUE["rotary_embedding"] = rotary_handler

    def concat_decode_handler(jit_ctx, input, *, num_heads=None, **kwargs):
        del num_heads, kwargs
        tensor_type = input.mlir_value.type
        shape = [int(dimension) for dimension in tensor_type.shape]
        batch, heads, head_dim = shape[-3:]
        with tracer.InsertionPoint(jit_ctx.func_bb), tracer.Location.unknown(jit_ctx.ctx):
            result_shape = [1, 1, batch, heads * head_dim]
            result_type = tracer.RankedTensorType.get(result_shape, tensor_type.element_type)
            return tracer.ttir.reshape(result=result_type, input=input.mlir_value, shape=result_shape)

    # The current runtime returns rank four (with a tile-padded batch axis),
    # whereas this advisor branch's stock handler models a rank-three result.
    tracer._EXPERIMENTAL_VALUE["nlp_concat_heads_decode"] = concat_decode_handler

    def sdpa_decode_handler(
        jit_ctx,
        q,
        k,
        v,
        *,
        cur_pos_tensor=None,
        is_causal=True,
        scale=None,
        **kwargs,
    ):
        del kwargs
        tensor_type = q.mlir_value.type
        shape = [int(dimension) for dimension in tensor_type.shape]
        with tracer.InsertionPoint(jit_ctx.func_bb), tracer.Location.unknown(jit_ctx.ctx):
            result_type = tracer.RankedTensorType.get(shape, tensor_type.element_type)
            return tracer.ttir.scaled_dot_product_attention_decode(
                result=result_type,
                query=q.mlir_value,
                key=k.mlir_value,
                value=v.mlir_value,
                cur_pos_tensor=(cur_pos_tensor.mlir_value if cur_pos_tensor is not None else None),
                is_causal=is_causal,
                scale=scale,
            )

    tracer._TRANSFORMER_VALUE["scaled_dot_product_attention_decode"] = sdpa_decode_handler


def _host_tensor(shape, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, torch_dtype=torch.bfloat16):
    # Empty values are sufficient: the interceptor needs shapes/dtypes only.
    return ttnn.from_torch(torch.empty(shape, dtype=torch_dtype), layout=layout, dtype=dtype)


_DECODER = None


def make_inputs(device):
    del device
    global _DECODER

    _install_capture_handlers()
    from models.autoports.openai_gpt_oss_20b.tt.optimized_decoder import OptimizationConfig, OptimizedDecoder

    # Keep the isolated advisor venv independent of the runtime's Transformers
    # installation.  These are the checked-in gpt-oss-20b config dimensions.
    config = SimpleNamespace(
        hidden_size=2880,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        intermediate_size=2880,
        num_local_experts=32,
        num_experts_per_tok=4,
        rope_theta=150000.0,
        rms_norm_eps=1e-5,
    )
    hidden = int(config.hidden_size)
    q_width = int(config.num_attention_heads) * int(config.head_dim)
    kv_width = int(config.num_key_value_heads) * int(config.head_dim)
    intermediate = int(config.intermediate_size)
    experts = int(config.num_local_experts)
    norm_shape = (1, 1, hidden // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
    weights = {
        "input_norm": _host_tensor(norm_shape, layout=ttnn.ROW_MAJOR_LAYOUT),
        "post_attention_norm": _host_tensor(norm_shape, layout=ttnn.ROW_MAJOR_LAYOUT),
        "qkv_weight": _host_tensor((hidden, q_width + 2 * kv_width)),
        "qkv_bias": _host_tensor((1, 1, q_width + 2 * kv_width)),
        "o_weight": _host_tensor((q_width, hidden)),
        "o_bias": _host_tensor((1, 1, hidden)),
        "prefill_sinks": _host_tensor((1, int(config.num_attention_heads), 1, 1)),
        "decode_sinks": _host_tensor((int(config.num_attention_heads), ttnn.TILE_SIZE)),
        "router_weight": _host_tensor((hidden, experts)),
        "router_bias": _host_tensor(
            (1, experts),
            dtype=ttnn.float32,
            torch_dtype=torch.float32,
        ),
        "gate_up_weight": _host_tensor((experts, hidden, 2 * intermediate)),
        "gate_up_bias": _host_tensor((experts, 1, 2 * intermediate)),
        "down_weight": _host_tensor((experts, intermediate, hidden)),
        "down_bias": _host_tensor((experts, 1, hidden)),
    }
    _DECODER = OptimizedDecoder(
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=_CAPTURE_DEVICE,
        batch=BATCH,
        max_cache_len=MAX_CACHE_LEN,
        weights=weights,
        cos_cache=_host_tensor((1, 1, MAX_CACHE_LEN, int(config.head_dim))),
        sin_cache=_host_tensor((1, 1, MAX_CACHE_LEN, int(config.head_dim))),
        optimization_config=OptimizationConfig(
            use_shard_advisor_layouts=False,
            use_sparse_experts=False,
            explicit_sdpa_program_config=False,
        ),
    )
    return (
        _host_tensor((1, BATCH, 1, hidden)),
        _host_tensor((BATCH, int(config.num_key_value_heads), MAX_CACHE_LEN, int(config.head_dim))),
        _host_tensor((BATCH, int(config.num_key_value_heads), MAX_CACHE_LEN, int(config.head_dim))),
        _host_tensor(
            (BATCH,),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            torch_dtype=torch.int32,
        ),
    )


def decode(hidden, key_cache, value_cache, position):
    return _DECODER.decode_forward(
        hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=CURRENT_POS,
        cache_position_tensor=position,
    )
