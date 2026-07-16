# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shard-advisor capture target for the rewritten dense Llama decoder block."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

import ttnn

TT_METAL_ROOT = Path("/home/mvasiljevic/tt-metal")
MODEL_CONFIG_DIR = TT_METAL_ROOT / "models/tt_transformers/model_params/Llama-3.1-8B-Instruct"
LAYER_IDX = 16
BATCH = 32
MAX_CACHE_LEN = 128
CURRENT_POS = 18


class _CaptureDevice:
    """Shape-only mesh used by the advisor's interception tracer.

    The checked-out advisor bundles a TT-Metal/UMD revision which cannot map
    sysmem on this host's current Blackhole driver.  Capture does not execute
    model ops: the optimizer queries SYSTEM_DESC_PATH after interception.  A
    host-resident tensor plus the advisor's documented 8x8 grid therefore
    preserves the shapes/layouts it needs without mixing UMD revisions.
    """

    @staticmethod
    def get_num_devices():
        return 1

    @staticmethod
    def compute_with_storage_grid_size():
        return ttnn.CoreCoord(8, 8)

    @staticmethod
    def dram_grid_size():
        return ttnn.CoreCoord(8, 1)


_CAPTURE_DEVICE = _CaptureDevice()
_HOST_FROM_TORCH = ttnn.from_torch


def _from_torch_without_silicon(tensor, *args, device=None, **kwargs):
    del device
    return _HOST_FROM_TORCH(tensor, *args, **kwargs)


# ttnn-advise imports this module before it opens its setup device.  Keep all
# capture objects host-resident; the interception tracer replaces runtime ops.
ttnn.open_mesh_device = lambda *args, **kwargs: _CAPTURE_DEVICE
ttnn.close_mesh_device = lambda *args, **kwargs: None
ttnn.from_torch = _from_torch_without_silicon


def _install_host_advisor_runner():
    """Use the existing system descriptor when capture inputs are host tensors."""

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
            raise ValueError("The capture-only host runner supports ttnn/interception tracers")
        return self._advise_ir(ir, self.func.__name__, system_desc_path)

    ShardAdvisor.run = run


_install_host_advisor_runner()

# SETUP.md Part B requires appending so the advisor environment's installed
# ttnn package is not shadowed by tt-metal's source directory.
for path in (TT_METAL_ROOT.parent, TT_METAL_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.append(text)


def _synthetic_state(config):
    generator = torch.Generator().manual_seed(20260716)
    prefix = f"model.layers.{LAYER_IDX}."
    head_dim = config.hidden_size // config.num_attention_heads
    kv_width = config.num_key_value_heads * head_dim

    def normal(shape, scale=0.02):
        tensor = torch.empty(shape, dtype=torch.bfloat16)
        return tensor.normal_(mean=0.0, std=scale, generator=generator)

    return {
        prefix + "input_layernorm.weight": 1.0 + normal((config.hidden_size,), 0.01),
        prefix + "post_attention_layernorm.weight": 1.0 + normal((config.hidden_size,), 0.01),
        prefix + "self_attn.q_proj.weight": normal((config.hidden_size, config.hidden_size)),
        prefix + "self_attn.k_proj.weight": normal((kv_width, config.hidden_size)),
        prefix + "self_attn.v_proj.weight": normal((kv_width, config.hidden_size)),
        prefix + "self_attn.o_proj.weight": normal((config.hidden_size, config.hidden_size)),
        prefix + "mlp.gate_proj.weight": normal((config.intermediate_size, config.hidden_size)),
        prefix + "mlp.up_proj.weight": normal((config.intermediate_size, config.hidden_size)),
        prefix + "mlp.down_proj.weight": normal((config.hidden_size, config.intermediate_size)),
    }


def _to_device(tensor, device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor,
        device=device,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


_DECODER = None


def _install_generic_rotary_handler():
    """Bridge the advisor's missing generic-RoPE interception handler.

    The advisor branch models ``rotary_embedding_llama`` but not the generic
    rotate-half op used by the IR-derived decoder.  Layout advice only needs
    the shape/dtype dependency, so emit an identity-shape TTIR reshape.  This
    follows SETUP.md A.3 without mutating the external tt-mlir checkout.
    """

    from ttnn_jit._src import interception_tracer as tracer
    from ttnn_jit._src import ttnn_emit_tracer as emit_tracer

    def handler(jit_ctx, input, cos_cache, sin_cache, token_index=None, **kwargs):
        del cos_cache, sin_cache, token_index, kwargs
        tensor_type = input.mlir_value.type
        shape = [int(dimension) for dimension in tensor_type.shape]
        with tracer.InsertionPoint(jit_ctx.func_bb), tracer.Location.unknown(jit_ctx.ctx):
            result_type = tracer.RankedTensorType.get(shape, tensor_type.element_type)
            return tracer.ttir.reshape(
                result=result_type,
                input=input.mlir_value,
                shape=shape,
            )

    tracer._EXPERIMENTAL_VALUE["rotary_embedding"] = handler

    def emit_handler(jit_ctx, input, cos_cache, sin_cache, token_index=None, **kwargs):
        del kwargs
        shape = [int(dimension) for dimension in input.mlir_value.type.shape]
        with emit_tracer.InsertionPoint(jit_ctx.func_bb), emit_tracer.Location.unknown(jit_ctx.ctx):
            result_type = emit_tracer._retype(jit_ctx.ctx, input.mlir_value, shape)
            return emit_tracer.ttnn.rotary_embedding(
                result=result_type,
                input=input.mlir_value,
                cos_cache=cos_cache.mlir_value,
                sin_cache=sin_cache.mlir_value,
                token_index=token_index,
            )

    emit_tracer._EXPERIMENTAL_VALUE["rotary_embedding"] = emit_handler

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
        shape = [int(dimension) for dimension in q.mlir_value.type.shape]
        with emit_tracer.InsertionPoint(jit_ctx.func_bb), emit_tracer.Location.unknown(jit_ctx.ctx):
            result_type = emit_tracer._retype(jit_ctx.ctx, q.mlir_value, shape)
            return emit_tracer.ttnn.scaled_dot_product_attention_decode(
                result=result_type,
                query=q.mlir_value,
                key=k.mlir_value,
                value=v.mlir_value,
                is_causal=is_causal,
                cur_pos_tensor=(cur_pos_tensor.mlir_value if cur_pos_tensor is not None else None),
                scale=scale,
            )

    emit_tracer._TRANSFORMER_VALUE["scaled_dot_product_attention_decode"] = sdpa_decode_handler

    def ttir_sdpa_decode_handler(
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

    tracer._TRANSFORMER_VALUE["scaled_dot_product_attention_decode"] = ttir_sdpa_decode_handler


def make_inputs(device):
    global _DECODER

    _install_generic_rotary_handler()

    from transformers import AutoConfig

    from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (
        OptimizationConfig,
        OptimizedDecoder,
    )

    config = AutoConfig.from_pretrained(MODEL_CONFIG_DIR, local_files_only=True)
    _DECODER = OptimizedDecoder.from_state_dict(
        _synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        max_cache_len=MAX_CACHE_LEN,
        # Host BFP packing in the advisor's bundled UMD initializes silicon.
        # Dtype is not an advisor decision, so use BF16 for capture while
        # preserving the exact rewritten operation and layout topology.
        optimization_config=OptimizationConfig(
            attention_weight_dtype=ttnn.bfloat16,
            gate_up_weight_dtype=ttnn.bfloat16,
            down_weight_dtype=ttnn.bfloat16,
        ),
    )
    head_dim = config.hidden_size // config.num_attention_heads
    cache_shape = (BATCH, config.num_key_value_heads, MAX_CACHE_LEN, head_dim)
    hidden = _to_device(
        torch.randn((1, BATCH, 1, config.hidden_size), dtype=torch.bfloat16),
        device,
    )
    key_cache = _to_device(torch.zeros(cache_shape, dtype=torch.bfloat16), device)
    value_cache = _to_device(torch.zeros(cache_shape, dtype=torch.bfloat16), device)
    return hidden, key_cache, value_cache


def decode(hidden, key_cache, value_cache):
    return _DECODER.decode_forward(
        hidden,
        key_cache,
        value_cache,
        current_pos=CURRENT_POS,
    )
