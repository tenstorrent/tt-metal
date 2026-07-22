# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only shard-advisor target for the rewritten dense decoder graph.

The runtime-selected batch-one MoE uses ``ttnn.sparse_matmul``, for which this
advisor revision has no TTIR operation.  OPT-015 therefore captures the same
optimized decoder with its dense expert path selected.  The resulting advice
is the required first layout/program candidate for attention, residuals,
router, and the dense gate/up/down reference; sparse-expert grids are searched
separately on device.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch

import ttnn

TT_METAL_ROOT = Path("/home/mvasiljevic/tt-metal")
BATCH = 1
MAX_CACHE_LEN = 128
CURRENT_POS = 17


class _CaptureDevice:
    """Shape-only device used by interception capture."""

    @staticmethod
    def get_num_devices():
        return 1

    @staticmethod
    def compute_with_storage_grid_size():
        return ttnn.CoreCoord(11, 10)

    @staticmethod
    def dram_grid_size():
        return ttnn.CoreCoord(8, 1)

    @staticmethod
    def arch():
        return ttnn.Arch.BLACKHOLE


_CAPTURE_DEVICE = _CaptureDevice()
_HOST_FROM_TORCH = ttnn.from_torch


class _InputSpec:
    """Tracer metadata input that avoids host-side packed-dtype conversion."""

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


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
        # SETUP.md Part B requires append so the source checkout does not
        # shadow the advisor virtual environment's ttnn package.
        sys.path.append(text)


def _install_capture_handlers():
    """Bridge runtime spellings missing from the advisor branch."""

    from ttnn_jit._src import interception_tracer as tracer
    from ttnn_jit._src.jit_functions import PermuteOpHandler, ReductionOpHandler

    original_permute = PermuteOpHandler.create_operation

    def permute_handler(self, *args, **kwargs):
        if len(args) >= 2 and "permutation" not in kwargs:
            kwargs["permutation"] = args[1]
            args = args[:1]
        return original_permute(self, *args, **kwargs)

    PermuteOpHandler.create_operation = permute_handler

    original_reduction = ReductionOpHandler.create_operation

    def reduction_handler(self, *args, **kwargs):
        # Runtime accepts sum(x, [0], False); this advisor handler otherwise
        # reads only keyword dimensions and incorrectly models a scalar.
        if len(args) >= 2 and "dim" not in kwargs:
            dim = args[1]
            kwargs["dim"] = dim[0] if isinstance(dim, (list, tuple)) and len(dim) == 1 else dim
        if len(args) >= 3 and "keepdim" not in kwargs:
            kwargs["keepdim"] = args[2]
        return original_reduction(self, args[0], **kwargs)

    ReductionOpHandler.create_operation = reduction_handler

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

    tracer._EXPERIMENTAL_VALUE["nlp_concat_heads_decode"] = concat_decode_handler

    def paged_update_cache_handler(jit_ctx, cache, input, *, update_idxs_tensor=None, page_table=None, **kwargs):
        # This advisor branch can trace ttir.paged_update_cache but cannot
        # legalize it to TTNN in either scoped or full pipelines.  Cache update
        # has no tunable layout choice, so thread the selected BFP8 cache value
        # through unchanged and keep SDPA plus the complete dense block in the
        # graph.  Runtime cache semantics are covered by the optimized tests.
        del jit_ctx, input, update_idxs_tensor, page_table, kwargs
        return cache.mlir_value

    tracer._EXPERIMENTAL_VALUE["paged_update_cache"] = paged_update_cache_handler

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
    return ttnn.from_torch(torch.empty(shape, dtype=torch_dtype), layout=layout, dtype=dtype)


_DECODER = None


def _install_transformers_stubs():
    """Satisfy functional-decoder imports without adding HF to the advisor env.

    Capture constructs already-prepared tensors directly, so neither symbol is
    called.  Keeping the advisor venv independent of Transformers also follows
    the host-capture recipe used by earlier autoports.
    """

    transformers = types.ModuleType("transformers")
    integrations = types.ModuleType("transformers.integrations")
    mxfp4 = types.ModuleType("transformers.integrations.mxfp4")
    models = types.ModuleType("transformers.models")
    gpt_oss = types.ModuleType("transformers.models.gpt_oss")
    modeling = types.ModuleType("transformers.models.gpt_oss.modeling_gpt_oss")

    def unavailable_conversion(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("MXFP4 conversion is unavailable in host-only advisor capture")

    class UnavailableRotaryEmbedding:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("HF RoPE construction is unavailable in host-only advisor capture")

    mxfp4.convert_moe_packed_tensors = unavailable_conversion
    modeling.GptOssRotaryEmbedding = UnavailableRotaryEmbedding
    modules = {
        "transformers": transformers,
        "transformers.integrations": integrations,
        "transformers.integrations.mxfp4": mxfp4,
        "transformers.models": models,
        "transformers.models.gpt_oss": gpt_oss,
        "transformers.models.gpt_oss.modeling_gpt_oss": modeling,
    }
    sys.modules.update(modules)


def make_inputs(device):
    del device
    global _DECODER

    _install_capture_handlers()
    _install_transformers_stubs()
    from models.autoports.openai_gpt_oss_20b.tt.optimized_decoder import OptimizationConfig, OptimizedDecoder

    hidden = 2880
    num_heads = 64
    num_kv_heads = 8
    head_dim = 64
    intermediate = 2880
    experts = 32
    q_width = num_heads * head_dim
    kv_width = num_kv_heads * head_dim
    norm_shape = (1, 1, hidden // ttnn.TILE_SIZE, ttnn.TILE_SIZE)

    # Construction-time constant folding in FusedDecoder performs device ops.
    # Interception starts only after make_inputs returns, so create the object
    # without __init__ and populate its already-folded runtime state directly.
    _DECODER = OptimizedDecoder.__new__(OptimizedDecoder)
    attributes = {
        "mesh_device": _CAPTURE_DEVICE,
        "layer_idx": 12,
        "batch": BATCH,
        "max_cache_len": MAX_CACHE_LEN,
        "hidden_size": hidden,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate,
        "num_experts": experts,
        "experts_per_token": 4,
        "rms_norm_eps": 1e-5,
        "sliding_window": 128,
        "attention_window": 128,
        "swiglu_limit": 7.0,
        "scale": head_dim**-0.5,
        "compute_kernel_config": None,
        "input_norm": _host_tensor(norm_shape, layout=ttnn.ROW_MAJOR_LAYOUT),
        "post_attention_norm": _host_tensor(norm_shape, layout=ttnn.ROW_MAJOR_LAYOUT),
        "qkv_weight": _host_tensor((hidden, q_width + 2 * kv_width)),
        "qkv_bias": _host_tensor((1, 1, q_width + 2 * kv_width)),
        "output_weight": _host_tensor((q_width, hidden)),
        "output_bias": _host_tensor((1, 1, hidden)),
        "attention_sinks": _host_tensor((BATCH, num_heads, MAX_CACHE_LEN, 1)),
        "decode_attention_sinks": _host_tensor((num_heads, ttnn.TILE_SIZE)),
        "router_weight": _host_tensor((hidden, experts)),
        "router_bias": _host_tensor((1, experts), dtype=ttnn.float32, torch_dtype=torch.float32),
        "gate_up_weight": _host_tensor((experts, hidden, 2 * intermediate)),
        "gate_up_bias": _host_tensor((experts, 1, 2 * intermediate)),
        "down_weight": _host_tensor((experts, intermediate, hidden)),
        "down_bias": _host_tensor((experts, 1, hidden)),
        "rotary_cos": _host_tensor((1, 1, MAX_CACHE_LEN, head_dim)),
        "rotary_sin": _host_tensor((1, 1, MAX_CACHE_LEN, head_dim)),
        "attention_mask": None,
        "position_indices": _host_tensor(
            (MAX_CACHE_LEN,),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            torch_dtype=torch.int32,
        ),
        "decode_heads_mem_config": ttnn.create_sharded_memory_config(
            shape=(64, head_dim),
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
        "prefill_rotary_views": {},
        "decode_position_views": {},
        "moe_policy": "auto",
        # Capture from the rewritten dense graph before applying the advisor's
        # own layouts.  This both avoids seeding the capture with its previous
        # answer and keeps the harness reproducible when OptimizedDecoder's
        # production defaults enable advisor layouts.
        "optimization_config": OptimizationConfig(
            use_sparse_experts=False,
            use_shard_advisor_attention_layouts=False,
            use_shard_advisor_router_layouts=False,
        ),
        "experts": None,
    }
    for name, value in attributes.items():
        setattr(_DECODER, name, value)
    # __init__ was intentionally bypassed above. Recreate the non-device
    # configuration state required by the owned decode path; these helpers
    # only build memory/program/compute descriptors during make_inputs.
    _DECODER._configure_dram_attention_candidate()
    _DECODER._configure_attention_program_candidates()
    return (
        _host_tensor((1, BATCH, 1, hidden)),
        _InputSpec((BATCH, num_kv_heads, MAX_CACHE_LEN, head_dim), ttnn.bfloat8_b),
        _InputSpec((BATCH, num_kv_heads, MAX_CACHE_LEN, head_dim), ttnn.bfloat8_b),
    )


def decode(hidden, key_cache, value_cache):
    return _DECODER.decode_forward(
        hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        current_pos=CURRENT_POS,
    )
