# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only shard-advisor target for GPT-OSS TP4 dense decoder regions.

The production MoE uses ``ttnn.sparse_matmul``, which the advisor cannot
trace.  This target therefore captures the real per-rank packed-QKV and O
shapes plus the replicated residual, both RMSNorms, and router.  Sparse
expert layouts/programs remain an explicit device sweep.
"""

from __future__ import annotations

import os

import torch

import ttnn


class _CaptureDevice:
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


def _host_tensor(shape, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, torch_dtype=torch.bfloat16):
    return ttnn.from_torch(torch.empty(shape, dtype=torch_dtype), layout=layout, dtype=dtype)


def make_inputs(device):
    del device
    hidden = 2880
    local_attention_width = 1280
    local_attended_width = 1024
    experts = 32
    return (
        _host_tensor((1, 1, 1, hidden)),
        _host_tensor((hidden,)),
        _host_tensor((hidden, local_attention_width)),
        _host_tensor((1, 1, local_attention_width)),
        _host_tensor((1, 1, 1, local_attended_width)),
        _host_tensor((local_attended_width, hidden)),
        _host_tensor((1, 1, hidden)),
        _host_tensor((hidden,)),
        _host_tensor((hidden, experts)),
        _host_tensor((1, experts), dtype=ttnn.float32, torch_dtype=torch.float32),
    )


def decode(
    residual,
    input_norm_weight,
    qkv_weight,
    qkv_bias,
    attended,
    o_weight,
    o_bias,
    post_norm_weight,
    router_weight,
    router_bias,
):
    normalized = ttnn.rms_norm(residual, epsilon=1e-5, weight=input_norm_weight)
    qkv = ttnn.linear(normalized, qkv_weight, bias=qkv_bias, dtype=ttnn.bfloat16)
    projected = ttnn.linear(attended, o_weight, bias=o_bias, dtype=ttnn.bfloat16)
    # The advisor CLI accepts a single output.  Preserve the otherwise
    # independent QKV branch in the captured graph with a scalar marker whose
    # broadcast is irrelevant to layout selection.
    qkv_marker = ttnn.sum(qkv, dim=-1, keepdim=True)
    projected = ttnn.add(projected, qkv_marker)
    attention_residual = ttnn.add(residual, projected, dtype=ttnn.bfloat16)
    post_normalized = ttnn.rms_norm(attention_residual, epsilon=1e-5, weight=post_norm_weight)
    router_input = ttnn.typecast(ttnn.reshape(post_normalized, [1, 2880]), ttnn.float32)
    router_logits = ttnn.linear(
        router_input,
        router_weight,
        bias=router_bias,
        dtype=ttnn.float32,
    )
    return router_logits
