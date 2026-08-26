# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware correctness tests for the common WH Galaxy MLP2D.

The resource plan, ring core map, DRAM-sharded weight layout and HF reference
live in `models.common.tests.modules._mlp_2d_galaxy` so the Prefetcher2D
hardware suite can drive the identical payload; only the test bodies are here.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.modules.mlp.mlp_2d import MLP2D, MLP2DConfig, _load_input_device_tensor
from models.common.tests.modules._hf_reference import get_mlp_weights_from_ref_model
from models.common.tests.modules._mlp_2d_galaxy import assert_mlp_pcc as _assert_pcc
from models.common.tests.modules._mlp_2d_galaxy import decode_all_reduce_configs as _decode_all_reduce_configs
from models.common.tests.modules._mlp_2d_galaxy import decode_reduce_scatter_memcfg as _decode_reduce_scatter_memcfg
from models.common.tests.modules._mlp_2d_galaxy import decode_ring_config as _decode_ring_config
from models.common.tests.modules._mlp_2d_galaxy import lazy_activation as _lazy
from models.common.tests.modules._mlp_2d_galaxy import prefill_weight_lazies as _prefill_weight_lazies
from models.common.tests.modules._mlp_2d_galaxy import reference_mlp as _reference_mlp
from models.common.tests.modules._mlp_2d_galaxy import resources_config as _resources_config
from models.common.tests.modules._mlp_2d_galaxy import weight_lazies as _weight_lazies
from models.common.tests.modules._wh_galaxy_hardware import (
    compose_2d_sharded_tensor,
    deallocate_module_weights,
    deallocate_tensor,
    exact_tensor_resource,
    require_galaxy_hardware_resources,
)


def _invoke(
    module: MLP2D,
    resources,
    mesh_device: ttnn.MeshDevice,
    x: torch.Tensor,
    *,
    mode: str,
    expected: torch.Tensor,
    case: str,
) -> None:
    input_dtype = module.config.decode_activation_dtype if mode == "decode" else module.config.prefill_activation_dtype
    device_input = _load_input_device_tensor(_lazy(x, mesh_device, input_dtype), module.config, mode)
    resources.activate(mode)
    output = module(device_input, mode=mode)
    try:
        resources.synchronize(mode)
        actual = compose_2d_sharded_tensor(output, mesh_device)
        _assert_pcc(expected, actual, case=case)
    finally:
        deallocate_tensor(output)
        deallocate_tensor(device_input)


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "dim,hidden_dim",
    [(8192, 28672), (5120, 25600)],
    ids=["llama-8192x28672", "qwen-5120x25600"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@torch.no_grad()
def test_mlp_2d_wh_galaxy_decode_batch_32_repeat(mesh_device, dim, hidden_dim):
    torch.manual_seed(0)
    reference_mlp = _reference_mlp(dim, hidden_dim)
    w1, w2, w3 = get_mlp_weights_from_ref_model(reference_mlp)
    x = torch.randn(1, 1, 32, dim, dtype=torch.bfloat16)
    expected = reference_mlp(x)
    weight_dtype = ttnn.bfloat16 if dim == 5120 else ttnn.bfloat8_b
    activation_dtype = ttnn.bfloat8_b
    lazy_w1, lazy_w2, lazy_w3 = _weight_lazies(w1, w2, w3, mesh_device, weight_dtype)
    decode_ring = _decode_ring_config(dim, hidden_dim)
    reduce_scatter_memcfg = _decode_reduce_scatter_memcfg()
    all_reduce_output_memcfg, all_reduce_buffer_memcfg = _decode_all_reduce_configs(dim)
    resources = require_galaxy_hardware_resources(
        mesh_device,
        config=_resources_config(
            mesh_device,
            dim,
            hidden_dim,
            decode_w2_input_memcfg=decode_ring["decode_w2_input_memcfg"],
            decode_reduce_scatter_memcfg=reduce_scatter_memcfg,
            decode_all_reduce_buffer_memcfg=all_reduce_buffer_memcfg,
        ),
        prefetch_weights=(
            ("mlp.w1", lazy_w1.get_device_weight()),
            ("mlp.w3", lazy_w3.get_device_weight()),
            ("mlp.w2", lazy_w2.get_device_weight()),
        ),
    )
    module = None
    try:
        module = MLP2D.from_config(
            MLP2DConfig(
                w1=lazy_w1,
                w2=lazy_w2,
                w3=lazy_w3,
                mesh_device=mesh_device,
                tt_ccl=resources.ccl,
                collective_resource_selector=exact_tensor_resource,
                w1_w3_memcfg=lazy_w1.memory_config,
                w2_memcfg=lazy_w2.memory_config,
                **decode_ring,
                ff1_out_reduce_scatter_memcfg=reduce_scatter_memcfg,
                ff2_out_reduce_scatter_memcfg=all_reduce_output_memcfg,
                sharded_attn_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                decode_prefetch_context=resources.prefetch_context("decode"),
                prefill_prefetch_context=resources.prefetch_context("prefill"),
                activation_dtype=activation_dtype,
                ccl_dtype=activation_dtype,
                mul_dtype=activation_dtype,
            )
        )
        for invocation in range(2):
            _invoke(
                module,
                resources,
                mesh_device,
                x,
                mode="decode",
                expected=expected,
                case=f"decode invocation {invocation}",
            )
    finally:
        try:
            resources.cleanup()
        finally:
            deallocate_module_weights(module, "w1", "w2", "w3")


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "dim,hidden_dim",
    [(8192, 28672), (5120, 25600)],
    ids=["llama-8192x28672", "qwen-5120x25600"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@torch.no_grad()
def test_mlp_2d_wh_galaxy_prefill_128_then_2048_repeat(mesh_device, dim, hidden_dim):
    torch.manual_seed(1)
    reference_mlp = _reference_mlp(dim, hidden_dim)
    w1, w2, w3 = get_mlp_weights_from_ref_model(reference_mlp)
    weight_dtype = ttnn.bfloat16 if dim == 5120 else ttnn.bfloat8_b
    activation_dtype = ttnn.bfloat8_b
    lazy_w1, lazy_w2, lazy_w3 = _weight_lazies(w1, w2, w3, mesh_device, weight_dtype)
    prefill_w1, prefill_w2, prefill_w3 = _prefill_weight_lazies(w1, w2, w3, mesh_device, weight_dtype)
    decode_ring = _decode_ring_config(dim, hidden_dim)
    reduce_scatter_memcfg = _decode_reduce_scatter_memcfg()
    all_reduce_output_memcfg, all_reduce_buffer_memcfg = _decode_all_reduce_configs(dim)
    resources = require_galaxy_hardware_resources(
        mesh_device,
        config=_resources_config(
            mesh_device,
            dim,
            hidden_dim,
            decode_w2_input_memcfg=decode_ring["decode_w2_input_memcfg"],
            decode_reduce_scatter_memcfg=reduce_scatter_memcfg,
            decode_all_reduce_buffer_memcfg=all_reduce_buffer_memcfg,
        ),
        prefetch_weights=(
            ("mlp.w1", lazy_w1.get_device_weight()),
            ("mlp.w3", lazy_w3.get_device_weight()),
            ("mlp.w2", lazy_w2.get_device_weight()),
        ),
    )
    module = None
    try:
        module = MLP2D.from_config(
            MLP2DConfig(
                w1=lazy_w1,
                w2=lazy_w2,
                w3=lazy_w3,
                prefill_w1=prefill_w1,
                prefill_w2=prefill_w2,
                prefill_w3=prefill_w3,
                mesh_device=mesh_device,
                tt_ccl=resources.ccl,
                collective_resource_selector=exact_tensor_resource,
                w1_w3_memcfg=lazy_w1.memory_config,
                w2_memcfg=lazy_w2.memory_config,
                ff1_out_reduce_scatter_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                ff2_out_reduce_scatter_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                sharded_attn_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                decode_prefetch_context=resources.prefetch_context("decode"),
                prefill_prefetch_context=resources.prefetch_context("prefill"),
                activation_dtype=activation_dtype,
                ccl_dtype=activation_dtype,
                mul_dtype=activation_dtype,
            )
        )
        for invocation in range(2):
            for seq_len in (128, 2048):
                x = torch.randn(1, 1, seq_len, dim, dtype=torch.bfloat16)
                _invoke(
                    module,
                    resources,
                    mesh_device,
                    x,
                    mode="prefill",
                    expected=reference_mlp(x),
                    case=f"prefill {seq_len} invocation {invocation}",
                )
    finally:
        try:
            resources.cleanup()
        finally:
            deallocate_module_weights(module, "w1", "w2", "w3", "prefill_w1", "prefill_w2", "prefill_w3")
