# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Coverage for the ``use_small_m_matmul`` backend on Linear / ColParallelLinear.

Three things are worth testing separately here, because they fail in different places:

1. WEIGHT LAYOUT, decided at construction. The op reads in1 from a DRAM WIDTH_SHARDED (8-bank)
   buffer, so the layout is part of the op's contract and Parameter enforces it at load.
2. ADMISSION, also at construction. Every rejection covered below is a condition under which the
   op could not run at ANY M, so it must fail before a weight is laid out -- not on first forward.
3. NUMERICS of each dispatch path (plain, chunked, TP gather, fused addcmul).

All of these require Blackhole: the op is Blackhole-only.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole

from ...layers.linear import ColParallelLinear, Linear
from ...parallel.config import DiTParallelConfig
from ...parallel.manager import CCLManager
from ...utils.check import assert_quality
from ...utils.tensor import bf16_tensor
from ...utils.test import mesh_device_config_to_string

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="small_m_matmul is Blackhole-only")

# Shapes are small M / wide N with an N whose LOCAL width satisfies 7*ceil(Nt/8) < Nt.
# N=2048 -> Nt=64 and N=768 -> Nt=24 are both feasible; N=384 -> Nt=12 is deliberately not.
_M, _K, _N = 256, 2048, 2048


def _mesh_1x1():
    return pytest.mark.parametrize("mesh_device", [(1, 1)], ids=mesh_device_config_to_string, indirect=True)


def _torch_linear(K, N, bias=True):
    m = torch.nn.Linear(K, N, bias=bias).to(dtype=torch.bfloat16)
    m.eval()
    return m


# ---------------------------------------------------------------------------------------------
# 1. Weight layout
# ---------------------------------------------------------------------------------------------


@_mesh_1x1()
def test_small_m_weight_is_width_sharded(mesh_device: ttnn.MeshDevice) -> None:
    """The weight must land in the 8-bank DRAM width shard at LOAD time, with no per-forward reshard."""
    torch_model = _torch_linear(_K, _N)
    tt_model = Linear(_K, _N, mesh_device=mesh_device, use_small_m_matmul=True)
    tt_model.load_torch_state_dict(torch_model.state_dict())

    mem = tt_model.weight.data.memory_config()
    assert mem.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert mem.buffer_type == ttnn.BufferType.DRAM
    # Exactly 8 banks, one shard column each.
    assert mem.shard_spec.num_cores() == 8

    # The bias is untouched: only in1 carries the layout requirement.
    assert tt_model.bias.data.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


@_mesh_1x1()
def test_small_m_flag_off_leaves_layout_alone(mesh_device: ttnn.MeshDevice) -> None:
    """Opting out must not perturb the default interleaved weight."""
    tt_model = Linear(_K, _N, mesh_device=mesh_device)
    tt_model.load_torch_state_dict(_torch_linear(_K, _N).state_dict())
    assert tt_model.weight.data.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


# ---------------------------------------------------------------------------------------------
# 2. Admission
# ---------------------------------------------------------------------------------------------


@_mesh_1x1()
def test_small_m_rejects_infeasible_n(mesh_device: ttnn.MeshDevice, expect_error) -> None:
    """N=384 gives Nt=12, where 7*ceil(12/8)=14 >= 12: the trailing banks would be all padding."""
    with expect_error(ValueError, "cannot serve local N=384"):
        Linear(_K, 384, mesh_device=mesh_device, use_small_m_matmul=True)


@_mesh_1x1()
def test_small_m_rejects_swiglu(mesh_device: ttnn.MeshDevice, expect_error) -> None:
    with expect_error(ValueError, "does not implement fuse_swiglu"):
        Linear(_K, _N, activation_fn="swiglu", mesh_device=mesh_device, use_small_m_matmul=True)


@_mesh_1x1()
def test_small_m_rejects_non_bf16_weight(mesh_device: ttnn.MeshDevice, expect_error) -> None:
    with expect_error(ValueError, "requires a bfloat16 weight"):
        Linear(_K, _N, dtype=ttnn.bfloat8_b, mesh_device=mesh_device, use_small_m_matmul=True)


@_mesh_1x1()
def test_small_m_rejects_output_dtype_at_forward(mesh_device: ttnn.MeshDevice, expect_error) -> None:
    """Numerics are fixed, so an explicit non-bf16 output dtype is an error rather than ignored."""
    tt_model = Linear(_K, _N, mesh_device=mesh_device, use_small_m_matmul=True)
    tt_model.load_torch_state_dict(_torch_linear(_K, _N).state_dict())
    x = bf16_tensor(torch.randn((1, 1, _M, _K), dtype=torch.bfloat16), device=mesh_device)
    with expect_error(ValueError, "fixes the output dtype to bfloat16"):
        tt_model(x, dtype=ttnn.bfloat8_b)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((1, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True})],
    ids=mesh_device_config_to_string,
    indirect=True,
)
def test_small_m_rejects_fsdp(mesh_device: ttnn.MeshDevice, expect_error) -> None:
    """An FSDP-gathered weight materialises interleaved at runtime; re-sharding it per forward
    would cost more than the matmul saves, so the combination is refused up front."""
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    with expect_error(ValueError, "incompatible with FSDP weight gathering"):
        ColParallelLinear(
            _K,
            _N,
            mesh_device=mesh_device,
            mesh_axis=0,
            fsdp_mesh_axis=1,
            ccl_manager=ccl_manager,
            use_small_m_matmul=True,
        )


# ---------------------------------------------------------------------------------------------
# 3. Numerics
# ---------------------------------------------------------------------------------------------


@_mesh_1x1()
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("activation_fn", [None, "gelu_tanh", "silu"], ids=["none", "fused_gelu", "unfused_silu"])
def test_small_m_linear(mesh_device: ttnn.MeshDevice, bias: bool, activation_fn) -> None:
    """Covers both fusion routes: gelu_tanh rides the op's fused_activation, silu is applied after."""
    torch_model = _torch_linear(_K, _N, bias=bias)
    tt_model = Linear(_K, _N, bias=bias, activation_fn=activation_fn, mesh_device=mesh_device, use_small_m_matmul=True)
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch_input = torch.randn((1, 1, _M, _K), dtype=torch.bfloat16)
    with torch.no_grad():
        torch_output = torch_model(torch_input)
        if activation_fn == "gelu_tanh":
            torch_output = torch.nn.functional.gelu(torch_output, approximate="tanh")
        elif activation_fn == "silu":
            torch_output = torch.nn.functional.silu(torch_output)

    tt_output = tt_model(bf16_tensor(torch_input, device=mesh_device))
    assert_quality(torch_output, ttnn.to_torch(tt_output), pcc=0.999_500)


@_mesh_1x1()
@pytest.mark.parametrize("chunks", [2, 3], ids=["chunks2", "chunks3"])
def test_small_m_col_parallel_chunks(mesh_device: ttnn.MeshDevice, chunks: int) -> None:
    """Chunked output goes through small_m_matmul_split, which writes each chunk directly."""
    N = 768 * chunks
    torch_model = _torch_linear(_K, N)
    tt_model = ColParallelLinear(_K, N, mesh_device=mesh_device, mesh_axis=0, chunks=chunks, use_small_m_matmul=True)
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch_input = torch.randn((1, 1, _M, _K), dtype=torch.bfloat16)
    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_outputs = tt_model(bf16_tensor(torch_input, device=mesh_device))
    assert len(tt_outputs) == chunks
    tt_output = torch.cat([ttnn.to_torch(o) for o in tt_outputs], dim=-1)
    assert_quality(torch_output, tt_output, pcc=0.999_500)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((1, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True})],
    ids=mesh_device_config_to_string,
    indirect=True,
)
def test_small_m_col_parallel_tp_gather(mesh_device: ttnn.MeshDevice) -> None:
    """The target composition: stand-alone all-gather along K, then the single-chip small-M matmul.

    Input arrives fractured on K (that is what the fused AG-matmul would have gathered), output
    comes back fractured on N.
    """
    tp_axis = 1
    tp = mesh_device.shape[tp_axis]
    torch_model = _torch_linear(_K, _N)
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    tt_model = ColParallelLinear(
        _K,
        _N,
        mesh_device=mesh_device,
        mesh_axis=tp_axis,
        ccl_manager=ccl_manager,
        use_small_m_matmul=True,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    parallel_config = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(1, 0), tp=(tp, tp_axis))

    torch_input = torch.randn((1, 1, _M, _K), dtype=torch.bfloat16)
    with torch.no_grad():
        torch_output = torch_model(torch_input)

    # Fracture the activation on K across the TP axis, mirroring the fused op's pre-gather input.
    shard_dims = [None, None]
    shard_dims[tp_axis] = -1
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )

    tt_output = tt_model(tt_input, parallel_config=parallel_config)

    out_dims = [None, None]
    out_dims[tp_axis] = -1
    out_dims[1 - tp_axis] = 0
    tt_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=out_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    assert_quality(torch_output, tt_output[:1], pcc=0.999_500)


@_mesh_1x1()
def test_small_m_fused_addcmul(mesh_device: ttnn.MeshDevice) -> None:
    """forward_fused_addcmul: residual + scalar * (x@W + bias) * gate, in one op."""
    scalar = 0.5
    torch_model = _torch_linear(_K, _N)
    tt_model = ColParallelLinear(_K, _N, mesh_device=mesh_device, mesh_axis=0, use_small_m_matmul=True)
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch_input = torch.randn((1, 1, _M, _K), dtype=torch.bfloat16)
    torch_residual = torch.randn((1, 1, _M, _N), dtype=torch.bfloat16)
    torch_gate = torch.randn((1, 1, 1, _N), dtype=torch.bfloat16)
    with torch.no_grad():
        torch_output = torch_residual + scalar * torch_model(torch_input) * torch_gate

    tt_output = tt_model.forward_fused_addcmul(
        bf16_tensor(torch_input, device=mesh_device),
        bf16_tensor(torch_residual, device=mesh_device),
        bf16_tensor(torch_gate, device=mesh_device),
        scalar,
    )
    assert_quality(torch_output, ttnn.to_torch(tt_output), pcc=0.999_500)
