# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Coverage for the opt-in ``use_small_m_matmul`` backend of Linear / ColParallelLinear.

Three things fail in different places, so they are tested separately:

1. WEIGHT LAYOUT, decided at construction: the weight must load DRAM WIDTH_SHARDED across all of the
   device's DRAM banks (bank count taken from the device, not assumed to be 8).
2. ADMISSION, also at construction: every rejection is a condition under which the op could not run at
   any M, so it must fail before a weight is laid out.
3. NUMERICS (PCC vs torch) of each dispatch path: plain, fused / unfused activation, chunked, fused
   addcmul, and tensor parallel with and without the hoisted K all-gather.

Blackhole only: the op is Blackhole only.
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

PCC = 0.999_500
_MESH_1x1 = pytest.mark.parametrize("mesh_device", [(1, 1)], ids=mesh_device_config_to_string, indirect=True)


def _torch_linear(K, N, bias=True):
    m = torch.nn.Linear(K, N, bias=bias).to(dtype=torch.bfloat16)
    m.eval()
    return m


# ---------------------------------------------------------------------------------------------
# 1. Weight layout
# ---------------------------------------------------------------------------------------------


@_MESH_1x1
def test_small_m_weight_is_width_sharded(mesh_device: ttnn.MeshDevice) -> None:
    K, N = 2048, 2048
    tt_model = Linear(K, N, mesh_device=mesh_device, use_small_m_matmul=True)
    tt_model.load_torch_state_dict(_torch_linear(K, N).state_dict())

    mem = tt_model.weight.data.memory_config()
    expected = ttnn.create_small_m_weight_memory_config([K, N], ttnn.bfloat16, mesh_device)
    assert mem.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert mem.buffer_type == ttnn.BufferType.DRAM
    # One shard per DRAM bank of THIS device (8 on P150 / Galaxy, 7 on P100).
    assert mem.shard_spec.num_cores() == expected.shard_spec.num_cores()
    # Only the weight carries the layout requirement.
    assert tt_model.bias.data.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


@_MESH_1x1
def test_small_m_flag_off_leaves_layout_alone(mesh_device: ttnn.MeshDevice) -> None:
    tt_model = Linear(2048, 2048, mesh_device=mesh_device)
    tt_model.load_torch_state_dict(_torch_linear(2048, 2048).state_dict())
    assert tt_model.weight.data.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


# ---------------------------------------------------------------------------------------------
# 2. Admission
# ---------------------------------------------------------------------------------------------


@_MESH_1x1
def test_small_m_rejects_infeasible_n(mesh_device: ttnn.MeshDevice, expect_error) -> None:
    """N=384 (Nt=12) leaves the trailing bank all padding on 8 and on 7 banks."""
    with expect_error(ValueError, "cannot serve the local weight"):
        Linear(2048, 384, mesh_device=mesh_device, use_small_m_matmul=True)


@_MESH_1x1
def test_small_m_rejects_swiglu(mesh_device: ttnn.MeshDevice, expect_error) -> None:
    with expect_error(ValueError, "does not implement fuse_swiglu"):
        Linear(2048, 2048, activation_fn="swiglu", mesh_device=mesh_device, use_small_m_matmul=True)


@_MESH_1x1
def test_small_m_rejects_non_bf16_weight(mesh_device: ttnn.MeshDevice, expect_error) -> None:
    with expect_error(ValueError, "requires a bfloat16 weight"):
        Linear(2048, 2048, dtype=ttnn.bfloat8_b, mesh_device=mesh_device, use_small_m_matmul=True)


@_MESH_1x1
def test_small_m_rejects_output_dtype_at_forward(mesh_device: ttnn.MeshDevice, expect_error) -> None:
    tt_model = Linear(2048, 2048, mesh_device=mesh_device, use_small_m_matmul=True)
    tt_model.load_torch_state_dict(_torch_linear(2048, 2048).state_dict())
    x = bf16_tensor(torch.randn((1, 1, 32, 2048), dtype=torch.bfloat16), device=mesh_device)
    with expect_error(ValueError, "fixes the output dtype to bfloat16"):
        tt_model(x, dtype=ttnn.bfloat8_b)


# ---------------------------------------------------------------------------------------------
# 3. Numerics
# ---------------------------------------------------------------------------------------------


@_MESH_1x1
@pytest.mark.parametrize(
    ("M", "K", "N"),
    [
        (32, 256, 3072),  # FLUX 3 Klein time embedding linear_1
        (3, 3072, 3072),  # three modulation rows, padded to one tile
        (32, 6144, 6144),  # Flux2 time embedding linear_2
        (256, 2048, 2048),  # Mt = 8
    ],
    ids=["klein_time_in", "rows3", "flux2_time_l2", "m256"],
)
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("activation_fn", [None, "gelu_tanh", "silu"], ids=["none", "fused_gelu", "unfused_silu"])
def test_small_m_linear(mesh_device: ttnn.MeshDevice, M: int, K: int, N: int, bias: bool, activation_fn) -> None:
    """gelu_tanh rides the op's fused_activation; silu is applied after the op."""
    torch_model = _torch_linear(K, N, bias=bias)
    tt_model = Linear(K, N, bias=bias, activation_fn=activation_fn, mesh_device=mesh_device, use_small_m_matmul=True)
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch_input = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    with torch.no_grad():
        torch_output = torch_model(torch_input)
        if activation_fn == "gelu_tanh":
            torch_output = torch.nn.functional.gelu(torch_output, approximate="tanh")
        elif activation_fn == "silu":
            torch_output = torch.nn.functional.silu(torch_output)

    tt_output = tt_model(bf16_tensor(torch_input, device=mesh_device))
    assert_quality(torch_output, ttnn.to_torch(tt_output), pcc=PCC)


@_MESH_1x1
@pytest.mark.parametrize(
    ("K", "N", "chunks"),
    [(2048, 768 * 3, 3), (3072, 15360, 20)],
    ids=["chunks3", "klein_modulation_tp4_local"],
)
def test_small_m_col_parallel_chunks(mesh_device: ttnn.MeshDevice, K: int, N: int, chunks: int) -> None:
    """Chunked output goes through small_m_matmul_split, which writes each chunk directly."""
    torch_model = _torch_linear(K, N)
    tt_model = ColParallelLinear(K, N, mesh_device=mesh_device, mesh_axis=0, chunks=chunks, use_small_m_matmul=True)
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch_input = torch.randn((1, 1, 32, K), dtype=torch.bfloat16)
    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_outputs = tt_model(bf16_tensor(torch_input, device=mesh_device))
    assert len(tt_outputs) == chunks
    assert_quality(torch_output, torch.cat([ttnn.to_torch(o) for o in tt_outputs], dim=-1), pcc=PCC)


@_MESH_1x1
def test_small_m_col_parallel_addcmul(mesh_device: ttnn.MeshDevice) -> None:
    """Fused epilogue: residual + scalar * (x@W + bias) * gate; at TP=1, which the default path cannot fuse."""
    K, N, scalar = 3072, 3072, 0.5
    torch_model = _torch_linear(K, N)
    tt_model = ColParallelLinear(K, N, mesh_device=mesh_device, mesh_axis=0, use_small_m_matmul=True)
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch_input = torch.randn((1, 1, 32, K), dtype=torch.bfloat16)
    torch_residual = torch.randn((1, 1, 32, N), dtype=torch.bfloat16)
    torch_gate = torch.randn((1, 1, 1, N), dtype=torch.bfloat16)
    with torch.no_grad():
        torch_output = torch_residual + scalar * torch_model(torch_input) * torch_gate

    tt_output = tt_model(
        bf16_tensor(torch_input, device=mesh_device),
        addcmul_a=bf16_tensor(torch_residual, device=mesh_device),
        addcmul_b=bf16_tensor(torch_gate, device=mesh_device),
        addcmul_scalar=scalar,
    )
    assert_quality(torch_output, ttnn.to_torch(tt_output), pcc=PCC)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((4, 8), {"fabric_config": ttnn.FabricConfig.FABRIC_1D})],
    ids=mesh_device_config_to_string,
    indirect=True,
)
@pytest.mark.parametrize("input_fractured", [False, True], ids=["replicated_input", "k_fractured_input"])
@pytest.mark.parametrize("chunks", [None, 20], ids=["no_chunks", "chunks20"])
def test_small_m_col_parallel_tp(mesh_device: ttnn.MeshDevice, input_fractured: bool, chunks) -> None:
    """TP=4 column parallel on the BH Galaxy mesh (TP on axis 0, as FLUX 3 Klein runs). A K-fractured input is
    all-gathered stand-alone, then each device runs the single-chip small-M matmul on its local N slice. With
    20 chunks and N = 20 * 3072 this is exactly Klein's merged modulation projection."""
    tp_axis = 0
    tp = mesh_device.shape[tp_axis]
    K = 3072
    N = 3072 * (chunks or 4)
    torch_model = _torch_linear(K, N)
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    tt_model = ColParallelLinear(
        K,
        N,
        mesh_device=mesh_device,
        mesh_axis=tp_axis,
        ccl_manager=ccl_manager,
        chunks=chunks,
        use_small_m_matmul=True,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())
    parallel_config = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(1, 0), tp=(tp, tp_axis))

    torch_input = torch.randn((1, 1, 32, K), dtype=torch.bfloat16)
    with torch.no_grad():
        torch_output = torch_model(torch_input)

    if input_fractured:
        tt_input = bf16_tensor(torch_input, device=mesh_device, mesh_axis=tp_axis, shard_dim=-1)
    else:
        tt_input = bf16_tensor(torch_input, device=mesh_device)
    tt_output = tt_model(tt_input, parallel_config=parallel_config)

    outputs = tt_output if isinstance(tt_output, list) else [tt_output]
    assert len(outputs) == (chunks or 1)
    # The device at mesh coordinate (r, c) holds columns [d*N/tp, (d+1)*N/tp) with d its TP-axis coordinate,
    # split locally into the chunks.
    per_chunk = [[ttnn.to_torch(t) for t in ttnn.get_device_tensors(o)] for o in outputs]
    cols = mesh_device.shape[1]
    n_local = N // tp
    for i in range(len(per_chunk[0])):
        d = divmod(i, cols)[tp_axis]
        local = torch.cat([per_chunk[c][i] for c in range(len(outputs))], dim=-1)
        assert_quality(torch_output[..., d * n_local : (d + 1) * n_local], local, pcc=PCC)
