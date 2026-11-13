# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Single-device pieces of the Qwen3-VL decoder's dynamic-length path: one program for every row count."""

import pytest
import torch

import ttnn

from ...encoders.qwen3vl.model_qwen3vl import Qwen3VlContext, _MatmulRotateHalf, _rotate_half
from ...layers.linear import ColParallelLinear, RowParallelLinear
from ...utils.check import assert_quality
from ...utils.matmul import get_matmul_config, get_matmul_core_grid
from ...utils.tensor import bf16_tensor, to_torch

HEAD_DIM = 128
ROW_CAPACITY = 640
# Long, short, long: a program compiled at one row count must serve a larger and a smaller one.
ROWS = (320, 32, 64, 96, ROW_CAPACITY)

_SINGLE = pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)


@_SINGLE
@pytest.mark.parametrize(("heads", "programs"), [(16, 1), (2, 2)], ids=["q", "k"])
def test_rotate_half_matmul(mesh_device: ttnn.MeshDevice, heads: int, programs: int) -> None:
    """`x @ R` equals slice + neg + concat value for value. K changes program once, where its rows pass head_dim."""
    torch.manual_seed(0)
    rotate = _MatmulRotateHalf(mesh_device, HEAD_DIM)
    before = mesh_device.num_program_cache_entries()
    added = 0
    for rows in ROWS:
        x = bf16_tensor(torch.randn(1, heads, rows, HEAD_DIM) * 3, device=mesh_device)
        expected = to_torch(_rotate_half(x))
        entries = mesh_device.num_program_cache_entries()
        actual = to_torch(rotate(x))
        added += mesh_device.num_program_cache_entries() - entries
        assert torch.equal(actual, expected), f"rows={rows}"
    assert added == programs, f"{added} rotation programs over rows {ROWS}"
    assert mesh_device.num_program_cache_entries() > before


@_SINGLE
@pytest.mark.parametrize(
    ("layer", "in_features", "out_features"),
    [(ColParallelLinear, 5120, 2560), (RowParallelLinear, 6400, 5120)],
    ids=["qkv", "down"],
)
def test_pinned_projection(mesh_device: ttnn.MeshDevice, layer, in_features: int, out_features: int) -> None:
    """A pinned config keeps one program on both sides of the 128-row switch to `ttnn.linear`."""
    torch.manual_seed(0)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    torch_model = torch.nn.Linear(in_features, out_features, bias=False)
    tt_model = layer(
        in_features,
        out_features,
        bias=False,
        mesh_device=mesh_device,
        mesh_axis=None,
        compute_kernel_config=compute_kernel_config,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())
    pin = get_matmul_config(ROW_CAPACITY, in_features, out_features, get_matmul_core_grid(mesh_device))

    added = 0
    for rows in (*ROWS, 128, 160):
        x = torch.randn(1, rows, in_features)
        tt_x = bf16_tensor(x, device=mesh_device)
        entries = mesh_device.num_program_cache_entries()
        actual = to_torch(tt_model.forward(tt_x, matmul_config=pin))
        added += mesh_device.num_program_cache_entries() - entries
        with torch.no_grad():
            assert_quality(torch_model(x), actual, pcc=0.9999)
    assert added == 1, f"{added} projection programs"


def test_context_dynamic_length_needs_sp_and_capacity() -> None:
    with pytest.raises(ValueError, match="dynamic_length"):
        Qwen3VlContext(device=None, tp_axis=0, ccl_manager=object(), dynamic_length=True)
    with pytest.raises(ValueError, match="dynamic_length"):
        Qwen3VlContext(device=None, tp_axis=0, ccl_manager=object(), sp_axis=1, dynamic_length=True)


def test_encoder_dynamic_length_default() -> None:
    """The MiniMax-H3 pipeline turns the encoder flag on only where it is tested; an explicit value always wins."""
    from ...pipelines.minimax_h3.pipeline_minimax_h3 import _resolve_encoder_dynamic_length as resolve

    on = dict(dynamic_length=True, blackhole=True, task="t2va", topology=ttnn.Topology.Ring, mesh_shape=(4, 8))
    assert resolve(None, **on)
    assert not resolve(False, **on)
    for change in (
        {"dynamic_length": False},
        {"blackhole": False},
        {"task": "ref2va"},
        {"topology": ttnn.Topology.Linear},
        {"mesh_shape": (4, 32)},
    ):
        assert not resolve(None, **{**on, **change}), change
        assert resolve(True, **{**on, **change}), change
