# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Parity tests for the three LoRA-aware Linear subclasses, each exercised
under both ``lora_mode`` options:

  fuse     — bind merges ``scale * A.T @ B.T`` into the base weight in-place.
             Forward runs the un-modified base path.
  runtime  — bind uploads A, B to device; forward adds
             ``scale * (x @ A) @ B`` on top of the base output.

The unbind branch is checked in both modes — for ``fuse`` it subtracts
the delta back out of W; for ``runtime`` it frees the A/B tensors so the
forward path collapses to the base.

Mesh shape is 1x1 so sharding collapses to the replicated case for all
three variants. The tests still exercise the LoRA-specific upload and
forward code paths and catch class-level bugs in shape handling,
fused-activation fallback, and chunked-output handling.

Run with:
    pytest -xvs models/tt_dit/experimental/tests/test_lora_variants.py
"""
from __future__ import annotations

import pytest
import torch

import ttnn
from models.tt_dit.layers.linear import LoRAColParallelLinear, LoRALinear, LoRARowParallelLinear
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tensor import bf16_tensor


def _make_pair(in_features: int, out_features: int, rank: int, dtype: torch.dtype):
    A = torch.randn(rank, in_features, dtype=dtype) * 0.02
    B = torch.randn(out_features, rank, dtype=dtype) * 0.02
    return A, B


def _ref_lora(x, W, b, A, B, scale):
    """y = x @ (W + scale * B @ A).T + b"""
    W_eff = W + scale * (B @ A)
    return torch.nn.functional.linear(x, W_eff, b)


# --------------------------------------------------------------------
# LoRALinear (replicated)
# --------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("lora_mode", ["fuse", "runtime"])
@pytest.mark.parametrize(
    ("seq_len", "in_features", "out_features", "rank"),
    [
        (128, 256, 256, 8),
        (256, 512, 1024, 16),
    ],
)
def test_lora_linear_variant(
    mesh_device: ttnn.MeshDevice,
    lora_mode: str,
    seq_len: int,
    in_features: int,
    out_features: int,
    rank: int,
) -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    scale = 0.6

    torch_lin = torch.nn.Linear(in_features, out_features, bias=True).to(dtype=dtype)
    torch_lin.eval()
    A_t, B_t = _make_pair(in_features, out_features, rank, dtype)
    x_t = torch.randn(1, 1, seq_len, in_features, dtype=dtype)

    with torch.no_grad():
        y_base = torch_lin(x_t)
        y_lora = _ref_lora(x_t, torch_lin.weight, torch_lin.bias, A_t, B_t, scale)

    tt = LoRALinear(in_features, out_features, bias=True, mesh_device=mesh_device, lora_mode=lora_mode)
    tt.load_torch_state_dict(torch_lin.state_dict())
    idx = tt.register_lora(A_t, B_t, scale=scale)
    x_tt = bf16_tensor(x_t, device=mesh_device)

    # 1) no adapter bound → base behavior
    y = tt(x_tt)
    for t in ttnn.get_device_tensors(y):
        assert_quality(y_base, ttnn.to_torch(t), pcc=0.999_5)

    # 2) bind active → LoRA effect applied
    tt.bind_active(idx, scale=scale)
    y = tt(x_tt)
    for t in ttnn.get_device_tensors(y):
        assert_quality(y_lora, ttnn.to_torch(t), pcc=0.999_0)

    # 3) unbind collapses back to base
    tt.unbind_active()
    y = tt(x_tt)
    for t in ttnn.get_device_tensors(y):
        assert_quality(y_base, ttnn.to_torch(t), pcc=0.999_5)

    tt.deallocate_lora()


# --------------------------------------------------------------------
# LoRAColParallelLinear — single output and chunked output
# --------------------------------------------------------------------
# Chunked configs run fuse-only: the runtime delta path doesn't split
# its delta into the same chunk layout as the base output.
_COL_PARALLEL_PARAMS = [
    # (label, lora_mode, seq, in, out, rank, chunks, activation_fn)
    ("plain_fuse", "fuse", 128, 256, 512, 8, None, None),
    ("plain_runtime", "runtime", 128, 256, 512, 8, None, None),
    ("gelu_fuse", "fuse", 128, 256, 1024, 8, None, "gelu"),
    ("gelu_runtime", "runtime", 128, 256, 1024, 8, None, "gelu"),
    ("chunked_qkv_fuse", "fuse", 128, 256, 3 * 256, 8, 3, None),
]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    ("lora_mode", "seq_len", "in_features", "out_features", "rank", "chunks", "activation_fn"),
    [params[1:] for params in _COL_PARALLEL_PARAMS],
    ids=[params[0] for params in _COL_PARALLEL_PARAMS],
)
def test_lora_col_parallel_variant(
    mesh_device: ttnn.MeshDevice,
    lora_mode: str,
    seq_len: int,
    in_features: int,
    out_features: int,
    rank: int,
    chunks: int | None,
    activation_fn: str | None,
) -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    scale = 0.5

    torch_lin = torch.nn.Linear(in_features, out_features, bias=True).to(dtype=dtype)
    torch_lin.eval()
    A_t, B_t = _make_pair(in_features, out_features, rank, dtype)
    x_t = torch.randn(1, 1, seq_len, in_features, dtype=dtype)

    def torch_act(y):
        if activation_fn == "gelu":
            return torch.nn.functional.gelu(y)
        return y

    with torch.no_grad():
        y_base = torch_act(torch_lin(x_t))
        y_lora = torch_act(_ref_lora(x_t, torch_lin.weight, torch_lin.bias, A_t, B_t, scale))

    tt = LoRAColParallelLinear(
        in_features,
        out_features,
        bias=True,
        mesh_device=mesh_device,
        mesh_axis=0,
        activation_fn=activation_fn,
        chunks=chunks,
        lora_mode=lora_mode,
    )
    tt.load_torch_state_dict(torch_lin.state_dict())
    idx = tt.register_lora(A_t, B_t, scale=scale)
    x_tt = bf16_tensor(x_t, device=mesh_device)

    def _join(out):
        """Concat chunked outputs along last dim; pass through single tensors."""
        if isinstance(out, list):
            torch_chunks = [ttnn.to_torch(ttnn.get_device_tensors(o)[0]) for o in out]
            return torch.cat(torch_chunks, dim=-1)
        return ttnn.to_torch(ttnn.get_device_tensors(out)[0])

    # 1) no adapter bound → base behavior
    assert_quality(y_base, _join(tt(x_tt)), pcc=0.999_5)

    # 2) bind active → LoRA effect applied
    tt.bind_active(idx, scale=scale)
    assert_quality(y_lora, _join(tt(x_tt)), pcc=0.999_0)

    # 3) unbind collapses to base
    tt.unbind_active()
    assert_quality(y_base, _join(tt(x_tt)), pcc=0.999_0)

    tt.deallocate_lora()


# --------------------------------------------------------------------
# LoRARowParallelLinear — N_tp=1 (1x1 mesh) sanity
# --------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("lora_mode", ["fuse", "runtime"])
@pytest.mark.parametrize(
    ("seq_len", "in_features", "out_features", "rank"),
    [
        (128, 512, 256, 8),
        (256, 1024, 512, 16),
    ],
)
def test_lora_row_parallel_variant(
    mesh_device: ttnn.MeshDevice,
    lora_mode: str,
    seq_len: int,
    in_features: int,
    out_features: int,
    rank: int,
) -> None:
    from models.tt_dit.parallel.manager import CCLManager

    torch.manual_seed(0)
    dtype = torch.bfloat16
    scale = 0.4

    torch_lin = torch.nn.Linear(in_features, out_features, bias=True).to(dtype=dtype)
    torch_lin.eval()
    A_t, B_t = _make_pair(in_features, out_features, rank, dtype)
    x_t = torch.randn(1, 1, seq_len, in_features, dtype=dtype)

    with torch.no_grad():
        y_base = torch_lin(x_t)
        y_lora = _ref_lora(x_t, torch_lin.weight, torch_lin.bias, A_t, B_t, scale)

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    tt = LoRARowParallelLinear(
        in_features,
        out_features,
        bias=True,
        mesh_device=mesh_device,
        mesh_axis=0,
        ccl_manager=ccl_manager,
        lora_mode=lora_mode,
    )
    tt.load_torch_state_dict(torch_lin.state_dict())
    idx = tt.register_lora(A_t, B_t, scale=scale)
    x_tt = bf16_tensor(x_t, device=mesh_device)

    # baseline
    y = tt(x_tt)
    assert_quality(y_base, ttnn.to_torch(ttnn.get_device_tensors(y)[0]), pcc=0.999_5)

    # bind active
    tt.bind_active(idx, scale=scale)
    y = tt(x_tt)
    assert_quality(y_lora, ttnn.to_torch(ttnn.get_device_tensors(y)[0]), pcc=0.999_0)

    # unbind
    tt.unbind_active()
    y = tt(x_tt)
    assert_quality(y_base, ttnn.to_torch(ttnn.get_device_tensors(y)[0]), pcc=0.999_5)

    tt.deallocate_lora()


# --------------------------------------------------------------------
# Construction-time validation
# --------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_invalid_lora_mode_raises(mesh_device: ttnn.MeshDevice, expect_error) -> None:
    with expect_error(ValueError, "lora_mode"):
        LoRALinear(64, 64, bias=False, mesh_device=mesh_device, lora_mode="bogus")
