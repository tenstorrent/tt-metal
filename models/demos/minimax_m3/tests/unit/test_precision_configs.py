# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks (no device) that the non-expert matmuls run with the compute configs of tt/compute_configs.py:
every ttnn call is replaced by a recorder. The device-side numerics are covered by the *_vs_ref layer tests.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.minimax_m3.tt import dense_mlp, layer
from models.demos.minimax_m3.tt.attention import msa, operations
from models.demos.minimax_m3.tt.compute_configs import matmul_compute_config
from models.demos.minimax_m3.tt.moe.shared_overlap import DEST_TILES, cb_tiles

BH = ttnn.device.Arch.BLACKHOLE
WH = ttnn.device.Arch.WORMHOLE_B0


class _Dev:
    def __init__(self, arch=BH):
        self._arch = arch

    def arch(self):
        return self._arch


class _T:
    def __init__(self, name, shape=(1, 1, 1024, 6144), dtype=ttnn.bfloat16):
        self.name, self.shape, self.padded_shape, self.dtype = name, shape, shape, dtype
        self.freed = False

    def device(self):
        return _Dev()

    def memory_config(self):
        return ttnn.DRAM_MEMORY_CONFIG

    def deallocate(self, *a):
        self.freed = True


def _recorder(monkeypatch, *names):
    calls = []

    def make(name):
        def f(*a, **k):
            calls.append((name, a, k))
            return _T(name)

        return f

    for name in names:
        monkeypatch.setattr(ttnn, name, make(name))
    return calls


@pytest.mark.parametrize("arch", [BH, WH], ids=["bh", "wh"])
def test_configs(arch):
    mm = matmul_compute_config(_Dev(arch))
    assert (mm.math_fidelity, mm.fp32_dest_acc_en, mm.packer_l1_acc, mm.math_approx_mode) == (
        ttnn.MathFidelity.HiFi3,
        True,
        True,
        False,
    )
    assert matmul_compute_config(_Dev(arch)) is mm


def test_hifi3_exact_for_mxfp8_weights():
    """HiFi3 drops only the weight's low-slice x activation's low-slice product; an MXFP8-origin weight has no low
    slice: every e4m3 value times a UE8M0 scale keeps at most 1+4 significant bits in bf16."""
    codes = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    w = codes[torch.isfinite(codes)] * 2.0 ** torch.arange(-20, 4).float().unsqueeze(-1)
    w = w.flatten().to(torch.bfloat16).float()
    hi = (w.view(torch.int32) & ~((1 << (23 - 4)) - 1)).view(torch.float32)
    assert torch.equal(hi, w)


def test_qkv_and_o_proj(monkeypatch):
    calls = _recorder(monkeypatch, "linear", "matmul", "typecast")
    weights = SimpleNamespace(wqkv=_T("wqkv"), o_proj=_T("o_proj", dtype=ttnn.bfloat8_b))
    operations.apply_qkv_projection(_T("x"), weights)
    x = _T("attn")
    operations.apply_output_projection(x, weights, ttnn.bfloat16)
    assert [c[0] for c in calls] == ["linear", "matmul"], "o_proj must take the bf16 activation (no typecast)"
    for _, args, kw in calls:
        assert kw["compute_kernel_config"] is matmul_compute_config(_Dev())
        assert kw["dtype"] == ttnn.bfloat16
    assert calls[1][1][0] is x and not x.freed


def test_o_proj_fused_rs(monkeypatch):
    calls = _recorder(monkeypatch, "typecast")
    ccl = SimpleNamespace(
        mesh_device=_Dev(),
        num_links=2,
        topology=ttnn.Topology.Ring,
        get_rs_ping_pong_semaphore=lambda: None,
        get_barrier_semaphore=lambda: None,
    )
    monkeypatch.setattr(
        ttnn.experimental,
        "minimal_matmul_strided_reduce_scatter_async",
        lambda *a, **k: (calls.append(("mm_rs", a, k)), (_T("mm"), _T("rs")))[1],
    )
    x = _T("attn", shape=(1, 1, 1024, 2048))
    out = operations.apply_output_projection_fused_rs(
        x, SimpleNamespace(o_proj=_T("o_proj", shape=(1, 1, 2048, 6144))), SimpleNamespace(tp_axis=1), ccl
    )
    assert [c[0] for c in calls] == ["mm_rs"]
    _, args, kw = calls[0]
    assert args[0] is x and not x.freed
    assert kw["compute_kernel_config"] is matmul_compute_config(_Dev()) and kw["dtype"] == ttnn.bfloat16
    assert out.name == "rs"


def test_index_projections(monkeypatch):
    calls = _recorder(monkeypatch, "linear")
    monkeypatch.setattr(msa, "_split_index_heads", lambda t, d: t)
    monkeypatch.setattr(msa, "apply_rope", lambda t, *a, **k: t)
    norms = []
    monkeypatch.setattr(msa, "apply_qk_norm_per_head", lambda t, w, eps: norms.append(w) or t)
    w = SimpleNamespace(index_q_proj="wq", index_k_proj="wk", index_q_norm="nq", index_k_norm="nk")
    msa.index_branch_forward(_T("x"), w, None, None, index_dim=128, rms_norm_eps=1e-6)
    assert [c[1][1] for c in calls] == ["wq", "wk"] and norms == ["nq", "nk"]
    assert all(c[2]["compute_kernel_config"] is matmul_compute_config(_Dev()) for c in calls)


def _dense_mlp(monkeypatch):
    mlp = object.__new__(dense_mlp.DenseMLP)
    mlp.__dict__.update(
        mesh_device=_Dev(), gate_proj=_T("g"), up_proj=_T("u"), down_proj=_T("d"), swiglu_cfg=None, hidden_size=6144
    )
    monkeypatch.setattr(dense_mlp, "swiglu", lambda g, u, cfg: _T("act"))
    monkeypatch.setattr(dense_mlp, "apply_swiglu_fused", lambda g, u, cfg, sub_core_grids: _T("act"))
    return mlp


@pytest.mark.parametrize("sub_device", [False, True], ids=["full_grid", "sub_device"])
def test_dense_mlp_and_shared_expert(monkeypatch, sub_device):
    """Both paths, including the sub-device one whose program_config would otherwise drop the default to LoFi."""
    calls = _recorder(monkeypatch, "linear")
    mlp = _dense_mlp(monkeypatch)
    if sub_device:
        monkeypatch.setattr(dense_mlp, "shared_expert_program_configs", lambda *a: ("pg", "pu", "pd"))
        mlp.partial(_T("x"), sub_device=("sd", "cores"), keep_alive=[])
        assert [c[2]["program_config"] for c in calls] == ["pg", "pu", "pd"]
    else:
        mlp.partial(_T("x"))
    assert len(calls) == 3
    for _, _, kw in calls:
        assert kw["compute_kernel_config"] is matmul_compute_config(_Dev()) and kw["dtype"] == ttnn.bfloat16


def test_shared_expert_blocks_fit_fp32_dest():
    """fp32 DEST holds 4 tiles at half-sync; the fp32 partials block is a separate buffer of 2 bf16 tiles per tile."""
    assert DEST_TILES == 4
    assert cb_tiles(4, 2, 3, 8) == 2 * (2 + 3) * 4 + 3 * 2 * 3


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
def test_add_residual_stays_bf16(monkeypatch, dtype):
    calls = _recorder(monkeypatch, "add")
    residual, delta = _T("residual"), _T("delta", dtype=dtype)
    layer.add_residual(residual, delta)
    (_, args, kw) = calls[0]
    assert args == (residual, delta)
    if dtype == ttnn.bfloat16:
        assert kw == {"output_tensor": delta} and not delta.freed
    else:
        assert kw == {"dtype": ttnn.bfloat16} and delta.freed
