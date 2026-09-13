"""Unit tests for the pure-CPU accelerator-package stand-ins."""

import importlib
import sys

import pytest

torch = pytest.importorskip("torch")

from scripts.tt_hw_planner import cpu_compat


def _purge_mamba_ssm_from_sys_modules():
    for name in list(sys.modules):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            del sys.modules[name]


def test_install_makes_mamba_ssm_importable():
    _purge_mamba_ssm_from_sys_modules()
    installed = cpu_compat.install_cpu_compat()
    assert "mamba_ssm" in installed
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
    )

    assert callable(rmsnorm_fn)
    assert callable(selective_state_update)
    assert callable(mamba_chunk_scan_combined)
    assert callable(mamba_split_conv1d_scan_combined)


def test_install_is_idempotent_when_already_stubbed():
    _purge_mamba_ssm_from_sys_modules()
    cpu_compat.install_cpu_compat()
    second = cpu_compat.install_cpu_compat()
    assert second == []


def test_rmsnorm_fn_matches_reference_no_gate():
    _purge_mamba_ssm_from_sys_modules()
    cpu_compat.install_cpu_compat()
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn

    torch.manual_seed(0)
    x = torch.randn(2, 4, 16)
    w = torch.randn(16)
    out = rmsnorm_fn(x, w, eps=1e-5)
    rstd = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
    ref = (x.float() * rstd * w.float()).to(x.dtype)
    assert torch.allclose(out, ref, atol=1e-5)


def test_rmsnorm_fn_applies_silu_gate():
    _purge_mamba_ssm_from_sys_modules()
    cpu_compat.install_cpu_compat()
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn

    torch.manual_seed(1)
    x = torch.randn(3, 8)
    w = torch.ones(8)
    z = torch.randn(3, 8)
    out = rmsnorm_fn(x, w, z=z, eps=1e-6, norm_before_gate=False)
    import torch.nn.functional as F

    xg = x.float() * F.silu(z.float())
    rstd = torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = (xg * rstd * w.float()).to(x.dtype)
    assert torch.allclose(out, ref, atol=1e-5)


def test_fast_path_stub_raises_when_called():
    _purge_mamba_ssm_from_sys_modules()
    cpu_compat.install_cpu_compat()
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    with pytest.raises(RuntimeError):
        mamba_chunk_scan_combined(1, 2, 3)


def test_does_not_shadow_a_genuinely_installed_package():
    fake = importlib.util.spec_from_loader("tt_fake_real_pkg", loader=None)
    mod = importlib.util.module_from_spec(fake)
    sys.modules["tt_fake_real_pkg"] = mod
    try:
        assert cpu_compat._genuinely_importable("tt_fake_real_pkg") is True
    finally:
        del sys.modules["tt_fake_real_pkg"]
