# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure construction-time architecture composition tests for MLP1D."""

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest

import ttnn
from models.common.modules.mlp import mlp_1d
from models.common.modules.mlp.mlp_1d import MLP1DConfig, _resolve_mlp1d_config, resolve_mlp1d_arch_config


@pytest.fixture(autouse=True)
def _isolate_architecture_resolution(monkeypatch):
    """Keep these pure tests focused on the architecture/SKU resolution stage."""
    monkeypatch.setattr(mlp_1d, "_resolve_mlp1d_config", lambda config: config)


class _FakeMesh:
    def __init__(self, arch, *, dram_width=8, compute_grid=(8, 10), num_devices=4):
        self._arch = arch
        self._dram_width = dram_width
        self._compute_grid = compute_grid
        self._num_devices = num_devices
        self.arch_calls = 0

    def arch(self):
        self.arch_calls += 1
        return self._arch

    def dram_grid_size(self):
        return SimpleNamespace(x=self._dram_width, y=1)

    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=self._compute_grid[0], y=self._compute_grid[1])

    def get_num_devices(self):
        return self._num_devices


class _FakeWeight:
    def __init__(self, shape, device):
        self.source = SimpleNamespace(shape=shape)
        self.device = device


def _common_config(arch, *, dram_width=8):
    mesh = _FakeMesh(arch, dram_width=dram_width)
    w1 = _FakeWeight((5120, 25600), mesh)
    w2 = _FakeWeight((25600, 5120), mesh)
    w3 = _FakeWeight((5120, 25600), mesh)
    return MLP1DConfig(
        w1=w1,
        w2=w2,
        w3=w3,
        mesh_device=mesh,
        dim=5120,
        hidden_dim=25600,
    )


def _kernel(*, fidelity=ttnn.MathFidelity.HiFi2, fp32=False, approximate=False):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=approximate,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
        dst_full_sync_en=False,
    )


def _kernel_semantics(config):
    return (
        config.math_fidelity,
        config.math_approx_mode,
        config.fp32_dest_acc_en,
        config.packer_l1_acc,
        config.dst_full_sync_en,
        config.throttle_level,
    )


@pytest.mark.parametrize(
    "arch,expected_cutoff,dram_width,expected_shard_width",
    [
        (ttnn.device.Arch.WORMHOLE_B0, 1024, 12, 8),
        (ttnn.device.Arch.BLACKHOLE, 512, 8, 8),
        (ttnn.device.Arch.BLACKHOLE, 512, 7, 7),
    ],
)
def test_resolver_selects_architecture_and_effective_sku_defaults(
    arch, expected_cutoff, dram_width, expected_shard_width
):
    common = _common_config(arch, dram_width=dram_width)

    resolved = resolve_mlp1d_arch_config(common)

    assert resolved is not common
    assert resolved.prefill_len_cutoff == expected_cutoff
    assert resolved.prefill_dram_shard_grid_width == expected_shard_width
    assert resolved.prefill_ff1_ff3_grid == (8, 8)
    assert resolved.prefill_ff2_grid == (8, 8)
    assert common.mesh_device.arch_calls == 1


def test_model_cutoff_precedes_sku_default_without_mutating_common_config():
    common = _common_config(ttnn.device.Arch.BLACKHOLE)
    resolved = resolve_mlp1d_arch_config(replace(common, prefill_len_cutoff=256))

    assert resolved.prefill_len_cutoff == 256
    assert common.prefill_len_cutoff is None
    for field in (
        "ff1_3_compute_kernel_cfg",
        "ff2_compute_kernel_cfg",
        "decode_ff1_3_compute_kernel_cfg",
        "decode_ff2_compute_kernel_cfg",
    ):
        assert field in common.__dataclass_fields__
        assert getattr(common, field) is None


def test_four_explicit_common_slots_preserve_semantics_and_are_independent():
    common = _common_config(ttnn.device.Arch.BLACKHOLE)
    supplied = replace(
        common,
        ff1_3_compute_kernel_cfg=_kernel(fidelity=ttnn.MathFidelity.HiFi4, fp32=True),
        ff2_compute_kernel_cfg=_kernel(fidelity=ttnn.MathFidelity.LoFi),
        decode_ff1_3_compute_kernel_cfg=_kernel(approximate=True),
        decode_ff2_compute_kernel_cfg=_kernel(fidelity=ttnn.MathFidelity.HiFi4),
    )
    common.mesh_device.arch_calls = 0

    resolved = resolve_mlp1d_arch_config(supplied)

    slot_names = (
        "ff1_3_compute_kernel_cfg",
        "ff2_compute_kernel_cfg",
        "decode_ff1_3_compute_kernel_cfg",
        "decode_ff2_compute_kernel_cfg",
    )
    assert common.mesh_device.arch_calls == 1
    assert [_kernel_semantics(getattr(resolved, name)) for name in slot_names] == [
        _kernel_semantics(getattr(supplied, name)) for name in slot_names
    ]
    assert all(getattr(resolved, name) is not getattr(supplied, name) for name in slot_names)
    assert len({id(getattr(resolved, name)) for name in slot_names}) == 4


def test_independent_resolutions_do_not_share_compute_configs():
    common = _common_config(ttnn.device.Arch.BLACKHOLE)

    first = resolve_mlp1d_arch_config(common)
    second = resolve_mlp1d_arch_config(common)

    assert first is not second
    assert first.ff1_3_compute_kernel_cfg is not second.ff1_3_compute_kernel_cfg
    assert first.decode_ff2_compute_kernel_cfg is not second.decode_ff2_compute_kernel_cfg
    first_slots = (
        first.ff1_3_compute_kernel_cfg,
        first.ff2_compute_kernel_cfg,
        first.decode_ff1_3_compute_kernel_cfg,
        first.decode_ff2_compute_kernel_cfg,
    )
    assert len({id(config) for config in first_slots}) == 4


def test_resolver_returns_only_common_config_state():
    resolved = resolve_mlp1d_arch_config(_common_config(ttnn.device.Arch.BLACKHOLE))

    assert isinstance(resolved, MLP1DConfig)
    assert not hasattr(resolved, "arch")
    assert not hasattr(resolved, "mlp")


def test_illegal_blackhole_common_overrides_fail_closed(expect_error):
    base = _common_config(ttnn.device.Arch.BLACKHOLE, dram_width=7)

    with expect_error(ValueError, "positive multiple"):
        resolve_mlp1d_arch_config(replace(base, prefill_len_cutoff=0))
    with expect_error(ValueError, "does not match the resolved architecture/SKU"):
        resolve_mlp1d_arch_config(replace(base, prefill_dram_shard_grid_width=8))
    with expect_error(ValueError, "exceeds mesh compute grid"):
        resolve_mlp1d_arch_config(replace(base, prefill_ff2_grid=(9, 8)))
    with expect_error(ValueError, "missing fields"):
        resolve_mlp1d_arch_config(
            replace(base, decode_ff2_compute_kernel_cfg=SimpleNamespace(math_fidelity=ttnn.MathFidelity.HiFi2))
        )


def test_unsupported_architecture_fails_before_compute_config_construction(expect_error):
    common = _common_config(None)

    with expect_error(ValueError, "Unsupported MLP1D architecture"):
        resolve_mlp1d_arch_config(common)
    assert common.mesh_device.arch_calls == 1


def test_deferred_config_factories_contain_no_architecture_queries():
    source = inspect.getsource(_resolve_mlp1d_config)

    assert ".arch(" not in source
    assert "is_blackhole" not in source
    assert "get_arch_name" not in source
