# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure construction tests for Attention1D architecture composition."""

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest

import ttnn
from models.common.modules.attention import attention_1d
from models.common.modules.attention.attention_1d import Attention1DConfig, resolve_attention1d_arch_config


class _Mesh:
    def __init__(self, arch, dram_width=8):
        self._arch = arch
        self._dram_width = dram_width
        self.arch_calls = 0

    def arch(self):
        self.arch_calls += 1
        return self._arch

    def dram_grid_size(self):
        return SimpleNamespace(x=self._dram_width, y=1)

    def compute_with_storage_grid_size(self):
        return ttnn.CoreCoord(8, 10 if self._arch == "blackhole" else 8)


def _common(mesh, slots):
    return Attention1DConfig(
        wqkv=SimpleNamespace(device=mesh),
        wo=SimpleNamespace(device=mesh),
        mesh_device=mesh,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        **slots,
    )


def _slots():
    return {
        name: ttnn.init_device_compute_kernel_config(
            ttnn.device.Arch.BLACKHOLE,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        for name in attention_1d._ATTENTION_COMPUTE_SLOT_NAMES
    }


def _passthrough_resolver(config, *, _resolved_fields=None, **_):
    return replace(config, **(_resolved_fields or {}))


@pytest.mark.parametrize(
    ("architecture", "qkv_grid", "dram_width", "create_head_grid"),
    [
        ("wormhole", (8, 8), 8, None),
        ("blackhole", (8, 10), 7, (8, 4)),
    ],
)
def test_resolve_attention_arch_config_selects_internal_state_without_mutating_common(
    monkeypatch, architecture, qkv_grid, dram_width, create_head_grid
):
    mesh = _Mesh(architecture, dram_width=dram_width)
    slots = _slots()
    common = _common(mesh, {})
    before = dict(common.__dict__)
    monkeypatch.setattr(attention_1d, "_attention_architecture", lambda *_: architecture)
    monkeypatch.setattr(attention_1d, "_resolve_attention1d_config", _passthrough_resolver)
    monkeypatch.setattr(attention_1d, "_shared_attention_compute_defaults", lambda _: slots)
    supplied = common

    resolved = resolve_attention1d_arch_config(supplied)

    assert isinstance(resolved, Attention1DConfig)
    assert resolved is not common
    assert resolved.prefill_qkv_grid == qkv_grid
    assert resolved.dram_shard_grid_width == dram_width
    assert common.__dict__ == before
    assert mesh.arch_calls == 1
    if create_head_grid is None:
        assert resolved.decode_create_qkv_head_grid is None
    else:
        assert (resolved.decode_create_qkv_head_grid.x, resolved.decode_create_qkv_head_grid.y) == create_head_grid
    for name, value in slots.items():
        assert getattr(resolved, name) is not value


def test_resolve_attention_arch_config_returns_only_common_config(monkeypatch):
    mesh = _Mesh("blackhole")
    slots = _slots()
    common = _common(mesh, {})
    monkeypatch.setattr(attention_1d, "_attention_architecture", lambda *_: "blackhole")
    monkeypatch.setattr(attention_1d, "_shared_attention_compute_defaults", lambda _: slots)
    monkeypatch.setattr(attention_1d, "_resolve_attention1d_config", _passthrough_resolver)
    resolved = resolve_attention1d_arch_config(common)

    assert isinstance(resolved, Attention1DConfig)
    assert not hasattr(attention_1d, "_ResolvedAttention1DConfig")


def test_shared_attention_compute_defaults_preserve_six_explicit_slots(monkeypatch):
    calls = []

    def init_kernel(arch, **kwargs):
        value = (arch, kwargs)
        calls.append(value)
        return value

    monkeypatch.setattr(ttnn, "init_device_compute_kernel_config", init_kernel)
    defaults = attention_1d._shared_attention_compute_defaults("wormhole")

    assert set(defaults) == set(attention_1d._ATTENTION_COMPUTE_SLOT_NAMES)
    ordinary = defaults["li_qkv_decode_compute_kernel_cfg"]
    assert ordinary[1] == {
        "math_fidelity": ttnn.MathFidelity.HiFi2,
        "math_approx_mode": False,
        "fp32_dest_acc_en": False,
        "packer_l1_acc": True,
    }
    assert all(
        defaults[name] is not ordinary
        for name in attention_1d._ATTENTION_COMPUTE_SLOT_NAMES
        if name not in ("li_qkv_decode_compute_kernel_cfg", "sdpa_prefill_compute_kernel_cfg")
    )
    assert defaults["sdpa_prefill_compute_kernel_cfg"][1] == {
        "math_fidelity": ttnn.MathFidelity.HiFi4,
        "math_approx_mode": False,
        "fp32_dest_acc_en": True,
        "packer_l1_acc": True,
    }
    assert len(calls) == 6


def test_compute_slots_live_on_resolved_common_config(monkeypatch):
    common_fields = Attention1DConfig.__dataclass_fields__
    assert set(attention_1d._ATTENTION_COMPUTE_SLOT_NAMES) <= set(common_fields)
    assert all(common_fields[name].default is None for name in attention_1d._ATTENTION_COMPUTE_SLOT_NAMES)

    mesh = _Mesh("wormhole")
    common = _common(mesh, {})
    monkeypatch.setattr(attention_1d, "_attention_architecture", lambda *_: "wormhole")
    monkeypatch.setattr(attention_1d, "_shared_attention_compute_defaults", lambda _: _slots())
    monkeypatch.setattr(attention_1d, "_resolve_attention1d_config", _passthrough_resolver)
    resolved = resolve_attention1d_arch_config(common)
    assert isinstance(resolved, Attention1DConfig)
    assert resolved.dram_shard_grid_width == 8


def test_explicit_common_recipe_and_sku_overlay_take_precedence(monkeypatch):
    mesh = _Mesh("blackhole", dram_width=7)
    common = _common(mesh, {})
    slots = _slots()
    supplied = replace(
        common,
        **slots,
        prefill_qkv_grid=(6, 9),
        dram_shard_grid_width=7,
        decode_create_qkv_head_grid=ttnn.CoreGrid(y=3, x=6),
        decode_transformation_core_grid=ttnn.CoreCoord(6, 9),
    )
    monkeypatch.setattr(attention_1d, "_attention_architecture", lambda *_: "blackhole")
    monkeypatch.setattr(attention_1d, "_shared_attention_compute_defaults", lambda _: _slots())
    monkeypatch.setattr(attention_1d, "_resolve_attention1d_config", _passthrough_resolver)

    resolved = resolve_attention1d_arch_config(supplied)

    assert resolved.prefill_qkv_grid == (6, 9)
    assert resolved.dram_shard_grid_width == 7
    for name, value in slots.items():
        assert getattr(resolved, name) is not value


def test_explicit_common_invalid_compute_slot_fails_closed(monkeypatch, expect_error):
    mesh = _Mesh("wormhole")
    slots = _slots()
    slots["sdpa_prefill_compute_kernel_cfg"] = SimpleNamespace(math_fidelity=ttnn.MathFidelity.HiFi2)
    supplied = replace(
        _common(mesh, {}),
        **slots,
    )
    monkeypatch.setattr(attention_1d, "_attention_architecture", lambda *_: "wormhole")
    monkeypatch.setattr(attention_1d, "_shared_attention_compute_defaults", lambda _: _slots())

    with expect_error(ValueError, "sdpa_prefill_compute_kernel_cfg"):
        resolve_attention1d_arch_config(supplied)


def test_explicit_common_illegal_geometry_fails_closed(monkeypatch, expect_error):
    mesh = _Mesh("blackhole", dram_width=7)
    common = _common(mesh, {})
    monkeypatch.setattr(attention_1d, "_attention_architecture", lambda *_: "blackhole")
    monkeypatch.setattr(attention_1d, "_shared_attention_compute_defaults", lambda _: _slots())

    with expect_error(ValueError, "does not match resolved width"):
        resolve_attention1d_arch_config(replace(common, dram_shard_grid_width=8))
    with expect_error(ValueError, "prefill QKV grid"):
        resolve_attention1d_arch_config(replace(common, prefill_qkv_grid=(9, 10)))
    with expect_error(ValueError, "decode_create_qkv_head_grid"):
        resolve_attention1d_arch_config(replace(common, decode_create_qkv_head_grid=ttnn.CoreGrid(x=9, y=4)))
    with expect_error(ValueError, "decode_transformation_core_grid"):
        resolve_attention1d_arch_config(replace(common, decode_transformation_core_grid=ttnn.CoreCoord(8, 11)))


def test_attention_architecture_rejects_unsupported_mesh(monkeypatch, expect_error):
    mesh = _Mesh("unsupported")

    with expect_error(ValueError, "Unsupported Attention1D architecture"):
        attention_1d._attention_architecture(mesh, "unsupported")


def test_blackhole_common_config_uses_shared_baseline_and_mesh_sku_overlay(monkeypatch):
    mesh = _Mesh("blackhole")
    common = _common(mesh, {})
    slots = _slots()
    monkeypatch.setattr(attention_1d, "_attention_architecture", lambda *_: "blackhole")
    monkeypatch.setattr(attention_1d, "_shared_attention_compute_defaults", lambda _: slots)
    monkeypatch.setattr(attention_1d, "_resolve_attention1d_config", _passthrough_resolver)

    resolved = resolve_attention1d_arch_config(common)

    assert resolved.prefill_qkv_grid == (8, 10)
    assert resolved.dram_shard_grid_width == 8
    for name, value in slots.items():
        assert getattr(resolved, name) is not value


def test_already_resolved_architecture_avoids_second_mesh_query(monkeypatch):
    mesh = _Mesh("wormhole")
    common = _common(mesh, {})
    slots = _slots()
    monkeypatch.setattr(attention_1d, "_attention_architecture", lambda *_: "wormhole")
    monkeypatch.setattr(attention_1d, "_shared_attention_compute_defaults", lambda _: slots)
    monkeypatch.setattr(attention_1d, "_resolve_attention1d_config", _passthrough_resolver)

    resolve_attention1d_arch_config(common, _arch="wormhole")

    assert mesh.arch_calls == 0


def test_internal_config_resolution_contains_no_architecture_query():
    source = inspect.getsource(attention_1d._resolve_attention1d_config)

    assert ".arch(" not in source
    assert "is_blackhole" not in source
    assert "get_arch_name" not in source
