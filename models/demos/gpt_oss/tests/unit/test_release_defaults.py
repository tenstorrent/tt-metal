# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of where the gpt-oss-20b release settings switch on (models/demos/gpt_oss/__init__.py): a QuietBox 2
(exactly 4 Blackhole chips visible) by default, any machine with GPT_OSS_RELEASE_DEFAULTS=1, none with =0. The device
nodes and sysfs are faked in tmp_path."""

import pytest

import models.demos.gpt_oss as gpt_oss

BLACKHOLE, WORMHOLE = "0xb140", "0x401e"


def _machine(tmp_path, chips):
    """Fake /dev/tenstorrent and /sys/class/tenstorrent with one entry per chip id (chips: {id: PCI device id})."""
    dev, sys = tmp_path / "dev", tmp_path / "sys"
    dev.mkdir()
    for i, pci in chips.items():
        (dev / str(i)).write_text("")
        d = sys / f"tenstorrent!{i}" / "device"
        d.mkdir(parents=True)
        (d / "device").write_text(pci + "\n")
    (dev / "by-id").mkdir()
    return str(dev), str(sys)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for k in [*gpt_oss.RELEASE_DEFAULTS_20B, "GPT_OSS_RELEASE_DEFAULTS", "TT_VISIBLE_DEVICES"]:
        monkeypatch.setenv(k, "")  # records the value to restore, so what apply_release_defaults sets is undone
        monkeypatch.delenv(k)
    ckpt = tmp_path / "gpt-oss-20b"
    ckpt.mkdir()
    (ckpt / "model.safetensors.index.json").write_text("{}")
    monkeypatch.setenv("HF_MODEL", str(ckpt))


@pytest.mark.parametrize(
    "chips, expected",
    [
        ({i: BLACKHOLE for i in range(4)}, 4),  # QuietBox 2
        ({0: BLACKHOLE}, 1),  # P150
        ({0: BLACKHOLE, 1: BLACKHOLE}, 2),  # P300
        ({i: BLACKHOLE for i in range(8)}, 8),  # LoudBox
        ({i: WORMHOLE for i in range(8)}, 0),  # LLMBox
        ({}, 0),
    ],
)
def test_visible_blackhole_chips(tmp_path, chips, expected):
    assert gpt_oss.visible_blackhole_chips(*_machine(tmp_path, chips)) == expected


def test_visible_devices_narrow_the_set(tmp_path, monkeypatch):
    """QuietBox 2 emulation on a Blackhole Galaxy: 4 of its 32 chips."""
    machine = _machine(tmp_path, {i: BLACKHOLE for i in range(32)})
    assert gpt_oss.visible_blackhole_chips(*machine) == 32
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "0,1,4,5")
    assert gpt_oss.visible_blackhole_chips(*machine) == 4


def test_no_device_nodes(tmp_path):
    assert gpt_oss.visible_blackhole_chips(str(tmp_path / "missing"), str(tmp_path / "missing")) == 0


@pytest.mark.parametrize("quietbox_2", [True, False])
def test_default_on_quietbox_2_only(monkeypatch, quietbox_2):
    monkeypatch.setattr(gpt_oss, "is_quietbox_2", lambda: quietbox_2)
    applied = gpt_oss.apply_release_defaults()
    assert bool(applied) == quietbox_2
    if quietbox_2:
        assert set(applied) == set(gpt_oss.RELEASE_DEFAULTS_20B)


def test_forced_on_any_machine(monkeypatch):
    monkeypatch.setattr(gpt_oss, "is_quietbox_2", lambda: False)
    monkeypatch.setenv("GPT_OSS_RELEASE_DEFAULTS", "1")
    monkeypatch.setenv("GPT_OSS_BOUNDARY_FIDELITY", "HiFi2")  # a variable set already is kept
    applied = gpt_oss.apply_release_defaults()
    assert "GPT_OSS_BOUNDARY_FIDELITY" not in applied and len(applied) == len(gpt_oss.RELEASE_DEFAULTS_20B) - 1


def test_off_and_model_rules(monkeypatch, tmp_path):
    monkeypatch.setattr(gpt_oss, "is_quietbox_2", lambda: True)
    monkeypatch.setenv("GPT_OSS_RELEASE_DEFAULTS", "0")
    assert gpt_oss.apply_release_defaults() == []
    monkeypatch.setenv("GPT_OSS_RELEASE_DEFAULTS", "1")
    config_only = tmp_path / "configs" / "gpt-oss-20b"
    config_only.mkdir(parents=True)
    monkeypatch.setenv("HF_MODEL", str(config_only))  # random-weight unit-test directory
    assert gpt_oss.apply_release_defaults() == []
    monkeypatch.setenv("HF_MODEL", "openai/gpt-oss-120b")
    assert gpt_oss.apply_release_defaults() == []
