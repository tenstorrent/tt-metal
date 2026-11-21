# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-free checks on the VAE decode profile's accounting (`MiniMaxH3Vae._report_profile`)."""

from types import SimpleNamespace

import pytest
from loguru import logger

from ....models.vae.minimax_h3 import vae_minimax_h3 as vm


def _vae(*, synced: bool, log: bool = False):
    vae = object.__new__(vm.MiniMaxH3Vae)
    vae.profile = synced
    vae.log_profile = log
    vae.mesh_device = SimpleNamespace(get_num_devices=lambda: 32)
    profile = vm.MiniMaxH3Vae._empty_profile()
    # A served 5 s clip from run 14516: the host waits 0.24 s on the YUV finish pool.
    profile.update(
        device=0.37,
        readback=0.03,
        readback_join=0.24,
        tiling=0.01,
        upload=0.02,
        host_prep=0.01,
        blend=0.07,
        waves=7,
        units=196,
        readback_mb=301.0,
        readback_each=[0.004] * 7,
        device_each=[0.05] * 7,
    )
    vae._profile = profile
    return vae


@pytest.mark.parametrize("synced", [False, True])
def test_residual_does_not_repeat_readback_join(synced):
    vae = _vae(synced=synced)
    vae._report_profile(0.80)
    p = vae.last_decode_profile
    timed = 0.37 + 0.03 + 0.24 + 0.01 + 0.02 + 0.01 + 0.07
    assert p["residual"] == pytest.approx(0.80 - timed)
    assert p["residual"] < p["readback_join"]
    assert p["total"] == 0.80
    assert p["device_synced"] is synced


def test_residual_floors_at_zero():
    vae = _vae(synced=True)
    vae._report_profile(0.5)
    assert vae.last_decode_profile["residual"] == 0.0


@pytest.mark.parametrize("synced, label", [(False, "device (dispatch)"), (True, "device ")])
def test_log_labels_unsynced_device_as_dispatch(monkeypatch, synced, label):
    monkeypatch.setattr(vm.ttnn, "using_distributed_env", lambda: False)
    lines = []
    sink = logger.add(lambda message: lines.append(str(message)), format="{message}")
    try:
        _vae(synced=synced, log=True)._report_profile(0.80)
    finally:
        logger.remove(sink)
    device_lines = [line for line in lines if line.lstrip().startswith("device")]
    assert len(device_lines) == 2, lines
    assert all(line.lstrip().startswith(label) for line in device_lines), device_lines
    if synced:
        assert not any("dispatch" in line for line in lines)


def test_log_prints_blend(monkeypatch):
    # `blend` (the host clip assembly) was summed into the accounting but never printed, so a slow
    # first clip after a switch read as unexplained decode time.
    monkeypatch.setattr(vm.ttnn, "using_distributed_env", lambda: False)
    lines = []
    sink = logger.add(lambda message: lines.append(str(message)), format="{message}")
    try:
        _vae(synced=True, log=True)._report_profile(0.80)
    finally:
        logger.remove(sink)
    blend = [line for line in lines if line.lstrip().startswith("blend")]
    assert len(blend) == 1 and "0.07 s" in blend[0], lines


def test_reserve_decode_output_maps_the_last_assembled_size(monkeypatch):
    calls = []
    monkeypatch.setattr(vm.host_buffers, "reserve", lambda shape, count=None: calls.append(shape))
    vae = object.__new__(vm.MiniMaxH3Vae)
    vae._assembled_shapes = {(31, 24, 42): (124, 1152, 1344)}
    vae.reserve_decode_output(31, 24, 42)
    vae.reserve_decode_output(61, 24, 42)  # never decoded: nothing to map yet
    assert calls == [(124, 1152, 1344)]
