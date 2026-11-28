# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from tests.didt.didt_chip import DEVICE_ID_ENV, pick_submesh


class Submesh:
    def __init__(self, dev):
        self.dev = dev

    def get_device_ids(self):
        return [self.dev]


# 1x1 submesh order follows the mesh, not the device id: index 20 can be device 26.
SUBMESHES = [Submesh(d) for d in (3, 26, 16, 0)]


def test_chip_id_is_device_id_without_env(monkeypatch):
    monkeypatch.delenv(DEVICE_ID_ENV, raising=False)
    assert pick_submesh(SUBMESHES, 26).dev == 26  # the 2nd submesh, found by its id
    assert pick_submesh(SUBMESHES, 0).dev == 0  # the last submesh, not the first


def test_chip_id_not_in_mesh_fails_loudly(monkeypatch):
    monkeypatch.delenv(DEVICE_ID_ENV, raising=False)
    try:
        pick_submesh(SUBMESHES, 1)
    except ValueError as e:
        assert "device 1" in str(e) and "[0, 3, 16, 26]" in str(e)
    else:
        raise AssertionError("no error for a device the mesh does not have")


def test_device_id_env_selects_by_device(monkeypatch):
    monkeypatch.setenv(DEVICE_ID_ENV, "16")
    assert pick_submesh(SUBMESHES, 0).dev == 16


def test_unknown_device_id_fails_loudly(monkeypatch):
    monkeypatch.setenv(DEVICE_ID_ENV, "31")
    try:
        pick_submesh(SUBMESHES, 0)
    except ValueError as e:
        assert "31" in str(e) and "[0, 3, 16, 26]" in str(e)
    else:
        raise AssertionError("no error for a device the mesh does not have")
