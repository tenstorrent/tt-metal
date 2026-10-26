# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F194: a box whose cards sit in different IOMMU domains is reported at launch, and a profiling task does not start
on it (DeepSeek-V4.1 8x4 FIX.3: two Galaxy cards `identity` among `DMA-FQ` lost the real-time profiler)."""

from models.demos.common.bringup import orchestrator as O
from models.demos.common.bringup.orchestrator import DONE, STOPPED
from models.demos.common.bringup.selftest.test_orchestrator import orch  # noqa: F401 (fixture)


def _box(tmp_path, kinds):
    root = tmp_path / "class"
    root.mkdir()
    for i, k in enumerate(kinds):
        dev = tmp_path / "pci" / f"0000:{i:02x}:00.0"
        (dev / "iommu_group").mkdir(parents=True)
        (dev / "iommu_group" / "type").write_text(k + "\n")
        (root / f"tenstorrent!{i}").mkdir()
        (root / f"tenstorrent!{i}" / "device").symlink_to(dev)
    return str(root)


def test_the_odd_cards_out_are_named_with_the_fix(tmp_path):
    root = _box(tmp_path, ["DMA-FQ", "DMA-FQ", "identity", "DMA-FQ"])
    assert list(O.iommu_mismatch(root)) == [2]
    assert "/dev/tenstorrent/2" in O.iommu_problem(root) and "Fix (root" in O.iommu_problem(root)


def test_a_uniform_box_is_fine(tmp_path):
    assert O.iommu_problem(_box(tmp_path, ["DMA-FQ"] * 4)) is None
    assert O.iommu_problem(str(tmp_path / "none")) is None


def test_a_profiling_task_does_not_start_on_a_mixed_box(orch, tmp_path, monkeypatch):  # noqa: F811
    from models.demos.common.bringup.selftest.conftest import record_cmd

    monkeypatch.setattr(O, "SYSFS_TT", _box(tmp_path, ["DMA-FQ", "identity", "DMA-FQ"]))
    prof = {"id": "X.1", "title": "profile", "step": "perf", "gate": {"cmd": record_cmd(n=1), "metrics": {"n": "== 1"}}}
    o = orch([prof], {})
    assert o.run() == STOPPED
    assert "IOMMU domains differ" in o.led.state()["X.1"]["reason"][0]
    monkeypatch.setattr(O, "SYSFS_TT", str(tmp_path / "none"))
    o.led.update("X.1", status="TODO")
    assert o.run() == DONE
