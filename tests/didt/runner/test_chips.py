# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from tests.didt.runner.chips import Chip, by_dev, snapshot

TRAY_BUS_HIGH = {1: 0x0, 2: 0x4, 3: 0xC, 4: 0x8}


def _fake_sysfs(root, chips):
    """chips: list of (kmd index, bdf, asic_id or None). Mirrors /sys/class/tenstorrent/tenstorrent!N."""
    sysfs = root / "class" / "tenstorrent"
    sysfs.mkdir(parents=True)
    for kmd, bdf, asic_id in chips:
        pci = root / "devices" / "pci0000:00" / bdf
        pci.mkdir(parents=True)
        node = sysfs / f"tenstorrent!{kmd}"
        node.mkdir()
        (node / "device").symlink_to(pci)
        if asic_id is not None:
            (node / "tt_asic_id").write_text(f"{asic_id}\n")
    return sysfs


def _galaxy(skip=()):
    """32 chips, kmd = (tray-1)*8 + (asic-1), bus = tray high nibble | asic."""
    rows = []
    for tray in range(1, 5):
        for asic in range(1, 9):
            if (tray, asic) in skip:
                continue
            bdf = f"0000:{(TRAY_BUS_HIGH[tray] << 4) | asic:02x}:00.0"
            rows.append(((tray - 1) * 8 + asic - 1, bdf, f"0x{tray:02x}{asic:014x}"))
    return rows


def test_tray_asic_from_bus(tmp_path):
    sysfs = _fake_sysfs(
        tmp_path,
        [(27, "0000:84:00.0", "0x81caba39768f9912"), (16, "0000:c1:00.0", "0x1"), (0, "0000:01:00.0", "0x2")],
    )
    got = {c.bdf: (c.tray, c.asic) for c in snapshot(sysfs)}
    assert got == {"0000:84:00.0": (4, 4), "0000:c1:00.0": (3, 1), "0000:01:00.0": (1, 1)}


def test_dev_is_bdf_rank(tmp_path):
    sysfs = _fake_sysfs(tmp_path, [(5, "0000:c1:00.0", "0x3"), (0, "0000:01:00.0", "0x1"), (9, "0000:41:00.0", "0x2")])
    chips = snapshot(sysfs)
    assert [(c.dev, c.bdf, c.kmd) for c in chips] == [
        (0, "0000:01:00.0", 0),
        (1, "0000:41:00.0", 9),
        (2, "0000:c1:00.0", 5),
    ]


def test_asic_id_is_16_hex_digits(tmp_path):
    sysfs = _fake_sysfs(tmp_path, [(27, "0000:84:00.0", "0x81caba39768f9912"), (0, "0000:01:00.0", "1f")])
    assert [c.asic_id for c in snapshot(sysfs)] == ["0x000000000000001f", "0x81caba39768f9912"]


def test_missing_asic_id_is_omitted(tmp_path):
    sysfs = _fake_sysfs(tmp_path, [(0, "0000:01:00.0", None)])
    (chip,) = snapshot(sysfs)
    assert chip.asic_id is None
    assert chip.as_dict() == {"dev": 0, "bdf": "0000:01:00.0", "tray": 1, "asic": 1, "kmd": 0}


def test_join_with_missing_chip(tmp_path):
    full = by_dev(snapshot(_fake_sysfs(tmp_path / "full", _galaxy())))
    short = by_dev(snapshot(_fake_sysfs(tmp_path / "short", _galaxy(skip={(4, 2)}))))
    assert len(full) == 32 and len(short) == 31
    # BDF order is tray 1 (0x0_), tray 2 (0x4_), tray 4 (0x8_), tray 3 (0xC_); tray 4 asic 2 is the 18th chip.
    gone = full[17]
    assert (gone.tray, gone.asic) == (4, 2)
    for dev in range(17):
        assert short[dev] == full[dev]
    for dev in range(17, 31):
        moved, orig = short[dev], full[dev + 1]
        assert (moved.bdf, moved.tray, moved.asic, moved.asic_id, moved.kmd) == (
            orig.bdf,
            orig.tray,
            orig.asic,
            orig.asic_id,
            orig.kmd,
        )
        assert moved.dev == dev


def test_as_dict_field_order():
    c = Chip(dev=19, bdf="0000:84:00.0", tray=4, asic=4, asic_id="0x81caba39768f9912", kmd=27)
    assert list(c.as_dict()) == ["dev", "bdf", "tray", "asic", "asic_id", "kmd"]


def test_no_sysfs_is_empty(tmp_path):
    assert snapshot(tmp_path / "nothing") == []
