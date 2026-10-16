# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The CPU-staged BF4 corpus pins agree with the cache's packed layout: the binder's and the verifier's per-artifact
byte counts are the tensorbin of the canonical slot shape (the 8-byte prefix, the serialized header, the BF4 payload
of four devices), and the corpus total is the 49 slots' sum of those pins, not a literal of its own."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from models.demos.blackhole.qwen38_flash_next import diagnostic_bf4
from models.demos.blackhole.qwen38_flash_next.ttnn import bf4

MESH_DEVICES = 4
TENSORBIN_HEADER_BYTES = 1240  # the serialized header of a slot tensorbin, measured on the written cache (2026-09-25)
COMPACT_CORPUS_BYTES = 69_363_424_704  # 49 x (943,719,648 + 471,860,448)


def _verify_tool():
    path = Path(diagnostic_bf4.__file__).with_name("tools") / "verify_full_bf4_cpu.py"
    spec = importlib.util.spec_from_file_location("verify_full_bf4_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_binder_artifact_pins_are_the_packed_layouts_tensorbins():
    shapes = bf4.canonical_packed_shapes(ring_size=diagnostic_bf4.EXPECTED_RING_SIZE)
    per_device = dict(
        zip(("w0_w1", "w2"), bf4.packed_bf4_bytes_per_device(ring_size=diagnostic_bf4.EXPECTED_RING_SIZE))
    )
    assert set(diagnostic_bf4.ARTIFACT_SPECS) == {"w0_w1", "w2"}
    for name, spec in diagnostic_bf4.ARTIFACT_SPECS.items():
        assert spec.global_shape == shapes[name]
        assert spec.local_shape == (*shapes[name][:2], shapes[name][2] // MESH_DEVICES, *shapes[name][3:])
        payload = MESH_DEVICES * per_device[name]
        assert spec.bytes == bf4.TENSORBIN_HEADER_PREFIX_BYTES + TENSORBIN_HEADER_BYTES + payload
        assert TENSORBIN_HEADER_BYTES % bf4.TENSORBIN_HEADER_ALIGNMENT == 0
    assert diagnostic_bf4.ARTIFACT_SPECS["w0_w1"].bytes == 943_719_648
    assert diagnostic_bf4.ARTIFACT_SPECS["w2"].bytes == 471_860_448


def test_the_corpus_total_is_the_49_slots_sum_of_the_artifact_pins():
    assert len(diagnostic_bf4.EXPECTED_SLOTS) == 49
    per_slot = sum(spec.bytes for spec in diagnostic_bf4.ARTIFACT_SPECS.values())
    assert diagnostic_bf4.EXPECTED_TOTAL_BYTES == 49 * per_slot == COMPACT_CORPUS_BYTES
    assert diagnostic_bf4.EXPECTED_TOTAL_BYTES != 106_819_608_000  # the pre-compaction corpus (49 x 2,179,992,000)


def test_the_verifier_pins_the_same_artifacts_and_total():
    verify = _verify_tool()
    assert verify.EXPECTED_SLOTS == diagnostic_bf4.EXPECTED_SLOTS
    assert set(verify.ARTIFACT_SPECS) == set(diagnostic_bf4.ARTIFACT_SPECS)
    for name, spec in diagnostic_bf4.ARTIFACT_SPECS.items():
        record = verify.ARTIFACT_SPECS[name]
        assert record["filename"] == spec.filename
        assert record["bytes"] == spec.bytes
        assert tuple(record["local_shape"]) == spec.local_shape
        assert tuple(record["global_shape"]) == spec.global_shape
    total = len(verify.EXPECTED_SLOTS) * sum(record["bytes"] for record in verify.ARTIFACT_SPECS.values())
    assert total == diagnostic_bf4.EXPECTED_TOTAL_BYTES == COMPACT_CORPUS_BYTES
    source = Path(verify.__file__).read_text(encoding="utf-8")
    assert "106_819_608_000" not in source and "106819608000" not in source
