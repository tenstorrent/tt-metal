# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ensure the experiment fails closed if the shared reader changes."""

import pytest

from models.demos.qwen38_27b_qb2.tests.gdn_resident_padding import (
    INITIALIZATION,
    MODES,
    OVERLAP_ANCHORS,
    reader_variant,
)
from models.demos.qwen38_27b_qb2.tt.gdn_step.op import kernel_source


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("source", ["", INITIALIZATION * 2])
def test_changed_reader_is_rejected(mode, source, expect_error):
    with expect_error(ValueError, "anchor changed or is ambiguous"):
        reader_variant(source, mode)


def test_only_initialization_changes():
    source = kernel_source("reader.cpp")
    assert reader_variant(source, "zero") == source
    before, after = source.split(INITIALIZATION)
    for mode in ("skip", "poison"):
        variant = reader_variant(source, mode)
        assert variant.startswith(before) and variant.endswith(after)
    assert "vr[i] =" not in reader_variant(source, "skip")
    assert "vr[i] = 0x7fc12345" in reader_variant(source, "poison")


@pytest.mark.parametrize("mode", [None, "", "garbage"])
def test_unknown_mode_rejected(mode, expect_error):
    with expect_error(ValueError, "Padding mode must"):
        reader_variant(kernel_source("reader.cpp"), mode)


@pytest.mark.parametrize("mode", ["overlap_zero", "overlap_skip", "overlap_poison"])
def test_vector_and_state_dma_have_distinct_completion_boundaries(mode):
    source = reader_variant(kernel_source("reader.cpp"), mode)
    assert source.count("noc_async_read_barrier();") == 2
    vector_done = source.index("noc_async_read_barrier();")
    state_read = source.index("noc_async_read_tile(")
    format_vectors = source.index("const auto* values")
    state_done = source.rindex("noc_async_read_barrier();")
    published = source.index("cb_push_back(")
    assert vector_done < state_read < format_vectors < state_done < published


@pytest.mark.parametrize("anchor", OVERLAP_ANCHORS)
def test_overlap_rejects_changed_dma_or_publication_boundary(anchor, expect_error):
    source = kernel_source("reader.cpp").replace(anchor, "")
    with expect_error(ValueError, "overlap anchor changed or is ambiguous"):
        reader_variant(source, "overlap_zero")
