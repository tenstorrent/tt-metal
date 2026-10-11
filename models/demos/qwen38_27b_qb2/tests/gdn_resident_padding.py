# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated reader experiment; leave the production reader and its ABI unchanged."""

INITIALIZATION = """        // V's unused rows must be zero for the SFPU residual. Q/K column
        // broadcasts and scalar broadcasts never consume the other lanes.
        for (uint32_t i = 0; i < value_columns * 1024; ++i) {
            vr[i] = 0;
        }
"""

MODES = ("zero", "skip", "poison", "overlap_zero", "overlap_skip", "overlap_poison")
OVERLAP_ANCHORS = (
    "        const uint32_t state_l1 = get_write_ptr(5);\n",
    "        noc_async_read_barrier();\n        const auto* values",
    "        cb_push_back(0, 4);\n",
)


def reader_variant(source, mode):
    """Only resident, pre-normalized, one-column callers may use these sources.

    The resident SFPU row broadcast consumes delta row zero only. Poison the
    other rows to test that assumption; skip their stores only after that test.
    Source text is part of generic_op's program cache key.
    """
    if mode not in MODES:
        raise ValueError("Padding mode must be zero, skip, or poison, optionally prefixed with overlap_")
    if source.count(INITIALIZATION) != 1:
        raise ValueError("Reader initialization anchor changed or is ambiguous")
    overlap = mode.startswith("overlap_")
    padding = mode.removeprefix("overlap_")
    if padding != "zero":
        replacement = "        // Experimental resident path: only V row zero is consumed.\n"
        if padding == "poison":
            replacement += """        for (uint32_t i = 0; i < value_columns * 1024; ++i) {
            vr[i] = 0x7fc12345;  // Quiet NaN; live row is overwritten below.
        }
"""
        source = source.replace(INITIALIZATION, replacement)
    if overlap:
        if any(source.count(anchor) != 1 for anchor in OVERLAP_ANCHORS):
            raise ValueError("Reader overlap anchor changed or is ambiguous")
        # Finish vector DMA before reading scratch. State DMA uses separate
        # reserved CB5 storage and can run while vectors are formatted. Finish
        # state DMA before publishing any inputs. No buffer lifetime changes.
        source = source.replace(OVERLAP_ANCHORS[1], "        const auto* values")
        source = source.replace(
            OVERLAP_ANCHORS[0], "        noc_async_read_barrier();  // Vectors ready.\n" + OVERLAP_ANCHORS[0]
        )
        source = source.replace(
            OVERLAP_ANCHORS[2],
            "        noc_async_read_barrier();  // State ready before publication.\n" + OVERLAP_ANCHORS[2],
        )
    return source
