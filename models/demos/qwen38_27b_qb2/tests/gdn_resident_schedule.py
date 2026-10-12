# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated resident-compute scheduling; preserve FP32 operations and ownership."""

MODES = ("native", "hoist_decay", "block_state", "combined")
LOOP = """        for (uint32_t kr = 0; kr < 4; ++kr) {
            const uint32_t state = 4 + kr;
            const uint32_t product = kr == 0 ? 3 : 0;
            load(5, kr, state);
            broadcast<BroadcastType::SCALAR>(3, 0, 1);
            multiply(state, 1, state);
"""


def compute_variant(source, mode):
    if mode not in MODES:
        raise ValueError("Unknown resident scheduling mode")
    if source.count(LOOP) != 1:
        raise ValueError("Resident state-load anchor changed or is ambiguous")
    if mode == "native":
        return source
    prelude, replacement = "", LOOP
    if mode in ("block_state", "combined"):
        prelude += """        // Full DEST already owns slots 4..7. Load four consecutive
        // state tiles with one copy configuration before any SFPU arithmetic.
        reconfig_full_operand<SrcOrder::Regular>(5, 5);
        copy_init(5);
        copy_block(5, 0, 4, 4);
"""
        replacement = replacement.replace("            load(5, kr, state);\n", "")
    if mode in ("hoist_decay", "combined"):
        prelude += """        // Slot 2 is otherwise unused by resident compute. Retain the
        // scalar broadcast across all four separately-rounded decay multiplies.
        broadcast<BroadcastType::SCALAR>(3, 0, 2);
"""
        replacement = replacement.replace("            broadcast<BroadcastType::SCALAR>(3, 0, 1);\n", "")
        replacement = replacement.replace("multiply(state, 1, state)", "multiply(state, 2, state)")
    return source.replace(LOOP, prelude + replacement)
