# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in native BF16 persistence experiment; no TTNN import or default change.

This removes explicit integer RNE before the existing BF16 pack. Numerical
identity is unproven, including persisted state. Current FP32 readout arithmetic,
reader/writer, descriptors, state addresses and CB synchronization are unchanged.
"""

from contextlib import contextmanager

from . import recurrent_variants

VARIANT = "cumulative_native_pack"
BASELINE = "cumulative_exp_scalar"
ORIGINAL_BODY = """    TTI_SFPMOV(0,DATA,LREG_MASK,0);
    TTI_SFPSHFT((-16)&0xFFF,DATA,LREG_TMP,5);
    TTI_SFPAND(0,p_sfpu::LREG12,LREG_TMP,0);
    TTI_SFPIADD(0,p_sfpu::LREG13,LREG_MASK,sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0,LREG_TMP,LREG_MASK,sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPAND(0,p_sfpu::LREG14,LREG_MASK,0);
    TT_SFPSTORE(LREG_MASK,InstrModLoadStore::FP32,ADDR_MOD_7,address);"""
NATIVE_BODY = "    TT_SFPSTORE(DATA,InstrModLoadStore::FP32,ADDR_MOD_7,address);"
PREFIX = "inline void _gdn_store_bf16_rne_(uint32_t address) {\n"


def native_pack_sources(reader, compute):
    """Transform an already adapted cumulative_exp_scalar source pair only."""
    before = PREFIX + ORIGINAL_BODY + "\n}"
    after = PREFIX + NATIVE_BODY + "\n}"
    assert compute.count(before) == 1, "Frozen BF16 helper drift"
    return reader, compute.replace(before, after, 1)


@contextmanager
def register_variant():
    """Register once for an isolated CLI process, restore at exit (no per-call rebinding).

    Do not use concurrently with another experiment in the same Python process.
    Registration changes neither existing variant source text nor model defaults.
    """
    original_sources, original_variants = recurrent_variants.sources, recurrent_variants.VARIANTS
    assert VARIANT not in original_variants, "Native-pack variant already registered"
    evidence = {"candidate_source_calls": 0, "other_source_calls": 0}

    def sources(reader, compute, variant="baseline", *, grouped=True, poison_dead_padding=False):
        candidate = variant == VARIANT
        result = original_sources(
            reader,
            compute,
            BASELINE if candidate else variant,
            grouped=grouped,
            poison_dead_padding=poison_dead_padding,
        )
        evidence["candidate_source_calls" if candidate else "other_source_calls"] += 1
        return native_pack_sources(*result) if candidate else result

    recurrent_variants.VARIANTS = (*original_variants, VARIANT)
    recurrent_variants.sources = sources
    try:
        yield evidence
    finally:
        recurrent_variants.sources = original_sources
        recurrent_variants.VARIANTS = original_variants
