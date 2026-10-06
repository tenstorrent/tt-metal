"""Prompt-length independent prefill chunk rule (DSV41_PREFILL_ROW_TOKENS=auto2, default OFF; the current rule stays the default).

One row-token budget per users-per-row U, fixed at startup; chunk per user C = budget // U (a multiple of 128). With auto2 the chunk trace is sized once for the
build's max context (DSV41_PREFILL_SPAD_MAX defaults to the model's max_ctx) and C is never collapsed to a prompt-sized single chunk, so a later (longer or
shorter) prompt replays the same capture instead of re-capturing / OOMing. A prompt shorter than C is padded to ONE full chunk (cost = a full chunk of replay).
Table = largest budget that fits next to the 64k-sized trace on a 32-chip BH Galaxy (bf16 pool, 40 layers, Engram on) and is within ~3% of the best
measured ms/row-token, see PREFILL_CHUNKCAL_NOTES.md. U = 8 (B = 32) and U = 2 (B = 8) use the replicated MoE path (no column split), U in {1, 4, 16} the column split.
"""
import os

# U -> budget (tokens per mesh row per pass)
BASE_TABLE = {1: 2048, 2: 4096, 4: 4096, 8: 1024, 16: 2048}
UNIFIED_TABLE = {1: 1024, 4: 8192, 16: 2048}  # DSV41_PREFILL_MOE=unified (column-split layers only: U in {1, 4, 16})
MIN_FREE_MIB = 450.0  # free DRAM per bank after the model build below which the budget is halved (one step) to keep the compile-pass headroom


def auto2_budget(U, free_dram_mib=None, unified=None):
    """Row-token budget (tokens per mesh row per traced-chunk pass) for U users per mesh row; chunk per user = budget // U."""
    if unified is None:
        unified = os.environ.get("DSV41_PREFILL_MOE", "") == "unified"
    table = UNIFIED_TABLE if unified else BASE_TABLE
    if U > 16:  # B=128 does not fit at >= 32k: the smallest chunk
        return 128 * U
    if U in table:
        b = table[U]
    else:  # untabulated U: the nearest smaller tabulated U's chunk C, never above 2048 tokens per user
        c = table[max(u for u in table if u <= U)] // max(u for u in table if u <= U)
        b = U * min(c, 2048)
    if free_dram_mib is not None and free_dram_mib < MIN_FREE_MIB:
        b = max(128 * U, b // 2)
    return max(128 * U, b)
