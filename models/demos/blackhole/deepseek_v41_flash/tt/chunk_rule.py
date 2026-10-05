"""Prompt-length independent prefill chunk rule (DSV41_PREFILL_ROW_TOKENS=auto2, default OFF). Filled in from the calibration, see PREFILL_CHUNKCAL_NOTES.md."""


def auto2_budget(U, free_dram_mib=None):
    """Row-token budget (tokens per mesh row per traced-chunk pass) for U users per mesh row; chunk per user = budget // U."""
    return 2048 if U > 4 else 4096  # placeholder = the current auto rule at long context
