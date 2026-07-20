# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared matmul benchmark mode metadata for GEMM_FLOPS plots and reports."""

# Canonical benchmark mode order (OOB first, then explicit program configs).
MODE_ORDER = [
    "oob",
    "reuse_dram",
    "mcast_2d_l1",
    "mcast_2d_dram",
    "mcast_1d_in0",
    "mcast_1d_out",
    "dram_sharded",
]

# Short legend labels keyed to the underlying program factory / config.
MODE_DISPLAY = {
    "oob": "OOB (auto)",
    "reuse_dram": "Reuse (DRAM)",
    "mcast_2d_l1": "2D MCAST (L1)",
    "mcast_2d_dram": "2D MCAST (DRAM)",
    "mcast_1d_in0": "1D MCAST in0",
    "mcast_1d_out": "1D MCAST out",
    "dram_sharded": "DRAM sharded",
}

MODE_COLORS = {
    "oob": "black",
    "reuse_dram": "#9467bd",
    "mcast_2d_l1": "#d62728",
    "mcast_2d_dram": "#1f77b4",
    "mcast_1d_in0": "#ff7f0e",
    "mcast_1d_out": "#2ca02c",
    "dram_sharded": "#8c564b",
}

MODE_LINESTYLES = {
    "oob": "-",
    "reuse_dram": "-",
    "mcast_2d_l1": "--",
    "mcast_2d_dram": "-",
    "mcast_1d_in0": "-",
    "mcast_1d_out": "--",
    "dram_sharded": "-.",
}

# Map legacy CSV mode names from older benchmark runs.
LEGACY_MODE_ALIASES = {
    "tuned_2d_l1": "mcast_2d_l1",
    "tuned_2d_dram": "mcast_2d_dram",
}

TUNED_MODES = [mode for mode in MODE_ORDER if mode != "oob"]


def normalize_mode(mode):
    """Return the canonical mode name, mapping legacy aliases when needed."""
    return LEGACY_MODE_ALIASES.get(mode, mode)


def normalize_modes(df):
    """Normalize the mode column in a benchmark DataFrame in place."""
    if "mode" in df.columns:
        df["mode"] = df["mode"].apply(normalize_mode)
    return df


LEGACY_BASE_SHAPE_COLUMNS = ["base_m", "base_k", "base_n"]


def parse_grid_size(raw):
    cleaned = str(raw).strip("() ")
    grid_x, grid_y = [int(x.strip()) for x in cleaned.split(",")]
    return grid_x, grid_y


def add_shape_column(df):
    """Add a ``shape`` tuple column for grouping benchmark rows by full M/K/N."""
    if all(col in df.columns for col in ("m", "k", "n")):
        df["shape"] = list(zip(df["m"], df["k"], df["n"]))
    elif all(col in df.columns for col in LEGACY_BASE_SHAPE_COLUMNS) and "grid_size" in df.columns:
        # Very old CSVs stored per-core tile counts in base_*; scale to element dims.
        scaled = []
        for _, row in df.iterrows():
            grid_x, grid_y = parse_grid_size(row["grid_size"])
            scaled.append((row["base_m"] * grid_y, row["base_k"] * grid_x, row["base_n"] * grid_x))
        df["shape"] = scaled
    elif all(col in df.columns for col in LEGACY_BASE_SHAPE_COLUMNS):
        df["shape"] = list(zip(df["base_m"], df["base_k"], df["base_n"]))
    return df
