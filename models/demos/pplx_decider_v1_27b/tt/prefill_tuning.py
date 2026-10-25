# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-sequence-length prefill tuning for the pplx decider: the SDPA chunk sizes, the AGMM
(all_gather_minimal_matmul) configs and the reduce-scatter hyperparameters that beat the qwen36 defaults, applied
through the Qwen36ModelArgs hooks `prefill_sdpa_config_fn`, `prefill_agmm_config_fn` and `prefill_rs_hparams`
(qwen36 tp_common.prefill_sdpa_chunks / agmm_config_override / prefill_rs_kwargs; unset = the qwen36 defaults).

    PPLX_SDPA_TUNED=0       no SDPA override (qwen36 default q=k=128 chunks below, exp_approx off)
    PPLX_AGMM_TUNED=0       no AGMM override
    PPLX_AGMM_WIDE=0        drop the AGMM entries with grid_x > 8 (fallback if a wide grid fails in the model)
    PPLX_RS_TUNED=0         no reduce-scatter override (tt_all_reduce defaults: chunks_per_sync 10, 2 workers, 2 buffers)
    PPLX_SDPA_EXP_APPROX    1 (default) = the SDPA exp approximation on, 0 = off (the qwen36 default)
    PPLX_SDPA_SP_WINDOWED   1 (default) = the sequence-parallel SDPA runs the windowed plain SDPA (no paged KV fill)
                            where the table has a windowed entry, 0 = always the paged chunked SDPA

The tables come from standalone op sweeps on a BH Galaxy (1x8 and 1x4 submeshes, 2026-10): AGMM block / grid /
link configs per (S, shape), SDPA (q_chunk, k_chunk, exp_approx) per S (TP8: NH=3, NKV=1, HD=256), and reduce-scatter
workers / chunks_per_sync / buffers. AGMM override keys: {M_block, K_block, N_block, subblock_h, subblock_w, grid_x,
grid_y, num_links, num_workers_per_link}. Check a new winner in the full model before adding it (a standalone sweep
cannot see the model's other L1 tenants). Entries not in a table fall back to the qwen36 default for that call.
"""

import os

# --- SDPA ---------------------------------------------------------------------------------------------------------
# tp -> ((max S, (q_chunk, k_chunk)), ...) in ascending S, the last entry (None) covering every larger S. From the TP8
# SDPA sweep (NH=3 NKV=1 HD=256); TP4 reuses it (NH per device differs, the chunking is the same). At S=768 / 1536 the
# rule's (32,64) / (64,128) are within 2% of the best (64,64), below the 3% bar for an explicit case.
_SDPA_TABLE = ((1024, (32, 64)), (1536, (64, 128)), (None, (128, 128)))
SDPA_CHUNKS = {8: _SDPA_TABLE, 4: _SDPA_TABLE}


def sdpa_tuned_enabled():
    return os.environ.get("PPLX_SDPA_TUNED", "1") != "0"


def sdpa_exp_approx():
    return os.environ.get("PPLX_SDPA_EXP_APPROX", "1") != "0"


def make_sdpa_config_fn(tp):
    """fn(S) -> (q_chunk, k_chunk, exp_approx) for args.prefill_sdpa_config_fn; None = disabled / no table."""
    table = SDPA_CHUNKS.get(tp)
    if table is None or not sdpa_tuned_enabled():
        return None
    exp_approx = sdpa_exp_approx()

    def fn(S):
        for max_s, (q_chunk, k_chunk) in table:
            if max_s is None or S <= max_s:
                return q_chunk, k_chunk, exp_approx

    return fn


# Sequence-parallel SDPA (last row of R rows: queries S_total/R at chunk_start (R-1)*S_total/R, keys S_total).
# (tp, rows, S_total) -> {"paged": cfg, "windowed": cfg (optional)}, cfg = dict(variant, q, k, grid) with grid = (x, y)
# SDPA core grid or None (the full device grid). Entries come from the 2026-10-13 SP sweeps (grid + windowed:
# runs/s3_sdpa_sp_grid_short.jsonl, runs/s3_sdpa_sp_grid.jsonl, TP8, R=4): last-row us in the comments; "default" /
# "generic" = the generic table's paged config at the full grid. A "windowed" entry runs the plain windowed-causal SDPA
# on the gathered non-paged K/V (no paged fill, so no KV cache is written); the "paged" entry is the chunked paged
# SDPA, what PPLX_SDPA_SP_WINDOWED=0 selects. Only lengths where the generic table is >3% off.
def _sp_cfg(variant, q, k, grid=None):
    return {"variant": variant, "q": q, "k": k, "grid": grid}


SDPA_SP_CONFIGS = {
    (8, 4, 256): {
        "windowed": _sp_cfg("windowed", 32, 64, (3, 2)),  # 22.7 us vs generic 72.4
        "paged": _sp_cfg("paged", 64, 64, (4, 4)),  # 29.6 us
    },
    (8, 4, 512): {
        "windowed": _sp_cfg("windowed", 32, 64, (4, 3)),  # 38.5 us vs generic 126.5
        "paged": _sp_cfg("paged", 32, 64, (4, 3)),  # 56.8 us
    },
    (8, 4, 768): {
        "windowed": _sp_cfg("windowed", 32, 64, (6, 4)),  # 56.1 us vs full-grid paged q64 k64 105
        "paged": _sp_cfg("paged", 64, 64, (4, 3)),  # 75.9 us
    },
    (8, 4, 1024): {
        "windowed": _sp_cfg("windowed", 32, 64, (6, 4)),  # 88 us
        "paged": _sp_cfg("paged", 32, 64, (6, 4)),  # 107 us vs default 230
    },
    (8, 4, 1536): {
        "windowed": _sp_cfg("windowed", 64, 128, (6, 4)),  # 115.7 us vs full-grid paged q128 k128 201
        "paged": _sp_cfg("paged", 32, 128, (6, 5)),  # 177.3 us
    },
    (8, 4, 2048): {
        "windowed": _sp_cfg("windowed", 128, 128, (6, 4)),  # 255.2 us (also skips the paged fill)
        "paged": _sp_cfg("paged", 64, 128, (6, 5)),  # 242.5 us
    },
    (8, 4, 3072): {
        "windowed": _sp_cfg("windowed", 64, 128, (8, 8)),  # 329 us
        "paged": _sp_cfg("paged", 64, 128, (6, 4)),  # 356 us vs default 667
    },
    (8, 4, 4096): {
        "windowed": _sp_cfg("windowed", 128, 128, (6, 4)),  # 500 us
        "paged": _sp_cfg("paged", 64, 128, (8, 8)),  # 570 us vs default 873
    },
}


def sdpa_sp_windowed_enabled():
    return os.environ.get("PPLX_SDPA_SP_WINDOWED", "1") != "0"


def make_sdpa_sp_config_fn(tp):
    """fn(S_total, rows) -> dict(variant, q, k, exp[, grid]) | None for args.prefill_sdpa_sp_config_fn (the qwen36
    attention accepts this dict form or the old (q, k, exp) tuple; None = the generic per-S path)."""
    if not sdpa_tuned_enabled():
        return None
    exp_approx = sdpa_exp_approx()
    windowed = sdpa_sp_windowed_enabled()

    def fn(S_total, rows):
        entry = SDPA_SP_CONFIGS.get((tp, rows, S_total))
        if entry is None:
            return None
        cfg = dict(entry["windowed"] if windowed and "windowed" in entry else entry["paged"], exp=exp_approx)
        if cfg["grid"] is None:
            del cfg["grid"]
        return cfg

    return fn


# --- AGMM ---------------------------------------------------------------------------------------------------------
# (tp, S, shape_name) -> override dict, keys as qwen36 tp_common._agmm_launch_config (any subset; a missing key keeps
# the call site's default). shape_name: mlp_gateup | gdn_in | attn_qkv | gdn_out. S is the exact length the op runs at.
def _cfg(M, K, N, sub, grid=None, links=None, workers=None):
    cfg = {"M_block": M, "K_block": K, "N_block": N, "subblock_h": sub[0], "subblock_w": sub[1]}
    if grid is not None:
        cfg["grid_x"], cfg["grid_y"] = grid
    if links is not None:
        cfg["num_links"] = links
    if workers is not None:
        cfg["num_workers_per_link"] = workers
    return cfg


AGMM_CONFIGS = {
    # TP8. No entry = the call site's default config is the sweep's best (or within 3% of it): mlp_gateup S=1536 (M8)
    # and gdn_in/attn_qkv/gdn_out S=768.
    # Short per-row lengths of the sequence-parallel buckets (S = bucket / 4 rows), from the 2026-10-13 small-S sweep
    # (runs/s3_agmm_small_s.jsonl): us best vs default.
    (8, 64, "mlp_gateup"): _cfg(1, 4, 16, (1, 4), grid=(4, 9), links=2, workers=2),  # 158 vs 243
    (8, 128, "mlp_gateup"): _cfg(1, 4, 16, (1, 4), grid=(4, 9), links=2, workers=2),  # 158 vs 246
    (8, 192, "mlp_gateup"): _cfg(1, 4, 16, (1, 4), grid=(6, 9), links=1, workers=6),  # 160 vs 246
    (8, 384, "mlp_gateup"): _cfg(2, 4, 16, (1, 4)),  # 176 vs 251
    (8, 64, "gdn_in"): _cfg(1, 4, 8, (1, 1), grid=(2, 9), links=1, workers=2),  # 102 vs 109
    (8, 384, "gdn_in"): _cfg(4, 4, 8, (4, 1), grid=(4, 9), links=2, workers=2),  # 120 vs 124
    (8, 64, "attn_qkv"): _cfg(1, 4, 8, (1, 4), grid=(4, 8), links=2, workers=2),  # 101 vs 109
    (8, 128, "attn_qkv"): _cfg(1, 4, 8, (1, 4), grid=(4, 8), links=2, workers=2),  # 101 vs 110
    (8, 384, "attn_qkv"): _cfg(4, 4, 8, (1, 4), grid=(4, 9), links=2, workers=2),  # 119 vs 127 (partial sweep)
    (8, 64, "gdn_out"): _cfg(1, 8, 4, (1, 2), grid=(2, 7), links=1, workers=2),  # 55 vs 62
    (8, 128, "gdn_out"): _cfg(1, 8, 4, (1, 4), grid=(4, 9), links=2, workers=2),  # 58 vs 69
    (8, 256, "mlp_gateup"): _cfg(1, 4, 16, (1, 4)),
    (8, 512, "mlp_gateup"): _cfg(2, 4, 16, (1, 4)),
    (8, 768, "mlp_gateup"): _cfg(4, 4, 16, (1, 4)),  # 207 us vs default M8 255 (M2: 339)
    (8, 768, "gdn_out"): _cfg(4, 8, 4, (1, 2), grid=(8, 5)),  # 112 vs 116 us (mid-S sweep), links 2 x workers 4
    (8, 1024, "mlp_gateup"): _cfg(4, 4, 16, (1, 4)),
    (8, 1024, "gdn_out"): _cfg(4, 8, 4, (2, 1), grid=(8, 5)),  # 134 vs 140 us (mid-S sweep), links 2 x workers 4
    (8, 1536, "gdn_in"): _cfg(8, 4, 8, (1, 4)),  # 248 us vs default M4 297
    (8, 1536, "attn_qkv"): _cfg(8, 4, 8, (1, 4), grid=(8, 8)),  # 257 us vs default M4 297
    (8, 1536, "gdn_out"): _cfg(4, 8, 4, (2, 2), grid=(12, 5), links=2, workers=6),  # 186 us vs default 216
    (8, 2048, "gdn_in"): _cfg(8, 4, 8, (1, 4)),
    (8, 2048, "attn_qkv"): _cfg(8, 4, 8, (1, 4)),
    (8, 3072, "mlp_gateup"): _cfg(8, 4, 16, (1, 4), grid=(12, 9), links=2, workers=6),
    (8, 3072, "gdn_in"): _cfg(8, 4, 8, (1, 4), grid=(12, 9), links=2, workers=6),
    (8, 3072, "attn_qkv"): _cfg(8, 4, 8, (4, 1), grid=(12, 8), links=2, workers=6),
    (8, 3072, "gdn_out"): _cfg(8, 8, 4, (1, 2), grid=(12, 5), links=2, workers=6),
    (8, 4096, "gdn_in"): _cfg(8, 4, 8, (4, 1)),
    (8, 4096, "attn_qkv"): _cfg(8, 4, 8, (1, 4), grid=(8, 8)),
    # TP4
    (4, 256, "mlp_gateup"): _cfg(1, 8, 16, (1, 4)),
    (4, 512, "mlp_gateup"): _cfg(2, 8, 16, (1, 4)),
    (4, 768, "mlp_gateup"): _cfg(4, 8, 16, (1, 4)),  # M_block as the TP8 sweep (not swept at TP4)
    (4, 1024, "mlp_gateup"): _cfg(4, 8, 16, (1, 4)),
    (4, 2048, "mlp_gateup"): _cfg(8, 8, 16, (1, 4), grid=(12, 9), links=2, workers=6),
    (4, 2048, "gdn_in"): _cfg(8, 8, 8, (4, 1)),
    (4, 2048, "attn_qkv"): _cfg(4, 8, 4, (4, 1), grid=(11, 8), links=2, workers=6),
    (4, 2048, "gdn_out"): _cfg(8, 16, 4, (4, 1), grid=(11, 8), links=2, workers=6),
}


def agmm_tuned_enabled():
    return os.environ.get("PPLX_AGMM_TUNED", "1") != "0"


def agmm_wide_enabled():
    return os.environ.get("PPLX_AGMM_WIDE", "1") != "0"


def make_agmm_config_fn(tp, device_grid=None):
    """fn(S, shape_name) -> override dict | None for args.prefill_agmm_config_fn; None = disabled / no entry.

    device_grid (x, y): entries whose grid exceeds it are dropped, so those shapes keep the default config. The
    tables were tuned on a Galaxy chip (12x10 worker grid); a P150 exposes 11x10 and AGMM rejects a larger grid.
    """
    if not agmm_tuned_enabled():
        return None
    wide = agmm_wide_enabled()

    def usable(cfg):
        grid_x, grid_y = cfg.get("grid_x", 8), cfg.get("grid_y", 0)
        if not wide and grid_x > 8:
            return False
        return device_grid is None or (grid_x <= device_grid[0] and grid_y <= device_grid[1])

    table = {key: cfg for key, cfg in AGMM_CONFIGS.items() if key[0] == tp and usable(cfg)}
    if not table:
        return None

    def fn(S, shape_name):
        cfg = table.get((tp, S, shape_name))
        return dict(cfg) if cfg is not None else None

    return fn


# --- Reduce-scatter -----------------------------------------------------------------------------------------------
# (chunks_per_sync, num_workers_per_link, num_buffers_per_channel) of the prefill reduce-scatters (tt_all_reduce, 1xN
# branch) at every S and TP: the rs_hparam sweeps' best vs the tt_all_reduce default (10, 2, 2), ~1.2-1.3x per RS.
PREFILL_RS_HPARAMS = (4, 8, 2)


def rs_tuned_enabled():
    return os.environ.get("PPLX_RS_TUNED", "1") != "0"


def make_rs_hparams():
    """Value for args.prefill_rs_hparams; None = disabled."""
    return PREFILL_RS_HPARAMS if rs_tuned_enabled() else None
