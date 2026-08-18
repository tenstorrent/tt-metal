# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only inventory of which tuned MLA matmul configs are variant-agnostic.

`MLA_MATMUL_CONFIG` is keyed on `(weight_name, seq_len_local)` and nothing else, so a row that
declares no gating tag is applied to EVERY variant that reaches that seq_len_local -- tuned for it
or not. `ttMLA._cfg_fits_weight` only catches the subset where the block width fails to divide the
live weight's Kt, which raises a TT_FATAL; that is the *lucky* case. Where it happens to divide, the
other model's tiling is applied silently. The result is a perf question, not a numerics one -- the
matmul is still mathematically correct -- but "silently slower for a model nobody tuned" is exactly
the kind of thing that is never noticed.

This test does not forbid untagged rows. It pins the list, so adding one is a deliberate act that
shows up in review rather than an accident, and removing one (by tagging it, or by keying the table
on the real (K, N)) is visible progress.
"""

import pytest

from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.mla_config import MLA_MATMUL_CONFIG

# seq_len_local values whose tuned rows carry no dimensional/mode tag, as of 2026-08-18.
# 3200 = 25600/8 is the Mistral Small 4 production 25k ISL on an 8-way sequence split, i.e. these
# rows ARE being applied cross-variant today. Shrink this set; do not grow it.
KNOWN_UNTAGGED_SEQ_LENS = {3200, 4096}

# Keys that carry the tuning itself rather than gating it -- read unconditionally by
# `_resolve_act_mem_cfg` / the matmul kwargs in `mla.py`, never by `_cfg_matches`.
PAYLOAD_KEYS = frozenset({"program_config", "act_mem_config", "out_mem_config", "out_dtype"})


def _iter_configs():
    for weight_name, per_seq in MLA_MATMUL_CONFIG.items():
        for seq_len_local, entry in per_seq.items():
            candidates = entry if isinstance(entry, (list, tuple)) else (entry,)
            for cfg in candidates:
                yield weight_name, seq_len_local, cfg


def _is_tagged(cfg):
    return any(cfg.get(tag) is not None for tag in ttMLA._CFG_DIM_TAGS)


def test_mla_matmul_config_tags():
    untagged = {seq for _, seq, cfg in _iter_configs() if not _is_tagged(cfg)}

    unexpected = untagged - KNOWN_UNTAGGED_SEQ_LENS
    assert not unexpected, (
        f"new untagged tuned matmul rows at seq_len_local {sorted(unexpected)}: they will be applied "
        f"to every variant reaching that seq_len. Declare one of {ttMLA._CFG_DIM_TAGS} (or key the "
        f"row on the real (K, N)), or add it to KNOWN_UNTAGGED_SEQ_LENS with a reason."
    )

    fixed = KNOWN_UNTAGGED_SEQ_LENS - untagged
    assert not fixed, (
        f"seq_len_local {sorted(fixed)} no longer has untagged rows -- remove it from "
        "KNOWN_UNTAGGED_SEQ_LENS so the list keeps meaning what it says."
    )


@pytest.mark.parametrize("tag", ttMLA._CFG_DIM_TAGS)
def test_declared_tags_are_understood(tag):
    """Every tag a config declares must be one the resolver actually reads.

    A typo'd tag key is not an error today: `cfg.get("num_head")` returns None, the gate passes, and
    the config is applied to everything -- the exact failure the tag was added to prevent.
    """
    for weight_name, seq_len_local, cfg in _iter_configs():
        for key in cfg:
            if key in PAYLOAD_KEYS:
                continue
            assert key in ttMLA._CFG_DIM_TAGS, (
                f"{weight_name}@{seq_len_local} declares unknown key {key!r}; the resolver never "
                f"reads it, so it gates nothing. Known tags: {ttMLA._CFG_DIM_TAGS}"
            )
