# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The device check's prompts and the H2 store sequence's EXPECTED per-chunk patterns, for any chunk size (torch + numpy only:
``driver.py`` on the device host and the host tests both import it).

The patterns are derived by a dry run of the runner's own decisions -- ``PrefixRetention`` with the H2 ``ChunkStore`` -- driven
the way ``tt/v41/runtime.py`` / ``tt/v41/prefix_reuse.py`` drive them (begin_chunk; a computed chunk retained and, if wanted,
stored; spills flushed after a burst's last chunk and after each digest's EMPTY flush chunk; the S2 phase starts with the
retention and the store emptied), with the production store's slot counts at that chunk size. So the pattern verdict checks
that the device runner made the decisions the host model makes; it does not re-derive them independently. The hand-checked
tables for 8192 and 4096 are pinned in ``tests/v41_prefix_reuse/test_v41_prefix_store.py``.

Codes: S skipped (held on the device, H1), R restored from the store (H2), C computed, C+Ns computed after N silently rebuilt
chunks; a prompt's pattern is the codes joined ("RRC"), comma-separated when one is longer than a letter ("R,C+1s,C").
"""

from __future__ import annotations

import numpy as np
import torch

from models.demos.common.prefill.prefix_reuse import PAGE, ChunkStore, PrefixRetention

S2 = ["A", "U", "A2", "U", "A", "U", "AP", "U", "B", "A2", "A2"]  # after X with arm clear
TIP_BYTES = 21 * 128 * 512 * 2  # the 21 window carries, bf16


def _rng(seed: int, n: int) -> list:
    return np.random.default_rng(seed).integers(1000, 120000, size=n).tolist()


def prompts(chunk: int) -> dict:
    """The driver's prompts (fixed seeds; every length a multiple of 128)."""
    C = int(chunk)
    A = _rng(53, 3 * C + 4992)
    return {
        "X": _rng(7, 2 * C),
        "A": A,
        "B": A[: C + 7040] + _rng(17, 20096),
        "A2": A + _rng(11, 12032),
        "AP": A[: 2 * C],
        "U": _rng(29, 2 * C + 1280),
    }


def record_bytes(chunk: int) -> int:
    """One stored chunk: bf16 entries [C/r, 512] + index keys [C/r, 128] of sources 2 / 8 / 14 (r 2) and layer 20 (r 1)."""
    C = int(chunk)
    return 3 * (C // 2) * (512 + 128) * 2 + C * (512 + 128) * 2


def store_slots(chunk: int, store_gb: float, tip_gb: float | None = None, users: int = 1) -> tuple[int, int]:
    """(chunk slots, tip slots) of one slot's store (ChunkStore's split and page-rounded slots)."""
    rnd = lambda n: -(-int(n) // PAGE) * PAGE  # noqa: E731
    per = int(store_gb * 2**30) // users
    tip = per // 16 if tip_gb is None else int(tip_gb * 2**30) // users
    return (per - tip) // rnd(record_bytes(chunk)), tip // rnd(TIP_BYTES)


def expected_patterns(
    chunk: int,
    *,
    n_ckpts: int = 16,
    spill_tips: int = 2,
    store_gb: float = 8.0,
    tip_gb: float | None = None,
    seq: list | None = None,
) -> list:
    """The S2 sequence's per-prompt patterns at ``chunk`` (one per prompt of ``seq``, X excluded)."""
    C = int(chunk)
    P = prompts(C)
    n_chunks, n_tips = store_slots(C, store_gb, tip_gb)
    dummy = [torch.zeros(1, dtype=torch.int64)]  # one page per slot: only the slot COUNTS matter here
    silent = [0]
    pr = PrefixRetention(
        chunk=C,
        n_ckpts=n_ckpts,
        save=lambda k: None,
        restore=lambda k: None,
        export_carries=lambda k, end, final: bool(final),
        set_next_start=lambda pos: None,
        compute_silent=lambda ids, start: silent.__setitem__(0, silent[0] + 1),
        on_resume=lambda pos: None,
        log=lambda m: None,
        skip_partial_last=False,
        store=ChunkStore((n_chunks + n_tips) * PAGE, tip_budget_bytes=n_tips * PAGE, log=lambda m: None),
        restore_rows=lambda parts, ids, start: None,
        spill=lambda k: dummy,
        load_carries=lambda parts: None,
        spill_tips=spill_tips,
    )

    def run(name: str) -> str:
        toks = P[name]
        S = len(toks)
        parts = [(s, min(S, s + C)) for s in range(0, S, C)]
        out = []
        for k, (s, e) in enumerate(parts):
            ids = torch.tensor(toks[s:e], dtype=torch.int32)
            last = k == len(parts) - 1
            silent[0] = 0
            if pr.begin_chunk(ids, s, e, last):
                out.append("R" if pr.last_restored else "S")
            else:
                pr.end_compute(ids, s, e)
                if pr.wants_rows(s, e):
                    pr.store_rows(ids, s, e, dummy)
                out.append(f"C+{silent[0]}s" if silent[0] else "C")
            if last and pr.pending_spills:
                pr.flush_spills()
        pr.note_empty()  # the digest's EMPTY flush chunk (after_acks(True))
        if pr.pending_spills:
            pr.flush_spills()
        return "".join(out) if all(len(x) == 1 for x in out) else ",".join(out)

    run("X")  # arm clear: an empty retention and store, then X
    return [run(name) for name in (seq or S2)]
