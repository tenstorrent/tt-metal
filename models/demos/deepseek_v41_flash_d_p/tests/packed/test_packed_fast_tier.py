# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.T FAST tier (OWNER 2026-11-21; <= 5 min, in every accept gate next to the cold strict no-rebuild check): SLOT
ISOLATION AT CHIP BOUNDARIES on the serving entry (tests/packed/isolation_common.py: the contract, the two worlds,
what is compared).

Four packed chunks (8192 rows, 32 chips of 256 rows; every segment its whole chips), real text, DECODER_SKIP=decode,
each segment against the same request alone (a one-segment chunk in a slot of its own, the same boundaries):

    chunk 0  A[0,4096) 16 chips | B[0,1024) | C[0,256) 1 chip | H[0,2048) | D[0,512)            31 chips, 5 segments
    chunk 1  D[512,1536) split part 1 | C[256,456) 1 chip, partial | A[4096,8192) steady |
             B[896,3072) replay (write_start 1024) | E[0,1) 1 token (cap 0: snapshot + acks)    31 chips
    chunk 2  G[0,2048) in C's slot (reused) | A[8192,10240) (history 8192: the long candidate pool) |
             D[1536,2560) split part 2                                                         20 chips (12 empty)
    chunk 3  H[2048,4096) (parked through chunks 1-2) | B[3072,3112) acks only (cap 3072) |
             G[2048,2304) 1 chip exactly | A[10112,12288) replay (write_start 10240) | K[0,1024) in E's slot  23 chips
A replay is a resumed request as tt-d-gen sends it: its history [0, P) written, the chunk starting P - 128 (the window
replayed), write_start P (rows below P are never written by the chunk; a replay of rows nothing wrote would read the
slot's own leftovers, which legitimately differ between a reused and a fresh slot).

Slots: B, D, H take slots 0-2, which the engine's warm-up (runtime.compile) filled with junk; G reuses C's slot and K
E's (their records over [0, end) must not see the leftovers). Segment orders differ every chunk. Bystanders (whole
slabs, every layer / config / chip unchanged): H's slot while it is parked (after chunk 0 .. before chunk 3), E's slot
idle (after chunk 1 .. before chunk 3), and a canary slot nothing uses (before chunk 0 .. after chunk 3).

Metrics (gated): packed_fast_output_mismatches, packed_fast_served_mismatches, packed_fast_ack_mismatches,
packed_fast_bystander_mismatches (== 0), packed_fast_output_max_abs_diff (== 0), packed_fast_one_ring_layers (> 0: the
indexer layers took GluedCall's one ring call), packed_fast_hide_regions (> 0: the history gathers hid),
packed_fast_program_builds_after_warmup (== 0: every serving call after runtime.compile), packed_fast_tier_s
(<= 330: the test's wall, warm-up included). Recorded: packed_fast_long_pool_layers, packed_fast_segments,
packed_fast_fallback_chunks, packed_fast_compile_s.

    BRINGUP_CHUNK=8192 DS41_PACKED=on ... (the gate's switches) scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_packed_fast_tier.py
"""

from __future__ import annotations

import importlib
import time

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tests.packed import isolation_common as _ic

_ic = importlib.reload(_ic)  # (the resident runner keeps helper modules of an earlier request: take this file's)

S = spec()
pytestmark = device_timeout(S)
BATCH_SHARED = True  # (the gate's accuracy load: every layer, with the determinism check, OV-61)
CHUNK = 8192
# the slots' capacity: the engine's warm-up (runtime.compile -> tt/packed_device.warm_glued_chunks) needs >= ~112k (its
# continued 6-segment kinds and its "far" kinds; below it the far kinds raise on a negative start, DS41-372); the load's
# tables reach it through BATCH_MAX_SEQ (core/batch.hints; the accuracy load was 64512)
MAX_SEQ = 114688
BATCH_MAX_SEQ = 114688
CANARY = 9
REQS = {  # name: (text offset, packed slot)
    "A": (0, 6),
    "B": (13000, 0),
    "C": (17000, 7),
    "D": (18000, 1),
    "E": (21000, 8),
    "H": (25000, 2),
    "G": (22000, 7),  # C's slot, after C's last chunk
    "K": (30000, 8),  # E's slot, after E's last chunk
}
CHUNKS = [
    [("A", 0, 4096, None), ("B", 0, 1024, None), ("C", 0, 256, None), ("H", 0, 2048, None), ("D", 0, 512, None)],
    [
        ("D", 512, 1536, None),
        ("C", 256, 456, None),
        ("A", 4096, 8192, None),
        ("B", 896, 3072, 1024),
        ("E", 0, 1, None),
    ],
    [("G", 0, 2048, None), ("A", 8192, 10240, None), ("D", 1536, 2560, None)],
    [
        ("H", 2048, 4096, None),
        ("B", 3072, 3112, None),
        ("G", 2048, 2304, None),
        ("A", 10112, 12288, 10240),
        ("K", 0, 1024, None),
    ],
]
BYSTANDERS = [(2, 0, 3), (8, 1, 3), (CANARY, -1, 4)]  # (slot, after chunk a, before chunk b)


def run_tier(
    mesh_device,
    prefix: str,
    reqs: dict,
    chunks: list,
    bystanders: list,
    max_seq: int,
    slots: int,
    extra=None,
    alone_from=6,
):
    """One isolation tier: the bench, the alone world, the packed world, the metrics under ``prefix``. ``extra``:
    a callable (bench, text, verdict) run after the packed world (the deep sweep's more worlds); ``alone_from``: the
    first slot the alone world may take (slots 0-5 hold the warm-up's junk)."""
    from models.demos.deepseek_v41_flash_d_p.tt import gather_hide
    from models.demos.deepseek_v41_flash_d_p.tt import packed_device as PD
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    t_start = time.time()
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192 (the gate's packed load)"
    for k in ("PACKED", "PACKED_GLUED", "PACKED_CHIPS", "PACKED_GLUED_DYNAMIC", "POS_DYNAMIC"):
        assert settings.get(k) == "on", f"{k}={settings.get(k)}: the shipped packed path needs it on"
    text = Golden.for_rung(S, "s56320").tokens().reshape(-1).long()
    bench = _ic.Bench(mesh_device, S, max_seq, slots)
    v = _ic.Verdict()
    try:
        R = {n: _ic.Req(n, *v) for n, v in reqs.items()}  # (name: (text offset, packed slot[, reference]))
        w = _ic.Worlds(bench, text, R, chunks, bystanders, label=prefix)
        used = {r.slot for r in R.values()} | {s for s, _, _ in bystanders}
        free = [s for s in range(alone_from, slots) if s not in used]
        need = w.alone_names()
        assert len(free) >= len(need), f"{slots} slots: {len(free)} free for {len(need)} alone requests"
        for n, s in zip(need, free):
            R[n].alone_slot = s
        bench.hold(range(slots))
        for _, s, e, _ in (x for c in chunks for x in c):
            assert s + max(1, e - s) <= max_seq
        ref = w.alone()
        _ic.live_check(ref, (need[0], 0), (need[1], 0))
        fb0, hide0 = int(PD.STATS["fallback"]), int(gather_hide.STATS["regions"])
        with _ic.PathCount() as paths:
            w.packed(ref, v)
            if extra is not None:
                extra(bench, text, v)
        fallback = int(PD.STATS["fallback"]) - fb0
        hide = int(gather_hide.STATS["regions"]) - hide0
    finally:
        bench.close()
    wall = time.time() - t_start
    metrics.record(f"{prefix}_output_mismatches", int(v.out_mm))
    metrics.record(f"{prefix}_output_max_abs_diff", float(v.out_diff))
    metrics.record(f"{prefix}_served_mismatches", int(v.served_mm))
    metrics.record(f"{prefix}_ack_mismatches", int(v.ack_mm))
    metrics.record(f"{prefix}_bystander_mismatches", int(v.bystander_mm))
    metrics.record(f"{prefix}_fallback_mismatches", int(fallback))  # (a chunk that did not run the one call)
    metrics.record(f"{prefix}_one_ring_layers", len(paths.one_ring))
    metrics.record(f"{prefix}_long_pool_layers", len(paths.long_pool))
    metrics.record(f"{prefix}_grouped_ring_layers", len(paths.grouped))
    metrics.record(f"{prefix}_hide_regions", int(hide))
    metrics.record(f"{prefix}_program_builds_after_warmup", int(bench.builds))
    metrics.record(f"{prefix}_segments", int(v.segments))
    metrics.record(f"{prefix}_serving_calls", int(bench.serving_calls))
    metrics.record(f"{prefix}_compile_s", round(bench.compile_s, 1))
    print(
        f"ISOLATION {prefix}: {v.segments} segments vs alone: outputs {v.out_mm} (max |diff| {v.out_diff}), served "
        f"{v.served_mm}, acks {v.ack_mm}, bystanders {v.bystander_mm}; one-ring layers {sorted(paths.one_ring)}, "
        f"long pool {sorted(paths.long_pool)}, grouped {sorted(paths.grouped)}, hide regions {hide}, fallback "
        f"{fallback}; builds after warm-up {bench.builds} in {bench.serving_calls} serving calls; wall {wall:.1f} s "
        f"(compile {bench.compile_s:.1f} s)",
        flush=True,
    )
    for n in v.notes[:40]:
        print("  " + n)
    return v, paths, hide, fallback, bench, wall


@mesh_parametrize(timeout_s=1800)
def test_packed_fast_tier(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    t0 = time.time()
    v, paths, hide, fallback, bench, _ = run_tier(
        mesh_device, "packed_fast", REQS, CHUNKS, BYSTANDERS, MAX_SEQ, slots=CANARY + 1 + len(REQS)
    )
    wall = time.time() - t0
    metrics.record("packed_fast_tier_s", round(wall, 1))
    assert v.out_mm == 0 and v.served_mm == 0 and v.ack_mm == 0 and v.bystander_mm == 0, "slot isolation broken"
    assert v.out_diff == 0, v.out_diff
    assert fallback == 0, f"{fallback} packed chunks did not run the one call"
    assert paths.one_ring, "no indexer layer took the one ring call"
    assert hide > 0, "no history gather hid (PACKED_GATHER_HIDE)"
    assert bench.builds == 0, f"{bench.builds} programs built after runtime.compile"
    assert wall <= 330, f"the fast tier took {wall:.0f} s (> 330)"
