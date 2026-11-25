# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The prefill stack in FRONT of the decode model, one process: a prompt is prefilled by the prefill stack into its
own caches, the stack's weights are freed, the decode model is built ON those caches (DecodeModel(stores=)) and
generates behind the prompt. Nothing of the context comes from the host: the rows and the index keys of positions
[0, N - 1) are the prefill stack's, every later one is appended by the decode step, and every full-indexer layer
selects on the chips over both.

    HY4_MOCK=1 pytest <this file>                                 (the decode side on stores of the stack's layout; the stack is not built)
    HY4_TEST_TIMEOUT=2400 TT_METAL_OPERATION_TIMEOUT_SECONDS=30 pytest <this file> -x

Env (tests/test_decode_model.py's): HY4_DM_LAYERS (6 | 78), HY4_DM_PROMPT (31 | 1000 | 1022 | 3000), HY4_DM_TRACE,
HY4_DM_SPLIT, HY4_ROPE_PAIRING=interleaved with its HY4_GOLDEN_DIR. Its own: HY4_PD_MAX_SEQ (8192: positions a
row holds, the prefill caches' max_seq_len), HY4_PD_APPEND (truncate), HY4_PD_LATENT_SCALE (1.0430; "" for none).

Order (weights swapped, PLAN.md section 25: the two weight sets do not fit side by side at 78 layers):
  1. the prefill stack through tt/prefill_bridge.py, its caches for 8 users
  2. the golden's prompt (N tokens) into every user's slots
  3. PrefillBridge.release_model(): the weights go, the caches stay
  4. DecodeModel(stores=(the stack's latent storage, its index keys), append="truncate", latent_scale=1.0430):
     the stack's pack cuts toward zero, so the rows this model appends are cut the same way and one scalar answers
     both
  5. adopt_row(r, N - 1) for the 8 rows, then
     a. the golden's decode steps teacher-forced (step j feeds the golden's token N - 1 + j at position N - 1 + j):
        logits and greedy tokens against the golden's; the first step program by program (every full layer's
        selection: exactly positions 0 .. N - 1 below the top-k, against the golden's recorded picks above it)
     b. every row rewound to N - 1 (reset_row(keep=)), then free-running greedy generation on the device's own
        tokens next to the golden's continuation
Lines start with HY4_PD.
"""

import os
import time

from loguru import logger

from models.experimental.hy4_preview.fused.kv_store import device as kv_device
from models.experimental.hy4_preview.fused.kv_store import frames as kv_frames
from models.experimental.hy4_preview.fused.model.decode_model import DecodeModel
from models.experimental.hy4_preview.fused.model.program import ROWS, is_mock
from models.experimental.hy4_preview.fused.model.tests import test_decode_model as TD
from models.experimental.hy4_preview.fused.projections import prepare as PP
from models.experimental.hy4_preview.tt import prefill_bridge as pb

MAX_SEQ = int(os.environ.get("HY4_PD_MAX_SEQ", "8192"))
APPEND = os.environ.get("HY4_PD_APPEND", "truncate")
SCALE = os.environ.get("HY4_PD_LATENT_SCALE", "1.0430")
SCALE = float(SCALE) if SCALE else None


def mock_stores(mesh, layers):
    """Two tensors of the prefill stack's layout (the mock builds no stack)."""
    full = sum(PP.is_full(n) for n in range(layers))
    frames = MAX_SEQ // kv_frames.FRAME_TOKENS
    lat = kv_frames.Kind(
        "latent", layers, kv_frames.LATENT.width, kv_frames.LATENT.shard_bytes, kv_frames.LATENT.row_stride
    )
    idx = kv_frames.Kind("index", full, kv_frames.INDEX.width, kv_frames.INDEX.shard_bytes, 0)
    return tuple(kv_device.allocate(mesh, kv_frames.Geometry.contiguous(kind, frames), ROWS) for kind in (lat, idx))


@TD.galaxy
def test_prefill_then_decode(mesh_device):
    mesh, n = mesh_device, TD.LAYERS
    g = TD.golden_of()
    N, steps, ids = g["meta"]["prompt_len"], g["meta"]["steps"], g["input_ids"]
    assert N + steps <= MAX_SEQ, (N, steps, MAX_SEQ)
    label = (
        f"prefill stack -> decode model [{n} layers, P{N}, {'ONE trace a step' if TD.TRACE else 'untraced'}, "
        f"append {APPEND}, latent_scale {SCALE}, experts {TD.SPLIT}]"
    )
    marks = []

    def mark(what, t0):
        marks.append((what, round(time.perf_counter() - t0, 1), round(pb.dram(mesh)["allocated_gb"], 3)))
        logger.info(f"HY4_PD {label}: {what}: {marks[-1][1]} s, DRAM {marks[-1][2]} GB per chip")

    bridge = None
    try:
        if is_mock():
            stores = mock_stores(mesh, n)
        else:
            t0 = time.perf_counter()
            bridge = pb.PrefillBridge(mesh, num_layers=n, num_users=ROWS, max_seq_len=MAX_SEQ)
            bridge.build()
            mark("1 prefill stack built", t0)
            t0 = time.perf_counter()
            for user in range(ROWS):
                bridge.prefill([int(t) for t in ids[:N]], user=user)
            mark(f"2 prefill of {N} tokens into {ROWS} slots", t0)
            t0 = time.perf_counter()
            bridge.release_model()
            mark("3 prefill weights released", t0)
            assert bridge.kv_caches.index is not None
            stores = (bridge.kv_caches.kvpe.storage, bridge.kv_caches.index)
        t0 = time.perf_counter()
        model = DecodeModel(
            mesh,
            layers=n,
            max_positions=MAX_SEQ,
            trace=TD.TRACE,
            keep_logits=True,
            check_every=TD.CHECK_EVERY,
            mlp_split=TD.SPLIT,
            append=APPEND,
            latent_scale=SCALE,
            stores=stores,
        )
        mark("4 decode model built on the stack's caches", t0)
        rows = list(range(ROWS))
        feed = lambda token_of, s: ([token_of(r) for r in rows], [s] * ROWS)
        for r in rows:
            model.adopt_row(r, N - 1)
        if is_mock():
            model.step(*feed(lambda r: int(ids[N - 1]), N - 1))
            model.step_probed(*feed(lambda r: int(ids[N]), N), after=lambda name: None)
            logger.info(
                f"HY4_PD {label}: compiled on the mock: {len(model.programs) + 2} programs a step, program-cache entries "
                f"{mesh.num_program_cache_entries()}, L1 tensors {model.st.l1_bytes()} B per core"
            )
            return
        # a. the golden's steps, teacher-forced
        table, t0, times = [], time.perf_counter(), []
        for j in range(steps):
            s = N - 1 + j
            tokens, positions = feed(lambda r: int(ids[s]), s)
            t1 = time.perf_counter()
            if j == 0 or (not TD.TRACE and s + 1 > TD.TOPK):
                probe = TD.Probe(model, f"{label} step {j}", positions, [True] * ROWS, TD.golden_picks(g, j))
                got = model.step_probed(tokens, positions, after=probe)
                if probe.shared:
                    TD.log_selection(g, j, probe, label)
            else:
                got = model.step(tokens, positions)
            times.append(time.perf_counter() - t1)
            TD.compare_step(model, g, j, got, rows, table, None)
            if j < 3:
                logger.info(
                    f"HY4_PD {label} golden step {j} (position {s}, {times[-1]:.2f} s): tokens out {got}, the golden's {int(g['cached_logits_argmax'][j])}"
                )
        status = model.status()
        wrong = TD.summarise(table, label)
        plain = times[3:]
        logger.info(
            f"HY4_PD {label}: {steps} golden steps behind the prefilled prompt in {time.perf_counter() - t0:.0f} s on the host (a step "
            f"{1e3 * sum(plain) / max(len(plain), 1):.0f} ms with its host writes and reads; not a latency figure); tokens that differ "
            f"from the golden's: {len(wrong)} of {len(table)}; status {status or 'at rest'}; canary {model.st.canary_summary()}"
        )
        assert not status, status
        TD.check_runs(model, steps, label)
        # b. free-running generation from the same point
        for r in rows:
            model.reset_row(r, keep=N - 1)
        last, out = {r: int(ids[N - 1]) for r in rows}, {r: [] for r in rows}
        for j in range(steps):
            got = model.step(*feed(lambda r: last[r], N - 1 + j))
            for r in rows:
                last[r] = got[r]
                out[r].append(got[r])
        status = model.status()
        want = [int(v) for v in g["cached_logits_argmax"][:steps]]
        gold_margin = [round(float(v[0] - v[1]), 3) for v in g["logits"][:steps].float().topk(2, dim=-1).values]
        firsts = []
        for r in rows:
            first = next((k for k in range(steps) if out[r][k] != want[k]), None)
            firsts.append(first)
        logger.info(
            f"HY4_PD {label} FREE RUN: per row, generated tokens equal to the golden's continuation "
            f"{[sum(a == b for a, b in zip(out[r], want)) for r in rows]} of {steps}; the first step that differs per row {firsts}"
            + (
                ""
                if firsts[0] is None
                else f" (row 0: golden {want[firsts[0]]}, device {out[0][firsts[0]]}; the golden's own top-1 margin there {gold_margin[firsts[0]]})"
            )
        )
        logger.info(f"HY4_PD {label} FREE RUN row 0 generated {out[0]}")
        logger.info(f"HY4_PD {label} FREE RUN the golden's continuation {want}")
        logger.info(f"HY4_PD {label} phases (what, s, DRAM GB per chip): {marks}; status {status or 'at rest'}")
        assert not status, status
        assert min(t["pcc"] for t in table) > 0.7, "logits far from the golden's"
    finally:
        if bridge is not None:
            bridge.close()
