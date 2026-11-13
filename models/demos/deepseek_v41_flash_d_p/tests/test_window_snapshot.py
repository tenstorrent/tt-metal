# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WINDOW_KV_KEEP=snapshot128 (P.42): the served 128-row window snapshot at cap (device; run_safe_pytest.sh, the
spec's mesh).

One window cache (tt/attention.TtWindowCache, keep snapshot128, its own snapshot slab: the contract's ND-sharded
[1, 1, 128, 512] bf16 ROW_MAJOR, 32-row records ROUND_ROBIN_1D over the DRAM banks) takes random window rows through
both ring write paths (``write``: the chunk's natural rows on every chip; ``write_own``: each chip its own query rows,
the ring blocks owned by one chip, so the snapshot must gather them), then ``write_snapshot(end)``. After every write,
on every chip: snapshot row p % 128 == the row of position p for p in [cap - 128, cap), cap = end // 64 x 64
(KV_BLOCK_SIZE 64; P.49, DS41-219: decode's cap of actual_end = prompt_len - 1), and every row of no position < cap zero, byte for byte. Ends: 100 (cap 64 < 128), a whole chunk,
a mid-chunk end, a follow-up off the chunk period (start 32 mod 64) whose window spans two chunks, an end mid-record
over owned blocks, the server's pulled-back last chunk (its carry from the pin: TtWindowCache.pin), 40 (cap 0: all
zero). 8x4: ~10 s.

test_skip_snapshot (P.43, OWNER 2026-11-07; runs only under DECODER_SKIP=exact / replay128 with snapshot128, as the
gate sets them): the whole model (every layer, lm_head off), rung s10240's tokens, a fresh state per prompt. The
request's last chunk starts where the caller rule puts it, TtDS41Model.last_start(end) (SKIP_TAIL_START), the chunk
before it (last=False: the decoder skipped) ending there:
    5184   a multiple of 64: cap 5120 before P.49, 5184 since (DS41-219: end // 64 x 64) (exact: [0, 2432) + [2432, 5184); replay128: [0, 4992) + [4992, 5184))
    5150   not a multiple of 64: cap 5120 (exact: [0, 2432) + [2432, 5150); replay128: [0, 4992) + [4992, 5150))
    100    shorter than either mode's last-chunk minimum: cap 64, one chunk [0, 100), the request's last
After the last chunk every layer's served snapshot (chip (0, 0)'s rows; every chip the same bytes) is checked:
written at cap (snap_cap == cap, no missing rows), rows of no position < cap zero, and its rows p % 128 vs the golden
window KV of the positions of [cap - 128, cap) the golden holds (every layer's ring at 5120 = [4992, 5120): all 128
rows at cap 5120; at cap 64 the step-dump layers' rows), rel L2 recorded as snap_err_window_kv_L<i>_<end> with
snap_..._limit = calib x max(the official reference's drift, one rounding) on the same comparison (the no-skip
limit, F128): on the same rows where the official per-position window KV exists (the step-dump layers 0, 1, 2, 3, 14,
20, 21, 24: contract/official_kv.py, FIX.19, DS41-193: the official's own error on rows [0, 64) is 2.5x its error on the
ring at 10240 for layer 20, 0.180 vs 0.072), else the rung's ring drift (state_window_kv_L<i>). Gated:
under exact every layer (exact by the rule); under replay128 layers 0-20 (layer 20's window KV is a row-wise
projection of layer 19's exact output) and layers 21-39 recorded as snap_diag_window_kv_L<i>_<end> beside the same
limit, not gated: their rows come from the decoder tail, whose windows are truncated at the tail's start by design
(the treatment test_window_ring gives replay128's tail rows, P.31 owner, DS41-207). Under SKIP_TAIL_START=end (the
P.42 path, the A/B's off side) only layers 0-19 are gated; the decoder layers are recorded as snap_diag_*. Refusal
(cap): a last chunk starting past last_start(end) after a skipped chunk raises ValueError.
"""

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

WIDTH = 512
BATCH_SHARED = True  # FIX.29: takes the model of a bring-up batch process (core/batch.shareable)


def _set(monkeypatch, **kw):
    for k, v in kw.items():
        monkeypatch.setenv(settings.env_name(k), str(v))
    settings.refresh()


def test_snapshot_cap():
    from models.demos.deepseek_v41_flash_d_p.tt.attention import snapshot_cap

    # P.49 (DS41-219): ``e`` is the engine's actual_end = prompt_len - 1 under PD_DISAGG, so decode's cap
    # (prompt_len - 1) // 64 x 64 = e // 64 x 64 (2944: a 2945-token prompt, decode's cap 2944)
    assert [snapshot_cap(e, 64) for e in (1, 64, 65, 100, 128, 129, 2944, 3000, 5120, 5121)] == [
        0,
        64,
        64,
        64,
        128,
        128,
        2944,
        2944,
        5120,
        5120,
    ]


@mesh_parametrize
def test_window_snapshot(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tt.attention import (
        SNAP,
        TtWindowCache,
        WindowTables,
        _upload,
        snapshot_cap,
    )

    _set(monkeypatch, WINDOW_KV_KEEP="snapshot128", KV_BLOCK_SIZE=64)
    mesh = mesh_device
    chunk = int(spec().get("target.chunk"))
    tables = WindowTables(mesh, chunk)
    g = tables.geo
    cache = TtWindowCache(mesh, tables, 4 * chunk)
    assert cache.snap is not None and cache.ring is not None
    gen = torch.Generator().manual_seed(42)
    full = (torch.randn(4 * chunk, WIDTH, generator=gen) * 2).to(torch.bfloat16)

    def nat_rows(start, end):
        nat = torch.zeros(chunk, WIDTH, dtype=torch.bfloat16)
        nat[: end - start] = full[start:end]
        return nat

    def write_rep(start, end):
        t = _upload(mesh, nat_rows(start, end).view(1, 1, chunk, WIDTH), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        cache.write(t, start, end)
        ttnn.deallocate(t)

    def write_own(start, end):
        nat = nat_rows(start, end).view(g.sp, g.tp, g.q_rows, WIDTH)
        t = _upload(mesh, nat, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (0, 1))
        cache.write_own(t, start, end)
        ttnn.deallocate(t)

    fails, checked = [], []

    def check(end, what):
        cache.write_snapshot(end)
        ttnn.synchronize_device(mesh)
        cap = snapshot_cap(end, 64)
        want = torch.zeros(SNAP, WIDTH, dtype=torch.bfloat16)
        for p in range(max(0, cap - SNAP), cap):
            want[p % SNAP] = full[p]
        chips = cache.snapshot_host()
        for q, got in enumerate(chips):
            if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
                bad = torch.nonzero((got != want).any(dim=-1)).flatten()
                fails.append(f"{what} end {end} cap {cap}: chip {q} rows {bad[:8].tolist()} differ")
                break
        checked.append((what, end, cap, len(chips)))

    for start, end, how, what in (
        (0, 100, write_rep, "short prompt, cap < 128"),
        (0, chunk, write_own, "whole chunk, owned blocks"),
        (chunk, chunk + 1000, write_own, "mid-chunk end"),
        (chunk + 992, 2 * chunk + 70, write_rep, "follow-up at 32 mod 64, window over two chunks"),
        (2 * chunk + 64, 3 * chunk - 23, write_own, "owned blocks, end mid-record"),
    ):
        how(start, end)
        check(end, what)
    # the server's pulled-back last chunk (prefill_writer.cpp:63-67): (P, max_seq) with P = max_seq - chunk after a chunk
    # that ended past P; its carry [P - 128, P) is out of the 192-row ring, kept in the pin
    pin_at = 3 * chunk
    assert cache.pin_at == pin_at, (cache.pin_at, pin_at)
    cache.truncate(2 * chunk + 640)  # (its own carry is not the point here: masked, as begin_chunk does for a new slot)
    write_own(2 * chunk + 640, 3 * chunk + 640)
    cache.truncate(None)
    assert not cache.ring.holds(pin_at - 128, pin_at) and cache.carry_ready(pin_at)
    carry = cache._ring_carry(pin_at)
    for q, got in enumerate(ttnn.get_device_tensors(carry)):
        got = ttnn.to_torch(got).reshape(-1, WIDTH)
        if not torch.equal(got.view(torch.int16), full[pin_at - 128 : pin_at].view(torch.int16)):
            fails.append(f"pulled-back carry [{pin_at - 128}, {pin_at}): chip {q} differs")
            break
    ttnn.deallocate(carry)
    write_own(pin_at, 4 * chunk)
    check(4 * chunk, "pulled-back last chunk (max_seq - chunk, max_seq)")
    write_rep(0, 40)
    check(40, "cap 0: no position below cap")
    cache.free()
    print(f"[window snapshot] checked (what, end, cap, chips): {checked}")
    assert not fails, "\n".join(fails)


RUNG = "s10240"
PAD_ID = 0xFFFFFFFF  # the engine's pad (K-72): past the chunk's end
SKIP_ENDS = (5184, 5150, 100)


def _chunk(model, state, tokens, s: int, e: int, last: bool) -> None:
    """One chunk [s, e) of the slot, every layer (its output freed)."""
    ids = torch.full((model.chunk,), PAD_ID, dtype=torch.long)
    ids[: e - s] = tokens[s:e].long()
    state.note_tokens(ids, s, e)
    model.begin_chunk(ids, s, last=last)
    h = model.embed(ids)
    for i in model.layers:
        h2 = model.layer(i, h, s, state, e)
        model.free(h)
        h = h2
    model.free(model.final_norm(h))
    model.free(h)
    model.sync()


@mesh_parametrize
def test_skip_snapshot(mesh_device, expect_error):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.common.bringup.core import metrics
    from models.demos.common.bringup.reference.golden import Golden
    from models.demos.common.bringup.testing import e2e_limits as E
    from models.demos.common.bringup.testing.ladder import rel
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import official_kv
    from models.demos.deepseek_v41_flash_d_p.tt.attention import SNAP, snapshot_cap
    from models.demos.deepseek_v41_flash_d_p.tt.runners.kv_contract import golden_window

    settings.refresh()
    mode = settings.get("DECODER_SKIP")
    if mode == "off" or settings.get("WINDOW_KV_KEEP") != "snapshot128":
        pytest.skip("P.43: runs under DECODER_SKIP=exact / replay128 with WINDOW_KV_KEEP=snapshot128 (the gate's env)")
    S = spec()
    g = Golden.for_rung(S, RUNG)
    layers = [i for i in S.layers() if i in g.layers]
    assert layers == list(range(S.num_layers)), f"the snapshot of every layer needs every layer: {layers}"
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    assert model.decoder and model.skip_mode == mode, (model.decoder, model.skip_mode, mode)
    if model.replay:
        model.build_tail()
    from_cap = model.tail_from_cap()
    metrics.record("model_load_s", round(getattr(model, "load_seconds", 0.0), 1))
    drift = E.load(g.dir, RUNG)
    official = official_kv.load(g)  # {step-dump layer: [seq, 512]} (DS41-193: the limit from the same rows)
    tokens, chunk = g.tokens(), g.chunk
    cap_seq = 2 * chunk  # (every case ends inside the first two chunks)
    # the layers gated on their golden rows: cap: exact every layer, replay128 layers <= 20 (the rest from the tail);
    # end (P.42): the encoder (0-19) only
    if not from_cap:
        n_gated = model.decoder[0]
    else:
        n_gated = model.decoder[0] + 1 if model.replay else len(layers)
    failed, diag = [], []
    for end in SKIP_ENDS:
        cap = snapshot_cap(end)
        start = model.last_start(end)
        assert end - start <= chunk and start <= chunk, (end, start)
        state = model.new_state(cap_seq)
        model.steady_state(None)
        if start > 0:
            _chunk(model, state, tokens, 0, start, False)
        _chunk(model, state, tokens, start, end, True)
        tail_at = model._tail_at
        metrics.record(f"snap_last_start_{end}", start)
        print(
            f"DECODER_SKIP={mode} SKIP_TAIL_START={settings.get('SKIP_TAIL_START')} end {end} cap {cap}: chunks "
            + (f"[0, {start}) last=False, " if start else "")
            + f"[{start}, {end}) last=True"
            + (f", tail [{tail_at[0]}, {tail_at[1]})" if tail_at else "")
        )
        worst, vals = (None, 0.0), []
        for i in layers:
            gated = i < n_gated
            c = state.window[i]
            chips = c.snapshot_host()
            got = chips[0]
            where = f"end {end} cap {cap} L{i:02d}"
            bad = []
            if c.snap_cap != cap:
                bad.append(f"{where}: snapshot written at cap {c.snap_cap}, not {cap}")
            if c.snap_missing is not None:
                bad.append(f"{where}: snapshot rows of positions {c.snap_missing} missing (zero)")
            if any(not torch.equal(x.view(torch.int16), got.view(torch.int16)) for x in chips[1:]):
                bad.append(f"{where}: the chips' snapshot bytes differ")
            held = {p % SNAP for p in range(max(0, cap - SNAP), cap)}
            empty = [r for r in range(SNAP) if r not in held and bool(got[r].abs().sum() > 0)]
            if empty:
                bad.append(f"{where}: rows {empty[:4]} of no position < cap are not zero")
            pos, want = golden_window(g, i, max(0, cap - SNAP), cap)
            if len(pos) and want is not None:
                key = f"state_window_kv_L{i:02d}"
                lim_drift, src = drift, "ring"
                if i in official:  # (the official's drift on the same rows, one rounding at least: E.err_limit)
                    lim_drift = {key: {**drift[key], "rel": rel(official[i][pos].float(), want)}}
                    src = "rows"
                v, lim = rel(got[pos % SNAP].float(), want), E.err_limit(lim_drift, key)
                name = f"snap_{'err' if gated else 'diag'}_window_kv_L{i:02d}_{end}"
                metrics.record(name, v)
                metrics.record(f"{name}_limit", lim)
                vals.append(f"L{i:02d} {v:.4f}/{lim:.4f}{'' if gated else ' (diag)'}")
                if not v <= lim:
                    (bad if gated else diag).append(
                        f"{name}: {v:.5g} > limit {lim:.5g} (calib {E.CALIB} x drift {key}, {src})"
                    )
                if gated and v / lim > worst[1]:
                    worst = (i, v / lim)
            elif cap >= SNAP:
                bad.append(f"{where}: the golden holds no row of [{cap - SNAP}, {cap})")
            if gated or from_cap:  # (the structural checks: every layer under cap; under end the gated layers)
                failed += bad
            else:
                diag += bad
        print(f"end {end}: snapshot rel / limit: {', '.join(vals)}")
        print(f"end {end}: worst gated snapshot L{worst[0]} at {worst[1]:.3f} of its limit")
        state.free()
    # refusal: a last chunk past last_start(end) after a skipped chunk (cap only; end has no check)
    if from_cap:
        end = 5150
        lim = model.last_start(end)
        bad_start = lim + 32
        state = model.new_state(cap_seq)
        _chunk(model, state, tokens, 0, bad_start, False)
        ids = torch.full((model.chunk,), PAD_ID, dtype=torch.long)
        ids[: end - bad_start] = tokens[bad_start:end].long()
        state.note_tokens(ids, bad_start, end)
        model.begin_chunk(ids, bad_start, last=True)
        with expect_error(ValueError, "starts after"):
            model._check_last_start(bad_start, end, state)
        print(f"refusal: last chunk [{bad_start}, {end}) refused (last_start {lim})")
        state.free()
    for d in diag:
        print(f"diagnostic (not gated): {d}")
    assert not failed, "\n".join(failed)
