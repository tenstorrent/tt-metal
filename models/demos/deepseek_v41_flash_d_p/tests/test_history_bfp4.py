# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.77 (HISTORY_FORMAT=bfp4, OWNER DECISION 2026-11-18): the compressed history sparse attention reads as 320 B
bfloat4_b row records, packed by every chip from its own served bf16 rows before the whole-mesh gather (device; run
with DS41_HISTORY_FORMAT=bfp4 and the defaults STATIC_TABLES=off, HISTORY_GATHER=rows, HISTORY_GATHER_HIDE=on).

One load, DECODER_SKIP=decode (as test_skip_positions). At each point p (P77_POS, default 145408, 1024000) and start
offset o (P77_OFF, default 0, 2048): a fresh state, the previous chunk at p + o - chunk, then the chunk at p + o
(testing/profile.PositionGolden ids). Sides, each its own state on the same load (the buffers' format is taken at the
state's build): ``hide`` (bfp4, P.75's prefetch + tail), ``full`` (bfp4, the whole history gathered at the source's
layer), ``bf16`` (HISTORY_FORMAT=bf16, hide: the 16-bit path before P.77). Checks:
- every gathered source's history buffer (bfp4 sides): owner q's rows [q R, q R + L) on EVERY chip byte-equal to the
  host packer's records (ckv_bfp4_ttnn/tests/reference.encode_rows: tt-metal's bfloat4_b rule, round to nearest even)
  of owner q's served bf16 rows [0, L) (L = nb hs: the chunk periods the chunk's queries reach);
- layer 19 (the last encoder layer: every source's attention feeds it) bit-identical hide vs full;
- the hide side took one prefetch + one tail a source (history_gather.STATS).
Prints (not gated): layer 19 bfp4 vs bf16 (max abs, rel L2), per source the gathered bytes into each chip and the
synced time of the pack, the records gather and the bf16 gather of the same history (median of 5; an isolated op:
device time + its dispatch), and the history buffers' DRAM a chip of each format."""

from __future__ import annotations

import os
import statistics
import time

import pytest
import torch
from ttnn.bringup.ckv_bfp4_ttnn.tests.reference import encode_rows

import ttnn
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.positions import _free
from models.demos.common.bringup.testing.profile import PositionGolden
from models.demos.deepseek_v41_flash_d_p.tt import ccl, history_gather
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True  # FIX.29: shares the batch's model (STATIC_TABLES=off at its load)
LAST_ENC = 19  # (DECODER_SKIP=decode: layers 0-19 run in full)


def _ints(name: str, default: str) -> list[int]:
    return [int(x) for x in os.environ.get(name, default).split(",") if x.strip()]


def _sync(mesh):
    ttnn.synchronize_device(mesh)


def _timed(mesh, fn, reps: int = 5) -> float:
    fn()
    _sync(mesh)
    ts = []
    for _ in range(reps):
        t0 = time.time()
        fn()
        _sync(mesh)
        ts.append((time.time() - t0) * 1e3)
    return statistics.median(ts)


def _check_records(c, s0: int, chunk: int) -> tuple[str | None, int]:
    """Owner q's records in the source's buffer (every chip) == the host packer of its served rows [0, L)."""
    ck = c.ckv
    buf = history_gather._buffer(ck.mesh, history_gather._key(ck))
    assert buf is not None and buf.bfp4, "no bfp4 history buffer"
    R = ck.slab.rows
    hs = ck.stripe // ck.ratio
    L = -(-(s0 + chunk) // chunk) * hs
    sl = ck.slab.slice_rows(0, -(-L // 32) * 32)
    served = [ttnn.to_torch(t).reshape(-1, 512)[:L] for t in ttnn.get_device_tensors(sl)]
    ttnn.deallocate(sl)
    if ck.tp_axis is None:  # (a TP-replicated cache: mesh column 0's copy of every SP row)
        served = served[:: ck.geo.tp]
    assert len(served) == ck.owners, (len(served), ck.owners)
    want = torch.cat([encode_rows(x.float()) for x in served])  # [owners L, 320] uint8
    # (the persistent buffer read as it is: a full-range slice would return its own buffer, K-1)
    chips = [
        ttnn.to_torch(t).reshape(ck.owners, R, -1)[:, :L].reshape(-1, history_gather.REC)
        for t in ttnn.get_device_tensors(buf.tensor)
    ]
    for k, got in enumerate(chips):
        got = got.to(torch.uint8)
        if not torch.equal(got, want):
            bad = (got != want).any(-1).nonzero()
            return f"chip {k}: {bad.numel()} of {want.shape[0]} records differ (first row {int(bad[0])})", L
    return None, L


@mesh_parametrize(timeout_s=0)
def test_history_bfp4(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    if (
        settings.get("STATIC_TABLES") != "off"
        or settings.get("HISTORY_GATHER") != "rows"
        or settings.get("HISTORY_GATHER_HIDE") != "on"
    ):
        pytest.skip("needs STATIC_TABLES=off HISTORY_GATHER=rows HISTORY_GATHER_HIDE=on (load-time)")
    if os.environ.get(settings.env_name("ENGRAM_TABLE")) is None:  # (as test_skip_positions: the shipped default)
        monkeypatch.setenv(settings.env_name("ENGRAM_TABLE"), str(settings.table["ENGRAM_TABLE"].default))
    chunk = int(S.get("target.chunk"))
    points, offs = _ints("P77_POS", "145408,1024000"), _ints("P77_OFF", "0,2048")
    layers = list(S.layers())
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cases = [(p, o) for p in points for o in offs if p + o + chunk <= deploy]
    want = max([p + o + chunk for p, o in cases] + [int(S.data["target"]["seq"])])
    monkeypatch.setitem(S.data["target"], "seq", want)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    fmt_env = settings.env_name("HISTORY_FORMAT")
    fmt0 = os.environ.get(fmt_env)
    bad = []

    def set_fmt(fmt):
        os.environ[fmt_env] = fmt
        settings.refresh()

    try:
        for p, o in cases:
            s0 = p + o
            ids = PositionGolden(S, p, chunk).tokens()
            prev_t, cur_t = ids[p - chunk : p], ids[p : p + chunk]
            res, stats = {}, {}
            for side in ("hide", "full", "bf16"):
                set_fmt("bf16" if side == "bf16" else "bfp4")
                model._hide = side != "full"
                state = model.new_state(s0 + chunk)
                try:

                    def run(tokens, start, keep=False):
                        model.begin_chunk(tokens, start, last=False)
                        h = model.embed(tokens)
                        out = None
                        for i in layers:
                            h2 = model.layer(i, h, start, state)
                            model.free(h)
                            h = h2
                            if keep and i == LAST_ENC:
                                out = model.to_host(h)
                        model.free(h)
                        return out

                    run(prev_t, s0 - chunk)
                    before = dict(history_gather.STATS)
                    res[side] = run(cur_t, s0, keep=True)
                    stats[side] = {k: history_gather.STATS[k] - before[k] for k in before}
                    srcs = sorted(model.history_sources())
                    if side != "bf16":
                        for i in srcs:
                            err, L = _check_records(state.compress[i], s0, chunk)
                            print(f"[P.77] p{p}+{o} {side}: source {i} records [0, {L}) an owner: {err or 'exact'}")
                            if err:
                                bad.append(f"p{p}+{o} {side} source {i}: {err}")
                    if side == "hide":  # timing and bytes, per source (the bfp4 buffers are held by this state)
                        for i in srcs:
                            ck = state.compress[i].ckv
                            owners, axis = history_gather.owners_and_axis(ck.mesh, ck.tp_axis is not None)
                            hs = ck.stripe // ck.ratio
                            L = -(-(s0 + chunk) // chunk) * hs
                            buf = history_gather._buffer(ck.mesh, history_gather._key(ck))
                            tmp = history_gather.HistoryBuffer(ck.mesh, owners * ck.slab.rows, None, False)
                            try:
                                t_pack = _timed(mesh_device, lambda: history_gather._pack(ck, 0, L))
                                loc = history_gather._pack(ck, 0, L)
                                s_0, s_1 = ccl._sems(ck.mesh, axis)

                                def gather_rec():
                                    ttnn.experimental.high_bw_all_gather(
                                        loc,
                                        2,
                                        buf.tensor,
                                        cluster_axis=axis,
                                        num_links=ccl.AG_LINKS,
                                        gathered_dim_size=owners * L,
                                        ready_semaphore=s_0,
                                        data_valid_semaphore=s_1,
                                    )

                                t_rec = _timed(mesh_device, gather_rec)
                                t_full = _timed(mesh_device, lambda: history_gather._full(ck, buf, L, owners, axis))
                                t_bf16 = _timed(mesh_device, lambda: history_gather._full(ck, tmp, L, owners, axis))
                            finally:
                                tmp.free()
                            mb = owners * L / 1e6
                            print(
                                f"[P.77] p{p}+{o} source {i}: L {L} rows an owner x {owners}; gathered into each chip "
                                f"bfp4 {mb * history_gather.REC:.1f} MB vs bf16 {mb * 1024:.1f} MB; synced ms: pack "
                                f"{t_pack:.3f}, records gather {t_rec:.3f}, pack + gather {t_full:.3f}, bf16 gather "
                                f"{t_bf16:.3f}; buffer a chip bfp4 {owners * ck.slab.rows * 320 / 2**20:.1f} MiB vs "
                                f"bf16 {owners * ck.slab.rows * 1024 / 2**20:.1f} MiB"
                            )
                finally:
                    _free(state)
            a, b, c = res["hide"], res["full"], res["bf16"]
            same = torch.equal(a, b)
            d = a.float() - c.float()
            rel = float(d.norm() / c.float().norm())
            print(
                f"[P.77] p{p}+{o}: layer {LAST_ENC} hide vs full bit-identical {same}; bfp4 vs bf16 max abs "
                f"{float(d.abs().max()):.4g} rel L2 {rel:.4g}; gathers {stats}"
            )
            if not same:
                bad.append(f"p{p}+{o}: hide != full (max abs {float((a - b).abs().max())})")
            n_src = len(model.history_sources())
            if s0 >= chunk and stats["hide"] != {"prefetch": n_src, "tail": n_src, "full": 0}:
                bad.append(f"p{p}+{o}: hide side gathers {stats['hide']}")
    finally:
        if fmt0 is None:
            os.environ.pop(fmt_env, None)
        else:
            os.environ[fmt_env] = fmt0
        settings.refresh()
        model._hide = True
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    assert not bad, bad
