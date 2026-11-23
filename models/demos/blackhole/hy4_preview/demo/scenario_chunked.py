# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scenario ``chunked_pf`` (E7 of HY4_PORT_PLAN, G6 of the demo gates): chunked prefill interleaved with decode, driven in process (no vLLM, plugin or scheduler).

Engine API under test (tt/hy4_serve_engine.py, HY4_CHUNKED_PF, default off): ``prefill_begin(prompts_by_row) -> handle``, ``prefill_chunks(handle, n)`` (n x PF-row chunks of the eager pool loop), ``prefill_end(handle) ->
first tokens``, ``prefill_abort(handle)``. Reference = the monolithic ``prefill`` with the hybrid chain switched OFF (HY4_PF_HYBRID_FILE=0: both take the direct-to-pool path, so the logits md5 must be equal).

Cases (``cases=`` selects):
  c0  reference: the 3 decoders alone (greedy streams S_ref, ms/round) and the monolithic long prompt (lg_md5, first token, 48 greedy tokens, sampled pool-page md5s, wall)
  c1  grid chunks-per-group n x decode-rounds-per-group m: chunked lg_md5 == mono, first token + 48 tokens, pool md5s, the decoders' streams bit-identical to S_ref (nothing of theirs disturbed), decode tok/s during the
      prefill, rounds/s, prefill slowdown vs mono, state intact afterwards (decoders 64 more rounds, row 3 decodes)
  c2  two prompts of different lengths interleaved in two handles (round robin), both == mono
  c3  abort mid-prefill: row clean (not live, no chain, nothing owned), decoders unaffected, the row prefills correctly afterwards
  c4  chunked suffix prefill after a prefix hit (start > 0) == monolithic hit; last_cached equal
Gate lines: ``DEMO_GATE chunked_pf <case>:<name> PASS|FAIL|SKIP <value>``; perf rows are appended to ctx.perf (kind=chunked). Lengths default 16384 (cold 55 s at 3.4 ms/token [M]).
"""
import os
import tempfile
import time

from models.demos.blackhole.hy4_preview.demo import scenarios as S

R = S.R


def _hybrid_off(ctx):
    """live switch of the hybrid chain (engine reads HY4_PF_HYBRID_FILE per prefill call): 0 = pure pool prefill, the path the chunked API always takes."""
    if getattr(ctx, "hybrid_file", None) is None:
        ctx.hybrid_file = tempfile.NamedTemporaryFile("w", suffix=".hybrid", delete=False).name
    with open(ctx.hybrid_file, "w") as f:
        f.write("0\n")
    os.environ["HY4_PF_HYBRID_FILE"] = ctx.hybrid_file


def _round(eng, streams, rows, count_rows=None):
    """one greedy spec round; every row of ``rows`` that is live records its tokens; returns the tokens of ``count_rows`` (default: rows)."""
    c = [eng.K if eng.live[r] else 0 for r in range(R)]
    out = eng.round(c, False)
    n = 0
    for r in rows:
        if eng.live[r] and r in streams:
            em = out[r]["a"][: out[r]["m"] + 1]
            streams[r] += em
            if count_rows is None or r in count_rows:
                n += len(em)
    return n


def _drive(ctx, handles, n, m, streams, dec_rows, max_groups=100000):
    """round robin over the handles: one chunk group (n chunks) of each, then m decode rounds, until every handle is done. -> dict(wall, rounds, dec_tokens, chunk_wall, round_wall, groups)"""
    eng = ctx.eng
    t0 = time.perf_counter()
    rounds = toks = groups = 0
    cw = rw = 0.0
    while not all(eng.prefill_done(h) for h in handles) and groups < max_groups:
        for h in handles:
            if eng.prefill_done(h):
                continue
            t = time.perf_counter()
            out = eng.prefill_chunks(h, n)
            for r, tok in (
                out.get("first") or {}
            ).items():  # a row that finished (its first token) decodes from the next round on: its stream starts here
                streams[r] = [tok]
            cw += time.perf_counter() - t
            groups += 1
            if all(eng.prefill_done(x) for x in handles):
                break
            t = time.perf_counter()
            for _ in range(m):
                toks += _round(eng, streams, list(streams), dec_rows)
                rounds += 1
                if hasattr(eng, "note_decode_rounds"):
                    eng.note_decode_rounds(h, 1)
            rw += time.perf_counter() - t
    return dict(
        wall=time.perf_counter() - t0, rounds=rounds, dec_tokens=toks, chunk_wall=cw, round_wall=rw, groups=groups
    )


def _streams(ctx, firsts, dec=None, nrounds=96, want=48):
    """decode rounds until every row of ``firsts`` {row: first token} has ``want`` tokens; the OTHER live rows advance in the same rounds, so their streams (``dec``, updated in place) must be recorded too. -> {row: tokens[:want]}"""
    st = {r: [t] for r, t in firsts.items()}
    if dec is not None:
        st.update(dec)
    rows = list(st)
    for _ in range(nrounds):
        if all(len(st[r]) >= want for r in firsts):
            break
        _round(ctx.eng, st, rows)
    return {r: st[r][:want] for r in firsts}


def _cont(ctx, st, rows, nrounds=96, want=48):
    """continue decoding until every row of ``rows`` has ``want`` tokens in ``st`` (every stream of ``st`` is recorded; st updated in place). -> {row: tokens[:want]}"""
    for _ in range(nrounds):
        if all(len(st[r]) >= want for r in rows):
            break
        _round(ctx.eng, st, list(st))
    return {r: st[r][:want] for r in rows}


def _stream(ctx, row, first_tok, dec=None, nrounds=96, want=48):
    return _streams(ctx, {row: first_tok}, dec, nrounds, want)[row]


def _pool(eng, row):
    try:
        return eng.pool_md5(row) if hasattr(eng, "pool_md5") else None
    except Exception:  # noqa: BLE001  (layout of the device read not validated: the gate then reports SKIP, never PASS)
        return None


def chunked_pf(
    ctx,
    lens=(16384,),
    chunks_per_group=(1, 2, 4),
    rounds_per_group=(2, 8),
    rounds_ref=96,
    cases=("c0", "c1", "c2", "c3", "c4"),
    seed=6,
    dec_len=2048,
):
    s = "chunked_pf"
    eng = ctx.eng
    if not all(hasattr(eng, a) for a in ("prefill_begin", "prefill_chunks", "prefill_end")):
        return ctx.gate(
            s, "api_present", None, "engine has no prefill_begin/prefill_chunks/prefill_end (HY4_CHUNKED_PF)"
        )
    if not getattr(eng, "cpf_on", lambda: True)():
        return ctx.gate(s, "flag_on", None, "HY4_CHUNKED_PF=1 not set (default off)")
    _hybrid_off(ctx)
    S.set_pfx(ctx, False)  # cold arms; c4 switches the cache on
    short = {r: S.prompt(60 + r, dec_len) for r in range(3)}
    for L in lens:
        L3 = S.prompt(seed, L)
        # ---- c0 reference
        S.release_all(eng)
        f, _, _, _ = S.prefill(ctx, short)
        ref, ms_ref, acc_ref = S.decode(ctx, f, rounds_ref, rows=[0, 1, 2])
        base_tps = sum(acc_ref) / (sum(ms_ref) / 1e3) if ms_ref else 0.0
        S.release_all(eng)
        fm, mm, tmono, _ = S.prefill(ctx, {3: L3})
        pool_m = _pool(eng, 3)
        sm = _stream(ctx, 3, fm[3])
        eng.release(3)
        ctx.gate(
            s,
            f"L{L}:c0_reference",
            True,
            f"mono {tmono:.1f}s lg_md5 {mm.get(3)} decoders alone {base_tps:.1f} tok/s (3 rows)",
        )
        if "c1" in cases:
            for n in chunks_per_group:
                for m in rounds_per_group:
                    tag = f"L{L}_n{n}_m{m}_c1"
                    S.release_all(eng)
                    f, _, _, _ = S.prefill(ctx, short)
                    dec = {r: [f[r]] for r in range(3)}
                    h = eng.prefill_begin({3: L3})
                    d = _drive(ctx, [h], n, m, dec, [0, 1, 2])
                    first = eng.prefill_end(h)
                    md5c = getattr(eng, "prefill_md5", lambda _h: {})(h).get(3)
                    ctx.gate(
                        s, f"{tag}:lg_md5_equals_mono", md5c is not None and md5c == mm.get(3), f"{md5c} vs {mm.get(3)}"
                    )
                    ctx.gate(s, f"{tag}:first_token_equal", first.get(3) == fm.get(3), f"{first.get(3)} vs {fm.get(3)}")
                    pc = _pool(eng, 3)
                    ctx.gate(
                        s, f"{tag}:pool_md5_equal", None if pc is None or pool_m is None else pc == pool_m, str(pc)
                    )
                    sc = _cont(ctx, dec, [3])[3]
                    ctx.gate(s, f"{tag}:48_greedy_tokens_equal", sc == sm, S.md5_ints(sc))
                    for r in range(3):
                        ctx.gate(
                            s,
                            f"{tag}:decoder{r}_bit_identical",
                            len(dec[r]) > 4 and S.common_prefix_equal(ref[r], dec[r]),
                            f"{len(dec[r])} tokens",
                        )
                    # integrity afterwards: decoders keep going, still equal to the uninterrupted stream
                    for _ in range(64):
                        _round(eng, dec, [0, 1, 2])
                    for r in range(3):
                        ctx.gate(
                            s,
                            f"{tag}:decoder{r}_intact_after",
                            S.common_prefix_equal(ref[r], dec[r]),
                            f"{len(dec[r])} tokens",
                        )
                    tps = d["dec_tokens"] / d["wall"] if d["wall"] else 0.0
                    row = dict(
                        kind="chunked",
                        scen=s,
                        L=L,
                        chunks_per_group=n,
                        rounds_per_group=m,
                        wall_s=d["wall"],
                        mono_s=tmono,
                        slowdown_pct=100 * (d["wall"] / tmono - 1),
                        decode_rounds_per_s=d["rounds"] / d["wall"] if d["wall"] else 0.0,
                        decoder_tok_per_s_during=tps,
                        decoder_tok_per_s_alone=base_tps,
                        chunk_wall_s=d["chunk_wall"],
                        round_wall_s=d["round_wall"],
                        groups=d["groups"],
                    )
                    ctx.perf.append(row)
                    ctx.gate(
                        s,
                        f"{tag}:perf",
                        True,
                        f"wall {d['wall']:.1f}s vs mono {tmono:.1f}s ({row['slowdown_pct']:+.0f}%), decoders {tps:.1f} tok/s during vs {base_tps:.1f} alone, {d['rounds']} rounds",
                    )
        if "c2" in cases:
            tag = f"L{L}_c2"
            La, Lb = S.prompt(seed + 1, L), S.prompt(seed + 2, max(L // 2 // 512 * 512, 1024))
            S.release_all(eng)
            fa, ma, _, _ = S.prefill(ctx, {2: La})
            sa = _stream(ctx, 2, fa[2])
            eng.release(2)
            fb, mb, _, _ = S.prefill(ctx, {3: Lb})
            sb = _stream(ctx, 3, fb[3])
            eng.release(3)
            short2 = {r: short[r] for r in range(2)}
            f, _, _, _ = S.prefill(ctx, short2)
            ref2, _, _ = S.decode(ctx, f, rounds_ref, rows=[0, 1])
            S.release_all(eng)
            f, _, _, _ = S.prefill(ctx, short2)
            dec = {r: [f[r]] for r in range(2)}
            ha, hb = eng.prefill_begin({2: La}), eng.prefill_begin({3: Lb})
            _drive(ctx, [ha, hb], 1, 2, dec, [0, 1])
            fa2, fb2 = eng.prefill_end(ha), eng.prefill_end(hb)
            pm = eng.prefill_md5
            ctx.gate(s, f"{tag}:A_lg_md5_equals_mono", pm(ha).get(2) == ma.get(2), f"{pm(ha).get(2)} vs {ma.get(2)}")
            ctx.gate(s, f"{tag}:B_lg_md5_equals_mono", pm(hb).get(3) == mb.get(3), f"{pm(hb).get(3)} vs {mb.get(3)}")
            both = _cont(ctx, dec, [2, 3])
            ctx.gate(s, f"{tag}:A_stream_equal", both[2] == sa, "48 tokens")
            ctx.gate(s, f"{tag}:B_stream_equal", both[3] == sb, "48 tokens")
            for r in range(2):
                ctx.gate(
                    s,
                    f"{tag}:decoder{r}_bit_identical",
                    len(dec[r]) > 4 and S.common_prefix_equal(ref2[r], dec[r]),
                    f"{len(dec[r])} tokens",
                )
        if "c3" in cases:
            tag = f"L{L}_c3"
            S.release_all(eng)
            f, _, _, _ = S.prefill(ctx, short)
            dec = {r: [f[r]] for r in range(3)}
            h = eng.prefill_begin({3: L3})
            _drive(ctx, [h], 2, 2, dec, [0, 1, 2], max_groups=2)
            eng.prefill_abort(h)
            for _ in range(8):
                _round(eng, dec, [0, 1, 2])
            ctx.gate(
                s,
                f"{tag}:row_clean",
                (not eng.live[3]) and eng.ret[3] is None and not getattr(eng, "pf_owner", {}),
                f"live={eng.live[3]} ret={eng.ret[3] is not None}",
            )
            for r in range(3):
                ctx.gate(
                    s,
                    f"{tag}:decoder{r}_bit_identical",
                    len(dec[r]) > 4 and S.common_prefix_equal(ref[r], dec[r]),
                    f"{len(dec[r])} tokens",
                )
            h = eng.prefill_begin({3: L3})
            _drive(ctx, [h], 4, 1, dec, [0, 1, 2])
            first = eng.prefill_end(h)
            ctx.gate(
                s,
                f"{tag}:row_reusable_md5_equals_mono",
                eng.prefill_md5(h).get(3) == mm.get(3),
                f"{eng.prefill_md5(h).get(3)} vs {mm.get(3)}",
            )
            ctx.gate(s, f"{tag}:row_reusable_stream_equal", _cont(ctx, dec, [3])[3] == sm, "48 tokens")
        if "c4" in cases:
            tag = f"L{L}_c4"
            suffix = S.prompt(seed + 9, 1024)
            P2 = L3 + suffix
            S.set_pfx(ctx, True)
            S.release_all(eng)
            S.prefill(ctx, {3: L3})
            eng.release(3)
            f2, m2, _, lc2 = S.prefill(ctx, {3: P2})
            s2 = _stream(ctx, 3, f2[3])
            eng.release(3)
            S.prefill(ctx, {3: L3})  # restore the cold chain of L3 on row 3
            eng.release(3)
            S.release_all(eng)
            f, _, _, _ = S.prefill(ctx, short)
            dec = {r: [f[r]] for r in range(3)}
            h = eng.prefill_begin({3: P2})
            lcc = dict(getattr(eng, "last_cached", {}) or {})
            _drive(ctx, [h], 1, 2, dec, [0, 1, 2])
            first = eng.prefill_end(h)
            ctx.gate(
                s,
                f"{tag}:real_hit",
                lcc.get(3, 0) > 0 and lcc.get(3) == lc2.get(3),
                f"chunked {lcc.get(3)} mono {lc2.get(3)}",
            )
            ctx.gate(
                s,
                f"{tag}:lg_md5_equals_mono_hit",
                eng.prefill_md5(h).get(3) == m2.get(3),
                f"{eng.prefill_md5(h).get(3)} vs {m2.get(3)}",
            )
            ctx.gate(s, f"{tag}:stream_equal", _cont(ctx, dec, [3])[3] == s2, "48 tokens")
            for r in range(3):
                ctx.gate(
                    s,
                    f"{tag}:decoder{r}_bit_identical",
                    len(dec[r]) > 4 and S.common_prefix_equal(ref[r], dec[r]),
                    f"{len(dec[r])} tokens",
                )
            S.set_pfx(ctx, False)
    S.set_pfx(ctx, True)
    ctx.gate(s, "perf_recorded", True, f"{len(ctx.perf)} rows")
