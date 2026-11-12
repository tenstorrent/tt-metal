# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DECODER_SKIP=decode (P.51, owner 2026-11-12, decode's proposal): prefill never runs layers 21-39; an encoder replay
below the chunk's global_write_start writes no history-cache row.

The serving runtime (tt/runners/adapter.py, built and compiled as the engine does: caches.build, every layer, 3 slots,
chunk / max seq of the golden rung; the model loaded with DECODER_SKIP=decode, ``use_decoder_skip`` runs it as off for
the other side in the same process) gets each request the way tt-d-gen will hand it under decode's proposal: prompt
length N, prefill stop P = (N - 128) // KV_BLOCK_SIZE x KV_BLOCK_SIZE, chunks [k chunk, min((k + 1) chunk, P)) with
actual_end <= P, the H2D payload with the PAD_ID tail (server_rules), and the header's new field global_write_start.
Everything is read through the exported table over UMD, as the KV Manager reads it.

(1) No prefix hit (global_write_start = actual_start). Every request runs twice, DECODER_SKIP=off into slot 0 and
    decode into slot 1, both prompts (novel: the spec's prompt; code: tt-metal C++, testing/b2b) interleaved, so a
    record the decode run failed to write still holds the other prompt's bytes:
      served_bytes_differ (== 0 gates): records that differ between the two runs: every history config's records of
      [0, P) (compressed KV and index keys of every KV source, layer 20's included; re-index aliases included) and the
      0_window_kv snapshot of layers 0-19 (rows [P - 128, P) at p % 128);
      decoder_snapshot_written (== 0, asserted): snapshot records of layers 20-39 the decode run changed (they never
      run, so it never writes them);
      acks: one per served layer per chunk, in order, naming the engine's (slot, actual_start, actual_end), both sides.
(2) Prefix hit (global_write_start = Xp > actual_start). Slot 2: request A (decode) of N_A with P_A = Xp; then request
    B (decode) of N_B on the same prefix, one chunk [Xp - 128, P_B) with global_write_start Xp (DeepSeek's Encoder SWA
    Bounded Replay of 128 rows):
      replay_rows_written (== 0, asserted): history records below Xp that differ before and after B (never written);
      the rows from Xp on vs a fresh prefill of N_B (the decode side of (1), slot 1): relative error and PCC per config
      and layer, history configs and layers 0-19's snapshot; and both against the golden (the contract's limits:
      server_rules.state_threshold for the history configs, the official drift for a dumped layer's window rows,
      caches.golden_failures / snapshot_failures), recorded, not gated: approximate by design where an encoder window
      reaches before the replay start.
Prompt lengths N (P; chunks of 5120 on galaxy_8x4): 300 (128), chunk - 800 (4160), chunk + 128 (5120, one whole chunk),
chunk + 2200 (7168, two chunks), 2 chunk + 128 (10240, two whole chunks); scaled to the rung's chunk on other boxes
(P past max seq dropped). Prefix hit: Xp = chunk - 1024 (4096), N_B = chunk + 2200.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_decoder_skip_decode.py
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
from loguru import logger

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import b2b
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.prefill.runners.layer_completion_ring_writer import LayerCompletionSink
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C
from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
PROMPTS = ("novel", "code")
SLOT = {"off": 0, "decode": 1, "hit": 2}
WINDOW = 128


def decode_stop(n: int, block: int = R.KV_BLOCK) -> int:
    """Decode's proposal: prefill stops at P = (N - 128) // kv_block x kv_block (example N 5002 -> P 4864)."""
    return max(0, (int(n) - WINDOW) // block * block)


def prompt_lens(chunk: int, max_seq: int) -> list[int]:
    lens = [300, chunk - 800, chunk + 128, chunk + 2200, 2 * chunk + 128]
    return [n for n in lens if 0 < decode_stop(n) <= max_seq]


def chunks(start: int, end: int, chunk: int) -> list[tuple[int, int]]:
    out, lo = [], start
    while lo < end:
        out.append((lo, min(lo + chunk, end)))
        lo += chunk
    return out


def _decode_rows(raw: dict, cfg: dict) -> np.ndarray:
    return np.concatenate([C.decode(raw[k], cfg) for k in sorted(raw)]) if raw else np.zeros((0, cfg["width"]))


def _pcc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    if not a.size:
        return 1.0
    a, b = a - a.mean(), b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else (1.0 if np.array_equal(a, b) else 0.0)


@mesh_parametrize(timeout_s=2 * 3600)
def test_decoder_skip_decode(mesh_device, tmp_path, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn

    monkeypatch.setenv(settings.env_name("DECODER_SKIP"), "decode")  # (read at load; off via use_decoder_skip)
    settings.refresh()
    layers = C.served_prefix()
    cfgs = C.served_configs(layers)
    chunk, max_seq = R.CHUNK, R.MAX_SEQ
    lens = prompt_lens(chunk, max_seq)
    xp, n_b = chunk - 1024, chunk + 2200
    tokens = {wl: torch.tensor(b2b.ids(S, wl, max(lens + [n_b]))).numpy() for wl in PROMPTS}
    try:
        _, runtime, kv = C.build(mesh_device, layers, 3)
    except C.NotBuilt as e:
        pytest.fail(f"not built: {e}", pytrace=False)
    model = runtime.model
    assert model.skip_mode == "decode" and model.decoder, (model.skip_mode, model.decoder)
    enc = [i for i in layers if i not in model.decoder]
    dec = [i for i in layers if i in model.decoder]
    path = C.export_table(runtime, kv, mesh_device, str(tmp_path / "table.pb"), len(layers))
    reader = C.TableReader(path, C.device_map(mesh_device, str(tmp_path / "device_map.json")))
    hist = [c for c in cfgs if not c.get("snapshot")]
    snaps = [c for c in cfgs if c.get("snapshot")]
    assert snaps, "P.51 serves the window snapshot (WINDOW_KV_KEEP=snapshot128)"

    acks = []

    class Sink(LayerCompletionSink):
        def layers_completed(self, layer_start, layer_end, slot_id, pos_start, pos_end):
            for layer in range(layer_start, layer_end):
                acks.append((int(layer), (int(slot_id), int(pos_start), int(pos_end))))

    runtime.set_layer_completion_sink(Sink())
    walls = {}

    def run(wl: str, mode: str, slot: int, lo: int, hi: int, gws: int | None = None) -> list[str]:
        """One chunk [lo, hi) of prompt wl in ``slot`` under ``mode``; its ack failures."""
        model.use_decoder_skip(mode)
        inp = C.payload_tensor(mesh_device, R.server_payload(tokens[wl][lo : lo + chunk], lo, hi))
        n0 = len(acks)
        t0 = time.perf_counter()
        runtime.prefill_chunk(
            inp, kv, slot_id=slot, actual_start=lo, actual_end=hi, request_id=0, global_write_start=gws
        )
        ttnn.synchronize_device(mesh_device)
        walls.setdefault((mode, hi - lo), []).append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(inp)
        want = [(i, (slot, lo, hi)) for i in range(len(layers))]
        if acks[n0:] != want:
            return [
                f"{wl} {mode} slot {slot} chunk [{lo}, {hi}): acks {acks[n0:][:4]}.. want one per layer {want[:2]}.."
            ]
        return []

    def request(wl: str, mode: str, slot: int, n: int) -> list[str]:
        out = []
        for lo, hi in chunks(0, decode_stop(n), chunk):
            out += run(wl, mode, slot, lo, hi)
        return out

    def hist_recs(slot: int, lo: int, hi: int) -> dict:
        got = {}
        if hi > lo:
            for c in hist:
                got.update(
                    {(c["name"],) + k: b for k, b in reader.records(c["name"], slot, c["layers"], lo, hi).items()}
                )
        return got

    def snap_recs(slot: int, which: list[int]) -> dict:
        got = {}
        for c in snaps:
            got.update({(c["name"],) + k: b for k, b in reader.records(c["name"], slot, which, 0, 0).items()})
        return got

    _, kvc, _ = R.harness()
    gk = C.GoldenKV(window="drift")
    thr = R.state_threshold()
    gold_toks = gk.tokens().reshape(-1).numpy()

    def by_cfg(recs: dict, name: str) -> dict:
        return {k[1:]: b for k, b in recs.items() if k[0] == name}

    def prefix_hit(wl: str) -> dict:
        """(2): request A [0, Xp) then B's replay chunk [Xp - 128, P_B) with global_write_start Xp, in slot 2."""
        sl, out, rows = SLOT["hit"], [], []
        n_a, p_b, lo = xp + WINDOW, decode_stop(n_b), xp - WINDOW
        assert decode_stop(n_a) == xp and p_b - lo <= chunk, (n_a, xp, p_b, lo)
        out += request(wl, "decode", sl, n_a)
        before = hist_recs(sl, 0, xp)
        out += run(wl, "decode", sl, lo, p_b, gws=xp)
        after = hist_recs(sl, 0, xp)
        written = sorted(k for k in before if after.get(k) != before[k])
        if written:
            out.append(
                f"prefix hit {wl}: {len(written)} history records below global_write_start {xp} written {written[:6]}"
            )
        ref, p_ref = fresh[wl]
        assert p_ref == p_b, (p_ref, p_b)
        got = {**hist_recs(sl, xp, p_b), **snap_recs(sl, enc)}
        golden_ok = wl == "novel" and np.array_equal(tokens[wl][:p_b], gold_toks[:p_b])
        lim_fail = {"replay": [], "fresh": []}
        for c in hist + snaps:
            for layer in c["layers"] if not c.get("snapshot") else enc:
                g_c, f_c = by_cfg(got, c["name"]), by_cfg(ref, c["name"])
                if c.get("snapshot"):
                    a = C.assemble(g_c, c, layer, 0, C.SNAP)
                    b = C.assemble(f_c, c, layer, 0, C.SNAP)
                else:
                    a = C.valid_and_pad(g_c, c, layer, xp, p_b)[0]
                    b = C.valid_and_pad(f_c, c, layer, xp, p_b)[0]
                row = dict(cfg=c["name"], layer=layer, rel_vs_fresh=_rel(a, b), pcc_vs_fresh=_pcc(a, b))
                if golden_ok:
                    for side, recs, rws in (("replay", g_c, a), ("fresh", f_c, b)):
                        what = f"prefix hit {side}"
                        if c.get("snapshot"):
                            f = C.snapshot_failures(kvc, gk, c, recs, layer, p_b, thr, what)
                            nat = rws[np.arange(p_b - C.SNAP, p_b) % C.SNAP]
                            want = gk.select(c, layer, p_b - C.SNAP, p_b)
                            gi, gw = want[0], want[1]
                            row[f"pcc_golden_{side}"] = _pcc(nat[gi], gw) if len(gi) else None
                        else:
                            f = C.golden_failures(kvc, gk, c, rws, layer, xp, p_b, thr, what)
                            row[f"pcc_golden_{side}"] = _pcc(rws, gk.rows(c, layer, xp, p_b))
                        lim_fail[side] += f
                rows.append(row)
        worst = min(rows, key=lambda r: r["pcc_vs_fresh"])
        summary = (
            f"{len(written)} records below Xp {xp} written; rows [{xp}, {p_b}) vs fresh: worst pcc "
            f"{worst['pcc_vs_fresh']:.5f} ({worst['cfg']} L{worst['layer']:02d}), max rel "
            f"{max(r['rel_vs_fresh'] for r in rows):.5f}"
        )
        if golden_ok:
            summary += (
                f"; vs the golden's limits: replay {len(lim_fail['replay'])} over, fresh {len(lim_fail['fresh'])} over"
            )
            for side in ("replay", "fresh"):
                metrics.record(f"prefix_hit_limit_exceed_{side}", len(lim_fail[side]), first=lim_fail[side][:8])
        for kind in ("hist", "snap"):
            sel = [r for r in rows if (kind == "snap") == r["cfg"].startswith(C.CONFIG_ORDER[0])]
            if sel:
                metrics.record(f"prefix_hit_{wl}_{kind}_min_pcc_vs_fresh", min(r["pcc_vs_fresh"] for r in sel))
                metrics.record(f"prefix_hit_{wl}_{kind}_max_rel_vs_fresh", max(r["rel_vs_fresh"] for r in sel))
        metrics.record(f"prefix_hit_{wl}_rows", rows)
        for r in rows:
            logger.info(f"prefix hit {wl} {r}")
        for side in lim_fail:
            for f in lim_fail[side][:20]:
                logger.info(f"prefix hit {wl} (reported, not gated) {f}")
        return dict(fails=out, replay_rows_written=len(written), summary=summary)

    fails, cases, total_differ, total_recs, dec_written = [], [], 0, 0, 0
    fresh = {}
    hit = {}
    try:
        for n in lens:
            for wl in PROMPTS:
                p = decode_stop(n)
                fails += request(wl, "off", SLOT["off"], n)
                ref = {**hist_recs(SLOT["off"], 0, p), **snap_recs(SLOT["off"], enc)}
                dec_before = snap_recs(SLOT["decode"], dec)
                fails += request(wl, "decode", SLOT["decode"], n)
                got = {**hist_recs(SLOT["decode"], 0, p), **snap_recs(SLOT["decode"], enc)}
                dec_after = snap_recs(SLOT["decode"], dec)
                differ = sorted(k for k in ref if got.get(k) != ref[k])
                written = sorted(k for k in dec_before if dec_after.get(k) != dec_before[k])
                total_differ += len(differ)
                total_recs += len(ref)
                dec_written += len(written)
                case = dict(
                    prompt=wl, prompt_len=n, stop=p, chunks=len(chunks(0, p, chunk)), records=len(ref),
                    differ=len(differ), decoder_snapshot_written=len(written),
                )  # fmt: skip
                cases.append(case)
                logger.info(f"DECODER_SKIP=decode {case}" + (f" first differing {differ[:6]}" if differ else ""))
                if differ:
                    fails.append(f"{wl} N {n} P {p}: {len(differ)} of {len(ref)} records differ {differ[:6]}")
                if written:
                    fails.append(f"{wl} N {n} P {p}: the decode run wrote decoder snapshots {written[:6]}")
                if n == n_b:
                    fresh[wl] = (got, p)
                    hit[wl] = prefix_hit(wl)
    finally:
        model.use_decoder_skip("decode")
    if reader.replica_mismatch:
        fails.append(
            f"device-group replicas differ at {len(reader.replica_mismatch)} records: {reader.replica_mismatch[:3]}"
        )

    metrics.record("served_bytes_differ", total_differ)
    metrics.record("served_records_compared", total_recs)
    metrics.record("decoder_snapshot_written", dec_written)
    metrics.record("decode_cases", cases)
    for (mode, rows), w in sorted(walls.items()):
        if rows == chunk and len(w) > 1:
            metrics.record(f"decode_test_chunk_wall_ms_{mode}", float(np.median(w[1:])), runs=len(w) - 1)
    print(f"[decoder_skip_decode] layers {layers[0]}..{layers[-1]}, encoder {enc[0]}..{enc[-1]}, decoder {dec[0]}..")
    for c in cases:
        print(f"[decoder_skip_decode] {c}")
    for (mode, rows), w in sorted(walls.items()):
        print(f"[decoder_skip_decode] wall {mode} {rows}-row chunks (ms, with the per-layer acks' syncs): {w}")
    replay_written = sum(h["replay_rows_written"] for h in hit.values())
    metrics.record("replay_rows_written", replay_written)
    for wl, h in hit.items():
        fails += h["fails"]
        print(f"[decoder_skip_decode] prefix hit {wl}: {h['summary']}")
    assert not fails, "\n".join(fails[:40])


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return C.rel_err(a, b)
