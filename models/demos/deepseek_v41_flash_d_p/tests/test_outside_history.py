# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.39 (owner review 2026-11-18, item C): a slot whose history was written from OUTSIDE the model gives the same
chunk as a slot that wrote it itself: the served-state path (the KV Manager fills a slot's served caches after a store
load or a d2p pull; the model never sees the write, its host ``version`` counters do not move).

One model load (a bring-up batch's, BATCH_SHARED), run as DECODER_SKIP=decode (``use_decoder_skip``, host lists only,
FIX.28; put back after). Per variant, two fresh slot states A and B of 2 chunks (tt/model.DS41DeviceState, own slabs:
the same tt/served.CacheSlab code and layout the engine's slots use, one slab per cache):

    1. A runs chunk k = [0, C) (self-written; C = target.chunk, the novel prompt's ids).
    2. Every served cache of A (each layer's 0_window_kv snapshot, each KV source's compressed KV and its 192-byte
       index rows, every re-index mirror) is copied into B's as a raw DRAM image: ttnn.from_device of A's tensor, then
       ttnn.copy_host_to_device_tensor into B's (a host -> DRAM write of the same bytes, as the KV Manager's UMD write;
       nothing through the model's write path, no load_host; B's counters stay 0). Checked: B's images == A's.
    3. Chunk k + 1 runs in A, then in B, with the same ids and arguments, and every encoder layer's output (layers
       0-19, the decoder never runs under decode) and every served cache image after it are compared A vs B.

Chunk k + 1 is the replay chunk [C - 128, 2C - 128) with global_write_start C (DeepSeek's Encoder SWA Bounded
Replay, P.51): every window is truncated at C - 128 in both slots, so the window ring (item 2: carried in the slot's
own device ring, which no outside write fills) is not read, and no history row below C is written. The engram tail
(item 1: the compressed ids of the 3 positions before the chunk, host) needs the prefix's ids: B is told them with
``note_tokens(ids[0:C], 0, C)`` (host bookkeeping only, no device write), as a resumed slot must be. Variants:

    shipped        the shipped settings (STATIC_TABLES=off, INDEX_K_SOURCE=served): outside_history_max_abs_diff
                   (== 0, asserted), the max |A - B| over every layer output and served image
    static_on      STATIC_TABLES=on (the per-slot replicated AttnTable, item 3: written only by this process)
    index_private  INDEX_K_SOURCE=private (the private k8 index-key copy, item 4: written only by this process)
    no_prefix_ids  shipped, B NOT told the prefix's ids (item 1: the engram tail of the replay start)
    window_carry   shipped, chunk k + 1 = [C, 2C) without replay (item 2: B's ring holds no carry, so its window is
                   truncated at C while A's reads its carry)
The four documentation variants are recorded (outside_history_max_abs_diff_<variant>, the first differing layer, the
differing caches), never gated; STATIC_TABLES / INDEX_K_SOURCE are read at state build (and the sparse attentions'
``static`` flag, a host flag), so they are switched for their variant's states only, on the same load.

Also checked (FIX.39 item A): right after each begin_chunk no history-gather buffer holds a prefetch marker
(tt/history_gather.new_chunk).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_outside_history.py
"""

from __future__ import annotations

import os

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import b2b, batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tt import history_gather
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True  # FIX.29: takes the model of a bring-up batch process (core/batch.shareable)
REPLAY = 128

# name -> (settings for the variant's states, B told the prefix ids, replay chunk)
VARIANTS = {
    "shipped": ({}, True, True),
    "static_on": ({"STATIC_TABLES": "on"}, True, True),
    "index_private": ({"INDEX_K_SOURCE": "private"}, True, True),
    "no_prefix_ids": ({}, False, True),
    "window_carry": ({}, True, False),
}


def _served(st) -> dict:
    """{name: CacheSlab} of every served cache this slot state owns."""
    from models.demos.deepseek_v41_flash_d_p.tt.served import CacheSlab

    out = {}
    for i, c in st.window.items():
        for k in ("snap", "slab"):
            s = getattr(c, k, None)
            if isinstance(s, CacheSlab):
                out[f"L{i:02d}.window.{k}"] = s
    for i, c in st.compress.items():
        if i in st.shared_compress:
            continue
        s = getattr(c.ckv, "slab", None)
        if isinstance(s, CacheSlab):
            out[f"L{i:02d}.ckv"] = s
        ik = c.index_k
        for k in ("served", "slab"):
            s = getattr(ik, k, None)
            if isinstance(s, CacheSlab):
                out[f"L{i:02d}.index_k.{k}"] = s
        for n, s in enumerate(getattr(ik, "mirrors", None) or []):
            if isinstance(s, CacheSlab):
                out[f"L{i:02d}.index_k.mirror{n}"] = s
    return out


def _images(st) -> dict:
    """{name: [per-chip host tensor]} of every served cache (test read-back)."""
    import ttnn

    return {n: [ttnn.to_torch(d) for d in ttnn.get_device_tensors(s.tensor)] for n, s in _served(st).items()}


def _raw_copy(src_st, dst_st) -> int:
    """Every served cache of ``src_st`` -> ``dst_st``'s as a raw DRAM image (host -> DRAM write of the same bytes,
    the model's write path and counters untouched). Returns the number of tensors copied."""
    import ttnn

    src, dst = _served(src_st), _served(dst_st)
    assert sorted(src) == sorted(dst), (sorted(src)[:4], sorted(dst)[:4])
    for n in src:
        a, b = src[n].tensor, dst[n].tensor
        assert tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype, (n, a.shape, b.shape, a.dtype, b.dtype)
        ttnn.copy_host_to_device_tensor(ttnn.from_device(a), b)
    return len(src)


def _diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.dtype in (torch.uint8, torch.int32, torch.int64):
        return float((a.long() - b.long()).abs().max()) if a.numel() else 0.0
    return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0


def _image_diffs(x: dict, y: dict) -> dict:
    """{name: max |x - y| over every chip} of the caches that differ (uint8 rows: byte values)."""
    out = {}
    for n in x:
        d = max(_diff(p, q) for p, q in zip(x[n], y[n]))
        if d != 0:
            out[n] = d
    return out


def _sparse_attentions(model) -> list:
    from models.demos.deepseek_v41_flash_d_p.tt.sparse_attention import TtSparseAttention

    return [b.attention for b in model.blocks.values() if isinstance(getattr(b, "attention", None), TtSparseAttention)]


class _Switch:
    """The variant's settings for its states (environment + settings.refresh, back after); STATIC_TABLES also flips
    the sparse attentions' load-time ``static`` flag and the model's history-gather hide (host flags only)."""

    def __init__(self, model, values: dict):
        self.model, self.values, self.old_env, self.old = model, values, {}, {}

    def __enter__(self):
        for k, v in self.values.items():
            name = settings.env_name(k)
            self.old_env[name] = os.environ.get(name)
            os.environ[name] = v
        settings.refresh()
        for k, v in self.values.items():
            assert settings.get(k) == v, (k, settings.get(k), v)
        if "STATIC_TABLES" in self.values:
            on = self.values["STATIC_TABLES"] == "on"
            self.old["static"] = [(a, a.static) for a in _sparse_attentions(self.model)]
            self.old["hide"] = self.model._hide
            for a, _ in self.old["static"]:
                a.static = on
            self.model._hide = self.model._hide and not on
        return self

    def __exit__(self, *exc):
        for name, v in self.old_env.items():
            if v is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = v
        settings.refresh()
        for a, v in self.old.get("static", []):
            a.static = v
        if "hide" in self.old:
            self.model._hide = self.old["hide"]
        return False


def _markers() -> list:
    """History-gather buffers holding a prefetch marker (tt/history_gather.HistoryBuffer.pre)."""
    return [k for _, bufs in history_gather._BUF.values() for k, b in bufs.items() if b.pre is not None]


@mesh_parametrize(timeout_s=2 * 3600)
def test_outside_history(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    chunk = int(S.get("target.chunk"))
    layers = list(S.layers())
    seq = 2 * chunk
    ids_all = torch.tensor(b2b.ids(S, "novel", seq), dtype=torch.int64)
    assert ids_all.numel() >= seq, (ids_all.numel(), seq)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    assert model.max_seq >= seq, (model.max_seq, seq)
    loaded, prev_state = model.skip_mode, model.state
    if loaded != "decode":
        model.use_decoder_skip("decode")
    assert model.skip_mode == "decode" and model.decoder, (model.skip_mode, model.decoder)
    enc = [i for i in layers if i not in model.decoder]
    print(f"[outside_history] chunk {chunk}, encoder layers {enc[0]}..{enc[-1]}, decoder {model.decoder[0]}.. skipped")
    marker_fails = []

    def run(st, a: int, g: int | None, sink=None):
        """One chunk of C ids at ``a`` in slot state ``st`` (the adapter's calls); sink(layer, host output)."""
        assert a + chunk <= seq, (a, chunk, seq)
        ids = ids_all[a : a + chunk].clone()
        model.state = st
        st.note_tokens(ids, a, a + chunk)
        model.begin_chunk(ids, a, last=False, write_start=g)
        left = _markers()
        if left:
            marker_fails.append((a, left[:4]))
        h = model.embed(ids, a)
        for i in layers:
            h2 = model.layer(i, h, a, st, end=a + chunk)
            if h2 is not h:
                model.free(h)
            h = h2
            if sink is not None and i in enc:
                sink(i, model.to_host(h))
        model.free(h)
        model.sync()

    results, fails = {}, []
    try:
        for name, (values, tell_ids, replay) in VARIANTS.items():
            try:
                results[name] = _variant(model, run, enc, chunk, seq, values, tell_ids, replay, ids_all)
            except Exception as e:  # noqa: BLE001  (a documentation variant that cannot run is reported, not hidden)
                if name == "shipped":
                    raise
                results[name] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"}
            r = results[name]
            print(f"[outside_history] {name} ({values or 'shipped'}, prefix ids {tell_ids}, replay {replay}): {r}")
    finally:
        model.state = prev_state
        if loaded != "decode":
            model.use_decoder_skip(loaded)

    for name, r in results.items():
        suffix = "" if name == "shipped" else f"_{name}"
        if "error" in r:
            metrics.record(f"outside_history_error{suffix}", r["error"])
            continue
        metrics.record(f"outside_history_max_abs_diff{suffix}", r["max_abs_diff"])
        metrics.record(f"outside_history_layers_differ{suffix}", r["layers_differ"])
        metrics.record(f"outside_history_first_layer_differ{suffix}", r["first_layer"])
        metrics.record(f"outside_history_caches_differ{suffix}", len(r["caches_differ"]), first=r["caches_differ"][:8])
    sh = results["shipped"]
    metrics.record("outside_history_caches_copied", sh["copied"])
    metrics.record("outside_history_caches_written", sh["written"])
    metrics.record("outside_history_markers_left", len(marker_fails), first=marker_fails[:4])
    if sh["copy_bad"]:
        fails.append(f"the raw copy did not land: {sh['copy_bad'][:6]}")
    if not sh["written"]:
        fails.append("chunk k + 1 wrote no served cache (nothing compared)")
    if marker_fails:
        fails.append(f"history-gather prefetch markers left after begin_chunk: {marker_fails[:4]}")
    if sh["max_abs_diff"] != 0:
        fails.append(
            f"shipped: outside-written slot B != self-written slot A: max abs diff {sh['max_abs_diff']}, layers "
            f"{sh['layers_differ'][:8]}, caches {sh['caches_differ'][:8]}"
        )
    assert not fails, "\n".join(fails)


def _variant(model, run, enc, chunk, seq, values, tell_ids, replay, ids_all) -> dict:
    a1, g1 = (chunk - REPLAY, chunk) if replay else (chunk, None)
    with _Switch(model, values):
        st_a = model.new_state(seq)
        st_b = None
        try:
            run(st_a, 0, None)  # chunk k, self-written
            img_k = _images(st_a)
            st_b = model.new_state(seq)
            copied = _raw_copy(st_a, st_b)
            model.sync()
            copy_bad = sorted(_image_diffs(img_k, _images(st_b)))
            if tell_ids:  # (item 1: the slot's id history, host only)
                st_b.note_tokens(ids_all[:chunk].clone(), 0, chunk)
            outs = {}
            run(st_a, a1, g1, sink=lambda i, t: outs.__setitem__(i, t))
            img_a = _images(st_a)
            written = sorted(_image_diffs(img_k, img_a))
            layer_diff = {}

            def cmp(i, t):
                d = _diff(outs.pop(i), t)
                if d != 0:
                    layer_diff[i] = d

            run(st_b, a1, g1, sink=cmp)
            img_b = _images(st_b)
        finally:
            for st in (st_a, st_b):
                if st is not None:
                    st.free()
    caches = _image_diffs(img_a, img_b)
    worst = max([0.0] + list(layer_diff.values()) + list(caches.values()))
    return dict(
        max_abs_diff=worst,
        layers_differ=sorted(layer_diff),
        first_layer=min(layer_diff) if layer_diff else -1,
        first_layer_diff=layer_diff[min(layer_diff)] if layer_diff else 0.0,
        caches_differ=sorted(caches),
        copied=copied,
        written=len(written),
        copy_bad=copy_bad,
    )
