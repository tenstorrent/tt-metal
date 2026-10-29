# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests for the Qwen3.6/3.8 KV-transfer model hook (PD disaggregation; PHASE2_DESIGN.md 8.7, AM3).

ONE process, ONE (1,1) mesh, the real 27B loaded ONCE (TP=1 mode, bf8 paged KV, B=8 decode slots), a fake in-process
Sink/Source pair (host tensors) standing in for the transport. Checks, in order:

  (a) A0 round trip: export after a T-1 prefill of a 300-token prompt on the B=1 prefill scratch (the R2 prefill-node
      shape), import into another slot + other blocks, byte-compare rec/taps/K-V blocks, then 16 greedy decode tokens
      from the imported slot == a straight prefill_paged_slots + decode of the same prompt in that slot.
  (d) kv_both producer shape: export the same request from slot s of the B=8 state (dumpfile host-slice path) and
      byte-compare against the B=1 export.
  (b) ORDER (G1/I11) + isolation, both slot placements (s inside / outside the pow2 bucket): POSITIVE = install at the
      join step's begin and decode {live.., s} together -> bit-exact vs the straight-prefill reference, live rows
      bit-exact vs a run without any import (isolation); NEGATIVE CONTROL = install, one decode step of the live rows
      with s idle, then decode s -> asserts the divergence the composite TP=1 decode causes (fused-conv: recorded).
  (c) both GDN decode modes: run the file twice, QWEN36_GDN_DECODE_FUSED=0 (composite) and =2 (fused conv, AM3 M2).
  (e) byte sizes and an export/import/decode round trip at T=4096 (chunk-trace replay + masked tail).
  STRICT_SHAPES: TT_PD_STRICT_SHAPES=1 makes the hook raise on a request-time program compile; the test also asserts
  the hook saw none.

Run (chip 3):
  source profiles/pd/p2_env_chip3.sh
  QWEN36_GDN_DECODE_FUSED=0 TT_PD_STRICT_SHAPES=1 pytest -svq models/demos/blackhole/qwen36/tests/test_kv_transfer_hook.py
  QWEN36_GDN_DECODE_FUSED=2 TT_PD_ALLOW_FUSED_CONV=1 TT_PD_STRICT_SHAPES=1 pytest -svq ... (fused-conv decode, M2)
"""
import gc
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.kv_transfer import Qwen36KVTransfer, cdiv, row_major_nbytes, tile_nbytes
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

B = 8  # decode slots (max_num_seqs of the decode node)
BLOCK = 64
BPU = 80  # blocks per user region: 4095 tokens need 64 (+ decode headroom); 8 x 80 = 640 is a multiple of 32
NUM_BLOCKS = B * BPU
PAD = NUM_BLOCKS  # the extra last block (allocate NUM_BLOCKS + 1)


# ------------------------------------------------------------------------------------------------------------------- #
# fake in-process transport (host tensors); the hook only ever calls these methods (design 4.1)
# ------------------------------------------------------------------------------------------------------------------- #
class FakeSink:
    def __init__(self, spec):
        self.spec = spec
        self.supports_regions = False
        self.chunks = {}
        self.rows = None

    def write_from_device(self, device_tensor, *, chunk, blocking=True, cq_id=None):
        h = ttnn.from_device(device_tensor)  # raw D2H, bytes as on device (what dump_tensor would serialize)
        assert tuple(int(d) for d in h.shape) == tuple(self.spec.shape), (tuple(h.shape), self.spec.shape)
        self.chunks[chunk] = h

    def write_region_from_device(self, *a, **k):
        raise NotImplementedError("fake transport is dumpfile-shaped")

    def write_rows(self, rows):
        assert tuple(rows.shape) == tuple(self.spec.shape) and rows.dtype == torch.bfloat16
        self.rows = rows.clone()

    def write_host(self, host_tensor, *, chunk):
        assert not ttnn.is_tensor_storage_on_device(host_tensor)
        assert tuple(int(d) for d in host_tensor.shape) == tuple(self.spec.shape)
        self.chunks[chunk] = host_tensor


class FakeChunk:
    is_head_major = False
    is_device_readable = False

    def __init__(self, host):
        self.host = host

    def read_into_device(self, staging, *, cq_id=None):
        raise NotImplementedError

    def read_device(self, mesh):
        return ttnn.to_device(self.host, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class FakeSource:
    def __init__(self, sink):
        self.spec = sink.spec
        self._chunks = dict(sink.chunks)
        self._rows = sink.rows
        if self.spec.kind == "gdn_taps":
            self.nbytes_present = row_major_nbytes(self._rows.shape, "bfloat16")
        else:
            self.nbytes_present = sum(tile_nbytes(self.spec.shape, self.spec.dtype) for _ in self._chunks)
        self.spec_crc = 0

    def crc32c(self):
        return 0

    def chunk(self, c):
        return FakeChunk(self._chunks[c])

    def read_rows(self):
        return self._rows


def make_sinks(manifest):
    return {p.name: FakeSink(p) for p in manifest.parts}


def make_sources(sinks):
    return {name: FakeSource(s) for name, s in sinks.items()}


# ------------------------------------------------------------------------------------------------------------------- #
# model-level helpers
# ------------------------------------------------------------------------------------------------------------------- #
def region(u):
    return list(range(u * BPU, (u + 1) * BPU))


def decode_step(model, vocab, width, active):
    """One eager decode forward at bucket `width`. active: {slot: (token, pos, blocks)}; other rows idle
    (token 0, position -1 -> paged_update_cache skips them, exactly the runner's padding rows). Returns [width, vocab]."""
    tokens = torch.zeros(width, 1, dtype=torch.int32)
    pos = torch.full((width,), -1, dtype=torch.int32)
    pt = torch.full((width, BPU), PAD, dtype=torch.int32)
    for slot, (tok, p, blocks) in active.items():
        assert 0 <= slot < width, (slot, width)
        tokens[slot, 0] = int(tok)
        pos[slot] = int(p)
        pt[slot] = torch.tensor(blocks, dtype=torch.int32)
    dev = model.prepare_inputs_decode(tokens, pos, pt)
    out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
    logits = model.process_output_decode(out, width)[:, 0, :vocab].float().clone()
    ttnn.deallocate(out)
    return logits


def greedy_decode(model, vocab, slot, width, first_token, pos0, steps, blocks):
    toks, logits = [], []
    tok, pos = int(first_token), int(pos0)
    for _ in range(steps):
        lg = decode_step(model, vocab, width, {slot: (tok, pos, blocks)})
        row = lg[slot]
        logits.append(row)
        tok = int(torch.argmax(row))
        toks.append(tok)
        pos += 1
    return toks, logits


def snapshot_scratch(model):
    """Per GDN layer: (rec [1,Nv,Dk,Dv] fp32 host, [tap rows [D] bf16]) of the currently bound (B=1) buffers."""
    out = []
    for dn in (l.attention for l in model.layers if not l.is_full_attention):
        rec = ttnn.to_torch(dn.rec_state)
        taps = [ttnn.to_torch(dn.conv_states[m])[0, 0].clone() for m in range(dn.K)]
        out.append((rec.clone(), taps))
    return out


def slot_state(model, slot):
    out = []
    for dn in (l.attention for l in model.layers if not l.is_full_attention):
        rec = ttnn.to_torch(dn.rec_state)[slot : slot + 1].clone()
        taps = [ttnn.to_torch(dn.conv_states[m])[0, slot].clone() for m in range(dn.K)]
        out.append((rec, taps))
    return out


def states_equal(a, b):
    bad = []
    for j, ((ra, ta), (rb, tb)) in enumerate(zip(a, b)):
        if not torch.equal(ra, rb):
            bad.append(f"L{j}.rec(maxdiff={(ra - rb).abs().max().item():.3e})")
        for m in range(len(ta)):
            if not torch.equal(ta[m], tb[m]):
                bad.append(f"L{j}.tap{m}")
    return bad


def rows_equal(a, b, rows):
    return all(torch.equal(a[u], b[u]) for u in rows)


def kv_blocks(hook, blocks):
    """{(name, i): host block tensor} for the first len(blocks) blocks of every K/V tensor (runtime-start slices)."""
    out = {}
    g = hook._geometry()
    for i, b in enumerate(blocks):
        keep = hook._upload_starts(int(b))
        for name, t in hook._kv_tensors():
            s = hook._unit_slice(t, g["num_blocks"])
            out[(name, i)] = ttnn.to_torch(s)
            ttnn.deallocate(s)
        del keep
    return out


def producer_prefill_export(model, hook, prompt, n, blocks, sinks, vocab):
    """The R2 prefill-node shape: B=1 prefill scratch bound, prefill_traced_chunked(prompt[:n]) (masked bucket or chunk
    replay + tail), export straight from the layer buffers (dn.B == 1)."""
    prev = model._bind_gdn_prefill_scratch()
    try:
        pt = torch.tensor([blocks], dtype=torch.int32)
        t0 = time.perf_counter()
        lg = model.prefill_traced_chunked(prompt[:, :n], pt, actual_len=n)
        ttnn.synchronize_device(model.mesh_device)
        t1 = time.perf_counter()
        logits = ttnn.to_torch(lg).reshape(-1, vocab)[0].float().clone()
        ttnn.deallocate(lg)
        snap = snapshot_scratch(model)
        t2 = time.perf_counter()
        hook.export_request_state(blocks, n, 0, sinks)
        t3 = time.perf_counter()
        logger.info(f"[hook-test] producer prefill n={n}: prefill {1e3*(t1-t0):.0f} ms, export {1e3*(t3-t2):.0f} ms")
    finally:
        model._unbind_gdn_prefill_scratch(prev)
    return logits, snap, t3 - t2


def consumer_prefill(model, prompts, ns, slots):
    pt = torch.stack([torch.tensor(region(s), dtype=torch.int32) for s in slots])
    return model.prefill_paged_slots([p[:, :n] for p, n in zip(prompts, ns)], pt, list(slots), valid_lens=list(ns))


def import_all(hook, sources, blocks, n, slot):
    t0 = time.perf_counter()
    done = hook.import_kv_blocks(sources, blocks, n)
    ttnn.synchronize_device(hook.mesh)
    t1 = time.perf_counter()
    hook.validate_gdn_parts(sources)
    t2 = time.perf_counter()
    hook.install_gdn_state(sources, slot)
    t3 = time.perf_counter()
    logger.info(
        f"[hook-test] import n={n} slot={slot}: kv {1e3*(t1-t0):.0f} ms ({done} chunks), validate {1e3*(t2-t1):.1f} ms, "
        f"install {1e3*(t3-t2):.0f} ms"
    )
    return t1 - t0, t3 - t2


# ------------------------------------------------------------------------------------------------------------------- #
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 536870912}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 1), id="1x1")], indirect=True)
def test_kv_transfer_hook(mesh_device, reset_seeds, ensure_gc):
    os.environ.setdefault("TT_PD_TIMING", "1")
    fused_env = os.environ.get("QWEN36_GDN_DECODE_FUSED", "2") == "2"
    strict = os.environ.get("TT_PD_STRICT_SHAPES", "0") == "1"
    fails = []
    report = []

    def check(cond, msg):
        (report if cond else fails).append(("PASS " if cond else "FAIL ") + msg)
        logger.info(("PASS " if cond else "FAIL ") + msg)

    def note(msg):
        report.append("INFO " + msg)
        logger.info("INFO " + msg)

    torch.manual_seed(1234)
    t_load = time.perf_counter()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=8192)
    assert model.use_tp, "PD hook tests run the TP code path on one die (QWEN36_FORCE_TP_PATH=1)"
    args = model.args
    vocab = args.vocab_size
    model.allocate_kv_caches((NUM_BLOCKS + 1, args.n_local_kv_heads, BLOCK, args.head_dim), ttnn.bfloat8_b, batch_size=B)
    assert model._pad_kv_block == PAD
    hook = Qwen36KVTransfer(model)
    dns = hook.gdn_layers
    fused = bool(dns[0]._decode_fused_conv)
    note(f"model loaded in {time.perf_counter()-t_load:.0f} s; GDN layers={len(dns)} attn layers={len(hook.attn_layers)} "
         f"fused_conv_decode={fused} (env {fused_env}) rec dtype={dns[0].rec_state.dtype} kv dtype={hook.attn_layers[0].paged_k.dtype}")
    check(fused == fused_env, "GDN decode mode follows QWEN36_GDN_DECODE_FUSED")

    # ---- 1. decode compile warmup at every bucket width (before any trace is parked), like warmup_model_decode ----
    t0 = time.perf_counter()
    for w in (1, 2, 4, 8):
        decode_step(model, vocab, w, {0: (1, 0, region(0))})
    for dn in dns:
        dn.reset_state_inplace()
        dn._hist_packed_valid = False
    model.sync_gdn_decode_state()  # fused conv: rebuild the packed history from the (all-zero) conv_states
    ttnn.synchronize_device(mesh_device)
    note(f"decode compile warmup (widths 1,2,4,8): {time.perf_counter()-t0:.0f} s")

    # ---- 2. PD hook warmup (design 5.5) ----
    n_before = mesh_device.num_program_cache_entries()
    hook.warmup_kv_transfer(role="kv_both", mode="dumpfile", chunk_tokens=2048, slots=range(B))
    note(f"warmup_kv_transfer: +{mesh_device.num_program_cache_entries()-n_before} programs")

    # ---- 3. prefill warmup exactly as qwen36_vllm.warmup_model_prefill (batched: B=1 scratch + parked chunk trace) ----
    t0 = time.perf_counter()
    warmup_pt = torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, NUM_BLOCKS)
    prev = model._bind_gdn_prefill_scratch()
    try:
        model.capture_prefill_trace_chunked(mesh_device, warmup_pt, chunk_size=2048, capture_chunk_trace=True)
    finally:
        model._unbind_gdn_prefill_scratch(prev)
    assert model._chunked_trace_id is not None
    model.warmup_gdn_slot_write()
    note(f"prefill warmup (masked buckets + chunk trace): {time.perf_counter()-t0:.0f} s; "
         f"program cache entries={mesh_device.num_program_cache_entries()}")
    n_pc_after_warmup = mesh_device.num_program_cache_entries()

    # ---- (a) round trip, T = 300 ----
    T = 300
    n = T - 1
    P0 = torch.randint(0, vocab, (1, T), dtype=torch.long)
    manifest = hook.describe_request_state(n, region(7))
    check(manifest.nblk == cdiv(n, BLOCK) and manifest.num_tokens == n, f"manifest nblk={manifest.nblk} for {n} tokens")
    check(len(manifest.parts) == 2 * len(hook.attn_layers) + 2 * len(dns), f"manifest has {len(manifest.parts)} parts")
    sinks = make_sinks(manifest)
    pf_logits_src, snap, export_s = producer_prefill_export(model, hook, P0, n, region(7), sinks, vocab)
    # exported bytes == the scratch state
    bad = []
    for j in range(len(dns)):
        if not torch.equal(ttnn.to_torch(sinks[f"gdn.L{j}.rec"].chunks[0]), snap[j][0]):
            bad.append(f"L{j}.rec")
        if not torch.equal(sinks[f"gdn.L{j}.taps"].rows, torch.stack(snap[j][1])):
            bad.append(f"L{j}.taps")
    check(not bad, f"(a) exported rec/taps bytes == B=1 prefill state (bad: {bad[:6]})")
    check(all(len(sinks[nm].chunks) == manifest.nblk // 32 + (1 if manifest.nblk % 32 else 0) for nm, _ in hook._kv_tensors()),
          "(a) every K/V part has cdiv(nblk, 32) chunks")
    src_kv = kv_blocks(hook, region(7)[: manifest.nblk])
    sources = make_sources(sinks)

    # reference on the consumer: straight prefill_paged_slots into slot 2 / region 2, then decode
    s_ref = 2
    pf_ref = consumer_prefill(model, [P0], [n], [s_ref])[0].reshape(-1, vocab)[0].float()
    check(torch.equal(pf_ref, pf_logits_src) or torch.allclose(pf_ref, pf_logits_src, atol=0.0),
          f"(a) prefill logits scratch vs prefill_paged_slots (maxdiff {(pf_ref-pf_logits_src).abs().max().item():.3e})")
    st_ref = slot_state(model, s_ref)
    bad = states_equal(st_ref, snap)
    check(not bad, f"(a) slot {s_ref} state after prefill_paged_slots == exported bytes (bad: {bad[:6]})")

    # (d) kv_both producer shape: export from slot s_ref of the B=8 state (host-slice rec, host tap rows)
    sinks2 = make_sinks(manifest)
    t0 = time.perf_counter()
    hook.export_request_state(region(s_ref), n, s_ref, sinks2)
    export_b8_s = time.perf_counter() - t0
    bad = []
    for name in sinks:
        a, b = sinks[name], sinks2[name]
        if a.rows is not None:
            if not torch.equal(a.rows, b.rows):
                bad.append(name)
        else:
            for c in a.chunks:
                if not torch.equal(ttnn.to_torch(a.chunks[c]), ttnn.to_torch(b.chunks[c])):
                    bad.append(f"{name}.c{c}")
    check(not bad, f"(d) kv_both (B=8, slot {s_ref}) export bytes == B=1 export (bad: {bad[:6]}); "
                   f"B=1 export {1e3*export_s:.0f} ms, B=8 export {1e3*export_b8_s:.0f} ms")

    W = 4  # decode bucket used for the single active row 2 (rows 0, 1, 3 idle)
    toks_ref, lg_ref = greedy_decode(model, vocab, s_ref, W, P0[0, T - 1], T - 1, 16, region(s_ref))

    # import: K/V into region 3 (other blocks), GDN row into slot 2 (the reference slot, now advanced by 16 steps)
    kv_s, inst_s = import_all(hook, sources, region(3), n, s_ref)
    st_imp = slot_state(model, s_ref)
    bad = states_equal(st_imp, snap)
    check(not bad, f"(a) imported slot {s_ref} rec/taps == exported bytes (bad: {bad[:6]})")
    dst_kv = kv_blocks(hook, region(3)[: manifest.nblk])
    bad = [k for k in src_kv if not torch.equal(src_kv[k], dst_kv[k])]
    check(not bad, f"(a) imported K/V blocks (region 3) == source blocks (region 7), {len(src_kv)} block tensors (bad: {len(bad)})")
    if fused:
        check(all(dn._hist_packed_valid for dn in dns), "(a) fused conv: _hist_packed_valid stays True across the install")
    toks_imp, lg_imp = greedy_decode(model, vocab, s_ref, W, P0[0, T - 1], T - 1, 16, region(3))
    exact_steps = sum(torch.equal(a, b) for a, b in zip(lg_ref, lg_imp))
    maxdiff = max((a - b).abs().max().item() for a, b in zip(lg_ref, lg_imp))
    check(toks_imp == toks_ref, f"(a) 16 greedy tokens from the imported slot == straight prefill+decode: {toks_imp == toks_ref} "
                                f"(logits bit-exact on {exact_steps}/16 steps, max |diff| {maxdiff:.3e}); import kv {1e3*kv_s:.0f} ms install {1e3*inst_s:.0f} ms")
    note(f"(a) tokens ref={toks_ref}")
    note(f"(a) tokens imp={toks_imp}")

    # ---- (b) ORDER / isolation, both placements ----
    Ps = torch.randint(0, vocab, (1, T), dtype=torch.long)  # the remote request (T-1 prefilled by the producer)
    sinks_s = make_sinks(manifest)
    _, snap_s, _ = producer_prefill_export(model, hook, Ps, n, region(7), sinks_s, vocab)
    sources_s = make_sources(sinks_s)
    live_prompts = {u: torch.randint(0, vocab, (1, 256), dtype=torch.long) for u in range(4)}
    live_len = {u: 160 + 32 * u for u in range(4)}  # {160, 192, 224, 256}
    d_tok = torch.randint(0, vocab, (2, 4)).tolist()  # fixed decode tokens per step / user

    def live_active(users, step):
        return {u: (d_tok[step][u], live_len[u] + step, region(u)) for u in users}

    for placement, live, s, w1, w2 in (("inside", [0, 1, 2], 3, 4, 4), ("outside", [0, 1, 2, 3], 4, 4, 8)):
        tag = f"(b) s={s} {placement} bucket (live {live}, D1 width {w1}, D2 width {w2})"
        prompts = [live_prompts[u] for u in live]
        ns = [live_len[u] for u in live]

        # A': live rows only (reference trajectories, no import anywhere)
        consumer_prefill(model, prompts, ns, live)
        ref_d1 = decode_step(model, vocab, w1, live_active(live, 0))
        ref_d2 = decode_step(model, vocab, w2, live_active(live, 1))

        # A: reference for s = a straight prefill into slot s in a prefill step between D1 and D2 (the non-PD flow)
        consumer_prefill(model, prompts, ns, live)
        a_d1 = decode_step(model, vocab, w1, live_active(live, 0))
        # only the LIVE rows are compared: an idle padding row inside the bucket (row 3 in the "inside" placement)
        # decodes token 0 at position -1 from whatever its slot holds, so its logits are don't-care and run-dependent
        check(rows_equal(a_d1, ref_d1, live), f"{tag}: prefill+decode deterministic (D1 live rows bit-exact on replay)")
        consumer_prefill(model, [Ps], [n], [s])
        act = live_active(live, 1)
        act[s] = (int(Ps[0, T - 1]), T - 1, region(s))
        a_d2 = decode_step(model, vocab, w2, act)
        s_ref_logits = a_d2[s]
        live_same = all(torch.equal(a_d2[u], ref_d2[u]) for u in live)
        if fused:
            note(f"{tag}: [Phase-1 fused-conv probe] live rows after a prefill_paged_slots between their decodes: "
                 f"{'bit-exact' if live_same else 'CHANGED (write_slot full packed-history rebuild from stale conv_states)'}")
        else:
            check(live_same, f"{tag}: live rows unaffected by a prefill_paged_slots into s between D1 and D2")

        # B (POSITIVE): K/V land early, GDN row installed at the join step's begin, s decodes in that step
        consumer_prefill(model, prompts, ns, live)
        hook.import_kv_blocks(sources_s, region(s), n)
        ttnn.synchronize_device(mesh_device)
        hook.validate_gdn_parts(sources_s)
        b_d1 = decode_step(model, vocab, w1, live_active(live, 0))
        check(rows_equal(b_d1, ref_d1, live), f"{tag}: POSITIVE: D1 live rows (K/V imported, GDN not yet installed) == reference")
        hook.install_gdn_state(sources_s, s)
        b_d2 = decode_step(model, vocab, w2, act)
        check(torch.equal(b_d2[s], s_ref_logits),
              f"{tag}: POSITIVE: row s logits bit-exact vs straight prefill (max|diff| {(b_d2[s]-s_ref_logits).abs().max().item():.3e})")
        check(all(torch.equal(b_d2[u], ref_d2[u]) for u in live), f"{tag}: ISOLATION: live rows bit-exact vs the no-import run")

        # C (NEGATIVE CONTROL): install, then one decode step of the live rows with s idle, then decode s at T-1
        consumer_prefill(model, prompts, ns, live)
        hook.import_kv_blocks(sources_s, region(s), n)
        ttnn.synchronize_device(mesh_device)
        hook.install_gdn_state(sources_s, s)
        c_d1 = decode_step(model, vocab, w1, live_active(live, 0))
        check(rows_equal(c_d1, ref_d1, live), f"{tag}: NEGATIVE: an installed idle row does not disturb the live rows' D1")
        c_d2 = decode_step(model, vocab, w2, act)
        differs = not torch.equal(c_d2[s], s_ref_logits)
        top1_same = int(torch.argmax(c_d2[s])) == int(torch.argmax(s_ref_logits))
        if fused:
            note(f"{tag}: NEGATIVE (fused conv): row s after one idle decode step {'DIFFERS' if differs else 'is bit-exact'} "
                 f"(top1 same={top1_same}, max|diff| {(c_d2[s]-s_ref_logits).abs().max().item():.3e}) -- recorded, not asserted")
        else:
            check(differs, f"{tag}: NEGATIVE: row s after one idle decode step differs from the reference "
                           f"(max|diff| {(c_d2[s]-s_ref_logits).abs().max().item():.3e}, top1 same={top1_same}) -> early install is forbidden (I11)")
        check(all(torch.equal(c_d2[u], ref_d2[u]) for u in live), f"{tag}: NEGATIVE: live rows still bit-exact")

    # ---- eager decode step time at full width (op-dispatch bound; the served/traced number is the plugin's) ----
    consumer_prefill(model, [live_prompts[u % 4] for u in range(B)], [live_len[u % 4] for u in range(B)], list(range(B)))
    act8 = {u: (d_tok[0][u % 4], live_len[u % 4], region(u)) for u in range(B)}
    decode_step(model, vocab, B, act8)
    t0 = time.perf_counter()
    for _ in range(5):
        decode_step(model, vocab, B, act8)
    note(f"eager decode step at width {B} ({'fused-conv' if fused else 'composite'} GDN): {1e3*(time.perf_counter()-t0)/5:.1f} ms/step "
         f"(eager host dispatch incl. logits readback; NOT the traced serving TPOT)")

    # ---- (e) T = 4096: sizes + chunk-trace replay export/import/decode ----
    T4 = 4096
    n4 = T4 - 1
    m4 = hook.describe_request_state(n4, region(7))
    real_payload = m4.nblk * 32 * tile_nbytes((1, 4, 64, 256), m4.kv_dtype) + len(dns) * (
        tile_nbytes((1,) + hook._geometry()["rec_shape"], m4.rec_dtype) + row_major_nbytes((4, 10240), "bfloat16")
    )
    check(m4.nblk == 64 and real_payload == 297_533_440,
          f"(e) T=4096 sizes: nblk={m4.nblk}, real payload {real_payload:,} B (design 5.6: 297,533,440), on-wire {m4.total_nbytes:,} B")
    P4 = torch.randint(0, vocab, (1, T4), dtype=torch.long)
    sinks4 = make_sinks(m4)
    _, snap4, export4_s = producer_prefill_export(model, hook, P4, n4, region(7), sinks4, vocab)
    src4 = kv_blocks(hook, region(7)[: m4.nblk])
    sources4 = make_sources(sinks4)
    s6 = 6
    consumer_prefill(model, [P4], [n4], [s6])
    toks4_ref, _ = greedy_decode(model, vocab, s6, 8, P4[0, T4 - 1], T4 - 1, 4, region(s6))
    kv4_s, inst4_s = import_all(hook, sources4, region(5), n4, s6)
    bad = states_equal(slot_state(model, s6), snap4)
    dst4 = kv_blocks(hook, region(5)[: m4.nblk])
    badkv = [k for k in src4 if not torch.equal(src4[k], dst4[k])]
    check(not bad and not badkv, f"(e) T=4096 imported rec/taps/K-V bytes exact (bad gdn {bad[:4]}, bad kv {len(badkv)})")
    toks4_imp, _ = greedy_decode(model, vocab, s6, 8, P4[0, T4 - 1], T4 - 1, 4, region(5))
    check(toks4_imp == toks4_ref, f"(e) T=4096 4 greedy tokens imported == straight: {toks4_imp} vs {toks4_ref}; "
                                  f"dumpfile-shaped export {1e3*export4_s:.0f} ms, import kv {1e3*kv4_s:.0f} ms, install {1e3*inst4_s:.0f} ms")

    # ---- STRICT_SHAPES ----
    check(hook.request_time_compiles == 0,
          f"STRICT_SHAPES: hook calls compiled no programs after warmup (strict={strict}; program cache "
          f"{n_pc_after_warmup} -> {mesh_device.num_program_cache_entries()} incl. non-hook eager decode/prefill compiles)")

    logger.info("\n".join(["==== kv_transfer hook report ===="] + report + fails))
    model.free_kv_caches()
    gc.collect()
    assert not fails, "\n".join(fails)
