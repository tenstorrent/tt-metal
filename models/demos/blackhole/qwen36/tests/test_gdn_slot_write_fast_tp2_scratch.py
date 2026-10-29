# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH bit-exactness + timing check of the plain served slot write (prefill_paged_slots) on a (1,2) mesh
(SLOT_FAST_MESH=1x4: the (1,4) TP=4 mesh of a P300x2 / P150x4).

QWEN36_PLAIN_GDN_SLOT_FAST=0 is the host round trip (snapshot the B=1 scratch GDN state to host, re-upload it and
write it into the decode row); =1 (default) writes the fp32 recurrent state with ttnn.fill_cache on device and feeds
the packed-history repack from the tap snapshot. Run the SAME admission sequence once per knob value (one process
each; the traced prefill is deterministic) and compare the per-admission digests of every batched GDN buffer
(rec_state, conv_states[0..3], conv_hist_packed: all B rows, both devices) and of the logits row.

    SLOT_FAST_OUT=<json> QWEN36_PLAIN_GDN_SLOT_FAST=0|1 pytest -svq \\
        models/demos/blackhole/qwen36/tests/test_gdn_slot_write_fast_tp2_scratch.py
    python models/demos/blackhole/qwen36/tests/test_gdn_slot_write_fast_tp2_scratch.py a.json b.json   # compare

Env: SLOT_FAST_LAYERS (default 16 = 12 GDN + 4 attention layers), SLOT_FAST_B (default 32 = the tp2 max_num_seqs),
SLOT_FAST_DECODE (default 0): > 0 = the served interleaving -- the plugin's warmup order (model_runner phase 1 compiles
the decode widths 8 and B; phase 2 captures the chunk-prefill trace + warms the slot write, THEN captures the decode
traces, so every persistent prefill buffer exists before a decode trace is parked), and after every admission step
SLOT_FAST_DECODE traced decode steps
run over every admitted slot (inactive rows at position -1, width = 8 while all live slots are < 8, else B, like the
plugin's decode bucketing), so each later admission lands next to live, advanced decode rows (fused-conv decode:
stale taps, advanced packed history). Every decode step's logits rows and the full GDN state after the decode steps
are digested too. The gate compares the LIVE rows (admitted slots, per device) of every buffer: inactive decode rows
(position -1) attend over never-written KV blocks, so from the first attention layer on their GDN rows hold
run-dependent garbage (knob 0 vs knob 0 differs there); whole-buffer digests are still recorded and reported.
"""

import hashlib
import json
import os
import sys
import time

import pytest
import torch
from loguru import logger

import ttnn

_MESH = os.environ.get("SLOT_FAST_MESH", "1x2")
BLOCK = 64
BPU = 80  # blocks per slot region (5120 tokens)
# (slots, prompt lengths) per admission step: fresh slots, an overwrite of a live slot, a 2-user step, the last slot,
# the masked bucket (< 2048), the exact chunk (2048) and chunk + masked tail (> 2048).
ADMISSIONS = [
    ([0], [300]),
    ([1], [2048]),
    ([5], [777]),
    ([1], [129]),
    ([2, 3], [64, 3000]),
    ([31], [2047]),
    ([0], [4100]),
]


def _digest(t):
    return hashlib.sha1(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def _state_digests(model, comp, rows=None):
    """Whole-buffer digests; with rows, per-device digests of those slot rows only (key "<buf>[d<dev>s<slot>]")."""
    out = {}
    li = 0

    def put(key, t, row_dim):
        if rows is None:
            out[key] = _digest(ttnn.to_torch(t, mesh_composer=comp))
            return
        for d, dt in enumerate(ttnn.get_device_tensors(t)):
            x = ttnn.to_torch(dt)
            for s in rows:
                out[f"{key}[d{d}s{s}]"] = _digest(x.select(row_dim, s))

    for layer in model.layers:
        if layer.is_full_attention:
            continue
        dn = layer.attention
        put(f"L{li}.rec", dn.rec_state, 0)  # [B, Nv, Dk, Dv] per device
        for m, c in enumerate(dn.conv_states):
            put(f"L{li}.conv{m}", c, 1)  # [1, B, C] per device
        if getattr(dn, "conv_hist_packed", None) is not None:
            put(f"L{li}.hist", dn.conv_hist_packed, 0)  # [B, Nv, 4, 32, 32] per device
            out[f"L{li}.hist_valid"] = str(bool(dn._hist_packed_valid))
        li += 1
    return out


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576, "trace_region_size": 1073741824}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param(tuple(int(x) for x in _MESH.split("x")), id=_MESH)],
    indirect=True,
)
def test_gdn_slot_write_fast_tp2(mesh_device, reset_seeds, ensure_gc):
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    B = int(os.environ.get("SLOT_FAST_B", "32"))
    n_layers = int(os.environ.get("SLOT_FAST_LAYERS", "16"))
    out_path = os.environ.get("SLOT_FAST_OUT", "slot_fast.json")
    knob = os.environ.get("QWEN36_PLAIN_GDN_SLOT_FAST", "1")
    num_blocks = B * BPU
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=8192, n_layers=n_layers)
    assert model.use_tp
    args = model.args
    model.allocate_kv_caches(
        (num_blocks + 1, args.n_local_kv_heads, BLOCK, args.head_dim), ttnn.bfloat8_b, batch_size=B
    )
    warm_pt = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    n_decode = int(os.environ.get("SLOT_FAST_DECODE", "0"))
    pt_full = torch.stack([torch.arange(s * BPU, (s + 1) * BPU, dtype=torch.int32) for s in range(B)])
    dec_traces = {}
    if n_decode:
        from models.tt_transformers.tt.common import copy_host_to_device

        model.sync_gdn_decode_state()
        widths = sorted({min(8, B), B})
        # Compile EVERY width before capturing any trace (generator_interface.warmup_decode_buckets: a compile after a
        # trace is parked clobbers it -- compiling width B after parking width 8 hung the first width-B replay).
        for w in widths:
            tokens = torch.full((w, 1), 100, dtype=torch.int32)
            pos = torch.full((w,), -1, dtype=torch.int32)
            dev0 = model.prepare_inputs_decode(tokens, pos, page_table=pt_full[:w])
            model.ttnn_decode_forward(dev0[0], dev0[1], rot_mat_idxs=dev0[2], page_table=dev0[3])  # compile
        ttnn.synchronize_device(mesh_device)
    live_pos = {}  # slot -> next decode position
    # model_runner phase 2 order: chunk-prefill trace + slot-write warmup first, decode traces after (a buffer allocated
    # after a parked trace may alias that trace's intermediates and be clobbered by its replays).
    prev = model._bind_gdn_prefill_scratch()
    try:
        model.capture_prefill_trace_chunked(mesh_device, warm_pt, chunk_size=2048)
    finally:
        model._unbind_gdn_prefill_scratch(prev)
    model.warmup_gdn_slot_write()
    ttnn.synchronize_device(mesh_device)
    if n_decode:
        model.sync_gdn_decode_state()  # the prefill capture invalidated the packed history; rebuilding it reads the host
        for w in widths:
            tokens = torch.full((w, 1), 100, dtype=torch.int32)
            pos = torch.full((w,), -1, dtype=torch.int32)
            host = model.prepare_decode_inputs_host(tokens, pos, page_table=pt_full[:w])
            dev = copy_host_to_device(host, mesh_device=mesh_device)
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            dec_traces[w] = (tid, dev, out)
        logger.info(f"[slot_fast] decode traces captured at widths {sorted(dec_traces)}")
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    g = torch.Generator().manual_seed(1234)
    repeat = int(os.environ.get("SLOT_FAST_REPEAT", "1"))  # >1: re-run each admission (warm timing; same final state)
    acc = {}
    if os.environ.get("SLOT_FAST_PROFILE", "0") == "1":
        # Host wall accumulators around the slot-write pieces (no device sync inside: dispatch + blocking reads).
        import models.demos.blackhole.qwen36.tt.gdn.tp as gdn_tp

        def _wrap(owner, name, key):
            orig = getattr(owner, name)

            def _w(*a, **kw):
                t = time.perf_counter()
                try:
                    return orig(*a, **kw)
                finally:
                    acc[key] = acc.get(key, 0.0) + time.perf_counter() - t

            setattr(owner, name, _w)

        cls = gdn_tp.TPGatedDeltaNet
        for name in ("write_slot", "_write_index", "_sync_conv_hist_packed", "_packed_slot_tensor", "sync_conv_taps"):
            _wrap(cls, name, name)
        _wrap(ttnn, "from_torch", "ttnn.from_torch")
        _wrap(ttnn, "to_torch", "ttnn.to_torch")
        _wrap(ttnn, "fill_cache", "ttnn.fill_cache")
        _wrap(model, "_write_gdn_slot", "_write_gdn_slot")
    steps = []
    for slots, lens in ADMISSIONS:
        toks = [torch.randint(1000, 100000, (1, T), generator=g, dtype=torch.int64).to(torch.int32) for T in lens]
        rows = torch.zeros(len(slots), 1024, dtype=torch.int32)
        for i, (s, T) in enumerate(zip(slots, lens)):
            nblk = -(-T // BLOCK)
            rows[i, :nblk] = torch.arange(s * BPU, s * BPU + nblk, dtype=torch.int32)
        for _ in range(repeat):
            ttnn.synchronize_device(mesh_device)
            acc.clear()
            t0 = time.perf_counter()
            logits = model.prefill_paged_slots(toks, rows, slots, valid_lens=lens)
            ttnn.synchronize_device(mesh_device)
            dt = time.perf_counter() - t0
        if acc:
            logger.info(
                f"[slot_fast knob={knob}] slots={slots} pieces: "
                + " ".join(f"{k}={v * 1e3:.1f}" for k, v in sorted(acc.items(), key=lambda kv: -kv[1]))
            )
        rec = {
            "slots": slots,
            "lens": lens,
            "wall_ms": round(dt * 1e3, 1),
            "logits": [_digest(lg.float()) for lg in logits],
            "state": _state_digests(model, comp),
        }
        for s, T in zip(slots, lens):
            live_pos[s] = T
        rec["state_live"] = _state_digests(model, comp, rows=sorted(live_pos))
        if n_decode:
            w = min(8, B) if max(live_pos) < min(8, B) else B
            tid, dev, out = dec_traces[w]
            dec_logits = []
            for _ in range(n_decode):
                tokens = torch.full((w, 1), 100, dtype=torch.int32)
                pos = torch.full((w,), -1, dtype=torch.int32)
                for s, p in live_pos.items():
                    tokens[s, 0] = int(torch.randint(1000, 100000, (1,), generator=g))
                    pos[s] = p
                host = model.prepare_decode_inputs_host(tokens, pos, page_table=pt_full[:w])
                copy_host_to_device(host, device_tensors=dev)
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
                lg = model.process_output_decode(out, w)[:, 0, : model.vocab_size].float()
                dec_logits.append({str(s): _digest(lg[s].contiguous()) for s in sorted(live_pos)})
                for s in live_pos:
                    live_pos[s] += 1
            ttnn.synchronize_device(mesh_device)
            rec["decode_width"] = w
            rec["decode_logits"] = dec_logits
            rec["state_after_decode"] = _state_digests(model, comp)
            rec["state_after_decode_live"] = _state_digests(model, comp, rows=sorted(live_pos))
        logger.info(f"[slot_fast knob={knob}] slots={slots} lens={lens} prefill_paged_slots {dt * 1e3:.1f} ms")
        steps.append(rec)
    for tid, _, _ in dec_traces.values():
        ttnn.release_trace(mesh_device, tid)
    with open(out_path, "w") as f:
        json.dump({"knob": knob, "B": B, "n_layers": n_layers, "n_decode": n_decode, "steps": steps}, f, indent=1)
    logger.info(f"[slot_fast] wrote {out_path}")


def _compare(a_path, b_path):
    """Gate = logits + LIVE-row state (state_live / state_after_decode_live) when both files have them, else whole
    buffers (the pre-decode-mode files). Whole-buffer differences are always printed (inactive rows: information)."""
    a, b = json.load(open(a_path)), json.load(open(b_path))
    bad = 0
    live = all("state_live" in st for st in a["steps"] + b["steps"])
    sk, dk = ("state_live", "state_after_decode_live") if live else ("state", "state_after_decode")
    for i, (sa, sb) in enumerate(zip(a["steps"], b["steps"])):
        diff = [k for k in sa[sk] if sa[sk][k] != sb[sk].get(k)]
        ld = sa["logits"] != sb["logits"]
        bad += len(diff) + int(ld)
        if live:
            whole = sum(sa["state"][k] != sb["state"].get(k) for k in sa["state"])
            whole += sum(
                sa["state_after_decode"][k] != sb["state_after_decode"].get(k) for k in sa.get("state_after_decode", {})
            )
            print(
                f"step {i}: live rows {sorted({int(k.split('s')[-1][:-1]) for k in sa[sk] if '[' in k})}, "
                f"{len(sa[sk])} live-row digests; whole-buffer digests differing (incl. inactive rows): {whole}"
            )
        dec = ""
        if "decode_logits" in sa or "decode_logits" in sb:
            sd_a, sd_b = sa.get(dk, {}), sb.get(dk, {})
            ddiff = [k for k in sd_a if sd_a[k] != sd_b.get(k)] + (["<missing>"] if not sd_a or not sd_b else [])
            dl = [j for j, (x, y) in enumerate(zip(sa.get("decode_logits", []), sb.get("decode_logits", []))) if x != y]
            dl += ["<len>"] if len(sa.get("decode_logits", [])) != len(sb.get("decode_logits", [])) else []
            bad += len(ddiff) + len(dl)
            n_rows = sum(len(x) for x in sa.get("decode_logits", []))
            dec = (
                f"; +{len(sa.get('decode_logits', []))} decode steps @w{sa.get('decode_width')} ({n_rows} rows): "
                f"logits steps differ {dl if dl else 'none'}, state after decode {len(ddiff)} differ"
                f"{' ' + str(ddiff[:6]) if ddiff else ''}"
            )
        print(
            f"step {i} slots={sa['slots']} lens={sa['lens']}: {len(sa[sk])} {'live-row digests' if live else 'buffers'}, "
            f"{len(diff)} differ"
            f"{' ' + str(diff[:6]) if diff else ''}; logits {'DIFFER' if ld else 'equal'}{dec}; "
            f"wall {sa['wall_ms']} ms (knob {a['knob']}) vs {sb['wall_ms']} ms (knob {b['knob']})"
        )
    print("BIT-EXACT" if bad == 0 and len(a["steps"]) == len(b["steps"]) else f"MISMATCH ({bad})")
    return bad


if __name__ == "__main__":
    sys.exit(1 if _compare(sys.argv[1], sys.argv[2]) else 0)
