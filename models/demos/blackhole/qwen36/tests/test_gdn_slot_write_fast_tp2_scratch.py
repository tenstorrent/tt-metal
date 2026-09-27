# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH bit-exactness + timing check of the plain served slot write (prefill_paged_slots) on a (1,2) mesh.

QWEN36_PLAIN_GDN_SLOT_FAST=0 is the host round trip (snapshot the B=1 scratch GDN state to host, re-upload it and
write it into the decode row); =1 (default) writes the fp32 recurrent state with ttnn.fill_cache on device and feeds
the packed-history repack from the tap snapshot. Run the SAME admission sequence once per knob value (one process
each; the traced prefill is deterministic) and compare the per-admission digests of every batched GDN buffer
(rec_state, conv_states[0..3], conv_hist_packed: all B rows, both devices) and of the logits row.

    SLOT_FAST_OUT=<json> QWEN36_PLAIN_GDN_SLOT_FAST=0|1 pytest -svq \\
        models/demos/blackhole/qwen36/tests/test_gdn_slot_write_fast_tp2_scratch.py
    python models/demos/blackhole/qwen36/tests/test_gdn_slot_write_fast_tp2_scratch.py a.json b.json   # compare

Env: SLOT_FAST_LAYERS (default 16 = 12 GDN + 4 attention layers), SLOT_FAST_B (default 32 = the tp2 max_num_seqs).
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


def _state_digests(model, comp):
    out = {}
    li = 0
    for layer in model.layers:
        if layer.is_full_attention:
            continue
        dn = layer.attention
        out[f"L{li}.rec"] = _digest(ttnn.to_torch(dn.rec_state, mesh_composer=comp))
        for m, c in enumerate(dn.conv_states):
            out[f"L{li}.conv{m}"] = _digest(ttnn.to_torch(c, mesh_composer=comp))
        if getattr(dn, "conv_hist_packed", None) is not None:
            out[f"L{li}.hist"] = _digest(ttnn.to_torch(dn.conv_hist_packed, mesh_composer=comp))
            out[f"L{li}.hist_valid"] = str(bool(dn._hist_packed_valid))
        li += 1
    return out


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576, "trace_region_size": 1073741824}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2")], indirect=True)
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
    prev = model._bind_gdn_prefill_scratch()
    try:
        model.capture_prefill_trace_chunked(mesh_device, warm_pt, chunk_size=2048)
    finally:
        model._unbind_gdn_prefill_scratch(prev)
    model.warmup_gdn_slot_write()
    ttnn.synchronize_device(mesh_device)
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
        logger.info(f"[slot_fast knob={knob}] slots={slots} lens={lens} prefill_paged_slots {dt * 1e3:.1f} ms")
        steps.append(rec)
    with open(out_path, "w") as f:
        json.dump({"knob": knob, "B": B, "n_layers": n_layers, "steps": steps}, f, indent=1)
    logger.info(f"[slot_fast] wrote {out_path}")


def _compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    bad = 0
    for i, (sa, sb) in enumerate(zip(a["steps"], b["steps"])):
        diff = [k for k in sa["state"] if sa["state"][k] != sb["state"].get(k)]
        ld = sa["logits"] != sb["logits"]
        bad += len(diff) + int(ld)
        print(
            f"step {i} slots={sa['slots']} lens={sa['lens']}: {len(sa['state'])} buffers, {len(diff)} differ"
            f"{' ' + str(diff[:6]) if diff else ''}; logits {'DIFFER' if ld else 'equal'}; "
            f"wall {sa['wall_ms']} ms (knob {a['knob']}) vs {sb['wall_ms']} ms (knob {b['knob']})"
        )
    print("BIT-EXACT" if bad == 0 and len(a["steps"]) == len(b["steps"]) else f"MISMATCH ({bad})")
    return bad


if __name__ == "__main__":
    sys.exit(1 if _compare(sys.argv[1], sys.argv[2]) else 0)
