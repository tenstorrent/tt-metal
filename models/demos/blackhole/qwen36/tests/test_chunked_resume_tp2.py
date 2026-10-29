# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test (TP=2, (1,2) mesh): scheduler-driven chunked prefill == unchunked prefill (QWEN36_CHUNKED_PREFILL).

ONE process, ONE (1,2) mesh, the real 27B through prefill_paged_slots on the persistent B=1 GDN scratch (the batched
TP serving path). P1C_B sets max_batch_size (default 8; the tp2 servers run max_num_seqs=32, so P1C_B=32 is the served
decode-buffer shape) and P1C_LONG_SLOT the long prompt's decode slot (default 0; e.g. 13 = a high, odd slot, which
exercises the slot index and the parity-tagged packed conv history). Riders take the highest other slots (slots 1..7 at
the default B=8 / slot 0). For each prompt length T:

  reference : the prompt prefilled in ONE prefill_paged_slots call into the long slot (R_REF times: the envelope);
              every rider prompt is also prefilled alone once (its reference logits).
  chunked   : the SAME prompt prefilled in 2048-token calls (intermediate chunks final_mask=False, continuations
              resume_mask=True at the chunk start), with other prompts ("riders", own slots and KV blocks) interleaved in
              the two ways the plugin produces:
                * a call with ONLY a rider between two chunks (review B1: the partial must be parked before the rider
                  resets the scratch and unparked before its next chunk), and
                * a rider sharing the call with a chunk (first in call order; the continuation runs first).
              Repeated N_SEQ times back to back (review M5: several sequential chunked requests with riders, since the
              earlier device-side slot copies drifted only on later requests).

Compared per chunked request: the long prompt's last-position logits, its whole GDN decode-slot state (every layer's
recurrent row + conv taps + its packed fused-conv history row when that buffer is live), and every rider's logits against its reference. P1C_MODE=eager (QWEN36_PREFILL_BUCKET_TRACE=0
and no chunk trace: the eager TP path) must be BIT-EXACT; P1C_MODE=traced (the served path: chunk trace + traced masked
bucket) is judged against the reference-vs-reference envelope (review M5 of the state critique: traced TP=2 prefill was
run-to-run nondeterministic before the AGMM out-AG barrier). The program cache must not grow after the first chunked
request (traced mode; eager compiles on demand).

Env: P1C_MODE eager|traced (default traced), P1C_LENS (default 2049,4096,6444,32768), P1C_NSEQ (3), P1C_RREF (3),
P1C_B (8), P1C_LONG_SLOT (0),
P1C_LAYERS (all), P1C_OUT (JSON), QWEN36_CP_PARK_HOST (1 = host park fallback).

Run (chips 0,3): source profiles/opt_round4/laneC_env.sh; cd $TT_METAL_HOME; P1C_MODE=eager pytest -svq \
    models/demos/blackhole/qwen36/tests/test_chunked_resume_tp2.py
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

BLOCK = 64
C = 2048
LONG_BLOCKS = 520  # region of the long prompt (33k tokens)
RIDER_BLOCKS = 40  # 2560 tokens per rider region
N_RIDER_SLOTS = 7  # the 7 highest slots other than the long prompt's (1..7 at B=8, long slot 0)
RIDER_LENS = [512, 90, 1500, 2047, 300, 1024, 64]


def _ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 150000, (1, n), generator=g, dtype=torch.long)


def _row(blocks, width):
    r = torch.zeros(1, width, dtype=torch.int32)
    r[0, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return r


def _slot_state(model, slot):
    """Host copy of decode slot `slot`'s GDN state (both replicas): recurrent rows + conv taps of every GDN layer."""
    comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
    out = []
    for layer in model.layers:
        if layer.is_full_attention:
            continue
        dn = layer.attention
        B = dn.rec_state.shape[0]
        rec = ttnn.to_torch(dn.rec_state, mesh_composer=comp).reshape(model.num_devices, B, -1)[:, slot].clone()
        convs = [
            ttnn.to_torch(c, mesh_composer=comp).reshape(model.num_devices, -1, c.shape[-1])[:, slot].clone()
            for c in dn.conv_states
        ]
        packed = getattr(dn, "conv_hist_packed", None)
        if packed is not None and getattr(dn, "_hist_packed_valid", False):
            convs.append(ttnn.to_torch(packed, mesh_composer=comp).reshape(model.num_devices, B, -1)[:, slot].clone())
        out.append((rec, convs))
    return out


def _state_diff(a, b):
    d = 0.0
    for (ra, ca), (rb, cb) in zip(a, b):
        d = max(d, float((ra.float() - rb.float()).abs().max()))
        for x, y in zip(ca, cb):
            d = max(d, float((x.float() - y.float()).abs().max()))
    return d


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576, "trace_region_size": 1073741824}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2")], indirect=True)
def test_chunked_resume_tp2(mesh_device, reset_seeds, ensure_gc):
    mode = os.environ.get("P1C_MODE", "traced")
    assert mode in ("eager", "traced"), mode
    if mode == "eager":
        assert (
            os.environ.get("QWEN36_PREFILL_BUCKET_TRACE", "1") == "0"
        ), "eager mode needs QWEN36_PREFILL_BUCKET_TRACE=0"
    lens = [int(x) for x in os.environ.get("P1C_LENS", "2049,4096,6444,32768").split(",")]
    n_seq = int(os.environ.get("P1C_NSEQ", "3"))
    r_ref = int(os.environ.get("P1C_RREF", "3"))
    n_layers = int(os.environ["P1C_LAYERS"]) if os.environ.get("P1C_LAYERS") else None
    out_path = os.environ.get("P1C_OUT", os.path.join(os.getcwd(), f"p1c_chunked_resume_{mode}.json"))
    assert mesh_device.get_num_devices() == 2
    B = int(os.environ.get("P1C_B", "8"))
    long_slot = int(os.environ.get("P1C_LONG_SLOT", "0"))
    assert 0 <= long_slot < B and B - 1 >= N_RIDER_SLOTS, (B, long_slot)
    rider_slots = [s for s in range(B - 1, -1, -1) if s != long_slot][:N_RIDER_SLOTS][::-1]
    num_blocks = 1024  # multiple of 32 (chunk page-table width); +1 below = the pad block, never in a row
    assert LONG_BLOCKS + N_RIDER_SLOTS * RIDER_BLOCKS <= num_blocks
    assert max(lens) <= LONG_BLOCKS * BLOCK
    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=65536, n_layers=n_layers)
    args = model.args
    vocab = args.vocab_size
    model.allocate_kv_caches(
        (num_blocks + 1, args.n_local_kv_heads, BLOCK, args.head_dim), ttnn.bfloat8_b, batch_size=B
    )
    logger.info(
        f"[p1c] model loaded in {time.perf_counter() - t0:.0f}s mode={mode} lens={lens} B={B} long_slot={long_slot} "
        f"rider_slots={rider_slots}"
    )

    warm_pt = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    prev = model._bind_gdn_prefill_scratch()
    try:
        model.ensure_gdn_park_buffer()
        model.capture_prefill_trace_chunked(mesh_device, warm_pt, chunk_size=C, capture_chunk_trace=(mode == "traced"))
    finally:
        model._unbind_gdn_prefill_scratch(prev)
    model.warmup_gdn_slot_write()
    ttnn.synchronize_device(mesh_device)
    logger.info(
        f"[p1c] warmup done: chunk_trace={model._chunked_trace_id} mb_traces={sorted(model._mb_traces)} "
        f"programs={mesh_device.num_program_cache_entries()}"
    )

    long_row = _row(list(range(LONG_BLOCKS)), num_blocks)
    rider_rows = [
        _row(list(range(LONG_BLOCKS + i * RIDER_BLOCKS, LONG_BLOCKS + (i + 1) * RIDER_BLOCKS)), num_blocks)
        for i in range(N_RIDER_SLOTS)
    ]
    rider_ids = [_ids(n, 100 + i) for i, n in enumerate(RIDER_LENS)]

    def call(rows):
        """rows: list of (ids, row, slot, valid_len, start, resume, final) -> host logits list (call order)."""
        return model.prefill_paged_slots(
            [r[0] for r in rows],
            torch.cat([r[1] for r in rows], dim=0),
            [r[2] for r in rows],
            valid_lens=[r[3] for r in rows],
            start_positions=[r[4] for r in rows],
            resume_mask=[r[5] for r in rows],
            final_mask=[r[6] for r in rows],
        )

    # Rider references (alone, unchunked), in their own slots/regions.
    rider_ref = []
    for i, ids in enumerate(rider_ids):
        hl = call([(ids, rider_rows[i], rider_slots[i], ids.shape[1], 0, False, True)])
        rider_ref.append(hl[0].reshape(-1)[:vocab].clone())

    results = {}
    n_pc_first = None
    ok = True
    for T in lens:
        ids = _ids(T, T)
        # ---- references (unchunked, slot 0) ----
        refs, ref_states = [], []
        for _ in range(r_ref):
            hl = call([(ids, long_row, long_slot, T, 0, False, True)])
            refs.append(hl[0].reshape(-1)[:vocab].clone())
            ref_states.append(_slot_state(model, long_slot))
        env_logit = max(float((r - refs[0]).abs().max()) for r in refs)
        env_state = max(_state_diff(s, ref_states[0]) for s in ref_states)
        # ---- chunked sequences ----
        seq_recs = []
        for k in range(n_seq):
            ends = list(range(C, T, C)) + [T]
            rider_i = 0
            rider_out = []  # (rider index, logits)

            def next_rider():
                nonlocal rider_i
                i = (rider_i + k) % N_RIDER_SLOTS
                rider_i += 1
                ids_r = rider_ids[i]
                return i, (ids_r, rider_rows[i], rider_slots[i], ids_r.shape[1], 0, False, True)

            t1 = time.perf_counter()
            prev_end = 0
            lg_final = None
            for ci, end in enumerate(ends):
                final = end == T
                long_r = (ids, long_row, long_slot, end, prev_end, prev_end > 0, final)
                if ci % 2 == 1 and not final:
                    # a call with ONLY a rider between two chunks (B1), then the continuation alone
                    i, rr = next_rider()
                    rider_out.append((i, call([rr])[0]))
                    hl = call([long_r])
                    lg = hl[0]
                else:
                    # a rider sharing the call with the chunk, first in call order
                    i, rr = next_rider()
                    hl = call([rr, long_r])
                    rider_out.append((i, hl[0]))
                    lg = hl[1]
                if final:
                    lg_final = lg.reshape(-1)[:vocab].clone()
                else:
                    assert torch.count_nonzero(lg) == 0, "an intermediate chunk must return a zero logits row"
                prev_end = end
            ttnn.synchronize_device(mesh_device)
            dt = time.perf_counter() - t1
            st = _slot_state(model, long_slot)
            if n_pc_first is None:
                n_pc_first = mesh_device.num_program_cache_entries()
            d_logit = float((lg_final - refs[0]).abs().max())
            d_state = _state_diff(st, ref_states[0])
            top1 = int(torch.argmax(lg_final)) == int(torch.argmax(refs[0]))
            d_rider = max(float((l.reshape(-1)[:vocab] - rider_ref[i]).abs().max()) for i, l in rider_out)
            rec = {
                "seq": k,
                "calls": len(ends) + sum(1 for ci, e in enumerate(ends) if ci % 2 == 1 and e != T),
                "riders": len(rider_out),
                "logits_equal": bool(torch.equal(lg_final, refs[0])),
                "max_dlogit": d_logit,
                "state_equal": d_state == 0.0,
                "max_dstate": d_state,
                "top1_agree": top1,
                "rider_max_dlogit": d_rider,
                "wall_s": round(dt, 2),
                "programs": mesh_device.num_program_cache_entries(),
            }
            seq_recs.append(rec)
            logger.info(f"[p1c] T={T} seq {k}: {rec} (ref envelope logit={env_logit:.4g} state={env_state:.4g})")
            if mode == "eager":
                ok &= rec["logits_equal"] and rec["state_equal"] and d_rider == 0.0
            else:
                ok &= d_logit <= env_logit and d_state <= env_state and top1 and d_rider <= env_logit
        results[str(T)] = {
            "ref_envelope_logit": env_logit,
            "ref_envelope_state": env_state,
            "ref_top2": [int(i) for i in torch.topk(refs[0], 2).indices],
            "ref_top2_gap": float(torch.topk(refs[0], 2).values[0] - torch.topk(refs[0], 2).values[1]),
            "chunked": seq_recs,
        }
    grew = mesh_device.num_program_cache_entries() - (n_pc_first or 0)
    summary = {
        "mode": mode,
        "park": "host" if os.environ.get("QWEN36_CP_PARK_HOST", "0") == "1" else "device",
        "n_layers": n_layers,
        "max_batch_size": B,
        "long_slot": long_slot,
        "rider_slots": rider_slots,
        "program_cache_growth_after_first_chunked_request": grew,
        # Eager mode compiles on demand (no parked trace to clobber), so growth only gates the traced mode.
        "pass": bool(ok and (grew == 0 or mode == "eager")),
        "results": results,
    }
    json.dump(summary, open(out_path, "w"), indent=1)
    logger.info(f"[p1c] wrote {out_path}: pass={summary['pass']} program growth={grew}")
    if mode == "traced":
        assert grew == 0, f"program cache grew by {grew} after the first chunked request"
    assert ok, f"chunked prefill differs from the unchunked reference (mode={mode}); see {out_path}"
