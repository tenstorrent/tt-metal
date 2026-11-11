# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drive the PUP d-gen prefill adapter the way the common prefill runner does, on one (1, 4) column, and check the
published state.

Runner steps reproduced: adapter -> ``allocate_kv_cache`` -> ``build_runtime`` -> ``compile`` -> ``build_kv_chunk_table``
(serialized to a file, as ``PREFILL_MIGRATION_EXPORT_TO_FILE=1``) -> per request ``make_chunk_input`` +
``prefill_chunk(slot, actual_start=0, actual_end=L, media_refs=...)`` with the per-chunk ``MediaRef``s the engine would
send (#2467 / #63999 layout) and the encoder's stored values in an in-process embedding store. Not reproduced: the H2D
socket (the runtime's tokens come from ``make_chunk_input``), the layer-ack channel (a recording sink), the KV manager.

Checks per request (``--ref DIR``: today's prefill state dumps, ``<name>.pt`` HF layout + ``<name>_vision.pt`` the TT
vision rows that prefill spliced):
  export   the published caches (engine-owned, contract form) decode back to exactly the model's working state;
  ref      the published state vs the reference dump: K / V (bf8 as cached), GDN recurrent (fp32), conv (bf16):
           exact-equal fraction, max abs diff, pcc per layer; first token vs the reference's.
Timing: media (store read + upload) / prefill / export per request, synced phases, median of --reps.

Usage (one (1, 4) column, TT_VISIBLE_DEVICES = its 4 chips):
  python pup_dgen_harness.py --gold DIR --ref DIR --names t2i_0,i2i1_0,i2i10 --out results.json
"""

import argparse
import hashlib
import json
import os
import time
import types

import numpy as np
import torch
from loguru import logger


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    if a.numel() == 0:
        return 1.0
    if torch.equal(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _cmp(got, ref):
    got, ref = got.float(), ref.float()
    return dict(
        equal=float((got == ref).float().mean()),
        max_abs=float((got - ref).abs().max()) if got.numel() else 0.0,
        pcc=_pcc(got, ref),
    )


class _Sink:
    def __init__(self):
        self.records = []

    def layers_completed(self, layer_start, layer_end, slot_id, actual_start, actual_end):
        self.records.append((layer_start, layer_end, slot_id, actual_start, actual_end))


def _image_spans(ids, token, grids, merge=2):
    """[start, end) of each image's run of image tokens, in prompt order, sized by its grid."""
    spans, i, k = [], 0, 0
    ids = list(ids)
    while i < len(ids):
        if ids[i] == token:
            n = int(np.prod(grids[k])) // (merge * merge)
            assert ids[i : i + n] == [token] * n, f"image {k} run at {i}"
            spans.append((i, i + n))
            i += n
            k += 1
        else:
            i += 1
    assert k == len(grids), f"{k} image runs for {len(grids)} grids"
    return spans


def published_state(rt, kv, slot, L):
    """Decode the engine-owned caches (contract form) back to the HF layout of the reference dump."""
    import ttnn

    from models.demos.blackhole.qwen36.tt.runners import gdn_state as gs

    geo = kv.gdn.geometry
    out = {}
    dev = lambda t: [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]  # noqa: E731
    k_dev, v_dev = dev(kv.k), dev(kv.v)
    for pos, li in enumerate(kv.attn_layer_ids):
        b = slot * kv.num_attn_layers + pos
        out[li] = {
            "keys": torch.stack([x[b, 0, :L] for x in k_dev]).to(torch.bfloat16),
            "values": torch.stack([x[b, 0, :L] for x in v_dev]).to(torch.bfloat16),
        }
    rec_dev, conv_dev = dev(kv.gdn.recurrent), dev(kv.gdn.convolution)
    for pos, li in enumerate(kv.gdn_layer_ids):
        b = slot * kv.gdn.num_layers + pos
        rsegs, csegs = {}, {}
        for c in range(geo.tensor_parallel_size):
            rec = rec_dev[c][b].float()  # [8, D, D]
            for seg, _ in geo.local_segments("gdn_recurrent", c):
                _, h, band = geo.decompose_recurrent(seg)
                rsegs[seg] = rec[h][:, band * 32 : (band + 1) * 32]
            rows = conv_dev[c][b].to(torch.bfloat16)  # [32, 192]
            for seg, shard in geo.local_segments("gdn_convolution", c):
                csegs[seg] = rows[shard].reshape(geo.conv_history, gs.CONVOLUTION_PAGE_CHANNELS)
        out[li] = {
            "recurrent_state": gs.assemble_recurrent(rsegs, geo),
            "conv_state": gs.hf_conv_state(gs.assemble_convolution(csegs, geo)),
        }
    return out


def working_state(rt, slot, L):
    """The model's own state after the prefill (what the export copies), in the same HF layout."""
    import ttnn

    from models.demos.blackhole.qwen36.tt.runners import gdn_state as gs

    m = rt.model
    out = {}
    b0 = slot * rt.blocks_per_slot
    nb = -(-L // rt.block)
    for li in rt.attn_layer_ids:
        att = m.layers[li].attention
        kv = []
        for c in (att.paged_k, att.paged_v):
            per = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(c)]
            kv.append(torch.stack([x[b0 : b0 + nb, 0].reshape(-1, x.shape[-1])[:L] for x in per]).to(torch.bfloat16))
        out[li] = {"keys": kv[0], "values": kv[1]}
    for li in rt.gdn_layer_ids:
        dn = m.layers[li].attention
        rec = torch.cat([ttnn.to_torch(x)[0] for x in ttnn.get_device_tensors(dn.rec_state)], 0).float()
        cols = []
        for mi in range(dn.K):
            per = [ttnn.to_torch(x).reshape(-1) for x in ttnn.get_device_tensors(dn.conv_states[mi])]
            kd = dn.Nk * dn.Dk
            cols.append(torch.cat([torch.cat([p[a:b] for p in per]) for a, b in ((0, kd), (kd, 2 * kd), (2 * kd, per[0].shape[0]))]))
        conv = torch.stack(cols, 1).to(torch.bfloat16)
        conv[:, 0] = 0  # the shifted-out input is not migrated
        out[li] = {"recurrent_state": rec, "conv_state": conv}
    return out


def compare(a, b, label):
    res = {}
    for li in sorted(a):
        for key in a[li]:
            ga, gb = a[li][key], b[li][key]
            if key == "conv_state":
                ga, gb = ga[:, 1:], gb[:, 1:]
            if ga.shape != gb.shape:
                res[f"{li}.{key}"] = {"shape": [list(ga.shape), list(gb.shape)]}
                continue
            res[f"{li}.{key}"] = _cmp(ga, gb)
    worst = {}
    for k, v in res.items():
        kind = k.split(".", 1)[1]
        if "pcc" not in v:
            worst[kind] = {"shape_mismatch": v}
            continue
        w = worst.setdefault(kind, dict(equal=1.0, max_abs=0.0, pcc=1.0))
        w["equal"] = min(w["equal"], v["equal"])
        w["max_abs"] = max(w["max_abs"], v["max_abs"])
        w["pcc"] = min(w["pcc"], v["pcc"])
    logger.info(f"CMP {label}: " + "; ".join(f"{k} {v}" for k, v in worst.items()))
    return {"per_layer": res, "worst": worst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--names", default="t2i_0,i2i1_0,i2i10")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default="")
    ap.add_argument("--wait-ref", type=float, default=2400.0)
    ap.add_argument("--segment", type=int, default=0, help="override PupPrefillConfig.segment_tokens")
    a = ap.parse_args()

    import ttnn

    from models.demos.blackhole.qwen36.tt.runners import media as pm
    from models.demos.blackhole.qwen36.tt.runners.adapter import Qwen35PupPrefillAdapter
    from models.demos.blackhole.qwen36.tt.tp_common import fabric_router_config
    from models.demos.common.prefill.adapter import PrefillRunParams

    router = fabric_router_config()
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        **({"router_config": router} if router is not None else {}),
    )
    reader = pm.DictEmbeddingReader()
    adapter = Qwen35PupPrefillAdapter(media_reader=reader)
    if a.segment:
        from dataclasses import replace

        adapter.pup_config = replace(adapter.pup_config, segment_tokens=a.segment)
    results_cfg = vars(adapter.pup_config)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=adapter.l1_small_size,
        num_command_queues=2,
        trace_region_size=adapter.trace_region_size,
    )
    results = {"names": {}, "setup": {}, "config": {k: str(v) for k, v in results_cfg.items()}}
    try:
        t0 = time.time()
        hf_config = adapter.load_hf_config()
        cfg = adapter.pup_config
        params = PrefillRunParams(
            mesh_shape=(1, 4),
            num_layers=32,
            first_layer_idx=0,
            is_first_rank=True,
            is_last_rank=True,
            max_seq_len=cfg.max_seq_len,
            chunk_size=cfg.chunk_size,
            num_users=a.slots,
            capacity_factor=1,
            num_links=1,
            gate_mode_name="DEVICE_FP32",
            kv_only_last_layer=False,
            weight_cache_path=None,
        )
        kv = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf_config, params=params)
        rt = adapter.build_runtime(mesh_device=mesh, hf_config=hf_config, params=params)
        sink = _Sink()
        rt.set_layer_completion_sink(sink)
        t1 = time.time()
        rt.compile(kv)
        t2 = time.time()
        results["setup"] = dict(build_s=t1 - t0, compile_s=t2 - t1)
        if a.table:
            rt.build_kv_chunk_table(kv, a.table)
            results["setup"]["table"] = a.table
        logger.info(f"HARNESS build {t1 - t0:.1f}s compile {t2 - t1:.1f}s")

        image_token = rt.image_token
        for slot, name in enumerate(n for n in a.names.split(",") if n):
            slot %= a.slots
            gold = torch.load(os.path.join(a.gold, f"{name}.pt"), weights_only=False)
            need = [os.path.join(a.ref, f"{name}.pt")]
            if gold.get("grid") is not None:
                need.append(os.path.join(a.ref, f"{name}_vision.pt"))
            deadline = time.time() + a.wait_ref
            while not all(os.path.exists(f) for f in need) and time.time() < deadline:
                time.sleep(10)  # the reference prefill (another box) is still dumping
            if not all(os.path.exists(f) for f in need):
                logger.warning(f"SKIP {name}: reference files missing {need}")
                continue
            time.sleep(2)
            ref = torch.load(need[0], weights_only=False)
            ids = gold["ids"][0].tolist()
            L = len(ids)
            refs = []
            if gold.get("grid") is not None:
                vis = torch.load(os.path.join(a.ref, f"{name}_vision.pt"), weights_only=False)
                grids = [tuple(int(v) for v in g) for g in vis["grid"]]
                spans = _image_spans(ids, image_token, grids)
                rows = vis["rows"].to(torch.bfloat16)
                off = 0
                for (s, e), g in zip(spans, grids):
                    r = rows[off : off + (e - s)].view(torch.int16).numpy().view(np.uint16)
                    off += e - s
                    key = hashlib.sha256(r.tobytes()).digest()
                    reader.put(key, pm.stored_value(r, g, n_refs=len(grids)))
                    # tt-d-gen PrefillMediaRef (#2467 / tt-metal #63999): chunk_index, start_pos, end_pos, media_key
                    refs.append(types.SimpleNamespace(chunk_index=0, start_pos=s, end_pos=e, media_key=key))
                assert off == rows.shape[0], (off, rows.shape)
            timings = []
            for rep in range(a.reps + 1):
                rt.profile = True
                inp = rt.make_chunk_input(ids)
                rt.prefill_chunk(inp, kv, slot_id=slot, actual_start=0, actual_end=L, request_id=rep, media_refs=refs)
                ttnn.synchronize_device(mesh)
                if rep:
                    timings.append(dict(rt.last_timing))
            # unsynced end-to-end (the runner's mode)
            walls = []
            for rep in range(a.reps):
                rt.profile = False
                ttnn.synchronize_device(mesh)
                w0 = time.perf_counter()
                inp = rt.make_chunk_input(ids)
                rt.prefill_chunk(inp, kv, slot_id=slot, actual_start=0, actual_end=L, request_id=rep, media_refs=refs)
                ttnn.synchronize_device(mesh)
                walls.append((time.perf_counter() - w0) * 1e3)
            first = int(rt.last_logits.reshape(-1).argmax())
            med = {k: float(np.median([t[k] for t in timings])) for k in ("media_ms", "prefill_ms", "export_ms", "total_ms")}
            med["wall_ms"] = float(np.median(walls))
            logger.info(f"TIME {name} L={L} images={len(refs)} slot={slot}: {med}")
            pub = published_state(rt, kv, slot, L)
            work = working_state(rt, slot, L)
            torch.save({"published": pub, "working": work, "first_token": first, "L": L},
                       os.path.join(os.path.dirname(a.out), f"{name}_state.pt"))
            r_export = compare(pub, work, f"{name} published vs working")
            r_ref = compare(pub, ref["state"], f"{name} published vs reference")
            logger.info(f"TOKEN {name}: first {first} reference {ref['first_token']}")
            results["names"][name] = dict(
                L=L,
                images=len(refs),
                slot=slot,
                timing=med,
                timing_reps=timings,
                first_token=first,
                ref_first_token=int(ref["first_token"]),
                export_vs_working=r_export["worst"],
                published_vs_ref=r_ref["worst"],
                published_vs_ref_layers=r_ref["per_layer"],
                acks=sink.records[-1],
            )
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
        logger.info(f"HARNESS DONE -> {a.out}")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
