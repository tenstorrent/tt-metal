# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH bit-exactness + timing of the fast GDN slot remap (QWEN36_GDN_REMAP_FAST) on a (1,2) mesh.

test_remap_fast_buffers: TPGatedDeltaNet layer shells holding the tp2 decode-state buffers (B=32 rows; rec_state
[B, 24, 128, 128] fp32, 4 conv taps [1, B, 5120] bf16, packed conv history [B, 24, 4, 32, 32] bf16; per-device random
bits, incl. -0.0, subnormals, +-inf and NaN rows), identical copies A and B. Each remap runs the slice/concat path on A
and the fast path on B, then compares EVERY row of every buffer bit for bit (int views, so -0.0 vs 0.0 and NaN payloads
count). Negative controls: the fast path with the device parity swap disabled, and with a wrong remap, must MISMATCH.
REMAP_TIME_LAYERS > 0 also times both paths over that many layer shells (48 = the model's GDN layer count).

    pytest -svq models/demos/blackhole/qwen36/tests/test_gdn_remap_fast_tp2_scratch.py -k buffers
"""

import os
import random
import time

import pytest
import torch
from loguru import logger

import ttnn

B, NV, DK, DV, K, C = 32, 24, 128, 128, 4, 5120


def _rand_bits(shape, dtype, g, specials):
    """Random finite values plus (specials) -0.0 / subnormal / +-inf / NaN planted at fixed places."""
    t = torch.randn(*shape, generator=g, dtype=torch.float32).to(dtype)
    if specials:
        flat = t.view(-1)
        n = flat.numel()
        tiny = torch.finfo(dtype).tiny
        vals = [-0.0, tiny / 4, -tiny / 8, float("inf"), float("-inf"), float("nan")]
        per_slot = n // B  # plant every special in every slot row (the slot is dim 0, or dim 1 of the taps)
        for r in range(B):
            for j, v in enumerate(vals):
                flat[r * per_slot + (j * 7919 + 13 * r) % per_slot] = v
    return t


def _to_dev(mesh, per_dev, dtype):
    """per_dev: list (one per device) of torch tensors -> mesh tensor sharded so device d holds per_dev[d]."""
    return ttnn.from_torch(
        torch.cat(per_dev, dim=0),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _shell(mesh, host):
    """A TPGatedDeltaNet with only the decode-state fields remap_slots touches."""
    from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet

    dn = object.__new__(TPGatedDeltaNet)
    dn.mesh, dn.B, dn.K, dn.Nv, dn.Dk, dn.Dv, dn.qkv_dim_tp = mesh, B, K, NV, DK, DV, C
    dn._decode_fused_conv = True
    dn._conv_taps_stale = dn._conv_win_stale = False
    dn._conv_win_buf = None
    dn._cfg_onehot = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    dn.rec_state = _to_dev(mesh, host["rec"], ttnn.float32)
    # conv taps are [1, B, C] per device: shard a [n_dev, B, C] host tensor on dim 0
    dn.conv_states = [_to_dev(mesh, host[f"conv{m}"], ttnn.bfloat16) for m in range(K)]
    dn.conv_hist_packed = _to_dev(mesh, host["hist"], ttnn.bfloat16)
    dn._hist_packed_valid = True
    return dn


def _host_state(n_dev, seed, specials=True):
    g = torch.Generator().manual_seed(seed)
    st = {"rec": [_rand_bits((B, NV, DK, DV), torch.float32, g, specials) for _ in range(n_dev)]}
    for m in range(K):
        st[f"conv{m}"] = [_rand_bits((1, B, C), torch.bfloat16, g, specials) for _ in range(n_dev)]
    st["hist"] = [_rand_bits((B, NV, 4, 32, 32), torch.bfloat16, g, specials) for _ in range(n_dev)]
    return st


def _read(dn):
    out = {"rec": dn.rec_state, "hist": dn.conv_hist_packed}
    out.update({f"conv{m}": c for m, c in enumerate(dn.conv_states)})
    return {k: [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)] for k, t in out.items()}


def _bits(t):
    return t.contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int16)


def _compare(a, b):
    """Number of (buffer, device, row) triples whose bits differ; row = the slot index."""
    bad = []
    for k in a:
        dim = 1 if k.startswith("conv") else 0
        for d, (x, y) in enumerate(zip(a[k], b[k])):
            eq = (_bits(x) == _bits(y)).reshape(*x.shape).movedim(dim, 0).reshape(x.shape[dim], -1).all(dim=1)
            bad += [(k, d, int(s)) for s in torch.nonzero(~eq).flatten()]
    return bad


def _remaps():
    rnd = random.Random(7)
    perm = list(range(B))
    rnd.shuffle(perm)
    condense = list(range(B))  # request in slot 3 finished: the last live row (9) moves into slot 3
    condense[3], condense[9] = 9, 3
    shift = [(i + 1) % B for i in range(B)]  # every row moves, all cross parity
    plugin = [4, 0, 1, 2, 7, 5, 6, 3] + list(range(8, B))  # a 5-move mix of same- and cross-parity rows
    cycle = list(range(B))
    cycle[10], cycle[12], cycle[21] = 12, 21, 10  # a same-parity cycle and one cross-parity hop
    rev16 = list(range(15, -1, -1)) + list(range(16, B))  # 16 moved (the fill_cache limit), all cross parity
    return {"rev16": rev16, "condense": condense, "random": perm, "shift": shift, "plugin": plugin, "cycle": cycle}


def _apply(dn, remap, fast, no_swap=False):
    from models.demos.blackhole.qwen36.tt.gdn.tp import RemapCtx

    if not fast:
        dn.remap_slots(remap)
        return
    ctx = RemapCtx(dn.mesh, remap)
    orig = None
    if no_swap:
        orig = dn._swap_packed_parity_dev
        dn._swap_packed_parity_dev = lambda x, c: ttnn.clone(x)
    try:
        dn.remap_slots(remap, ctx=ctx)
    finally:
        ctx.close()
        if orig is not None:
            dn._swap_packed_parity_dev = orig


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2")], indirect=True)
def test_remap_fast_buffers(mesh_device):
    n_dev = mesh_device.get_num_devices()
    host = _host_state(n_dev, 1234)
    a, b = _shell(mesh_device, host), _shell(mesh_device, host)
    assert not _compare(_read(a), _read(b)), "copies differ before any remap"
    total = 0
    for name, remap in _remaps().items():
        _apply(a, remap, fast=False)
        _apply(b, remap, fast=True)
        bad = _compare(_read(a), _read(b))
        moved = sum(1 for i, s in enumerate(remap) if s != i)
        cross = sum(1 for i, s in enumerate(remap) if (i ^ s) & 1)
        logger.info(f"[remap_fast] {name}: moved {moved} cross {cross}: {len(bad)} rows differ {bad[:6]}")
        total += len(bad)
    # sequential remaps compound: A and B are now 5 remaps deep on the same starting bits
    assert total == 0, f"fast remap differs from the slice/concat path on {total} rows"
    # negative control 1: no device parity swap -> the cross-parity packed rows must differ
    remap = _remaps()["plugin"]
    _apply(a, remap, fast=False)
    _apply(b, remap, fast=True, no_swap=True)
    bad = _compare(_read(a), _read(b))
    cross_rows = sorted({i for i, s in enumerate(remap) if (i ^ s) & 1})
    logger.info(f"[remap_fast] NEG no-swap: {len(bad)} rows differ {bad[:8]} (cross rows {cross_rows})")
    assert bad and {r for k, _, r in bad} == set(cross_rows) and all(k == "hist" for k, _, _ in bad)
    # rebuild B from A's bits, then negative control 2: a wrong remap (two entries swapped) must differ on those rows
    b = _shell(mesh_device, {k: v for k, v in _read(a).items()})
    assert not _compare(_read(a), _read(b))
    good = _remaps()["condense"]
    wrong = list(good)
    wrong[0], wrong[1] = wrong[1], wrong[0]
    _apply(a, good, fast=False)
    _apply(b, wrong, fast=True)
    bad = _compare(_read(a), _read(b))
    logger.info(f"[remap_fast] NEG wrong remap: {len(bad)} rows differ, rows {sorted({r for _, _, r in bad})}")
    assert bad and {r for _, _, r in bad} == {0, 1}
    logger.info("[remap_fast] BIT-EXACT on every row; both negative controls mismatch")

    n_time = int(os.environ.get("REMAP_TIME_LAYERS", "0"))
    if not n_time:
        return
    shells = [a] + [_shell(mesh_device, _host_state(n_dev, 99 + i, specials=False)) for i in range(n_time - 1)]
    for name, remap in _remaps().items():
        res = {}
        for fast in (False, True, False, True):
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            if fast:
                from models.demos.blackhole.qwen36.tt.gdn.tp import RemapCtx

                ctx = RemapCtx(mesh_device, remap)
                for dn in shells:
                    dn.remap_slots(remap, ctx=ctx)
                ctx.close()
            else:
                for dn in shells:
                    dn.remap_slots(remap)
            t_disp = time.perf_counter() - t0
            ttnn.synchronize_device(mesh_device)
            res.setdefault(fast, []).append((1e3 * t_disp, 1e3 * (time.perf_counter() - t0)))
        phases = {}
        for fast in (False, True):  # per-phase breakdown (device synced between phases)
            from models.demos.blackhole.qwen36.tt.gdn.tp import RemapCtx

            tm = {}
            ctx = RemapCtx(mesh_device, remap) if fast else None
            for dn in shells:
                dn.remap_slots(remap, timing=tm, ctx=ctx)
            if ctx is not None:
                ctx.close()
            phases[fast] = " ".join(f"{k}={1e3 * v:.1f}" for k, v in tm.items() if k.endswith("_s"))
        logger.info(f"[remap_fast] PHASES {name}: old {phases[False]} | fast {phases[True]}")
        moved = sum(1 for i, s in enumerate(remap) if s != i)
        cross = sum(1 for i, s in enumerate(remap) if (i ^ s) & 1)
        logger.info(
            f"[remap_fast] TIME {name} moved {moved} cross {cross} x{n_time} layers: "
            f"old dispatch/total ms {res[False]}  fast {res[True]}"
        )


# ---------------------------------------------------------------------------------------------------------------------
# test_remap_fast_model: the served shape on the real model (REMAP_LAYERS, default 16 = 12 GDN + 4 attention layers,
# B = 32, fused-conv decode). Admissions (traced chunk prefill + the plain slot write) and traced decode steps between
# plugin-shaped remaps, i.e. the remaps act on real, advanced decode state (stale taps, advanced packed history).
#  * in process, at every remap: the GDN state is snapshotted, remapped by the slice/concat path, digested (every row,
#    bits), restored, remapped by the fast path and digested again -> must be identical on EVERY row;
#  * across processes: the run then continues on the path QWEN36_GDN_REMAP_FAST selects; per-step decode logits of the
#    live rows and the live-row state are written to REMAP_OUT, and `python <this file> a.json b.json` compares a
#    knob-0 and a knob-1 run (the traced prefill and decode are deterministic run to run).
# ---------------------------------------------------------------------------------------------------------------------
BLOCK = 64
BPU = 80  # blocks per KV region (5120 tokens)


def _digest(t):
    import hashlib

    return hashlib.sha1(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def _gdn_layers(model):
    return [layer.attention for layer in model.layers if not layer.is_full_attention]


def _snapshot(model):
    snap = []
    for dn in _gdn_layers(model):
        bufs = [dn.rec_state, *dn.conv_states, dn.conv_hist_packed]
        snap.append([[ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)] for t in bufs])
    return snap


def _restore(model, snap):
    for dn, per in zip(_gdn_layers(model), snap):
        bufs = [dn.rec_state, *dn.conv_states, dn.conv_hist_packed]
        for t, host in zip(bufs, per):
            src = _to_dev(model.mesh_device, host, t.dtype)
            ttnn.copy(src, t)
            ttnn.deallocate(src)


def _row_digests(model, rows=None):
    """{buf[d<dev>s<row>]: digest} for every row (or `rows`) of every GDN buffer, both devices."""
    out = {}
    for li, per in enumerate(_snapshot(model)):
        for bi, devs in enumerate(per):
            dim = 1 if 1 <= bi <= K else 0
            name = ["rec", "conv0", "conv1", "conv2", "conv3", "hist"][bi]
            for d, x in enumerate(devs):
                for s in range(x.shape[dim]) if rows is None else rows:
                    out[f"L{li}.{name}[d{d}s{s}]"] = _digest(x.select(dim, s))
    return out


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576, "trace_region_size": 1073741824}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2")], indirect=True)
def test_remap_fast_model(mesh_device, reset_seeds, ensure_gc):
    import json

    from models.demos.blackhole.qwen36.tt.gdn.tp import RemapCtx, gdn_remap_fast_enabled
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from models.tt_transformers.tt.common import copy_host_to_device

    n_layers = int(os.environ.get("REMAP_LAYERS", "16"))
    n_decode = int(os.environ.get("REMAP_DECODE", "3"))
    out_path = os.environ.get("REMAP_OUT", "remap_fast.json")
    fast = gdn_remap_fast_enabled(mesh_device)
    n_regions = B + 8
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=8192, n_layers=n_layers)
    args = model.args
    model.allocate_kv_caches(
        (n_regions * BPU + 1, args.n_local_kv_heads, BLOCK, args.head_dim), ttnn.bfloat8_b, batch_size=B
    )
    region_pt = torch.stack([torch.arange(r * BPU, (r + 1) * BPU, dtype=torch.int32) for r in range(n_regions)])
    warm_pt = torch.arange(n_regions * BPU, dtype=torch.int32).reshape(1, -1)
    model.sync_gdn_decode_state()
    widths = [8, B]
    for w in widths:  # compile every width before any trace is parked
        dev0 = model.prepare_inputs_decode(
            torch.full((w, 1), 100, dtype=torch.int32),
            torch.full((w,), -1, dtype=torch.int32),
            page_table=region_pt[:w],
        )
        model.ttnn_decode_forward(dev0[0], dev0[1], rot_mat_idxs=dev0[2], page_table=dev0[3])
    ttnn.synchronize_device(mesh_device)
    prev = model._bind_gdn_prefill_scratch()
    try:
        model.capture_prefill_trace_chunked(mesh_device, warm_pt, chunk_size=2048)
    finally:
        model._unbind_gdn_prefill_scratch(prev)
    model.warmup_gdn_slot_write()
    model.sync_gdn_decode_state()
    d0 = _row_digests(model)
    model.warmup_gdn_remap()  # compiles the fast-path programs (knob on); its 4 remaps compose to the identity
    d1 = _row_digests(model)
    warm_diff = sum(d0[k] != d1[k] for k in d0)
    logger.info(f"[remap_model] warmup_gdn_remap left {warm_diff} of {len(d0)} row digests changed")
    assert warm_diff == 0
    traces = {}
    for w in widths:
        host = model.prepare_decode_inputs_host(
            torch.full((w, 1), 100, dtype=torch.int32),
            torch.full((w,), -1, dtype=torch.int32),
            page_table=region_pt[:w],
        )
        dev = copy_host_to_device(host, mesh_device=mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        traces[w] = (tid, dev, out)

    g = torch.Generator().manual_seed(4321)
    live = {}  # row -> {"pos": next position, "region": KV region}
    free_regions = list(range(n_regions))
    steps = []

    def admit(slots, lens):
        toks = [torch.randint(1000, 100000, (1, T), generator=g, dtype=torch.int64).to(torch.int32) for T in lens]
        rows = torch.zeros(len(slots), 1024, dtype=torch.int32)
        for i, (s, T) in enumerate(zip(slots, lens)):
            if s in live:
                free_regions.append(live[s]["region"])
            r = free_regions.pop(0)
            nblk = -(-T // BLOCK)
            rows[i, :nblk] = region_pt[r, :nblk]
            live[s] = {"pos": T, "region": r}
        logits = model.prefill_paged_slots(toks, rows, slots, valid_lens=lens)
        return [_digest(lg.float()) for lg in logits]

    def decode(n):
        w = 8 if max(live) < 8 else B
        tid, dev, out = traces[w]
        res = []
        for _ in range(n):
            tokens = torch.full((w, 1), 100, dtype=torch.int32)
            pos = torch.full((w,), -1, dtype=torch.int32)
            pt = region_pt[:w].clone()
            for s, st in live.items():
                tokens[s, 0] = int(torch.randint(1000, 100000, (1,), generator=g))
                pos[s] = st["pos"]
                pt[s] = region_pt[st["region"]]
            host = model.prepare_decode_inputs_host(tokens, pos, page_table=pt)
            copy_host_to_device(host, device_tensors=dev)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            lg = model.process_output_decode(out, w)[:, 0, : model.vocab_size].float()
            res.append({str(s): _digest(lg[s].contiguous()) for s in sorted(live)})
            for st in live.values():
                st["pos"] += 1
        ttnn.synchronize_device(mesh_device)
        return w, res

    def remap(want, gone=()):
        """Plugin _decode_state_slot_remap: rows 0..len(want)-1 read want's slots, the rest the untaken slots."""
        for s in gone:
            free_regions.append(live.pop(s)["region"])
        idx = list(want) + [s for s in range(B) if s not in set(want)]
        snap = _snapshot(model)
        t0 = time.perf_counter()
        for dn in _gdn_layers(model):
            dn.remap_slots(idx)
        ttnn.synchronize_device(mesh_device)
        t_old = time.perf_counter() - t0
        d_old = _row_digests(model)
        _restore(model, snap)
        t0 = time.perf_counter()
        ctx = RemapCtx(mesh_device, idx)
        for dn in _gdn_layers(model):
            dn.remap_slots(idx, ctx=ctx)
        ctx.close()
        ttnn.synchronize_device(mesh_device)
        t_fast = time.perf_counter() - t0
        d_fast = _row_digests(model)
        diff = sorted(k for k in d_old if d_old[k] != d_fast[k])
        if not fast:  # continue on the old path's result
            _restore(model, snap)
            for dn in _gdn_layers(model):
                dn.remap_slots(idx)
        # the snapshot/restore round trip itself must be exact: the continued state equals the digested one
        d_now = _row_digests(model)
        cont = sorted(k for k in d_now if d_now[k] != (d_fast if fast else d_old)[k])
        new_live = {}
        for i, s in enumerate(idx):
            if s in live:
                new_live[i] = live[s]
        live.clear()
        live.update(new_live)
        moved = sum(1 for i, s in enumerate(idx) if s != i)
        cross = sum(1 for i, s in enumerate(idx) if (i ^ s) & 1)
        logger.info(
            f"[remap_model] remap moved {moved} cross {cross}: {len(diff)} of {len(d_old)} row digests differ old vs "
            f"fast {diff[:4]}; continued-state mismatches {len(cont)}; old {1e3 * t_old:.0f} ms fast {1e3 * t_fast:.0f} ms"
        )
        return {"idx": idx, "moved": moved, "cross": cross, "rows_differ": len(diff), "cont_differ": len(cont)}

    plan = [
        ("admit", [0], [300]),
        ("admit", [1], [2048]),
        ("admit", [5], [777]),
        ("admit", [2, 3], [64, 3000]),
        ("admit", [6], [129]),
        ("remap", [0, 1, 2, 3, 6, 5], []),  # a leave-free reorder (the admission order the plugin saw)
        ("remap", [0, 1, 5, 3, 4], [2]),  # row 2 finished: the last row (5) moves into the hole
        ("admit", [7], [2100]),  # a join into a slot next to moved rows
        ("remap", [7, 0, 1, 2, 3, 4], []),  # the joined row moves to the front (all rows shift, cross parity)
        ("admit", [12, 31], [500, 64]),  # width B from here on
        ("remap", [31, 5, 4, 3, 2, 1, 0, 12], []),
        ("remap", [0, 2, 4, 6], [1, 3, 5, 7]),  # 4 leave at once
    ]
    for op, a, b in plan:
        rec = {"op": op}
        if op == "admit":
            rec["slots"], rec["lens"], rec["logits"] = a, b, admit(a, b)
        else:
            rec.update(remap(a, b))
        rec["decode_width"], rec["decode_logits"] = decode(n_decode)
        rec["state_live"] = _row_digests(model, rows=sorted(live))
        steps.append(rec)
    for tid, _, _ in traces.values():
        ttnn.release_trace(mesh_device, tid)
    with open(out_path, "w") as f:
        json.dump({"fast": fast, "n_layers": n_layers, "steps": steps}, f, indent=1)
    bad = sum(st.get("rows_differ", 0) + st.get("cont_differ", 0) for st in steps)
    logger.info(f"[remap_model] fast={fast} wrote {out_path}; in-process old vs fast row mismatches: {bad}")
    assert bad == 0


def _compare_runs(a_path, b_path):
    import json

    a, b = json.load(open(a_path)), json.load(open(b_path))
    bad = 0
    for i, (sa, sb) in enumerate(zip(a["steps"], b["steps"])):
        dl = [j for j, (x, y) in enumerate(zip(sa["decode_logits"], sb["decode_logits"])) if x != y]
        ds = [k for k in sa["state_live"] if sa["state_live"][k] != sb["state_live"].get(k)]
        lg = sa.get("logits") != sb.get("logits")
        bad += len(dl) + len(ds) + int(lg) + int(len(sa["decode_logits"]) != len(sb["decode_logits"]))
        print(
            f"step {i} {sa['op']} {sa.get('slots', sa.get('idx', [])[:8])}: prefill logits {'DIFFER' if lg else 'equal'}; "
            f"{len(sa['decode_logits'])} decode steps @w{sa['decode_width']} logits differ {dl or 'none'}; "
            f"live-row state {len(ds)}/{len(sa['state_live'])} differ"
        )
    print("BIT-EXACT" if bad == 0 and len(a["steps"]) == len(b["steps"]) else f"MISMATCH ({bad})")
    return bad


if __name__ == "__main__":
    import sys

    sys.exit(1 if _compare_runs(sys.argv[1], sys.argv[2]) else 0)
