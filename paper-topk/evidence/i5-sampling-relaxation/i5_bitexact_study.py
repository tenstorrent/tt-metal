#!/usr/bin/env python3
"""I5 bit-exactness study: OLD (production indices_tensor/stable) vs NEW (relaxed/routed)
ttnn.topk call forms on identical seeded inputs, plus a real Sampling1D end-to-end
old-vs-new comparison. All comparisons in the int-bits domain.

Shapes (rows=32, k=32, bf16 TILE interleaved DRAM):
  A qwen36_tp4    : W=65536 padded (valid 37984, pad = finfo(bf16).min)   -> gate True
  B 1chip_split   : full row 128256 randn, halves W=64128 each            -> gate True
      B1 sampling_1d form  : old idx uint16 iota[0..64127] both halves, no stable
      B2 tt_sampling form  : old idx uint32 global iota, stable=True;
                             new half1 = +64128 offset via int32 add (patch pattern)
  C tp8_pow2 ctrl : W=32768 padded (valid 19008, pad = finfo(bf16).min)   -> gate False
                    (new call form == old call form; require bit identity)
"""

import json
import sys
import time

import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.sampling._utils import topk_would_route_to_large_indices

OUT_JSON = "/tmp/claude-1000/-home-nachiket-tt-metal/9f8f10d4-baba-4138-8904-cb9bdebdbd08/scratchpad/night/i5-sampling-relaxation/i5_bitexact_results.json"
N_TRIALS = 20
B = 32
K = 32
BF16_MIN = torch.finfo(torch.bfloat16).min  # -3.3895e38, finite; production mask value


def idx_long(t: torch.Tensor) -> torch.Tensor:
    """uint16/uint32/int32 -> exact int64."""
    if t.dtype == torch.uint16:
        return t.view(torch.int16).long() & 0xFFFF
    if t.dtype == torch.uint32:
        return t.view(torch.int32).long() & 0xFFFFFFFF
    return t.long()


def bf16_bits(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.bfloat16
    return t.view(torch.int16).long() & 0xFFFF


def dev_tensor(torch_t, mesh, dtype, layout=None):
    return ttnn.from_torch(
        torch_t,
        device=mesh,
        dtype=dtype,
        layout=layout if layout is not None else ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class Acc:
    """Per-variant accumulators (row-granular)."""

    def __init__(self, name):
        self.name = name
        self.rows = 0
        self.val_multiset_ok = 0
        self.val_seq_ok = 0
        self.idx_seq_ok = 0
        self.idx_set_ok = 0
        self.diff_positions = 0
        self.diff_tie_proven = 0
        self.nontie_failures = []  # fatal
        self.pick_rows = 0
        self.pick_ok = 0
        self.pick_diffs = []
        self.pad_wins = 0  # winners landing in padding lanes (must stay 0)

    def as_dict(self):
        d = dict(self.__dict__)
        d["nontie_failures"] = self.nontie_failures[:20]
        d["pick_diffs"] = self.pick_diffs[:20]
        return d


def compare_rows(acc: Acc, inp_bits_rows, old_vals, old_idx, new_vals, new_idx, valid=None, trial=0):
    """inp_bits_rows: [B, W] int64 bf16 bit patterns of the (padded) input row.
    old/new vals: [B, K] bf16 tensors; old/new idx: [B, K] int64 (row positions,
    same index space for old and new)."""
    ov_b, nv_b = bf16_bits(old_vals), bf16_bits(new_vals)
    for r in range(B):
        acc.rows += 1
        o_v, n_v = ov_b[r], nv_b[r]
        o_i, n_i = old_idx[r], new_idx[r]
        if valid is not None:
            pw = int((o_i >= valid).sum() + (n_i >= valid).sum())
            acc.pad_wins += pw
        acc.val_multiset_ok += int(torch.equal(o_v.sort().values, n_v.sort().values))
        acc.val_seq_ok += int(torch.equal(o_v, n_v))
        seq_eq = torch.equal(o_i, n_i)
        acc.idx_seq_ok += int(seq_eq)
        acc.idx_set_ok += int(torch.equal(o_i.sort().values, n_i.sort().values))
        if not seq_eq:
            row_bits = inp_bits_rows[r]
            for p in torch.nonzero(o_i != n_i).flatten().tolist():
                acc.diff_positions += 1
                oi, ni = int(o_i[p]), int(n_i[p])
                # a diff is a proven tie iff the input values at both positions are bit-equal
                if 0 <= oi < row_bits.numel() and 0 <= ni < row_bits.numel() and int(row_bits[oi]) == int(row_bits[ni]):
                    acc.diff_tie_proven += 1
                else:
                    acc.nontie_failures.append(
                        dict(
                            trial=trial,
                            row=r,
                            pos=p,
                            old_idx=oi,
                            new_idx=ni,
                            old_val_bits=int(o_v[p]),
                            new_val_bits=int(n_v[p]),
                            inp_old_bits=int(row_bits[oi]) if 0 <= oi < row_bits.numel() else None,
                            inp_new_bits=int(row_bits[ni]) if 0 <= ni < row_bits.numel() else None,
                        )
                    )


def greedy_pick(vals_bf16_row, idx_row):
    """Documented _adjust_values_for_tiebreak guarantee: among gathered candidates,
    boost the lowest-global-index holder of the row max -> argmax picks it.
    Returns that index."""
    v = vals_bf16_row.float()
    bits = bf16_bits(vals_bf16_row)
    tied = bits == bits[v.argmax()]
    return int(idx_row[tied].min())


def compare_pick(acc: Acc, old_vals, old_gidx, new_vals, new_gidx, trial):
    for r in range(B):
        acc.pick_rows += 1
        po = greedy_pick(old_vals[r], old_gidx[r])
        pn = greedy_pick(new_vals[r], new_gidx[r])
        if po == pn:
            acc.pick_ok += 1
        else:
            acc.pick_diffs.append(dict(trial=trial, row=r, old_pick=po, new_pick=pn))


def timed(mesh, fn, sink):
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    out = fn()
    ttnn.synchronize_device(mesh)
    sink.append(time.perf_counter() - t0)
    return out


def make_padded_input(trial_seed, valid, W):
    torch.manual_seed(trial_seed)
    x = torch.full((1, 1, B, W), BF16_MIN, dtype=torch.float32)
    x[..., :valid] = torch.randn(1, 1, B, valid, dtype=torch.float32)
    return x.to(torch.bfloat16)


def study_single(mesh, name, W, valid, stock_idx_dtype, stable, expect_gate, expected_dtype, results, timers):
    """Shapes A and C. OLD: topk(x, indices_tensor=stock iota(+ -1 pad), stable=stable).
    NEW (gate True): bare topk -> typecast to stock dtype if different.
    NEW (gate False): identical old-form call, require bit identity."""
    acc = Acc(name)
    idx_row = torch.full((1, 1, B, W), -1, dtype=torch.int32)
    idx_row[..., :valid] = torch.arange(valid, dtype=torch.int32)
    stock_idx_tt = dev_tensor(idx_row, mesh, stock_idx_dtype)
    t_old, t_new = [], []
    for trial in range(N_TRIALS):
        xh = make_padded_input(31337 + trial, valid, W)
        inp_bits = bf16_bits(xh)[0, 0]  # [B, W]
        x_tt = dev_tensor(xh, mesh, ttnn.bfloat16)
        gate = topk_would_route_to_large_indices(x_tt, K, mesh)
        assert gate == expect_gate, f"{name}: gate={gate}, expected {expect_gate}"

        ov, oi = timed(
            mesh,
            lambda: ttnn.topk(x_tt, k=K, dim=-1, sub_core_grids=None, indices_tensor=stock_idx_tt, stable=stable),
            t_old,
        )
        if expect_gate:

            def new_form():
                v, i = ttnn.topk(x_tt, k=K, dim=-1)
                if i.dtype != expected_dtype:
                    i2 = ttnn.typecast(i, expected_dtype)
                    ttnn.deallocate(i)
                    i = i2
                return v, i

            nv, ni = timed(mesh, new_form, t_new)
        else:
            # patched code emits the identical old-form call
            nv, ni = timed(
                mesh,
                lambda: ttnn.topk(x_tt, k=K, dim=-1, sub_core_grids=None, indices_tensor=stock_idx_tt, stable=stable),
                t_new,
            )
        ovh = to_torch_auto_compose(ov)[0, 0]
        oih = idx_long(to_torch_auto_compose(oi))[0, 0]
        nvh = to_torch_auto_compose(nv)[0, 0]
        nih = idx_long(to_torch_auto_compose(ni))[0, 0]
        compare_rows(acc, inp_bits, ovh, oih, nvh, nih, valid=valid, trial=trial)
        compare_pick(acc, ovh, oih, nvh, nih, trial)
        for t in (ov, oi, nv, ni, x_tt):
            ttnn.deallocate(t)
    results[name] = acc.as_dict()
    timers[name] = dict(old_ms=sorted(t_old)[len(t_old) // 2] * 1e3, new_ms=sorted(t_new)[len(t_new) // 2] * 1e3)
    return acc


def study_split(mesh, results, timers):
    """Shape B: full row 128256, halves W=64128. Variants:
    b1_1d  : sampling_1d form (old idx uint16 0-based both halves, no stable; new = bare)
    b2_tts : tt_sampling form as CORRECTED (old = stock + stable=True, whose single-core
             factory ignores the supplied indices tensor and emits 0-based positions,
             GH #36329; new = bare routed + dtype normalize (no cast fires at u16)).
             Gathered pick emulates the downstream tt_indices_device_offsets add
             (+0 / +HALF) identically for old and new."""
    FULL, HALF = 128256, 64128
    acc_1d = Acc("b1_split_1d")
    acc_tts = Acc("b2_split_tts_stable")
    iota_half_u16 = dev_tensor(
        torch.arange(HALF, dtype=torch.int32).reshape(1, 1, 1, HALF).expand(1, 1, B, HALF).contiguous(),
        mesh,
        ttnn.uint16,
    )
    t_old, t_new = [], []
    for trial in range(N_TRIALS):
        torch.manual_seed(60451 + trial)
        xh = torch.randn(1, 1, B, FULL, dtype=torch.float32).to(torch.bfloat16)
        inp_bits_full = bf16_bits(xh)[0, 0]  # [B, FULL]
        halves_h = [xh[..., :HALF], xh[..., HALF:]]
        gathered = {}  # variant -> (vals list, gidx list)
        for variant in ("b1_1d", "b2_tts"):
            gathered[variant] = ([], [])
        for i in range(2):
            x_tt = dev_tensor(halves_h[i].contiguous(), mesh, ttnn.bfloat16)
            gate = topk_would_route_to_large_indices(x_tt, K, mesh)
            assert gate, f"split half {i}: gate must be True"

            # ---- NEW (shared bare routed call) ----
            def new_bare():
                return ttnn.topk(x_tt, k=K, dim=-1)

            nv, ni = timed(mesh, new_bare, t_new)
            assert ni.dtype == ttnn.uint16, f"routed idx dtype {ni.dtype} != uint16 at W=64128"
            # corrected patch: no offset, no cast fires (route u16 == stock u16 contract);
            # both b1 and b2 consume the same bare routed result, local 0-based
            nih_local = idx_long(to_torch_auto_compose(ni))[0, 0]
            nvh = to_torch_auto_compose(nv)[0, 0]

            # ---- OLD b1: sampling_1d form ----
            ov1, oi1 = timed(
                mesh,
                lambda: ttnn.topk(x_tt, k=K, dim=-1, sub_core_grids=None, indices_tensor=iota_half_u16),
                t_old,
            )
            ovh1 = to_torch_auto_compose(ov1)[0, 0]
            oih1 = idx_long(to_torch_auto_compose(oi1))[0, 0]
            # local index space; tie proof against the HALF's rows
            half_bits = inp_bits_full[:, i * HALF : (i + 1) * HALF]
            compare_rows(acc_1d, half_bits, ovh1, oih1, nvh, nih_local, valid=HALF, trial=trial)
            gathered["b1_1d"][0].append((ovh1, nvh))
            gathered["b1_1d"][1].append((oih1 + i * HALF, nih_local + i * HALF))

            # ---- OLD b2: tt_sampling form (stable=True; indices tensor values are
            # ignored by the stock single-core factory, GH #36329 -> local positions) ----
            ov2, oi2 = timed(
                mesh,
                lambda: ttnn.topk(x_tt, k=K, dim=-1, sub_core_grids=None, indices_tensor=iota_half_u16, stable=True),
                t_old,
            )
            ovh2 = to_torch_auto_compose(ov2)[0, 0]
            oih2 = idx_long(to_torch_auto_compose(oi2))[0, 0]
            # local index space; tie proof against the HALF's rows
            compare_rows(acc_tts, half_bits, ovh2, oih2, nvh, nih_local, valid=HALF, trial=trial)
            # downstream tt_indices_device_offsets emulation: +i*HALF for both forms
            gathered["b2_tts"][0].append((ovh2, nvh))
            gathered["b2_tts"][1].append((oih2 + i * HALF, nih_local + i * HALF))

            for t in (nv, ni, ov1, oi1, ov2, oi2, x_tt):
                ttnn.deallocate(t)

        # gathered (concat of both halves) greedy pick, old vs new, per variant
        for variant, acc in (("b1_1d", acc_1d), ("b2_tts", acc_tts)):
            vals, gidx = gathered[variant]
            old_v = torch.cat([vals[0][0], vals[1][0]], dim=-1)
            new_v = torch.cat([vals[0][1], vals[1][1]], dim=-1)
            old_g = torch.cat([gidx[0][0], gidx[1][0]], dim=-1)
            new_g = torch.cat([gidx[0][1], gidx[1][1]], dim=-1)
            compare_pick(acc, old_v, old_g, new_v, new_g, trial)
    results[acc_1d.name] = acc_1d.as_dict()
    results[acc_tts.name] = acc_tts.as_dict()
    timers["split_halves"] = dict(
        old_ms=sorted(t_old)[len(t_old) // 2] * 1e3, new_ms=sorted(t_new)[len(t_new) // 2] * 1e3
    )
    return acc_1d, acc_tts


def study_e2e_sampling1d(mesh, results):
    """Real Sampling1D (vocab 128256, 1x1 mesh -> split path, routed fires) old-vs-new.
    OLD = gate monkeypatched to False (today's call form); NEW = patched module as-is.
    Greedy k=1 and random k=10/p=0.9 with a fixed seed tensor, 20 steps each."""
    import models.common.modules.sampling.sampling_1d as s1d
    from models.common.modules.sampling.sampling_1d import Sampling1D

    VOCAB = 128256
    sampler = Sampling1D(vocab_size=VOCAB, mesh_device=mesh)
    sampler.load_device_buffers()
    real_gate = s1d.topk_would_route_to_large_indices

    # route-fires proof at the call surface: record ttnn.topk kwargs
    calls = []
    real_topk = ttnn.topk

    def recording_topk(*a, **kw):
        calls.append(sorted(kw.keys()))
        return real_topk(*a, **kw)

    def params(k_val, p_val, temp_val):
        mk = lambda vals, dt: ttnn.from_torch(
            vals, device=mesh, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return (
            mk(torch.full((B,), k_val, dtype=torch.int32), ttnn.uint32),
            mk(torch.full((B,), p_val, dtype=torch.float32).bfloat16(), ttnn.bfloat16),
            mk(torch.full((B,), temp_val, dtype=torch.float32).bfloat16(), ttnn.bfloat16),
        )

    seed_tensor = ttnn.from_torch(
        torch.arange(B, dtype=torch.int32) + 7,
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = {}
    for mode, (k_val, p_val, temp_val) in (("greedy_k1", (1, 0.0, 1.0)), ("random_k10_p09", (10, 0.9, 1.0))):
        k, p, temp = params(k_val, p_val, temp_val)
        same = 0
        diffs = []
        call_shapes = {"old": None, "new": None}
        for step in range(N_TRIALS):
            torch.manual_seed(90210 + step)
            logits_host = torch.randn(1, 1, B, VOCAB, dtype=torch.bfloat16)
            row_bits = bf16_bits(logits_host)[0, 0]
            toks = {}
            for form in ("old", "new"):
                s1d.topk_would_route_to_large_indices = (lambda *a, **kw: False) if form == "old" else real_gate
                ttnn.topk = recording_topk
                calls.clear()
                logits_tt = ttnn.from_torch(logits_host, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                tt_tok, _ = sampler.decode_forward(logits_tt, k=k, p=p, temp=temp, seeds=seed_tensor)
                toks[form] = to_torch_auto_compose(tt_tok).flatten()[:B].long()
                call_shapes[form] = list(calls)
                ttnn.topk = real_topk
            s1d.topk_would_route_to_large_indices = real_gate
            eq = toks["old"] == toks["new"]
            same += int(eq.all())
            for r in torch.nonzero(~eq).flatten().tolist():
                to, tn = int(toks["old"][r]), int(toks["new"][r])
                tie = int(row_bits[r][to]) == int(row_bits[r][tn])
                diffs.append(dict(step=step, row=r, old_tok=to, new_tok=tn, bitequal_logits=tie))
        out[mode] = dict(
            steps=N_TRIALS,
            steps_identical=same,
            token_diffs=diffs[:40],
            n_token_diffs=len(diffs),
            n_nontie_token_diffs=sum(1 for d in diffs if not d["bitequal_logits"]),
            old_topk_kwargs=call_shapes["old"],
            new_topk_kwargs=call_shapes["new"],
        )
    results["e2e_sampling1d_v128256"] = out

    # control: pow2 vocab -> gate inert at module level (call surface identical to old)
    sampler2 = s1d.Sampling1D(vocab_size=32768, mesh_device=mesh)
    sampler2.load_device_buffers()
    k, p, temp = params(1, 0.0, 1.0)
    ttnn.topk = recording_topk
    calls.clear()
    torch.manual_seed(4242)
    logits_tt = ttnn.from_torch(
        torch.randn(1, 1, B, 32768, dtype=torch.bfloat16), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    sampler2.decode_forward(logits_tt, k=k, p=p, temp=temp, seeds=seed_tensor)
    ttnn.topk = real_topk
    results["e2e_control_v32768_topk_kwargs"] = list(calls)


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768)
    results, timers = {}, {}
    fatal = []
    try:
        accs = []
        accs.append(
            study_single(mesh, "a_qwen36_w65536", 65536, 37984, ttnn.uint16, True, True, ttnn.uint16, results, timers)
        )
        a1, a2 = study_split(mesh, results, timers)
        accs += [a1, a2]
        accs.append(
            study_single(
                mesh, "c_tp8_w32768_ctrl", 32768, 19008, ttnn.uint16, True, False, ttnn.uint16, results, timers
            )
        )
        # control must be bit-identical in every field
        c = results["c_tp8_w32768_ctrl"]
        if c["val_seq_ok"] != c["rows"] or c["idx_seq_ok"] != c["rows"]:
            fatal.append("tp8 control not bit-identical")
        for a in accs:
            if a.nontie_failures:
                fatal.append(f"{a.name}: {len(a.nontie_failures)} NON-TIE index diffs")
            if a.pad_wins:
                fatal.append(f"{a.name}: {a.pad_wins} winners in padding lanes")
            if a.val_multiset_ok != a.rows:
                fatal.append(f"{a.name}: value multiset mismatch in {a.rows - a.val_multiset_ok} rows")
            if a.pick_ok != a.pick_rows:
                fatal.append(f"{a.name}: post-tiebreak pick differs in {a.pick_rows - a.pick_ok} rows")
        study_e2e_sampling1d(mesh, results)
        e2e = results["e2e_sampling1d_v128256"]
        for mode, r in e2e.items():
            if r["n_nontie_token_diffs"]:
                fatal.append(f"e2e {mode}: {r['n_nontie_token_diffs']} non-tie token diffs")
    finally:
        results["timers_coarse_host_ms"] = timers
        results["fatal"] = fatal
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=1, default=str)
        ttnn.close_mesh_device(mesh)

    print("\n================ I5 BIT-EXACTNESS SUMMARY ================")
    for name in ("a_qwen36_w65536", "b1_split_1d", "b2_split_tts_stable", "c_tp8_w32768_ctrl"):
        a = results[name]
        pct = lambda n, d: f"{100.0 * n / d:6.2f}%" if d else "n/a"
        print(
            f"{name:22s} rows={a['rows']:4d} valmset={pct(a['val_multiset_ok'], a['rows'])} "
            f"valseq={pct(a['val_seq_ok'], a['rows'])} idxseq={pct(a['idx_seq_ok'], a['rows'])} "
            f"idxset={pct(a['idx_set_ok'], a['rows'])} diffs={a['diff_positions']} "
            f"ties={a['diff_tie_proven']} nontie={len(a['nontie_failures'])} "
            f"padwins={a['pad_wins']} pick={pct(a['pick_ok'], a['pick_rows'])}"
        )
    print("timers (coarse host ms, median):", json.dumps(results["timers_coarse_host_ms"]))
    print("e2e:", json.dumps(results.get("e2e_sampling1d_v128256", {}), indent=1)[:2000])
    print("e2e control kwargs:", results.get("e2e_control_v32768_topk_kwargs"))
    if fatal:
        print("FATAL:", fatal)
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


if __name__ == "__main__":
    main()
