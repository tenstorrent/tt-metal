# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare two DeepSeek V3 B1 KV-cache dump files (.pt) using PCC and error stats."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from models.common.utility_functions import comp_pcc


def _load_kv_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, got {type(obj)}")
    return obj


def _per_position_max_abs(t: torch.Tensor) -> torch.Tensor:
    """Max absolute value over batch, head, and last dim for each sequence index (dim 2)."""
    # (B, H, S, D) -> (S, B*H*D)
    t = t.abs().to(torch.float32)
    s = t.shape[2]
    flat = t.permute(2, 0, 1, 3).reshape(s, -1)
    return flat.amax(dim=1)


def _auto_filled_len(a: torch.Tensor, b: torch.Tensor, *, min_len: int = 16, eps: float = 1e-12) -> int:
    ma = _per_position_max_abs(a)
    mb = _per_position_max_abs(b)
    combined = torch.maximum(ma, mb)
    mask = combined > eps
    if not bool(mask.any()):
        return min_len
    last = int(torch.nonzero(mask, as_tuple=False)[-1].item())
    detected = last + 1
    s = a.shape[2]
    return min(max(detected, min_len), s)


def _compare_slice(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    diff = (a32 - b32).abs()
    return a32, diff


def _per_position_diff_max_mean(diff: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Max and mean absolute diff over batch, head, and KV dim for each sequence index (dim 2)."""
    max_pp = diff.amax(dim=(0, 1, 3))
    mean_pp = diff.mean(dim=(0, 1, 3))
    return max_pp, mean_pp


def _per_position_pcc(a_s: torch.Tensor, b_s: torch.Tensor) -> list[float]:
    """Pearson correlation per sequence index (slow: one comp_pcc per position)."""
    s = a_s.shape[2]
    out: list[float] = []
    for t in range(s):
        _, p = comp_pcc(a_s[:, :, t, :], b_s[:, :, t, :], pcc=0.0)
        out.append(float(p) if not math.isnan(float(p)) else 0.0)
    return out


def _write_per_position_csv(
    path: Path,
    max_pp: torch.Tensor,
    mean_pp: torch.Tensor,
    pcc_pp: list[float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if pcc_pp is not None:
            f.write("position_id,max_abs,mean_abs,pcc\n")
            for t in range(max_pp.numel()):
                f.write(f"{t},{max_pp[t].item():.8g},{mean_pp[t].item():.8g},{pcc_pp[t]:.8g}\n")
        else:
            f.write("position_id,max_abs,mean_abs\n")
            for t in range(max_pp.numel()):
                f.write(f"{t},{max_pp[t].item():.8g},{mean_pp[t].item():.8g}\n")


def _finite_stats(x: torch.Tensor) -> tuple[float, float, float, float, int, int]:
    """min, max, mean, std, nan_count, inf_count for float tensor."""
    xf = x.detach().float().flatten()
    n_nan = int(torch.isnan(xf).sum().item())
    n_inf = int(torch.isinf(xf).sum().item())
    m = torch.isfinite(xf)
    if not bool(m.any()):
        return (float("nan"),) * 4 + (n_nan, n_inf)
    y = xf[m]
    return (
        float(y.min().item()),
        float(y.max().item()),
        float(y.mean().item()),
        float(y.std(unbiased=False).item()),
        n_nan,
        n_inf,
    )


def _inspect_positions(
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    positions: list[int],
    *,
    file_a_name: str,
    file_b_name: str,
    preview_slot: int,
    preview_dim: int,
    slots_summary: int,
    top_dims: int,
) -> None:
    """Print diagnostics for selected sequence indices (dim 2)."""
    num_slots, _, kv_d = a_s.shape[0], a_s.shape[1], a_s.shape[3]
    for t in positions:
        print(f"\n=== inspect position_id={t} ({file_a_name} vs {file_b_name}) ===")
        ap = a_s[:, :, t, :]
        bp = b_s[:, :, t, :]
        apf = ap.float()
        bpf = bp.float()
        d = (apf - bpf).abs()

        print(f"slice shape (num_slots, n_heads, kv_dim) = {tuple(ap.shape)}")
        ma = _finite_stats(apf)
        mb = _finite_stats(bpf)
        print(
            f"{file_a_name} float32: min={ma[0]:.6g} max={ma[1]:.6g} mean={ma[2]:.6g} std={ma[3]:.6g} "
            f"nan={ma[4]} inf={ma[5]}"
        )
        print(
            f"{file_b_name} float32: min={mb[0]:.6g} max={mb[1]:.6g} mean={mb[2]:.6g} std={mb[3]:.6g} "
            f"nan={mb[4]} inf={mb[5]}"
        )
        print(f"|A-B| float32: max={d.max().item():.6g} mean={d.mean().item():.6g} " f"median={d.median().item():.6g}")

        s_slots, s_heads, s_dim = d.shape
        d_flat = d.reshape(-1)
        flat_max = int(d_flat.argmax().item())
        slot_i = flat_max // (s_heads * s_dim)
        rem = flat_max % (s_heads * s_dim)
        head_i = rem // s_dim
        dim_i = rem % s_dim
        print(f"argmax |A-B| at (slot={slot_i}, head={head_i}, kv_dim={dim_i}) " f"value={d_flat[flat_max].item():.6g}")

        n_show = min(max(0, slots_summary), num_slots)
        if n_show > 0:
            print(f"per-slot max |A-B| at pos {t} (first {n_show} slots, head 0):")
            for s in range(n_show):
                print(f"  slot {s:2d}: max={d[s, 0, :].max().item():.6g} mean={d[s, 0, :].mean().item():.6g}")

        if preview_slot < 0 or preview_slot >= num_slots:
            print(f"(skip vector preview: inspect-slot {preview_slot} out of range [0,{num_slots - 1}])")
            continue

        kd = min(preview_dim, kv_d)
        a_vec = ap[preview_slot, 0, :kd].float()
        b_vec = bp[preview_slot, 0, :kd].float()
        dv = d[preview_slot, 0, :kd]
        print(
            f"vector preview slot={preview_slot} head=0 first {kd} kv_dims "
            f"(bf16 equal count in full dim: {int(torch.eq(ap[preview_slot], bp[preview_slot]).all().item())})"
        )
        print(f"  A f32: {a_vec.tolist()}")
        print(f"  B f32: {b_vec.tolist()}")
        print(f"  |Δ|  : {dv.tolist()}")

        tk = min(max(0, top_dims), kv_d)
        if tk > 0:
            topv, topi = torch.topk(d[preview_slot, 0, :], k=tk)
            print(
                f"largest |Δ| at slot {preview_slot} (top {tk} kv_dims): "
                + ", ".join(f"d[{int(i)}]={float(v):.6g}" for v, i in zip(topv.tolist(), topi.tolist()))
            )


def _report_per_position(
    *,
    max_pp: torch.Tensor,
    mean_pp: torch.Tensor,
    pcc_pp: list[float] | None,
    notice_thr: float,
    top_k: int,
    print_limit: int | None,
) -> None:
    s = int(max_pp.numel())
    mask = max_pp > notice_thr
    if bool(mask.any()):
        first = int(torch.nonzero(mask, as_tuple=False)[0].item())
        n_bad = int(mask.sum().item())
    else:
        first = -1
        n_bad = 0

    print("\n--- per-position divergence (seq index = position_id along KV dim 2) ---")
    print(f"notice_threshold (max_abs): {notice_thr}")
    if first >= 0:
        print(f"first_position_max_abs_gt_threshold: {first}")
        print(f"num_positions_max_abs_gt_threshold: {n_bad} / {s}")
    else:
        print(f"first_position_max_abs_gt_threshold: (none in 0..{s - 1})")
        print(f"num_positions_max_abs_gt_threshold: 0 / {s}")

    flat_idx = torch.argsort(max_pp, descending=True)
    k = min(top_k, s)
    print(f"top_{k}_positions_by_max_abs:")
    for i in range(k):
        t = int(flat_idx[i].item())
        line = f"  pos={t} max_abs={max_pp[t].item():.8g} mean_abs={mean_pp[t].item():.8g}"
        if pcc_pp is not None:
            line += f" pcc={pcc_pp[t]:.8g}"
        print(line)

    if print_limit is not None and print_limit > 0:
        n = min(print_limit, s)
        print(f"\nper_position_table (first {n} rows):")
        hdr = "position_id max_abs mean_abs"
        if pcc_pp is not None:
            hdr += " pcc"
        print(hdr)
        for t in range(n):
            row = f"{t} {max_pp[t].item():.8g} {mean_pp[t].item():.8g}"
            if pcc_pp is not None:
                row += f" {pcc_pp[t]:.8g}"
            print(row)
        if s > n:
            print(f"... ({s - n} more positions; use --per-position-csv for full table)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two KV cache .pt dumps (PCC + stats).")
    parser.add_argument("file_a", type=Path, help="First dump (.pt)")
    parser.add_argument("file_b", type=Path, help="Second dump (.pt)")
    parser.add_argument("--threshold", type=float, default=0.99, help="PCC pass threshold (default: 0.99)")
    parser.add_argument("--seq-len", type=int, default=None, metavar="N", help="Compare [:, :, :N, :] only")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Compare the full sequence dimension (no prefix slice)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print one pipe-separated summary line only (for scripting)",
    )
    parser.add_argument(
        "--per-position",
        action="store_true",
        help="Report max/mean |Δ| per sequence index (position_id); see also --per-position-csv",
    )
    parser.add_argument(
        "--per-position-pcc",
        action="store_true",
        help="With --per-position, also compute PCC per position (slow)",
    )
    parser.add_argument(
        "--per-position-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write all positions to CSV (implies --per-position stats; adds pcc column if --per-position-pcc)",
    )
    parser.add_argument(
        "--divergence-notice",
        type=float,
        default=1e-3,
        metavar="THR",
        help="Highlight first position where max_abs exceeds this (default: 1e-3)",
    )
    parser.add_argument(
        "--per-position-top-k",
        type=int,
        default=15,
        metavar="K",
        help="How many worst positions to list (default: 15)",
    )
    parser.add_argument(
        "--per-position-print-limit",
        type=int,
        default=32,
        metavar="N",
        help="Print first N rows of the per-position table (0 = skip table; default: 32)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print value / |Δ| diagnostics at sequence indices (see --inspect-pos)",
    )
    parser.add_argument(
        "--inspect-pos",
        action="append",
        type=int,
        default=None,
        metavar="T",
        help="Sequence index to inspect (repeatable). Default: 0 when --inspect is set.",
    )
    parser.add_argument(
        "--inspect-slot",
        type=int,
        default=0,
        metavar="S",
        help="Slot (batch dim) for vector preview (default: 0)",
    )
    parser.add_argument(
        "--inspect-dim",
        type=int,
        default=32,
        metavar="K",
        help="Print first K KV dimensions in vector preview (default: 32)",
    )
    parser.add_argument(
        "--inspect-slots",
        type=int,
        default=8,
        metavar="N",
        help="Show per-slot max |Δ| for first N slots at each inspect position (default: 8; 0=skip)",
    )
    parser.add_argument(
        "--inspect-top-dims",
        type=int,
        default=8,
        metavar="K",
        help="List K largest |Δ| kv_dims for --inspect-slot (default: 8; 0=skip)",
    )
    args = parser.parse_args(argv)

    if args.full and args.seq_len is not None:
        parser.error("Use only one of --full and --seq-len")

    want_per_pos = args.per_position or args.per_position_csv is not None
    inspect_positions: list[int] = []
    if args.inspect:
        raw = args.inspect_pos if args.inspect_pos is not None else [0]
        seen: set[int] = set()
        for t in raw:
            if t not in seen:
                seen.add(t)
                inspect_positions.append(t)

    if args.compact and want_per_pos:
        parser.error("Do not combine --compact with --per-position / --per-position-csv")
    if args.compact and args.inspect:
        parser.error("Do not combine --compact with --inspect")
    if args.per_position_pcc and not want_per_pos:
        parser.error("--per-position-pcc requires --per-position or --per-position-csv")

    a = _load_kv_tensor(args.file_a)
    b = _load_kv_tensor(args.file_b)

    if a.dim() != 4 or b.dim() != 4:
        raise ValueError(f"Expected 4D KV tensors, got {a.dim()}D and {b.dim()}D")

    seq_full = a.shape[2]
    if args.full:
        filled_len = seq_full
    elif args.seq_len is not None:
        if args.seq_len <= 0 or args.seq_len > seq_full:
            parser.error(f"--seq-len must be in [1, {seq_full}]")
        filled_len = args.seq_len
    else:
        filled_len = _auto_filled_len(a, b)

    a_s = a[:, :, :filled_len, :].contiguous()
    b_s = b[:, :, :filled_len, :].contiguous()

    a32, diff = _compare_slice(a_s, b_s)
    b32 = b_s.to(torch.float32)

    cal_atol = float(diff.max().item())
    cal_rtol = float((diff / b32.abs().clamp_min(1e-20)).max().item())
    passing, cal_pcc = comp_pcc(a32, b32, pcc=args.threshold)
    cal_pcc = float(cal_pcc)
    if math.isnan(cal_pcc):
        cal_pcc = 0.0
        passing = False
    mean_abs = float(diff.mean().item())
    rms = float(math.sqrt((diff**2).mean().item()))
    eq_frac = float(torch.eq(a_s, b_s).float().mean().item())

    ok = passing
    status = "PASS" if ok else "FAIL"
    base_a = args.file_a.name
    base_b = args.file_b.name

    if args.compact:
        print(
            "KV_CACHE_CMP|"
            f"{base_a}|{base_b}|{filled_len}|{cal_pcc:.8g}|{cal_atol:.8g}|{mean_abs:.8g}|{rms:.8g}|{eq_frac:.8g}|{status}",
            flush=True,
        )
        return 0 if ok else 1

    print(f"file_a: {args.file_a.resolve()}")
    print(f"file_b: {args.file_b.resolve()}")
    print(f"shape: {tuple(a.shape)}")
    print(f"compare_seq_len: {filled_len} (full_seq={seq_full})")
    print(f"PCC: {cal_pcc}")
    print(f"max_abs (atol): {cal_atol}")
    print(f"max_rel (rtol vs |b|): {cal_rtol}")
    print(f"mean_abs: {mean_abs}")
    print(f"rms_abs: {rms}")
    print(f"frac_eq (exact, same dtype): {eq_frac}")
    print(f"Max ATOL Delta: {cal_atol}, Max RTOL Delta: {cal_rtol}, PCC: {cal_pcc}")
    print(
        f"SUMMARY|{base_a}|{base_b}|{filled_len}|{cal_pcc:.8g}|{cal_atol:.8g}|{mean_abs:.8g}|{status}",
        flush=True,
    )

    if inspect_positions:
        for t in inspect_positions:
            if t < 0 or t >= filled_len:
                raise ValueError(
                    f"--inspect-pos {t} out of range for current slice [0, {filled_len - 1}] "
                    f"(use --seq-len / --full to widen)"
                )
        _inspect_positions(
            a_s,
            b_s,
            inspect_positions,
            file_a_name=base_a,
            file_b_name=base_b,
            preview_slot=args.inspect_slot,
            preview_dim=max(0, args.inspect_dim),
            slots_summary=max(0, args.inspect_slots),
            top_dims=max(0, args.inspect_top_dims),
        )

    if want_per_pos:
        max_pp, mean_pp = _per_position_diff_max_mean(diff)
        pcc_list: list[float] | None = _per_position_pcc(a_s, b_s) if args.per_position_pcc else None
        if args.per_position_csv is not None:
            _write_per_position_csv(args.per_position_csv, max_pp, mean_pp, pcc_list)
            print(f"\nWrote per-position CSV: {args.per_position_csv.resolve()}")
        if args.per_position:
            lim = args.per_position_print_limit if args.per_position_print_limit > 0 else None
            _report_per_position(
                max_pp=max_pp,
                mean_pp=mean_pp,
                pcc_pp=pcc_list,
                notice_thr=args.divergence_notice,
                top_k=max(1, args.per_position_top_k),
                print_limit=lim,
            )
        elif args.per_position_csv is not None and not args.per_position:
            print("\n(per-position CSV only; add --per-position for console summary / worst-positions list)")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
