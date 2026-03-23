#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import torch

NUM_BLOCKS = 30


def _load_pt(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_compare_tensor(obj: Any, name: str) -> tuple[torch.Tensor, str]:
    if isinstance(obj, torch.Tensor):
        return obj, ""

    if isinstance(obj, (tuple, list)) and len(obj) > 0 and isinstance(obj[0], torch.Tensor):
        note = f"{name}: tuple/list detected, comparing first element only."
        return obj[0], note

    if isinstance(obj, (int, float)):
        return torch.tensor([obj]), f"{name}: scalar detected, converted to tensor."

    raise TypeError(f"{name}: unsupported saved object type: {type(obj)}")


def _align_tt_to_torch(tt_t: torch.Tensor, torch_t: torch.Tensor) -> torch.Tensor:
    """Expand or permute TT tensor to match torch shape for apple-to-apple comparison."""
    tt_t = tt_t.detach().cpu().float()
    torch_t = torch_t.detach().cpu().float()
    if tt_t.shape == torch_t.shape:
        return tt_t
    # Common singleton-axis swap case: (L,1,C) <-> (1,L,C)
    if tt_t.ndim == 3 and torch_t.ndim == 3:
        if (
            tt_t.shape[1] == 1
            and torch_t.shape[0] == 1
            and tt_t.shape[0] == torch_t.shape[1]
            and tt_t.shape[2] == torch_t.shape[2]
        ):
            return tt_t.permute(1, 0, 2).contiguous()
        if (
            tt_t.shape[0] == 1
            and torch_t.shape[1] == 1
            and tt_t.shape[1] == torch_t.shape[0]
            and tt_t.shape[2] == torch_t.shape[2]
        ):
            return tt_t.permute(1, 0, 2).contiguous()
    # Same ndim and numel: try permute (e.g. TT (L,B,C) vs torch (B,L,C))
    if tt_t.ndim == torch_t.ndim and tt_t.numel() == torch_t.numel():
        ss_tt = tuple(sorted(tt_t.shape))
        ss_torch = tuple(sorted(torch_t.shape))
        if ss_tt == ss_torch:
            available = list(range(tt_t.ndim))
            perm: list[int] = []
            for i in range(tt_t.ndim):
                target = int(torch_t.shape[i])
                found = next((j for j in available if int(tt_t.shape[j]) == target), None)
                if found is None:
                    break
                perm.append(found)
                available.remove(found)
            if len(perm) == tt_t.ndim:
                return tt_t.permute(perm).contiguous()
    while tt_t.ndim < torch_t.ndim:
        tt_t = tt_t.unsqueeze(1)
    if tt_t.ndim != torch_t.ndim:
        raise ValueError(f"Cannot align tt {tt_t.shape} to torch {torch_t.shape}: ndim mismatch")
    for i in range(tt_t.ndim):
        if tt_t.shape[i] != torch_t.shape[i] and tt_t.shape[i] != 1:
            raise ValueError(f"Cannot align tt {tt_t.shape} to torch {torch_t.shape}: dim {i} mismatch")
    return tt_t.expand(torch_t.shape).clone()


def _video_bcfhw_to_reference_seq(
    tt_video: torch.Tensor, torch_seq_shape: torch.Size, patch_size=(1, 2, 2)
) -> torch.Tensor:
    """
    Convert TT video tensor (B, C, F, H, W) to reference sequence order (B, L*n, C),
    matching reference: rearrange('b l (n c) -> b (l n) c').
    """
    if tt_video.ndim != 5:
        raise ValueError(f"Expected 5D TT video tensor, got {tt_video.shape}")
    B, C, F, H, W = tt_video.shape
    p_t, p_h, p_w = patch_size
    if F % p_t != 0 or H % p_h != 0 or W % p_w != 0:
        raise ValueError(f"Video shape {tt_video.shape} is not divisible by patch size {patch_size}")
    pf, ph, pw = F // p_t, H // p_h, W // p_w
    # Inverse of data_seq_to_patch in reference.utils.
    x = tt_video.reshape(B, C, pf, p_t, ph, p_h, pw, p_w)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()  # B, pf, ph, pw, p_t, p_h, p_w, C
    x = x.reshape(B, pf * ph * pw * p_t * p_h * p_w, C).contiguous()
    if tuple(x.shape) != tuple(torch_seq_shape):
        raise ValueError(f"Converted TT video seq shape {x.shape} does not match torch seq shape {torch_seq_shape}")
    return x


def _flatten_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = a.detach().cpu().float().flatten()
    b = b.detach().cpu().float().flatten()
    if a.numel() != b.numel():
        raise ValueError(f"numel mismatch: torch={a.numel()} vs tt={b.numel()}")
    return a, b


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    am = a - a.mean()
    bm = b - b.mean()
    denom = torch.linalg.norm(am) * torch.linalg.norm(bm)
    if float(denom) == 0.0:
        return float("nan")
    return float((am * bm).sum().item() / denom.item())


def _metrics(torch_t: torch.Tensor, tt_t: torch.Tensor) -> dict[str, float]:
    a, b = _flatten_pair(torch_t, tt_t)
    diff = a - b
    mse = float(torch.mean(diff * diff).item())
    rmse = float(math.sqrt(mse))
    mae = float(torch.mean(torch.abs(diff)).item())
    max_abs = float(torch.max(torch.abs(diff)).item())
    mean_abs = float(torch.mean(torch.abs(diff)).item())
    rel_l2 = float((torch.linalg.norm(diff) / (torch.linalg.norm(a) + 1e-12)).item())
    cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    pcc = _pcc(a, b)
    return {
        "pcc": pcc,
        "rmse": rmse,
        "mse": mse,
        "mae": mae,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel_l2": rel_l2,
        "cosine": cos,
    }


def _dump_suffix(iter_id: int | None, path: str | None) -> str:
    """Suffix for per-iteration, per-path dumps: _iter0_video, _iter0_action, etc."""
    if iter_id is None or path is None:
        return ""
    return f"_iter{iter_id}_{path}"


def _stage_pairs(
    out_dir: Path,
    iter_id: int | None = None,
    path: str | None = None,
) -> list[tuple[str, Path, Path]]:
    suffix = _dump_suffix(iter_id, path)
    legacy = iter_id is None and path is None
    pairs = [
        ("input", out_dir / f"input_torch{suffix}.pt", out_dir / f"input_tt{suffix}.pt"),
        (
            "text_hidden",
            out_dir / f"text_hidden_states_torch{suffix}.pt",
            out_dir / f"text_hidden_states_tt{suffix}.pt",
        ),
        ("temb", out_dir / f"temb_torch{suffix}.pt", out_dir / f"temb_tt{suffix}.pt"),
        (
            "timestep_proj",
            out_dir / f"timestep_proj_torch{suffix}.pt",
            out_dir / f"timestep_proj_tt{suffix}.pt",
        ),
    ]
    if legacy:
        block_range = range(1, 6)  # old naming: block_1..block_5
        block_fmt_torch = "transformer_torch_{}.pt"
        block_fmt_tt = "transformers_tt_{}.pt"
    else:
        block_range = range(NUM_BLOCKS)
        block_fmt_torch = f"transformer_torch_{{}}{suffix}.pt"
        block_fmt_tt = f"transformers_tt_{{}}{suffix}.pt"
    for i in block_range:
        pairs.append(
            (
                f"block_{i}",
                out_dir / block_fmt_torch.format(i),
                out_dir / block_fmt_tt.format(i),
            )
        )
    pairs.extend(
        [
            ("norm_out", out_dir / f"norm_out_torch{suffix}.pt", out_dir / f"norm_out_tt{suffix}.pt"),
        ]
    )
    # final_pre_rearrange and final_out_video only saved for video path
    if path != "action":
        pairs.append(
            (
                "final_pre_rearr",
                out_dir / f"final_pre_rearrange_torch{suffix}.pt",
                out_dir / f"final_pre_rearrange_tt{suffix}.pt",
            )
        )
        pairs.append(
            (
                "final_out_video",
                out_dir / f"final_out_video_torch{suffix}.pt",
                out_dir / f"final_out_video_tt{suffix}.pt",
            )
        )
    # final_out_action only for action path
    if path == "action":
        pairs.append(
            (
                "final_out_action",
                out_dir / f"final_out_action_torch{suffix}.pt",
                out_dir / f"final_out_action_tt{suffix}.pt",
            )
        )
    return pairs


def _discover_dump_pairs(out_dir: Path) -> list[tuple[int, str]]:
    """Discover (iter_id, path) from filenames like input_torch_iter0_video.pt."""
    seen: set[tuple[int, str]] = set()
    pattern = re.compile(r"input_torch_iter(\d+)_(video|action)\.pt")
    for p in out_dir.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if m:
            seen.add((int(m.group(1)), m.group(2)))
    return sorted(seen)


def _compare_one_pair(
    out_dir: Path,
    iter_id: int | None,
    path: str | None,
    min_pcc: float | None,
    max_rmse: float | None,
) -> int:
    pairs = _stage_pairs(out_dir, iter_id=iter_id, path=path)
    failures = 0
    print(
        f"{'stage':<14} {'torch_shape':<22} {'tt_shape':<22} {'pcc':>10} {'rmse':>10} {'mae':>10} {'max_abs':>10} {'rel_l2':>10} {'cosine':>10}"
    )
    print("-" * 120)

    for stage, torch_path, tt_path in pairs:
        if not torch_path.exists() or not tt_path.exists():
            print(f"{stage:<14} MISSING FILE(S): {torch_path.name} / {tt_path.name}")
            failures += 1
            continue

        torch_raw = _load_pt(torch_path)
        tt_raw = _load_pt(tt_path)

        torch_t, torch_note = _to_compare_tensor(torch_raw, f"{stage}.torch")
        tt_t, tt_note = _to_compare_tensor(tt_raw, f"{stage}.tt")

        if stage == "text_hidden":
            if isinstance(tt_t, torch.Tensor):
                if tt_t.dim() == 4 and tt_t.shape[2] == 1:
                    tt_t = tt_t.squeeze(2)
                if tt_t.dim() == 4 and tt_t.shape[1] == 1:
                    tt_t = tt_t.squeeze(1)
                while tt_t.dim() > 3 and tt_t.shape[0] == 1:
                    tt_t = tt_t.squeeze(0)
            if isinstance(torch_t, torch.Tensor):
                while torch_t.dim() > 3 and torch_t.shape[0] == 1:
                    torch_t = torch_t.squeeze(0)

        try:
            # final_out_video: TT returns (B, C, F, H, W), torch is (B, L*n, C).
            # Use exact inverse patch-order mapping to match reference sequence order.
            if stage == "final_out_video" and tt_t.ndim == 5 and torch_t.ndim == 3 and tt_t.numel() == torch_t.numel():
                tt_t = _video_bcfhw_to_reference_seq(tt_t, torch_t.shape)
            # text_hidden: TT (L,B,C) vs torch (B,L,C) — permute to match when same sizes
            if (
                stage == "text_hidden"
                and torch_t.shape != tt_t.shape
                and tt_t.ndim == 3
                and torch_t.ndim == 3
                and tt_t.numel() == torch_t.numel()
            ):
                ss_tt = tuple(sorted(int(d) for d in tt_t.shape))
                ss_torch = tuple(sorted(int(d) for d in torch_t.shape))
                if ss_tt == ss_torch:
                    available = list(range(3))
                    perm = []
                    for i in range(3):
                        target = int(torch_t.shape[i])
                        j = next((j for j in available if int(tt_t.shape[j]) == target), None)
                        if j is not None:
                            perm.append(j)
                            available.remove(j)
                    if len(perm) == 3:
                        tt_t = tt_t.permute(perm).contiguous()
            if torch_t.shape != tt_t.shape:
                tt_t = _align_tt_to_torch(tt_t, torch_t)
            m = _metrics(torch_t, tt_t)
        except Exception as exc:
            print(f"{stage:<14} ERROR: {exc}")
            failures += 1
            continue

        print(
            f"{stage:<14} {str(tuple(torch_t.shape)):<22} {str(tuple(tt_t.shape)):<22} "
            f"{m['pcc']:>10.6f} {m['rmse']:>10.6f} {m['mae']:>10.6f} {m['max_abs']:>10.6f} {m['rel_l2']:>10.6f} {m['cosine']:>10.6f}"
        )

        if torch_note:
            print(f"  note: {torch_note}")
        if tt_note:
            print(f"  note: {tt_note}")

        if min_pcc is not None and (math.isnan(m["pcc"]) or m["pcc"] < min_pcc):
            print(f"  FAIL: pcc {m['pcc']:.6f} < min_pcc {min_pcc:.6f}")
            failures += 1
        if max_rmse is not None and m["rmse"] > max_rmse:
            print(f"  FAIL: rmse {m['rmse']:.6f} > max_rmse {max_rmse:.6f}")
            failures += 1

    return failures


def compare_saved_transformer_dumps(
    out_dir: Path,
    iter_id: int | None = None,
    path: str | None = None,
    min_pcc: float | None = None,
    max_rmse: float | None = None,
) -> int:
    if iter_id is not None and path is not None:
        pairs_to_compare = [(iter_id, path)]
    else:
        pairs_to_compare = _discover_dump_pairs(out_dir)
        if not pairs_to_compare:
            # Fallback: legacy naming (no suffix)
            pairs_to_compare = [(None, None)]

    total_failures = 0
    print(f"Comparing saved tensors in: {out_dir}")
    for i, p in pairs_to_compare:
        print("-" * 120)
        if i is not None and p is not None:
            print(f" iter={i} path={p}")
        else:
            print(" legacy naming (no iter/path suffix)")
        print("-" * 120)
        total_failures += _compare_one_pair(out_dir, i, p, min_pcc, max_rmse)
        print("-" * 120)

    if total_failures == 0:
        print("Comparison complete: no failures.")
    else:
        print(f"Comparison complete: failures={total_failures}")
    return total_failures


def main() -> None:
    default_out_dir = Path(__file__).resolve().parent / "out_inference"
    parser = argparse.ArgumentParser(description="Compare saved torch/tt transformer block dumps.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help="Directory containing dump files (e.g. input_torch_iter0_video.pt).",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Compare only this diffusion step index. If omitted, discover and compare all (iter, path) pairs.",
    )
    parser.add_argument(
        "--path",
        type=str,
        choices=("video", "action"),
        default=None,
        help="Compare only this path. Requires --iter. If omitted, discover all pairs.",
    )
    parser.add_argument("--min-pcc", type=float, default=None, help="Optional threshold. Fail if PCC is below this.")
    parser.add_argument("--max-rmse", type=float, default=None, help="Optional threshold. Fail if RMSE is above this.")
    args = parser.parse_args()

    if (args.iter is None) != (args.path is None):
        parser.error("--iter and --path must be given together or both omitted.")

    failures = compare_saved_transformer_dumps(
        args.out_dir,
        iter_id=args.iter,
        path=args.path,
        min_pcc=args.min_pcc,
        max_rmse=args.max_rmse,
    )
    raise SystemExit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
