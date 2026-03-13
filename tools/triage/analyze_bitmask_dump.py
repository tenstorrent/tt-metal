#!/usr/bin/env python3
"""Offline analyzer for TT_DEBUG_BITMASK tensor dumps.

Usage:
  python tools/triage/analyze_bitmask_dump.py --dump-dir /tmp/tt_debug_bitmask
  python tools/triage/analyze_bitmask_dump.py --dump-dir /tmp/tt_debug_bitmask --step 1
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


FILE_RE = re.compile(r"^step(?P<step>\d+?)_(?P<name>.+?)(?:_shard(?P<shard>\d+))?\.pt$")

NAMES = [
    "packed_bitmask_host",
    "packed_bitmask_device",
    "bitmask_to_broadcast",
    "broadcast_unpacked_rshift",
    "broadcast_unpacked_and1",
    "unpacked_bitmask_reshape",
    "converted_bitmask_tile",
    "unpacked_penalty_mask",
    "logits_before_mask",
    "logits_after_mask",
]


@dataclass
class StepShard:
    tensors: Dict[str, torch.Tensor]
    host_packed: Optional[torch.Tensor]


def load_dump_dir(dump_dir: Path) -> Dict[int, Dict[int, StepShard]]:
    by_step_shard: Dict[int, Dict[int, StepShard]] = {}
    host_by_step: Dict[int, torch.Tensor] = {}

    for p in sorted(dump_dir.glob("*.pt")):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        step = int(m.group("step"))
        name = m.group("name")
        shard_s = m.group("shard")
        tensor = torch.load(p, map_location="cpu")

        if name == "packed_bitmask_host":
            host_by_step[step] = tensor
            continue

        if shard_s is None:
            # Only packed host is expected without shard suffix.
            continue
        shard = int(shard_s)

        by_step_shard.setdefault(step, {})
        if shard not in by_step_shard[step]:
            by_step_shard[step][shard] = StepShard(tensors={}, host_packed=None)
        by_step_shard[step][shard].tensors[name] = tensor

    for step, shard_map in by_step_shard.items():
        host = host_by_step.get(step)
        for shard in shard_map:
            shard_map[shard].host_packed = host

    return by_step_shard


def tensor_stats(t: torch.Tensor) -> str:
    s = f"shape={tuple(t.shape)} dtype={t.dtype}"
    if t.numel() == 0:
        return s + " empty"
    if t.is_floating_point():
        finite = t[torch.isfinite(t)]
        if finite.numel() > 0:
            s += f" min={finite.min().item():.6g} max={finite.max().item():.6g}"
        else:
            s += " min/max=non-finite-only"
    else:
        s += f" min={t.min().item()} max={t.max().item()}"
    uniq = torch.unique(t)
    if uniq.numel() <= 12:
        s += f" unique={uniq.tolist()}"
    else:
        s += f" unique_count={uniq.numel()}"
    return s


def first_mismatch(a: torch.Tensor, b: torch.Tensor, atol: float = 0.0, rtol: float = 0.0) -> Optional[str]:
    # Compare with trailing dims flattened so tensors like
    # (B, W, 32) can be directly compared against (B, W*32).
    if a.ndim >= 2:
        a = a.reshape(a.shape[0], -1)
    else:
        a = a.reshape(1, -1)
    if b.ndim >= 2:
        b = b.reshape(b.shape[0], -1)
    else:
        b = b.reshape(1, -1)

    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a2 = a[:h, :w]
    b2 = b[:h, :w]
    if a2.is_floating_point() or b2.is_floating_point():
        close = torch.isclose(a2.to(torch.float32), b2.to(torch.float32), atol=atol, rtol=rtol)
        neq = ~close
    else:
        neq = a2 != b2
    if not torch.any(neq):
        return None
    idx = torch.nonzero(neq, as_tuple=False)[0].tolist()
    got = a2[idx[0], idx[1]].item()
    exp = b2[idx[0], idx[1]].item()
    return f"first mismatch idx={idx} got={got} expected={exp}"


def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    a = a.reshape(a.shape[0], -1) if a.ndim >= 2 else a.reshape(1, -1)
    b = b.reshape(b.shape[0], -1) if b.ndim >= 2 else b.reshape(1, -1)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    d = (a[:h, :w].to(torch.float32) - b[:h, :w].to(torch.float32)).abs()
    if d.numel() == 0:
        return 0.0, 0.0
    finite = d[torch.isfinite(d)]
    if finite.numel() == 0:
        return 0.0, 0.0
    return finite.max().item(), finite.mean().item()


def expected01_from_packed(packed_local: torch.Tensor) -> torch.Tensor:
    ar = torch.arange(32, dtype=torch.int32, device=packed_local.device)
    return ((packed_local[:, :, None] >> ar[None, None, :]) & 1).reshape(packed_local.shape[0], -1)


def _join_shards(tensors_by_shard: Dict[int, torch.Tensor], shard_ids: list[int]) -> Optional[torch.Tensor]:
    if not shard_ids:
        return None
    ts = [tensors_by_shard[s] for s in shard_ids if s in tensors_by_shard]
    if len(ts) != len(shard_ids):
        return None
    ndim = ts[0].ndim
    if ndim in (2, 3):
        dim = 1
    elif ndim == 4:
        dim = 3
    else:
        # Unexpected shape; keep per-shard behavior by not joining.
        return None
    return torch.cat(ts, dim=dim)


def _chunked_groups(shard_ids: list[int], group_size: int) -> list[list[int]]:
    shard_ids = sorted(shard_ids)
    if group_size <= 1 or len(shard_ids) < group_size or len(shard_ids) % group_size != 0:
        return [[s] for s in shard_ids]
    return [shard_ids[i : i + group_size] for i in range(0, len(shard_ids), group_size)]


def _infer_group_size(host_packed: Optional[torch.Tensor], packed_by_shard: Dict[int, torch.Tensor]) -> int:
    if host_packed is None or not packed_by_shard:
        return 1
    first = packed_by_shard[sorted(packed_by_shard.keys())[0]]
    if host_packed.ndim != 2 or first.ndim != 2:
        return 1
    local_w = first.shape[1]
    if local_w <= 0 or host_packed.shape[1] % local_w != 0:
        return 1
    return host_packed.shape[1] // local_w


def analyze_step_grouped(
    step: int,
    shard_map: Dict[int, StepShard],
    verbose: bool,
    show_ok: bool,
    ignore_logits_nan: bool,
    check_all_replicas: bool,
) -> int:
    errs = 0
    host_packed: Optional[torch.Tensor] = None
    by_name: Dict[str, Dict[int, torch.Tensor]] = {}
    for shard, data in shard_map.items():
        if data.host_packed is not None:
            host_packed = data.host_packed
        for name, tensor in data.tensors.items():
            by_name.setdefault(name, {})[shard] = tensor

    packed_by_shard = by_name.get("packed_bitmask_device", {})
    if not packed_by_shard:
        print(f"\n=== step={step} ===\n  ! missing packed_bitmask_device")
        return 1

    # Llama-galaxy bitmask uses ShardTensor2dMesh(dims=(-1, None), mesh_shape=(8,4)):
    # - mesh row dim shards tensor last dim
    # - mesh col dim replicates
    # get_device_tensors() returns coords in row-major mesh order.
    shard_ids_all = sorted(packed_by_shard.keys())
    local_w = packed_by_shard[shard_ids_all[0]].shape[1]
    if host_packed is not None and host_packed.ndim == 2 and local_w > 0 and host_packed.shape[1] % local_w == 0:
        mesh_rows = host_packed.shape[1] // local_w
    else:
        # Fallback: old grouping heuristic.
        mesh_rows = _infer_group_size(host_packed, packed_by_shard)
    mesh_cols = len(shard_ids_all) // mesh_rows if mesh_rows > 0 else 1
    if mesh_rows <= 0 or mesh_cols <= 0 or mesh_rows * mesh_cols != len(shard_ids_all):
        print(
            f"\n=== step={step} ===\n"
            f"  ! invalid shard mesh inference rows={mesh_rows} cols={mesh_cols} "
            f"num_shards={len(shard_ids_all)}"
        )
        return 1

    # One logical group per step; reassemble via canonical representative col=0 for each row.
    groups = [shard_ids_all]

    for group_idx, shard_ids in enumerate(groups):
        lines = [f"\n=== step={step} group={group_idx} shards={shard_ids[0]}..{shard_ids[-1]} ==="]
        local_errs = 0

        # Strict mesh-semantics ordering:
        # row-major shard list layout is [r0c0,r0c1..r0cC-1, r1c0...]
        # canonical logical reconstruction uses c0 from each row in row order.
        join_order = [shard_ids[r * mesh_cols] for r in range(mesh_rows)]
        if verbose:
            lines.append(f"  inferred mesh rows={mesh_rows} cols={mesh_cols}; " f"join_order(c0-per-row)={join_order}")

        # Replica consistency check on packed stage (must be exact across cols).
        def _check_stage_replicas(stage_name: str) -> None:
            nonlocal local_errs
            stage = by_name.get(stage_name, {})
            if not stage:
                return
            for r in range(mesh_rows):
                ref = stage[shard_ids[r * mesh_cols]]
                ref = ref.to(torch.float32) if ref.is_floating_point() else ref.to(torch.int32)
                for c in range(1, mesh_cols):
                    sid = shard_ids[r * mesh_cols + c]
                    cur = stage[sid]
                    cur = cur.to(torch.float32) if cur.is_floating_point() else cur.to(torch.int32)
                    msg = first_mismatch(cur, ref, atol=2.0e6 if ref.is_floating_point() else 0.0, rtol=0.0)
                    if msg:
                        lines.append(f"  ! {stage_name} replica mismatch row={r} col={c} shard={sid} -> {msg}")
                        local_errs += 1
                        break

        _check_stage_replicas("packed_bitmask_device")
        if check_all_replicas:
            for stage_name in (
                "bitmask_to_broadcast",
                "broadcast_unpacked_rshift",
                "broadcast_unpacked_and1",
                "unpacked_bitmask_reshape",
                "converted_bitmask_tile",
                "unpacked_penalty_mask",
            ):
                _check_stage_replicas(stage_name)

        joined: Dict[str, torch.Tensor] = {}
        for name in NAMES:
            if name == "packed_bitmask_host":
                continue
            if name in by_name:
                j = _join_shards(by_name[name], join_order)
                if j is not None:
                    joined[name] = j
                    if verbose:
                        lines.append(f"  {name}: {tensor_stats(j)}")

        packed = joined.get("packed_bitmask_device")
        if packed is None:
            lines.append("  ! failed to build joined packed_bitmask_device")
            print("\n".join(lines))
            errs += 1
            continue

        packed_i32 = packed.to(torch.int32)
        torch_bitmask_to_broadcast = packed_i32[:, :, None]
        ar = torch.arange(32, dtype=torch.int32, device=packed_i32.device)
        torch_rshift = torch_bitmask_to_broadcast >> ar[None, None, :]
        torch_and1 = torch_rshift & 1
        torch_unpacked = torch_and1.reshape(packed_i32.shape[0], -1)
        torch_penalty = torch.where(
            torch_unpacked != 0,
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(-1e9, dtype=torch.float32),
        )

        if host_packed is not None:
            msg = first_mismatch(packed_i32, host_packed.to(torch.int32))
            if msg:
                lines.append(f"  ! packed_bitmask_device(reassembled) != packed_bitmask_host -> {msg}")
                local_errs += 1
            elif verbose:
                lines.append("  packed_bitmask_device(reassembled) matches packed_bitmask_host")

        stage_expected = {
            "bitmask_to_broadcast": torch_bitmask_to_broadcast,
            "broadcast_unpacked_rshift": torch_rshift,
            "broadcast_unpacked_and1": torch_and1,
            "unpacked_bitmask_reshape": torch_unpacked,
            "converted_bitmask_tile": torch_unpacked,
            "unpacked_penalty_mask": torch_penalty,
        }
        for name, exp in stage_expected.items():
            got = joined.get(name)
            if got is None:
                continue
            if exp.is_floating_point():
                msg = first_mismatch(got.to(torch.float32), exp.to(torch.float32))
            else:
                msg = first_mismatch(got.to(torch.int32), exp.to(torch.int32))
            if msg:
                lines.append(f"  ! {name} != torch_ref -> {msg}")
                local_errs += 1
            elif verbose:
                lines.append(f"  {name} matches torch_ref")

        if "logits_before_mask" in joined and "logits_after_mask" in joined:
            before = joined["logits_before_mask"].to(torch.float32)
            after = joined["logits_after_mask"].to(torch.float32)
            h = min(before.shape[-2], torch_penalty.shape[-2]) if before.ndim >= 2 else None
            w = min(before.shape[-1], torch_penalty.shape[-1]) if before.ndim >= 1 else None
            if h is not None and w is not None and before.ndim == 4:
                delta = after[0, 0, :h, :w] - before[0, 0, :h, :w]
                penalty_slice = torch_penalty[:h, :w]
                if ignore_logits_nan:
                    valid = torch.isfinite(delta) & torch.isfinite(penalty_slice)
                    delta = torch.where(valid, delta, torch.zeros_like(delta))
                    penalty_slice = torch.where(valid, penalty_slice, torch.zeros_like(penalty_slice))
                msg = first_mismatch(delta, penalty_slice, atol=2.0e6, rtol=0.0)
                if msg:
                    max_err, mean_err = max_abs_err(delta, penalty_slice)
                    lines.append(
                        f"  ! logits delta != penalty_ref -> {msg} "
                        f"(max_abs_err={max_err:.6g}, mean_abs_err={mean_err:.6g})"
                    )
                    local_errs += 1
                elif verbose:
                    max_err, mean_err = max_abs_err(delta, penalty_slice)
                    lines.append(
                        "  logits_after - logits_before matches penalty_ref "
                        f"(within tol, max_abs_err={max_err:.6g}, mean_abs_err={mean_err:.6g})"
                    )

        if local_errs > 0 or verbose or show_ok:
            if local_errs == 0 and not verbose and show_ok:
                lines.append("  OK")
            print("\n".join(lines))
        errs += local_errs
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", type=Path, required=True, help="Directory containing stepXXXX_*.pt files")
    ap.add_argument("--step", type=int, default=None, help="Analyze only one step")
    ap.add_argument("--verbose", action="store_true", help="Print full per-shard stats")
    ap.add_argument("--show-ok", action="store_true", help="Print one line for passing shards")
    ap.add_argument(
        "--per-shard",
        action="store_true",
        help="Use legacy shard-by-shard checks instead of joined/grouped checks",
    )
    ap.add_argument(
        "--no-ignore-logits-nan",
        action="store_true",
        help="Treat non-finite logits delta positions as mismatches",
    )
    ap.add_argument(
        "--check-all-replicas",
        action="store_true",
        help="Validate row/col replica consistency for all dumped bitmask stages",
    )
    args = ap.parse_args()

    by_step = load_dump_dir(args.dump_dir)
    if not by_step:
        print("No dump files found.")
        return

    total_errs = 0
    steps = sorted(by_step.keys())
    if args.step is not None:
        steps = [s for s in steps if s == args.step]
    if not steps:
        print(f"No matching step for --step={args.step}")
        return

    for step in steps:
        if args.per_shard:
            for shard in sorted(by_step[step].keys()):
                total_errs += analyze_step_grouped(
                    step,
                    {shard: by_step[step][shard]},
                    verbose=args.verbose,
                    show_ok=args.show_ok,
                    ignore_logits_nan=not args.no_ignore_logits_nan,
                    check_all_replicas=args.check_all_replicas,
                )
        else:
            total_errs += analyze_step_grouped(
                step,
                by_step[step],
                verbose=args.verbose,
                show_ok=args.show_ok,
                ignore_logits_nan=not args.no_ignore_logits_nan,
                check_all_replicas=args.check_all_replicas,
            )

    print(f"\nDone. Total issues found: {total_errs}")


if __name__ == "__main__":
    main()
