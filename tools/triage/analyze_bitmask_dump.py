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
from typing import Dict, Optional

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


def first_mismatch(a: torch.Tensor, b: torch.Tensor) -> Optional[str]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1]) if a.ndim >= 2 and b.ndim >= 2 else None
    if w is not None:
        a2 = a[:h, :w]
        b2 = b[:h, :w]
    else:
        n = min(a.numel(), b.numel())
        a2 = a.reshape(-1)[:n]
        b2 = b.reshape(-1)[:n]
    neq = a2 != b2
    if not torch.any(neq):
        return None
    idx = torch.nonzero(neq, as_tuple=False)[0].tolist()
    got = a2[tuple(idx)].item() if isinstance(idx, list) else a2[idx].item()
    exp = b2[tuple(idx)].item() if isinstance(idx, list) else b2[idx].item()
    return f"first mismatch idx={idx} got={got} expected={exp}"


def find_host_slice_match(host: torch.Tensor, packed_local: torch.Tensor) -> Optional[int]:
    if host.ndim != 2 or packed_local.ndim != 2:
        return None
    if packed_local.shape[0] != host.shape[0] or packed_local.shape[1] > host.shape[1]:
        return None
    shard_w = packed_local.shape[1]
    for col_start in range(0, host.shape[1] - shard_w + 1, shard_w):
        if torch.equal(packed_local, host[:, col_start : col_start + shard_w]):
            return col_start
    return None


def expected01_from_packed(packed_local: torch.Tensor) -> torch.Tensor:
    ar = torch.arange(32, dtype=torch.int32, device=packed_local.device)
    return ((packed_local[:, :, None] >> ar[None, None, :]) & 1).reshape(packed_local.shape[0], -1)


def analyze_step_shard(step: int, shard: int, data: StepShard) -> int:
    errs = 0
    t = data.tensors
    print(f"\n=== step={step} shard={shard} ===")
    for name in NAMES:
        if name in t:
            print(f"  {name}: {tensor_stats(t[name])}")

    packed = t.get("packed_bitmask_device")
    if packed is None:
        print("  ! missing packed_bitmask_device")
        return 1

    if data.host_packed is not None:
        col = find_host_slice_match(data.host_packed.to(torch.int32), packed.to(torch.int32))
        if col is None:
            print("  ! packed shard does NOT match any host packed slice")
            errs += 1
        else:
            print(f"  host slice match at col_start={col}")

    expected01 = expected01_from_packed(packed.to(torch.int32))
    expected_penalty = torch.where(
        expected01 != 0,
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(-1e9, dtype=torch.float32),
    )

    for name in ("broadcast_unpacked_and1", "unpacked_bitmask_reshape", "converted_bitmask_tile"):
        if name in t:
            got = t[name].to(torch.int32)
            msg = first_mismatch(got, expected01)
            if msg:
                print(f"  ! {name} != expected01 -> {msg}")
                errs += 1
            else:
                print(f"  {name} matches expected01")

    if "unpacked_penalty_mask" in t:
        got = t["unpacked_penalty_mask"].to(torch.float32)
        msg = first_mismatch(got, expected_penalty)
        if msg:
            print(f"  ! unpacked_penalty_mask != expected_penalty -> {msg}")
            errs += 1
        else:
            print("  unpacked_penalty_mask matches expected_penalty")

    if "logits_before_mask" in t and "logits_after_mask" in t and "unpacked_penalty_mask" in t:
        before = t["logits_before_mask"].to(torch.float32)
        after = t["logits_after_mask"].to(torch.float32)
        penalty = t["unpacked_penalty_mask"].to(torch.float32)
        h = min(before.shape[-2], penalty.shape[-2]) if before.ndim >= 2 else None
        w = min(before.shape[-1], penalty.shape[-1]) if before.ndim >= 1 else None
        if h is not None and w is not None and before.ndim == 4 and penalty.ndim == 2:
            delta = after[0, 0, :h, :w] - before[0, 0, :h, :w]
            msg = first_mismatch(delta, penalty[:h, :w])
            if msg:
                print(f"  ! logits delta != penalty -> {msg}")
                errs += 1
            else:
                print("  logits_after - logits_before matches penalty")

    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", type=Path, required=True, help="Directory containing stepXXXX_*.pt files")
    ap.add_argument("--step", type=int, default=None, help="Analyze only one step")
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
        for shard in sorted(by_step[step].keys()):
            total_errs += analyze_step_shard(step, shard, by_step[step][shard])

    print(f"\nDone. Total issues found: {total_errs}")


if __name__ == "__main__":
    main()
