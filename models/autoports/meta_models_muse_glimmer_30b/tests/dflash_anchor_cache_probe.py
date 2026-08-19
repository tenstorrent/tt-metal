# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Does the persistent anchor K/V cache compute the same drafter forward as the padded path?

The cache replaces "re-upload and re-project the whole context every iteration" with
``O(new rows)`` work, so the question is whether the arithmetic survives being split across
``fill_cache`` writes at tile-aligned rows.  Graded three ways, none of which is token
equality (F2 rules that out for this port):

* **PCC and argmax** of ``forward_anchored`` against ``forward`` at the exact context length,
  which is the reference the 13/13 device PCC actually graded.
* **Accumulation**, i.e. appending the context in 32-row tiles with varying committed counts
  the way a real generation does, rather than in one shot.  This is where a row/position
  mis-mapping would show up and a single-shot test would not.
* **Program cache stability** across that accumulation -- the direct F6 regression test.  F6
  measured 82x from recompilation because both the delta length and the cache length were
  shapes; if either still is, the entry count grows here.

Drafter only: 5.11 GB and no 30B target, so this runs in about a minute.
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import dflash_checkpoint as R
from models.autoports.meta_models_muse_glimmer_30b.tt.dflash_drafter import (
    DFlashAnchorCache,
    DFlashDrafter,
    config_from_hf,
    context_bucket,
)

TILE = 32


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.flatten().to(torch.float32), b.flatten().to(torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    return float((x @ y) / denom) if float(denom) else float("nan")


def program_cache_entries(mesh) -> int:
    fn = getattr(mesh, "num_program_cache_entries", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:  # noqa: BLE001 - probe only
            pass
    return -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-len", type=int, default=67)
    parser.add_argument(
        "--commits",
        default="3,13,2,2,9,4,2,2",
        help="committed counts to replay, as a real generation produces them",
    )
    parser.add_argument("--mesh", default="1,4")
    parser.add_argument(
        "--capacity",
        type=int,
        default=0,
        help="override the cache capacity, to reproduce a real run's width (0 = derive it)",
    )
    parser.add_argument(
        "--time-iterations",
        type=int,
        default=0,
        help=(
            "after the correctness check, time this many propose+append iterations. The runner "
            "at OSL 256 sits CPU-bound for minutes, so the question is whether the drafter path "
            "itself is slow or whether it is something the runner adds around it."
        ),
    )
    args = parser.parse_args()

    commits = [int(c) for c in args.commits.split(",") if c.strip()]
    rows, cols = (int(v) for v in args.mesh.split(","))

    hf_config = R.draft_config()
    config = config_from_hf(hf_config)
    block = config.block_size
    fan_in = config.context_fan_in

    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols), trace_region_size=0)
    try:
        drafter = DFlashDrafter.from_state_dict(
            R.draft_state_dict(),
            hf_config=hf_config,
            mesh_device=mesh,
            weight_dtype=ttnn.bfloat8_b,
            activation_dtype=ttnn.bfloat16,
        )
        ttnn.synchronize_device(mesh)

        torch.manual_seed(0)
        total = args.prompt_len + sum(commits)
        capacity = args.capacity or context_bucket(total + block)
        print(f"prompt {args.prompt_len}, commits {commits} -> {total} anchors; capacity {capacity}")

        # Stand-in for the target's tapped hidden states, at the real width.
        taps_host = torch.normal(0.0, 0.02, (1, 1, total, fan_in), dtype=torch.float32)
        noise_host = torch.normal(0.0, 0.02, (1, 1, block, config.hidden_size), dtype=torch.float32)

        def up(t: torch.Tensor) -> ttnn.Tensor:
            return ttnn.from_torch(
                t.to(torch.bfloat16),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def to_host(t: ttnn.Tensor) -> torch.Tensor:
            return ttnn.to_torch(ttnn.get_device_tensors(t)[0])

        # ---- reference: the shipped exact-length path, context in one shot -------------
        noise_ref = up(noise_host)
        ctx_ref = up(taps_host)
        ref = drafter.forward(
            noise_ref,
            ctx_ref,
            position_ids=torch.arange(total + block),
        )
        ttnn.synchronize_device(mesh)
        ref_host = to_host(ref)
        ttnn.deallocate(ref)
        ttnn.deallocate(ctx_ref)

        # ---- candidate: accumulate into the cache the way a generation does ------------
        cache = DFlashAnchorCache(
            config=config,
            mesh_device=mesh,
            capacity=capacity,
            num_layers=len(drafter.layers),
        )
        entries = [program_cache_entries(mesh)]

        # Iteration 0 carries the whole prompt, padded up to a tile.
        def append(start: int, count: int) -> None:
            """Append ``count`` anchors starting at absolute position ``start``.

            The write is tile-aligned and tile-sized, which is what keeps every shape constant;
            rows past ``count`` inside that tile are stale and excluded by ``kv_valid``.
            """
            padded = ((count + TILE - 1) // TILE) * TILE
            dest = (start // TILE) * TILE
            lead = start - dest
            piece = torch.zeros((1, 1, padded, fan_in), dtype=torch.float32)
            piece[0, 0, : lead + count] = taps_host[0, 0, dest : dest + lead + count]
            tt_piece = up(piece)
            drafter.append_anchors(
                tt_piece,
                cache=cache,
                dest_row=dest,
                positions=torch.arange(dest, dest + padded),
            )
            ttnn.deallocate(tt_piece)

        append(0, args.prompt_len)
        cache.note_committed(args.prompt_len)
        entries.append(program_cache_entries(mesh))

        position = args.prompt_len
        for n in commits:
            append(position, n)
            cache.note_committed(n)
            position += n
            entries.append(program_cache_entries(mesh))

        noise_cand = up(noise_host)
        got = drafter.forward_anchored(noise_cand, cache=cache, noise_start=total)
        ttnn.synchronize_device(mesh)
        got_host = to_host(got)
        ttnn.deallocate(got)

        a = ref_host.reshape(-1, config.hidden_size)[:block]
        b = got_host.reshape(-1, config.hidden_size)[:block]
        agree = int((a.argmax(dim=-1) == b.argmax(dim=-1)).sum())

        print()
        # Two warm-up shapes are expected and are not churn: the prompt append is a one-off
        # row count, and the first 32-row append compiles the steady-state programs.  What must
        # not grow is everything after that -- entries[2:] -- because those appends differ only
        # in destination row, which fill_cache takes as a runtime argument.
        steady = entries[2:]
        print(f"program cache entries : {entries}")
        print(f"  warm-up (prompt, first tile): {entries[0]} -> {entries[1]} -> {entries[2]}")
        print(
            f"  steady state stable   : {len(set(steady)) <= 1}  "
            f"({len(steady)} appends, entries {min(steady)}..{max(steady)})"
        )
        print(f"PCC vs exact-length   : {pcc(a, b):.6f}")
        print(f"max abs delta         : {float((a.to(torch.float32) - b.to(torch.float32)).abs().max()):.5f}")
        print(f"argmax agreement      : {agree}/{block}")
        print(f"all finite            : {bool(torch.isfinite(b.to(torch.float32)).all())}")

        if args.time_iterations:
            print()
            print(f"timing {args.time_iterations} propose+append iterations at capacity {capacity}")
            per_iter = []
            entries_seen = [program_cache_entries(mesh)]
            for i in range(args.time_iterations):
                start = time.perf_counter()
                noise_i = up(noise_host)
                out_i = drafter.forward_anchored(noise_i, cache=cache, noise_start=cache.valid_len)
                # One 32-row append, exactly what the runner does after a verify.
                dest = (cache.valid_len // TILE) * TILE
                piece = torch.zeros((1, 1, TILE, fan_in), dtype=torch.float32)
                tt_piece = up(piece)
                drafter.append_anchors(tt_piece, cache=cache, dest_row=dest, positions=torch.arange(dest, dest + TILE))
                ttnn.deallocate(tt_piece)
                ttnn.deallocate(out_i)
                ttnn.synchronize_device(mesh)
                per_iter.append((time.perf_counter() - start) * 1000.0)
                cache.note_committed(1)
                entries_seen.append(program_cache_entries(mesh))
            print(f"  per-iteration ms : {[round(v, 1) for v in per_iter]}")
            warm = per_iter[1:] or per_iter
            print(f"  first / median warm: {per_iter[0]:.1f} ms / {sorted(warm)[len(warm)//2]:.1f} ms")
            print(
                f"  program cache     : {entries_seen[0]} -> {entries_seen[-1]}  "
                f"(stable after first: {len(set(entries_seen[2:])) <= 1})"
            )

        cache.release()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
