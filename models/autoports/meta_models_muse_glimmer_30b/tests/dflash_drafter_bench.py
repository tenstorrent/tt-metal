# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone perf harness for the TTNN DFlash drafter.

The drafter call is the whole DFlash bottleneck (~1.09 s/call against an 88 ms
target verify forward), so it needs to be iterated on quickly.  Loading the 30 B
target to measure it would cost minutes per attempt, and none of the target is
needed: the drafter's two inputs are a ``[1, 1, block, hidden]`` noise embedding
and a ``[1, 1, T, 5*hidden]`` context tensor, both of which can be synthesised.

So this harness loads **only** the 5.11 GB drafter and drives
``forward_cached`` with the shape sequence a real generation produces, which is
what makes a measure-fix-measure loop take a minute instead of twenty.

Two things it deliberately does:

* **Replays real context lengths.**  Iteration 0 passes the whole prompt; every
  later iteration passes only ``n_matches + 1`` newly accepted rows.  Cost per
  call is therefore *not* constant -- the cache grows -- so a single-shape
  benchmark would mismeasure it.  ``--matches`` defaults to the sequence a
  measured device run produced.
* **Attributes time per ttnn op.**  ``--breakdown`` wraps the ttnn entry points
  the drafter uses and synchronises after each, which perturbs the total but is
  the only way to see *which* op owns the second.  Read the totals from a run
  *without* ``--breakdown``.

Usage::

    # attribution: where does the 1.09 s go?
    python .../tests/dflash_drafter_bench.py --breakdown

    # clean wall-clock per call, 1x4 mesh, bf16 weights
    python .../tests/dflash_drafter_bench.py --mesh 1x4 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import dflash_checkpoint as R
from models.autoports.meta_models_muse_glimmer_30b.tt.dflash_drafter import (
    DFlashDrafter,
    DFlashDrafterCache,
    config_from_hf,
)

#: Per-block accepted counts from the first working device run (commit dd9505fd35b).
#: Replayed so the context-length sequence -- and therefore the cache growth --
#: matches a real generation rather than a flat guess.
DEFAULT_MATCHES = (2, 13, 3, 10, 2, 1, 0, 2, 0, 1, 2)

DTYPES = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "bfloat4_b": ttnn.bfloat4_b}


def assert_expected_source(expected_root: str | None = None) -> str:
    """Fail loudly if the code under test is not the tree we think it is.

    ``models`` is a **namespace package with several roots**.  This venv's
    ``ttnn-custom.pth`` unconditionally puts a second tt-metal checkout on
    ``sys.path``, so the same import can resolve into a different branch -- or into
    ``tt-train/sources/ttml/models``, which is a different project entirely.  The
    dangerous case is not the loud ``ModuleNotFoundError``; it is silently
    benchmarking a *mixture* of an optimised drafter and another branch's model
    code, which looks like a clean result.

    ``expected_root`` defaults to ``$DFLASH_EXPECTED_ROOT``; checking is skipped
    only when neither is set.
    """
    import models.autoports.meta_models_muse_glimmer_30b as package

    resolved = os.path.realpath(list(package.__path__)[0])
    expected = expected_root or os.environ.get("DFLASH_EXPECTED_ROOT")
    if expected and not resolved.startswith(os.path.realpath(expected)):
        raise RuntimeError(
            f"model code resolved to {resolved}, which is not under {expected}. "
            "Set PYTHONPATH to the tree under test so it precedes the .pth entries; "
            "measuring a mixture of two checkouts would silently look like a valid result."
        )
    return resolved


class OpTimer:
    """Wrap the ttnn ops the drafter calls, synchronising after each.

    Attribution only.  Every wrapped call becomes a device barrier, so the total
    here is an upper bound and the *shares* are the signal, not the sum.
    """

    WRAPPED = (
        "linear",
        "matmul",
        "mul",
        "add",
        "neg",
        "concat",
        "softmax",
        "permute",
        "reshape",
        "rms_norm",
        "repeat_interleave",
        "silu",
        "slice",
        "pad",
        "from_torch",
        "to_torch",
        "transpose",
    )

    def __init__(self, mesh_device: ttnn.MeshDevice) -> None:
        self.mesh_device = mesh_device
        self.totals: dict[str, float] = collections.defaultdict(float)
        self.counts: dict[str, int] = collections.defaultdict(int)
        self._saved: dict[str, object] = {}

    def _wrap(self, name: str, fn):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            out = fn(*args, **kwargs)
            ttnn.synchronize_device(self.mesh_device)
            self.totals[name] += time.perf_counter() - start
            self.counts[name] += 1
            return out

        return wrapper

    def __enter__(self) -> "OpTimer":
        for name in self.WRAPPED:
            fn = getattr(ttnn, name, None)
            if fn is None:
                continue
            self._saved[name] = fn
            setattr(ttnn, name, self._wrap(name, fn))
        return self

    def __exit__(self, *exc) -> None:
        for name, fn in self._saved.items():
            setattr(ttnn, name, fn)
        self._saved.clear()

    def report(self) -> list[tuple[str, float, int]]:
        rows = [(name, seconds, self.counts[name]) for name, seconds in self.totals.items()]
        rows.sort(key=lambda row: row[1], reverse=True)
        return rows


def _upload(tensor: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.reshape(1, 1, *tensor.shape[-2:]).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def parse_mesh(text: str) -> ttnn.MeshShape:
    rows, _, cols = text.partition("x")
    return ttnn.MeshShape(int(rows), int(cols))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="1x4", help="mesh shape, e.g. 1x4 or 1x1")
    parser.add_argument("--dtype", default="bfloat8_b", choices=sorted(DTYPES))
    parser.add_argument("--activation-dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--prompt-len", type=int, default=67, help="iteration-0 context length")
    parser.add_argument(
        "--matches",
        default=",".join(str(m) for m in DEFAULT_MATCHES),
        help="per-iteration accepted counts to replay",
    )
    parser.add_argument("--warmup", type=int, default=1, help="untimed leading iterations")
    parser.add_argument("--breakdown", action="store_true", help="per-op attribution (perturbs the total)")
    parser.add_argument(
        "--fixed-shape",
        type=int,
        default=0,
        metavar="CTX",
        help=(
            "diagnostic: run every iteration at exactly CTX context rows against a FRESH cache, "
            "so all calls are shape-identical. Isolates ttnn program-cache compilation from real work: "
            "if call 1 is slow and calls 2+ collapse, the cost is recompilation driven by shape churn, "
            "not arithmetic."
        ),
    )
    parser.add_argument("--iterations", type=int, default=0, help="override iteration count (--fixed-shape only)")
    parser.add_argument(
        "--mode",
        default="cached",
        choices=["cached", "uncached_padded"],
        help=(
            "cached: today's forward_cached, whose context/cache lengths change every call. "
            "uncached_padded: the PCC-validated uncached forward over a FIXED --context-cap rows, "
            "zero-padded, so every call is shape-identical. Costs O(cap) instead of O(new rows) but "
            "hits the program cache; this is the candidate design."
        ),
    )
    parser.add_argument("--context-cap", type=int, default=128, help="padded context rows for uncached_padded")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(f"model code: {assert_expected_source()}", flush=True)
    matches = [int(m) for m in args.matches.split(",") if m.strip()]
    mesh_shape = parse_mesh(args.mesh)

    hf_config = R.draft_config()
    config = config_from_hf(hf_config)
    block = config.block_size

    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, trace_region_size=0)
    try:
        load_start = time.perf_counter()
        drafter = DFlashDrafter.from_state_dict(
            R.draft_state_dict(),
            hf_config=hf_config,
            mesh_device=mesh_device,
            weight_dtype=DTYPES[args.dtype],
            activation_dtype=DTYPES[args.activation_dtype],
        )
        ttnn.synchronize_device(mesh_device)
        load_seconds = time.perf_counter() - load_start
        print(f"drafter loaded in {load_seconds:.1f}s  mesh={args.mesh}  weights={args.dtype}", flush=True)

        noise_host = torch.normal(0.0, 0.02, (1, block, config.hidden_size), dtype=torch.float32)

        if args.fixed_shape:
            count = args.iterations or len(matches)
            context_lens = [args.fixed_shape] * count
        else:
            # Replay the real shape sequence: iteration 0 carries the whole prompt,
            # later iterations carry only the newly accepted rows.
            context_lens = [args.prompt_len] + [m + 1 for m in matches[:-1]]

        per_call: list[float] = []
        timer_ctx = OpTimer(mesh_device) if args.breakdown else contextlib.nullcontext()
        cache = DFlashDrafterCache(config.num_hidden_layers)
        anchor_pos = args.prompt_len
        context_start = 0

        with timer_ctx as timer:
            for index, context_len in enumerate(context_lens):
                if args.fixed_shape:
                    # Fresh cache and fixed positions: every call is shape-identical, so
                    # anything still slow after call 1 is real work rather than compilation.
                    cache.release()
                    cache = DFlashDrafterCache(config.num_hidden_layers)
                    context_start, anchor_pos = 0, context_len
                context_host = torch.normal(0.0, 1.0, (1, context_len, config.context_fan_in), dtype=torch.float32)
                context_positions = torch.arange(context_start, context_start + context_len)
                noise_positions = torch.arange(anchor_pos, anchor_pos + block)

                if args.mode == "uncached_padded":
                    # Every call sees exactly (cap + block) rows however many context rows
                    # are real.  That is the whole point: identical shapes hit the ttnn
                    # program cache instead of recompiling every op.
                    cap = args.context_cap
                    padded = torch.zeros(1, cap, config.context_fan_in, dtype=torch.float32)
                    valid = min(context_len, cap)
                    padded[:, :valid, :] = context_host[:, :valid, :]
                    position_ids = torch.cat([torch.arange(cap), torch.arange(anchor_pos, anchor_pos + block)])
                    start = time.perf_counter()
                    tt_context = _upload(padded, mesh_device)
                    tt_noise = _upload(noise_host, mesh_device)
                    out = drafter(tt_noise, tt_context, position_ids=position_ids)
                else:
                    start = time.perf_counter()
                    tt_context = _upload(context_host, mesh_device)
                    tt_noise = _upload(noise_host, mesh_device)
                    out = drafter.forward_cached(
                        tt_noise,
                        tt_context,
                        context_positions=context_positions,
                        noise_positions=noise_positions,
                        cache=cache,
                    )
                ttnn.synchronize_device(mesh_device)
                elapsed = time.perf_counter() - start
                ttnn.deallocate(tt_context)
                ttnn.deallocate(out)

                tag = "warmup" if index < args.warmup else "timed"
                if tag == "timed":
                    per_call.append(elapsed)
                print(
                    f"  iter {index:2d}  ctx={context_len:4d}  cache={cache.length:5d}  "
                    f"{elapsed * 1000:8.1f} ms  [{tag}]",
                    flush=True,
                )

                context_start = anchor_pos
                anchor_pos += matches[index] + 1 if index < len(matches) else 1

        cache.release()

        mean_ms = 1000.0 * sum(per_call) / len(per_call) if per_call else 0.0
        print(f"\nmean drafter call: {mean_ms:.1f} ms over {len(per_call)} timed iterations")

        report = None
        if args.breakdown:
            total = sum(timer.totals.values())
            print(f"\nper-op attribution (total {total:.2f}s across all iterations, syncs included):")
            print(f"  {'op':22s} {'seconds':>9s} {'share':>7s} {'calls':>7s} {'ms/call':>9s}")
            report = []
            for name, seconds, count in timer.report():
                share = 100.0 * seconds / total if total else 0.0
                print(f"  {name:22s} {seconds:9.3f} {share:6.1f}% {count:7d} {1000.0 * seconds / count:9.3f}")
                report.append({"op": name, "seconds": seconds, "share": share, "calls": count})

        if args.out:
            with open(args.out, "w") as handle:
                json.dump(
                    {
                        "mesh": args.mesh,
                        "weight_dtype": args.dtype,
                        "activation_dtype": args.activation_dtype,
                        "prompt_len": args.prompt_len,
                        "matches": matches,
                        "per_call_seconds": per_call,
                        "mean_ms": mean_ms,
                        "load_seconds": load_seconds,
                        "breakdown": report,
                    },
                    handle,
                    indent=2,
                )
            print(f"\nwrote {args.out}")
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
