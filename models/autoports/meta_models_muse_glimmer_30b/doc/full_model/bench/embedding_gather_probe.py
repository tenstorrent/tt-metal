# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The nondeterminism is in the embedding.  Is it the lookup or the all-gather?

``prefill_divergence_probe.py`` localised the sporadic prefill nondeterminism to
``embed_prefill`` -- the *first* stage of the graph, before any layer -- with the
layers merely carrying it forward.  ``embed_prefill`` does two things: a local
``ttnn.embedding`` over this device's quarter of the hidden dimension, and
:meth:`MuseGlimmerModel._all_gather_async`, which replicates the residual stream
using semaphores shared across every shape and every caller.

This separates them and picks the fix in one run:

* **which step** -- the local lookup is read back per device before the gather, so
  a stable lookup with an unstable gather is conclusive;
* **which columns** -- the gathered row is compared per 1664-column device shard.
  If device 0's own quarter is stable and the other three move, the gather is
  losing or duplicating remote data rather than the lookup being wrong;
* **which variant is reproducible** -- the shipped shared-semaphore
  ``all_gather_async``, the same op with semaphores created per call, and the
  composite ``ttnn.all_gather``.  Each arm repeats until it diverges or runs out
  of attempts, so the arms are compared on divergence *rate*, not on one sample.

Rate matters here: ``prefill_sync_bisect.py`` was misleading precisely because a
3-run sample can come back clean by luck.

Usage::

    python doc/full_model/bench/embedding_gather_probe.py [--length 128] [--repeats 25]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
REPO = ROOT.parents[2]  # the tt-metal checkout
sys.path.insert(0, str(REPO))

from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (  # noqa: E402
    DEFAULT_TRACE_REGION_SIZE,
    build_generator,
    clear_generator_cache,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    CCL_TOPOLOGY,
    close_multichip_mesh,
    open_multichip_mesh,
)

VOCAB = 202048
DEVICES = 4


def say(*args) -> None:
    print(*args, flush=True)


def prompt_of(length: int, *, seed: int = 41) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return [int(t) for t in torch.randint(0, VOCAB, (length,), generator=gen).tolist()]


def per_device(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(tensor)]


def gather_shipped(model, local4):
    return model._all_gather_async(local4)


def gather_fresh_sems(model, local4):
    """The same op, but with semaphores created for this call only."""
    grid = model.mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})

    def sem():
        return ttnn.create_global_semaphore(model.mesh_device, crs, 0, ttnn.BufferType.L1_SMALL)

    out = ttnn.experimental.all_gather_async(
        local4,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=[sem(), sem()],
        barrier_semaphore=sem(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=CCL_TOPOLOGY,
    )
    ttnn.deallocate(local4)
    return out


def gather_composite(model, local4):
    out = ttnn.all_gather(local4, dim=3, topology=CCL_TOPOLOGY)
    ttnn.deallocate(local4)
    return out


def gather_cloned_shipped(model, local4):
    """Copy the embedding output into a fresh buffer, then gather that.

    ``gather_and_untilize_logits`` already clones before gathering, which is a
    candidate explanation for why the *logits* gather never misbehaved while the
    embedding gather does.
    """
    fresh = ttnn.clone(local4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(local4)
    return model._all_gather_async(fresh)


def gather_cloned_composite(model, local4):
    fresh = ttnn.clone(local4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(local4)
    out = ttnn.all_gather(fresh, dim=3, topology=CCL_TOPOLOGY)
    ttnn.deallocate(fresh)
    return out


ARMS = {
    "shipped_shared_sems": gather_shipped,
    "fresh_sems_per_call": gather_fresh_sems,
    "composite_all_gather": gather_composite,
    # Same gather, same memory context, but a host-staged constant instead of the
    # embedding's output as the input.
    "staged_input_shared_sems": gather_shipped,
    "staged_input_composite": gather_composite,
    "cloned_shipped": gather_cloned_shipped,
    "cloned_composite": gather_cloned_composite,
    "native4d_shipped": gather_shipped,
    "native4d_composite": gather_composite,
}
STAGED_ARMS = {"staged_input_shared_sems", "staged_input_composite"}
NATIVE4D_ARMS = {"native4d_shipped", "native4d_composite"}


def staged_input(model, rows: int) -> ttnn.Tensor:
    """A known constant of the embedding's shape, staged straight from host.

    ``ccl_reproducibility_probe.py`` gathers exactly this, outside any model, and
    is clean and exactly correct at every row count.  Staging it *inside* the built
    model separates the two remaining explanations: the gather misbehaving in this
    memory context, or the embedding output being a special input.
    """
    hidden = int(model.embed_weight.shape[-1]) * DEVICES
    torch.manual_seed(17)
    reference = torch.randn(1, 1, rows, hidden, dtype=torch.bfloat16)
    return ttnn.from_torch(
        reference,
        device=model.mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(model.mesh_device, dim=3),
    )


def embed_4d(model, ids: list[int]) -> ttnn.Tensor:
    """The embedding output as a *natively* 4D tensor, with no unsqueeze view.

    ``ttnn.embedding`` gives rank(input)+1, so a ``[1, 1, n]`` token tensor yields
    ``[1, 1, n, 1664]`` directly.  The shipped path instead embeds a ``[1, n]``
    tensor and reshapes the ``[1, n, 1664]`` result with ``ttnn.unsqueeze_to_4D``,
    which is the one structural difference left between the embedding output (which
    gathers nondeterministically) and a host-staged tensor of identical shape and
    contents (which does not).
    """
    padded = ((len(ids) + 31) // 32) * 32
    host_ids = torch.full((1, 1, padded), model.embed_pad_id, dtype=torch.int32)
    host_ids[0, 0, : len(ids)] = torch.tensor(list(ids), dtype=torch.int32)
    tokens = ttnn.from_torch(
        host_ids,
        device=model.mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
    )
    out = ttnn.embedding(tokens, model.embed_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn.deallocate(tokens)
    return out


def one_run(
    model, ids: list[int], gather, *, staged: bool = False, native4d: bool = False
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """``(local lookup per device, gathered device-0 copy)`` for one embedding."""
    if native4d:
        local4 = embed_4d(model, ids)
    elif staged:
        local4 = staged_input(model, ((len(ids) + 31) // 32) * 32)
        ttnn.synchronize_device(model.mesh_device)
    else:
        tokens, _ = model.prefill_tokens_to_device(ids)
        local = ttnn.embedding(tokens, model.embed_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.deallocate(tokens)
        local4 = ttnn.unsqueeze_to_4D(local)
        if local4 is not local:
            ttnn.deallocate(local)
    local_hosts = per_device(local4)
    gathered = gather(model, local4)
    gathered_host = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0]).float()
    ttnn.deallocate(gathered)
    return local_hosts, gathered_host


def run_arm(model, arm: str, ids: list[int], repeats: int) -> dict:
    gather = ARMS[arm]
    staged = arm in STAGED_ARMS
    native4d = arm in NATIVE4D_ARMS
    ref_local, ref_gathered = one_run(model, ids, gather, staged=staged, native4d=native4d)
    hidden = int(ref_gathered.shape[-1])
    shard = hidden // DEVICES
    diverged_at = None
    local_diverged = False
    shard_diffs = [0.0] * DEVICES
    worst = 0.0
    for index in range(1, repeats + 1):
        local, gathered = one_run(model, ids, gather, staged=staged, native4d=native4d)
        if any(not torch.equal(a, b) for a, b in zip(ref_local, local)):
            local_diverged = True
        if not torch.equal(ref_gathered, gathered):
            diverged_at = index
            worst = float((ref_gathered - gathered).abs().max())
            for device in range(DEVICES):
                lo, hi = device * shard, (device + 1) * shard
                shard_diffs[device] = float((ref_gathered[..., lo:hi] - gathered[..., lo:hi]).abs().max())
            break
    row = {
        "arm": arm,
        "diverged_on_run": diverged_at,
        "attempts": repeats,
        "reproducible": diverged_at is None,
        "local_lookup_diverged": local_diverged,
        "gathered_max_abs_diff": worst,
        "per_device_shard_max_abs_diff": shard_diffs,
    }
    say(f"GARM {json.dumps(row)}")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=128)
    # Sweeping row counts answers the two questions a fix depends on: whether the
    # 32-row *decode* gather through the same op is also at risk (it runs inside a
    # trace, so a race there is worse), and whether the composite op is clean at the
    # larger fixed shape a bounded-program fix would adopt.
    parser.add_argument("--lengths", default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--out", default="embedding_gather_probe.json")
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")] if args.lengths else [args.length]
    ids = prompt_of(args.length)
    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    generator = None
    results = []
    try:
        # One layer: the embedding is what is under test, and a 52-layer build
        # would cost 160 s for nothing.
        generator = build_generator(ROOT, mesh, max_seq_len=args.max_seq_len, max_batch_size=1, layer_indices=[0])
        for length in lengths:
            say(f"--- rows={length}")
            for arm in args.arms.split(","):
                try:
                    row = run_arm(generator.model, arm, prompt_of(length), args.repeats)
                    row["rows"] = length
                    results.append(row)
                except Exception as exc:  # noqa: BLE001
                    say(f"GARM {arm} rows={length} FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
                    results.append({"arm": arm, "rows": length, "error": str(exc)[:400]})
        out = ROOT / "doc/full_model" / args.out
        out.write_text(json.dumps({"length": args.length, "arms": results}, indent=2) + "\n")
        say(f"GATHER wrote {out}")
        say("GATHER_OK")
        return 0
    finally:
        if generator is not None:
            generator.teardown()
        clear_generator_cache()
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
