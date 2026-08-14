# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Narrow the stage-05 watcher assert to a raw ``all_gather_async`` call.

``sampler_watcher_ab.py`` showed the assert reproduces on the **first** eager
call of ``Sampling1D``'s force-argmax path, with **or without** a barrier
semaphore, while the split top-k path is clean. That rules out the barrier
semaphore -- the README's leading hypothesis -- and points at the gather itself.

This probe removes ``Sampling1D`` too. It calls
``ttnn.experimental.all_gather_async`` directly on a 1x4 ring and sweeps one
knob at a time from the sampler's spelling towards the decoder layer's
(watcher-clean) spelling, so the upstream report can name a parameter rather
than a module.

One leg per process -- the watcher aborts the process on the first trip:

    TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
      python .../probes/ccl_watcher_ab.py --leg sampler_argmax_shape

``--leg list`` prints the legs.

**Result (archived in ../watcher_ab.log).** The minimal trigger is
``topology=Topology::Linear`` together with ``num_workers_per_link=1``. Either
alone is clean; both together trip a BRISC ``ASSERT`` in
``minimal_default_writer.cpp`` on the first call, at the sampler's 37984-wide
per-die shape and equally at the decoder layer's 512-wide one -- so it is the
op's parameters, not the model's tensors. ``--leg linear_workers1`` is the
upstream reproducer.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from models.common.modules.tt_ccl import TT_CCL  # noqa: E402

DEVICES = 4
SLOTS = 32
LOCAL_VOCAB = 37984  # 151936 / 4, the column-parallel LM head's per-die width
LAYER_WIDTH = 512  # 2048 / 4, what the decoder layer's all-gather sees

#: name -> (per-die last-dim width, kwargs overlaid on the base call)
LEGS: dict[str, tuple[int, dict]] = {
    # exactly what ``Sampling1D._argmax_all_gather`` ships on a ring mesh
    "sampler_argmax_shape": (LOCAL_VOCAB, dict(chunks_per_sync=24, num_workers_per_link=4, num_buffers_per_channel=2)),
    # same shape, with the barrier semaphore the layer always passes
    "sampler_argmax_barrier": (
        LOCAL_VOCAB,
        dict(chunks_per_sync=24, num_workers_per_link=4, num_buffers_per_channel=2, barrier=True),
    ),
    # same shape, but every tuning knob left at its default -- the layer's spelling
    "sampler_shape_default_knobs": (LOCAL_VOCAB, dict(barrier=True)),
    "sampler_shape_default_knobs_nobarrier": (LOCAL_VOCAB, dict()),
    # the sampler's knobs at the layer's *width*: is it the shape or the knobs?
    "layer_width_sampler_knobs": (
        LAYER_WIDTH,
        dict(chunks_per_sync=24, num_workers_per_link=4, num_buffers_per_channel=2),
    ),
    # the decoder layer's own spelling, known watcher-clean, as the control
    "layer_width_default_knobs": (LAYER_WIDTH, dict(barrier=True)),
    # one knob at a time away from the sampler's spelling, at the sampler's shape
    "sampler_shape_workers1": (
        LOCAL_VOCAB,
        dict(chunks_per_sync=24, num_workers_per_link=1, num_buffers_per_channel=2),
    ),
    "sampler_shape_chunks10": (
        LOCAL_VOCAB,
        dict(chunks_per_sync=10, num_workers_per_link=4, num_buffers_per_channel=2),
    ),
    # The branch ``Sampling1D._argmax_all_gather`` ACTUALLY takes on a 1x4
    # Blackhole ring: ``default_topology`` is not Ring here, so the ring branch
    # with its no-barrier comment is dead code on this mesh, and the fallback
    # runs -- Linear, cluster_axis=1, barrier semaphore, chunks_per_sync=10,
    # num_workers_per_link=1, num_links clamped by ``get_num_links``.
    "sampler_shipped_branch": (
        LOCAL_VOCAB,
        dict(
            chunks_per_sync=10,
            num_workers_per_link=1,
            num_buffers_per_channel=2,
            barrier=True,
            linear=True,
            cluster_axis=1,
        ),
    ),
    "sampler_shipped_branch_no_cluster_axis": (
        LOCAL_VOCAB,
        dict(chunks_per_sync=10, num_workers_per_link=1, num_buffers_per_channel=2, barrier=True, linear=True),
    ),
    "sampler_shipped_branch_nobarrier": (
        LOCAL_VOCAB,
        dict(chunks_per_sync=10, num_workers_per_link=1, num_buffers_per_channel=2, linear=True, cluster_axis=1),
    ),
    "layer_width_shipped_branch": (
        LAYER_WIDTH,
        dict(
            chunks_per_sync=10,
            num_workers_per_link=1,
            num_buffers_per_channel=2,
            barrier=True,
            linear=True,
            cluster_axis=1,
        ),
    ),
    # Minimal-combination bisection between the clean Linear leg
    # (chunks 24 / workers 4 / no barrier) and the tripping shipped branch
    # (chunks 10 / workers 1 / barrier).
    "linear_workers1": (
        LOCAL_VOCAB,
        dict(chunks_per_sync=24, num_workers_per_link=1, num_buffers_per_channel=2, linear=True),
    ),
    "linear_chunks10": (
        LOCAL_VOCAB,
        dict(chunks_per_sync=10, num_workers_per_link=4, num_buffers_per_channel=2, linear=True),
    ),
    "linear_barrier": (
        LOCAL_VOCAB,
        dict(chunks_per_sync=24, num_workers_per_link=4, num_buffers_per_channel=2, linear=True, barrier=True),
    ),
    "linear_default_knobs_nobarrier": (LOCAL_VOCAB, dict(linear=True)),
    "sampler_shape_linear": (
        LOCAL_VOCAB,
        dict(chunks_per_sync=24, num_workers_per_link=4, num_buffers_per_channel=2, linear=True),
    ),
}


def run_leg(mesh, ccl, name, reps):
    width, opts = LEGS[name]
    torch.manual_seed(0)
    host = torch.randn(1, 1, SLOTS, width * DEVICES)
    x = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )
    assert int(x.shape[-1]) == width, x.shape

    kwargs = dict(
        persistent_output_buffer=None,
        dim=3,
        num_links=1,
        memory_config=x.memory_config(),
        topology=ttnn.Topology.Linear if opts.get("linear") else ttnn.Topology.Ring,
    )
    for key in ("chunks_per_sync", "num_workers_per_link", "num_buffers_per_channel"):
        if key in opts:
            kwargs[key] = opts[key]

    axis = opts.get("cluster_axis")
    if axis is not None:
        kwargs["cluster_axis"] = axis

    def call():
        extra = {}
        if opts.get("barrier"):
            extra["barrier_semaphore"] = ccl.get_and_cycle_barrier_semaphore_handle(axis)
        return ttnn.experimental.all_gather_async(
            x, multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(axis), **kwargs, **extra
        )

    print(
        f"[{name}] width={width} kwargs={ {k: str(v) for k, v in kwargs.items() if k != 'memory_config'} } "
        f"barrier={bool(opts.get('barrier'))}",
        flush=True,
    )
    try:
        out = call()
        ttnn.synchronize_device(mesh)
        print(f"[{name}] first call OK, out width {int(out.shape[-1])}", flush=True)
        for i in range(reps):
            call()
            ttnn.synchronize_device(mesh)
        print(f"[{name}] {reps} calls OK", flush=True)
        print(f"[{name}] RESULT clean", flush=True)
    except Exception:
        traceback.print_exc()
        print(f"[{name}] RESULT exception", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg", default="list")
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    if args.leg == "list":
        for name in LEGS:
            print(name)
        return

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, DEVICES), trace_region_size=20_000_000)
    try:
        from models.common.modules.tt_ccl import default_topology

        ccl = TT_CCL(mesh)
        # Which branch of ``_argmax_all_gather`` this mesh actually takes.
        print(
            f"default_topology(mesh) = {default_topology(mesh)}  "
            f"num_devices = {mesh.get_num_devices()}  "
            f"get_num_links(1) = {ccl.get_num_links(1)}",
            flush=True,
        )
        names = list(LEGS) if args.leg == "all" else [args.leg]
        for name in names:
            run_leg(mesh, ccl, name, args.reps)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
