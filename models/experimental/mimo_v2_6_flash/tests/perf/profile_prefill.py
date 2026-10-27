# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill profile of the 48-layer MiMo, one user per Galaxy row (each row a TP=8 replica) (#61418).

Zones (default): the cache is filled with positions [0, context - chunk) chunk by chunk, the last chunk runs once to
compile its programs, then runs again inside the ``profiled_chunk`` zone, so the profile is one chunk attending to
a full context. The device profiler is read after every chunk outside the zone (its buffer holds a limited number
of programs) and once after the zone, never inside it. Needs ``MIMO_PROFILE_ZONES=1`` and a tracy run; roll the CSV
up with parse_prefill_zones.py:

    MIMO_PROFILE_ZONES=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 python -m tracy -v -r -p \\
        models/experimental/mimo_v2_6_flash/tests/perf/profile_prefill.py --context 32768

``--ttft``: no profiler; wall-clock time of each whole prompt (prefill and its last-token logits), the second of
two runs, for every ``--contexts`` length.

Prompts are random token ids (seed 0), as in prefill_time.py.
"""

import argparse
import os
import time

import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference.checkpoint import MiMoCheckpoint
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common
from models.experimental.mimo_v2_6_flash.tt import profiling
from models.experimental.mimo_v2_6_flash.tt.generator import MiMoGalaxyGenerator
from models.experimental.mimo_v2_6_flash.tt.model import prefill_padded_len

READ_EVERY = 3  # layers between device profiler reads (a 3-layer chunk read is known good)


def fabric_router_config(packet_bytes):
    """A FabricRouterConfig with this packet payload size, or None for the default (4352 B on this Galaxy)."""
    if not packet_bytes:
        return None
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = packet_bytes
    return config


def build(mesh, max_seq_len, replicate_global_kv=False, users=4):
    cfg = MiMoConfig.from_model_path(common.MODEL_PATH)
    gen = MiMoGalaxyGenerator(
        mesh,
        cfg,
        MiMoCheckpoint(common.MODEL_PATH, cfg),
        os.path.join(common.CACHE_ROOT, "model_bfp4experts"),
        dp=4,
        users_per_replica=users,
        max_seq_len=max_seq_len,
        replicate_global_kv=replicate_global_kv,
    )
    tokens = torch.randint(0, cfg.vocab_size, (max_seq_len,), generator=torch.Generator().manual_seed(0))
    return gen.models, tokens


def profile(models, tokens, context, chunk, layers, fill, graph_report=None):
    """``models``: the rows to profile, each running the same chunks (one user each). One row is the default: the
    device log of one 48-layer chunk on 32 devices is ~30 GB.

    ``fill`` "skip": the earlier chunks only take their SWA blocks (host bookkeeping) and the cache keeps its zeros.
    The profiled chunk does the same work (the attention reads every cached position whatever it holds), but routing
    sees other hidden states, and the profile of one chunk at 128K stays a few GB instead of all 32 chunks'."""
    start = context - chunk

    def drain():
        for model in models:
            ttnn.ReadDeviceProfiler(model.mesh_device)

    def reader(model):
        """Synchronize and read the row's profiler every READ_EVERY layers: one read of a whole 48-layer chunk
        (~1700 programs per device) fails the profiler's marker matching. Waiting between layers adds idle time
        between ops but does not change their device-kernel time, which is what the zones sum."""

        def after_layer(i):
            if (i + 1) % READ_EVERY == 0:
                ttnn.synchronize_device(model.mesh_device)
                ttnn.ReadDeviceProfiler(model.mesh_device)

        return after_layer

    readers = [reader(model) for model in models]

    def run(s):
        t0 = time.time()
        for model, after_layer in zip(models, readers):
            x, _ = model._prefill_chunk(tokens[s : s + chunk], 0, s, None, layers, after_layer=after_layer)
            x.deallocate(True)
        for model in models:
            ttnn.synchronize_device(model.mesh_device)
        return time.time() - t0

    for s in range(0, start, chunk):
        if fill == "skip":
            for model in models:
                model.swa_table.prefill(0, s, s + chunk, s + prefill_padded_len(chunk))
            continue
        run(s)
        drain()
    logger.info(f"cache {'filled' if fill == 'real' else 'skipped (zeros)'} for positions [0, {start})")
    warm = run(start)  # compiles the last chunk's programs (it rewrites the same K / V when run again)
    drain()
    if graph_report:  # ttnn-visualizer memory / graph report of the profiled chunk only (ttnn.graph_report)
        ttnn.graph.begin_graph_capture()
    with profiling.zone("profiled_chunk"):
        wall = run(start)
    if graph_report:
        ttnn.graph.end_graph_capture_to_file(graph_report)
    drain()
    logger.info(
        f"context {context}: chunk of {chunk} at {start}, {len(layers) if layers else 48} layers: "
        f"{wall * 1e3:.1f} ms wall under the profiler (first run {warm * 1e3:.1f} ms)"
    )


def ttft(model, tokens, contexts, chunk):
    for n in contexts:
        times = []
        for _ in range(2):
            t0 = time.time()
            model.prefill_read(model.prefill_enqueue(tokens[:n], 0, chunk_size=chunk))
            times.append(time.time() - t0)
        logger.info(f"TTFT {n} tokens in chunks of {chunk}: {times[1]:.3f} s ({n / times[1]:.0f} tokens/s)")
        print(f"MIMO_TTFT context={n} chunk={chunk} s={times[1]:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=4096, help="zones: the profiled chunk ends here")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--layers", default=None, help="zones: comma-separated layer indices (default all)")
    parser.add_argument("--fill", choices=("real", "skip"), default="real", help="zones: the earlier chunks (profile)")
    parser.add_argument("--rows", type=int, default=1, help="zones: Galaxy rows to profile (1-4)")
    parser.add_argument("--fabric-packet", type=int, default=0, help="fabric packet payload bytes (0: default)")
    parser.add_argument(
        "--replicate-global-kv",
        action="store_true",
        help="both chips of a global layer's pair keep the K / V (vLLM's layout): later chunks split the heads",
    )
    parser.add_argument(
        "--graph-report",
        default=None,
        help="zones: also capture the profiled chunk's graph to this JSON; then `python -m ttnn.graph_report <json> "
        "<dir>` makes the ttnn-visualizer memory report",
    )
    parser.add_argument("--ttft", action="store_true", help="time whole prompts instead (no profiler)")
    parser.add_argument("--contexts", default="4096,32768,131072", help="--ttft: prompt lengths")
    parser.add_argument(
        "--users", type=int, default=4, help="users per replica the caches are sized for (1 lets a 1M prompt fit)"
    )
    args = parser.parse_args()
    contexts = [int(c) for c in args.contexts.split(",")]
    assert args.ttft or profiling.ENABLED, "set MIMO_PROFILE_ZONES=1 (or pass --ttft)"
    assert args.context % args.chunk == 0, "--context must be a multiple of --chunk"
    layers = [int(i) for i in args.layers.split(",")] if args.layers else None

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        *([router] if (router := fabric_router_config(args.fabric_packet)) else []),
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=65536)
    try:
        models, tokens = build(mesh, max(contexts) if args.ttft else args.context, args.replicate_global_kv, args.users)
        if args.ttft:
            ttft(models[0], tokens, contexts, args.chunk)
        else:
            profile(models[: args.rows], tokens, args.context, args.chunk, layers, args.fill, args.graph_report)
    finally:
        # Submeshes first, as the pytest mesh_device fixture does: closing the parent with its row submeshes still
        # open hangs in the profiler read of the close.
        for submesh in mesh.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
