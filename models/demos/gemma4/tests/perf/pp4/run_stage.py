# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run ONE Gemma 4 pipeline stage in a single process, with no MPI and no sockets.

This is the cheap half of pipeline bring-up. It carves an [8,1] column out of the galaxy the same
way the four-rank job does (``create_submeshes(MeshShape(8,1))`` on a plain FABRIC_2D 8x4 mesh --
the same fabric config test_factory gives the 8x4 baseline, and the one Gemma 4's Topology.Linear
ring_joint wants), builds the stage's layer window, and drives
chunks through it -- exercising every model-side change except the D2D hand-off itself.

Two things it is for:

* A stage with ``--first-layer 0 --last`` proves TP=1 works at all: no TP collectives, 4x the heads
  and 4x the MLP width per device, matmul program configs that were sized for TP=4.
* A stage with a NON-ZERO ``--first-layer`` and neither ``--first`` nor ``--last`` proves the
  global-index threading: it must build full attention where its GLOBAL layers are, and it must
  reject a local-index ``layer_scalar`` lookup. That failure does not raise inside the pipeline --
  it computes the wrong thing -- so catching it here, with no fabric or MPI in the picture, is
  worth a lot more than catching it at 256k.

It also POPULATES the TP=1 weight cache for its window, which is why the four windows are run
sequentially before the pipeline job: a single [8,1] mesh cannot hold all 60 layers at TP=1
(~29 GB/chip of weights), so the cache has to be built a window at a time.

Usage (galaxy idle; run detached, never in a foreground call with a timeout):
    python models/demos/gemma4/tests/perf/pp4/run_stage.py \\
        --first-layer 17 --num-layers 13 --max-seq 32768 --chunks 4
"""

import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-layer", type=int, default=0)
    ap.add_argument("--num-layers", type=int, default=15)
    ap.add_argument("--column", type=int, default=0, help="which [8,1] column of the 8x4 galaxy to use")
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--num-users", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=4, help="how many chunks to push through the stage")
    ap.add_argument("--first", action="store_true", help="this stage embeds tokens (rank 0)")
    ap.add_argument("--last", action="store_true", help="this stage runs the final norm (rank N-1)")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--build-cache-only", action="store_true", help="compile and exit; populates the weight cache")
    args = ap.parse_args()

    import ttnn
    from loguru import logger

    from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
    from models.demos.common.prefill.runners.runner_utils import _create_fabric_router_config

    adapter = get_adapter("gemma4_31b")
    hf_config = adapter.load_hf_config()
    text_config = getattr(hf_config, "text_config", hf_config)
    layer_types = list(text_config.layer_types)
    window = layer_types[args.first_layer : args.first_layer + args.num_layers]
    n_global = sum(1 for t in window if t == "full_attention")
    global_idx = [args.first_layer + i for i, t in enumerate(window) if t == "full_attention"]
    logger.warning(
        f"[stage] layers [{args.first_layer}, {args.first_layer + args.num_layers}) : "
        f"{n_global} full_attention at {global_idx}, {len(window) - n_global} sliding | "
        f"is_first={args.first} is_last={args.last}"
    )

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D,
        ttnn.FabricReliabilityMode.RELAXED_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        _create_fabric_router_config(max_payload_size=adapter.model_config.FABRIC_PAYLOAD_SIZE),
    )
    trace_region = 0 if args.no_trace else int(os.environ.get("GEMMA4_PREFILL_TRACE_REGION_SIZE", 256 * 1024 * 1024))
    full = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(8, 4),
        l1_small_size=adapter.l1_small_size,
        trace_region_size=trace_region,
    )
    rc = 1
    try:
        columns = full.create_submeshes(ttnn.MeshShape(8, 1))
        mesh = columns[args.column]
        logger.warning(f"[stage] column {args.column} devices={list(mesh.get_device_ids())} shape={tuple(mesh.shape)}")

        params = PrefillRunParams(
            mesh_shape=(8, 1),
            num_layers=args.num_layers,
            first_layer_idx=args.first_layer,
            is_first_rank=args.first,
            is_last_rank=args.last,
            max_seq_len=args.max_seq,
            chunk_size=args.chunk,
            num_users=args.num_users,
            capacity_factor=8,
            num_links=2,
            gate_mode_name="DEVICE_FP32",
            kv_only_last_layer=True,
            weight_cache_path=adapter.weight_cache_path((8, 1)),
            use_trace=not args.no_trace,
        )
        logger.warning(f"[stage] weight cache -> {params.weight_cache_path}")

        t0 = time.perf_counter()
        runtime = adapter.build_runtime(mesh_device=mesh, hf_config=hf_config, params=params)
        kv = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf_config, params=params)
        logger.warning(f"[stage] kv allocated in {time.perf_counter() - t0:.1f}s")

        t1 = time.perf_counter()
        runtime.compile(kv)
        logger.warning(f"[stage] MODEL_READY compile={time.perf_counter() - t1:.1f}s total={time.perf_counter() - t0:.1f}s")
        if args.build_cache_only:
            logger.warning("[stage] --build-cache-only: weight cache populated, exiting before the chunk loop")
            rc = 0
            return

        if not args.no_trace:
            t2 = time.perf_counter()
            runtime.capture_trace(kv)
            logger.warning(f"[stage] trace captured in {time.perf_counter() - t2:.1f}s")

        # prefill_chunk OWNS its input (it deallocates what the socket handed it), so build a
        # fresh one per chunk exactly as the runner's H2D / D2D receive does.
        tokens = list(range(args.chunk)) if args.first else []
        n = min(args.chunks, args.max_seq // args.chunk)
        per_chunk = []
        out = None
        for c in range(n):
            start = c * args.chunk
            chunk_input = runtime.make_chunk_input(tokens)
            ttnn.synchronize_device(mesh)
            t = time.perf_counter()
            out = runtime.prefill_chunk(
                chunk_input,
                kv,
                slot_id=0,
                actual_start=start,
                actual_end=start + args.chunk,
                request_id=c,
            )
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t) * 1000.0
            per_chunk.append(ms)
            shape = tuple(int(d) for d in out.shape) if out is not None else None
            logger.warning(f"[stage] CHUNK c={c} [{start},{start + args.chunk}) ms={ms:.1f} out={shape}")

        if args.last:
            assert out is None, "a last rank must not return an activation"
        else:
            assert out is not None, "a non-last rank must return the stage activation for D2D"
            expected = (1, 1, args.chunk // 8, text_config.hidden_size)
            got = tuple(int(d) for d in out.shape)
            assert got == expected, f"stage activation {got} != expected {expected}"

        # Discard the first chunk: it is the only one whose ring gather has no prefix to read,
        # and under trace it is also the first replay. Later chunks grow with KV depth, which is
        # the point -- report them individually, not just the mean.
        warm = per_chunk[1:] or per_chunk
        logger.warning(
            f"[stage] RESULT layers={args.num_layers} globals={n_global} chunks={n} "
            f"first={per_chunk[0]:.1f}ms warm_mean={sum(warm) / len(warm):.1f}ms "
            f"per_layer_warm={sum(warm) / len(warm) / args.num_layers:.2f}ms "
            f"all={[round(x, 1) for x in per_chunk]}"
        )
        runtime.release_trace()
        rc = 0
    except Exception:
        logger.exception("[stage] FAILED")
    finally:
        ttnn.close_mesh_device(full)
    sys.exit(rc)


if __name__ == "__main__":
    sys.path.insert(0, os.environ.get("TT_METAL_HOME", "."))
    main()
