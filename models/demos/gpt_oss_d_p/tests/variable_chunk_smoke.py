# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Variable chunk length smoke test.

Builds ONE prefill runtime supporting two chunk sizes {1024, 10240} and times processing the SAME
1024-token prompt two ways — as a native 1024 chunk vs padded into a 10240 chunk — to show the small
chunk avoids the ~10x padding waste on a typical short request.

Both ways are ONE-SHOT (actual_start=0, gather-Q AllGather) — the primary use case for a short request
and the exact path variable chunk optimizes. One-shot uses FABRIC_1D (linear); NO torus / ring needed,
so this runs on any healthy galaxy node. We skip runtime.compile() (which would warm the multi-chunk
ring path and require the torus); the timing loop's first iter JIT-warms, the median of 3 is warm.

GALAXY-GATED. Env: HF_MODEL, GPT_OSS_WEIGHTS_FROM_CACHE=1, EXPERT_DTYPE=bf8,
TT_MESH_GRAPH_DESC_PATH=.../single_bh_galaxy_mesh_graph_descriptor.textproto, PREFILL_NUM_LAYERS(opt).
"""

import os
import resource
import statistics
import sys
import time

import ttnn

ROWS, COLS = 4, 8
GALAXY_NUM_DEVICES = 32
# Sizes overridable via env: 1k/8k is the 128k-context pair (131072 = 128*1024 = 16*8192);
# 10240 was the original large size (does NOT divide 128k).
SMALL = int(os.getenv("VARCHUNK_SMALL", "1024"))
LARGE = int(os.getenv("VARCHUNK_LARGE", "10240"))


def _raise_nproc_limit():
    """Raise RLIMIT_NPROC to the hard limit so the cold JIT kernel build (a burst of g++/collect2
    procs) doesn't fail with posix_spawn "Operation not permitted". Same as the galaxy PCC harness."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
            print(f"[varchunk] raised RLIMIT_NPROC soft {soft} -> {hard}", flush=True)
        except (ValueError, OSError) as e:
            print(f"[varchunk] WARNING: could not raise RLIMIT_NPROC (soft={soft}): {e}", file=sys.stderr)


def main():
    _raise_nproc_limit()
    if ttnn.get_num_devices() < GALAXY_NUM_DEVICES:
        print(f"[varchunk] SKIP: needs galaxy ({GALAXY_NUM_DEVICES}); have {ttnn.get_num_devices()}", flush=True)
        return 0
    from models.demos.gpt_oss_d_p.tt.model_config import ModelArgs
    from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    # Since #53153 chunk 0 uses cache-backed RingJointSDPA whenever max_seq_len > chunk (i.e. the small
    # size here) -> needs FABRIC_1D_RING + the torus descriptor + a fresh node without tt-smi -r.
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(ROWS, COLS))
    print(f"[varchunk] mesh {tuple(mesh.shape)} ndev={mesh.get_num_devices()}", flush=True)
    try:
        model_args = ModelArgs(mesh_device=mesh)
        hf_config = model_args.hf_config
        num_layers = hf_config.num_hidden_layers
        nl = os.getenv("PREFILL_NUM_LAYERS")
        if nl:
            num_layers = int(nl)
            hf_config.num_hidden_layers = num_layers
        expert_dtype = ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b
        cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
        if os.getenv("GPT_OSS_WEIGHTS_FROM_CACHE") == "1":
            state_dict = {}
        else:
            state_dict = ModelArgs.load_state_dict(model_args.weights_path)

        cfg = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=LARGE,
            mesh_shape=(ROWS, COLS),
            default_chunk_size=LARGE,
            additional_chunk_sizes=(SMALL,),
            num_users=1,
            expert_weight_dtype=expert_dtype,
            cache_dtype=ttnn.bfloat8_b,
            weight_cache_path=cache_path,
            owns_kv_cache=True,
        )
        rt = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
        del state_dict
        print(f"[varchunk] built; supported chunk_sizes={rt.chunk_sizes} num_layers={num_layers}", flush=True)

        def time_prefill(chunk_size, real=SMALL, iters=3):
            """Prefill `real` real tokens as a single one-shot chunk of width `chunk_size`. Median ms.
            No compile() warmup here, so the first of `iters` JIT-warms and is discarded by the median."""
            toks = [0] * chunk_size
            ts = []
            for _ in range(iters):
                inp = rt.make_chunk_input(toks, chunk_size)
                t0 = time.perf_counter()
                rt.prefill_chunk(inp, slot_id=0, actual_start=0, actual_end=real, chunk_size=chunk_size)
                ttnn.synchronize_device(mesh)
                ts.append(time.perf_counter() - t0)
            return statistics.median(ts)

        t_small = time_prefill(SMALL)  # 1024 real tokens as a 1024 chunk (no waste)
        t_large = time_prefill(LARGE)  # SAME 1024 real tokens padded into a 10240 chunk (~10x work)
        print(
            f"[varchunk] RESULT ({num_layers}L): prefill 1024 real tokens — "
            f"as 1k chunk = {t_small * 1000:.1f} ms, as 10k chunk (padded) = {t_large * 1000:.1f} ms, "
            f"speedup = {t_large / t_small:.1f}x",
            flush=True,
        )
        print("[varchunk] DONE", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
