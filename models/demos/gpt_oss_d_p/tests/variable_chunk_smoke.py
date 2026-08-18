# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Variable chunk length smoke test.

Builds ONE prefill runtime supporting two chunk sizes {1024, 10240} and times processing the SAME
1024-token prompt two ways — as a native 1024 chunk vs padded into a 10240 chunk — to show the small
chunk avoids the ~10x padding waste on a typical short request. One-shot (actual_start=0) both ways.

GALAXY-GATED. Run on a fresh bh_sc36_2 node WITHOUT tt-smi -r (compile warms the small-size ring, which
needs the torus). Env: HF_MODEL, GPT_OSS_WEIGHTS_FROM_CACHE=1, EXPERT_DTYPE=bf8,
TT_MESH_GRAPH_DESC_PATH=.../single_bh_galaxy_torus_xy_graph_descriptor.textproto, PREFILL_NUM_LAYERS(opt).
"""
import os
import statistics
import sys
import time

import ttnn

ROWS, COLS = 4, 8
GALAXY_NUM_DEVICES = 32
SMALL, LARGE = 1024, 10240


def main():
    if ttnn.get_num_devices() < GALAXY_NUM_DEVICES:
        print(f"[varchunk] SKIP: needs galaxy ({GALAXY_NUM_DEVICES}); have {ttnn.get_num_devices()}", flush=True)
        return 0
    from models.demos.gpt_oss_d_p.tt.model_config import ModelArgs
    from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)  # compile warms the small-size ring path
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
            chunk_size=LARGE,
            extra_chunk_sizes=(SMALL,),
            num_users=1,
            expert_weight_dtype=expert_dtype,
            cache_dtype=ttnn.bfloat8_b,
            weight_cache_path=cache_path,
            owns_kv_cache=True,
        )
        rt = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
        del state_dict
        print(f"[varchunk] built; supported chunk_sizes={rt.chunk_sizes} num_layers={num_layers}", flush=True)
        rt.compile()
        print("[varchunk] compiled both sizes", flush=True)

        def time_prefill(chunk_size, real=SMALL, iters=3):
            """Prefill `real` real tokens as a single chunk of width `chunk_size` (pad tail). Median ms."""
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
