# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer HOST-DISPATCH profile for the eager prefill trace-gap investigation (companion to
galaxy_prefill_traced.py). Profiles the eager forward of exactly ONE dense and ONE sparse decoder layer
so a Tracy run isolates per-op HOST DURATION and OP-TO-OP LATENCY without the full 60-layer op count
overflowing Tracy's 1000-op-per-device buffer.

M3 layers 0-2 are dense and sparse layers start at index 3, and weights/type are index-keyed, so the
layers are built 0..3 from the tilized cache and then model.layers is spliced to [layer 0, layer 3] — the
forward runs only the dense and the sparse layer, back to back. A signpost after each layer
(on_layer_complete) segments the ops CSV; ReadDeviceProfiler flushes per forward so the buffer never drops
ops. (Feeding layer 3 with layer 0's output makes the numeric output meaningless — this measures dispatch,
not correctness.)

Run:
  HF_MODEL=/mnt/models/MiniMaxAI/MiniMax-M3-ref \
    python -m tracy -p -r -v -m pytest models/demos/minimax_m3/tests/layer_host_dispatch_profile.py -q
Then in the ops CSV: filter DEVICE ID == 0 (mesh ops are logged once per device) and segment by the
LAYER_0_DONE (end of dense) / LAYER_1_DONE (end of sparse) signpost rows.
"""
import math
import random
import resource

import ttnn

try:
    from tracy import signpost
except ImportError:

    def signpost(**kw):  # no-op when not run under tracy
        pass


SEQ_LEN = 5120  # one-shot prompt -> 640 tokens/chip at SP=8
REPS = 1  # measured forwards after one warmup
DENSE_IDX, SPARSE_IDX, BUILD_N = 0, 3, 4  # build 0..3; profile layer 0 (dense) + layer 3 (MSA attn + MoE)
MESH = (8, 4)


def _raise_nproc():
    _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
    except (ValueError, OSError):
        pass


def test_layer_host_dispatch_profile():
    from models.demos.minimax_m3.tt.attention import allocate_kv_caches
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
    from models.demos.minimax_m3.tt.weight_cache import weight_cache_is_complete

    _raise_nproc()
    rng = random.Random(0)  # perf is op-graph-shaped, not token-dependent
    total = max(16 * 128, math.ceil(SEQ_LEN / 1024) * 1024)
    tokens = [rng.randrange(1000) for _ in range(SEQ_LEN)] + [0] * (total - SEQ_LEN)
    rows, cols = MESH

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), trace_region_size=200_000_000)
    try:
        model_args = ModelArgs(mesh_device=mesh)
        hf_config = model_args.hf_config
        hf_config.num_hidden_layers = BUILD_N
        cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
        cache_only = weight_cache_is_complete(cache_path, hf_config, BUILD_N, ttnn.bfloat4_b)
        state_dict = {} if cache_only else ModelArgs.load_state_dict(model_args.weights_path)
        cfg = TtPrefillRuntimeConfig(
            num_layers=BUILD_N,
            max_seq_len=total,
            mesh_shape=(rows, cols),
            chunk_size=total,
            num_users=1,
            weight_cache_path=cache_path,
        )
        runtime = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
        del state_dict
        layers = runtime.model.layers
        runtime.model.layers = [layers[DENSE_IDX], layers[SPARSE_IDX]]

        kv_cache = allocate_kv_caches(
            mesh, num_layers=BUILD_N, max_seq_len=total, num_users=1, head_dim=hf_config.head_dim
        )
        runtime.compile(kv_cache)

        model = runtime.model
        tok = runtime.make_chunk_input(tokens)
        x_persist = runtime._embed_tokens(tok)
        ttnn.deallocate(tok)

        def fwd():
            return model.prefill_forward(
                ttnn.clone(x_persist),
                rot_mats_global=runtime.rope_indexed,
                kv_cache=kv_cache,
                cached_len=0,
                user_id=0,
                get_last_token=-1,
                skip_lm_head=True,
                indexed_rope=True,
                on_layer_complete=lambda i: signpost(header=f"LAYER_{i}_DONE"),  # i=0 dense, i=1 sparse
            )

        for measured in [False] + [True] * REPS:
            if measured:
                signpost(header="FWD_START")
            o = fwd()
            ttnn.synchronize_device(mesh)
            if o is not None:
                o.deallocate(True)
            ttnn.ReadDeviceProfiler(mesh)  # flush per forward -> Tracy 1000-op device buffer never overflows
    finally:
        ttnn.close_mesh_device(mesh)
