# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3 chunked-prefill zone profile / KV check on an SPxTP sub-mesh (e.g. 4x2) carved from the 8x4 galaxy.

Wraps models/demos/minimax_m3/tests/perf/profile_prefill.py (plan, load_tokens, build_runtime) and adds what
TP != 4 needs, as harness-local monkeypatches (model code untouched):
  * K/V cache with num_kv_heads / TP heads per chip: tt/attention/kv_cache.py allocates one head per chip
    and update_padded_kv_cache asserts cache heads == input heads;
  * MSA cache read that gathers a multi-head slot one head at a time: high_bw_all_gather's selected-batch
    path needs singleton dims between batch and the gather dim. The per-head slice / concat copies are
    charged to ag_kv (FINE sub-zones head_slice / head_concat), so ag_kv is inflated vs a native TP=2 path.

Env, besides profile_prefill.py's PROFILE_CHUNK / PROFILE_CACHE / PROFILE_NUM_LAYERS / PROFILE_LAYER_IDS /
PROFILE_READ_EVERY / PROFILE_SKIP_PREFIX / PROFILE_SKIP_COMPILE / PREFILL_TRACE_DIR / M3_FABRIC /
M3_CCL_TOPOLOGY / EXPERT_DTYPE / HF_MODEL / TT_CACHE_PATH:
  PROFILE_MESH     SPxTP, e.g. 4x2. Sub-mesh k = create_submeshes(MeshShape(SP, TP))[k]         [required]
  PROFILE_STAGE    sub-mesh / stage index; stage k owns layers [k*60/S, (k+1)*60/S), S=(8/SP)*(4/TP)  [0]
  PROFILE_KV_PCC   1 -> after the last chunk, per-layer K / V / index_k PCC vs PREFILL_TRACE_DIR/kv_cache
                   over the first min(capacity, golden length) tokens: the tokens are the golden's own,
                   tiled past its end, and a causal prefix does not depend on what follows           [0]
  PROFILE_KV_PCC_MIN  fail below this min PCC                                                       [0.88]
  PROFILE_KV_DUMP  dir -> save natural-order K / V / index_k [layers, heads, n_tokens, hd] (bf16)   [unset]
For KV runs set M3_PROFILE_ZONES=0 TT_METAL_DEVICE_PROFILER=0 and run without tracy.

Compare two dumps (e.g. 4x2 vs 2x4), no device needed:
  python3 profile_4x2.py --compare <dumpA> <dumpB>
"""

import json
import os
import sys
import time
from pathlib import Path


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def compare(dir_a, dir_b):
    import torch

    A, B = torch.load(Path(dir_a) / "kv.pt"), torch.load(Path(dir_b) / "kv.pt")
    assert A["layers"] == B["layers"], f"layer sets differ: {A['layers']} vs {B['layers']}"
    n = min(A["n_tokens"], B["n_tokens"])
    print(f"A={dir_a} mesh={A['mesh']}  B={dir_b} mesh={B['mesh']}  tokens={n}")
    worst = 1.0
    for i, L in enumerate(A["layers"]):
        line = f"  layer {L:>2}:"
        for name in ("k", "v", "index_k"):
            a, b = A[name][i][:, :n], B[name][i][:, :n]
            if name == "index_k" and a.abs().sum() == 0 and b.abs().sum() == 0:
                continue  # dense layers never write index_k
            p = _pcc(a, b)
            worst = min(worst, p)
            line += f" {name}={p:.5f}"
        print(line)
    print(f"worst PCC {worst:.5f}")
    return 0


def _hf_num_kv_heads():
    try:
        c = json.load(open(Path(os.environ["HF_MODEL"]) / "config.json"))
        return int(c.get("text_config", c)["num_key_value_heads"])
    except (KeyError, OSError):
        return 4


def _install_multihead_kv(n_kv):
    """allocate_kv_caches with n_kv / TP K/V heads per chip; index_k keeps its single shared head."""
    import torch

    import models.demos.minimax_m3.tt.attention as attn_pkg
    import ttnn
    from models.demos.minimax_m3.tt.attention import kv_cache as kvc

    orig = attn_pkg.allocate_kv_caches

    def allocate(
        mesh_device, *, num_layers, max_seq_len, sp_axis=0, num_users=1, head_dim=128, cache_dtype=ttnn.bfloat8_b
    ):
        n_local = n_kv // mesh_device.shape[1 - sp_axis]
        kw = dict(num_layers=num_layers, max_seq_len=max_seq_len, sp_axis=sp_axis, num_users=num_users)
        if n_local == 1:
            return orig(mesh_device, head_dim=head_dim, cache_dtype=cache_dtype, **kw)
        sp = mesh_device.shape[sp_axis]
        assert max_seq_len % sp == 0, f"max_seq_len ({max_seq_len}) must be divisible by sp ({sp})"
        seq_local = max_seq_len // sp
        grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(kvc.BH_NUM_DRAM_BANKS)]
        )
        spec = ttnn.NdShardSpec(
            shard_shape=[1, 1, kvc.NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
            grid=grid,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        mem = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)

        def alloc(n_heads, dtype):
            return ttnn.from_torch(
                torch.zeros(num_users * num_layers, n_heads, seq_local, head_dim),
                dtype=dtype,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        index_dtype = ttnn.bfloat16 if os.getenv("M3_INDEX_CACHE_BF16") == "1" else cache_dtype
        print(f"[4x2] KV cache: {n_local} K/V heads per chip (TP={mesh_device.shape[1 - sp_axis]})", flush=True)
        return kvc.MiniMaxKVCache(
            k=alloc(n_local, cache_dtype),
            v=alloc(n_local, cache_dtype),
            index_k=alloc(1, index_dtype),
            num_users=num_users,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            sp=sp,
        )

    attn_pkg.allocate_kv_caches = allocate


def _install_multihead_cache_read():
    """msa_sp_attention_cache_read for a cache with > 1 K/V head per chip; the 1-head path is unchanged."""
    import models.demos.minimax_m3.tt.attention.msa as msa
    import models.demos.minimax_m3.tt.attention.prefill as prefill
    import ttnn
    from models.demos.minimax_m3.utils.profiler_utils import FINE, zone

    orig = msa.msa_sp_attention_cache_read

    def cache_read(
        q,
        index_q,
        kv_cache,
        *,
        slot,
        mesh_config,
        ccl_manager,
        cached_len,
        chunk_local,
        scale,
        block_size,
        topk_blocks,
        num_groups=1,
    ):
        kw = dict(
            slot=slot,
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            cached_len=cached_len,
            chunk_local=chunk_local,
            scale=scale,
            block_size=block_size,
            topk_blocks=topk_blocks,
            num_groups=num_groups,
        )
        if kv_cache.k.shape[1] == 1:
            return orig(q, index_q, kv_cache, **kw)
        sp_axis = mesh_config.sp_axis
        device = ccl_manager.mesh_device
        sp = device.shape[sp_axis]
        assert sp > 1 and sp_axis == 0, f"multi-head cache read needs sp > 1 on axis 0 (sp={sp}, axis={sp_axis})"
        seq_local = kv_cache.k.shape[2]
        kv_len, n_rows = msa.msa_cache_read_extent(cached_len, chunk_local, sp, block_size)
        assert n_rows <= seq_local and kv_len <= seq_local * sp, f"cache read past capacity: {n_rows} > {seq_local}"

        def gather(key, t, batch_index):
            buf = ccl_manager.get_high_bw_gather_buffer(key, (1, 1, seq_local * sp, t.shape[3]), t.dtype)
            return msa.high_bw_sp_gather(
                t, mesh_config, ccl_manager, buf, input_batch_index=batch_index, gathered_dim_size=n_rows * sp
            )

        def gather_heads(key, cache_t):
            nh, hd = cache_t.shape[1], cache_t.shape[3]
            heads = []
            for h in range(nh):
                with zone("head_slice", FINE):
                    # DRAM interleaved: slicing into the cache's ND-shard spec trips the 1-D DRAM bank grid check
                    one = ttnn.slice(
                        cache_t,
                        [slot, h, 0, 0],
                        [slot + 1, h + 1, seq_local, hd],
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                heads.append(gather(f"{key}_h{h}", one, None))
                one.deallocate(True)
            with zone("head_concat", FINE):
                return ttnn.concat(heads, dim=1)

        with zone("ag_kv"):
            k_full = gather_heads("msa_cache_k", kv_cache.k)
            v_full = gather_heads("msa_cache_v", kv_cache.v)
        with zone("ag_index_k"):
            index_k_full = gather("msa_cache_index_k", kv_cache.index_k, slot)
        out = msa.msa_indexer_sparse(
            index_q,
            index_k_full,
            q,
            k_full,
            v_full,
            chunk_start_idx=cached_len,
            scale=scale,
            num_groups=num_groups,
            block_size=block_size,
            topk_blocks=topk_blocks,
            device=device,
            cluster_axis=sp_axis,
            block_cyclic_sp_axis=sp_axis,
            block_cyclic_chunk_local=chunk_local,
            kv_len=kv_len,
        )
        k_full.deallocate(True)  # concat results, not the persistent gather buffers
        v_full.deallocate(True)
        return out

    msa.msa_sp_attention_cache_read = cache_read
    prefill.msa_sp_attention_cache_read = cache_read


def _natural_kv(runtime, kv_cache, n_tokens):
    from models.demos.minimax_m3.tt.runners.prefill_kv_validation import naturalize_kv_block

    blocks = runtime.read_slot_kv(kv_cache, 0, n_tokens)
    sp, chunk, seq = runtime.config.sp_factor, runtime.config.chunk_size, runtime.read_seq_len(n_tokens)
    return [
        [naturalize_kv_block(blk[i], n_tokens, sp, chunk, seq) for i in range(blk.shape[0])] for blk in blocks
    ]  # [k, v, index_k] x layers x [heads, n_tokens, hd]


def _kv_pcc(nat, golden_dir, n_tokens, layer_ids, hf_config, threshold):
    import torch
    from safetensors import safe_open

    head_dim = hf_config.head_dim
    rotary_dim = getattr(hf_config, "rotary_dim", head_dim)
    half = rotary_dim // 2
    src = list(range(head_dim))
    for m in range(rotary_dim):
        src[m] = half * (m % 2) + (m // 2)
    src = torch.tensor(src, dtype=torch.long)  # HF half-split -> device Meta interleave (K, index_k)
    worst = 1.0
    for i, L in enumerate(layer_ids):
        with safe_open(str(Path(golden_dir) / "kv_cache" / f"layer_{L}.safetensors"), framework="pt") as h:
            keys = set(h.keys())
            g = {
                "k": h.get_tensor(f"key_cache_layer_{L}").float()[0, :, :n_tokens, :][..., src],
                "v": h.get_tensor(f"value_cache_layer_{L}").float()[0, :, :n_tokens, :],
            }
            if f"index_k_cache_layer_{L}" in keys:
                g["index_k"] = h.get_tensor(f"index_k_cache_layer_{L}").float()[0, :, :n_tokens, :][..., src]
        line = f"  layer {L:>2}:"
        for j, name in enumerate(("k", "v", "index_k")):
            if name in g:
                p = _pcc(g[name], nat[j][i])
                worst = min(worst, p)
                line += f" {name}={p:.5f}"
        print(line, flush=True)
    print(f"[4x2] KV PCC vs golden: min {worst:.5f} (threshold {threshold})", flush=True)
    return worst


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        return compare(sys.argv[2], sys.argv[3])

    mesh_env = os.getenv("PROFILE_MESH")
    if not mesh_env:
        raise SystemExit("set PROFILE_MESH=SPxTP, e.g. 4x2")
    sp_req, tp_req = (int(x) for x in mesh_env.lower().split("x"))
    assert 8 % sp_req == 0 and 4 % tp_req == 0, f"PROFILE_MESH={mesh_env} does not tile the 8x4 galaxy"
    stages = (8 // sp_req) * (4 // tp_req)
    stage = int(os.getenv("PROFILE_STAGE", "0"))
    assert 0 <= stage < stages, f"PROFILE_STAGE={stage} out of range for {stages} sub-meshes"

    from models.demos.minimax_m3.tests.perf import profile_prefill as pp  # sets M3_PROFILE_ZONES default, imports ttnn

    import ttnn

    pp._raise_nproc_limit()
    chunk = int(os.getenv("PROFILE_CHUNK", "5120"))
    cache_req = int(os.getenv("PROFILE_CACHE", "25600"))
    read_every = int(os.getenv("PROFILE_READ_EVERY", "1"))
    num_layers_override = os.getenv("PROFILE_NUM_LAYERS")
    layer_ids = [int(x) for x in os.getenv("PROFILE_LAYER_IDS", "").split(",") if x.strip()] or None
    do_pcc = os.getenv("PROFILE_KV_PCC") == "1"
    dump_dir = os.getenv("PROFILE_KV_DUMP")
    pcc_min = float(os.getenv("PROFILE_KV_PCC_MIN", "0.88"))
    fabric_config = pp.fabric_config_from_env()

    n_chunks, cache, total = pp.plan(chunk, cache_req)
    print(
        f"[4x2] sub-mesh {sp_req}x{tp_req}, {stages} sub-meshes, stage {stage}; one {chunk}-token chunk attending "
        f"{cache} cached tokens ({n_chunks} chunks, capacity {total})",
        flush=True,
    )
    tokens = pp.load_tokens(total)
    if do_pcc:
        n_src = len(json.load(open(Path(os.environ["PREFILL_TRACE_DIR"]) / "metadata.json"))["token_ids"])
        n_pcc = min(total, n_src)
        print(f"[4x2] KV PCC over the first {n_pcc} tokens (golden has {n_src})", flush=True)
        assert os.getenv("PROFILE_SKIP_PREFIX") != "1", "KV PCC needs the real prefix fill"
    if os.getenv("PROFILE_DRY_RUN") == "1":
        return 0

    _install_multihead_kv(_hf_num_kv_heads())
    _install_multihead_cache_read()

    pp.set_fabric_config_from_env(fabric_config)
    galaxy = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), l1_small_size=pp.L1_SMALL_SIZE)
    try:
        from models.demos.minimax_m3.utils.profiler_utils import COARSE, ZONES_ENABLED, read_profiler, zone

        mesh = galaxy.create_submeshes(ttnn.MeshShape(sp_req, tp_req))[stage] if stages > 1 else galaxy
        print(f"[4x2] mesh {tuple(mesh.shape)} ndev={mesh.get_num_devices()} fabric={fabric_config}", flush=True)
        sp, tp = tuple(mesh.shape)
        t0 = time.perf_counter()
        runtime, kv_cache, hf_config, layers = pp.build_runtime(
            mesh, chunk, total, num_layers_override, layer_ids, stages=stages, stage=stage
        )
        print(f"[4x2] built layers {layers} in {time.perf_counter() - t0:.1f}s", flush=True)

        state = {"in_chunk": False}

        def on_layer_complete(layer_idx):
            if not state["in_chunk"] and read_every > 0 and (layer_idx + 1) % read_every == 0:
                read_profiler(mesh)

        runtime._on_layer_complete = on_layer_complete

        t0 = time.perf_counter()
        skip_compile = os.getenv("PROFILE_SKIP_COMPILE") == "1"
        if not skip_compile:
            runtime.compile(kv_cache)
        print(f"[4x2] warmup done in {time.perf_counter() - t0:.1f}s (SP={sp} x TP={tp}, EP={sp * tp})", flush=True)

        def prefill_chunk(c):
            a = c * chunk
            inp = runtime.make_chunk_input(tokens[a : a + chunk])
            out = runtime.prefill_chunk(inp, kv_cache, slot_id=0, actual_start=a, actual_end=a + chunk)
            if out is not None:
                out.deallocate(True)

        if os.getenv("PROFILE_SKIP_PREFIX") != "1" and n_chunks > 1:
            t0 = time.perf_counter()
            for c in range(n_chunks - 1):
                prefill_chunk(c)
            ttnn.synchronize_device(mesh)
            print(f"[4x2] prefix of {n_chunks - 1} chunks filled in {time.perf_counter() - t0:.1f}s", flush=True)
        if skip_compile:
            if do_pcc or dump_dir:
                raise SystemExit("PROFILE_SKIP_COMPILE=1 re-runs the last chunk to warm it; not valid for KV checks")
            for _ in range(2):
                prefill_chunk(n_chunks - 1)
            ttnn.synchronize_device(mesh)

        state["in_chunk"] = True
        t0 = time.perf_counter()
        with zone("profiled_chunk", COARSE):
            prefill_chunk(n_chunks - 1)
            ttnn.synchronize_device(mesh)
        wall = time.perf_counter() - t0
        state["in_chunk"] = False
        read_profiler(mesh)
        print(
            f"\n[zone-prof] PROFILED CHUNK: {chunk} tok @ {cache} cache, {len(layers)} layers "
            f"(stage {stage}/{stages}, mesh {sp}x{tp}, fabric {fabric_config}, zones {'ON' if ZONES_ENABLED else 'OFF'})\n"
            f"  wall-clock: {wall * 1e3:.1f} ms",
            flush=True,
        )

        rc = 0
        if do_pcc or dump_dir:
            nat = _natural_kv(runtime, kv_cache, total)
            if dump_dir:
                import torch

                Path(dump_dir).mkdir(parents=True, exist_ok=True)
                torch.save(
                    {name: torch.stack(nat[j]).to(torch.bfloat16) for j, name in enumerate(("k", "v", "index_k"))}
                    | {"layers": list(layers), "n_tokens": total, "mesh": [sp, tp]},
                    Path(dump_dir) / "kv.pt",
                )
                print(f"[4x2] KV dump -> {dump_dir}/kv.pt", flush=True)
            if do_pcc:
                nat_pcc = [[t[:, :n_pcc] for t in per_layer] for per_layer in nat]
                worst = _kv_pcc(nat_pcc, os.environ["PREFILL_TRACE_DIR"], n_pcc, layers, hf_config, pcc_min)
                rc = 0 if worst >= pcc_min else 1
        print("[zone-prof] DONE", flush=True)
        return rc
    finally:
        for sub in galaxy.get_submeshes():
            ttnn.close_mesh_device(sub)
        ttnn.close_mesh_device(galaxy)


if __name__ == "__main__":
    sys.exit(main())
