# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-user M3 prefill bench for the 16-chip (4x4, SP4 x TP4, EP16) prefill partition.

Target request: ``BENCH_NEW`` (5120) new tokens on top of ``BENCH_PREFIX`` (51200) cached tokens, run through
the same TtPrefillRuntime.prefill_chunk path the production runner (models/demos/common/prefill) uses, with the
production mesh bring-up (FABRIC_2D, RELAXED_INIT, 6144 B router payload, L1_SMALL 1152) and the production
per-layer ack behaviour (``ttnn.synchronize_device`` + callback after every layer) unless overridden.

Flow:
  1. open the mesh, build the 60-layer model from the tilized cache, allocate the KV cache, runtime.compile()
  2. prefill the prefix [0, PREFIX) in chunk-size pieces (chunk-aligned starts; the last piece ragged when
     PREFIX % CHUNK != 0) and sync  -> "PREFIX" line (untimed for the metric)
  3. time the request [PREFIX, PREFIX + NEW) ``BENCH_ITERS`` times (+1 untimed warm-up): the request starts at
     actual_start=PREFIX (mid-slab when PREFIX % CHUNK != 0, as the engine's prefix reuse sends it), one
     prefill_chunk per chunk-size piece, the last piece ragged. Each repeat rewrites the same cache rows with
     the same values (deterministic), so the prefix is filled once.
     -> "REQUEST ... wall median X ms ... tok/s" (the target metric; production-equivalent E2E when
        BENCH_LAYER_ACK=sync)
  4. optional: a per-layer timing pass, a logits pass (LM head on the request rows, for teacher-forced
     agreement) and a KV dump of [0, PREFIX + NEW) for the L2 KV gate (compare with kagent_prefill_compare.py).

Env (defaults = the deployed 16p16d prefill config):
  BENCH_MESH          rows x cols                                                    [4x4]
  M3_FABRIC           1d | 2d | 1d_ring | 2d_torus_xy (production: 2d)              [2d]
  M3_CCL_TOPOLOGY     linear | ring                                                  [linear]
  BENCH_CHUNK         prefill chunk size (production 4096)                           [4096]
  BENCH_CAPACITY      KV capacity in tokens, multiple of the chunk (production 1M)   [1048576]
  BENCH_PREFIX        cached tokens before the request                               [51200]
  BENCH_NEW           request tokens                                                 [5120]
  BENCH_TOKENS        dir with metadata.json {"token_ids": [...]} (>= PREFIX+NEW, tiled otherwise)  [required]
  BENCH_ITERS         timed repeats of the request                                   [5]
  BENCH_LAYER_ACK     sync  : production — synchronize_device + callback after every layer
                      event : device fence — record a MeshEvent after every layer; a publisher thread
                              event_synchronize()s it and then acks (same ordering, host runs ahead)
                      none  : no per-layer ack (galaxy_prefill_kv_pcc.py LAST CHUNK style)       [sync]
  BENCH_NUM_LAYERS    build only the first N layers                                  [60]
  BENCH_LAYER_IDS     explicit global layer ids (profiling: e.g. 0,1,2,3,4,5)        [unset]
  BENCH_PER_LAYER     1 -> extra request pass with per-layer host timestamps         [1]
  BENCH_LOGITS        1 -> extra request pass with the LM head; top-5 per request row saved to the dump dir [0]
  BENCH_DUMP_DIR      dir for kv_{k,v,ik}.pt (bf16, natural order, [L, H, PREFIX+NEW, 128]) + logits_top5.pt [unset]
  BENCH_PROFILE       1 -> device-profiler zones: the timed request runs once inside the `profiled_chunk` zone
                      (run under `python -m tracy -r -p -v --no-web-server`); BENCH_ITERS forced to 1     [0]
  BENCH_SWEEP         interleaved in-process A/B after the main timing: "name:ack=sync;name2:ack=event,ROPE_FUSED=1"
                      (keys: ack, ROPE_FUSED, SKIP_IDX_SPLIT, MOE_SINGLE_RS); BENCH_SWEEP_ROUNDS repeats   [unset]
  BENCH_FINAL         sweep variant the per-layer / logits / dump passes run with      [the env config]
  BENCH_FINALS        comma list of sweep variants: for each, re-fill the prefix under it, then the logits pass
                      and the KV dump into BENCH_DUMP_DIR/<variant> (several gated dumps from one model build)
  BENCH_RESULTS_JSONL append one JSON line with every number                         [unset]
  EXPERT_DTYPE        bf4 | bf8                                                      [bf4]
"""

import json
import math
import os
import resource
import statistics
import sys
import time
from pathlib import Path

if os.getenv("BENCH_PROFILE", "0") == "1":
    os.environ.setdefault("M3_PROFILE_ZONES", "1")
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")

import torch  # noqa: E402
from loguru import logger  # noqa: E402

import ttnn  # noqa: E402
from models.demos.common.prefill.chunk_layout import rotated_chunk_positions  # noqa: E402
from models.demos.minimax_m3.tt.ccl import L1_SMALL_SIZE  # noqa: E402
from models.demos.minimax_m3.utils.fabric_env import ccl_topology_from_env, fabric_config_from_env  # noqa: E402

FABRIC_PAYLOAD_SIZE = 6144  # MiniMaxM3Config.FABRIC_PAYLOAD_SIZE (production runner_utils.open_mesh_device)


def log(msg):
    print(f"[m3-bench] {msg}", flush=True)


def _raise_nproc_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
        except (ValueError, OSError):
            pass


def open_mesh(rows, cols):
    """Production bring-up (models/demos/common/prefill/runners/runner_utils.open_mesh_device)."""
    fabric = fabric_config_from_env(default="2d")
    router = ttnn._ttnn.fabric.FabricRouterConfig()
    router.max_packet_payload_size_bytes = FABRIC_PAYLOAD_SIZE
    ttnn.set_fabric_config(
        fabric,
        ttnn.FabricReliabilityMode.RELAXED_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        router,
    )
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(rows, cols), l1_small_size=L1_SMALL_SIZE, trace_region_size=0
    )
    log(f"mesh {tuple(mesh.shape)} ndev={mesh.get_num_devices()} fabric={fabric} ccl={ccl_topology_from_env()}")
    return mesh


def maybe_stop(phase):
    """Graceful early exit between phases: touch BENCH_STOP_FILE and the run closes the mesh and exits rc 3
    (never kill a device job -- tt-partition-run treats a signal death as a hang and sets the dirty flag)."""
    f = os.getenv("BENCH_STOP_FILE")
    if f and os.path.exists(f):
        log(f"STOP file {f} present before {phase}: exiting cleanly")
        raise SystemExit(3)


def load_tokens(n):
    d = os.environ["BENCH_TOKENS"]
    src = json.load(open(Path(d) / "metadata.json"))["token_ids"]
    if len(src) < n:
        log(f"WARNING: {d} has {len(src)} tokens < {n}: tiling cyclically")
    return [src[i % len(src)] for i in range(n)]


def main():
    _raise_nproc_limit()
    rows, cols = (int(x) for x in os.getenv("BENCH_MESH", "4x4").lower().split("x"))
    chunk = int(os.getenv("BENCH_CHUNK", "4096"))
    capacity = int(os.getenv("BENCH_CAPACITY", "1048576"))
    prefix = int(os.getenv("BENCH_PREFIX", "51200"))
    new = int(os.getenv("BENCH_NEW", "5120"))
    iters = int(os.getenv("BENCH_ITERS", "5"))
    ack_mode = os.getenv("BENCH_LAYER_ACK", "sync")
    profile = os.getenv("BENCH_PROFILE", "0") == "1"
    per_layer = os.getenv("BENCH_PER_LAYER", "1") == "1" and not profile
    want_logits = os.getenv("BENCH_LOGITS", "0") == "1" and not profile
    dump_dir = os.getenv("BENCH_DUMP_DIR")
    layer_ids = [int(x) for x in os.getenv("BENCH_LAYER_IDS", "").split(",") if x.strip()] or None
    if profile:
        iters = 1
    assert capacity % chunk == 0, f"capacity {capacity} must be a multiple of the chunk {chunk}"
    assert prefix + new <= capacity, "request does not fit the capacity"
    assert ack_mode in ("sync", "event", "none"), ack_mode
    total = prefix + new
    tokens = load_tokens(total)
    env_echo = {
        k: os.environ.get(k)
        for k in (
            "M3_FABRIC",
            "M3_CCL_TOPOLOGY",
            "LOGURU_LEVEL",
            "TT_LOGGER_LEVEL",
            "TT_METAL_SHM_TRACKING_DISABLED",
            "EXPERT_DTYPE",
            "M3_INDEX_CACHE_BF16",
            "TT_METAL_RUNTIME_ROOT",
        )
        + tuple(k for k in os.environ if k.startswith("M3_KA_"))
    }
    log(
        f"config mesh={rows}x{cols} chunk={chunk} capacity={capacity} prefix={prefix} new={new} iters={iters} "
        f"ack={ack_mode} layers={layer_ids or os.getenv('BENCH_NUM_LAYERS', '60')} env={env_echo}"
    )

    mesh = open_mesh(rows, cols)
    result = {
        "tag": os.getenv("BENCH_TAG"),
        "chunk": chunk,
        "capacity": capacity,
        "prefix": prefix,
        "new": new,
        "ack": ack_mode,
        "env": env_echo,
    }
    try:
        from models.demos.minimax_m3.tt.attention import allocate_kv_caches
        from models.demos.minimax_m3.tt.model_config import ModelArgs
        from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
        from models.demos.minimax_m3.tt.weight_cache import weight_cache_is_complete
        from models.demos.minimax_m3.utils.profiler_utils import COARSE, read_profiler, zone

        model_args = ModelArgs(mesh_device=mesh)
        hf_config = model_args.hf_config
        num_layers = len(layer_ids) if layer_ids else int(os.getenv("BENCH_NUM_LAYERS", hf_config.num_hidden_layers))
        if not layer_ids:
            os.environ.setdefault("M3_LOAD_NLAYERS", str(num_layers))
        hf_config.num_hidden_layers = num_layers
        expert_dtype = ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b
        cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
        complete = weight_cache_is_complete(cache_path, hf_config, num_layers, expert_dtype)
        if not complete and os.getenv("M3_WEIGHTS_FROM_CACHE") != "1":
            raise SystemExit(f"weight cache incomplete at {cache_path}; refusing the 869 GB source read")
        cfg = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=capacity,
            mesh_shape=(rows, cols),
            chunk_size=chunk,
            num_users=1,
            expert_weight_dtype=expert_dtype,
            weight_cache_path=cache_path,
            layer_indices=layer_ids,
            topology=ccl_topology_from_env(),
        )
        t0 = time.perf_counter()
        runtime = TtPrefillRuntime(mesh, hf_config, {}, cfg)
        result["build_s"] = time.perf_counter() - t0
        kv_cache = allocate_kv_caches(
            mesh, num_layers=num_layers, max_seq_len=capacity, num_users=1, head_dim=hf_config.head_dim
        )
        log(f"model built in {result['build_s']:.1f} s (cache {cache_path})")

        # ---- per-layer ack (production: host callback after synchronize_device) ----------------------------
        layer_stamps = []  # (layer, perf_counter) — filled by the ack sink in sync/event mode
        state = {"profiling_chunk": False}

        def ack_sink(layer_idx):
            layer_stamps.append((layer_idx, time.perf_counter()))
            if profile and not state["profiling_chunk"]:
                read_profiler(mesh)  # drain outside the profiled chunk only

        publishers = {}

        def set_ack(mode):
            # "event" goes through the runtime's own fence (M3_LAYER_ACK_FENCE=event path) so the bench
            # exercises exactly what the production runner would run.
            if mode == "sync":
                runtime.layer_ack_fence = None
                runtime._on_layer_complete = ack_sink
            elif mode == "event":
                if "event" not in publishers:
                    from models.demos.minimax_m3.tt.tt_prefill_runtime import EventLayerAckFence

                    publishers["event"] = EventLayerAckFence()
                runtime.layer_ack_fence = publishers["event"]
                runtime._on_layer_complete = ack_sink
            else:
                runtime.layer_ack_fence = None
                # profiling needs the per-layer drains before the chunk
                runtime._on_layer_complete = ack_sink if profile else None

        set_ack(ack_mode)

        def sync():
            ttnn.synchronize_device(mesh)
            for pub in publishers.values():
                pub.drain()

        t0 = time.perf_counter()
        runtime.compile(kv_cache)
        sync()
        result["compile_s"] = time.perf_counter() - t0
        log(f"compile (3 warm-up variants) {result['compile_s']:.1f} s")

        def run_span(start, end, **kw):
            """prefill [start, end) in chunk-size pieces; piece boundaries follow the chunk grid of the
            first start only when start is chunk-aligned (prefix fill), else start, start+chunk, ... (request)."""
            outs = []
            a = start
            while a < end:
                b = min(a + chunk, end)
                piece = tokens[a:b] + [0] * (chunk - (b - a))
                inp = runtime.make_chunk_input(piece, a)
                outs.append((a, b, runtime.prefill_chunk(inp, kv_cache, slot_id=0, actual_start=a, actual_end=b, **kw)))
                a = b
            return outs

        maybe_stop("prefix fill")
        # ---- prefix fill -------------------------------------------------------------------------------------
        t0 = time.perf_counter()
        run_span(0, prefix)
        sync()
        result["prefix_s"] = time.perf_counter() - t0
        log(f"PREFIX {prefix} tok filled in {result['prefix_s']*1e3:.1f} ms ({prefix / result['prefix_s']:.0f} tok/s)")

        # ---- the timed request ---------------------------------------------------------------------------------
        times, enq = [], []
        for it in range(iters + 1):
            layer_stamps.clear()
            if profile:
                state["profiling_chunk"] = True
            t0 = time.perf_counter()
            with zone("profiled_chunk", COARSE):
                run_span(prefix, total)
                t_enq = time.perf_counter()
                sync()
            dt = time.perf_counter() - t0
            if profile:
                state["profiling_chunk"] = False
                read_profiler(mesh)
            if it == 0 and not profile:
                log(f"request warm-up (untimed): {dt*1e3:.1f} ms")
                continue
            times.append(dt)
            enq.append(t_enq - t0)
            log(f"request iter {len(times)-1}: wall {dt*1e3:.1f} ms  host-enqueue-return {enq[-1]*1e3:.1f} ms")
        med = statistics.median(times)
        result.update(
            request_ms=[t * 1e3 for t in times],
            request_ms_median=med * 1e3,
            request_ms_min=min(times) * 1e3,
            request_ms_max=max(times) * 1e3,
            request_tok_s=new / med,
            enqueue_ms_median=statistics.median(enq) * 1e3,
        )
        log(
            f"REQUEST {new} tok @ {prefix} cache (chunk {chunk}, capacity {capacity}, ack {ack_mode}): "
            f"wall median {med*1e3:.1f} ms [min {min(times)*1e3:.1f}, max {max(times)*1e3:.1f}] over {len(times)} "
            f"-> {new/med:.1f} tok/s; host enqueue-return median {statistics.median(enq)*1e3:.1f} ms"
        )

        # ---- interleaved in-process A/B sweep (same model, cache and host state; variants alternate) -----------
        # BENCH_SWEEP="name:ack=sync;name2:ack=event,ROPE_FUSED=1;..." keys: ack and any kagent_flags boolean.
        # Each variant rewrites the same request rows from the same prefix, so every repeat is self-consistent.
        from models.demos.minimax_m3.utils import kagent_flags

        flag_names = ("ROPE_FUSED", "SKIP_IDX_SPLIT", "MOE_SINGLE_RS")
        flag_defaults = {f: getattr(kagent_flags, f) for f in flag_names}

        mm_default = set(kagent_flags.MM_FIDELITY)

        def apply_variant(d):
            set_ack(d.get("ack", ack_mode))
            for f in flag_names:
                setattr(kagent_flags, f, d[f] == "1" if f in d else flag_defaults[f])
            # MM_FIDELITY=qkv+shared (groups joined with '+')
            kagent_flags.MM_FIDELITY = (
                set(filter(None, d["MM_FIDELITY"].split("+"))) if "MM_FIDELITY" in d else set(mm_default)
            )

        variants = []
        for spec in filter(None, os.getenv("BENCH_SWEEP", "").split(";")):
            name, _, kv = spec.partition(":")
            variants.append((name, dict(x.split("=") for x in kv.split(",") if x)))
        if variants:
            maybe_stop("sweep")
            rounds = int(os.getenv("BENCH_SWEEP_ROUNDS", "5"))
            sw = {name: [] for name, _ in variants}
            for r in range(rounds + 1):  # round 0 = warm-up (program-cache misses of new variants)
                for name, d in variants:
                    apply_variant(d)
                    t0 = time.perf_counter()
                    run_span(prefix, total)
                    sync()
                    if r:
                        sw[name].append(time.perf_counter() - t0)
            base_name = variants[0][0]
            base_med = statistics.median(sw[base_name])
            result["sweep"] = {}
            for name, d in variants:
                m = statistics.median(sw[name])
                result["sweep"][name] = {"spec": d, "ms": [x * 1e3 for x in sw[name]], "median_ms": m * 1e3}
                log(
                    f"SWEEP {name:<14} {json.dumps(d):<60} median {m*1e3:8.1f} ms [min {min(sw[name])*1e3:.1f}, "
                    f"max {max(sw[name])*1e3:.1f}] -> {new/m:7.1f} tok/s ({(m/base_med-1)*100:+.1f}% vs {base_name})"
                )
        final = dict(variants).get(os.getenv("BENCH_FINAL", ""), {}) if variants else {}
        apply_variant(final)  # the variant the per-layer / logits / dump passes below run with
        if final:
            log(f"per-layer / logits / dump passes run variant {os.getenv('BENCH_FINAL')}: {final}")
            # re-fill the prefix under the final variant so a dump reflects that variant end to end
            run_span(0, prefix)
            sync()

        # ---- per-layer pass: layer wrapper timestamps (host enqueue per layer) + ack timestamps --------------
        if per_layer:
            enq_stamps = []
            layers = runtime.model.layers
            cls = type(layers[0])
            cls_call = cls.__call__
            idx_of = {id(L): i for i, L in enumerate(layers)}

            def patched(self, *a, **k):
                s = time.perf_counter()
                out = cls_call(self, *a, **k)
                enq_stamps.append((idx_of[id(self)], s, time.perf_counter()))
                return out

            cls.__call__ = patched
            layer_stamps.clear()
            t0 = time.perf_counter()
            run_span(prefix, total)
            sync()
            wall = time.perf_counter() - t0
            cls.__call__ = cls_call
            # per-layer host enqueue (sum over request chunks) and per-layer ack-to-ack interval
            enq_by_layer = [0.0] * len(layers)
            for i, s, e in enq_stamps:
                enq_by_layer[i] += e - s
            ack_iv = []
            prev = t0
            for i, t in layer_stamps:
                ack_iv.append((i, t - prev))
                prev = t
            ack_by_layer = [0.0] * len(layers)
            for i, d in ack_iv:
                ack_by_layer[i] += d
            result["per_layer"] = {
                "wall_ms": wall * 1e3,
                "host_enqueue_ms": [x * 1e3 for x in enq_by_layer],
                "ack_interval_ms": [x * 1e3 for x in ack_by_layer] if layer_stamps else None,
            }
            dense = [x * 1e3 for i, x in enumerate(enq_by_layer) if runtime.model.layers[i].is_dense]
            sparse = [x * 1e3 for i, x in enumerate(enq_by_layer) if not runtime.model.layers[i].is_dense]
            msg = (
                f"PER-LAYER pass wall {wall*1e3:.1f} ms; host enqueue per layer (sum over request chunks): "
                f"dense {statistics.mean(dense) if dense else 0:.2f} ms, sparse median "
                f"{statistics.median(sparse) if sparse else 0:.2f} ms (total {sum(enq_by_layer)*1e3:.1f} ms)"
            )
            if layer_stamps:
                ad = [x * 1e3 for i, x in enumerate(ack_by_layer) if runtime.model.layers[i].is_dense]
                asp = [x * 1e3 for i, x in enumerate(ack_by_layer) if not runtime.model.layers[i].is_dense]
                msg += (
                    f"; ack interval per layer: dense {statistics.mean(ad) if ad else 0:.2f} ms, sparse median "
                    f"{statistics.median(asp) if asp else 0:.2f} ms"
                )
            log(msg)

        def gated_pass(dump_dir):
            """Logits pass + KV dump of the current variant (prefix must already be filled under it)."""
            # ---- logits pass (LM head on the request rows) --------------------------------------------------------
            if want_logits:
                sp = rows
                top_ids, top_vals, positions, nlls = [], [], [], []
                vocab = hf_config.vocab_size
                composer = ttnn.ConcatMesh2dToTensor(mesh, dims=(2, 3), mesh_shape=mesh.shape)
                # BENCH_CAPTURE_MSA=3,30,59: also keep the MSA top-16 block ids those layers select for the request
                # rows (realistic index patterns for op-level sparse_sdpa_msa work).
                capture = {int(x) for x in os.getenv("BENCH_CAPTURE_MSA", "").split(",") if x.strip()}
                captured = {}
                if capture:
                    import models.demos.minimax_m3.tt.attention.msa as msa_mod

                    orig_ixs = msa_mod.msa_indexer_sparse
                    layer_cls = type(runtime.model.layers[0])
                    orig_layer_call = layer_cls.__call__
                    cur = {"layer": None}

                    def layer_call(self, *a, **k):
                        cur["layer"] = self.layer_idx
                        return orig_layer_call(self, *a, **k)

                    def ixs(*a, **k):
                        if cur["layer"] in capture:
                            o, ids = orig_ixs(*a, return_block_ids=True, **k)
                            captured.setdefault(cur["layer"], []).append(ids)
                            return o
                        return orig_ixs(*a, **k)

                    layer_cls.__call__ = layer_call
                    msa_mod.msa_indexer_sparse = ixs
                spans = []
                for a, b, out in run_span(prefix, total, skip_lm_head=False):
                    spans.append((a, b))
                    logits = ttnn.to_torch(out, mesh_composer=composer).float()[0, 0]  # [chunk, padded_vocab]
                    ttnn.deallocate(out)
                    pos = [p for row in rotated_chunk_positions(a, sp, chunk // sp) for p in row]
                    keep = [j for j, p in enumerate(pos) if p < b]
                    lg = logits[keep, :vocab]
                    v, ix = torch.topk(lg, 5, dim=-1)
                    top_ids.append(ix)
                    top_vals.append(v)
                    kp = [pos[j] for j in keep]
                    positions += kp
                    # teacher-forced NLL of the true next prompt token (-1 past the end -> nan)
                    tgt = torch.tensor([tokens[p_ + 1] if p_ + 1 < total else 0 for p_ in kp])
                    nll_c = torch.logsumexp(lg, dim=-1) - lg.gather(1, tgt[:, None])[:, 0]
                    nll_c[torch.tensor([p_ + 1 >= total for p_ in kp])] = float("nan")
                    nlls.append(nll_c)
                sync()
                if capture:
                    layer_cls.__call__ = orig_layer_call
                    msa_mod.msa_indexer_sparse = orig_ixs
                    ids_comp = ttnn.ConcatMesh2dToTensor(mesh, dims=(2, 1), mesh_shape=mesh.shape)
                    cap_out = {}
                    for L, lst in captured.items():
                        rows_pos, rows_ids = [], []
                        for (a, b), ids in zip(spans, lst):
                            h = ttnn.to_torch(ids, mesh_composer=ids_comp)  # [1, tp, chunk, topk]
                            pos = [p for row in rotated_chunk_positions(a, sp, chunk // sp) for p in row]
                            keep = [j for j, p in enumerate(pos) if p < b]
                            rows_pos += [pos[j] for j in keep]
                            rows_ids.append(h[0][:, keep].to(torch.int64) & 0xFFFFFFFF)
                        order_c = torch.tensor(rows_pos).argsort()
                        cap_out[L] = {
                            "positions": torch.tensor(rows_pos)[order_c],
                            "block_ids": torch.cat(rows_ids, dim=1)[
                                :, order_c
                            ],  # [kv_heads(tp), rows, topk] natural ids
                        }
                    if dump_dir:
                        os.makedirs(dump_dir, exist_ok=True)
                        torch.save(cap_out, Path(dump_dir) / "msa_block_ids.pt")
                    log(f"captured MSA block ids for layers {sorted(cap_out)}")
                order = torch.tensor(positions).argsort()
                top_ids = torch.cat(top_ids)[order]
                top_vals = torch.cat(top_vals)[order]
                positions = torch.tensor(positions)[order]
                nxt = torch.tensor(tokens[prefix + 1 : total] + [-1])
                acc = (top_ids[:-1, 0] == nxt[:-1]).float().mean().item()
                nll = torch.cat(nlls)[order]
                mean_nll = nll[~nll.isnan()].mean().item()
                result.setdefault("teacher_forced_top1_vs_text", {})[str(dump_dir)] = acc
                result.setdefault("teacher_forced_nll", {})[str(dump_dir)] = mean_nll
                log(
                    f"LOGITS pass: {len(positions)} request rows; top-1 == next prompt token on {acc*100:.2f}%; "
                    f"mean NLL {mean_nll:.4f} (ppl {math.exp(mean_nll):.3f})"
                )
                if dump_dir:
                    os.makedirs(dump_dir, exist_ok=True)
                    torch.save(
                        {
                            "positions": positions,
                            "top_ids": top_ids,
                            "top_vals": top_vals,
                            "next_tokens": nxt,
                            "nll": nll,
                        },
                        Path(dump_dir) / "logits_top5.pt",
                    )

            # ---- KV dump ---------------------------------------------------------------------------------------------
            if dump_dir:
                from models.demos.minimax_m3.tt.runners.prefill_kv_validation import naturalize_kv_block

                os.makedirs(dump_dir, exist_ok=True)
                t0 = time.perf_counter()
                blocks = runtime.read_slot_kv(kv_cache, 0, total)
                seq = runtime.read_seq_len(total)
                for name, blk in zip(("k", "v", "ik"), blocks):
                    nat = torch.stack([naturalize_kv_block(blk[L], total, rows, chunk, seq) for L in range(num_layers)])
                    torch.save(nat.to(torch.bfloat16), Path(dump_dir) / f"kv_{name}.pt")
                result["dump_dir"] = dump_dir
                log(f"KV dump [0, {total}) x {num_layers} layers -> {dump_dir} in {time.perf_counter()-t0:.1f} s")
                json.dump(result, open(Path(dump_dir) / "bench_result.json", "w"), indent=1)

        finals = [x for x in os.getenv("BENCH_FINALS", "").split(",") if x]
        if finals:
            root = dump_dir
            for name in finals:
                maybe_stop(f"gated pass {name}")
                apply_variant(dict(variants)[name])
                run_span(0, prefix)  # whole path under this variant
                sync()
                log(f"gated pass for variant {name}: {dict(variants)[name]}")
                gated_pass(os.path.join(root, name) if root else None)
        else:
            gated_pass(dump_dir)

        for pub in publishers.values():
            pub.close()
        if os.getenv("BENCH_RESULTS_JSONL"):
            with open(os.environ["BENCH_RESULTS_JSONL"], "a") as fh:
                fh.write(json.dumps(result) + "\n")
        log("RESULT " + json.dumps({k: v for k, v in result.items() if k != "per_layer"}))
        log("DONE")
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
