"""Instrumented bisect of the request-13 prefill cliff: WHERE inside prefill does the time go?

The regression triggers at request index ~13 in every multi-request run (13/13/13/15
across four configs whose wall clock at onset varies 1.8x and whose blocks-per-request
varies 5x), so it is neither time-based nor block-based. Twelve mechanisms are refuted.

This answers the one question that cannot be seen from outside: is the ~8-12 s ONE stall
or ~60 small ones? ``prefill_prompt_tokens`` (generate.py:291-319) is
``embed_host_tokens`` -> ``tt_model(is_decode=False)`` -> ``logits.deallocate``, and the
30-layer forward performs 60 blocking ``ttnn.to_torch`` readbacks (two per layer, in
``sparse_moe._ragged_metadata_host`` at :185-189, reached from :340).

Instrumentation is monkeypatched from the harness, never in production code, and is armed
ONLY for the iterations in INSTRUMENT_FROM..INSTRUMENT_TO so it cannot perturb the ramp
that produces the cliff. Per armed iteration it reports:
  * embed / forward / dealloc coarse spans (synchronized),
  * per-readback wall times inside _ragged_metadata_host: count, sum, max, and the worst
    few -- this is what separates one big stall from sixty small ones,
  * live trace count and capture_events (if capture_events moves off 1, recapture is
    happening and that alone explains the cliff),
  * program-cache entry count and free DRAM.

Env: see plan.md section 5. Requires DG_TRACE_REGION_SIZE > 0.
"""

import gc
import json
import os
import time

import torch

import ttnn

GIB = 2**30
REQUESTS = int(os.getenv("INSTR_REQUESTS", "16"))
BLOCKS = int(os.getenv("INSTR_BLOCKS", "3"))
PROMPT_LEN = int(os.getenv("INSTR_PROMPT_LEN", "232"))
FROM = int(os.getenv("INSTR_FROM", "11"))
TO = int(os.getenv("INSTR_TO", "15"))
P_MAX = int(os.getenv("DG_DENOISE_REVEAL_PMAX", "16384"))

_armed = False
_readbacks = []
_allreduces = []


def free_gib(mesh):
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return round(view.num_banks * view.total_bytes_free_per_bank / GIB, 4)


def program_cache_entries(mesh):
    for name in ("num_program_cache_entries", "get_program_cache_size"):
        fn = getattr(mesh, name, None)
        if fn is not None:
            try:
                return fn()
            except Exception:
                pass
    return None


def trace_stats(wrapper):
    adapter = getattr(wrapper, "_persistent_adapter", None)
    ctrl = getattr(adapter, "_upfront_traced_denoise_controller", None)
    if ctrl is None:
        return {}
    try:
        s = ctrl.stats()
        return {k: s[k] for k in ("capture_events", "traces_captured", "execute_trace_calls",
                                  "adapter_rebinds") if k in s}
    except Exception:
        return {}


def install_allreduce_probe(mesh):
    """Time EVERY ttnn.all_reduce, and record program-cache entries around each one.

    This is the decisive measurement. The live stack of a wedged prefill was
    ``ttnn.all_reduce -> reduce_scatter -> ReduceScatterProgram::create_mesh_workload ->
    create_global_semaphore -> GlobalSemaphore::setup_buffer -> reset_semaphore_value ->
    enqueue_write_mesh_buffer -> FDMeshCommandQueue::finish_nolock``, i.e. a cache-missing
    all_reduce mints a GlobalSemaphore whose setup does a synchronous device write that
    drains the whole command queue. Prefill issues ~90 all_reduces (3/layer x 30 layers),
    so this says directly whether the 8-12 s is 90 drains of ~100 ms or one 8 s stall --
    and, by sampling the cache size across each call, whether entries GROW per all_reduce
    (hash includes the semaphore identity) or stay flat (eviction / invalidation).

    Patched at ``ttnn.all_reduce`` rather than at the three DG call sites
    (diffusion_attention.py:491, expert_operations.py:53, concat_moe.py:376) because the
    prefill forward runs the shared gemma4 model, whose collectives are not those sites.
    """
    original = ttnn.all_reduce

    def timed(*args, **kwargs):
        if not _armed:
            return original(*args, **kwargs)
        before = program_cache_entries(mesh)
        t0 = time.perf_counter()
        out = original(*args, **kwargs)
        dt = time.perf_counter() - t0
        _allreduces.append((dt, before, program_cache_entries(mesh)))
        return out

    ttnn.all_reduce = timed
    return original


def install_readback_probe():
    """Time every ttnn.to_torch executed inside sparse_moe._ragged_metadata_host."""
    from models.experimental.diffusion_gemma.tt import sparse_moe

    original = sparse_moe.ttnn.to_torch

    def timed(*args, **kwargs):
        if not _armed:
            return original(*args, **kwargs)
        t0 = time.perf_counter()
        out = original(*args, **kwargs)
        _readbacks.append(time.perf_counter() - t0)
        return out

    sparse_moe.ttnn.to_torch = timed
    return original


def main():
    from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
    from models.experimental.diffusion_gemma.config import DiffusionConfig
    from models.experimental.diffusion_gemma.demo.text_demo import _close_mesh_device, _open_mesh_device
    from models.experimental.diffusion_gemma.tt import generate as dg_generate
    from models.experimental.diffusion_gemma.tt.generate import tokenize_prompt

    global _armed

    os.environ.setdefault("DG_VLLM_GUMBEL_MODE", "device")
    os.environ["DG_DENOISE_REVEAL_PMAX"] = str(P_MAX)

    mesh = _open_mesh_device(os.environ.get("DG_MESH", "P150x4"))
    print(f"[instr] requests={REQUESTS} blocks={BLOCKS} armed={FROM}..{TO} p_max={P_MAX}", flush=True)
    try:
        bundle = build_tt_model_from_checkpoint_dir(
            mesh, os.environ["DG_CKPT"], tokenizer_kwargs={"local_files_only": True},
            max_seq_len=P_MAX, create_kv_cache=True,
        )
        aligned = ((PROMPT_LEN + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        os.environ["DG_UPFRONT_PREFILL_WARMUP_LENS"] = f"{ttnn.TILE_SIZE},{aligned}"

        from models.experimental.diffusion_gemma.tt import generator_vllm

        wrapper = generator_vllm.DiffusionGemmaForCausalLM(
            [bundle.tt_model], [bundle.model_args], bundle.tt_model.mesh_device,
            dg_state_dict=bundle.state_dict, tokenizer=bundle.tokenizer,
            config=DiffusionConfig(), gumbel_mode="device",
        )
        wrapper.warmup_model_prefill(None, False, True)
        wrapper.warmup_model_prefill(None, True, True)
        print(f"[instr] captured; free={free_gib(mesh)} pc_entries={program_cache_entries(mesh)}", flush=True)

        install_readback_probe()
        install_allreduce_probe(mesh)

        # coarse spans inside prefill_prompt_tokens, without touching production code
        spans = {}
        orig_embed = dg_generate.embed_host_tokens

        def timed_embed(*a, **k):
            if not _armed:
                return orig_embed(*a, **k)
            t0 = time.perf_counter()
            out = orig_embed(*a, **k)
            ttnn.synchronize_device(mesh)
            spans["embed_s"] = time.perf_counter() - t0
            return out

        dg_generate.embed_host_tokens = timed_embed

        text = "Explain, step by step and in careful detail, why the sky appears blue during "
        tokens = tokenize_prompt(bundle.tokenizer, text * 24)[:, :PROMPT_LEN]

        rows = []
        for req in range(REQUESTS):
            _armed = FROM <= req <= TO
            _readbacks.clear()
            _allreduces.clear()
            spans.clear()

            t0 = time.perf_counter()
            wrapper.prefill_forward(tokens, prompt_lens=[PROMPT_LEN])
            session = wrapper._sessions.get(0)
            prefill_s = float(getattr(session, "prefill_time_s", float("nan")))

            emitted = 1
            for _ in range(BLOCKS - 1):
                s = wrapper._sessions.get(0)
                if s is None or getattr(s, "finished", False):
                    break
                wrapper.decode_forward()
                emitted += 1
            wrapper.release_request(0)
            gc.collect()

            row = {
                "req": req, "prefill_s": round(prefill_s, 4), "emitted": emitted,
                "free_gib": free_gib(mesh), "pc_entries": program_cache_entries(mesh),
                **trace_stats(wrapper),
            }
            if _armed:
                ar = sorted((d for d, _, _ in _allreduces), reverse=True)
                growth = [(b, a) for _, b, a in _allreduces if b is not None and a is not None and a != b]
                row.update({
                    "ar_count": len(ar),
                    "ar_sum_s": round(sum(ar), 4),
                    "ar_max_s": round(ar[0], 4) if ar else None,
                    "ar_top5_s": [round(x, 4) for x in ar[:5]],
                    # how many all_reduces ADDED a program-cache entry -> hypothesis A
                    "ar_grew_cache": len(growth),
                })
                rb = sorted(_readbacks, reverse=True)
                row.update({
                    "rb_count": len(rb),
                    "rb_sum_s": round(sum(rb), 4),
                    "rb_max_s": round(rb[0], 4) if rb else None,
                    "rb_top5_s": [round(x, 4) for x in rb[:5]],
                    **{k: round(v, 4) for k, v in spans.items()},
                })
            rows.append(row)
            print(f"[instr] {json.dumps(row)}", flush=True)

        out = os.getenv("INSTR_JSON")
        if out:
            with open(out, "w") as fh:
                json.dump(rows, fh, indent=2)
            print(f"[instr] wrote {out}", flush=True)

        print("\n[instr] program-cache entry curve (A=unbounded growth, B=plateau+evict, C=flat+miss):", flush=True)
        print("    " + " ".join(f"{r['req']}:{r['pc_entries']}" for r in rows), flush=True)

        spiked = [r for r in rows if r["prefill_s"] > 2.0]
        print(f"\n[instr] spiked requests: {[r['req'] for r in spiked]}", flush=True)
        for r in spiked:
            if "rb_sum_s" in r:
                share_rb = 100.0 * r["rb_sum_s"] / max(r["prefill_s"], 1e-9)
                share_ar = 100.0 * r.get("ar_sum_s", 0.0) / max(r["prefill_s"], 1e-9)
                print(f"[instr] req {r['req']}: prefill {r['prefill_s']:.3f}s | "
                      f"{r.get('ar_count')} all_reduce summing {r.get('ar_sum_s')}s ({share_ar:.0f}%), "
                      f"worst {r.get('ar_max_s')}s, {r.get('ar_grew_cache')} grew the cache | "
                      f"{r['rb_count']} readbacks summing {r['rb_sum_s']:.3f}s ({share_rb:.0f}%), "
                      f"worst {r['rb_max_s']:.3f}s", flush=True)
                if r.get("ar_max_s") and r["ar_max_s"] > 0.5 * r["prefill_s"]:
                    print("[instr]   -> ONE all_reduce dominates: a single drain, not 90", flush=True)
                elif r.get("ar_sum_s", 0) > 0.5 * r["prefill_s"]:
                    print("[instr]   -> SPREAD across all_reduces: many drains", flush=True)

        wrapper.release_persistent_capture()
    finally:
        _close_mesh_device(mesh)


if __name__ == "__main__":
    main()
