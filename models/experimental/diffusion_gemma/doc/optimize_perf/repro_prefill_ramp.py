"""Production-faithful N-request harness for the per-request prefill ramp.

The 198-question GPQA run (~/dg_runs/gpqa_full_fp32_fullcanvas, 2026-07-30) shows
``prefill_s`` climbing 0.99 -> 12.16 s over 200 sequential requests at constant prompt
length, while the traced denoise step stays flat at 197.4 ms. Eleven mechanisms were
refuted on device first -- fixed-shape repetition, low free DRAM, fragmentation,
allocator entry count, ``page_table``, async/enqueue timing, program recompilation,
``max_seq_len=16384``, shape churn, metric mis-attribution, and non-idempotent
monkeypatch stacking.

Every one of those repros drove ``prefill_prompt_tokens`` in a bare loop: no denoise,
no commit, and no captured traces. This harness closes that gap by driving the real
vLLM entry points in the real order -- ``prefill_forward`` -> ``decode_forward`` x N ->
``release_request`` -- against ONE startup-captured persistent adapter, exactly as
``generator_vllm`` does in production.

Reads ``session.prefill_time_s`` (the same field that feeds the ``prefill_block0``
metric's ``prefill_s``) before releasing the row, so the number is production's number.

Env: see plan.md section 5. Requires DG_TRACE_REGION_SIZE > 0.

Knobs (the bisect axes):
  HARNESS_REQUESTS   sequential requests to serve            (default 30)
  HARNESS_BLOCKS     256-token blocks emitted per request    (default 3)
  HARNESS_PROMPT_LEN prompt tokens, fixed across requests    (default 232)
  HARNESS_LABEL      arm name for the summary line
  DG_UPFRONT_CAPTURE 0 selects the eager arm (no 48 traces, no persistent adapter)
"""

import gc
import json
import os
import statistics as st
import time

import torch

import ttnn

GIB = 2**30
REQUESTS = int(os.getenv("HARNESS_REQUESTS", "30"))
BLOCKS = int(os.getenv("HARNESS_BLOCKS", "3"))
PROMPT_LEN = int(os.getenv("HARNESS_PROMPT_LEN", "232"))
LABEL = os.getenv("HARNESS_LABEL", "base")
P_MAX = int(os.getenv("DG_DENOISE_REVEAL_PMAX", "16384"))
UPFRONT = os.getenv("DG_UPFRONT_CAPTURE", "1") != "0"


def dram(mesh_device):
    """Free/used DRAM plus any live-block counter the memory view exposes."""
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    banks = view.num_banks
    out = {
        "free_gib": round(banks * view.total_bytes_free_per_bank / GIB, 4),
        "used_gib": round(banks * view.total_bytes_allocated_per_bank / GIB, 4),
    }
    # free-bytes drift is only ~1.28 MiB/request and both fragmentation and allocator
    # entry count are already refuted, so what matters here is the LIVE BLOCK COUNT.
    for name in (
        "num_allocated_blocks",
        "total_num_allocated_blocks",
        "num_blocks",
        "largest_contiguous_block_size_per_bank",
    ):
        val = getattr(view, name, None)
        if val is not None:
            out[name] = val
    return out


def main():
    from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
    from models.experimental.diffusion_gemma.config import DiffusionConfig
    from models.experimental.diffusion_gemma.demo.text_demo import _close_mesh_device, _open_mesh_device
    from models.experimental.diffusion_gemma.tt import generator_vllm

    os.environ.setdefault("DG_VLLM_GUMBEL_MODE", "device")
    os.environ["DG_DENOISE_REVEAL_PMAX"] = str(P_MAX)

    mesh = _open_mesh_device(os.environ.get("DG_MESH", "P150x4"))
    print(f"[ramp] label={LABEL} upfront={UPFRONT} p_max={P_MAX} requests={REQUESTS} blocks={BLOCKS}", flush=True)
    print(f"[ramp] dram after open: {dram(mesh)}", flush=True)

    try:
        bundle = build_tt_model_from_checkpoint_dir(
            mesh,
            os.environ["DG_CKPT"],
            tokenizer_kwargs={"local_files_only": True},
            max_seq_len=P_MAX,
            create_kv_cache=True,
        )
        print(f"[ramp] dram after model build: {dram(mesh)}", flush=True)

        # one fixed aligned prompt length, so any ramp is unambiguous
        aligned = ((PROMPT_LEN + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        os.environ["DG_UPFRONT_PREFILL_WARMUP_LENS"] = f"{ttnn.TILE_SIZE},{aligned}"

        wrapper = generator_vllm.DiffusionGemmaForCausalLM(
            [bundle.tt_model],
            [bundle.model_args],
            bundle.tt_model.mesh_device,
            dg_state_dict=bundle.state_dict,
            tokenizer=bundle.tokenizer,
            config=DiffusionConfig(),
            gumbel_mode="device",
        )
        t_cap = time.perf_counter()
        wrapper.warmup_model_prefill(None, False, True)   # compile-only phase
        wrapper.warmup_model_prefill(None, True, True)    # capture phase
        print(f"[ramp] warmup+capture took {time.perf_counter() - t_cap:.1f}s; dram={dram(mesh)}", flush=True)

        # A REAL tokenized prompt, not random ids. Random ids are gibberish, the canvas
        # never settles, and the degeneracy guard ends the request after 2-3 blocks -- so
        # HARNESS_BLOCKS silently stops mattering and every arm measures the same 3 blocks.
        # (`_make_session` already passes `stop_token_ids=[]`, so EOS is not the cause.)
        from models.experimental.diffusion_gemma.tt.generate import tokenize_prompt

        text = "Explain, step by step and in careful detail, why the sky appears blue during "
        tokens = tokenize_prompt(bundle.tokenizer, text * 24)[:, :PROMPT_LEN]
        if tokens.shape[1] < PROMPT_LEN:
            raise RuntimeError(f"seed text tokenized to {tokens.shape[1]} < {PROMPT_LEN}; lengthen it")
        rows = []
        print(
            f"\n{'req':>4} {'prefill_s':>10} {'ttft_s':>8} {'blocks_s':>9} {'free_gib':>9} ",
            flush=True,
        )
        for req in range(REQUESTS):
            t0 = time.perf_counter()
            wrapper.prefill_forward(tokens, prompt_lens=[PROMPT_LEN])
            ttft_s = time.perf_counter() - t0
            # read prefill_time_s from the LIVE session -- release_request drops it
            session = wrapper._sessions.get(0)
            prefill_s = float(getattr(session, "prefill_time_s", float("nan")))

            t1 = time.perf_counter()
            emitted = 1
            for _ in range(BLOCKS - 1):
                session = wrapper._sessions.get(0)
                # a finished session turns decode_forward into a no-op stop block, which
                # would make blocks_s look flat while emitting nothing -- count for real
                if session is None or getattr(session, "finished", False):
                    break
                wrapper.decode_forward()
                emitted += 1
            blocks_s = time.perf_counter() - t1

            wrapper.release_request(0)
            gc.collect()

            d = dram(mesh)
            live = d.get("num_allocated_blocks", d.get("total_num_allocated_blocks", "-"))
            rows.append({"req": req, "emitted": emitted, "prefill_s": prefill_s, "ttft_s": ttft_s, "blocks_s": blocks_s, **d})
            print(
                f"{req:>4} {prefill_s:>10.3f} {ttft_s:>8.3f} {blocks_s:>9.3f} "
                f"{d['free_gib']:>9.4f} {emitted:>5}",
                flush=True,
            )

        pre = [r["prefill_s"] for r in rows]
        first5, last5 = pre[:5], pre[-5:]
        ratio = st.median(last5) / max(st.median(first5), 1e-9)
        print(f"\n[ramp] label={LABEL} prefill_s first5_p50={st.median(first5):.3f} "
              f"last5_p50={st.median(last5):.3f} ratio={ratio:.2f}x", flush=True)
        print(f"[ramp] free_gib {rows[0]['free_gib']:.4f} -> {rows[-1]['free_gib']:.4f}", flush=True)
        print("[ramp] VERDICT: " + ("RAMP REPRODUCES" if ratio > 2.0 else "no ramp on this arm"), flush=True)

        out = os.getenv("HARNESS_JSON")
        if out:
            with open(out, "w") as fh:
                json.dump({"label": LABEL, "upfront": UPFRONT, "requests": REQUESTS,
                           "blocks": BLOCKS, "prompt_len": PROMPT_LEN, "rows": rows}, fh, indent=2)
            print(f"[ramp] wrote {out}", flush=True)

        if UPFRONT:
            wrapper.release_persistent_capture()
    finally:
        _close_mesh_device(mesh)


if __name__ == "__main__":
    main()
