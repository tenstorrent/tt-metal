"""Is the exhausted resource consumed per PREFILL CALL or per SESSION lifecycle?

The prefill regression triggers at request index ~13 in every multi-request run
(13/13/13/15 across four configs whose wall clock at onset varies 1.8x and whose
blocks-per-request varies 5x), so it is neither time-based nor block-based. But in the
vLLM path a session, a prefill call and a block-0 emission are 1:1:1, so those three
candidates are still tangled.

This decouples them by driving the session API directly: N_SESSIONS session lifecycles,
each doing PREFILLS_PER_SESSION prefills, and NO decode blocks at all.

  spike at the 13th PREFILL   (iteration ~7 at 2/session) -> consumed per prefill call.
      Then look at what prefill allocates once per call and never frees -- the 60
      blocking `ttnn.to_torch` readbacks in sparse_moe._ragged_metadata_host are the
      prime suspect and DG_PREFILL_MOE_RAGGED=0 becomes a real discriminator.
  spike at the 13th SESSION   (iteration 13 at 2/session) -> consumed per lifecycle.
      Then it is attach_persistent_adapter / rebind_prompt / reset, not the MoE readbacks.

Runs with the 48 traces captured and the concat-MoE relayout resident, i.e. production's
memory state (free DRAM ~4.1 GiB), because the four earlier bare-prefill repros lacked
both and never triggered.

Env: see plan.md section 5. Requires DG_TRACE_REGION_SIZE > 0.
"""

import gc
import json
import os
import time

import torch

import ttnn

GIB = 2**30
N_SESSIONS = int(os.getenv("PERCALL_SESSIONS", "13"))
PREFILLS_PER_SESSION = int(os.getenv("PERCALL_PREFILLS", "2"))
PROMPT_LEN = int(os.getenv("PERCALL_PROMPT_LEN", "232"))
P_MAX = int(os.getenv("DG_DENOISE_REVEAL_PMAX", "16384"))


def free_gib(mesh_device):
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return round(view.num_banks * view.total_bytes_free_per_bank / GIB, 4)


def main():
    from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
    from models.experimental.diffusion_gemma.config import DiffusionConfig
    from models.experimental.diffusion_gemma.demo.text_demo import _close_mesh_device, _open_mesh_device
    from models.experimental.diffusion_gemma.tt import generator_vllm
    from models.experimental.diffusion_gemma.tt.generate import tokenize_prompt

    os.environ.setdefault("DG_VLLM_GUMBEL_MODE", "device")
    os.environ["DG_DENOISE_REVEAL_PMAX"] = str(P_MAX)

    mesh = _open_mesh_device(os.environ.get("DG_MESH", "P150x4"))
    print(f"[percall] sessions={N_SESSIONS} prefills/session={PREFILLS_PER_SESSION} "
          f"total_prefills={N_SESSIONS * PREFILLS_PER_SESSION} p_max={P_MAX}", flush=True)
    try:
        bundle = build_tt_model_from_checkpoint_dir(
            mesh, os.environ["DG_CKPT"],
            tokenizer_kwargs={"local_files_only": True},
            max_seq_len=P_MAX, create_kv_cache=True,
        )
        aligned = ((PROMPT_LEN + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        os.environ["DG_UPFRONT_PREFILL_WARMUP_LENS"] = f"{ttnn.TILE_SIZE},{aligned}"

        wrapper = generator_vllm.DiffusionGemmaForCausalLM(
            [bundle.tt_model], [bundle.model_args], bundle.tt_model.mesh_device,
            dg_state_dict=bundle.state_dict, tokenizer=bundle.tokenizer,
            config=DiffusionConfig(), gumbel_mode="device",
        )
        wrapper.warmup_model_prefill(None, False, True)
        wrapper.warmup_model_prefill(None, True, True)
        print(f"[percall] traces captured; free={free_gib(mesh)} GiB", flush=True)

        text = "Explain, step by step and in careful detail, why the sky appears blue during "
        tokens = tokenize_prompt(bundle.tokenizer, text * 24)[:, :PROMPT_LEN]

        rows = []
        n_prefill = 0
        print(f"\n{'sess':>5} {'call':>5} {'nth_prefill':>12} {'prefill_s':>10} {'free_gib':>9}", flush=True)
        for sess_idx in range(N_SESSIONS):
            session = wrapper._make_session()
            session.attach_persistent_adapter(wrapper._persistent_adapter)
            try:
                for call_idx in range(PREFILLS_PER_SESSION):
                    t0 = time.perf_counter()
                    session.prefill(tokens)
                    ttnn.synchronize_device(mesh)
                    wall = time.perf_counter() - t0
                    n_prefill += 1
                    rows.append({
                        "session": sess_idx, "call": call_idx, "nth_prefill": n_prefill,
                        "prefill_time_s": round(float(session.prefill_time_s), 6),
                        "wall_s": round(wall, 6), "free_gib": free_gib(mesh),
                    })
                    print(f"{sess_idx:>5} {call_idx:>5} {n_prefill:>12} "
                          f"{session.prefill_time_s:>10.3f} {rows[-1]['free_gib']:>9.4f}", flush=True)
            finally:
                session.reset()
                gc.collect()

        spikes = [r for r in rows if r["prefill_time_s"] > 2.0]
        print("\n[percall] spikes (>2 s):", flush=True)
        for s in spikes:
            print(f"    session={s['session']} nth_prefill={s['nth_prefill']} "
                  f"prefill_s={s['prefill_time_s']:.3f}", flush=True)
        if not spikes:
            print("    NONE -- neither axis triggered; the resource is not consumed by "
                  "prefill or session alone (blocks/denoise/commit are the remaining axis)", flush=True)
        else:
            first = spikes[0]
            verdict = ("PER PREFILL CALL" if first["nth_prefill"] <= N_SESSIONS
                       else "PER SESSION LIFECYCLE")
            print(f"[percall] first spike at nth_prefill={first['nth_prefill']} "
                  f"session={first['session']} -> {verdict}", flush=True)

        out = os.getenv("PERCALL_JSON")
        if out:
            with open(out, "w") as fh:
                json.dump({"sessions": N_SESSIONS, "prefills_per_session": PREFILLS_PER_SESSION,
                           "rows": rows}, fh, indent=2)
            print(f"[percall] wrote {out}", flush=True)

        wrapper.release_persistent_capture()
    finally:
        _close_mesh_device(mesh)


if __name__ == "__main__":
    main()
