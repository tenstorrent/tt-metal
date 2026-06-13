# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""EAGER decode profiler on the REAL generator/paged path (Qwen3.6 BH_GLX, 32-chip).

Why this file exists (the existing harnesses are all unusable for eager per-op
device-kernel profiling of the *production* decode step):

  * ``demo/tracy_perf_1L_delta.py`` / ``demo/tracy_perf_1L_fullattn.py`` call
    ``model.forward(mode="decode")`` directly with ``page_table=None,
    kv_cache=None`` -> the NON-paged decode path -> different KV/SDPA kernels
    than the real demo & server.
  * ``tests/test_decode_perf_intrace.py`` uses the correct paged path but
    captures + replays a TRACE; traced profiling WRAPS per-op device-kernel
    durations into a single ~6.86e8 ns bucket (garbage), and has no eager toggle.
  * ``tests/test_decode_generator_profile.py`` delegates to the full demo whose
    long-context prefill seq-scan (``gated_delta_attn_seq``) overflows L1 by
    ~900 B under the profiler and crashes BEFORE decode is reached.

This harness instead drives ``Generator.prefill_forward_text`` (eager, OUTSIDE
the profiled region) once to prime the paged KV + DeltaNet recurrent state,
then runs ``Generator.decode_forward(..., enable_trace=False, ...)`` EAGER for a
handful of steps INSIDE a ``signpost("start")`` / ``signpost("stop")`` window.
Eager => Tracy records true per-op device-kernel durations (no trace wrap).

It is configurable down to a tiny layer count (``QWEN36_N_LAYERS``) and to a
SINGLE block type so the per-op tables are attributable to ONE block:
``QWEN36_PROFILE_LAYER_TYPE=linear`` (GDN, default) or ``full`` (full-attention).

Run command (exact):

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && QWEN36_N_LAYERS=1 QWEN36_PROFILE_LAYER_TYPE=linear \\
           python -m tracy -p -v -r --op-support-count 20000 -m pytest \\
               --noconftest \\
               models/demos/qwen3_6_galaxy_v2/tests/profile_decode_eager.py -s

Env knobs:
  * QWEN36_N_LAYERS            (default 1)        decoder block count to build.
  * QWEN36_PROFILE_DECODE_STEPS(default 4)        eager decode steps inside the signpost window.
  * QWEN36_PROFILE_LAYER_TYPE  (default 'linear') 'linear' -> GDN block(s); 'full' -> full-attention block(s).
  * QWEN36_PROFILE_ISL         (default 128)      prime-prefill ISL (kept short on purpose).
"""
from __future__ import annotations

import faulthandler
import os
import threading
import time


def _arm_teardown_watchdog(seconds=150):
    """Arm a watchdog that force-exits the process if teardown (or any post-decode
    code, incl. fixture teardown) blocks past ``seconds``.

    The full Generator/paged-model decode HANGS in teardown after a mid-decode
    TT_FATAL (the ``__del__`` chain: ``tt_ccl.close`` ->
    ``reset_sub_device_stall_group``, trace releases, ``close_mesh_device`` all
    issue device ops that block when the device has pending work). Killing such
    a process externally WEDGES the fabric and forces a human reset. This
    watchdog self-terminates the process instead, so we NEVER need an external
    kill.

    CRITICAL: the hang is a GIL-holding C-level device call, so a pure-Python
    ``threading.Timer`` callback can NEVER run (it can't preempt the GIL). We use
    TWO mechanisms that DO fire from outside the Python interpreter loop:
      1. ``faulthandler.dump_traceback_later`` — a dedicated C watchdog thread that
         dumps the stuck traceback to stderr after ``seconds`` (works under a held
         GIL). With ``exit=True`` it then calls ``_exit(1)`` at the C level,
         terminating the process without an external kill even mid-C-call.
      2. ``signal.setitimer(ITIMER_REAL)`` as a backstop — delivers SIGALRM to the
         process; the default SIGALRM action terminates it. (Belt and suspenders;
         the C-level faulthandler is the primary.)
    A daemon ``threading.Timer`` is kept too for the case where the hang is in
    Python (GIL released) so we still get a clean ``os._exit(0)``.
    """
    # Primary: C-level watchdog thread — fires even while the GIL is held in a
    # blocking device call. dump_traceback_later(exit=True) hard-exits at the C
    # level after dumping where it was stuck.
    try:
        faulthandler.dump_traceback_later(seconds, exit=True)
    except Exception:  # noqa: BLE001
        pass
    # Backstop: SIGALRM (delivered by the kernel, independent of the GIL).
    try:
        import signal

        signal.setitimer(signal.ITIMER_REAL, seconds + 10)
    except Exception:  # noqa: BLE001
        pass

    # Tertiary: Python Timer for the GIL-released case (clean exit code 0).
    def _boom():
        faulthandler.dump_traceback()
        os._exit(0)

    t = threading.Timer(seconds, _boom)
    t.daemon = True
    t.start()
    return t


import pytest
import torch

import ttnn

# Reuse the proven generator-demo helpers + the 32-chip mesh fixture. Importing
# ``bh_glx_mesh`` registers the fixture in this module; the rest are the exact
# model-build / prompt-load / config constants the demo (and server) use.
from models.demos.qwen3_6_galaxy_v2.demo.text_demo_qwen36 import (  # noqa: F401
    _PAGED_BLOCK_SIZE,
    _PAGED_MAX_NUM_BLOCKS,
    _SNAPSHOT,
    _build_tt_model_paged_kv,
    _load_full_state_dict,
    _load_prompt_for_isl,
    bh_glx_mesh,
)

_N_LAYERS = int(os.environ.get("QWEN36_N_LAYERS", "1"))
_DECODE_STEPS = int(os.environ.get("QWEN36_PROFILE_DECODE_STEPS", "4"))
_LAYER_TYPE = os.environ.get("QWEN36_PROFILE_LAYER_TYPE", "linear").lower()
_ISL = int(os.environ.get("QWEN36_PROFILE_ISL", "128"))

assert _LAYER_TYPE in ("linear", "full"), f"QWEN36_PROFILE_LAYER_TYPE must be 'linear' or 'full', got {_LAYER_TYPE!r}"


def _canonical_layer_indices(layer_type: str, n_layers: int) -> list[int]:
    """Pick ``n_layers`` source layer indices from the canonical
    ``[linear, linear, linear, full] * 16`` Qwen3.6 pattern that match the
    requested block type, so the HF ``self_attn.*`` / DeltaNet keys we relabel
    line up with what the single-type TtTransformer expects.

      * 'full'   -> slots 3, 7, 11, ...   (every 4th, offset 3)
      * 'linear' -> slots 0, 1, 2, 4, 5, 6, 8, ...  (the non-full slots)

    For ``n_layers == 1`` this yields ``[3]`` for full (mirrors
    ``tracy_perf_1L_fullattn.py`` / ``test_layer3_full_attention_forward_pcc.py``,
    which load layer 3's full-attention weights into slot 0) and ``[0]`` for GDN.
    """
    canonical = ["linear", "linear", "linear", "full"] * 16  # 64 slots
    matching = [i for i, t in enumerate(canonical) if t == layer_type]
    if n_layers > len(matching):
        raise ValueError(
            f"requested {n_layers} '{layer_type}' layers but only {len(matching)} exist in the 64-layer pattern"
        )
    return matching[:n_layers]


def _relabel_layers_to_contiguous(state_dict: dict, src_indices: list[int]) -> dict:
    """Re-index the chosen source decoder layers to contiguous slots 0..n-1 so
    the reduced TtTransformer (which only builds ``n_layers`` blocks and reads
    ``model.language_model.layers.{0..n-1}.*``) finds the keys.

    Mirrors ``_relabel_layer_idx`` in ``test_layer3_full_attention_forward_pcc.py``
    and the layer re-index loop in ``tracy_perf_1L_fullattn.py``. Non-layer keys
    (embed_tokens, final norm, lm_head) pass through untouched.
    """
    # src_indices[k] -> destination slot k. Keep only the chosen source layers'
    # weights (drop the other 60+ layers' keys so they can't collide on re-index).
    src_to_dst = {src: dst for dst, src in enumerate(src_indices)}
    src_prefixes = {src: f"model.language_model.layers.{src}." for src in src_indices}
    all_layer_prefix = "model.language_model.layers."

    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith(all_layer_prefix):
            # Is this key one of our chosen source layers?
            matched = False
            for src, pfx in src_prefixes.items():
                if k.startswith(pfx):
                    dst = src_to_dst[src]
                    out[f"model.language_model.layers.{dst}." + k[len(pfx) :]] = v
                    matched = True
                    break
            # Drop keys for non-chosen layers entirely.
            if not matched:
                continue
        else:
            out[k] = v
    return out


@pytest.mark.hardware
def test_profile_decode_eager(bh_glx_mesh):
    """Eager (un-traced) paged-generator decode profile, single block type, tiny N_LAYERS."""
    from transformers import AutoTokenizer

    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig
    from models.demos.qwen3_6_galaxy_v2.tt.generator import Generator
    from models.demos.qwen3_6_galaxy_v2.tt.generator_vllm import allocate_vllm_kv_cache

    try:
        from tracy import signpost
    except ImportError:
        signpost = lambda *_a, **_k: None  # noqa: E731

    # Bake the known-good qwen3.6 decode-CCL config defaults (same set as the
    # generator demo) so the eager decode runs through the production kernels.
    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
        "QWEN36_FULLATTN_WO_TUNED": "1",
        "QWEN36_DELTA_OP_TUNED": "1",
        "QWEN36_CCL_NUM_LINKS_DELTA": "2",
    }.items():
        os.environ.setdefault(_k, _v)

    pattern_type = "linear_attention" if _LAYER_TYPE == "linear" else "full_attention"
    pattern = [pattern_type] * _N_LAYERS
    src_indices = _canonical_layer_indices(_LAYER_TYPE, _N_LAYERS)
    print(
        f"[profile-eager] layer_type={_LAYER_TYPE} ({pattern_type})  n_layers={_N_LAYERS}  "
        f"decode_steps={_DECODE_STEPS}  ISL={_ISL}  src_layer_indices={src_indices}"
    )

    tok = AutoTokenizer.from_pretrained(str(_SNAPSHOT), trust_remote_code=True)
    prompt = _load_prompt_for_isl(_ISL)
    ids = tok(prompt, return_tensors="pt").input_ids
    T_prompt = int(ids.shape[-1])
    if T_prompt > _ISL:
        ids = ids[:, :_ISL]
        T_prompt = _ISL
    real_tokens = ids[:, :T_prompt].to(torch.long)  # [1, T_prompt]
    print(f"[profile-eager] real prompt tokens = {T_prompt}")

    # ---- load + relabel weights to the chosen contiguous single-type layers ----
    state_dict = _load_full_state_dict(_SNAPSHOT)
    state_dict = _relabel_layers_to_contiguous(state_dict, src_indices)
    print(f"[profile-eager] state dict relabeled to {len(src_indices)} contiguous '{_LAYER_TYPE}' layer(s)")

    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    # use_paged_kv_cache=True -> the Generator (prefill_forward_text / decode_forward)
    # + tt-inference-server contract.
    model, args = _build_tt_model_paged_kv(bh_glx_mesh, state_dict, pattern, _N_LAYERS, paged_attention_config)
    print(f"[profile-eager] TT model built (paged kv cache, pattern={pattern})")

    # Host page table (reverse-permutation block map) — identical to the generator demo.
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        args.max_batch_size, paged_attention_config.max_num_blocks // args.max_batch_size
    )
    _kv_shape = (
        paged_attention_config.max_num_blocks,
        1,  # num_kv_heads_per_dev (allocate_vllm_kv_cache rebuilds the row-shard)
        paged_attention_config.block_size,
        args.head_dim,
    )
    tt_kv_cache = allocate_vllm_kv_cache(
        _kv_shape, torch.bfloat16, args.n_layers, model, args.weight_cache_path(ttnn.bfloat8_b)
    )

    generator = Generator(model, args, bh_glx_mesh, tokenizer=tok)
    # Server contract: prefill runs eager (no prefill trace), warmup pre-done.
    generator._disable_prefill_tracing = True
    generator.prefill_warmup_completed = True

    # =====================================================================
    # PRIME: one EAGER prefill OUTSIDE the profiled region (NOT signposted).
    # This seeds the paged KV (full-attn) + DeltaNet recurrent/conv state
    # (GDN). The seq-scan prefill kernel is fine here: eager + un-signposted,
    # exactly like the warm prefill the older tracy scripts ran successfully.
    # =====================================================================
    print("[profile-eager] PRIME prefill_forward_text (eager, return_logits, un-signposted) ...")
    _t0 = time.perf_counter()
    prefill_logits = generator.prefill_forward_text(
        real_tokens,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[T_prompt],
        enable_trace=False,
        sampling_params=None,  # return_logits -> host first-token argmax
    )
    ttnn.synchronize_device(bh_glx_mesh)
    print(f"[profile-eager] PRIME prefill done in {(time.perf_counter() - _t0) * 1000.0:.1f} ms")

    _logits = torch.as_tensor(prefill_logits).float().reshape(-1)[: args.vocab_size]
    first_decode_token = int(_logits.argmax().item())
    print(f"[profile-eager] first decode token (greedy from prefill) = {first_decode_token}")

    # Flush the device profiler DRAM ring buffer after the un-profiled prime so the
    # subsequent eager-decode events start from a clean buffer (mirrors tracy_perf_1L_*).
    ttnn.ReadDeviceProfiler(bh_glx_mesh)

    # =====================================================================
    # PROFILED REGION: EAGER decode steps (enable_trace=False). This routes
    # through Generator._decode_forward_no_trace_text -> the real paged decode
    # forward, so Tracy captures true per-op device-kernel durations (NO trace
    # wrap). sampling_params=None -> return_logits + host argmax (greedy), the
    # same eager isolation path the demo's QWEN36_GEN_HOST_SAMPLE=1 branch uses.
    # =====================================================================
    current_pos = torch.tensor([T_prompt], dtype=torch.long)
    cur_tok = first_decode_token
    step_ms: list[float] = []
    next_toks: list[int] = []

    signpost("start")
    try:
        for it in range(_DECODE_STEPS):
            _t0 = time.perf_counter()
            out = generator.decode_forward(
                torch.tensor([cur_tok], dtype=torch.long).reshape(1, 1),
                current_pos,
                enable_trace=False,  # EAGER — the whole point (no traced duration wrap).
                page_table=page_table,
                kv_cache=tt_kv_cache,
                read_from_device=True,
                sampling_params=None,  # return logits, host-sample (greedy) -> deterministic, eager.
                reset_inputs=True,
                batch_size=1,
            )
            ttnn.synchronize_device(bh_glx_mesh)
            dt = (time.perf_counter() - _t0) * 1000.0
            step_ms.append(dt)
            _logits = out[0] if isinstance(out, (tuple, list)) else out
            _l = torch.as_tensor(_logits).float().reshape(-1)[: args.vocab_size]
            cur_tok = int(_l.argmax().item())
            next_toks.append(cur_tok)
            current_pos = current_pos + 1
            print(f"[profile-eager]   decode step {it}: {dt:.2f} ms  (next_tok={cur_tok})")
        signpost("stop")
        print(f"[profile-eager] next_tok sequence = {next_toks}")
    finally:
        # Arm the teardown watchdog: any hang in the rest of the test + the
        # fixture/__del__ teardown chain is now bounded to ~150s and the process
        # exits on its own — NEVER needs an external kill (which wedges the fabric).
        _arm_teardown_watchdog(150)

    mean_ms = sum(step_ms) / len(step_ms) if step_ms else 0.0
    print("\n[profile-eager] === summary ===")
    print(f"[profile-eager]   layer_type     : {_LAYER_TYPE} ({pattern_type})")
    print(f"[profile-eager]   n_layers       : {_N_LAYERS}")
    print(f"[profile-eager]   decode steps   : {_DECODE_STEPS} (EAGER, enable_trace=False)")
    print(f"[profile-eager]   per-step wall  : {[round(x, 2) for x in step_ms]} ms")
    print(f"[profile-eager]   mean wall      : {mean_ms:.2f} ms/step")

    # Reduced layer count -> generated text is intentionally incoherent; the
    # device-op capture is what matters. Swallow any coherence assertion the way
    # test_decode_generator_profile.py does (there is none here, but guard anyway
    # in case the generator path adds one downstream).
    assert first_decode_token >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
