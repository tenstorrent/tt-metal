"""Minimal repro: does ``extract_speaker_embedding`` hang when called after a
Talker decode trace has executed?

Reproduces the ad-hoc voice-clone hang from the inference-server runner without
needing the server stack. Runs inside the parent process (no subprocess) since
the trace-in-subprocess bug is already fixed.

Sequence:
  1. Open device, build model.
  2. Call ``extract_speaker_embedding(audio)`` — first call. Should succeed.
  3. Build + execute a Talker decode trace (mimics what ``run_inference`` does).
  4. Call ``extract_speaker_embedding(audio)`` — SECOND call after trace exec.
     This is where the inference-server runner hangs.

Usage:
  cd tt-metal && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=wormhole_b0
  source python_env/bin/activate
  python models/demos/qwen3_tts/tests/repro_speaker_embedding_after_trace.py
"""
import os
import sys
import time

os.environ.setdefault("TT_QWEN3_CP_FP32", "1")


def main():
    import torch

    import ttnn
    from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import encode_reference_audio, load_weights
    from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS

    # First call uses jim (short, 4s); second post-trace call uses Satoshi
    # (longer, 12s) to mimic the runner's ad-hoc-clone request shape.
    ref_path_short = "models/demos/qwen3_tts/demo/jim_reference.wav"
    ref_path_long = "/local/ttuser/ssinghal/tts2/tts-models/tts-2/prompts/Satoshi_ja.wav"
    ref_path = ref_path_short
    print(f"[repro] short clip: {ref_path_short}")
    print(f"[repro] long clip:  {ref_path_long}")

    print("[repro] loading weights...")
    main_weights, _ = load_weights()

    print("[repro] opening device...")
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=512_000_000, num_command_queues=1)
    device.enable_program_cache()

    print("[repro] building model...")
    model = Qwen3TTS(device=device, state_dict=main_weights)

    print("[repro] encoding reference audio...")
    _, audio_data = encode_reference_audio(ref_path, main_weights=None)
    print(f"[repro] audio_data shape={tuple(audio_data.shape)} samples")

    # ── Mirror runner order: traces FIRST, then ECAPA precompute ─────────
    print("[repro] init_server_context (traces captured, NOT yet executed) ...")
    from transformers import AutoTokenizer

    from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import (
        TTSConfig,
        create_icl_embedding_ttnn,
        init_server_context,
        run_inference,
    )
    from models.demos.qwen3_tts.demo.reference_icl_utils import trim_reference_for_icl_conditioning

    config = TTSConfig(max_new_tokens=64)  # small to keep test quick
    config.repetition_penalty = 1.15
    ctx = init_server_context(device, model, config, main_weights)
    print("[repro] server context ready.")

    # ── TEST A: untraced vs traced ECAPA, same time, same audio ─────────
    print("[repro] === TEST A: untraced vs traced (no run_inference yet) ===")
    _, audio_jim_a = encode_reference_audio(ref_path_short, main_weights=None)
    print("[repro]   pass 1: untraced (SE traces NOT active)")
    emb_untraced = model.extract_speaker_embedding(audio_jim_a)
    print(f"[repro]     untraced norm={emb_untraced.norm().item():.4f}")
    print("[repro]   pass 2: activate traces, run again")
    model.speaker_encoder.activate_traced_extract()
    emb_traced = model.extract_speaker_embedding(audio_jim_a)
    print(f"[repro]     traced   norm={emb_traced.norm().item():.4f}")
    diffA = (emb_untraced - emb_traced).abs().max().item()
    cosA = torch.nn.functional.cosine_similarity(
        emb_untraced.flatten().unsqueeze(0), emb_traced.flatten().unsqueeze(0)
    ).item()
    print(f"[repro]   |untraced - traced|_max = {diffA:.4f}  cos = {cosA:.6f}")
    if cosA > 0.9999:
        print("[repro]   TEST A PASS — traced matches untraced")
    else:
        print("[repro]   TEST A FAIL — trace path itself is lossy")
    # Switch BACK to untraced for the original test.
    model.speaker_encoder._se_traces_active = False

    # ── Mimic runner: 3 ECAPA precomputes AFTER trace capture, BEFORE exec ─
    print("[repro] precompute_speaker_embeddings: jim, ashley, satoshi ...")
    voices = [
        ("jim", "models/demos/qwen3_tts/demo/jim_reference.wav"),
        ("ashley", "/local/ttuser/ssinghal/tts2/tts-models/tts-2/prompts/Ashley_en.wav"),
        ("satoshi", ref_path_long if os.path.isfile(ref_path_long) else ref_path_short),
    ]
    cached_embeds = {}
    for name, path in voices:
        if not os.path.isfile(path):
            print(f"[repro]   skipping {name} ({path} missing)")
            continue
        _, a = encode_reference_audio(path, main_weights=None)
        t0 = time.time()
        cached_embeds[name] = model.extract_speaker_embedding(a)
        print(f"[repro]   cached {name}: shape={tuple(cached_embeds[name].shape)} took {time.time()-t0:.2f}s")
    emb1 = cached_embeds.get("jim")

    print("[repro] running one inference (executes CP + Talker decode traces) ...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    target_text = "Hello, this is a test."
    ref_text = "Jason, can you put up the high level overview slides."
    ref_codes, _ = encode_reference_audio(ref_path, main_weights=None)
    ref_codes_trim, audio_trim = trim_reference_for_icl_conditioning(
        ref_codes, audio_data, tokenizer, ref_text, target_text
    )
    inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, _ = create_icl_embedding_ttnn(
        target_text=target_text,
        ref_text=ref_text,
        ref_codes=ref_codes_trim,
        speaker_embedding=emb1,
        tokenizer=tokenizer,
        model=model,
        device=device,
        config=config,
        main_weights=main_weights,
    )
    codes, _, _ = run_inference(
        ctx, model, device, inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, config, use_2cq=False
    )
    print(f"[repro] inference done, generated {0 if codes is None else len(codes)} frames.")

    # ── Cache hypothesis test: is the prepared-conv-weight cache stale? ──
    cache_size = len(model.speaker_encoder._conv1d_prepared_cache)
    print(f"[repro] prepared-conv cache has {cache_size} entries before CALL 2")
    if os.environ.get("REPRO_CLEAR_CACHE", "0") == "1":
        print("[repro] *** clearing prepared-conv cache (REPRO_CLEAR_CACHE=1) ***")
        model.speaker_encoder._conv1d_prepared_cache.clear()

    # ── Call 2: re-extract on the SAME audio used by precompute(jim) ─────
    # If embeddings are bit-identical, on-device ECAPA is deterministic
    # across init-time and post-trace calls.
    print("[repro] CALL 2: re-extract on jim audio (same as cached jim) ...")
    _, audio_jim = encode_reference_audio(ref_path_short, main_weights=None)
    t0 = time.time()
    emb2 = model.extract_speaker_embedding(audio_jim)
    dt2 = time.time() - t0
    print(f"[repro]   shape={tuple(emb2.shape)} took {dt2:.2f}s")
    if "jim" in cached_embeds and cached_embeds["jim"] is not None:
        a = cached_embeds["jim"]
        b = emb2
        print(
            f"[repro] cached_jim:    min={a.min().item():.6f} max={a.max().item():.6f} norm={a.norm().item():.6f} nan_count={a.isnan().sum().item()} inf_count={a.isinf().sum().item()}"
        )
        print(
            f"[repro] runtime_jim:   min={b.min().item():.6f} max={b.max().item():.6f} norm={b.norm().item():.6f} nan_count={b.isnan().sum().item()} inf_count={b.isinf().sum().item()}"
        )
        diff = (a - b).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
        print(f"[repro] |cached_jim - runtime_jim|_max = {diff:.4f}  cos = {cos:.6f}")
        if cos > 0.99 and not a.isnan().any() and not b.isnan().any() and not a.isinf().any() and not b.isinf().any():
            print(f"[repro] PASS — cos={cos:.4f} (functional embedding match)")
        else:
            print("[repro] FAIL — embeddings DIVERGE between init-time and post-trace")
    ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main() or 0)
