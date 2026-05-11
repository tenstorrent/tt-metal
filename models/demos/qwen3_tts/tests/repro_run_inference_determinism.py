"""Minimal repro: are two consecutive ``run_inference`` calls with identical
inputs bit-identical?

The server shows ECAPA cos drift (0.94–0.98) across requests with the same text
and voice. Demo (fresh process) is byte-identical run-to-run. This pins the
question: is ``run_inference`` itself non-deterministic across calls in the
same process, or is the variance introduced by something the runner does
around it (ICL build, voice resolve, etc.)?

Usage:
  cd tt-metal && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=wormhole_b0
  source python_env/bin/activate
  python models/demos/qwen3_tts/tests/repro_run_inference_determinism.py
"""
import os
import sys

os.environ.setdefault("TT_QWEN3_CP_FP32", "1")


def main():
    import torch
    from transformers import AutoTokenizer

    import ttnn
    from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import (
        TTSConfig,
        create_icl_embedding_ttnn,
        encode_reference_audio,
        init_server_context,
        load_weights,
        run_inference,
    )
    from models.demos.qwen3_tts.demo.reference_icl_utils import trim_reference_for_icl_conditioning
    from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS

    print("[repro] loading weights...")
    main_weights, _ = load_weights()

    print("[repro] opening device...")
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=512_000_000, num_command_queues=2)
    device.enable_program_cache()

    print("[repro] building model...")
    model = Qwen3TTS(device=device, state_dict=main_weights)

    config = TTSConfig(max_new_tokens=128)
    config.repetition_penalty = 1.0
    config.greedy = bool(int(os.environ.get("REPRO_GREEDY", "1")))
    print(f"[repro] config: greedy={config.greedy}, rep_penalty={config.repetition_penalty}")

    print("[repro] init_server_context (warm + capture traces)...")
    ctx = init_server_context(device, model, config, main_weights)

    # Build ONE set of identical inputs; reuse for both calls.
    ref_path = "models/demos/qwen3_tts/demo/jim_reference.wav"
    ref_text = "Jason, can you put up the high level overview slides."
    target_text = "Hello, this is a quick test of on device sampling for the Talker decode loop."
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    ref_codes, audio_data = encode_reference_audio(ref_path, main_weights=None)
    ref_codes_trim, audio_trim = trim_reference_for_icl_conditioning(
        ref_codes, audio_data, tokenizer, ref_text, target_text
    )

    # One speaker embedding shared across both calls (server-style).
    speaker_embedding = model.extract_speaker_embedding(audio_data)

    # ── PROBE A: back-to-back Talker prefill, identical inputs, ONE call apart ──
    # Build ICL once, snapshot input handle, zero-reset KV, run prefill twice,
    # compare prefill logits directly. Bypasses sampling, decode traces, ICL build.
    print("[repro] === PROBE A: back-to-back Talker prefill, same inputs ===")
    inputs_embeds_tt_a, _, _, _ = create_icl_embedding_ttnn(
        target_text=target_text,
        ref_text=ref_text,
        ref_codes=ref_codes_trim,
        speaker_embedding=speaker_embedding,
        tokenizer=tokenizer,
        model=model,
        device=device,
        config=config,
        main_weights=main_weights,
    )
    real_seq_len = inputs_embeds_tt_a.shape[2]

    def _padded(L):
        from models.demos.qwen3_tts.tt.server import get_padded_prefill_len

        return get_padded_prefill_len(L)

    pad = _padded(real_seq_len)
    if pad > real_seq_len:
        pass

        talker_h = model.talker_config.hidden_size
        pad_zeros = ttnn.from_torch(
            torch.zeros(1, 1, pad - real_seq_len, talker_h, dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        inputs_embeds_tt_a = ttnn.concat([inputs_embeds_tt_a, pad_zeros], dim=2)

    from models.demos.qwen3_tts.tt.rope import get_rope_tensors

    head_dim = model.talker_config.head_dim

    def _run_prefill():
        # Zero-reset Talker KV cache (mirror run_inference)
        kv_caches = ctx.talker_kv_caches_by_bucket[pad]
        zero_hosts = ctx.talker_kv_zero_hosts_by_bucket[pad]
        for (_zk, _zv), (_kc, _vc) in zip(zero_hosts, kv_caches):
            ttnn.copy_host_to_device_tensor(_zk, _kc)
            ttnn.copy_host_to_device_tensor(_zv, _vc)

        prefill_pos = torch.arange(pad)
        cos_tt, sin_tt = get_rope_tensors(device, head_dim, pad, prefill_pos, model.talker_config.rope_theta)
        ttnn.synchronize_device(device)
        hidden, _ = model.talker.forward_from_hidden(
            inputs_embeds_tt_a,
            cos_tt,
            sin_tt,
            ctx.talker_trans_mat,
            kv_caches=kv_caches,
            start_pos=0,
            mode="prefill",
        )
        logits = model.talker.get_codec_logits(hidden)
        ttnn.synchronize_device(device)
        return ttnn.to_torch(logits).squeeze(1).float()[0, real_seq_len - 1, :].clone()

    pf_logits_1 = _run_prefill()
    pf_logits_2 = _run_prefill()
    pf_logits_3 = _run_prefill()
    diff_12 = (pf_logits_1 - pf_logits_2).abs().max().item()
    diff_23 = (pf_logits_2 - pf_logits_3).abs().max().item()
    same_top1_12 = pf_logits_1.argmax().item() == pf_logits_2.argmax().item()
    print(
        f"[repro]   prefill_logits |1-2|max={diff_12:.6e}  |2-3|max={diff_23:.6e}  "
        f"argmax_1={pf_logits_1.argmax().item()} argmax_2={pf_logits_2.argmax().item()} "
        f"argmax_3={pf_logits_3.argmax().item()} top1_match_1_2={same_top1_12}"
    )

    # PROBE B is implemented by interleaving _run_prefill calls with full
    # run_inference passes below.

    print("[repro] === PROBE B: prefill_before -> run_inference -> prefill_after ===")
    pf_before = _run_prefill()
    inputs_embeds_tt_1, trailing_1, pad_1, _ = create_icl_embedding_ttnn(
        target_text=target_text,
        ref_text=ref_text,
        ref_codes=ref_codes_trim,
        speaker_embedding=speaker_embedding,
        tokenizer=tokenizer,
        model=model,
        device=device,
        config=config,
        main_weights=main_weights,
    )
    codes_1, _, _ = run_inference(ctx, model, device, inputs_embeds_tt_1, trailing_1, pad_1, config, use_2cq=True)
    print(f"[repro]   codes_1: shape={tuple(codes_1.shape)}  first5_code0={codes_1[:5, 0].tolist()}")
    pf_after = _run_prefill()
    diff_ba = (pf_before - pf_after).abs().max().item()
    print(
        f"[repro] PROBE B result: pf_before.argmax={pf_before.argmax().item()} "
        f"pf_after.argmax={pf_after.argmax().item()} |before-after|max={diff_ba:.6e}  "
        f"drift={'YES' if diff_ba > 0 else 'NO'}"
    )

    print("[repro] === Pass 2: rebuild ICL + run_inference ===")
    inputs_embeds_tt_2, trailing_2, pad_2, _ = create_icl_embedding_ttnn(
        target_text=target_text,
        ref_text=ref_text,
        ref_codes=ref_codes_trim,
        speaker_embedding=speaker_embedding,
        tokenizer=tokenizer,
        model=model,
        device=device,
        config=config,
        main_weights=main_weights,
    )
    codes_2, _, _ = run_inference(ctx, model, device, inputs_embeds_tt_2, trailing_2, pad_2, config, use_2cq=True)
    print(f"[repro]   codes_2: shape={tuple(codes_2.shape)}  first5_code0={codes_2[:5, 0].tolist()}")

    # Compare.
    if codes_1.shape != codes_2.shape:
        print(f"[repro] FAIL — different shapes: {codes_1.shape} vs {codes_2.shape}")
        diverge_at = "shape"
    else:
        eq = torch.equal(codes_1, codes_2)
        if eq:
            print("[repro] PASS — codes_1 == codes_2 bit-equal")
            diverge_at = None
        else:
            mismatch = (codes_1 != codes_2).any(dim=-1)
            first_div = int(mismatch.nonzero(as_tuple=True)[0][0])
            print(f"[repro] FAIL — first diverging frame: {first_div}")
            print(f"[repro]   codes_1[{first_div}] = {codes_1[first_div].tolist()}")
            print(f"[repro]   codes_2[{first_div}] = {codes_2[first_div].tolist()}")
            diverge_at = first_div

    ttnn.close_device(device)
    return 0 if diverge_at is None else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
