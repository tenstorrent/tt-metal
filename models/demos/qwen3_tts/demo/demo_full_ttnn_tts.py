# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS CLI demo — thin wrapper on top of ``models.demos.qwen3_tts.tt.server``.

The reusable server-side implementation (init_server_context, run_inference,
ICL embed builder, Mimi encode/decode, KV-cache allocation, etc.) lives in
``tt/server.py`` so the inference-server runner and any other consumer can
import it directly without dragging in argparse / CLI orchestration.

For backwards compatibility this module re-exports the public server API,
so existing imports of ``models.demos.qwen3_tts.demo.demo_full_ttnn_tts``
continue to resolve to the same names.

Usage:
    python models/demos/qwen3_tts/demo/demo_full_ttnn_tts.py \\
        --text "Hello, how are you today?" \\
        --ref-audio /path/to/reference.wav \\
        --ref-text "Reference audio transcript" \\
        --output /tmp/ttnn_tts_output.wav \\
        --seed 42

    Trace + KV cache + 2CQ are always on.
"""

import argparse
import time
from typing import Optional

import soundfile as sf
import torch

import ttnn

# ---------------------------------------------------------------------------
# Server-side implementation lives in tt/server.py — re-export the public API
# so existing call sites (web_demo.py, runner, tests) keep working.
# ---------------------------------------------------------------------------
from models.demos.qwen3_tts.tt.server import (  # noqa: F401  (re-exported)
    TTSConfig,
    TTSServerContext,
    _argmax_into,
    _DeviceSampler,
    allocate_kv_cache,
    build_cp_decode_trace_h2d_constants,
    build_prefill_attn_mask,
    build_talker_decode_trace_h2d_constants,
    create_icl_embedding_ttnn,
    deallocate_kv_cache,
    decode_audio,
    encode_reference_audio,
    generate_codes_ttnn,
    get_padded_prefill_len,
    init_server_context,
    load_weights,
    run_inference,
    sample_from_tt_vocab_logits,
    sample_token,
    warmup_all_buckets,
    warmup_bucket,
)


def run_full_ttnn_tts(
    text: str,
    ref_audio: str,
    ref_text: str,
    output_path: str = "/tmp/ttnn_tts_output.wav",
    max_new_tokens: int = 256,
    device_id: int = 0,
    language: str = "english",
    greedy: bool = False,
    repetition_penalty: float = 1.0,
    seed: Optional[int] = None,
    ref_cache: str = None,
    trim_frames: int = 4,
    load_cpu_inputs: str = None,
):
    """Run full TTNN TTS pipeline (CLI orchestrator)."""
    demo_start = time.time()
    print("=" * 80)
    print("Full TTNN TTS Demo")
    print("=" * 80)
    print(f"Text: {text}")
    print(f"Reference: {ref_audio}")
    print(f"Max tokens: {max_new_tokens}")
    print(f"Decoding: {'greedy' if greedy else f'sampling (temp=0.9, top_k=50, rep_penalty={repetition_penalty})'}")
    print(f"KV cache: enabled")
    if seed is not None:
        print(f"RNG seed: {seed} (torch.manual_seed before codec generation)")
    else:
        print("RNG seed: default — sampling is non-deterministic; use --seed for repeatable benchmarks")
    print()

    timings = {}

    # Load weights
    load_start = time.time()
    main_weights, decoder_weights = load_weights()
    timings["load_weights"] = time.time() - load_start

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", trust_remote_code=True)

    # Open device with explicit trace region.
    print(f"\nOpening TT device {device_id}...")
    _ncq = 2
    device = ttnn.open_device(
        device_id=device_id,
        l1_small_size=32768,
        trace_region_size=200000000,
        num_command_queues=_ncq,
    )
    device.enable_program_cache()

    try:
        print("\nInitializing TTNN model...")
        init_start = time.time()

        from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS

        model = Qwen3TTS(device=device, state_dict=main_weights)
        timings["model_init"] = time.time() - init_start
        print(f"  Model initialized in {timings['model_init']:.2f}s")

        config = TTSConfig()
        config.max_new_tokens = max_new_tokens
        config.greedy = greedy
        config.repetition_penalty = repetition_penalty
        config.trim_codec_frames = trim_frames

        if load_cpu_inputs:
            print(f"\n  Loading CPU-computed ICL inputs from: {load_cpu_inputs}")
            cpu_data = torch.load(load_cpu_inputs, map_location="cpu", weights_only=True)
            inputs_embeds_cpu = cpu_data["inputs_embeds"].float()
            trailing_text_hidden = cpu_data["trailing_text_hidden"].float()
            tts_pad_embed = cpu_data["tts_pad_embed"].float()
            if "ref_codes" in cpu_data:
                ref_codes_original = cpu_data["ref_codes"]
            else:
                ref_codes_original, _ = encode_reference_audio(ref_audio, main_weights, cache_path=ref_cache)
            ref_codes = ref_codes_original
            inputs_embeds_tt = ttnn.from_torch(
                inputs_embeds_cpu.unsqueeze(1),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            code_pred_embeds = []
            for i in range(config.num_code_groups - 1):
                key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
                if key in main_weights:
                    code_pred_embeds.append(main_weights[key].float())
            print(f"  inputs_embeds: {inputs_embeds_cpu.shape}")
            print(f"  trailing_text_hidden: {trailing_text_hidden.shape}")
            print(f"  code_pred_embeds: {len(code_pred_embeds)}")
            timings["encode_ref"] = 0.0
            timings["speaker_embed"] = 0.0
            timings["icl_embed"] = 0.0
        else:
            encode_start = time.time()
            ref_codes_original, audio_data = encode_reference_audio(ref_audio, main_weights, cache_path=ref_cache)
            ref_codes = ref_codes_original
            timings["encode_ref"] = time.time() - encode_start

            spk_start = time.time()
            speaker_embedding = model.extract_speaker_embedding(audio_data)
            timings["speaker_embed"] = time.time() - spk_start
            print(f"  Speaker embedding: {speaker_embedding.shape} (extracted with TTNN)")

            icl_start = time.time()
            inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, code_pred_embeds = create_icl_embedding_ttnn(
                target_text=text,
                ref_text=ref_text,
                ref_codes=ref_codes,
                speaker_embedding=speaker_embedding,
                tokenizer=tokenizer,
                model=model,
                device=device,
                config=config,
                main_weights=main_weights,
                language=language,
            )
            timings["icl_embed"] = time.time() - icl_start

        if seed is not None:
            torch.manual_seed(seed)

        from models.demos.qwen3_tts.reference.functional import (
            SpeechTokenizerDecoderConfig,
            speech_tokenizer_decoder_forward,
        )
        from models.demos.qwen3_tts.tt.generator import StreamingAudioDecoder

        _decoder_cfg = SpeechTokenizerDecoderConfig()

        def _streaming_decoder_fn(codes_input: torch.Tensor) -> torch.Tensor:
            codes_filtered = codes_input.clone().clamp(max=2047)
            return speech_tokenizer_decoder_forward(codes_filtered, decoder_weights, _decoder_cfg)

        streaming_decoder = StreamingAudioDecoder(_streaming_decoder_fn, chunk_size=50, sample_rate=24000)
        streaming_decoder.start()
        for _ref_frame in ref_codes_original:
            streaming_decoder.add_tokens(_ref_frame.long())

        gen_start = time.time()
        codes, compile_timings = generate_codes_ttnn(
            model=model,
            device=device,
            inputs_embeds_tt=inputs_embeds_tt,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            code_pred_embeds=code_pred_embeds,
            config=config,
            streaming_decoder=streaming_decoder,
        )
        timings["generation"] = time.time() - gen_start
        timings["warmup"] = compile_timings["warmup"]
        timings["trace_capture"] = compile_timings["trace_capture"]
        timings["avg_decode_ms"] = compile_timings.get("avg_decode_ms", 0.0)
        timings["steady_avg_decode_ms"] = compile_timings.get("steady_avg_decode_ms", 0.0)
        timings["steady_frames_per_sec"] = compile_timings.get("steady_frames_per_sec", 0.0)

        if codes is None:
            print("ERROR: Failed to generate codes")
            return

        ref_codes_len = ref_codes_original.shape[0]
        codes_for_decode = torch.cat([ref_codes_original, codes], dim=0)
        total_codes_len = codes_for_decode.shape[0]
        print(f"  Decoding: {ref_codes_len} ref (original) + {len(codes)} gen = {total_codes_len} total frames")

        decode_start = time.time()
        _drain_t0 = time.time()
        while not streaming_decoder.token_queue.empty() and time.time() - _drain_t0 < 5.0:
            time.sleep(0.001)
        audio = streaming_decoder.get_all_audio()
        streaming_decoder.stop()
        timings["decode"] = time.time() - decode_start

        audio_np = audio.squeeze().detach().cpu().float().numpy()
        cut_samples = int(ref_codes_len / total_codes_len * len(audio_np))
        audio_np = audio_np[cut_samples:]
        print(f"  HF-style trim: removed {cut_samples} samples ({cut_samples/24000:.2f}s) of reference")

        sf.write(output_path, audio_np, 24000)

        # Summary
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        num_frames = len(codes) if codes is not None else 0
        inference_time = (
            timings["speaker_embed"]
            + timings["icl_embed"]
            + timings["generation"]
            - timings.get("warmup", 0.0)
            - timings.get("trace_capture", 0.0)
        )

        print(f"\n{'Phase':<30} {'Time (ms)':<15} {'Component'}")
        print("-" * 70)
        print(f"{'Load weights':<30} {timings['load_weights']*1000:>10.1f}   PyTorch")
        print(f"{'Model init':<30} {timings['model_init']*1000:>10.1f}   TTNN")
        print(f"{'Encode ref audio':<30} {timings['encode_ref']*1000:>10.1f}   Reference (Speech Tok Enc)")
        print(f"{'  Warmup (compile)':<30} {timings.get('warmup', 0)*1000:>10.1f}   TTNN [excluded from inference]")
        print(f"{'  Trace capture':<30} {timings.get('trace_capture', 0)*1000:>10.1f}   TTNN [excluded from inference]")
        print(f"{'Speaker embedding':<30} {timings['speaker_embed']*1000:>10.1f}   TTNN")
        print(f"{'ICL embedding':<30} {timings['icl_embed']*1000:>10.1f}   TTNN")
        print(f"{'Generation (' + str(num_frames) + ' frames)':<30} {timings['generation']*1000:>10.1f}   TTNN")
        print(f"{'Decode audio':<30} {timings['decode']*1000:>10.1f}   Reference (Speech Tok Dec)")
        print("-" * 70)
        print(f"{'Inference time (no compile)':<30} {inference_time*1000:>10.1f}   speaker+ICL+prefill+decode")
        print(f"{'2 CQ (H2D / trace overlap)':<30} {'yes':>10}   device queues")
        if timings.get("avg_decode_ms", 0) > 0:
            print(f"{'Avg decode (all steps)':<30} {timings['avg_decode_ms']:>10.1f}   ms/frame (fair vs other runs)")
        if timings.get("steady_avg_decode_ms", 0) > 0:
            print(
                f"{'Steady decode (step 2+)':<30} {timings['steady_avg_decode_ms']:>10.1f}   ms/frame (excludes 1st decode)"
            )
            print(
                f"{'Steady throughput':<30} {timings['steady_frames_per_sec']:>10.2f}   frames/sec (matches line above)"
            )
        print(
            "  Note: Total generation ms scales with EOS frame count when sampling; compare steady ms/sec across runs."
        )
        print("  (TTFT and decode throughput breakdown printed above during generation)")

        total_time = time.time() - demo_start
        print(f"\nOutput saved to: {output_path}")
        print(f"Audio duration: {len(audio_np) / 24000:.2f}s")
        print(f"Total wall time: {total_time:.2f}s")
        print("=" * 80)

        result = {
            "prefill_ms": float(compile_timings.get("prefill_ms", 0.0)),
            "steady_ms_per_frame": float(compile_timings.get("steady_avg_decode_ms", 0.0)),
            "steady_frames_per_sec": float(compile_timings.get("steady_frames_per_sec", 0.0)),
            "num_frames": int(num_frames),
            "output_wav": output_path,
        }
    finally:
        ttnn.close_device(device)
        print("\nDevice closed")

    return result


def get_default_reference_path():
    """Get path to included Jim reference audio."""
    import os

    return os.path.join(os.path.dirname(__file__), "jim_reference.wav")


def _load_ref_text_for(ref_audio_path: str) -> str:
    """Return the transcript stored next to a reference audio file, if present.

    Looks for a sibling ``<name>.txt`` of the ref audio. Raises if neither
    --ref-text was given nor a sibling .txt exists, so ad-hoc users supply
    their own transcript explicitly.
    """
    import os

    base, _ = os.path.splitext(ref_audio_path)
    txt_path = base + ".txt"
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            return f.read().strip()
    raise SystemExit(
        f"No --ref-text provided and no sibling transcript at {txt_path}. "
        "Pass --ref-text explicitly when using an ad-hoc reference audio."
    )


def main():
    parser = argparse.ArgumentParser(description="Full TTNN TTS Demo")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio path (default: included jim_reference.wav)",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help=(
            "Reference audio transcript. If unset and --ref-audio is the bundled "
            "jim_reference.wav, falls back to jim_reference.txt next to it."
        ),
    )
    parser.add_argument("--output", type=str, default="/tmp/ttnn_tts_output.wav", help="Output path")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--device-id", type=int, default=0, help="TT device ID")
    parser.add_argument("--language", type=str, default="english", help="Language")
    parser.add_argument(
        "--greedy", action="store_true", help="Use greedy decoding (causes repetitive output - not recommended)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty >1.0 discourages repetition (e.g., 1.1-1.3, default: 1.0)",
    )
    parser.add_argument(
        "--trim-frames",
        type=int,
        default=4,
        help="Codec frames to trim from start (removes reference echo, default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="torch.manual_seed before codec generation (reproducible sampling)",
    )
    parser.add_argument(
        "--ref-cache",
        type=str,
        default=None,
        help="Path to cached reference encoding (.pt). Auto-derived from --ref-audio if not set.",
    )
    parser.add_argument(
        "--load-cpu-inputs",
        type=str,
        default=None,
        help="Load CPU-computed ICL embeddings from .pt file (skips speaker encoder & ICL construction)",
    )
    args = parser.parse_args()

    ref_audio = args.ref_audio if args.ref_audio else get_default_reference_path()
    ref_text = args.ref_text if args.ref_text else _load_ref_text_for(ref_audio)

    run_full_ttnn_tts(
        text=args.text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        output_path=args.output,
        max_new_tokens=args.max_tokens,
        device_id=args.device_id,
        language=args.language,
        greedy=args.greedy,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        ref_cache=args.ref_cache,
        trim_frames=args.trim_frames,
        load_cpu_inputs=args.load_cpu_inputs,
    )


if __name__ == "__main__":
    main()
