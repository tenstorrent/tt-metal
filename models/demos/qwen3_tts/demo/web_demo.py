# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS Web Demo

Starts on port 7777. On startup, pre-compiles all prefill bucket kernels and
pre-captures all Talker + CP traces so every subsequent request has zero
compile or trace-capture overhead.

Usage:
    export ARCH_NAME=wormhole_b0
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    source python_env/bin/activate
    python models/demos/qwen3_tts/demo/web_demo.py
"""

import tempfile
import time
from pathlib import Path

import gradio as gr
import soundfile as sf

import ttnn
from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import (
    TTSConfig,
    TTSServerContext,
    create_icl_embedding_ttnn,
    decode_audio,
    encode_reference_audio,
    init_server_context,
    load_weights,
    run_inference,
)
from models.demos.qwen3_tts.demo.reference_icl_utils import trim_reference_for_icl_conditioning
from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEMO_DIR = Path(__file__).parent
DEFAULT_REF_AUDIO = str(_DEMO_DIR / "jim_reference.wav")
DEFAULT_REF_TEXT = "Jason, can you put up the high level overview slides."

# ---------------------------------------------------------------------------
# Global server state (initialised once at startup)
# ---------------------------------------------------------------------------
_device = None
_model = None
_tokenizer = None
_main_weights = None
_decoder_weights = None
_config = None
_ctx: TTSServerContext = None
_lock = None  # threading.Lock set at startup


def _startup():
    """Load weights, open device, init model, pre-warm all buckets and capture traces."""
    global _device, _model, _tokenizer, _main_weights, _decoder_weights, _config, _ctx, _lock

    import threading

    _lock = threading.Lock()

    from transformers import AutoTokenizer

    print("=" * 70)
    print("Qwen3-TTS Web Demo — Server Startup")
    print("=" * 70)

    t0 = time.time()

    print("\n[1/5] Loading model weights...")
    _main_weights, _decoder_weights = load_weights()

    print("\n[2/5] Loading tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", trust_remote_code=True)

    print("\n[3/5] Opening TT device...")
    _device = ttnn.open_device(
        device_id=0,
        l1_small_size=32768,
        trace_region_size=100_000_000,
    )
    _device.enable_program_cache()

    print("\n[4/5] Initialising TTNN model...")
    _model = Qwen3TTS(device=_device, state_dict=_main_weights)

    _config = TTSConfig(max_new_tokens=1500)
    _config.repetition_penalty = 1.15  # prevents codec token repetition loops on long texts

    print("\n[5/5] Pre-warming all buckets and capturing traces (one-time cost)...")
    _ctx = init_server_context(_device, _model, _config, _main_weights)

    elapsed = time.time() - t0
    print(f"\nServer ready — startup took {elapsed:.1f}s")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Per-request handler
# ---------------------------------------------------------------------------
def generate(ref_audio_path, ref_text: str, target_text: str):
    """
    Gradio handler: synthesise speech and return (wav_path, perf_text).

    ref_audio_path: file path string from gr.Audio (or None → use default)
    ref_text:       transcript of the reference audio
    target_text:    text to synthesise
    """
    if not target_text or not target_text.strip():
        return None, "Please enter text to synthesise."

    if not ref_audio_path:
        ref_audio_path = DEFAULT_REF_AUDIO
    if not ref_text or not ref_text.strip():
        ref_text = DEFAULT_REF_TEXT

    t_request_start = time.time()

    with _lock:
        try:
            # 1. Encode reference audio (uses .refcache.pt if available)
            t0 = time.time()
            ref_codes, audio_data = encode_reference_audio(ref_audio_path, _main_weights)
            ref_codes, audio_data = trim_reference_for_icl_conditioning(
                ref_codes, audio_data, _tokenizer, ref_text, target_text
            )
            t_encode_ref = time.time() - t0

            # 2. Speaker embedding
            t0 = time.time()
            speaker_embedding = _model.extract_speaker_embedding(audio_data)
            t_speaker = time.time() - t0

            # 3. ICL embeddings
            t0 = time.time()
            inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, _ = create_icl_embedding_ttnn(
                target_text=target_text,
                ref_text=ref_text,
                ref_codes=ref_codes,
                speaker_embedding=speaker_embedding,
                tokenizer=_tokenizer,
                model=_model,
                device=_device,
                config=_config,
                main_weights=_main_weights,
            )
            t_icl = time.time() - t0

            # 4. Run inference (zero compile / trace overhead)
            codes, inf_timings, perf_text_core = run_inference(
                ctx=_ctx,
                model=_model,
                device=_device,
                inputs_embeds_tt=inputs_embeds_tt,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                config=_config,
            )

            if codes is None:
                return None, "Generation failed (EOS at prefill or no frames)."

            # Trim leading reference echo (default 4 frames)
            if _config.trim_codec_frames > 0 and len(codes) > _config.trim_codec_frames:
                codes = codes[_config.trim_codec_frames :]

            # 5. Decode to audio
            t0 = time.time()
            audio = decode_audio(codes, _decoder_weights)
            t_audio_decode = time.time() - t0

            audio_np = audio.squeeze().detach().cpu().float().numpy()

            # Save to temp file for Gradio to serve
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp_wav.name, audio_np, 24000)

            # Build full performance summary
            t_total = time.time() - t_request_start
            audio_dur = len(audio_np) / 24000.0
            lines = [
                f"{'Phase':<35} {'Time (ms)':>10}",
                "-" * 48,
                f"{'Encode ref audio':<35} {t_encode_ref*1000:>10.1f}",
                f"{'Speaker embedding':<35} {t_speaker*1000:>10.1f}",
                f"{'ICL embedding':<35} {t_icl*1000:>10.1f}",
            ]
            lines += perf_text_core.splitlines()
            lines += [
                f"{'Audio decode':<35} {t_audio_decode*1000:>10.1f}",
                "-" * 48,
                f"{'Total request time':<35} {t_total*1000:>10.1f}",
                f"{'Audio duration':<35} {audio_dur:>10.2f}s",
            ]
            perf_text = "\n".join(lines)

            return tmp_wav.name, perf_text

        except Exception as e:
            import traceback

            err = traceback.format_exc()
            print(err)
            return None, f"Error: {e}\n\n{err}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def _build_ui():
    with gr.Blocks(title="Qwen3-TTS — Tenstorrent N150") as demo:
        gr.Markdown(
            """
# Qwen3-TTS on Tenstorrent N150
**Voice-cloned text-to-speech** powered by TTNN.\n
Record or upload a reference voice, enter the text you want synthesised, and click **Generate**.
"""
        )

        with gr.Row():
            # ---- Left column: reference ----
            with gr.Column():
                gr.Markdown("### Reference Voice")
                ref_audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Reference Audio",
                )
                ref_text = gr.Textbox(
                    label="Reference Text (what is said in the reference audio)",
                    lines=2,
                    placeholder=DEFAULT_REF_TEXT,
                )
                use_default_btn = gr.Button("🎙 Use Jim's Voice (default)", variant="secondary")

            # ---- Right column: target ----
            with gr.Column():
                gr.Markdown("### Text to Synthesise")
                target_text = gr.Textbox(
                    label="Text",
                    lines=6,
                    placeholder="Enter the text you want to hear spoken...",
                )
                generate_btn = gr.Button("▶ Generate", variant="primary")

        output_audio = gr.Audio(label="Generated Speech", autoplay=True)
        perf_box = gr.Textbox(
            label="Performance",
            interactive=False,
            lines=14,
            placeholder="Timing breakdown will appear here after generation...",
        )

        # ---- Button handlers ----
        def _use_default():
            return str(Path(DEFAULT_REF_AUDIO).resolve()), DEFAULT_REF_TEXT

        use_default_btn.click(
            fn=_use_default,
            inputs=[],
            outputs=[ref_audio, ref_text],
        )

        generate_btn.click(
            fn=generate,
            inputs=[ref_audio, ref_text, target_text],
            outputs=[output_audio, perf_box],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _startup()

    demo = _build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7777,
        share=False,
        allowed_paths=[str(_DEMO_DIR)],
    )
