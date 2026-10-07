# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Device speech decoder sharing the chip with the traced talker (serving path).

Mirrors tt-media-server's qwen3_tts_runner: production device config, the full
``init_server_context`` (talker / CP / ECAPA traces), and real ICL requests through
``run_inference`` before every decode. Each decode is scored against the CPU
reference on the same codes.

``order=before`` builds and warms the decoder before any trace is captured, as the
runner must. ``mode=continue`` is the serving default (host front-end from a cached
reference state, device back-end over 12 context + generated frames); ``mode=full``
runs the whole decoder on device over cat(ref, generated). ``order=after`` builds it after capture and is a negative control.
Whether the talker's own traces free memory where a late decoder lands depends on the
talker stack's allocation order (the original stack did; the PR #56212 generation stack
does not), so the control also captures a "scribble" trace right before building the
decoder: 64 intermediates of 8 MiB, all alive during capture and freed together after
it, so the late decoder is allocated into memory the scribble trace rewrites every time
it replays (before each decode). The control must fail (strict xfail); if it ever
passes, this test no longer detects the hazard.

Run (~2 min per case):
    pytest -s models/demos/qwen3_tts/tests/test_device_decoder_coexistence.py
"""

import math
import os
import time

import pytest
import torch

import ttnn

os.environ.setdefault("TT_QWEN3_CP_FP32", "1")  # as tt-media-server

from models.demos.qwen3_tts.tt import server as api

HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MAX_NEW_TOKENS = 256
MIN_SNR_DB = float(os.environ.get("TT_QWEN3_DECODE_MIN_SNR", "15"))
REQUESTS = ("Hello, this is a test.", "Could we push the meeting back to Thursday?")

# Qwen3TTSConstants in tt-media-server.
SERVING_DEVICE_PARAMS = {
    "l1_small_size": int(os.environ.get("TT_QWEN3_L1_SMALL_SIZE", "32768")),
    "trace_region_size": 512_000_000,
    "num_command_queues": 2,
}


def _capture_scribble_trace(device, count: int = 64, rows: int = 4096):
    """Capture a trace whose ``count`` intermediates (``rows`` x 1024 bf16, 8 MiB each) are
    freed after capture. Anything allocated afterwards can land on them; replaying the trace
    rewrites them. Returns the trace id; the caller replays it with ``ttnn.execute_trace``."""
    import ttnn

    x = ttnn.from_torch(
        torch.ones(1, 1, rows, 1024, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def body():
        held, y = [], x
        for _ in range(count):
            y = ttnn.multiply(y, 1.0001, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            held.append(y)
        return held

    for t in body():  # compile outside the trace
        ttnn.deallocate(t)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    held = body()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    for t in held:  # freed after capture: the region a late allocation reuses
        ttnn.deallocate(t)
    ttnn.synchronize_device(device)
    return tid, x


def _snr_db(test: torch.Tensor, ref: torch.Tensor) -> float:
    test, ref = test.flatten().float(), ref.flatten().float()
    n = min(test.numel(), ref.numel())
    noise = (ref[:n] - test[:n]).pow(2).mean().item()
    return 10.0 * math.log10(ref[:n].pow(2).mean().item() / noise) if noise > 0 else math.inf


@pytest.mark.parametrize("device_params", [SERVING_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "order, mode",
    [
        ("before", "continue"),
        ("before", "full"),
        pytest.param(
            "after",
            "continue",
            marks=pytest.mark.xfail(strict=True, reason="negative control: built after capture"),
        ),
    ],
)
def test_device_decoder_survives_talker_traces(device, order, mode):
    from transformers import AutoTokenizer

    from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import get_default_reference_path
    from models.demos.qwen3_tts.tt.model_config import talker_config_for_hf_id
    from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS

    main_weights, decoder_weights = api.load_weights(HF_ID)
    ref_wav = get_default_reference_path()
    ref_text = open(os.path.splitext(ref_wav)[0] + ".txt", encoding="utf-8").read().strip()
    ref_codes, audio_data = api.encode_reference_audio(ref_wav, main_weights=None)

    def build_decoder():
        decoder, _ = api.prepare_device_decoder(
            device, decoder_weights, int(ref_codes.shape[0]), MAX_NEW_TOKENS, icl_continue=mode == "continue"
        )
        return decoder

    decoder = build_decoder() if order == "before" else None

    talker_config = talker_config_for_hf_id(HF_ID)
    model = Qwen3TTS(device=device, state_dict=main_weights, talker_config=talker_config)
    config = api.TTSConfig(max_new_tokens=MAX_NEW_TOKENS)
    config.greedy = False
    config.repetition_penalty = 1.15
    config.hidden_size = talker_config.hidden_size
    ctx = api.init_server_context(device, model, config, main_weights)

    scribble = None
    if decoder is None:
        scribble = _capture_scribble_trace(device)
        decoder = build_decoder()

    tokenizer = AutoTokenizer.from_pretrained(HF_ID, trust_remote_code=True, revision=api.hf_revision(HF_ID))
    speaker_embedding = model.extract_speaker_embedding(audio_data)
    ref_state = api.prepare_icl_decoder_state(ref_codes, decoder_weights)
    torch.manual_seed(0)

    results = []
    for text in REQUESTS:
        inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, _ = api.create_icl_embedding_ttnn(
            target_text=text,
            ref_text=ref_text,
            ref_codes=ref_codes,
            speaker_embedding=speaker_embedding,
            tokenizer=tokenizer,
            model=model,
            device=device,
            config=config,
            main_weights=main_weights,
            language="english",
        )
        codes, _, _ = api.run_inference(
            ctx=ctx,
            model=model,
            device=device,
            inputs_embeds_tt=inputs_embeds_tt,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            config=config,
            use_2cq=True,
        )
        if scribble is not None:
            ttnn.execute_trace(device, scribble[0], cq_id=0, blocking=True)
        t0 = time.perf_counter()
        if mode == "continue":
            dev = api.decode_icl_audio(ref_codes, codes, decoder_weights, ref_state=ref_state, device_decoder=decoder)
        else:
            dev = api.decode_audio_device(ref_codes, codes, decoder)
        decode_ms = (time.perf_counter() - t0) * 1e3
        cpu = api.decode_icl_audio(ref_codes, codes, decoder_weights, ref_state=ref_state)
        assert dev.shape[-1] == cpu.shape[-1] == codes.shape[0] * 1920
        snr = _snr_db(dev, cpu)
        results.append(
            (text, int(codes.shape[0]), snr, dev.pow(2).mean().sqrt().item(), cpu.pow(2).mean().sqrt().item())
        )
        print(
            f"[{order}/{mode}] {text!r}: {codes.shape[0]} frames, decode {decode_ms:.0f} ms, snr={snr:.2f} dB, "
            f"rms dev/cpu={results[-1][3]:.4f}/{results[-1][4]:.4f}"
        )

    if scribble is not None:
        ttnn.release_trace(device, scribble[0])

    for text, frames, snr, _, _ in results:
        assert snr >= MIN_SNR_DB, f"{text!r} ({frames} frames): {snr:.2f} dB < {MIN_SNR_DB} dB"
