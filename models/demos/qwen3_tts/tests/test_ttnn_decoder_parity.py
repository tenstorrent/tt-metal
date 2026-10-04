# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""On-device speech-decoder parity harness (branch: tvardhineni/sdawle/qwen3-tts-decoder-on-device).

Goal: quantify how far the TTNN speech decoder (``SpeechTokenizer`` with
``use_reference=False``) is from the validated CPU reference
(``speech_tokenizer_decoder_forward``), which is what ``decode_icl_audio`` uses.

For each frame count we decode the SAME codes both ways and report SNR (dB) and
the per-decode wall time. This is the gate before wiring the device decoder onto
the serving path: it must match the reference (incl. >72 frames, where the
sliding-window mask matters) before it can replace the CPU decode.

Run:
    python models/demos/qwen3_tts/tests/test_ttnn_decoder_parity.py
"""

import time
from pathlib import Path

import numpy as np
import torch

import ttnn
from models.demos.qwen3_tts.reference.functional import SpeechTokenizerDecoderConfig, speech_tokenizer_decoder_forward
from models.demos.qwen3_tts.tt.speech_tokenizer import TtSpeechTokenizerDecoder

DEMO = Path(__file__).resolve().parents[1] / "demo"
REF_CACHE = DEMO / "jim_reference.refcache.pt"
SPF = 1920


def _load_decoder_weights():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    sd = load_file(hf_hub_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "speech_tokenizer/model.safetensors"))
    return {k[len("decoder.") :]: v.float() for k, v in sd.items() if k.startswith("decoder.")}


def _snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    n = min(len(ref), len(test))
    ref, test = ref[:n], test[:n]
    noise = ref - test
    p_sig = float(np.mean(ref**2))
    p_noise = float(np.mean(noise**2))
    if p_noise == 0:
        return float("inf")
    return 10.0 * np.log10(p_sig / p_noise)


def main():
    torch.set_num_threads(8)
    codes_all = torch.load(REF_CACHE, weights_only=True)["ref_codes"].long()  # [51, 16]
    decoder_weights = _load_decoder_weights()
    cfg = SpeechTokenizerDecoderConfig()

    device = ttnn.open_device(device_id=0, l1_small_size=131072, trace_region_size=0)
    try:
        st = TtSpeechTokenizerDecoder(device, decoder_weights, use_reference=False)
        for n in (25, 51, 76):
            if n <= codes_all.shape[0]:
                codes = codes_all[:n]
            else:
                reps = (n + codes_all.shape[0] - 1) // codes_all.shape[0]
                codes = codes_all.repeat(reps, 1)[:n]
            token_ids = codes.T.unsqueeze(0)  # [1, 16, n]

            t = time.perf_counter()
            with torch.no_grad():
                ref_audio = speech_tokenizer_decoder_forward(
                    codes.clone().clamp(max=2047).T.unsqueeze(0), decoder_weights, cfg
                )
            ref_s = time.perf_counter() - t
            ref_np = ref_audio.squeeze().detach().cpu().float().numpy()

            try:
                t = time.perf_counter()
                tt_audio = st.forward(token_ids.clone())
                tt_s = time.perf_counter() - t
                tt_np = np.asarray(tt_audio.squeeze().detach().cpu().float(), dtype=np.float32)
                snr = _snr_db(ref_np, tt_np)
                print(
                    f"RESULT frames={n} ref_samples={len(ref_np)} tt_samples={len(tt_np)} "
                    f"snr_db={snr:.2f} ref_s={ref_s:.3f} tt_s={tt_s:.3f} "
                    f"ref_first={ref_np[0]:.4f} tt_first={tt_np[0]:.4f}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"RESULT frames={n} TTNN_FORWARD_FAILED {type(e).__name__}: {e}", flush=True)
        print("DONE", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
