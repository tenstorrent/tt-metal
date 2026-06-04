# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare HF vs TT speech→text ``generate()`` on a fixed WAV (demo ASR parity check).

Runs ASR (and optionally S2TT) on the same audio file with ``hf_aligned_generation_kwargs``
and the demo's TT flags (trace + 2CQ). Useful to see why demo ASR token counts differ from
the PCC test (PCC chains HF T2ST audio; demo chains TT T2ST audio).

Example (from repo root):

  python models/experimental/seamless_m4t_v2_large/scripts/compare_speech_asr_hf_tt.py

  python models/experimental/seamless_m4t_v2_large/scripts/compare_speech_asr_hf_tt.py \\
      --wav models/experimental/seamless_m4t_v2_large/demo/outputs/t2st_hindi_speech.wav \\
      --task asr s2tt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np
import torch
import ttnn
from transformers import AutoProcessor, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.common import (
    hf_aligned_generation_kwargs,
    to_torch_replicated_first_shard,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import open_seamless_mesh_device
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_seamless_m4t_v2_model_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    TTSeamlessM4Tv2GreedySearchOutput,
    TTSeamlessM4Tv2Model,
)

_DEFAULT_WAV = _REPO_ROOT / "models/experimental/seamless_m4t_v2_large/demo/outputs/t2st_hindi_speech.wav"


def _load_mono_wav(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sw == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sw} in {path}")
    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)
    return pcm.astype(np.float32), int(sr)


def _hf_text_ids(out: Any) -> List[int]:
    if hasattr(out, "sequences"):
        return out.sequences[0].cpu().tolist()
    return out[0].cpu().tolist()


def _tt_text_ids(out: TTSeamlessM4Tv2GreedySearchOutput) -> List[int]:
    ids = to_torch_replicated_first_shard(out.sequences).long().cpu().reshape(-1).tolist()
    ttnn.deallocate(out.sequences)
    return ids


def _decode(tokenizer: Any, ids: Iterable[int]) -> str:
    return tokenizer.batch_decode([list(ids)], skip_special_tokens=True)[0]


def _lcp(a: List[int], b: List[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _make_tt_model(device: ttnn.Device, model: torch.nn.Module, cfg: Any, t2u_cfg: Any) -> TTSeamlessM4Tv2Model:
    params = create_seamless_m4t_v2_model_parameters(model, device=device)
    return TTSeamlessM4Tv2Model(
        device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        encoder_layers=cfg.encoder_layers,
        encoder_attention_heads=cfg.encoder_attention_heads,
        decoder_layers=cfg.decoder_layers,
        decoder_attention_heads=cfg.decoder_attention_heads,
        hidden_size=cfg.hidden_size,
        feature_projection_input_dim=cfg.feature_projection_input_dim,
        speech_encoder_attention_heads=cfg.speech_encoder_attention_heads,
        speech_encoder_intermediate_size=cfg.speech_encoder_intermediate_size,
        speech_encoder_layers=cfg.speech_encoder_layers,
        speech_encoder_chunk_size=cfg.speech_encoder_chunk_size,
        speech_encoder_left_chunk_num=cfg.speech_encoder_left_chunk_num,
        pad_token_id=cfg.pad_token_id,
        decoder_start_token_id=cfg.decoder_start_token_id,
        vocab_size=cfg.vocab_size,
        adaptor_kernel_size=cfg.adaptor_kernel_size,
        adaptor_stride=cfg.adaptor_stride,
        t2u_eos_token_id=cfg.t2u_eos_token_id,
        t2u_pad_token_id=t2u_cfg.pad_token_id,
        vocoder_offset=cfg.vocoder_offset,
        t2u_layer_norm_eps=t2u_cfg.layer_norm_eps,
        t2u_encoder_layers=t2u_cfg.encoder_layers,
        t2u_encoder_attention_heads=t2u_cfg.encoder_attention_heads,
        t2u_decoder_layers=t2u_cfg.decoder_layers,
        t2u_decoder_attention_heads=t2u_cfg.decoder_attention_heads,
        variance_predictor_embed_dim=t2u_cfg.variance_predictor_embed_dim,
        variance_predictor_hidden_dim=t2u_cfg.variance_predictor_hidden_dim,
        variance_predictor_kernel_size=t2u_cfg.variance_predictor_kernel_size,
        vocoder_config=cfg,
        generation_config=model.generation_config,
        hf_config=cfg,
    )


def _torch_feats_to_ttnn(device: ttnn.Device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.bfloat16).cpu().contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _torch_ids_to_ttnn(device: ttnn.Device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.int32).cpu(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run_task(
    *,
    task: str,
    tgt_lang: str,
    hf_model: torch.nn.Module,
    tt_model: TTSeamlessM4Tv2Model,
    device: ttnn.Device,
    sp_features: torch.Tensor,
    sp_attn: torch.Tensor,
    gen_common: dict,
    tt_extra: dict,
    tokenizer: Any,
) -> None:
    label = task.upper()
    print(f"\n{'=' * 72}")
    print(f"  {label}  (speech → text, tgt_lang={tgt_lang!r})")
    print(f"{'=' * 72}")

    t0 = time.perf_counter()
    with torch.no_grad():
        hf_out = hf_model.generate(
            input_features=sp_features.float(),
            attention_mask=sp_attn,
            generate_speech=False,
            tgt_lang=tgt_lang,
            **{k: v for k, v in gen_common.items() if k not in tt_extra},
        )
    hf_ms = (time.perf_counter() - t0) * 1000.0
    hf_ids = _hf_text_ids(hf_out)

    feats_tt = _torch_feats_to_ttnn(device, sp_features)
    attn_tt = _torch_ids_to_ttnn(device, sp_attn)
    t0 = time.perf_counter()
    tt_out = tt_model.generate(
        input_features=feats_tt,
        attention_mask=attn_tt,
        generate_speech=False,
        tgt_lang=tgt_lang,
        **gen_common,
    )
    tt_ms = (time.perf_counter() - t0) * 1000.0
    if not isinstance(tt_out, TTSeamlessM4Tv2GreedySearchOutput):
        raise TypeError(f"TT {label} expected GreedySearchOutput, got {type(tt_out)}")
    tt_ids = _tt_text_ids(tt_out)

    lcp = _lcp(hf_ids, tt_ids)
    seed = 2
    seed_ok = hf_ids[:seed] == tt_ids[:seed]

    print(f"  HF: {len(hf_ids)} tokens in {hf_ms:.1f} ms")
    print(f"  TT: {len(tt_ids)} tokens in {tt_ms:.1f} ms")
    print(f"  longest common prefix: {lcp}  |  seed[{seed}] match: {seed_ok}")
    print(f"\n  HF text:\n    {_decode(tokenizer, hf_ids)}\n")
    print(f"  TT text:\n    {_decode(tokenizer, tt_ids)}\n")
    if hf_ids != tt_ids:
        i = lcp
        print(
            f"  first mismatch at index {i}: HF={hf_ids[i] if i < len(hf_ids) else '—'} TT={tt_ids[i] if i < len(tt_ids) else '—'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HF vs TT speech→text on a fixed WAV.")
    parser.add_argument(
        "--wav",
        type=Path,
        default=_DEFAULT_WAV,
        help=f"Input mono speech WAV (default: {_DEFAULT_WAV})",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        choices=("asr", "s2tt"),
        default=("asr",),
        help="Tasks to run: asr (hin→hin) and/or s2tt (hin→eng). Default: asr",
    )
    parser.add_argument("--tgt-asr", default="hin", help="ASR target language code (default: hin)")
    parser.add_argument("--tgt-s2tt", default="eng", help="S2TT target language code (default: eng)")
    parser.add_argument("--no-trace", action="store_true", help="Disable TT decode trace + 2CQ")
    args = parser.parse_args()

    wav_path = args.wav.expanduser().resolve()
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    weights_dir = (
        Path(os.environ.get("SEAMLESS_M4T_V2_WEIGHTS", "")).expanduser()
        if os.environ.get("SEAMLESS_M4T_V2_WEIGHTS")
        else ensure_seamless_m4t_v2_large_weights()
    )
    path = os.fspath(weights_dir)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    pcm, file_sr = _load_mono_wav(wav_path)
    torch.manual_seed(0)
    hf_model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = hf_model.t2u_model.config
    model_sr = int(getattr(cfg, "sampling_rate", 16000))
    if file_sr != model_sr:
        print(f"  warning: WAV sample rate {file_sr} != model {model_sr}; processor resamples")

    audio_inputs = processor(audios=pcm, sampling_rate=model_sr, return_tensors="pt")
    sp_features = audio_inputs["input_features"]
    sp_attn = audio_inputs["attention_mask"]
    mel_frames = int(sp_attn.sum().item())

    use_trace = not args.no_trace
    gen_common = hf_aligned_generation_kwargs(
        hf_model.generation_config,
        use_kv_cache=True,
        use_decode_trace=use_trace,
        use_2cq=use_trace,
    )
    tt_extra = {k: gen_common[k] for k in ("use_kv_cache", "use_decode_trace", "use_2cq") if k in gen_common}

    print("=" * 72)
    print("  HF vs TT speech→text comparison")
    print("=" * 72)
    print(f"  WAV: {wav_path}")
    print(f"  samples: {pcm.size} ({pcm.size / model_sr:.2f} s @ {model_sr} Hz)")
    print(f"  mel_frames (attn sum): {mel_frames}")
    print(f"  generation kwargs: { {k: gen_common[k] for k in gen_common if k not in tt_extra} }")
    print(f"  TT-only flags: {tt_extra}")

    try:
        original_default = ttnn.GetDefaultDevice()
    except Exception:
        original_default = None

    device, mesh_shape = open_seamless_mesh_device(
        enable_decode_trace=bool(gen_common.get("use_decode_trace")),
        enable_2cq=bool(gen_common.get("use_2cq")),
    )
    print(f"  device: MeshShape{mesh_shape}")
    ttnn.SetDefaultDevice(device)

    tt_model = None
    try:
        tt_model = _make_tt_model(device, hf_model, cfg, t2u_cfg)
        tt_model.prewarm_speech_encoder([int(sp_features.shape[-1])])
        ttnn.synchronize_device(device)

        if "asr" in args.task:
            _run_task(
                task="asr",
                tgt_lang=args.tgt_asr.replace("__", ""),
                hf_model=hf_model,
                tt_model=tt_model,
                device=device,
                sp_features=sp_features,
                sp_attn=sp_attn,
                gen_common=gen_common,
                tt_extra=tt_extra,
                tokenizer=tokenizer,
            )
        if "s2tt" in args.task:
            tt_model.clear_runtime_program_cache()
            ttnn.synchronize_device(device)
            tt_model.prewarm_speech_encoder([int(sp_features.shape[-1])])
            _run_task(
                task="s2tt",
                tgt_lang=args.tgt_s2tt.replace("__", ""),
                hf_model=hf_model,
                tt_model=tt_model,
                device=device,
                sp_features=sp_features,
                sp_attn=sp_attn,
                gen_common=gen_common,
                tt_extra=tt_extra,
                tokenizer=tokenizer,
            )
    finally:
        if tt_model is not None:
            tt_model.release_generation_runtime()
        ttnn.close_mesh_device(device)
        ttnn.SetDefaultDevice(original_default)


if __name__ == "__main__":
    main()
