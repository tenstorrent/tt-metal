#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare HF vs TT speech-path metadata: units, t_audio proxy, waveform samples."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import ttnn
from transformers import AutoProcessor, AutoTokenizer

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.experimental.seamless_m4t_v2_large.demo.demo import (  # noqa: E402
    _isolated_task_session,
    _process_jit_preflight,
    torch_feats_to_ttnn,
    torch_ids_to_ttnn,
)
from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (  # noqa: E402
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import (  # noqa: E402
    ensure_seamless_m4t_v2_large_weights,
)
from models.experimental.seamless_m4t_v2_large.tt.common import (  # noqa: E402
    hf_aligned_generation_kwargs,
    to_torch_replicated_first_shard,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_code_hifigan import (  # noqa: E402
    _host_hifigan_output_length,
)

_PROMPT = (
    "going along slushy country roads and speaking to damp audiences in draughty schoolrooms "
    "day after day for a fortnight he'll have to put in an appearance at some place of worship "
    "on sunday morning and he can come to us immediately afterwards"
)


def _hf_waveform(out) -> tuple:
    if hasattr(out, "waveform"):
        wav_t = out.waveform
        lens_t = getattr(out, "waveform_lengths", None)
    else:
        wav_t, lens_t = out[0], out[1]
    wav = wav_t.detach().float().squeeze().cpu().numpy()
    n = int(wav.size)
    if lens_t is not None:
        n = int(lens_t.reshape(-1)[0].item())
    return wav[:n], n


def _hf_samples(out) -> int:
    return _hf_waveform(out)[1]


def _tt_meta(out) -> dict:
    samples = int(to_torch_replicated_first_shard(out.waveform_lengths).long().reshape(-1)[0].item())
    units = to_torch_replicated_first_shard(out.unit_sequences).long().reshape(-1).tolist()
    pad = 0
    valid_units = sum(1 for u in units if int(u) != pad)
    t_audio = getattr(out, "_t_audio", None)
    return {"samples": samples, "units_len": len(units), "valid_units": valid_units, "t_audio": t_audio}


def main() -> None:
    weights_dir = ensure_seamless_m4t_v2_large_weights()
    path = str(weights_dir)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    torch.manual_seed(0)
    hf_model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = hf_model.t2u_model.config
    sr = int(getattr(cfg, "sampling_rate", 16000))

    text_enc = tokenizer([_PROMPT], return_tensors="pt", padding=True)
    gen_common = hf_aligned_generation_kwargs(
        hf_model.generation_config,
        use_kv_cache=True,
        use_decode_trace=True,
        use_2cq=True,
    )

    try:
        original_default = ttnn.GetDefaultDevice()
    except Exception:
        original_default = None

    session_kw = dict(
        hf_model=hf_model,
        cfg=cfg,
        t2u_cfg=t2u_cfg,
        gen_common=gen_common,
        original_default=original_default,
    )
    _process_jit_preflight(
        session_kw=session_kw,
        input_ids=text_enc["input_ids"],
        input_text_attn=text_enc["attention_mask"],
        gen_common=gen_common,
    )

    print("=== T2ST (eng text -> hin speech) ===")
    with torch.no_grad():
        hf_out = hf_model.generate(
            input_ids=text_enc["input_ids"],
            attention_mask=text_enc["attention_mask"],
            generate_speech=True,
            tgt_lang="hin",
            speaker_id=0,
            **{k: v for k, v in gen_common.items() if k not in ("use_kv_cache", "use_decode_trace", "use_2cq")},
        )
        hf_wav, hf_n = _hf_waveform(hf_out)
        print(f"  HF samples={hf_n}")

    with _isolated_task_session(**session_kw) as (device, tt_model):
        ids_tt = torch_ids_to_ttnn(device, text_enc["input_ids"])
        attn_tt = torch_ids_to_ttnn(device, text_enc["attention_mask"])
        tt_out = tt_model.generate(
            input_ids=ids_tt,
            attention_mask=attn_tt,
            generate_speech=True,
            return_intermediate_token_ids=True,
            tgt_lang="hin",
            speaker_id=0,
            **gen_common,
        )
        ttnn.synchronize_device(device)
        tt_n = int(to_torch_replicated_first_shard(tt_out.waveform_lengths).long().reshape(-1)[0].item())
        t_audio = getattr(tt_model.vocoder, "_last_t_audio", None)
        unit_seq = getattr(tt_model.vocoder, "_last_unit_seq", None)
        expected_from_t = _host_hifigan_output_length(tt_model.vocoder.cfg, int(t_audio)) if t_audio else None
        print(f"  TT samples={tt_n}  vocoder _last_t_audio={t_audio}  _last_unit_seq={unit_seq}")
        print(f"  TT expected samples from t_audio formula={expected_from_t}")
        print(f"  delta HF-TT samples={hf_n - tt_n}")

        # S2ST on HF T2ST audio
        audio_inputs = processor(audios=hf_wav, sampling_rate=sr, return_tensors="pt")
        print("\n=== S2ST (HF hin speech -> spa speech) ===")
        with torch.no_grad():
            hf_s2st = hf_model.generate(
                input_features=audio_inputs["input_features"].float(),
                attention_mask=audio_inputs["attention_mask"],
                generate_speech=True,
                tgt_lang="spa",
                speaker_id=0,
                **{k: v for k, v in gen_common.items() if k not in ("use_kv_cache", "use_decode_trace", "use_2cq")},
            )
        hf_s2_wav, hf_s2_n = _hf_waveform(hf_s2st)
        print(f"  HF samples={hf_s2_n}")

        feats_tt = torch_feats_to_ttnn(device, audio_inputs["input_features"])
        attn_sp = torch_ids_to_ttnn(device, audio_inputs["attention_mask"])
        tt_s2st = tt_model.generate(
            input_features=feats_tt,
            attention_mask=attn_sp,
            generate_speech=True,
            return_intermediate_token_ids=True,
            tgt_lang="spa",
            speaker_id=0,
            **gen_common,
        )
        ttnn.synchronize_device(device)
        tt_s2_n = int(to_torch_replicated_first_shard(tt_s2st.waveform_lengths).long().reshape(-1)[0].item())
        t_audio2 = getattr(tt_model.vocoder, "_last_t_audio", None)
        unit_seq2 = getattr(tt_model.vocoder, "_last_unit_seq", None)
        print(f"  TT samples={tt_s2_n}  vocoder _last_t_audio={t_audio2}  _last_unit_seq={unit_seq2}")
        print(f"  delta HF-TT samples={hf_s2_n - tt_s2_n}")

        ttnn.deallocate(tt_out.waveform)
        ttnn.deallocate(tt_out.waveform_lengths)
        ttnn.deallocate(tt_out.sequences)
        ttnn.deallocate(tt_out.unit_sequences)
        ttnn.deallocate(tt_s2st.waveform)
        ttnn.deallocate(tt_s2st.waveform_lengths)
        ttnn.deallocate(tt_s2st.sequences)
        ttnn.deallocate(tt_s2st.unit_sequences)


if __name__ == "__main__":
    main()
