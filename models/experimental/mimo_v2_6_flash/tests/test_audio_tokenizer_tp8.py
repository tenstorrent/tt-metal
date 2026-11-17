# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The TTNN audio tokenizer encoder (tt/audio/tokenizer.py) on one 1x8 row vs the fp32 reference, real weights.

On speech clips of 5 s to 300 s (one to five 6000-frame segments): the 25 Hz hidden states' error, the code frames
that differ from the fp32 reference (with their RVQ decision margins, tests/perf/audio_codes_margin.py), and the LLM
embeddings of the device codes through the device patch encoder against the fp32 reference path (codes and patch
encoder in fp32) and the served host path. Checks: the outputs are the same bit for bit run to run; ``warm()``
compiles every program a call runs and a call leaves no DRAM allocated; embeddings PCC >= 0.998 and at most 8 % of the
code frames changed (sanity bounds). The relaxed gate for making the device encoder the default is logged, not
asserted, because it is not met: code agreement >= the fp64 encoder's (~99.99 %; the device changes ~4 % of the
frames: Blackhole matmuls keep 11 bits per 32-term tile dot product, see the module docstring) and embeddings PCC
>= 0.999 (0.9989-0.9991 measured). Logs the time per clip.

Run on one Galaxy row:
    TT_VISIBLE_DEVICES=<the row's 8 chips> pytest models/experimental/mimo_v2_6_flash/tests/test_audio_tokenizer_tp8.py
Needs MIMO_MODEL_PATH with ``audio_tokenizer/``, MIMO_AUDIO_CLIPS; converted weights go to MIMO_AUDIO_CACHE.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference import audio as ref
from models.experimental.mimo_v2_6_flash.tests.common import GALAXY_DEVICE_PARAMS, pcc
from models.experimental.mimo_v2_6_flash.tests.perf.audio_codes_margin import margins
from models.experimental.mimo_v2_6_flash.tests.test_audio_reference_host import (
    audio_model_path,
    load_wav_24k,
    require_clips,
    vllm_processor_or_none,
)
from models.experimental.mimo_v2_6_flash.tt.audio.patch import TtAudioPatchEncoder
from models.experimental.mimo_v2_6_flash.tt.audio.tokenizer import TtAudioTokenizerEncoder

CLIPS = ("speech_5s", "speech_30s", "speech_65s", "speech_120s", "speech_300s")
PCC = 0.998  # sanity bound; the gate's 0.999 is missed by ~1e-4 on some clips (logged)
GATE_PCC = 0.999
MAX_CHANGED = 0.08  # sanity bound on the changed code frames (measured 3.5-5 %)
# Edge lengths, cut from the clips' mels (name -> (clip, first mel frame, frames)): fewer mel frames than one LLM
# token's 16 (and than 10); one 25 Hz frame; and a last segment after a full 6000-frame one whose 50 Hz row count is
# odd, so its last pooled frame repeats a row. A few code frames say nothing as a fraction, so these are held to the
# lengths and the hidden states (HIDDEN_PCC), and their code / embedding differences are logged.
EDGE_CLIPS = {
    "edge_7_frames": ("speech_5s", 100, 7),
    "edge_3_frames": ("speech_5s", 200, 3),
    "edge_odd_last_segment": ("speech_65s", 0, 6000 + 37),
}
HIDDEN_PCC = 0.99


def edge_mels(mels):
    out = {}
    for name, (clip, start, frames) in EDGE_CLIPS.items():
        assert mels[clip].shape[0] >= start + frames, (name, mels[clip].shape)
        out[name] = mels[clip][start : start + frames].clone()
    return out


def cache_dir(path, what):
    root = os.environ.get("MIMO_AUDIO_CACHE", os.path.join(os.environ.get("MIMO_TT_CACHE", "/tmp"), "audio"))
    return os.path.join(root, os.path.basename(os.path.normpath(path)), what)


@pytest.fixture(scope="module")
def host():
    torch.set_num_threads(min(16, torch.get_num_threads()))
    path = audio_model_path()
    proc = vllm_processor_or_none()
    if proc is None:
        pytest.skip("vllm or torchaudio not importable (the clips are resampled with vLLM's resampler)")
    weights = ref.load_audio_weights(path)
    cfg = ref.MiMoAudioConfig.from_model_path(path)
    enc = ref.AudioTokenizerEncoder(weights, cfg)
    p32 = ref.AudioPatchEncoder(weights, cfg, torch.float32)
    p16 = ref.AudioPatchEncoder(weights, cfg, torch.bfloat16)
    clips = {}
    mels = {c: ref.log_mel_spectrogram(load_wav_24k(wav, proc[1]), cfg) for c, wav in require_clips(CLIPS).items()}
    mels.update(edge_mels(mels))
    for c, mel in mels.items():
        lens = torch.tensor(ref.segment_lengths(mel.shape[0], cfg.segment_size))
        with torch.inference_mode():
            h, _ = enc.features(mel, lens)
            codes = enc.quantize(h).t().contiguous()
            g = ref.pad_and_group_codes(codes, cfg)
            clips[c] = {
                "mel": mel,
                "hidden": h,
                "codes": codes,
                "margins": margins(enc, h),
                "emb32": p32(g),
                "emb16": p16(g),
            }
    return path, cfg, clips


def dram_allocated(mesh_device):
    mv = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return mv.total_bytes_allocated_per_bank * mv.num_banks


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_audio_tokenizer_tp8(mesh_device, host):
    path, cfg, clips = host
    dram0 = dram_allocated(mesh_device)
    t0 = time.time()
    tok = TtAudioTokenizerEncoder.from_checkpoint(mesh_device, path, cache_dir=cache_dir(path, "tokenizer"))
    dram1 = dram_allocated(mesh_device)
    patch = TtAudioPatchEncoder.from_checkpoint(mesh_device, path, cache_dir=cache_dir(path, "patch"))
    logger.info(f"tokenizer encoder loaded in {time.time() - t0:.1f} s: {(dram1 - dram0) / 2**20:.0f} MB per chip")
    t0 = time.time()
    times = tok.warm()
    patch.warm()
    dram2 = dram_allocated(mesh_device)
    logger.info(
        f"warmed in {time.time() - t0:.0f} s ({ {r: round(t, 1) for r, t in times.items()} }); after warm-up "
        f"{(dram2 - dram1) / 2**20:.0f} MB per chip more (RoPE tables, programs, the patch encoder)"
    )
    programs = mesh_device.num_program_cache_entries()
    rows = []
    for clip, c in clips.items():
        tok.hidden(c["mel"])
        lat = []
        for _ in range(3):
            t = time.perf_counter()
            h = tok.hidden(c["mel"])
            lat.append(time.perf_counter() - t)
        assert torch.equal(h, tok.hidden(c["mel"])), f"{clip}: run to run"
        assert h.shape == c["hidden"].shape, (clip, h.shape, c["hidden"].shape)
        codes = tok.quantizer.quantize(h).t().contiguous()
        assert codes.shape == c["codes"].shape == (ref.num_codes(c["mel"].shape[0], cfg), 20), (clip, codes.shape)
        t = time.perf_counter()
        emb = patch.embed_audio_codes(codes)
        t_patch = time.perf_counter() - t
        changed = (codes != c["codes"]).any(1)
        m = c["margins"][changed]
        assert emb.shape == c["emb32"].shape, (clip, emb.shape, c["emb32"].shape)
        p32, p16 = pcc(emb.float(), c["emb32"]), pcc(emb.float(), c["emb16"].float())
        rows.append((clip, changed.float().mean().item(), p32, pcc(h, c["hidden"])))
        if p32 < GATE_PCC:
            logger.warning(f"{clip}: embeddings PCC {p32:.5f} < the gate's {GATE_PCC}")
        logger.info(
            f"{clip}: {1e3 * min(lat):.0f} ms tokenizer encoder + {1e3 * t_patch:.0f} ms patch encoder; hidden rel err "
            f"{((h - c['hidden']).norm() / c['hidden'].norm()).item():.1e}; {int(changed.sum())}/{changed.numel()} "
            f"code frames changed ({100 * changed.float().mean():.2f} %; margins of the changed frames median "
            f"{m.median().item() if m.numel() else 0:.1e}, max {m.max().item() if m.numel() else 0:.1e}; all frames "
            f"median {c['margins'].median().item():.1e}); embeddings PCC vs fp32 {p32:.5f}, vs the host bf16 path {p16:.5f}"
        )
    assert mesh_device.num_program_cache_entries() == programs, "a call compiled programs warm() did not"
    assert dram_allocated(mesh_device) == dram2, "a call left device tensors allocated"
    assert {clip for clip, *_ in rows} == set(CLIPS) | set(EDGE_CLIPS)  # none dropped out
    for clip, frac, p, p_hidden in rows:
        assert p_hidden >= HIDDEN_PCC, (clip, p_hidden)
        if clip in EDGE_CLIPS:
            continue
        assert p >= PCC, (clip, p)
        assert frac <= MAX_CHANGED, (clip, frac)
