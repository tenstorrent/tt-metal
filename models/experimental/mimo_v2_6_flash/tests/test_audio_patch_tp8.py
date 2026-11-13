# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The TTNN audio patch encoder (tt/audio/patch.py) on one 1x8 row vs the host paths, real weights and codes.

The codes are the fp32 tokenizer encoder's for speech clips (reference/make_audio_clips.py: 5 s to 300 s, one pass of
32 to 1024 groups, and two passes). Checks against the host patch encoder the engine runs today (bf16) and the fp32
reference: PCC >= 0.999 for each; the embeddings' shape and dtype; that padding a pass to its bucket leaves the
real rows alone (garbage instead of zeros in the padding rows, bit for bit); that ``warm()`` compiles every
program a call runs and a call leaves no device tensor behind; and the engine's path (generator_vllm's
``audio_device_patch`` builds it, ``AudioCodesBackend`` routes an item's codes to it through tt/multimodal.encode_item).
Logs the time per clip after warm-up and the encoder's DRAM (weights, warmed programs).

Run on one Galaxy row, as a row engine does:
    TT_VISIBLE_DEVICES=<the row's 8 chips> pytest models/experimental/mimo_v2_6_flash/tests/test_audio_patch_tp8.py
Needs MIMO_MODEL_PATH with ``audio_tokenizer/``, MIMO_AUDIO_CLIPS, vllm and the TT plugin. The converted weights go
to MIMO_AUDIO_CACHE (default: an ``audio`` cache next to the LLM's MIMO_TT_CACHE), per checkpoint.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference import audio as ref
from models.experimental.mimo_v2_6_flash.tests.common import GALAXY_DEVICE_PARAMS, pcc
from models.experimental.mimo_v2_6_flash.tests.test_audio_reference_host import (
    audio_model_path,
    load_wav_24k,
    require_clips,
    vllm_processor_or_none,
)
from models.experimental.mimo_v2_6_flash.tt.audio.host_backend import HostAudioBackend
from models.experimental.mimo_v2_6_flash.tt.audio.patch import GROUP_BUCKETS, TtAudioPatchEncoder, padded_groups

CLIPS = ("speech_5s", "speech_30s", "speech_120s", "speech_300s")
# Edge lengths, cut from speech_5s's codes (name -> code frames): one code frame (a group that is three quarters
# padding), and counts that are not a multiple of the group size (4), one of them below one group.
EDGE_CODES = {"edge_1_code": 1, "edge_3_codes": 3, "edge_5_codes": 5}
PCC = 0.999


def audio_cache_root():
    return os.environ.get("MIMO_AUDIO_CACHE", os.path.join(os.environ.get("MIMO_TT_CACHE", "/tmp"), "audio"))


@pytest.fixture(scope="module")
def host():
    """Codes per clip (fp32 tokenizer encoder) and the host patch encoders' embeddings (bf16 as served, fp32)."""
    torch.set_num_threads(min(16, torch.get_num_threads()))
    path = audio_model_path()
    proc = vllm_processor_or_none()
    if proc is None:
        pytest.skip("vllm or torchaudio not importable (the clips are resampled with vLLM's resampler)")
    weights = ref.load_audio_weights(path)
    tok = HostAudioBackend(path, num_threads=16, weights=weights)
    patch32 = ref.AudioPatchEncoder(weights, tok.cfg, torch.float32)
    clips = {}
    all_codes = {}
    for c, wav in require_clips(CLIPS).items():  # skips when one is missing: no case drops out quietly
        all_codes[c] = tok.codes(ref.log_mel_spectrogram(load_wav_24k(wav, proc[1]), tok.cfg))
    for name, n in EDGE_CODES.items():
        all_codes[name] = all_codes["speech_5s"][20 : 20 + n].clone()
    for c, codes in all_codes.items():
        clips[c] = {
            "codes": codes,
            "bf16": tok.embed_audio_codes(codes),
            "fp32": patch32(ref.pad_and_group_codes(codes, tok.cfg)),
        }
    return path, tok.cfg, clips


def dram_allocated(mesh_device):
    mv = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return mv.total_bytes_allocated_per_bank * mv.num_banks


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_audio_patch_tp8(mesh_device, host, monkeypatch):
    from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gen
    from models.experimental.mimo_v2_6_flash.tt import multimodal as mm
    from models.experimental.mimo_v2_6_flash.tt.audio.host_backend import AudioCodesBackend

    path, cfg, clips = host
    monkeypatch.setenv("MIMO_AUDIO_CACHE", audio_cache_root())
    dram0 = dram_allocated(mesh_device)
    t0 = time.time()
    enc = gen.audio_device_patch(path, mesh_device)  # what the engine builds for a replica (MIMO_AUDIO_PATCH=device)
    assert isinstance(enc, TtAudioPatchEncoder), type(enc)
    dram1 = dram_allocated(mesh_device)
    logger.info(f"device patch encoder loaded in {time.time() - t0:.1f} s, {(dram1 - dram0) / 2**20:.1f} MB per chip")
    t0 = time.time()
    times = enc.warm()
    logger.info(
        f"warmed {len(times)} buckets in {time.time() - t0:.1f} s ({(dram_allocated(mesh_device) - dram1) / 2**20:.1f}"
        f" MB per chip more): { {g: round(t, 2) for g, t in times.items()} }"
    )
    programs = mesh_device.num_program_cache_entries()
    before = dram_allocated(mesh_device)
    scores = {}
    for clip, c in clips.items():
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            out = enc.embed_audio_codes(c["codes"].to(torch.int32))
            times.append(time.perf_counter() - t0)
        g = c["bf16"].shape[0]
        assert out.shape == (g, cfg.out_hidden_size) and out.dtype == torch.bfloat16, (out.shape, out.dtype)
        p16, p32 = pcc(out.float(), c["bf16"].float()), pcc(out.float(), c["fp32"].float())
        scores[clip] = (p16, p32)
        logger.info(
            f"{clip}: {c['codes'].shape[0]} code frames, {g} groups in {-(-g // enc.max_groups)} pass(es) (bucket "
            f"{padded_groups(min(g, enc.max_groups))}): PCC vs host bf16 {p16:.6f}, vs fp32 {p32:.6f}; "
            f"{1e3 * min(times):.1f} ms (min of 3; host embedding sum and transfers included)"
        )
    # The engine's path: an item's codes through encode_item and AudioCodesBackend reach the device encoder.
    c = clips["speech_30s"]
    n = c["bf16"].shape[0]
    item = mm.MMItem("audio", "a", 10, n, None, {"audio_codes": c["codes"].to(torch.int32), "audio_token_lens": n})
    backend = AudioCodesBackend(object(), path, device_patch=enc)
    assert torch.equal(mm.encode_item(backend, mm.parse_item(item)), enc.embed_audio_codes(c["codes"]))
    assert mesh_device.num_program_cache_entries() == programs, "a call compiled programs warm() did not"
    assert dram_allocated(mesh_device) == before, "a call left device tensors allocated"

    # Padding isolation: garbage instead of zeros past the item's groups gives the same embeddings bit for bit.
    grouped = ref.pad_and_group_codes(clips["speech_30s"]["codes"], cfg)
    g = grouped.shape[0]
    rows = padded_groups(g) * cfg.group_size
    x = enc.embed_codes(grouped)
    junk = torch.randn(rows - x.shape[0], x.shape[1], generator=torch.Generator().manual_seed(1)).to(x.dtype)
    a = enc.forward_pass(torch.cat([x, torch.zeros_like(junk)]), g)
    b = enc.forward_pass(torch.cat([x, junk]), g)
    assert g < rows // cfg.group_size and padded_groups(g) in GROUP_BUCKETS
    assert torch.equal(a, b), "padding rows leak into the real rows"
    logger.info(f"speech_30s: {g} groups padded to {rows // cfg.group_size}: garbage padding changes nothing")
    assert set(scores) == set(CLIPS) | set(EDGE_CODES)  # none dropped out
    for clip, (p16, p32) in scores.items():
        assert p16 >= PCC and p32 >= PCC, (clip, p16, p32)
