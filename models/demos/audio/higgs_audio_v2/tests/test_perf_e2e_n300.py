# SPDX-FileCopyrightText: (c) 2026 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
"""Directly-measured END-TO-END RTF on N300: traced decode + codec, both chips.

Reuses the data-parallel traced decode (one stream/chip, _build_stream from the
perf test) AND runs the TTNN codec on each chip, all in one process, and times
the real combined wall-clock. Token values don't affect timing, so the codec is
fed placeholder frames (same as the perf-path decode uses placeholder tokens).

Reports aggregate end-to-end RTF = (2 streams * N/25 s of audio) / wall.
"""
import json
import os
import pathlib
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.audio.higgs_audio_v2.tests.test_perf_dp_n300 import _build_stream
from models.demos.audio.higgs_audio_v2.tt.codec import TtDacDecoder
from models.demos.audio.higgs_audio_v2.tt.reference import HiggsAudioV2Config
from models.tt_transformers.tt.generator import create_submeshes

HIGGS_MODEL_DIR = "/data/hf_cache/higgs"
FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "baseline_tts_short.json"
CODEC_FRAME_RATE_HZ = 25.0
WARMUP = 4
STEPS = int(os.environ.get("E2E_STEPS", "200"))  # frames per stream (~8s audio)


@pytest.fixture(scope="module")
def mesh_device():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=200000000, l1_small_size=98304)
    yield dev
    ttnn.close_mesh_device(dev)


def test_e2e_rtf_n300(mesh_device):
    assert mesh_device.get_num_devices() == 2, "needs N300 (2 chips)"
    from models.demos.audio.higgs_audio_v2.tt.reference import load_higgs_v2_state_dict

    with open(FIXTURE) as _fh:
        fixture = json.load(_fh)
    prompt_ids = torch.tensor(fixture["prompt_text_tokens"], dtype=torch.int64)
    higgs_cfg = HiggsAudioV2Config.from_json(pathlib.Path(HIGGS_MODEL_DIR) / "config.json")
    _, state_dict = load_higgs_v2_state_dict(HIGGS_MODEL_DIR)

    submeshes = create_submeshes(mesh_device, 2)
    streams = [_build_stream(sm, higgs_cfg, state_dict, prompt_ids, "performance") for sm in submeshes]
    logger.info("both decode streams traced")

    # Codec per chip (host-loaded HF acoustic_decoder weights, TTNN execution).
    from transformers import AutoModel

    codec_hf = AutoModel.from_pretrained(HIGGS_MODEL_DIR + "/tokenizer").eval()
    C0 = codec_hf.acoustic_decoder.conv1.in_channels
    codecs = [TtDacDecoder(sm, codec_hf.acoustic_decoder) for sm in submeshes]
    qa = torch.randn(1, C0, STEPS) * 0.5  # placeholder frames (timing is token-independent)

    def run_decode(st, pos):
        st["advance_pos"](pos)
        ttnn.execute_trace(st["submesh"], st["trace_id"], cq_id=0, blocking=False)

    # warmup decode + codec compile
    for i in range(WARMUP):
        for st in streams:
            run_decode(st, st["S"] + i)
    for st in streams:
        ttnn.synchronize_device(st["submesh"])
    for c in codecs:
        _ = c.forward(qa)
    for sm in submeshes:
        ttnn.synchronize_device(sm)

    # ---- timed end-to-end: decode STEPS frames (both chips) then codec (both chips) ----
    t0 = time.perf_counter()
    for i in range(STEPS):
        for st in streams:
            run_decode(st, st["S"] + WARMUP + i)
    for st in streams:
        ttnn.synchronize_device(st["submesh"])
    t_decode = time.perf_counter()
    for c in codecs:
        _ = c.forward(qa)
    for sm in submeshes:
        ttnn.synchronize_device(sm)
    wall = time.perf_counter() - t0
    decode_wall = t_decode - t0
    codec_wall = wall - decode_wall

    audio_s = 2 * STEPS / CODEC_FRAME_RATE_HZ  # two streams
    rtf = wall / audio_s
    logger.info(f"==== N300 END-TO-END (traced decode + codec, 2 streams) ====")
    logger.info(f"  decode {decode_wall*1e3:.0f}ms  codec {codec_wall*1e3:.0f}ms  total {wall*1e3:.0f}ms")
    logger.info(f"  audio={audio_s:.2f}s  END-TO-END aggregate RTF = {rtf:.4f}")
    print(
        f"PERF_E2E_N300 rtf={rtf:.4f} decode_ms={decode_wall*1e3:.0f} codec_ms={codec_wall*1e3:.0f} "
        f"per_step_ms={1e3*decode_wall/STEPS:.2f}"
    )

    for st in streams:
        ttnn.release_trace(st["submesh"], st["trace_id"])
