# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end performance for the single-chip Kokoro-82M plbert encoder (P150).

Standard models-perf harness: one compile+run, then N measured eager prefills,
reported via ``prep_perf_report``.

    pytest -m models_performance_bare_metal \
        models/demos/audio/kokoro/tests/test_perf_optimized.py
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.audio.kokoro.tt.optimized_decoder import OptimizedDecoder
from models.perf.perf_utils import prep_perf_report

MODEL_ID = "hexgrad/Kokoro-82M"
PERF_ITERS = int(os.environ.get("KOKORO_PERF_ITERS", "10"))
_REP_SYMBOLS = "ɐɚɛɜɪʊʌəɹɾɡ aioueɑɔbdfhjklmnprstvwzˈˌ "
# Measured on P150 (single Blackhole chip): ~2.5 ms @T128, ~3.0 ms @T512 eager prefill
# (launch/dispatch-bound). Small margin over measured.
_EXPECTED_INFERENCE_S = {128: 0.0030, 512: 0.0035}


def _build(seq_len):
    from huggingface_hub import hf_hub_download
    from transformers import AlbertConfig

    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    ac = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    pool = [cfg["vocab"][c] for c in _REP_SYMBOLS if c in cfg["vocab"]]
    g = torch.Generator().manual_seed(0)
    body = [pool[i] for i in torch.randint(len(pool), (seq_len - 2,), generator=g).tolist()]
    ids = torch.tensor([[0, *body, 0]], dtype=torch.long)
    return ac, sd, ids


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("seq_len", [128, 512])
def test_perf_encoder(device, seq_len):
    ac, sd, ids = _build(seq_len)
    dec = OptimizedDecoder.from_state_dict(sd, hf_config=ac, mesh_device=device)
    prep = OptimizedDecoder.prepare_inputs(ids, device)

    def run():
        return dec.prefill_forward(
            prep["input_ids"],
            prep["position_ids"],
            prep["token_type_ids"],
            prep["attention_mask"],
            batch=prep["batch"],
            seq_len=prep["padded_seq_len"],
        )

    t0 = time.time()
    out = run()
    ttnn.synchronize_device(device)
    inference_and_compile_time = time.time() - t0
    ttnn.deallocate(out)

    t0 = time.time()
    for _ in range(PERF_ITERS):
        out = run()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    inference_time = (time.time() - t0) / PERF_ITERS

    expected = _EXPECTED_INFERENCE_S[seq_len]
    logger.info(
        f"plbert encoder T={seq_len}: compile+run {inference_and_compile_time:.2f}s, inference {inference_time*1000:.2f}ms"
    )
    prep_perf_report(
        model_name=f"kokoro82m_plbert_encoder_T{seq_len}",
        batch_size=1,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=60.0,
        expected_inference_time=expected,
        comments=f"seq{seq_len}",
    )
