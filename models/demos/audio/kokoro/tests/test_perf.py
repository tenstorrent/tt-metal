# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill + traced warmed decode performance harness for the Kokoro-82M
functional decoder (plbert / ALBERT encoder).

Run under Tracy, one measured window per invocation, e.g.:

  python -m tracy -r -p -v -m pytest \
    models/demos/audio/kokoro/tests/test_perf.py -k prefill
  python -m tracy -r -p -v -m pytest \
    models/demos/audio/kokoro/tests/test_perf.py -k decode

Then feed the newest ops_perf_results CSV to tt-perf-report with the matching
signposts (PERF_PREFILL / PERF_DECODE). See doc/functional_decoder/README.md.
"""
import json
import os
import random

import pytest
import torch
import tracy

import ttnn
from models.demos.audio.kokoro.tt.functional_decoder import FunctionalDecoder

MODEL_ID = "hexgrad/Kokoro-82M"
PERF_SEQ_LEN = int(os.environ.get("KOKORO_PERF_SEQ_LEN", "512"))  # max advertised context
PERF_ITERS = int(os.environ.get("KOKORO_PERF_ITERS", "20"))
_REP_SYMBOLS = "ɐɚɛɜɪʊʌəɹɾɡ aioueɑɔbdfhjklmnprstvwzˈˌ "


def _build():
    from huggingface_hub import hf_hub_download
    from transformers import AlbertConfig, AlbertModel  # noqa: F401

    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    ac = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    pool = [cfg["vocab"][c] for c in _REP_SYMBOLS if c in cfg["vocab"]]
    rng = random.Random(0)
    body = [rng.choice(pool) for _ in range(PERF_SEQ_LEN - 2)]
    ids = torch.tensor([[0, *body, 0]], dtype=torch.long)
    return ac, sd, ids


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def test_perf_prefill(device):
    ac, sd, ids = _build()
    dec = FunctionalDecoder.from_state_dict(sd, hf_config=ac, mesh_device=device)
    prep = FunctionalDecoder.prepare_inputs(ids, device)

    def run():
        return dec.prefill_forward(
            prep["input_ids"],
            prep["position_ids"],
            prep["token_type_ids"],
            prep["attention_mask"],
            batch=prep["batch"],
            seq_len=prep["padded_seq_len"],
        )

    out = run()  # warm-up / compile
    ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    tracy.signpost("PERF_PREFILL")
    for _ in range(PERF_ITERS):
        out = run()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    tracy.signpost("PERF_PREFILL_END")


def test_perf_decode(device):
    ac, sd, ids = _build()
    dec = FunctionalDecoder.from_state_dict(sd, hf_config=ac, mesh_device=device)
    prep = FunctionalDecoder.prepare_inputs(ids, device)

    # capture trace + warm-up replay
    dec.decode_forward(
        prep["input_ids"],
        prep["position_ids"],
        prep["token_type_ids"],
        prep["attention_mask"],
        batch=prep["batch"],
        seq_len=prep["padded_seq_len"],
    )
    ttnn.synchronize_device(device)
    record = dec._traces[(prep["batch"], prep["padded_seq_len"])]

    tracy.signpost("PERF_DECODE")
    for _ in range(PERF_ITERS):
        ttnn.execute_trace(device, record["trace_id"], cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    tracy.signpost("PERF_DECODE_END")
    dec.release_traces()
