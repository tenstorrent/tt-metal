# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the EmbeddingGemma 2 tests.

The device is opened once per test module (`eg2_device`) so one loaded model serves every test in
that module. The HF fp32 reference and the checkpoint tensors are session scoped.
"""

import os

import pytest

import ttnn
from models.demos.blackhole.embedding_gemma2.tt.config import TextModelConfig, default_weight_cache_dir
from models.demos.blackhole.embedding_gemma2.tt.weights import (
    load_text_config_dict,
    load_text_state_dict,
    resolve_snapshot,
)

MODEL_ID = os.environ.get("HF_MODEL", "google/embeddinggemma-2")
# RoPE tables cover the largest bucket any test may use.
TEST_MAX_SEQ_LEN = int(os.environ.get("EMBEDDING_GEMMA2_MAX_SEQ_LEN", "32768"))
DEVICE_PARAMS = {"l1_small_size": 0, "trace_region_size": 128 * 1024 * 1024, "num_command_queues": 1}


@pytest.fixture(scope="session")
def eg2_snapshot():
    return resolve_snapshot(MODEL_ID)


@pytest.fixture(scope="session")
def eg2_state_dict(eg2_snapshot):
    return load_text_state_dict(eg2_snapshot)


@pytest.fixture(scope="session")
def eg2_config(eg2_snapshot):
    return TextModelConfig.from_hf(load_text_config_dict(eg2_snapshot), max_seq_len=TEST_MAX_SEQ_LEN)


@pytest.fixture(scope="session")
def eg2_cache_dir(eg2_snapshot):
    d = default_weight_cache_dir() / eg2_snapshot.name
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def hf_text_model(eg2_snapshot, eg2_state_dict):
    from models.demos.blackhole.embedding_gemma2.reference.hf_reference import load_reference_text_model

    model, _ = load_reference_text_model(eg2_snapshot, eg2_state_dict)
    return model


@pytest.fixture(scope="session")
def reference_for(eg2_snapshot, hf_text_model):
    """reference_for(lengths) -> (input_ids [B,L], attention_mask [B,L], ReferenceOutputs).

    Captures every layer's in/out plus attention/MLP in/out for layers 0 (sliding) and 5 (full).
    Lengths >= 16384 capture layer in/out only for those two layers (host RAM).
    """
    from models.demos.blackhole.embedding_gemma2.reference.hf_reference import run_reference
    from models.demos.blackhole.embedding_gemma2.tests.common import make_batch

    cache = {}

    def get(lengths):
        key = tuple(lengths)
        if key not in cache:
            if len(cache) >= 2:
                cache.pop(next(iter(cache)))
            ids, mask = make_batch(eg2_snapshot, list(lengths))
            layers = None if max(lengths) < 16384 else (0, 5)
            cache[key] = (
                ids,
                mask,
                run_reference(hf_text_model, ids, mask, capture_layers=layers, capture_sublayers=(0, 5)),
            )
        return cache[key]

    return get


@pytest.fixture(scope="session")
def eg2_tokenizer(eg2_snapshot):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(eg2_snapshot))


@pytest.fixture(scope="module")
def eg2_device(request):
    device_id = int(request.config.getoption("device_id", 0) or 0)
    device = ttnn.open_device(device_id=device_id, **DEVICE_PARAMS)
    yield device
    ttnn.close_device(device)


@pytest.fixture(scope="module")
def tt_text_model(eg2_device, eg2_state_dict, eg2_config, eg2_cache_dir):
    from models.demos.blackhole.embedding_gemma2.tt.text_model import TextModel

    return TextModel.from_state_dict(eg2_state_dict, eg2_config, eg2_device, cache_dir=eg2_cache_dir)
