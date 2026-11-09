# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the model descriptor the prefill runner writes for tt-d-gen."""

import json
import os

import pytest

from models.demos.common.prefill.runners import model_descriptor

ARGS = dict(
    model_id="moonshotai/Kimi-K2.7-Code",
    max_context_len=262144,
    max_slots=16,
    kv_depth=131072,
    num_layers=61,
    page_tokens=32,
    chunk_size=5120,
    sp_factor=8,
    max_chunks_per_batch=2,
    num_mtp_tokens=1,
    service_id="kimi27_prefill",
    ack_shm_name="/tt_prefill_layer_completions_kimi27_prefill",
)


def test_the_descriptor_carries_the_runners_settings():
    d = model_descriptor.prefill_descriptor(**ARGS)
    producer = d.pop("producer")
    assert producer["name"] == "tt-metal-prefill" and producer["pid"] == os.getpid()
    assert producer["started_at"] == pytest.approx(model_descriptor.process_started_at(), abs=0.01)
    assert d == {
        "schema_version": 1,
        "role": "prefill",
        "model": {"id": "moonshotai/Kimi-K2.7-Code", "max_context_len": 262144},
        "capacity": {"max_slots": 16, "kv_depth": 131072},
        "kv": {"num_layers": 61, "layer_begin": 0, "layer_end": 61, "page_tokens": 32},
        "prefill": {"chunk_size": 5120, "sp_factor": 8, "max_chunks_per_batch": 2, "num_mtp_tokens": 1},
        "io": {"service_id": "kimi27_prefill", "ack_shm_name": "/tt_prefill_layer_completions_kimi27_prefill"},
    }


def test_write_replaces_the_file_in_one_rename(tmp_path):
    path = tmp_path / "model.json"
    path.write_text("{}")
    model_descriptor.write(str(path), model_descriptor.prefill_descriptor(**ARGS))
    assert json.loads(path.read_text())["prefill"]["chunk_size"] == 5120
    assert [p.name for p in tmp_path.iterdir()] == ["model.json"]
