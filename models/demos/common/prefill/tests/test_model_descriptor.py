# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the model descriptor the prefill runner writes for tt-d-gen."""

import json
import os
from types import SimpleNamespace

import pytest

from models.demos.common.prefill.runners import model_descriptor

ARGS = dict(
    model_id="moonshotai/Kimi-K2.7-Code",
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
        "model": {"id": "moonshotai/Kimi-K2.7-Code"},
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


def test_remove_drops_a_previous_runs_descriptor(tmp_path):
    path = tmp_path / "model.json"
    path.write_text("{}")
    model_descriptor.remove(str(path))
    assert not path.exists()
    model_descriptor.remove(str(path))


def test_a_master_rank_launched_without_the_descriptor_path_stops_the_run(monkeypatch, expect_error):
    """Only the master rank writes the descriptor, so the ranks agree on its path before any of them sets up."""
    from models.demos.common.prefill.runners import prefill_runner as runner

    monkeypatch.setattr(runner, "ADAPTER", SimpleNamespace(name="kimi_k3", default_sparse_kv_cache_format=None))
    monkeypatch.setenv(model_descriptor.PATH_ENV, "/run/model.json")
    rank0 = []
    monkeypatch.setattr(runner.ttnn, "distributed_context_allgather_int", lambda d: rank0.append(d) or [d, d])
    runner._assert_ranks_agree_on_config(0, 2)
    monkeypatch.delenv(model_descriptor.PATH_ENV)
    monkeypatch.setattr(runner.ttnn, "distributed_context_allgather_int", lambda d: [rank0[0], d])
    with expect_error(RuntimeError, "DIFFERENT prefill configs"):
        runner._assert_ranks_agree_on_config(1, 2)


def test_the_checkpoint_falls_back_to_the_default_when_prefill_hf_model_is_empty(monkeypatch):
    from models.demos.common.prefill.adapter import PrefillModelAdapter

    adapter = SimpleNamespace(hf_model_default="/ckpt/default")
    monkeypatch.setenv("PREFILL_HF_MODEL", "")
    assert PrefillModelAdapter.hf_model_path.fget(adapter) == "/ckpt/default"
    monkeypatch.setenv("PREFILL_HF_MODEL", "/ckpt/override")
    assert PrefillModelAdapter.hf_model_path.fget(adapter) == "/ckpt/override"


def test_an_mla_model_names_the_checkpoint_its_weights_come_from(monkeypatch):
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter

    adapter = SimpleNamespace(env_var="KIMI_K3_HF_MODEL", hf_model_default="/ckpt/default")
    monkeypatch.delenv("PREFILL_HF_MODEL", raising=False)
    monkeypatch.setenv("KIMI_K3_HF_MODEL", "/ckpt/k3")
    assert MLAPrefillAdapter.hf_model_path.fget(adapter) == "/ckpt/k3"
    monkeypatch.setenv("PREFILL_HF_MODEL", "/ckpt/override")
    assert MLAPrefillAdapter.hf_model_path.fget(adapter) == "/ckpt/override"


def test_minimax_names_an_hf_model_checkpoint(monkeypatch):
    """MiniMax-M3 loads ``$HF_MODEL`` when ``$PREFILL_HF_MODEL`` is unset; the descriptor must say so."""
    from models.demos.minimax_m3.tt.runners.adapters.minimax_m3 import MiniMaxM3PrefillAdapter

    adapter = SimpleNamespace(variant=SimpleNamespace(weights_dir="/ckpt/variant"))
    monkeypatch.delenv("PREFILL_HF_MODEL", raising=False)
    monkeypatch.delenv("HF_MODEL", raising=False)
    assert MiniMaxM3PrefillAdapter.hf_model_path.fget(adapter) == "/ckpt/variant"
    monkeypatch.setenv("HF_MODEL", "/ckpt/m3")
    assert MiniMaxM3PrefillAdapter.hf_model_path.fget(adapter) == "/ckpt/m3"
