# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the M3 adapter opts into packed forwards and builds its runtime at the runner's forward width, with the
media reader its env selects, and every M3 runner binding starts under the runner's packed-forward checks."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.common.prefill.prefill_chunk import packed_forward_refusals
from models.demos.minimax_m3.multimodal.stub_reader import StubEmbeddingReader
from models.demos.minimax_m3.tt import tt_prefill_runtime, weight_cache
from models.demos.minimax_m3.tt.runners.adapters.minimax_m3 import MiniMaxM3PrefillAdapter

MANIFESTS = Path(__file__).parents[2] / "tt" / "runners" / "manifests"
REPO_ROOT = Path(__file__).parents[5]


def run_params(chunk_size, forward_chunks):
    return PrefillRunParams(
        mesh_shape=(8, 4),
        num_layers=2,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=10240,
        chunk_size=chunk_size,
        num_users=2,
        capacity_factor=1,
        num_links=1,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=False,
        weight_cache_path=None,
        forward_chunks=forward_chunks,
    )


@pytest.mark.parametrize("chunk_size, forward_chunks", [(5120, 1), (1024, 4), (1024, 8)])
def test_build_runtime_forward_width(monkeypatch, chunk_size, forward_chunks):
    built = {}
    monkeypatch.setattr(weight_cache, "weight_cache_is_complete", lambda *args, **kwargs: True)
    monkeypatch.setattr(tt_prefill_runtime, "TtPrefillRuntime", lambda **kwargs: built.update(kwargs))
    MiniMaxM3PrefillAdapter().build_runtime(
        mesh_device=None, hf_config=None, params=run_params(chunk_size, forward_chunks)
    )
    config = built["config"]
    assert config.chunk_size == chunk_size
    assert config.forward_tokens_budget == forward_chunks * chunk_size
    assert config.forward_chunks == forward_chunks


STORE = {"PREFILL_MEDIA_STORE_MASTER": "master:50051"}
REFS = {"PREFILL_MEDIA_MAX_REFS": "4"}
STUB = {"PREFILL_MEDIA_STUB_EMBEDDINGS": "1"}


@pytest.mark.parametrize(
    "env,is_first_rank,want",
    [
        ({}, True, None),
        (STORE, True, None),
        (STORE | STUB, True, None),
        (REFS | STORE, True, "mooncake"),
        (REFS | STORE | STUB, True, "mooncake"),
        (REFS | STUB, True, StubEmbeddingReader),
        (REFS | STORE, False, None),
    ],
    ids=["text_only", "store_without_refs", "stub_without_refs", "store", "store_over_stub", "stub", "later_rank"],
)
def test_build_runtime_media_reader(monkeypatch, env, is_first_rank, want):
    """The first rank opens a media reader only when it takes media refs: the store when configured, else the stub."""
    built = {}
    monkeypatch.setattr(weight_cache, "weight_cache_is_complete", lambda *args, **kwargs: True)
    monkeypatch.setattr(tt_prefill_runtime, "TtPrefillRuntime", lambda **kwargs: built.update(kwargs))
    mooncake = SimpleNamespace(MooncakeEmbeddingReader=SimpleNamespace(open=lambda config: "mooncake"))
    monkeypatch.setitem(sys.modules, "models.common.multimedia_store.mooncake_store", mooncake)
    for name in (*STORE, *REFS, *STUB):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PREFILL_MEDIA_STORE_SEGMENT", "host")
    monkeypatch.setenv("PREFILL_MEDIA_STORE_METADATA", "http://master:8080/metadata")
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    params = replace(run_params(1024, 1), is_first_rank=is_first_rank)
    MiniMaxM3PrefillAdapter().build_runtime(mesh_device=None, hf_config=SimpleNamespace(hidden_size=64), params=params)

    reader = built["media_embedding_reader"]
    assert isinstance(reader, want) if isinstance(want, type) else reader == want


def test_build_runtime_media_refs_need_a_reader(monkeypatch, expect_error):
    monkeypatch.setattr(weight_cache, "weight_cache_is_complete", lambda *args, **kwargs: True)
    for name in (*STORE, *STUB):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PREFILL_MEDIA_MAX_REFS", "4")
    with expect_error(ValueError, "PREFILL_MEDIA_STUB_EMBEDDINGS=1"):
        MiniMaxM3PrefillAdapter().build_runtime(mesh_device=None, hf_config=None, params=run_params(1024, 1))


def test_packed_forward_accepted():
    assert MiniMaxM3PrefillAdapter.supports_packed_forward
    assert packed_forward_refusals(4, supports_packed_forward=MiniMaxM3PrefillAdapter.supports_packed_forward) == []


@pytest.mark.parametrize("binding", sorted(MANIFESTS.glob("m3_binding_*.yaml")), ids=lambda p: p.stem)
def test_binding_env_starts(binding):
    """The env the runner sees: the binding's global_env, then the model manifest's env as defaults."""
    env = dict(yaml.safe_load(binding.read_text())["global_env"])
    manifest = REPO_ROOT / env["PREFILL_MANIFEST"].replace("${TT_METAL_HOME}/", "")
    for key, val in json.loads(manifest.read_text())["env"].items():
        env.setdefault(key, str(val))
    chunk_size = int(env["PREFILL_CHUNK_SIZE"])
    budget = int(env.get("PREFILL_FORWARD_TOKENS_BUDGET", chunk_size))
    assert budget % chunk_size == 0
    assert env.get("PREFILL_USE_TRACE", "0") == "0", "TtPrefillRuntime has no capture_trace"
    refusals = packed_forward_refusals(
        budget // chunk_size,
        supports_packed_forward=MiniMaxM3PrefillAdapter.supports_packed_forward,
        use_trace=False,
        d2h_acks=env.get("PREFILL_LAYER_ACK_D2H", "0") == "1",
    )
    assert refusals == []
