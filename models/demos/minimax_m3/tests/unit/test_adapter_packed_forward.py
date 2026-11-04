# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the M3 adapter opts into packed forwards and builds its runtime at the runner's forward width, and
every M3 runner binding starts under the runner's packed-forward checks."""

import json
from pathlib import Path

import pytest
import yaml

from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.common.prefill.prefill_chunk import packed_forward_refusals
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
