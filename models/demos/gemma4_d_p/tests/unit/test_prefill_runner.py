# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import (
    Gemma4PrefillAdapter,
    Gemma4ServiceConfig,
    validate_params,
)
from models.demos.gemma4_d_p.tt.runners.prepare_prefill_inputs import write_producer_manifest
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime


def service_params():
    return PrefillRunParams(
        mesh_shape=(8, 4),
        num_layers=60,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=262144,
        chunk_size=Gemma4ServiceConfig.CHUNK_SIZE,
        num_users=6,
        capacity_factor=8,
        num_links=2,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=True,
        weight_cache_path=None,
        use_trace=True,
    )


def test_adapter_uses_canonical_weight_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("PREFILL_TTNN_CACHE", raising=False)
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path))
    (tmp_path / "tensor_cache_bf16_mesh8x4").mkdir()
    adapter = get_adapter("gemma4_d_p")
    assert isinstance(adapter, Gemma4PrefillAdapter)
    assert adapter.weight_cache_path((8, 4)) == tmp_path / "tensor_cache_bf16_mesh8x4"
    validate_params(service_params())


@pytest.mark.parametrize(
    "override",
    [{"num_users": 7}, {"mesh_shape": (4, 8)}, {"num_layers": 30}, {"chunk_size": 2 * Gemma4ServiceConfig.CHUNK_SIZE}],
)
def test_unsupported_service_shapes(override, expect_error):
    with expect_error(ValueError, "Gemma4|KV slot|Chunk|Slot"):
        validate_params(replace(service_params(), **override))


def test_slot_offsets_and_partial_final_chunk(expect_error):
    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = service_params()
    c, last = runtime.config.chunk_size, runtime.config.max_seq_len
    runtime.slot_ends = [c, 2 * c, 0, 0, 0, last - c]
    runtime.validate_chunk(0, c, 2 * c)
    runtime.validate_chunk(1, 2 * c, 2 * c + 1)
    runtime.validate_chunk(5, last - c, last)
    runtime.validate_chunk(5, 0, c)
    for chunk in [(6, 0, c), (0, 2 * c, 3 * c), (0, 1, c + 1), (5, last - c, last + 1)]:
        with expect_error(ValueError, "Gemma4|KV slot|Chunk|Slot"):
            runtime.validate_chunk(*chunk)


@pytest.mark.parametrize("chunk_size, step", [(2048, 256), (4096, 512), (8192, 1024)])
def test_mid_chunk_starts(chunk_size, step, expect_error):
    """A chunk may start on any slab boundary (chunk / CP), and its padded chunk must fit the slot."""
    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = replace(service_params(), chunk_size=chunk_size)
    start, late = 3 * chunk_size + step, runtime.config.max_seq_len - chunk_size + step
    runtime.slot_ends = [start, late, 0, 0, 0, 0]
    runtime.validate_chunk(0, start, start + chunk_size)
    runtime.validate_chunk(0, start - step, start)  # rewinds into the cached prefix
    with expect_error(ValueError, "multiple of"):
        runtime.validate_chunk(0, start + step // 2, start + chunk_size)
    with expect_error(ValueError, "runs past"):
        runtime.validate_chunk(1, late, late + 1)
    with expect_error(ValueError, "holds"):
        runtime.validate_chunk(0, start + step, start + chunk_size)


@pytest.mark.parametrize("chunk_size", [2048, 4096, 8192])
def test_simulator_resumes_at_the_runtime_granularity(chunk_size):
    """The AgentX simulator's --align resume rounds to the runtime's granularity."""
    from models.demos.gemma4_d_p.agentx_sim.agentx_sim import resume_step

    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = replace(service_params(), chunk_size=chunk_size)
    assert runtime.resume_granularity == resume_step(chunk_size, "resume")


@pytest.mark.parametrize("slabs", [0, 3, 7], ids=["aligned", "slab3", "slab7"])
def test_chunk_positions_follow_the_rotated_tokens(slabs):
    """A chunk's staged positions are its tokens' positions after the producer's rotation: natural order at an
    aligned start, rotated at a mid-chunk one."""
    from types import SimpleNamespace

    from models.demos.common.prefill.chunk_layout import rotate_chunk_tokens

    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = service_params()
    runtime.mesh_config = SimpleNamespace(cp_degree=8)
    c = runtime.config.chunk_size
    start = 4 * c + slabs * (c // 8)
    natural = list(range(start, start + c))
    assert runtime._chunk_positions(start) == rotate_chunk_tokens(natural, start, 8)
    assert (runtime._chunk_positions(start) == natural) == (start % c == 0)


# Positions as functions of the service chunk c: (slot_id, previous_end, actual_start, actual_end).
@pytest.mark.parametrize(
    "case",
    [
        lambda c: (0, c + 321, c, c + c // 2),
        lambda c: (1, 8 * c, c, c + c // 4),
        lambda c: (5, 2 * c, c, 2 * c),
    ],
    ids=["partial-tail-replay", "shorter-shared-prefix", "full-chunk-replay"],
)
def test_prefix_reuse_replays_computed_chunks(case):
    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = service_params()
    slot_id, previous_end, actual_start, actual_end = case(runtime.config.chunk_size)
    runtime.slot_ends = [0] * runtime.config.num_users
    runtime.slot_ends[slot_id] = previous_end
    runtime.validate_chunk(slot_id, actual_start, actual_end)


# (slot_id, previous_end, actual_start, actual_end) as functions of the service chunk c.
@pytest.mark.parametrize(
    "case",
    [lambda c: (0, 2 * c, 0, c), lambda c: (1, 8 * c, c, c + c // 2)],
    ids=["restart-at-zero", "shorter-shared-prefix"],
)
def test_prefill_replay_invalidates_old_suffix_before_continuing(monkeypatch, expect_error, case):
    import ttnn

    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = service_params()
    c = runtime.config.chunk_size
    slot_id, previous_end, actual_start, actual_end = case(c)
    runtime.mesh_config = SimpleNamespace(cp_degree=8)
    runtime.mesh_device = object()
    runtime.input_tokens = object()
    runtime.trace_id = 1
    runtime.d2h_service = None
    runtime.layer_completion_sink = None
    runtime.slot_ends = [c] * runtime.config.num_users
    runtime.slot_ends[slot_id] = previous_end
    monkeypatch.setattr(runtime, "_check_cache", Mock())
    monkeypatch.setattr(runtime, "_stage_positions", Mock())
    for name in ("reshape", "copy", "execute_trace", "synchronize_device", "deallocate"):
        monkeypatch.setattr(ttnn, name, Mock())

    kv_cache = object()

    def prefill(start, end):
        runtime.prefill_chunk(object(), kv_cache, slot_id=slot_id, actual_start=start, actual_end=end)

    prefill(actual_start, actual_end)
    expected_ends = [c] * runtime.config.num_users
    expected_ends[slot_id] = actual_end
    assert runtime.slot_ends == expected_ends

    with expect_error(ValueError, f"Slot {slot_id} holds {actual_end} tokens, got a chunk at {previous_end}"):
        prefill(previous_end, previous_end + runtime.config.chunk_size)

    # Recompute the suffix, replaying the partial tail when needed, then extend past the old end.
    chunk_size = runtime.config.chunk_size
    for start in range(actual_end // chunk_size * chunk_size, previous_end + chunk_size, chunk_size):
        prefill(start, start + chunk_size)
        expected_ends[slot_id] = start + chunk_size
        assert runtime.slot_ends == expected_ends


# (previous_end, actual_start, actual_end, error) as functions of the service chunk c. A start c + c // 16 is half a
# slab (c / 8) past a boundary, so never on one.
@pytest.mark.parametrize(
    "case",
    [
        lambda c: (c + 321, 2 * c, 3 * c, f"Slot 0 holds {c + 321} tokens, got a chunk at {2 * c}"),
        lambda c: (0, c, c + c // 2, f"Slot 0 holds 0 tokens, got a chunk at {c}"),
        lambda c: (c + 321, c + c // 16, c + c // 2, "Chunk start"),
        lambda c: (c + 321, c, c, "Chunk must contain"),
        lambda c: (c + 321, c, 2 * c + 1, "Chunk must contain"),
    ],
    ids=["gap", "empty-slot", "unaligned-replay", "empty-chunk", "oversized-chunk"],
)
def test_prefix_reuse_rejects_invalid_chunks(case, expect_error):
    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = service_params()
    previous_end, actual_start, actual_end, error = case(runtime.config.chunk_size)
    runtime.slot_ends = [previous_end] + [0] * (runtime.config.num_users - 1)
    with expect_error(ValueError, error):
        runtime.validate_chunk(0, actual_start, actual_end)


def test_prepared_prompts_use_shared_producer_format(tmp_path):
    from models.demos.common.prefill.runners.runner_utils import load_trace_token_ids

    prompts = [[1, 2, 3], [4, 5]]
    manifest = json.loads(write_producer_manifest(tmp_path, prompts).read_text())
    assert manifest["env"]["PREFILL_MODEL"] == "gemma4_d_p"
    assert manifest["env"]["PREFILL_NUM_USERS"] == "2"
    assert manifest["workload"]["max_requests"] == 2
    assert manifest["workload"]["interleave"] == "round_robin"
    assert not manifest["workload"]["check_pcc"]
    assert [load_trace_token_ids(path) for path in manifest["workload"]["slot_prompts"]] == prompts


@pytest.mark.parametrize("prefill_override", [False, True])
def test_tt_cache_resolution(monkeypatch, tmp_path, prefill_override):
    import ttnn
    from models.demos.gemma4_d_p.tt.model_config import resolve_cache_dir_from_tt_cache_path

    monkeypatch.delenv("PREFILL_TTNN_CACHE", raising=False)
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path / "regular"))
    expected = tmp_path / "regular/tensor_cache_bf16_mesh8x4"
    expected.mkdir(parents=True)
    assert (
        resolve_cache_dir_from_tt_cache_path(tmp_path / "regular", dtype=ttnn.bfloat16, mesh_shape=(8, 4)) == expected
    )
    if prefill_override:
        monkeypatch.setenv("PREFILL_TTNN_CACHE", str(tmp_path / "service"))
        expected = tmp_path / "service/tensor_cache_bf16_mesh8x4"
        expected.mkdir(parents=True)
    assert Gemma4PrefillAdapter().weight_cache_path((8, 4)) == expected


def test_tt_cache_requires_configured_root(monkeypatch, tmp_path, expect_error):
    import ttnn
    from models.demos.gemma4_d_p.tt.model_config import resolve_cache_dir_from_tt_cache_path

    monkeypatch.delenv("TT_CACHE_PATH", raising=False)
    monkeypatch.delenv("PREFILL_TTNN_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_MODEL", str(tmp_path))
    with expect_error(ValueError, "tt_cache_path must be provided"):
        resolve_cache_dir_from_tt_cache_path(None, dtype=ttnn.bfloat16, mesh_shape=(8, 4))
    with expect_error(ValueError, "tt_cache_path must be provided"):
        Gemma4PrefillAdapter().weight_cache_path((8, 4))
