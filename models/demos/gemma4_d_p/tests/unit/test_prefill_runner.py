# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import Gemma4PrefillAdapter, validate_params
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
        chunk_size=8192,
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
    "override", [{"num_users": 7}, {"mesh_shape": (4, 8)}, {"num_layers": 30}, {"chunk_size": 4096}]
)
def test_unsupported_service_shapes(override, expect_error):
    with expect_error(ValueError, "Gemma4|KV slot|Chunk|Slot"):
        validate_params(replace(service_params(), **override))


def test_slot_offsets_and_partial_final_chunk(expect_error):
    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = service_params()
    runtime.slot_ends = [8192, 16384, 0, 0, 0, 253952]
    runtime.validate_chunk(0, 8192, 16384)
    runtime.validate_chunk(1, 16384, 16385)
    runtime.validate_chunk(5, 253952, 262144)
    runtime.validate_chunk(5, 0, 8192)
    for chunk in [(6, 0, 8192), (0, 16384, 24576), (0, 1, 8193), (5, 253952, 262145)]:
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


@pytest.mark.parametrize("start", [16384, 16384 + 3 * 1024, 16384 + 7 * 1024])
def test_chunk_positions_follow_the_rotated_tokens(start):
    """A chunk's staged positions are its tokens' positions after the producer's rotation: natural order at an
    aligned start, rotated at a mid-chunk one."""
    from types import SimpleNamespace

    from models.demos.common.prefill.chunk_layout import rotate_chunk_tokens

    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = service_params()
    runtime.mesh_config = SimpleNamespace(cp_degree=8)
    natural = list(range(start, start + 8192))
    assert runtime._chunk_positions(start) == rotate_chunk_tokens(natural, start, 8)
    assert (runtime._chunk_positions(start) == natural) == (start % 8192 == 0)


@pytest.mark.parametrize(
    "slot_id,previous_end,actual_start,actual_end",
    [
        (0, 8513, 8192, 12473),
        (1, 32768, 8192, 12000),
        (5, 16384, 8192, 16384),
    ],
    ids=["partial-tail-replay", "shorter-shared-prefix", "full-chunk-replay"],
)
def test_prefix_reuse_replays_computed_chunks(slot_id, previous_end, actual_start, actual_end):
    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = service_params()
    runtime.slot_ends = [0] * runtime.config.num_users
    runtime.slot_ends[slot_id] = previous_end
    runtime.validate_chunk(slot_id, actual_start, actual_end)


@pytest.mark.parametrize(
    "slot_id,previous_end,actual_start,actual_end",
    [(0, 16384, 0, 8192), (1, 32768, 8192, 12000)],
    ids=["restart-at-zero", "shorter-shared-prefix"],
)
def test_prefill_replay_invalidates_old_suffix_before_continuing(
    monkeypatch, expect_error, slot_id, previous_end, actual_start, actual_end
):
    import ttnn

    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = service_params()
    runtime.mesh_config = SimpleNamespace(cp_degree=8)
    runtime.mesh_device = object()
    runtime.input_tokens = object()
    runtime.trace_id = 1
    runtime.d2h_service = None
    runtime.layer_completion_sink = None
    runtime.slot_ends = [8192] * runtime.config.num_users
    runtime.slot_ends[slot_id] = previous_end
    monkeypatch.setattr(runtime, "_check_cache", Mock())
    monkeypatch.setattr(runtime, "_stage_positions", Mock())
    for name in ("reshape", "copy", "execute_trace", "synchronize_device", "deallocate"):
        monkeypatch.setattr(ttnn, name, Mock())

    kv_cache = object()

    def prefill(start, end):
        runtime.prefill_chunk(object(), kv_cache, slot_id=slot_id, actual_start=start, actual_end=end)

    prefill(actual_start, actual_end)
    expected_ends = [8192] * runtime.config.num_users
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


@pytest.mark.parametrize(
    "previous_end,actual_start,actual_end,error",
    [
        (8513, 16384, 24576, "Slot 0 holds 8513 tokens, got a chunk at 16384"),
        (0, 8192, 12473, "Slot 0 holds 0 tokens, got a chunk at 8192"),
        (8513, 8512, 12473, "Chunk start"),
        (8513, 8192, 8192, "Chunk must contain"),
        (8513, 8192, 16385, "Chunk must contain"),
    ],
    ids=["gap", "empty-slot", "unaligned-replay", "empty-chunk", "oversized-chunk"],
)
def test_prefix_reuse_rejects_invalid_chunks(previous_end, actual_start, actual_end, error, expect_error):
    runtime = Gemma4PrefillRuntime.__new__(Gemma4PrefillRuntime)
    runtime.config = service_params()
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
