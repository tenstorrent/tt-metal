# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.autoports.qwen_qwen3_32b.tests.test_functional_decoder import (
    _assert_pcc,
    _config,
    _hf_layer,
    _real_state,
    _reference_layer,
    _synthetic_state,
    _to_host,
)
from models.autoports.qwen_qwen3_32b.tests.test_optimized_decoder import _recorded_hidden
from models.autoports.qwen_qwen3_32b.tt.functional_decoder import (
    EMITTED_BATCH,
    REPRESENTATIVE_LAYER,
)
from models.autoports.qwen_qwen3_32b.tt.multichip_decoder import (
    PAGE_BLOCK_SIZE,
    TARGET_MESH_SHAPE,
    TP_DEGREE,
    MultichipDecoder,
)
from models.autoports.qwen_qwen3_32b.tt.optimized_decoder import OptimizedDecoder

BASELINE_PATH_ENV = "QWEN3_32B_MULTICHIP_BASELINE_PATH"
RESULTS_DIR_ENV = "QWEN3_32B_MULTICHIP_RESULTS_DIR"


def _optional_env_int(name: str) -> int | None:
    return int(os.environ[name]) if name in os.environ else None


def _write_result(name: str, payload: dict) -> Path:
    output_dir = Path(
        os.getenv(
            RESULTS_DIR_ENV,
            Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder/results",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    decoder_path = Path(__file__).resolve().parents[1] / "tt/multichip_decoder.py"
    test_path = Path(__file__).resolve()
    payload["provenance"] = {
        "decoder_sha256": hashlib.sha256(decoder_path.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "hardware": "4x Blackhole p300c, 1x4 FABRIC_1D_RING",
    }
    output_path = output_dir / name
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def _dram_snapshot(mesh_device) -> dict:
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return {
        "num_banks": view.num_banks,
        "total_bytes": view.num_banks * view.total_bytes_per_bank,
        "allocated_bytes": view.num_banks * view.total_bytes_allocated_per_bank,
        "free_bytes": view.num_banks * view.total_bytes_free_per_bank,
        "largest_contiguous_bytes_free_per_bank": view.largest_contiguous_bytes_free_per_bank,
    }


def _tp_hidden(host: torch.Tensor, mesh_device):
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )


def _replicated_hidden(host: torch.Tensor, mesh_device):
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _compose_hidden(tensor, mesh_device) -> torch.Tensor:
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))


def _compose_cache(tensor, mesh_device) -> torch.Tensor:
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))


def _compose_padded_sdpa_heads(tensor) -> torch.Tensor:
    local_heads = []
    for shard in ttnn.get_device_tensors(tensor):
        host = ttnn.to_torch(shard)
        local_heads.append(torch.cat([host[:, :, :8, :], host[:, :, 16:24, :]], dim=2))
    return torch.cat(local_heads, dim=2)


def _unpage_cache(physical: torch.Tensor, page_table: torch.Tensor) -> torch.Tensor:
    users = []
    for row in page_table:
        users.append(torch.cat([physical[int(block)] for block in row], dim=1))
    return torch.stack(users)


def _copy_replicated_page_table(host_table: torch.Tensor, device_table, mesh_device) -> None:
    """Refresh a trace-bound replicated page table without changing its address."""

    host_mesh = ttnn.from_torch(
        host_table.to(torch.int32).contiguous(),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.copy_host_to_device_tensor(host_mesh, device_table)


def test_multichip_contract_is_optimized_owned_and_host_free():
    assert issubclass(MultichipDecoder, OptimizedDecoder)
    assert TARGET_MESH_SHAPE == (1, 4)
    assert TP_DEGREE == 4
    for method in (
        MultichipDecoder._all_gather_hidden,
        MultichipDecoder._reduce_scatter_hidden,
        MultichipDecoder._fused_decode_row_parallel,
        MultichipDecoder._prefill_linear,
        MultichipDecoder._prefill_row_parallel,
        MultichipDecoder._fill_prefill_cache,
        MultichipDecoder._prefill_mlp_chunk,
        MultichipDecoder._prefill_mlp,
        MultichipDecoder.prefill_forward,
        MultichipDecoder._decode_mlp,
        MultichipDecoder.decode_forward,
    ):
        source = inspect.getsource(method)
        for token in ("torch", "from_torch", "to_torch", "super()"):
            assert token not in source, f"{method.__name__} contains runtime fallback token {token!r}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_multichip_synthetic_non_aligned_prefill_matches_hf(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    model = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
        precision_policy="mlp_bfp4_attention_lofi",
    )
    plan = model.mesh_plan_summary()
    assert plan["mesh_shape"] == [1, 4]
    assert plan["local_shapes"] == {
        "residual": 1280,
        "q_heads": 16,
        "kv_heads": 2,
        "qkv": 2560,
        "attention": 2048,
        "intermediate": 6400,
    }

    seq_len = 31
    generator = torch.Generator().manual_seed(32041)
    hidden = torch.randn(
        (1, EMITTED_BATCH, seq_len, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    reference_layer = _hf_layer(state, config)
    reference, reference_key, reference_value, reference_cache = _reference_layer(
        reference_layer, hidden, config
    )
    key_cache, value_cache = model.allocate_kv_cache()
    output = model.prefill_forward(_tp_hidden(hidden, mesh_device), key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)

    _assert_pcc(reference, _compose_hidden(output, mesh_device), 0.99, "multichip non-aligned prefill")
    _assert_pcc(
        reference_key,
        _compose_cache(key_cache, mesh_device)[:, :, :seq_len, :],
        0.99,
        "multichip local-head key cache",
    )
    _assert_pcc(
        reference_value,
        _compose_cache(value_cache, mesh_device)[:, :, :seq_len, :],
        0.99,
        "multichip local-head value cache",
    )

    stacked_reference, _, _, stacked_reference_cache = _reference_layer(
        reference_layer, reference, config
    )
    stacked_key, stacked_value = model.allocate_kv_cache()
    stacked_output = model.prefill_forward(output, stacked_key, stacked_value)
    ttnn.synchronize_device(mesh_device)
    assert tuple(stacked_output.shape) == (1, EMITTED_BATCH, seq_len, config.hidden_size // TP_DEGREE)
    _assert_pcc(
        stacked_reference,
        _compose_hidden(stacked_output, mesh_device),
        0.99,
        "direct TP-sharded decoder-to-decoder prefill contract",
    )

    decode_hidden = torch.randn(
        (1, EMITTED_BATCH, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    expected_decode, expected_key, expected_value, _ = _reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=seq_len,
        cache=reference_cache,
    )
    decode_output = model.decode_forward(
        _tp_hidden(decode_hidden, mesh_device),
        key_cache,
        value_cache,
        current_pos=seq_len,
    )
    ttnn.synchronize_device(mesh_device)
    _assert_pcc(
        expected_key,
        _compose_cache(key_cache, mesh_device)[:, :, seq_len : seq_len + 1, :],
        0.99,
        "multichip decode key cache",
    )
    _assert_pcc(
        expected_value,
        _compose_cache(value_cache, mesh_device)[:, :, seq_len : seq_len + 1, :],
        0.99,
        "multichip decode value cache",
    )
    _assert_pcc(
        expected_decode,
        _compose_hidden(decode_output, mesh_device),
        0.98,
        "multichip conservative synthetic decode",
    )
    for shard in ttnn.get_device_tensors(decode_output):
        assert shard.is_sharded()
        assert shard.memory_config().buffer_type == ttnn.BufferType.L1
    stacked_expected_decode, _, _, _ = _reference_layer(
        reference_layer,
        expected_decode,
        config,
        start_pos=seq_len,
        cache=stacked_reference_cache,
    )
    stacked_decode_output = model.decode_forward(
        decode_output,
        stacked_key,
        stacked_value,
        current_pos=seq_len,
    )
    ttnn.synchronize_device(mesh_device)
    _assert_pcc(
        stacked_expected_decode,
        _compose_hidden(stacked_decode_output, mesh_device),
        0.98,
        "direct TP-sharded decoder-to-decoder decode contract",
    )
    ttnn.deallocate(stacked_decode_output, True)

    permutation = torch.arange(2 * EMITTED_BATCH - 1, -1, -1, dtype=torch.int32)
    host_page_table = permutation.reshape(EMITTED_BATCH, 2)
    page_table = model.allocate_page_table(permutation=permutation)
    paged_key, paged_value = model.allocate_kv_cache(paged=True)
    paged_prefill = model.prefill_forward(
        _tp_hidden(hidden, mesh_device),
        paged_key,
        paged_value,
        page_table=page_table,
    )
    paged_decode = model.decode_forward(
        _tp_hidden(decode_hidden, mesh_device),
        paged_key,
        paged_value,
        current_pos=seq_len,
        page_table=page_table,
    )
    ttnn.synchronize_device(mesh_device)
    _assert_pcc(
        _compose_hidden(output, mesh_device),
        _compose_hidden(paged_prefill, mesh_device),
        0.999,
        "paged versus contiguous prefill",
    )
    _assert_pcc(
        _compose_hidden(decode_output, mesh_device),
        _compose_hidden(paged_decode, mesh_device),
        0.999,
        "paged versus contiguous decode",
    )
    logical_key = _unpage_cache(_compose_cache(paged_key, mesh_device), host_page_table)
    logical_value = _unpage_cache(_compose_cache(paged_value, mesh_device), host_page_table)
    _assert_pcc(
        _compose_cache(key_cache, mesh_device)[:, :, : seq_len + 1, :],
        logical_key[:, :, : seq_len + 1, :],
        0.999,
        "paged logical key-cache layout",
    )
    _assert_pcc(
        _compose_cache(value_cache, mesh_device)[:, :, : seq_len + 1, :],
        logical_value[:, :, : seq_len + 1, :],
        0.999,
        "paged logical value-cache layout",
    )

    trace_hidden_host = torch.randn(
        (1, EMITTED_BATCH, 1, config.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    )
    expected_trace_32, expected_trace_key_32, expected_trace_value_32, reference_cache = _reference_layer(
        reference_layer,
        trace_hidden_host,
        config,
        start_pos=seq_len + 1,
        cache=reference_cache,
    )
    trace_hidden = _tp_hidden(trace_hidden_host, mesh_device)
    position_buffers = model.allocate_decode_position_buffers(seq_len + 1)
    warm = model.decode_forward(
        trace_hidden,
        key_cache,
        value_cache,
        current_pos=seq_len + 1,
        position_buffers=position_buffers,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm, True)
    model.prepare_decode_position_buffers(position_buffers, seq_len + 2)
    model.prepare_decode_position_buffers(position_buffers, seq_len + 1)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.decode_forward(
        trace_hidden,
        key_cache,
        value_cache,
        current_pos=seq_len + 1,
        position_buffers=position_buffers,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        replay_outputs = []
        for _ in range(10):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            replay_outputs.append(_compose_hidden(trace_output, mesh_device))
        first_trace = replay_outputs[0]
        final_same_position_trace = replay_outputs[-1]
        _assert_pcc(
            expected_trace_32,
            first_trace,
            0.98,
            "multichip traced decode versus HF at position 32",
        )
        _assert_pcc(
            expected_trace_key_32,
            _compose_cache(key_cache, mesh_device)[:, :, seq_len + 1 : seq_len + 2, :],
            0.99,
            "multichip traced key-cache update at position 32",
        )
        _assert_pcc(
            expected_trace_value_32,
            _compose_cache(value_cache, mesh_device)[:, :, seq_len + 1 : seq_len + 2, :],
            0.99,
            "multichip traced value-cache update at position 32",
        )

        model.prepare_decode_position_buffers(position_buffers, seq_len + 2)
        for shard in ttnn.get_device_tensors(position_buffers.update_indices):
            assert torch.equal(
                ttnn.to_torch(shard),
                torch.full((EMITTED_BATCH,), seq_len + 2, dtype=torch.int32),
            )
        expected_trace_33, expected_trace_key_33, expected_trace_value_33, _ = _reference_layer(
            reference_layer,
            trace_hidden_host,
            config,
            start_pos=seq_len + 2,
            cache=reference_cache,
        )
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        advanced_trace = _compose_hidden(trace_output, mesh_device)
        _assert_pcc(
            expected_trace_33,
            advanced_trace,
            0.98,
            "multichip traced decode versus HF at advanced position 33",
        )
        _assert_pcc(
            expected_trace_key_33,
            _compose_cache(key_cache, mesh_device)[:, :, seq_len + 2 : seq_len + 3, :],
            0.99,
            "multichip traced key-cache update at advanced position 33",
        )
        _assert_pcc(
            expected_trace_value_33,
            _compose_cache(value_cache, mesh_device)[:, :, seq_len + 2 : seq_len + 3, :],
            0.99,
            "multichip traced value-cache update at advanced position 33",
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    assert torch.isfinite(first_trace).all()
    _assert_pcc(
        first_trace,
        final_same_position_trace,
        0.99999,
        "multichip 10-replay trace determinism",
    )


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_capture_real_optimized_single_chip_baseline(mesh_device):
    output_path_text = os.getenv(BASELINE_PATH_ENV)
    if not output_path_text:
        pytest.skip(f"Set {BASELINE_PATH_ENV} to capture the independent optimized baseline")
    config = _config()
    state = _real_state()
    baseline = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
        precision_policy="all_bfp4_lofi",
        decode_target_cores=40,
        decode_mlp_target_cores=40,
        decode_down_target_cores=32,
    )
    prefill_hidden, _ = _recorded_hidden(0, 17)
    baseline_key, baseline_value = baseline.allocate_kv_cache()
    baseline_prefill = baseline.prefill_forward(
        ttnn.from_torch(
            prefill_hidden,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        baseline_key,
        baseline_value,
    )
    decode_hidden, _ = _recorded_hidden(17, 1)
    captured = {}
    baseline_decode_mlp = baseline._decode_mlp
    baseline_concat_heads = ttnn.experimental.nlp_concat_heads_decode
    baseline_sdpa = ttnn.transformer.scaled_dot_product_attention_decode

    def capture_baseline_attention_residual(residual):
        captured["attention_residual"] = _to_host(residual)
        return baseline_decode_mlp(residual)

    def capture_baseline_concat_heads(*args, **kwargs):
        result = baseline_concat_heads(*args, **kwargs)
        captured["concat_heads"] = _to_host(result)
        return result

    def capture_baseline_sdpa(*args, **kwargs):
        captured["query"] = _to_host(args[0])
        result = baseline_sdpa(*args, **kwargs)
        captured["sdpa"] = _to_host(result)
        return result

    baseline._decode_mlp = capture_baseline_attention_residual
    ttnn.experimental.nlp_concat_heads_decode = capture_baseline_concat_heads
    ttnn.transformer.scaled_dot_product_attention_decode = capture_baseline_sdpa
    try:
        baseline_decode = baseline.decode_forward(
            ttnn.from_torch(
                decode_hidden,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            baseline_key,
            baseline_value,
            current_pos=17,
        )
    finally:
        ttnn.experimental.nlp_concat_heads_decode = baseline_concat_heads
        ttnn.transformer.scaled_dot_product_attention_decode = baseline_sdpa
    ttnn.synchronize_device(mesh_device)
    output_path = Path(output_path_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # Materialize compact storages.  Several captures are narrow views
            # into full cache/head buffers; saving the views would retain their
            # much larger backing storages in the provenance artifact.
            "prefill": _to_host(baseline_prefill).contiguous(),
            "decode": _to_host(baseline_decode).contiguous(),
            "attention_residual": captured["attention_residual"].contiguous(),
            "concat_heads": captured["concat_heads"].contiguous(),
            "query": captured["query"].contiguous(),
            "sdpa": captured["sdpa"].contiguous(),
            "key_cache": _to_host(baseline_key)[:, :, :18, :].contiguous(),
            "value_cache": _to_host(baseline_value)[:, :, :18, :].contiguous(),
            "metadata": {
                "baseline": "OptimizedDecoder",
                "precision_policy": "all_bfp4_lofi",
                "decode_target_cores": 40,
                "decode_down_target_cores": 32,
                "prefill_length": 17,
                "decode_position": 17,
            },
        },
        output_path,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_multichip_real_layer_matches_optimized_single_chip_baseline(mesh_device):
    baseline_path_text = os.getenv(BASELINE_PATH_ENV)
    if not baseline_path_text or not Path(baseline_path_text).is_file():
        pytest.skip(f"Run the baseline capture first with {BASELINE_PATH_ENV} set")
    baseline = torch.load(baseline_path_text, map_location="cpu", weights_only=True)
    config = _config()
    state = _real_state()
    multichip = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
        precision_policy="all_bfp4_lofi",
        decode_target_cores=20,
        decode_down_target_cores=20,
    )
    prefill_hidden, _ = _recorded_hidden(0, 17)
    multichip_key, multichip_value = multichip.allocate_kv_cache()
    multichip_prefill = multichip.prefill_forward(
        _tp_hidden(prefill_hidden, mesh_device), multichip_key, multichip_value
    )
    decode_hidden, _ = _recorded_hidden(17, 1)
    captured = {}
    multichip_decode_mlp = multichip._decode_mlp
    multichip_concat_heads = ttnn.experimental.nlp_concat_heads_decode
    multichip_sdpa = ttnn.transformer.scaled_dot_product_attention_decode

    def capture_multichip_attention_residual(residual):
        captured["attention_residual"] = _compose_hidden(residual, mesh_device)
        return multichip_decode_mlp(residual)

    def capture_multichip_concat_heads(*args, **kwargs):
        result = multichip_concat_heads(*args, **kwargs)
        captured["concat_heads"] = _compose_hidden(result, mesh_device)
        return result

    def capture_multichip_sdpa(*args, **kwargs):
        captured["query"] = _compose_padded_sdpa_heads(args[0])
        result = multichip_sdpa(*args, **kwargs)
        captured["sdpa"] = _compose_padded_sdpa_heads(result)
        return result

    multichip._decode_mlp = capture_multichip_attention_residual
    ttnn.experimental.nlp_concat_heads_decode = capture_multichip_concat_heads
    ttnn.transformer.scaled_dot_product_attention_decode = capture_multichip_sdpa
    try:
        multichip_decode = multichip.decode_forward(
            _tp_hidden(decode_hidden, mesh_device),
            multichip_key,
            multichip_value,
            current_pos=17,
        )
    finally:
        ttnn.experimental.nlp_concat_heads_decode = multichip_concat_heads
        ttnn.transformer.scaled_dot_product_attention_decode = multichip_sdpa
    ttnn.synchronize_device(mesh_device)
    _assert_pcc(baseline["query"], captured["query"], 0.99, "multichip real query versus optimized baseline")
    _assert_pcc(baseline["sdpa"], captured["sdpa"], 0.99, "multichip real SDPA output versus optimized baseline")
    _assert_pcc(
        baseline["concat_heads"],
        captured["concat_heads"],
        0.99,
        "multichip real concatenated attention versus optimized baseline",
    )
    _assert_pcc(
        baseline["attention_residual"],
        captured["attention_residual"],
        0.99,
        "multichip real attention residual versus optimized baseline",
    )
    _assert_pcc(
        baseline["prefill"],
        _compose_hidden(multichip_prefill, mesh_device),
        0.99,
        "multichip real prefill versus optimized baseline",
    )
    _assert_pcc(
        baseline["key_cache"][:, :, :17, :],
        _compose_cache(multichip_key, mesh_device)[:, :, :17, :],
        0.99,
        "multichip real prefill key cache versus optimized baseline",
    )
    _assert_pcc(
        baseline["value_cache"][:, :, :17, :],
        _compose_cache(multichip_value, mesh_device)[:, :, :17, :],
        0.99,
        "multichip real prefill value cache versus optimized baseline",
    )
    _assert_pcc(
        baseline["decode"],
        _compose_hidden(multichip_decode, mesh_device),
        0.99,
        "multichip real decode versus optimized baseline",
    )
    _assert_pcc(
        baseline["key_cache"][:, :, 17:18, :],
        _compose_cache(multichip_key, mesh_device)[:, :, 17:18, :],
        0.99,
        "multichip real decode key cache versus optimized baseline",
    )
    _assert_pcc(
        baseline["value_cache"][:, :, 17:18, :],
        _compose_cache(multichip_value, mesh_device)[:, :, 17:18, :],
        0.99,
        "multichip real decode value cache versus optimized baseline",
    )
    if os.getenv("TT_METAL_WATCHER"):
        repo_root = Path(__file__).resolve().parents[4]
        watcher_path = repo_root / "generated/watcher/watcher.log"
        watcher_text = watcher_path.read_text(errors="replace")
        fault_patterns = ("error", "assert", "hang", "stuck", "timeout")
        matches = [pattern for pattern in fault_patterns if pattern in watcher_text.lower()]
        assert not matches, f"watcher fault signatures: {matches}"
        retained_watcher_path = (
            Path(__file__).resolve().parents[1]
            / "doc/optimized_multichip_decoder/results/watcher_clean.log"
        )
        retained_watcher_path.write_text(watcher_text)
        _write_result(
            "watcher_clean.json",
            {
                "enabled": os.environ["TT_METAL_WATCHER"],
                "test": "real layer-32 prefill/decode versus optimized TTNN baseline",
                "log_path": str(retained_watcher_path.relative_to(repo_root)),
                "disabled_features": ["ETH"] if os.getenv("TT_METAL_WATCHER_DISABLE_ETH") else [],
                "eth_disable_reason": (
                    "full Watcher instrumentation makes the active-Ethernet ring firmware 27,920 bytes, "
                    "exceeding the 25,600-byte kernel-config buffer"
                    if os.getenv("TT_METAL_WATCHER_DISABLE_ETH")
                    else None
                ),
                "fault_patterns": list(fault_patterns),
                "matches": matches,
                "log_sha256": hashlib.sha256(watcher_text.encode()).hexdigest(),
            },
        )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_multichip_paged_trace_refresh_matches_eager(mesh_device):
    """Exercise the final policy's paged SDPA path through capture and replay.

    The second page of every user is unused by the 64-token prefill.  That lets
    a replay change its physical mapping in place, overwrite position 64 under
    the new mapping, and then advance to 65 without invalidating prior tokens.
    """

    config = _config()
    state = _real_state()
    model = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
        precision_policy="all_bfp4_lofi",
        decode_target_cores=20,
        decode_down_target_cores=20,
    )
    recorded_hidden, activation_metadata = _recorded_hidden(0, 21)
    # The retained prompt artifact is 21 tokens.  Cycle it deterministically to
    # reach the second page; this test compares traced/eager TTNN paths, so no
    # invented reference activation is needed.
    prefill_hidden = recorded_hidden.repeat(1, 1, 4, 1)[:, :, :64, :]
    decode_hidden = recorded_hidden[:, :, :1, :]
    activation_metadata = dict(activation_metadata)
    activation_metadata["transform"] = "cycled 21-token prompt-derived sequence to 64 tokens"
    stable_decode = _tp_hidden(decode_hidden, mesh_device)

    table_a = torch.arange(2 * EMITTED_BATCH, dtype=torch.int32).reshape(EMITTED_BATCH, 2)
    table_b = table_a.clone()
    table_b[:, 1] = table_a.flip(0)[:, 1]
    assert sorted(table_b.flatten().tolist()) == list(range(2 * EMITTED_BATCH))
    page_table = model.allocate_page_table(permutation=table_a.flatten())
    page_table_a_ref = model.allocate_page_table(permutation=table_a.flatten())
    page_table_b_ref = model.allocate_page_table(permutation=table_b.flatten())

    trace_key, trace_value = model.allocate_kv_cache(paged=True)
    eager_a_key, eager_a_value = model.allocate_kv_cache(paged=True)
    eager_b_key, eager_b_value = model.allocate_kv_cache(paged=True)
    for key_cache, value_cache, table in (
        (trace_key, trace_value, page_table),
        (eager_a_key, eager_a_value, page_table_a_ref),
        (eager_b_key, eager_b_value, page_table_b_ref),
    ):
        prefill_output = model.prefill_forward(
            _tp_hidden(prefill_hidden, mesh_device),
            key_cache,
            value_cache,
            page_table=table,
        )
        ttnn.deallocate(prefill_output, True)

    eager_a_64 = model.decode_forward(
        stable_decode,
        eager_a_key,
        eager_a_value,
        current_pos=64,
        page_table=page_table_a_ref,
    )
    eager_b_64 = model.decode_forward(
        stable_decode,
        eager_b_key,
        eager_b_value,
        current_pos=64,
        page_table=page_table_b_ref,
    )
    eager_b_65 = model.decode_forward(
        stable_decode,
        eager_b_key,
        eager_b_value,
        current_pos=65,
        page_table=page_table_b_ref,
    )

    position_buffers = model.allocate_decode_position_buffers(64)
    warm = model.decode_forward(
        stable_decode,
        trace_key,
        trace_value,
        current_pos=64,
        page_table=page_table,
        position_buffers=position_buffers,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm, True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.decode_forward(
        stable_decode,
        trace_key,
        trace_value,
        current_pos=64,
        page_table=page_table,
        position_buffers=position_buffers,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        original_table_pcc = _assert_pcc(
            _compose_hidden(eager_a_64, mesh_device),
            _compose_hidden(trace_output, mesh_device),
            0.99,
            "paged trace under original page table",
        )

        _copy_replicated_page_table(table_b, page_table, mesh_device)
        model.prepare_decode_position_buffers(position_buffers, 64)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        refreshed_table_pcc = _assert_pcc(
            _compose_hidden(eager_b_64, mesh_device),
            _compose_hidden(trace_output, mesh_device),
            0.99,
            "paged trace after page-table refresh",
        )
        logical_trace_key = _unpage_cache(_compose_cache(trace_key, mesh_device), table_b)
        logical_eager_key = _unpage_cache(_compose_cache(eager_b_key, mesh_device), table_b)
        logical_trace_value = _unpage_cache(_compose_cache(trace_value, mesh_device), table_b)
        logical_eager_value = _unpage_cache(_compose_cache(eager_b_value, mesh_device), table_b)
        refreshed_key_pcc = _assert_pcc(
            logical_eager_key[:, :, 64:65, :],
            logical_trace_key[:, :, 64:65, :],
            0.99,
            "paged traced key cache under refreshed page table",
        )
        refreshed_value_pcc = _assert_pcc(
            logical_eager_value[:, :, 64:65, :],
            logical_trace_value[:, :, 64:65, :],
            0.99,
            "paged traced value cache under refreshed page table",
        )

        model.prepare_decode_position_buffers(position_buffers, 65)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        advanced_position_pcc = _assert_pcc(
            _compose_hidden(eager_b_65, mesh_device),
            _compose_hidden(trace_output, mesh_device),
            0.99,
            "paged trace after advancing current position",
        )
        logical_trace_key = _unpage_cache(_compose_cache(trace_key, mesh_device), table_b)
        logical_eager_key = _unpage_cache(_compose_cache(eager_b_key, mesh_device), table_b)
        logical_trace_value = _unpage_cache(_compose_cache(trace_value, mesh_device), table_b)
        logical_eager_value = _unpage_cache(_compose_cache(eager_b_value, mesh_device), table_b)
        advanced_key_pcc = _assert_pcc(
            logical_eager_key[:, :, 65:66, :],
            logical_trace_key[:, :, 65:66, :],
            0.99,
            "paged traced key cache at advanced position",
        )
        advanced_value_pcc = _assert_pcc(
            logical_eager_value[:, :, 65:66, :],
            logical_trace_value[:, :, 65:66, :],
            0.99,
            "paged traced value cache at advanced position",
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    _write_result(
        "paged_trace_refresh.json",
        {
            "activation_kind": "prompt-derived HF layer-32 input",
            "activation_metadata": activation_metadata,
            "precision_policy": "all_bfp4_lofi",
            "prefill_sequence_length": 64,
            "page_block_size": PAGE_BLOCK_SIZE,
            "original_decode_position": 64,
            "advanced_decode_position": 65,
            "page_table_change": "reverse only the previously-unused second block across 32 users",
            "original_table_output_pcc": original_table_pcc,
            "refreshed_table_output_pcc": refreshed_table_pcc,
            "refreshed_key_cache_pcc": refreshed_key_pcc,
            "refreshed_value_cache_pcc": refreshed_value_pcc,
            "advanced_position_output_pcc": advanced_position_pcc,
            "advanced_key_cache_pcc": advanced_key_pcc,
            "advanced_value_cache_pcc": advanced_value_pcc,
        },
    )


@pytest.mark.skipif(
    os.getenv("QWEN3_32B_MULTICHIP_RUN_TOPOLOGY") != "1",
    reason="manual compiler-provenance topology benchmark",
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.timeout(1800)
def test_multichip_compiler_provenance_topology(mesh_device):
    """Measure the compiler's replicated-boundary/two-all-reduce baseline."""

    config = _config()
    state = _real_state()
    selected = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
        precision_policy="all_bfp4_lofi",
        decode_target_cores=20,
        decode_down_target_cores=20,
    )
    provenance = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
        precision_policy="all_bfp4_lofi",
        decode_target_cores=20,
        decode_down_target_cores=20,
        use_persistent_decode_collectives=False,
        residual_contract="replicated_provenance",
    )
    decode_hidden, activation_metadata = _recorded_hidden(0, 1)
    selected_hidden = _tp_hidden(decode_hidden, mesh_device)
    provenance_hidden = _replicated_hidden(decode_hidden, mesh_device)
    selected_key, selected_value = selected.allocate_kv_cache()
    provenance_key, provenance_value = provenance.allocate_kv_cache()
    selected_position = selected.allocate_decode_position_buffers(0)
    provenance_position = provenance.allocate_decode_position_buffers(0)

    def capture_trace(model, hidden, key_cache, value_cache, position_buffers):
        warm = model.decode_forward(
            hidden,
            key_cache,
            value_cache,
            current_pos=0,
            position_buffers=position_buffers,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(warm, True)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output = model.decode_forward(
            hidden,
            key_cache,
            value_cache,
            current_pos=0,
            position_buffers=position_buffers,
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        return trace_id, output

    selected_trace, selected_output = capture_trace(
        selected, selected_hidden, selected_key, selected_value, selected_position
    )
    replays = int(os.getenv("QWEN3_32B_MULTICHIP_DECODE_REPLAYS", "100"))
    trials = int(os.getenv("QWEN3_32B_MULTICHIP_DECODE_TRIALS", "7"))

    def time_trace(trace_id):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        samples = []
        for _ in range(trials):
            start = time.perf_counter()
            for _ in range(replays):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter() - start) * 1000.0 / replays)
        return samples

    try:
        selected_samples = time_trace(selected_trace)
        selected_host = _compose_hidden(selected_output, mesh_device)
    finally:
        ttnn.release_trace(mesh_device, selected_trace)

    provenance_warm = provenance.decode_forward(
        provenance_hidden,
        provenance_key,
        provenance_value,
        current_pos=0,
        position_buffers=provenance_position,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(provenance_warm, True)
    provenance_samples = []
    provenance_output = None
    for _ in range(trials):
        start = time.perf_counter()
        provenance_output = provenance.decode_forward(
            provenance_hidden,
            provenance_key,
            provenance_value,
            current_pos=0,
            position_buffers=provenance_position,
        )
        ttnn.synchronize_device(mesh_device)
        provenance_samples.append((time.perf_counter() - start) * 1000.0)
    provenance_host = ttnn.to_torch(ttnn.get_device_tensors(provenance_output)[0])
    output_pcc = _assert_pcc(
        selected_host,
        provenance_host,
        0.99,
        "selected sharded topology versus compiler-provenance all-reduce topology",
    )
    key_pcc = _assert_pcc(
        _compose_cache(selected_key, mesh_device)[:, :, :1, :],
        _compose_cache(provenance_key, mesh_device)[:, :, :1, :],
        0.99,
        "topology-family key-cache comparison",
    )
    value_pcc = _assert_pcc(
        _compose_cache(selected_value, mesh_device)[:, :, :1, :],
        _compose_cache(provenance_value, mesh_device)[:, :, :1, :],
        0.99,
        "topology-family value-cache comparison",
    )

    selected_median = statistics.median(selected_samples)
    provenance_median = statistics.median(provenance_samples)
    _write_result(
        "topology_family_benchmark.json",
        {
            "activation_kind": "prompt-derived HF layer-32 input",
            "activation_metadata": activation_metadata,
            "precision_policy": "all_bfp4_lofi",
            "decode_position": 0,
            "replays_per_trial": replays,
            "selected": {
                "family": "sharded layer boundary; 2 all-gathers + 2 reduce-scatters",
                "measurement": "warmed trace replay",
                "samples_ms": selected_samples,
                "median_ms": selected_median,
                "mesh_plan": selected.mesh_plan_summary(),
            },
            "compiler_provenance": {
                "family": "replicated layer boundary; 2 Ring all-reduces",
                "measurement": "warmed eager; trace capture/replay rejected after >4 minute stall and board reset",
                "samples_ms": provenance_samples,
                "median_ms": provenance_median,
                "mesh_plan": provenance.mesh_plan_summary(),
            },
            "selected_speedup": provenance_median / selected_median,
            "provenance_trace_attempt": {
                "status": "device_stall",
                "observation": "no replay completion for more than four minutes; terminated process left chip 0 requiring reset",
                "recovery": "tt-smi -r all followed by healthy 4-board tt-smi discovery",
            },
            "output_pcc": output_pcc,
            "key_cache_pcc": key_pcc,
            "value_cache_pcc": value_pcc,
        },
    )


@pytest.mark.skipif(os.getenv("QWEN3_32B_MULTICHIP_RUN_PERF") != "1", reason="manual multichip performance gate")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.timeout(1800)
def test_multichip_warmed_prefill_and_traced_decode(mesh_device):
    baseline_path_text = os.getenv(BASELINE_PATH_ENV)
    if not baseline_path_text or not Path(baseline_path_text).is_file():
        pytest.skip(f"Run the baseline capture first with {BASELINE_PATH_ENV} set")
    baseline = torch.load(baseline_path_text, map_location="cpu", weights_only=True)
    config = _config()
    state = _real_state()
    ccl_dtype_name = os.getenv("QWEN3_32B_MULTICHIP_CCL_DTYPE", "bf16")
    precision_policy = os.getenv("QWEN3_32B_MULTICHIP_PRECISION_POLICY", "all_bfp4_lofi")
    ccl_dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[ccl_dtype_name]
    use_persistent_collectives = os.getenv("QWEN3_32B_MULTICHIP_PERSISTENT_CCL", "1") == "1"
    use_fused_reduce_scatter = os.getenv("QWEN3_32B_MULTICHIP_FUSED_RS", "0") == "1"
    model = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
        precision_policy=precision_policy,
        decode_target_cores=int(os.getenv("QWEN3_32B_MULTICHIP_DECODE_CORES", "20")),
        decode_down_target_cores=int(os.getenv("QWEN3_32B_MULTICHIP_DOWN_CORES", "20")),
        decode_qkv_target_cores=_optional_env_int("QWEN3_32B_MULTICHIP_QKV_CORES"),
        decode_o_target_cores=_optional_env_int("QWEN3_32B_MULTICHIP_O_CORES"),
        decode_gate_target_cores=_optional_env_int("QWEN3_32B_MULTICHIP_GATE_CORES"),
        decode_packed_target_cores=_optional_env_int("QWEN3_32B_MULTICHIP_PACKED_CORES"),
        decode_qkv_in0_block_w_limit=_optional_env_int("QWEN3_32B_MULTICHIP_QKV_IN0"),
        decode_o_in0_block_w_limit=_optional_env_int("QWEN3_32B_MULTICHIP_O_IN0"),
        decode_gate_in0_block_w_limit=_optional_env_int("QWEN3_32B_MULTICHIP_GATE_IN0"),
        decode_packed_in0_block_w_limit=(
            _optional_env_int("QWEN3_32B_MULTICHIP_PACKED_IN0")
            or _optional_env_int("QWEN3_32B_MULTICHIP_GATE_IN0")
        ),
        decode_down_in0_block_w_limit=_optional_env_int("QWEN3_32B_MULTICHIP_DOWN_IN0"),
        ccl_payload_dtype=ccl_dtype,
        use_persistent_decode_collectives=use_persistent_collectives,
        use_fused_decode_reduce_scatter=use_fused_reduce_scatter,
        use_packed_mlp=os.getenv("QWEN3_32B_MULTICHIP_PACKED_MLP", "1") == "1",
        packed_mlp_split_layout=os.getenv("QWEN3_32B_MULTICHIP_PACKED_SPLIT", "dram"),
        keep_decode_output_sharded=os.getenv("QWEN3_32B_MULTICHIP_KEEP_OUTPUT_SHARDED", "1") == "1",
        use_distributed_decode_norm=os.getenv("QWEN3_32B_MULTICHIP_DISTRIBUTED_NORM", "0") == "1",
        use_fused_decode_all_gather_matmul=os.getenv("QWEN3_32B_MULTICHIP_FUSED_AGMM", "0") == "1",
        decode_matmul_mode=os.getenv("QWEN3_32B_MULTICHIP_DECODE_MATMUL_MODE", "dram_sharded"),
        prefill_matmul_input_l1=os.getenv("QWEN3_32B_MULTICHIP_PREFILL_INPUT_L1", "0") == "1",
    )
    prefill_hidden, activation_metadata = _recorded_hidden(0, 17)
    decode_hidden, _ = _recorded_hidden(17, 1)
    key_cache, value_cache = model.allocate_kv_cache()
    stable_prefill = _tp_hidden(prefill_hidden, mesh_device)

    warm_prefill = model.prefill_forward(stable_prefill, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm_prefill, True)
    prefill_samples = []
    for _ in range(int(os.getenv("QWEN3_32B_MULTICHIP_PREFILL_TRIALS", "7"))):
        start = time.perf_counter()
        output = model.prefill_forward(stable_prefill, key_cache, value_cache)
        ttnn.synchronize_device(mesh_device)
        prefill_samples.append((time.perf_counter() - start) * 1000.0)
        ttnn.deallocate(output, True)

    stable_decode = _tp_hidden(decode_hidden, mesh_device)
    position_buffers = model.allocate_decode_position_buffers(17)
    warm_decode = model.decode_forward(
        stable_decode,
        key_cache,
        value_cache,
        current_pos=17,
        position_buffers=position_buffers,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm_decode, True)
    model.prepare_decode_position_buffers(position_buffers, 18)
    model.prepare_decode_position_buffers(position_buffers, 17)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.decode_forward(
        stable_decode,
        key_cache,
        value_cache,
        current_pos=17,
        position_buffers=position_buffers,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    trace_samples = []
    replays = int(os.getenv("QWEN3_32B_MULTICHIP_DECODE_REPLAYS", "100"))
    trials = int(os.getenv("QWEN3_32B_MULTICHIP_DECODE_TRIALS", "7"))
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        for _ in range(trials):
            start = time.perf_counter()
            for _ in range(replays):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            trace_samples.append((time.perf_counter() - start) * 1000.0 / replays)
        actual_decode = _compose_hidden(trace_output, mesh_device)
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    prefill_pcc = _assert_pcc(
        baseline["prefill"],
        _compose_hidden(model.prefill_forward(stable_prefill, key_cache, value_cache), mesh_device),
        0.99,
        "timed multichip prefill versus optimized baseline",
    )
    decode_pcc = _assert_pcc(
        baseline["decode"],
        actual_decode,
        0.99,
        "timed multichip trace versus optimized baseline",
    )
    payload = {
        "activation_kind": "prompt-derived HF layer-32 input",
        "activation_metadata": activation_metadata,
        "batch": EMITTED_BATCH,
        "sequence_length": 17,
        "decode_position": 17,
        "prefill_pcc": prefill_pcc,
        "decode_pcc": decode_pcc,
        "prefill_samples_ms": prefill_samples,
        "prefill_median_ms": statistics.median(prefill_samples),
        "decode_trace_samples_ms": trace_samples,
        "decode_trace_median_ms": statistics.median(trace_samples),
        "decode_replays_per_trial": replays,
        "mesh_plan": model.mesh_plan_summary(),
    }
    result_name = os.getenv(
        "QWEN3_32B_MULTICHIP_RESULT_NAME",
        f"candidate_{ccl_dtype_name}_{model.decode_target_cores}c_down{model.decode_down_target_cores}c"
        f"{'_persistent' if use_persistent_collectives else ''}"
        f"{'_fused_rs' if use_fused_reduce_scatter else ''}.json",
    )
    output_path = _write_result(result_name, payload)
    print(
        f"MULTICHIP_PERF prefill_ms={payload['prefill_median_ms']:.6f} "
        f"traced_decode_ms={payload['decode_trace_median_ms']:.6f} result={output_path}"
    )


@pytest.mark.skipif(os.getenv("QWEN3_32B_MULTICHIP_RUN_PROFILE") != "1", reason="manual Tracy profile")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.timeout(1800)
def test_profile_selected_multichip_decoder(mesh_device):
    from tracy import signpost

    config = _config()
    state = _real_state()
    model = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
        precision_policy=os.getenv("QWEN3_32B_MULTICHIP_PRECISION_POLICY", "all_bfp4_lofi"),
        decode_target_cores=int(os.getenv("QWEN3_32B_MULTICHIP_DECODE_CORES", "20")),
        decode_down_target_cores=int(os.getenv("QWEN3_32B_MULTICHIP_DOWN_CORES", "20")),
        decode_qkv_target_cores=_optional_env_int("QWEN3_32B_MULTICHIP_QKV_CORES"),
        decode_o_target_cores=_optional_env_int("QWEN3_32B_MULTICHIP_O_CORES"),
        decode_gate_target_cores=_optional_env_int("QWEN3_32B_MULTICHIP_GATE_CORES"),
        decode_packed_target_cores=_optional_env_int("QWEN3_32B_MULTICHIP_PACKED_CORES"),
        decode_qkv_in0_block_w_limit=_optional_env_int("QWEN3_32B_MULTICHIP_QKV_IN0"),
        decode_o_in0_block_w_limit=_optional_env_int("QWEN3_32B_MULTICHIP_O_IN0"),
        decode_gate_in0_block_w_limit=_optional_env_int("QWEN3_32B_MULTICHIP_GATE_IN0"),
        decode_packed_in0_block_w_limit=(
            _optional_env_int("QWEN3_32B_MULTICHIP_PACKED_IN0")
            or _optional_env_int("QWEN3_32B_MULTICHIP_GATE_IN0")
        ),
        decode_down_in0_block_w_limit=_optional_env_int("QWEN3_32B_MULTICHIP_DOWN_IN0"),
        ccl_payload_dtype={
            "bf16": ttnn.bfloat16,
            "bfp8": ttnn.bfloat8_b,
        }[os.getenv("QWEN3_32B_MULTICHIP_CCL_DTYPE", "bf16")],
        use_persistent_decode_collectives=os.getenv("QWEN3_32B_MULTICHIP_PERSISTENT_CCL", "1") == "1",
        use_fused_decode_reduce_scatter=os.getenv("QWEN3_32B_MULTICHIP_FUSED_RS", "0") == "1",
        use_packed_mlp=os.getenv("QWEN3_32B_MULTICHIP_PACKED_MLP", "1") == "1",
        packed_mlp_split_layout=os.getenv("QWEN3_32B_MULTICHIP_PACKED_SPLIT", "dram"),
        keep_decode_output_sharded=os.getenv("QWEN3_32B_MULTICHIP_KEEP_OUTPUT_SHARDED", "1") == "1",
        use_distributed_decode_norm=os.getenv("QWEN3_32B_MULTICHIP_DISTRIBUTED_NORM", "0") == "1",
        use_fused_decode_all_gather_matmul=os.getenv("QWEN3_32B_MULTICHIP_FUSED_AGMM", "0") == "1",
        decode_matmul_mode=os.getenv("QWEN3_32B_MULTICHIP_DECODE_MATMUL_MODE", "dram_sharded"),
        prefill_matmul_input_l1=os.getenv("QWEN3_32B_MULTICHIP_PREFILL_INPUT_L1", "0") == "1",
    )
    prefill_hidden, activation_metadata = _recorded_hidden(0, 17)
    decode_hidden, _ = _recorded_hidden(17, 1)
    key_cache, value_cache = model.allocate_kv_cache()
    stable_prefill = _tp_hidden(prefill_hidden, mesh_device)
    warm = model.prefill_forward(stable_prefill, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm, True)
    signpost(header="MULTICHIP_PREFILL")
    start = time.perf_counter()
    profile_prefill = model.prefill_forward(stable_prefill, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    profiled_prefill_wall_ms = (time.perf_counter() - start) * 1000.0
    signpost(header="MULTICHIP_PREFILL_END")

    stable_decode = _tp_hidden(decode_hidden, mesh_device)
    position_buffers = model.allocate_decode_position_buffers(17)
    warm = model.decode_forward(
        stable_decode,
        key_cache,
        value_cache,
        current_pos=17,
        position_buffers=position_buffers,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm, True)
    model.prepare_decode_position_buffers(position_buffers, 18)
    model.prepare_decode_position_buffers(position_buffers, 17)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.decode_forward(
        stable_decode,
        key_cache,
        value_cache,
        current_pos=17,
        position_buffers=position_buffers,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        signpost(header="MULTICHIP_DECODE")
        start = time.perf_counter()
        for _ in range(3):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        profiled_wall_ms_per_replay = (time.perf_counter() - start) * 1000.0 / 3
        signpost(header="MULTICHIP_DECODE_END")
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    _write_result(
        "profile_run.json",
        {
            "activation_kind": "prompt-derived HF layer-32 input",
            "activation_metadata": activation_metadata,
            "batch": EMITTED_BATCH,
            "sequence_length": 17,
            "decode_position": 17,
            "trace_replays": 3,
            "profiled_prefill_wall_ms": profiled_prefill_wall_ms,
            "profiled_wall_ms_per_replay": profiled_wall_ms_per_replay,
            "mesh_plan": model.mesh_plan_summary(),
        },
    )
    for tensor in (
        profile_prefill,
        trace_output,
        stable_prefill,
        stable_decode,
        key_cache,
        value_cache,
    ):
        ttnn.deallocate(tensor)


@pytest.mark.skipif(
    os.getenv("QWEN3_32B_MULTICHIP_RUN_FUSED_PROBES") != "1",
    reason="manual fused collective compatibility probes",
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.timeout(600)
def test_multichip_fused_collective_candidates(mesh_device):
    config = _config()
    state = _real_state()
    model = MultichipDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=mesh_device,
        max_cache_len=128,
        precision_policy="all_bfp4_lofi",
        decode_target_cores=20,
        decode_down_target_cores=20,
        use_persistent_decode_collectives=True,
    )
    prefill_hidden, _ = _recorded_hidden(0, 17)
    decode_hidden, _ = _recorded_hidden(17, 1)
    key_cache, value_cache = model.allocate_kv_cache()
    model.prefill_forward(_tp_hidden(prefill_hidden, mesh_device), key_cache, value_cache)

    captured = {}
    original_matmul = ttnn.matmul
    original_reduce_scatter = model._reduce_scatter_hidden
    reduce_scatter_calls = 0

    def capture_o_input(*args, **kwargs):
        if args[1] is model.output_weight:
            captured["o_input"] = args[0]
        return original_matmul(*args, **kwargs)

    def capture_attention_shard(tensor, *, memory_config, decode):
        nonlocal reduce_scatter_calls
        result = original_reduce_scatter(tensor, memory_config=memory_config, decode=decode)
        if decode and reduce_scatter_calls == 0:
            captured["attention_shard"] = _compose_hidden(result, mesh_device)
        reduce_scatter_calls += 1
        return result

    ttnn.matmul = capture_o_input
    model._reduce_scatter_hidden = capture_attention_shard
    try:
        model.decode_forward(
            _tp_hidden(decode_hidden, mesh_device),
            key_cache,
            value_cache,
            current_pos=17,
        )
    finally:
        ttnn.matmul = original_matmul
        model._reduce_scatter_hidden = original_reduce_scatter
    ttnn.synchronize_device(mesh_device)
    assert "o_input" in captured and "attention_shard" in captured

    probe_results = {"matmul_reduce_scatter": [], "all_gather_matmul": []}
    for offset in ((0, 6), (0, 4)):
        try:
            _, fused_rs = ttnn.experimental.matmul_reduce_scatter_async(
                captured["o_input"],
                model.output_weight,
                persistent_intermediate_buffer=model._decode_rs_persistent_buffers[0][0],
                persistent_output_buffer=model._decode_rs_persistent_buffers[0][1],
                dim=3,
                multi_device_global_semaphore=model.tt_ccl.get_and_cycle_rs_semaphore_handles(),
                reduce_scatter_core_grid_offset=offset,
                barrier_semaphore=model.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=model.num_links,
                memory_config_rs=model.local_residual_memory_config,
                intermediate_memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                topology=model.topology,
                memory_config_mm=model.o_partial_memory_config,
                dtype=ttnn.bfloat16,
                program_config=model.o_decode_program_config,
                compute_kernel_config=model.attention_compute_config,
            )
            ttnn.synchronize_device(mesh_device)
            pcc = _assert_pcc(
                captured["attention_shard"],
                _compose_hidden(fused_rs, mesh_device),
                0.99,
                f"fused O matmul+reduce-scatter offset {offset}",
            )
            probe_results["matmul_reduce_scatter"].append(
                {"offset": list(offset), "status": "pass", "pcc": pcc}
            )
            break
        except RuntimeError as error:
            probe_results["matmul_reduce_scatter"].append(
                {"offset": list(offset), "status": "runtime_error", "error": str(error).splitlines()[0]}
            )

    o_weight = state[f"model.layers.{REPRESENTATIVE_LAYER}.self_attn.o_proj.weight"].to(torch.bfloat16)
    if not any(result["status"] == "pass" for result in probe_results["matmul_reduce_scatter"]):
        fused_rs_weight = ttnn.from_torch(
            o_weight.T.contiguous(),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-2),
        )
        fused_rs_input = ttnn.to_memory_config(captured["o_input"], ttnn.DRAM_MEMORY_CONFIG)
        fused_rs_intermediate = ttnn.from_torch(
            torch.zeros((1, 1, EMITTED_BATCH, config.hidden_size), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        fused_rs_output = ttnn.from_torch(
            torch.zeros((1, 1, EMITTED_BATCH, config.hidden_size // TP_DEGREE), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        fused_rs_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 6),
            in0_block_w=model.local_attention_width // 32 // 8,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=model.hidden_size // 32 // 8,
            out_block_w=model.hidden_size // 32 // 8 // 2,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        try:
            _, adapted_fused_rs = ttnn.experimental.matmul_reduce_scatter_async(
                fused_rs_input,
                fused_rs_weight,
                persistent_intermediate_buffer=fused_rs_intermediate,
                persistent_output_buffer=fused_rs_output,
                dim=3,
                multi_device_global_semaphore=model.tt_ccl.get_and_cycle_rs_semaphore_handles(),
                reduce_scatter_core_grid_offset=(0, 6),
                barrier_semaphore=model.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=model.num_links,
                memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                intermediate_memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                topology=model.topology,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                program_config=fused_rs_program_config,
                compute_kernel_config=model.attention_compute_config,
            )
            ttnn.synchronize_device(mesh_device)
            pcc = _assert_pcc(
                captured["attention_shard"],
                _compose_hidden(adapted_fused_rs, mesh_device),
                0.99,
                "adapted interleaved 2-D fused O matmul+reduce-scatter",
            )
            probe_results["matmul_reduce_scatter"].append(
                {"variant": "interleaved_2d_multicast", "status": "pass", "pcc": pcc}
            )
        except RuntimeError as error:
            probe_results["matmul_reduce_scatter"].append(
                {
                    "variant": "interleaved_2d_multicast",
                    "status": "runtime_error",
                    "error": str(error).splitlines()[0],
                }
            )

    fused_o_weight = ttnn.from_torch(
        o_weight.T.contiguous().reshape(1, 1, model.attention_width, model.hidden_size),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )
    fused_ag_memory_config = ttnn.create_sharded_memory_config(
        shape=(32, model.local_attention_width // 8),
        core_grid=ttnn.CoreGrid(x=8, y=1),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    fused_ag_input = ttnn.to_memory_config(captured["o_input"], fused_ag_memory_config)
    fused_ag_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 1),
        in0_block_w=model.attention_width // 32 // 8,
        out_subblock_h=1,
        out_subblock_w=5,
        per_core_M=1,
        per_core_N=model.local_hidden_size // 32 // 8,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    for offset in ((0, 4), (0, 6)):
        try:
            _, fused_ag_output = ttnn.experimental.all_gather_matmul_async(
                fused_ag_input,
                fused_o_weight,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=model.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                all_gather_core_grid_offset=offset,
                barrier_semaphore=model.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=1,
                memory_config_ag=fused_ag_memory_config,
                topology=model.topology,
                memory_config_mm=model.local_residual_memory_config,
                dtype=ttnn.bfloat16,
                program_config=fused_ag_program_config,
                compute_kernel_config=model.attention_compute_config,
            )
            ttnn.synchronize_device(mesh_device)
            pcc = _assert_pcc(
                captured["attention_shard"],
                _compose_hidden(fused_ag_output, mesh_device),
                0.99,
                f"fused O all-gather+matmul offset {offset}",
            )
            probe_results["all_gather_matmul"].append(
                {"offset": list(offset), "status": "pass", "pcc": pcc}
            )
            break
        except RuntimeError as error:
            probe_results["all_gather_matmul"].append(
                {"offset": list(offset), "status": "runtime_error", "error": str(error).splitlines()[0]}
            )

    if not any(result["status"] == "pass" for result in probe_results["all_gather_matmul"]):
        adapted_ag_memory_config = ttnn.create_sharded_memory_config(
            shape=(32, model.attention_width // 8),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        adapted_ag_output_memory_config = ttnn.create_sharded_memory_config(
            shape=(32, model.local_hidden_size // 8),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        adapted_ag_input = ttnn.to_memory_config(captured["o_input"], adapted_ag_memory_config)
        try:
            _, adapted_ag_output = ttnn.experimental.all_gather_matmul_async(
                adapted_ag_input,
                fused_o_weight,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=model.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                all_gather_core_grid_offset=(0, 4),
                barrier_semaphore=model.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=1,
                memory_config_ag=adapted_ag_memory_config,
                topology=model.topology,
                memory_config_mm=adapted_ag_output_memory_config,
                dtype=ttnn.bfloat16,
                program_config=fused_ag_program_config,
                compute_kernel_config=model.attention_compute_config,
            )
            ttnn.synchronize_device(mesh_device)
            passed, pcc = comp_pcc(
                captured["attention_shard"].float(),
                _compose_hidden(adapted_ag_output, mesh_device).float(),
                pcc=0.99,
            )
            print(f"adapted fused O all-gather+matmul: {pcc}")
            probe_results["all_gather_matmul"].append(
                {
                    "variant": "full_gather_width_shard",
                    "status": "pass" if passed else "wrong_result",
                    "pcc": pcc,
                }
            )
        except RuntimeError as error:
            probe_results["all_gather_matmul"].append(
                {
                    "variant": "full_gather_width_shard",
                    "status": "runtime_error",
                    "error": str(error).splitlines()[0],
                }
            )

    _write_result("fused_collective_probes.json", probe_results)
    assert probe_results["matmul_reduce_scatter"]
    assert probe_results["all_gather_matmul"]


@pytest.mark.skipif(
    os.getenv("QWEN3_32B_MULTICHIP_RUN_CAPACITY") != "1",
    reason="manual full-stack DRAM-capacity probe",
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.timeout(600)
def test_multichip_full_stack_capacity(mesh_device):
    """Reserve the exact per-device static bytes and worst prefill live set.

    Empty tensors avoid host RAM and kernel execution while retaining the real
    BFP4 weight, BFP8 cache, BF16 RoPE/CCL, and activation tile accounting.
    The live set is the maximum retained by the ownership-aware chunked-QKV
    implementation: local residual, norm output, accumulated QKV chunks, and
    the QKV concat output.  The preceding gather is released before projection.
    """

    sequence_length = int(os.environ["QWEN3_32B_MULTICHIP_CAPACITY_SEQUENCE"])
    physical_cache_length = PAGE_BLOCK_SIZE * math.ceil(sequence_length / PAGE_BLOCK_SIZE)
    expected = os.getenv("QWEN3_32B_MULTICHIP_CAPACITY_EXPECT", "pass")
    if expected not in ("pass", "fail"):
        raise ValueError("QWEN3_32B_MULTICHIP_CAPACITY_EXPECT must be pass or fail")
    allocations = []
    snapshots = {"mesh_open": _dram_snapshot(mesh_device)}
    failed_stage = None
    error_text = None

    def reserve(shape, dtype, *, layout=ttnn.TILE_LAYOUT):
        tensor = ttnn.empty(
            shape,
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        allocations.append(tensor)
        return tensor

    try:
        failed_stage = "duplicated_prefill_decode_weights"
        # Local TP shapes for all 64 layers.  Each role is reserved twice
        # because the final decoder owns interleaved prefill and DRAM-sharded
        # decode layouts simultaneously.
        local_weight_shapes = (
            (64, 1, 5120, 2560),
            (64, 1, 2048, 5120),
            (64, 1, 5120, 6400),
            (64, 1, 5120, 6400),
            (64, 1, 6400, 5120),
        )
        for _ in range(2):
            for shape in local_weight_shapes:
                reserve(shape, ttnn.bfloat4_b)

        failed_stage = "all_layer_kv_cache"
        # Split the leading layer*batch dimension to keep every logical tensor
        # comfortably below the signed-32-bit element-count boundary.
        for _ in range(2):
            for _ in range(8):
                reserve((256, 2, physical_cache_length, 128), ttnn.bfloat8_b)

        failed_stage = "rope_norm_and_persistent_ccl"
        # Each layer owns cos+sin in both TILE form (prefill) and ROW_MAJOR
        # form (decode).  Reserve the two layout sets separately so capacity
        # matches MultichipDecoder.from_state_dict rather than counting only
        # half of the retained RoPE tables.
        reserve((128, 1, sequence_length, 128), ttnn.bfloat16)
        reserve((128, 1, sequence_length, 128), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        reserve((1, 1, 32, 1_474_560), ttnn.bfloat16)
        reserve((1, 1, 32, 20_992), ttnn.bfloat16)
        snapshots["full_stack_static"] = _dram_snapshot(mesh_device)

        failed_stage = "prefill_live_set"
        reserve((1, 32, sequence_length, 1280), ttnn.bfloat16)
        reserve((1, 32, sequence_length, 5120), ttnn.bfloat16)
        reserve((1, 32, sequence_length, 2560), ttnn.bfloat16)
        reserve((1, 32, sequence_length, 2560), ttnn.bfloat16)
        snapshots["prefill_peak"] = _dram_snapshot(mesh_device)
        failed_stage = None
    except RuntimeError as error:
        error_text = str(error).split("backtrace:", maxsplit=1)[0].strip()
        snapshots["at_failure"] = _dram_snapshot(mesh_device)
    finally:
        payload = {
            "sequence_length": sequence_length,
            "physical_cache_length": physical_cache_length,
            "page_block_size": PAGE_BLOCK_SIZE,
            "batch": EMITTED_BATCH,
            "result": "pass" if failed_stage is None else "expected_out_of_memory",
            "failed_stage": failed_stage,
            "error": error_text,
            "snapshots": snapshots,
            "accounting": {
                "duplicated_bfp4_weight_bytes_per_device": 8_776_581_120,
                "bfp8_kv_bytes_per_device_per_token": 1_114_112,
                "bf16_rope_bytes_per_device_per_token": 65_536,
                "persistent_ccl_bytes_per_device": 94_371_840,
                "norm_weight_bytes_per_device": 1_343_488,
                "prefill_peak_live_bytes_per_device_per_token": 737_280,
                "trace_region_size_parameter_bytes_per_bank": 200_000_000,
                "trace_reserved_bytes_per_device": 1_600_000_000,
            },
        }
        _write_result(f"capacity_seq{sequence_length}.json", payload)
        for tensor in reversed(allocations):
            ttnn.deallocate(tensor, True)

    if expected == "pass":
        assert failed_stage is None, f"capacity probe failed at {failed_stage}: {error_text}"
    else:
        assert failed_stage is not None, "capacity probe unexpectedly fit"
