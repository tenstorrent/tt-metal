# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Contract, correctness, cache, trace, and performance gates for TP=4 Gemma-4."""

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path

import pytest

import models.autoports.google_gemma_4_26b_a4b_it.tests.test_functional_decoder as functional_tests
import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import HIDDEN_SIZE
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import (
    LOCAL_MLP_INTERMEDIATE_SIZE,
    LOCAL_MOE_INTERMEDIATE_SIZE,
    LOCAL_Q_HEADS,
    PADDED_MLP_INTERMEDIATE_SIZE,
    PADDED_MOE_INTERMEDIATE_SIZE,
    TP_SIZE,
    MultichipDecoder,
)
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_multichip_shape_contract():
    assert TP_SIZE == 4
    assert LOCAL_Q_HEADS == 4
    assert PADDED_MLP_INTERMEDIATE_SIZE == 2176
    assert LOCAL_MLP_INTERMEDIATE_SIZE == 544
    assert PADDED_MOE_INTERMEDIATE_SIZE == 768
    assert LOCAL_MOE_INTERMEDIATE_SIZE == 192
    assert PADDED_MLP_INTERMEDIATE_SIZE % (TP_SIZE * 32) == 0
    assert PADDED_MOE_INTERMEDIATE_SIZE % (TP_SIZE * 32) == 0


def test_multichip_inherits_optimized_baseline_and_has_no_host_hot_path():
    assert issubclass(
        MultichipDecoder,
        __import__(
            "models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder", fromlist=["OptimizedDecoder"]
        ).OptimizedDecoder,
    )
    hot_methods = (
        MultichipDecoder._attention_prefill,
        MultichipDecoder._attention_decode,
        MultichipDecoder._dense_mlp,
        MultichipDecoder._moe_prefill,
        MultichipDecoder._moe_decode,
    )
    forbidden = ("torch.", "ttnn.from_torch", "ttnn.to_torch", ".cpu(", ".numpy(")
    for method in hot_methods:
        source = inspect.getsource(method)
        assert not any(token in source for token in forbidden), (method.__name__, source)


def test_multichip_preserves_active_expert_execution():
    source = inspect.getsource(MultichipDecoder._moe_decode) + inspect.getsource(MultichipDecoder._moe_prefill)
    assert "super()._moe_decode" in source
    assert "super()._moe_prefill" in source
    inherited = inspect.getsource(MultichipDecoder.__mro__[1]._moe_decode_single_user)
    assert "ttnn.sparse_matmul" in inherited
    assert "TOP_K_EXPERTS" in inherited


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_tp4_ring_all_reduce_smoke(mesh_device):
    """Shape-faithful hidden payload smoke with the exact final ring topology."""
    import torch

    if tuple(mesh_device.shape) != (1, 4):
        pytest.skip(f"requires target 1x4 mesh, got {tuple(mesh_device.shape)}")
    shards = torch.cat(
        [torch.full((1, 1, 32, 2816), float(rank + 1), dtype=torch.bfloat16) for rank in range(4)], dim=0
    )
    value = ttnn.from_torch(
        shards,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    reduced = ttnn.all_reduce(value, cluster_axis=1, topology=ttnn.Topology.Ring)
    host_shards = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(reduced.cpu())]
    for shard in host_shards:
        assert torch.equal(shard, torch.full_like(shard, 10.0))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_capture_optimized_single_chip_reference(mesh_device, monkeypatch, layer_idx):
    """Explicit opt-in producer for the checked-in direct-comparison reference."""
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    if os.environ.get("GEMMA4_CAPTURE_SINGLE_CHIP_REFERENCE") != "1":
        pytest.skip("reference capture is an explicit hardware evidence step")
    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    cfg = functional_tests._load_text_config()
    state = functional_tests._load_layer_state(layer_idx)
    layer_type = cfg.layer_types[layer_idx]
    seq_len = 32
    torch.manual_seed(4200 + layer_idx)
    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, torch.arange(seq_len).unsqueeze(0), layer_type=layer_type)
    decode_cos, decode_sin = rotary(decode_hidden, torch.tensor([[seq_len]]), layer_type=layer_type)

    def run(decoder_cls, target_mesh, local):
        decoder = decoder_cls.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=target_mesh)
        if layer_type == "full_attention":
            blocks, heads, block, dim = 2, (1 if local else 2), 128, 512
        else:
            blocks, heads, block, dim = 4, (2 if local else 8), 64, 256
        page_table = functional_tests._as_tt(
            target_mesh,
            torch.arange(blocks, dtype=torch.int32).view(1, blocks),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cache = tuple(
            functional_tests._as_tt(target_mesh, torch.zeros((blocks, heads, block, dim), dtype=torch.bfloat16))
            for _ in range(2)
        )
        prefill = decoder.prefill_forward(
            functional_tests._as_tt(target_mesh, hidden.unsqueeze(1)),
            position_cos=functional_tests._as_tt(target_mesh, cos.unsqueeze(1)),
            position_sin=functional_tests._as_tt(target_mesh, sin.unsqueeze(1)),
            page_table=page_table,
            kv_cache=cache,
        )
        decode = decoder.decode_forward(
            hidden_states=functional_tests._as_tt(target_mesh, decode_hidden.unsqueeze(1)),
            position_cos=functional_tests._as_tt(target_mesh, decode_cos.unsqueeze(1)),
            position_sin=functional_tests._as_tt(target_mesh, decode_sin.unsqueeze(1)),
            current_pos=functional_tests._as_tt(
                target_mesh,
                torch.tensor([seq_len], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            page_table=page_table,
            kv_cache=cache,
        )
        return tuple(ttnn.to_torch(ttnn.get_device_tensors(result.cpu())[0]) for result in (prefill, decode))

    baseline_prefill, baseline_decode = run(OptimizedDecoder, mesh_device, False)
    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"prefill": baseline_prefill, "decode": baseline_decode},
        artifact_dir / f"optimized_reference_layer{layer_idx}.pt",
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_multichip_matches_optimized_single_chip(mesh_device, device_params, monkeypatch, layer_idx):
    """Compare TP output directly with separately captured optimized TTNN output."""
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    cfg = functional_tests._load_text_config()
    state = functional_tests._load_layer_state(layer_idx)
    layer_type = cfg.layer_types[layer_idx]
    seq_len = 32
    torch.manual_seed(4200 + layer_idx)
    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, torch.arange(seq_len).unsqueeze(0), layer_type=layer_type)
    decode_cos, decode_sin = rotary(decode_hidden, torch.tensor([[seq_len]]), layer_type=layer_type)
    decoder = MultichipDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    if layer_type == "full_attention":
        blocks, heads, block, dim = 2, 1, 128, 512
    else:
        blocks, heads, block, dim = 4, 2, 64, 256
    page_table = functional_tests._as_tt(
        mesh_device,
        torch.arange(blocks, dtype=torch.int32).view(1, blocks),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache = tuple(
        functional_tests._as_tt(mesh_device, torch.zeros((blocks, heads, block, dim), dtype=torch.bfloat16))
        for _ in range(2)
    )
    prefill = decoder.prefill_forward(
        functional_tests._as_tt(mesh_device, hidden.unsqueeze(1)),
        position_cos=functional_tests._as_tt(mesh_device, cos.unsqueeze(1)),
        position_sin=functional_tests._as_tt(mesh_device, sin.unsqueeze(1)),
        page_table=page_table,
        kv_cache=cache,
    )
    decode = decoder.decode_forward(
        hidden_states=functional_tests._as_tt(mesh_device, decode_hidden.unsqueeze(1)),
        position_cos=functional_tests._as_tt(mesh_device, decode_cos.unsqueeze(1)),
        position_sin=functional_tests._as_tt(mesh_device, decode_sin.unsqueeze(1)),
        current_pos=functional_tests._as_tt(
            mesh_device,
            torch.tensor([seq_len], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        page_table=page_table,
        kv_cache=cache,
    )
    multichip_prefill, multichip_decode = (
        ttnn.to_torch(ttnn.get_device_tensors(result.cpu())[0]) for result in (prefill, decode)
    )
    reference = torch.load(
        Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
        / f"optimized_reference_layer{layer_idx}.pt",
        weights_only=True,
    )
    assert_with_pcc(reference["prefill"], multichip_prefill, 0.995)
    assert_with_pcc(reference["decode"], multichip_decode, 0.995)


def _run_traced_batch32(decoder_cls, mesh_device, cfg, state, layer_idx, *, local_cache):
    """Run an identical batch-32 TTNN decode regime for baseline and TP4."""
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    batch, current_position = 32, 32
    layer_type = cfg.layer_types[layer_idx]
    torch.manual_seed(4300 + layer_idx)
    decode_hidden = torch.randn(batch, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    positions = torch.full((batch, 1), current_position, dtype=torch.long)
    cos, sin = rotary(decode_hidden, positions, layer_type=layer_type)
    decoder = decoder_cls.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    if layer_type == "full_attention":
        blocks_per_user, heads, block, dim = 2, (1 if local_cache else 2), 128, 512
    else:
        blocks_per_user, heads, block, dim = 4, (2 if local_cache else 8), 64, 256
    page_table = functional_tests._as_tt(
        mesh_device,
        torch.arange(batch * blocks_per_user, dtype=torch.int32).view(batch, blocks_per_user),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache_shape = (batch * blocks_per_user, heads, block, dim)
    kv_cache = tuple(
        functional_tests._as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)) for _ in range(2)
    )
    if layer_type == "full_attention":
        torch.manual_seed(4400 + layer_idx)
        prefix = torch.randn(1, current_position, HIDDEN_SIZE, dtype=torch.bfloat16).expand(batch, -1, -1).clone()
        prefix_positions = torch.arange(current_position).view(1, -1).expand(batch, -1)
        prefix_cos, prefix_sin = rotary(prefix, prefix_positions, layer_type=layer_type)
        for user_id in range(batch):
            decoder.prefill_forward(
                functional_tests._as_tt(mesh_device, prefix[user_id : user_id + 1].unsqueeze(1)),
                position_cos=functional_tests._as_tt(mesh_device, prefix_cos[user_id : user_id + 1].unsqueeze(1)),
                position_sin=functional_tests._as_tt(mesh_device, prefix_sin[user_id : user_id + 1].unsqueeze(1)),
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user_id,
            )
    tt_cos = cos.unsqueeze(0) if layer_type == "sliding_attention" else cos.transpose(0, 1).unsqueeze(0)
    tt_sin = sin.unsqueeze(0) if layer_type == "sliding_attention" else sin.transpose(0, 1).unsqueeze(0)
    decode_args = {
        "hidden_states": functional_tests._as_tt(mesh_device, decode_hidden.transpose(0, 1).unsqueeze(0)),
        "position_cos": functional_tests._as_tt(mesh_device, tt_cos),
        "position_sin": functional_tests._as_tt(mesh_device, tt_sin),
        "current_pos": functional_tests._as_tt(
            mesh_device,
            torch.full((batch,), current_position, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    eager = decoder.decode_forward(**decode_args)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = decoder.decode_forward(**decode_args)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    for _ in range(5):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    first = ttnn.to_torch(ttnn.get_device_tensors(traced.cpu())[0])
    iterations = 30
    started = time.perf_counter()
    for _ in range(iterations):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    elapsed_ms = (time.perf_counter() - started) * 1000 / iterations
    second = ttnn.to_torch(ttnn.get_device_tensors(traced.cpu())[0])
    eager_host = ttnn.to_torch(ttnn.get_device_tensors(eager.cpu())[0])
    ttnn.release_trace(mesh_device, trace_id)
    assert torch.equal(first, second)
    assert torch.equal(eager_host, second)
    return second, elapsed_ms, cache_shape


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_capture_optimized_batch32_reference(mesh_device, device_params, monkeypatch, layer_idx):
    import torch

    if os.environ.get("GEMMA4_CAPTURE_SINGLE_CHIP_REFERENCE") != "1":
        pytest.skip("reference capture is an explicit hardware evidence step")
    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    cfg = functional_tests._load_text_config()
    state = functional_tests._load_layer_state(layer_idx)
    output, latency_ms, cache_shape = _run_traced_batch32(
        OptimizedDecoder, mesh_device, cfg, state, layer_idx, local_cache=False
    )
    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
    torch.save(output, artifact_dir / f"optimized_batch32_layer{layer_idx}.pt")
    (artifact_dir / f"optimized_batch32_layer{layer_idx}.json").write_text(
        json.dumps(
            {
                "layer_idx": layer_idx,
                "batch": 32,
                "trace_replay_ms": latency_ms,
                "cache_shape": cache_shape,
                "repeat_bit_exact": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_multichip_batch32_trace_and_optimized_pcc(mesh_device, device_params, monkeypatch, layer_idx):
    import torch

    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    cfg = functional_tests._load_text_config()
    state = functional_tests._load_layer_state(layer_idx)
    output, latency_ms, cache_shape = _run_traced_batch32(
        MultichipDecoder, mesh_device, cfg, state, layer_idx, local_cache=True
    )
    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
    reference = torch.load(artifact_dir / f"optimized_batch32_layer{layer_idx}.pt", weights_only=True)
    assert_with_pcc(reference, output, 0.995)
    baseline = json.loads((artifact_dir / f"optimized_batch32_layer{layer_idx}.json").read_text())
    (artifact_dir / f"multichip_batch32_layer{layer_idx}.json").write_text(
        json.dumps(
            {
                "layer_idx": layer_idx,
                "layer_type": cfg.layer_types[layer_idx],
                "batch": 32,
                "trace_replay_ms": latency_ms,
                "single_chip_trace_replay_ms": baseline["trace_replay_ms"],
                "speedup": baseline["trace_replay_ms"] / latency_ms,
                "tp_efficiency": baseline["trace_replay_ms"] / latency_ms / TP_SIZE,
                "local_cache_shape": cache_shape,
                "repeat_bit_exact": True,
                "optimized_pcc_threshold": 0.995,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical,decode_pcc",
    [
        pytest.param(0, False, 0.995, id="sliding_attention"),
        pytest.param(5, False, 0.995, id="full_attention"),
    ],
)
def test_multichip_real_weights_prefill_decode(
    mesh_device, device_params, monkeypatch, layer_idx, shared_physical, decode_pcc
):
    """Reuse the established HF oracle with TP-local cache geometry."""

    def local_cache_shape(layer_type, *, shared_physical, token_capacity=None):
        if layer_type == "full_attention":
            block_size, heads, head_dim, default_blocks = 128, 1, 512, 2
        else:
            block_size, heads, head_dim, default_blocks = 64, 2, 256, 4
        blocks = default_blocks if token_capacity is None else (token_capacity + block_size - 1) // block_size
        return blocks, heads, block_size, head_dim

    def replicated_output(_mesh, tensor):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor.cpu())[0])

    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", MultichipDecoder)
    monkeypatch.setattr(functional_tests, "_cache_shape", local_cache_shape)
    monkeypatch.setattr(functional_tests, "_to_torch", replicated_output)
    monkeypatch.setattr(
        functional_tests,
        "ARTIFACT_DIR",
        Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts"),
    )
    functional_tests.test_functional_decoder_real_weights_prefill_decode(
        mesh_device, device_params, layer_idx, shared_physical, decode_pcc
    )


def _install_multichip_functional_harness(monkeypatch, mesh_device):
    """Adapt capacity tests to TP-local cache geometry and first replicated rank."""
    monkeypatch.setattr(functional_tests, "FunctionalDecoder", MultichipDecoder)
    monkeypatch.setattr(functional_tests, "SLIDING_NUM_KV_HEADS", 2)
    monkeypatch.setattr(functional_tests, "FULL_NUM_KV_HEADS", 1)
    monkeypatch.setattr(
        functional_tests,
        "_to_torch",
        lambda _mesh, tensor: ttnn.to_torch(ttnn.get_device_tensors(tensor.cpu())[0]),
    )
    monkeypatch.setattr(
        functional_tests,
        "ARTIFACT_DIR",
        Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts"),
    )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_multichip_advertised_context_traced_decode(mesh_device, device_params, monkeypatch, layer_idx):
    _install_multichip_functional_harness(monkeypatch, mesh_device)
    functional_tests.test_advertised_context_traced_decode(mesh_device, device_params, layer_idx)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_multichip_prefill_capacity(mesh_device, device_params, monkeypatch, layer_idx):
    _install_multichip_functional_harness(monkeypatch, mesh_device)
    functional_tests.test_prefill_capacity_probe(mesh_device, device_params, layer_idx)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_multichip_bounded_modulo_tail_integrity(mesh_device, device_params, monkeypatch):
    _install_multichip_functional_harness(monkeypatch, mesh_device)
    functional_tests.test_bounded_modulo_prefill_tail_cache_integrity(mesh_device, device_params)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,shared_physical", [(0, True), (5, False)], ids=["sliding_attention", "full_attention"]
)
def test_multichip_perf_profile(mesh_device, device_params, monkeypatch, layer_idx, shared_physical):
    _install_multichip_functional_harness(monkeypatch, mesh_device)
    functional_tests.test_functional_decoder_perf_profile(mesh_device, device_params, layer_idx, shared_physical, 32)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_multichip_non_aligned_prefill_and_decode_trace(mesh_device, device_params, monkeypatch, layer_idx):
    """Exercise logical S=33, local cache ownership, and repeat trace replay."""
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    monkeypatch.setenv("GEMMA4_RANGE_DOWNLOAD", "1")
    cfg = functional_tests._load_text_config()
    state = functional_tests._load_layer_state(layer_idx)
    layer_type = cfg.layer_types[layer_idx]
    seq_len = 33
    torch.manual_seed(4100 + layer_idx)
    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    cos, sin = rotary(hidden, torch.arange(seq_len).unsqueeze(0), layer_type=layer_type)
    decode_cos, decode_sin = rotary(decode_hidden, torch.tensor([[seq_len]]), layer_type=layer_type)
    decoder = MultichipDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device)
    if layer_type == "full_attention":
        blocks, heads, block, dim = 2, 1, 128, 512
    else:
        blocks, heads, block, dim = 4, 2, 64, 256
    page_table = functional_tests._as_tt(
        mesh_device,
        torch.arange(blocks, dtype=torch.int32).view(1, blocks),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kv_cache = tuple(
        functional_tests._as_tt(mesh_device, torch.zeros((blocks, heads, block, dim), dtype=torch.bfloat16))
        for _ in range(2)
    )
    prefill = decoder.prefill_forward(
        functional_tests._as_tt(mesh_device, hidden.unsqueeze(1)),
        position_cos=functional_tests._as_tt(mesh_device, cos.unsqueeze(1)),
        position_sin=functional_tests._as_tt(mesh_device, sin.unsqueeze(1)),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    assert tuple(prefill.shape) == (1, 1, seq_len, HIDDEN_SIZE)
    decode_args = {
        "hidden_states": functional_tests._as_tt(mesh_device, decode_hidden.unsqueeze(1)),
        "position_cos": functional_tests._as_tt(mesh_device, decode_cos.unsqueeze(1)),
        "position_sin": functional_tests._as_tt(mesh_device, decode_sin.unsqueeze(1)),
        "current_pos": functional_tests._as_tt(
            mesh_device,
            torch.tensor([seq_len], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "page_table": page_table,
        "kv_cache": kv_cache,
    }
    eager = decoder.decode_forward(**decode_args)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = decoder.decode_forward(**decode_args)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    first = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(traced.cpu())]
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    second = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(traced.cpu())]
    eager_shards = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(eager.cpu())]
    for rank in range(TP_SIZE):
        assert torch.equal(first[rank], second[rank])
        assert torch.equal(first[0], first[rank]), "replicated residual diverged across TP ranks"
        assert torch.equal(eager_shards[rank], first[rank])
    page_shards = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(page_table.cpu())]
    pos_shards = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(decode_args["current_pos"].cpu())]
    for rank in range(1, TP_SIZE):
        assert torch.equal(page_shards[0], page_shards[rank])
        assert torch.equal(pos_shards[0], pos_shards[rank])
    cache_shards = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(kv_cache[0].cpu())]
    assert all(tuple(shard.shape) == (blocks, heads, block, dim) for shard in cache_shards)
    if layer_type == "full_attention":
        assert torch.equal(cache_shards[0], cache_shards[1]), "full KV head 0 pair diverged"
        assert torch.equal(cache_shards[2], cache_shards[3]), "full KV head 1 pair diverged"
        assert not torch.equal(cache_shards[0], cache_shards[2]), "distinct full KV heads collapsed"
    assert decoder.multichip_path_counters["attention_tp"] >= 3
    assert decoder.multichip_path_counters["expert_tp"] >= 3
    assert decoder.multichip_path_counters["all_reduce"] >= 9
    for _ in range(5):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    iterations = 30
    started = time.perf_counter()
    for _ in range(iterations):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    multi_ms = (time.perf_counter() - started) * 1000 / iterations
    single_ms = 1.272 if layer_type == "sliding_attention" else 1.270
    artifact_dir = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / f"trace_{layer_type}_batch1.json").write_text(
        json.dumps(
            {
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "logical_prefill_length": seq_len,
                "decode_batch": 1,
                "warmups": 5,
                "iterations": iterations,
                "single_chip_optimized_baseline_ms": single_ms,
                "single_chip_source": "doc/optimized_decoder/README.md",
                "multichip_trace_replay_ms": multi_ms,
                "speedup": single_ms / multi_ms,
                "tp_efficiency": single_ms / multi_ms / TP_SIZE,
                "repeat_bit_exact": True,
                "replicas_bit_exact": True,
                "page_table_replicated": True,
                "current_position_replicated": True,
                "local_cache_shape": [blocks, heads, block, dim],
                "full_kv_pair_duplication_verified": layer_type == "full_attention",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    ttnn.release_trace(mesh_device, trace_id)
