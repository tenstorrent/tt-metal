# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v4_flash.converter import convert_hf_checkpoint
from models.demos.deepseek_v4_flash.cpu_reference import (
    combine_routed_experts,
    compressor_prefill,
    indexer_topk,
    rms_norm,
    sparse_attention,
    swiglu_expert,
    v4_router,
)
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

ttnn = pytest.importorskip("ttnn")

from models.demos.deepseek_v4_flash.expert_abi import load_packed_expert_weight
from models.demos.deepseek_v4_flash.ttnn_attention_projection import (
    TtAttentionProjection,
    grouped_output_projection_a,
    load_attention_projection_weights,
)
from models.demos.deepseek_v4_flash.ttnn_prefill_attention_block import TtPrefillAttentionBlock, load_attention_sink
from models.demos.deepseek_v4_flash.ttnn_prefill_compressor import TtPrefillCompressor, load_prefill_compressor_weights
from models.demos.deepseek_v4_flash.ttnn_routed_expert import TtRoutedExpertMLP
from models.demos.deepseek_v4_flash.ttnn_router import TtRouter, load_router_weights
from models.demos.deepseek_v4_flash.ttnn_shared_expert import TtSharedExpertMLP
from models.demos.deepseek_v4_flash.ttnn_sparse_attention import TtSparsePrefillAttention


@pytest.fixture(scope="module")
def tiny_tt_preprocessed_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    artifact_root = os.environ.get("DSV4_FLASH_ARTIFACT_DIR")
    if artifact_root:
        base = Path(artifact_root) / "pytest" / f"deepseek_v4_flash_device_smoke_{os.getpid()}"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)
    else:
        base = tmp_path_factory.mktemp("deepseek_v4_flash_device_smoke")

    source = generate_tiny_hf_checkpoint(base / "hf_source", num_hidden_layers=3)
    return convert_hf_checkpoint(source, base / "tt_preprocessed")


@pytest.fixture(scope="module")
def device():
    try:
        num_devices = ttnn.GetNumAvailableDevices()
    except Exception as exc:
        pytest.skip(f"Unable to query TT devices: {exc}")
    if num_devices < 1:
        pytest.skip("No TT devices available")

    device_id = int(os.environ.get("TTNN_DEVICE_ID", "0"))
    device = ttnn.open_device(device_id=device_id, num_command_queues=1)
    try:
        yield device
    finally:
        ttnn.close_device(device)


def test_tiny_hyperconnection_residual_shape_roundtrips_on_device(tiny_tt_preprocessed_checkpoint: Path, device):
    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    hidden_size = manifest["config"]["hidden_size"]
    hc_mult = manifest["config"]["hc_mult"]
    torch_tensor = torch.linspace(-1.0, 1.0, steps=hc_mult * hidden_size, dtype=torch.float32).reshape(
        1, 1, hc_mult, hidden_size
    )
    torch_tensor = torch_tensor.to(torch.bfloat16)

    tt_tensor = ttnn.from_torch(torch_tensor, device=device, dtype=ttnn.bfloat16)
    torch_output = ttnn.to_torch(tt_tensor)

    torch.testing.assert_close(torch_output, torch_tensor)


def test_tiny_shared_expert_w1_projection_matches_torch(tiny_tt_preprocessed_checkpoint: Path, device):
    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    non_expert_path = tiny_tt_preprocessed_checkpoint / manifest["artifacts"]["non_expert_safetensors"][0]
    with safe_open(non_expert_path, framework="pt", device="cpu") as handle:
        w1 = handle.get_tensor("layers.0.ffn.shared_experts.w1.weight").to(torch.bfloat16)

    hidden_size = w1.shape[-1]
    intermediate_size = w1.shape[0]
    torch_input = torch.linspace(-0.5, 0.5, steps=32 * hidden_size, dtype=torch.float32).reshape(1, 1, 32, hidden_size)
    torch_input = torch_input.to(torch.bfloat16)
    torch_weight = w1.T.contiguous().reshape(1, 1, hidden_size, intermediate_size)
    expected = torch.matmul(torch_input.float(), torch_weight.float())

    tt_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_output = ttnn.matmul(tt_input, tt_weight)
    torch_output = ttnn.to_torch(tt_output)

    torch.testing.assert_close(torch_output.float(), expected, rtol=2e-2, atol=2e-2)


def test_tiny_shared_expert_mlp_module_matches_torch(tiny_tt_preprocessed_checkpoint: Path, device):
    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    non_expert_path = tiny_tt_preprocessed_checkpoint / manifest["artifacts"]["non_expert_safetensors"][0]
    with safe_open(non_expert_path, framework="pt", device="cpu") as handle:
        w1 = handle.get_tensor("layers.0.ffn.shared_experts.w1.weight").to(torch.bfloat16)
        w2 = handle.get_tensor("layers.0.ffn.shared_experts.w2.weight").to(torch.bfloat16)
        w3 = handle.get_tensor("layers.0.ffn.shared_experts.w3.weight").to(torch.bfloat16)

    hidden_size = w1.shape[-1]
    torch_input = torch.linspace(-0.25, 0.25, steps=32 * hidden_size, dtype=torch.float32).reshape(
        1, 1, 32, hidden_size
    )
    torch_input = torch_input.to(torch.bfloat16)
    expected = swiglu_expert(
        torch_input.reshape(-1, hidden_size),
        w1,
        w2,
        w3,
        swiglu_limit=float(manifest["config"]["swiglu_limit"]),
    ).view_as(torch_input)

    tt_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    module = TtSharedExpertMLP.from_preprocessed(tiny_tt_preprocessed_checkpoint, device=device)
    torch_output = ttnn.to_torch(module(tt_input))

    torch.testing.assert_close(torch_output.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_tiny_routed_expert_mlp_module_matches_torch(tiny_tt_preprocessed_checkpoint: Path, device):
    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    expert_id = 2
    weights = {
        projection: load_packed_expert_weight(
            tiny_tt_preprocessed_checkpoint, layer=0, expert=expert_id, projection=projection
        ).dequantize(dtype=torch.bfloat16)
        for projection in ("w1", "w2", "w3")
    }

    hidden_size = weights["w1"].shape[-1]
    intermediate_size = weights["w1"].shape[0]
    torch_input = torch.linspace(-0.2, 0.2, steps=32 * hidden_size, dtype=torch.float32).reshape(1, 1, 32, hidden_size)
    torch_input = torch_input.to(torch.bfloat16)
    route_weights = torch.linspace(0.25, 1.0, steps=32, dtype=torch.float32).reshape(1, 32, 1)
    route_indices = torch.full((1, 32, 1), expert_id, dtype=torch.int64)
    expected = combine_routed_experts(
        torch_input.reshape(1, 32, hidden_size),
        route_weights,
        route_indices,
        {expert_id: (weights["w1"], weights["w2"], weights["w3"])},
        swiglu_limit=float(manifest["config"]["swiglu_limit"]),
    ).reshape_as(torch_input)

    tt_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_route_weights = ttnn.from_torch(
        route_weights.reshape(1, 1, 32, 1).expand(1, 1, 32, intermediate_size).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    module = TtRoutedExpertMLP.from_preprocessed(tiny_tt_preprocessed_checkpoint, device=device, expert=expert_id)
    torch_output = ttnn.to_torch(module(tt_input, route_weight=tt_route_weights))

    torch.testing.assert_close(torch_output.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_tiny_router_module_matches_torch_hash_layer(tiny_tt_preprocessed_checkpoint: Path, device):
    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    weights = load_router_weights(tiny_tt_preprocessed_checkpoint, manifest=manifest, layer=0)

    hidden_size = weights.gate_weight.shape[-1]
    seq_len = 32
    torch_input = torch.linspace(-0.15, 0.2, steps=seq_len * hidden_size, dtype=torch.float32).reshape(
        1, 1, seq_len, hidden_size
    )
    torch_input = torch_input.to(torch.bfloat16)
    input_ids = torch.zeros(1, seq_len, dtype=torch.int64)
    input_ids[:, seq_len // 2 :] = 1
    expected_weights, expected_indices = v4_router(
        torch_input[:, 0],
        weights.gate_weight,
        topk=int(manifest["config"]["num_experts_per_tok"]),
        route_scale=float(manifest["config"]["routed_scaling_factor"]),
        scoring_func=str(manifest["config"]["scoring_func"]),
        input_ids=input_ids,
        tid2eid=weights.tid2eid,
    )

    tt_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    module = TtRouter.from_preprocessed(tiny_tt_preprocessed_checkpoint, device=device, layer=0)
    route_weights, route_indices = module(tt_input, input_ids=input_ids)

    torch.testing.assert_close(route_indices, expected_indices)
    torch.testing.assert_close(route_weights.float(), expected_weights.float(), rtol=5e-2, atol=5e-2)


def test_tiny_prefill_compressor_overlap_uses_host_ratio_pooling_and_matches_torch(
    tiny_tt_preprocessed_checkpoint: Path, device
):
    """Smoke the first compressor prefill slice; overlap pooling is still host-side."""

    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    layer = 2
    weights = load_prefill_compressor_weights(tiny_tt_preprocessed_checkpoint, manifest=manifest, layer=layer)
    hidden_size = manifest["config"]["hidden_size"]
    head_dim = manifest["config"]["head_dim"]
    compress_ratio = manifest["config"]["compress_ratios"][layer]
    seq_len = 32
    torch_input = torch.linspace(-0.1, 0.1, steps=seq_len * hidden_size, dtype=torch.float32).reshape(
        1, 1, seq_len, hidden_size
    )
    torch_input = torch_input.to(torch.bfloat16)
    expected = compressor_prefill(
        torch_input[:, 0],
        weights.wkv,
        weights.wgate,
        weights.ape,
        weights.norm_weight,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
        norm_eps=float(manifest["config"]["rms_norm_eps"]),
        overlap=True,
    )

    tt_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    module = TtPrefillCompressor.from_preprocessed(tiny_tt_preprocessed_checkpoint, device=device, layer=layer)
    torch_output = ttnn.to_torch(module(tt_input))[:, 0]

    assert torch_output.shape == expected.shape
    torch.testing.assert_close(torch_output.float(), expected.float(), rtol=8e-2, atol=8e-2)


def test_tiny_attention_q_projection_module_matches_torch(tiny_tt_preprocessed_checkpoint: Path, device):
    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    weights = load_attention_projection_weights(
        tiny_tt_preprocessed_checkpoint,
        manifest=manifest,
        layer=0,
        include_output_projection=True,
    )
    hidden_size = manifest["config"]["hidden_size"]
    seq_len = 32
    torch_input = torch.linspace(-0.12, 0.18, steps=seq_len * hidden_size, dtype=torch.float32).reshape(
        1, 1, seq_len, hidden_size
    )
    torch_input = torch_input.to(torch.bfloat16)
    q_rank = F.linear(torch_input[:, 0].float(), weights.wq_a.float()).to(torch.bfloat16)
    expected = F.linear(
        rms_norm(q_rank, weights.q_norm, float(manifest["config"]["rms_norm_eps"])).float(),
        weights.wq_b.float(),
    ).unsqueeze(1)

    tt_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    module = TtAttentionProjection.from_preprocessed(tiny_tt_preprocessed_checkpoint, device=device, layer=0)
    torch_output = ttnn.to_torch(module.project_q(tt_input))

    assert torch_output.shape == expected.shape
    torch.testing.assert_close(torch_output.float(), expected.float(), rtol=8e-2, atol=8e-2)


def test_tiny_attention_output_projection_host_grouped_wo_a_matches_torch(
    tiny_tt_preprocessed_checkpoint: Path, device
):
    """Smoke output projection; grouped wo_a is an explicit host fallback."""

    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    weights = load_attention_projection_weights(
        tiny_tt_preprocessed_checkpoint,
        manifest=manifest,
        layer=0,
        include_output_projection=True,
    )
    num_heads = manifest["config"]["num_attention_heads"]
    head_dim = manifest["config"]["head_dim"]
    seq_len = 32
    attention_dim = num_heads * head_dim
    attention_output = torch.linspace(-0.2, 0.25, steps=seq_len * attention_dim, dtype=torch.float32).reshape(
        1, 1, seq_len, attention_dim
    )
    attention_output = attention_output.to(torch.bfloat16)
    output_rank = grouped_output_projection_a(
        attention_output[:, 0],
        weights.wo_a,
        o_groups=int(manifest["config"]["o_groups"]),
    )
    expected = F.linear(output_rank.float(), weights.wo_b.float()).unsqueeze(1)

    tt_attention_output = ttnn.from_torch(
        attention_output,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    module = TtAttentionProjection.from_preprocessed(tiny_tt_preprocessed_checkpoint, device=device, layer=0)
    torch_output = ttnn.to_torch(module.project_output(tt_attention_output))

    assert torch_output.shape == expected.shape
    torch.testing.assert_close(torch_output.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_tiny_prefill_attention_block_host_fallbacks_match_torch(tiny_tt_preprocessed_checkpoint: Path, device):
    """Integrated prefill block; compressor pooling, sparse attention, and grouped wo_a are host fallbacks."""

    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    layer = 2
    projection_weights = load_attention_projection_weights(
        tiny_tt_preprocessed_checkpoint,
        manifest=manifest,
        layer=layer,
        include_output_projection=True,
    )
    compressor_weights = load_prefill_compressor_weights(
        tiny_tt_preprocessed_checkpoint,
        manifest=manifest,
        layer=layer,
    )
    attn_sink = load_attention_sink(tiny_tt_preprocessed_checkpoint, manifest=manifest, layer=layer)

    hidden_size = manifest["config"]["hidden_size"]
    num_heads = manifest["config"]["num_attention_heads"]
    head_dim = manifest["config"]["head_dim"]
    compress_ratio = manifest["config"]["compress_ratios"][layer]
    seq_len = 32
    torch_input = torch.linspace(-0.18, 0.22, steps=seq_len * hidden_size, dtype=torch.float32).reshape(
        1, 1, seq_len, hidden_size
    )
    torch_input = torch_input.to(torch.bfloat16)

    q_rank = F.linear(torch_input[:, 0].float(), projection_weights.wq_a.to(torch.bfloat16).float()).to(torch.bfloat16)
    q = F.linear(
        rms_norm(
            q_rank,
            projection_weights.q_norm.to(torch.bfloat16),
            float(manifest["config"]["rms_norm_eps"]),
        ).float(),
        projection_weights.wq_b.to(torch.bfloat16).float(),
    ).to(torch.bfloat16)
    compressed_kv = compressor_prefill(
        torch_input[:, 0],
        compressor_weights.wkv.to(torch.bfloat16),
        compressor_weights.wgate.to(torch.bfloat16),
        compressor_weights.ape,
        compressor_weights.norm_weight,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
        norm_eps=float(manifest["config"]["rms_norm_eps"]),
        overlap=True,
    ).to(torch.bfloat16)
    q_heads = q.reshape(1, seq_len, num_heads, head_dim)
    index_weights = torch.linspace(0.5, 1.25, steps=num_heads, dtype=torch.float32).reshape(1, 1, num_heads)
    index_weights = index_weights.expand(1, seq_len, num_heads)
    topk_idxs = indexer_topk(
        q_heads,
        compressed_kv,
        index_weights,
        index_topk=int(manifest["config"]["index_topk"]),
        compress_ratio=compress_ratio,
        start_pos=0,
        offset=0,
    )
    attention_output = sparse_attention(
        q_heads,
        compressed_kv,
        attn_sink,
        topk_idxs,
        softmax_scale=head_dim**-0.5,
    )
    attention_output = attention_output.reshape(1, seq_len, num_heads * head_dim).to(torch.bfloat16)
    output_rank = grouped_output_projection_a(
        attention_output,
        projection_weights.wo_a,
        o_groups=int(manifest["config"]["o_groups"]),
    )
    expected = F.linear(output_rank.float(), projection_weights.wo_b.to(torch.bfloat16).float()).unsqueeze(1)

    tt_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    module = TtPrefillAttentionBlock.from_preprocessed(tiny_tt_preprocessed_checkpoint, device=device, layer=layer)
    torch_output = ttnn.to_torch(module(tt_input, topk_idxs=topk_idxs))

    assert torch.any(topk_idxs < 0)
    assert torch_output.shape == expected.shape
    passing, pcc_message = comp_pcc(expected.float(), torch_output.float(), pcc=0.999)
    assert passing, f"Prefill attention block output PCC below 0.999: {pcc_message}"


def test_tiny_sparse_prefill_attention_abi_smoke_matches_torch(device):
    """ABI/hardware smoke path; sparse gather, sink-softmax, and reduction are still host-side."""

    batch_size, seq_len, num_heads, head_dim, cache_len = 1, 32, 2, 32, 32
    q = torch.linspace(-0.35, 0.45, steps=batch_size * seq_len * num_heads * head_dim, dtype=torch.float32)
    q = q.reshape(batch_size, seq_len, num_heads, head_dim).to(torch.bfloat16)
    kv = torch.linspace(0.25, -0.4, steps=batch_size * cache_len * head_dim, dtype=torch.float32)
    kv = kv.reshape(batch_size, cache_len, head_dim).to(torch.bfloat16)
    attn_sink = torch.tensor([0.2, -0.15], dtype=torch.float32)
    token_ids = torch.arange(seq_len).view(1, seq_len)
    topk_idxs = torch.stack(
        [
            token_ids % cache_len,
            (token_ids + 5) % cache_len,
            (token_ids + 11) % cache_len,
        ],
        dim=-1,
    ).to(torch.int64)
    topk_idxs[:, 4::7, -1] = -1
    softmax_scale = 0.5
    expected = sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale)

    tt_q = ttnn.from_torch(
        q.reshape(batch_size, seq_len, num_heads * head_dim).unsqueeze(1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_kv = ttnn.from_torch(kv.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    module = TtSparsePrefillAttention(
        device=device,
        num_heads=num_heads,
        head_dim=head_dim,
        softmax_scale=softmax_scale,
    )
    torch_output = ttnn.to_torch(module(tt_q, tt_kv, attn_sink=attn_sink, topk_idxs=topk_idxs))[:, 0]
    torch_output = torch_output.reshape(batch_size, seq_len, num_heads, head_dim)

    assert torch_output.shape == expected.shape
    torch.testing.assert_close(torch_output.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_tiny_indexer_topk_feeds_sparse_prefill_attention_abi_smoke(device):
    """CPU indexer feed-through smoke for the sparse attention ABI boundary."""

    batch_size, seq_len, num_heads, head_dim, cache_len = 1, 32, 2, 32, 32
    q = torch.arange(batch_size * seq_len * num_heads * head_dim, dtype=torch.float32)
    q = ((q % 29) - 14).reshape(batch_size, seq_len, num_heads, head_dim).to(torch.bfloat16) / 17
    kv = torch.arange(batch_size * cache_len * head_dim, dtype=torch.float32)
    kv = ((kv % 23) - 8).reshape(batch_size, cache_len, head_dim).to(torch.bfloat16) / 19
    weights = torch.tensor([[[1.0, 0.5]]], dtype=torch.float32).expand(batch_size, seq_len, num_heads)
    attn_sink = torch.tensor([0.0, -0.25], dtype=torch.float32)
    softmax_scale = 0.25
    topk_idxs = indexer_topk(q, kv, weights, index_topk=4, compress_ratio=1, start_pos=0, offset=0)
    expected = sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale)

    tt_q = ttnn.from_torch(
        q.reshape(batch_size, seq_len, num_heads * head_dim).unsqueeze(1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_kv = ttnn.from_torch(kv.unsqueeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    module = TtSparsePrefillAttention(
        device=device,
        num_heads=num_heads,
        head_dim=head_dim,
        softmax_scale=softmax_scale,
    )
    torch_output = ttnn.to_torch(module(tt_q, tt_kv, attn_sink=attn_sink, topk_idxs=topk_idxs))[:, 0]
    torch_output = torch_output.reshape(batch_size, seq_len, num_heads, head_dim)

    assert torch.any(topk_idxs < 0)
    torch.testing.assert_close(torch_output.float(), expected.float(), rtol=1e-2, atol=1e-2)
