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
    v4_router,
)
from models.demos.deepseek_v4_flash.expert_abi import load_packed_expert_weight
from models.demos.deepseek_v4_flash.expert_plan import plan_batch1_decode_expert_placements
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint

ttnn = pytest.importorskip("ttnn")

from models.demos.deepseek_v4_flash.ttnn_attention_projection import (
    grouped_output_projection_a,
    load_attention_projection_weights,
)
from models.demos.deepseek_v4_flash.ttnn_decoder_layer import TtDecoderLayer, load_decoder_layer_norm_weights
from models.demos.deepseek_v4_flash.ttnn_expert_group import (
    TtPlannedRoutedExpertGroup,
    route_weights_by_expert,
    unique_route_expert_ids,
)
from models.demos.deepseek_v4_flash.ttnn_model import (
    TtDeepSeekV4FlashTinyModel,
    embed_input_ids_host,
    load_model_embedding_head_weights,
)
from models.demos.deepseek_v4_flash.ttnn_moe_block import TtMoEFeedForwardBlock
from models.demos.deepseek_v4_flash.ttnn_prefill_attention_block import load_attention_sink
from models.demos.deepseek_v4_flash.ttnn_prefill_compressor import load_prefill_compressor_weights
from models.demos.deepseek_v4_flash.ttnn_prefill_indexer import load_prefill_indexer_weights
from models.demos.deepseek_v4_flash.ttnn_routed_expert import TtRoutedExpertMLP
from models.demos.deepseek_v4_flash.ttnn_router import TtRouter, load_router_weights

pytestmark = pytest.mark.t3k_compat


def _skip_unless_t3k() -> None:
    try:
        cluster_type = ttnn.cluster.get_cluster_type()
        num_devices = ttnn.get_num_devices()
    except Exception as exc:
        pytest.skip(f"Unable to query TT cluster for T3K expert-plan test: {exc}")

    if cluster_type != ttnn.cluster.ClusterType.T3K or num_devices != 8:
        pytest.skip(f"Requires T3K with 8 devices, found cluster_type={cluster_type}, num_devices={num_devices}")


@pytest.fixture(scope="module")
def tiny_tt_preprocessed_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    artifact_root = os.environ.get("DSV4_FLASH_ARTIFACT_DIR")
    if artifact_root:
        base = Path(artifact_root) / "pytest" / f"deepseek_v4_flash_t3k_expert_plan_{os.getpid()}"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)
    else:
        base = tmp_path_factory.mktemp("deepseek_v4_flash_t3k_expert_plan")

    source = generate_tiny_hf_checkpoint(base / "hf_source", num_hidden_layers=1)
    return convert_hf_checkpoint(source, base / "tt_preprocessed")


@pytest.fixture(scope="module")
def tiny_three_layer_tt_preprocessed_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    artifact_root = os.environ.get("DSV4_FLASH_ARTIFACT_DIR")
    if artifact_root:
        base = Path(artifact_root) / "pytest" / f"deepseek_v4_flash_t3k_decoder_layer_{os.getpid()}"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)
    else:
        base = tmp_path_factory.mktemp("deepseek_v4_flash_t3k_decoder_layer")

    source = generate_tiny_hf_checkpoint(base / "hf_source", num_hidden_layers=3)
    return convert_hf_checkpoint(source, base / "tt_preprocessed")


@pytest.fixture(scope="module")
def tiny_compressed_stack_tt_preprocessed_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    artifact_root = os.environ.get("DSV4_FLASH_ARTIFACT_DIR")
    if artifact_root:
        base = Path(artifact_root) / "pytest" / f"deepseek_v4_flash_t3k_model_stack_{os.getpid()}"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)
    else:
        base = tmp_path_factory.mktemp("deepseek_v4_flash_t3k_model_stack")

    source = generate_tiny_hf_checkpoint(base / "hf_source", num_hidden_layers=3, compress_ratios=(4, 0, 4))
    return convert_hf_checkpoint(source, base / "tt_preprocessed")


@pytest.fixture
def t3k_mesh():
    _skip_unless_t3k()
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4))
    try:
        yield mesh_device
    finally:
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)


def test_t3k_planned_primary_replica_runs_tiny_routed_expert(
    tiny_tt_preprocessed_checkpoint: Path,
    t3k_mesh,
) -> None:
    plan = plan_batch1_decode_expert_placements((2, 4), (0, 2), replicas_per_expert=1)
    expert_id = 2
    primary_replica = plan.placements[1].primary_replica

    assert primary_replica.expert_id == expert_id
    assert primary_replica.mesh_coord == (1, 0)
    assert plan.devices_for_expert(expert_id) == ((1, 0),)

    submesh = t3k_mesh.create_submesh(
        ttnn.MeshShape(1, 1),
        offset=ttnn.MeshCoordinate(*primary_replica.mesh_coord),
    )

    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    weights = _load_expert_weights(tiny_tt_preprocessed_checkpoint, expert_id)

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

    tt_input = ttnn.from_torch(torch_input, device=submesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_route_weights = ttnn.from_torch(
        route_weights.reshape(1, 1, 32, 1).expand(1, 1, 32, intermediate_size).to(torch.bfloat16),
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    module = TtRoutedExpertMLP.from_preprocessed(
        tiny_tt_preprocessed_checkpoint,
        device=submesh,
        expert=expert_id,
    )
    torch_output = ttnn.to_torch(module(tt_input, route_weight=tt_route_weights))

    torch.testing.assert_close(torch_output.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_t3k_planned_routed_expert_group_host_combines_two_tiny_experts(
    tiny_tt_preprocessed_checkpoint: Path,
    t3k_mesh,
) -> None:
    plan = plan_batch1_decode_expert_placements((2, 4), (0, 2), replicas_per_expert=1)

    assert plan.devices_for_expert(0) == ((0, 0),)
    assert plan.devices_for_expert(2) == ((1, 0),)

    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    weights_by_expert = {
        expert_id: _load_expert_weights(tiny_tt_preprocessed_checkpoint, expert_id)
        for expert_id in plan.activated_expert_ids
    }

    hidden_size = weights_by_expert[0]["w1"].shape[-1]
    torch_input = torch.linspace(-0.3, 0.3, steps=32 * hidden_size, dtype=torch.float32).reshape(1, 1, 32, hidden_size)
    torch_input = torch_input.to(torch.bfloat16)
    route_weights = torch.stack(
        [
            torch.linspace(0.2, 0.8, steps=32, dtype=torch.float32),
            torch.linspace(0.7, 0.1, steps=32, dtype=torch.float32),
        ],
        dim=-1,
    ).reshape(1, 32, 2)
    route_indices = torch.tensor(plan.activated_expert_ids, dtype=torch.int64).reshape(1, 1, 2).expand(1, 32, 2)
    expected = combine_routed_experts(
        torch_input.reshape(1, 32, hidden_size),
        route_weights,
        route_indices,
        {expert_id: (weights["w1"], weights["w2"], weights["w3"]) for expert_id, weights in weights_by_expert.items()},
        swiglu_limit=float(manifest["config"]["swiglu_limit"]),
    ).reshape_as(torch_input)

    group = TtPlannedRoutedExpertGroup.from_preprocessed(tiny_tt_preprocessed_checkpoint, t3k_mesh, plan)
    torch_output = group.run_torch_host_combine(
        torch_input,
        {
            expert_id: route_weights[:, :, topk_index : topk_index + 1]
            for topk_index, expert_id in enumerate(plan.activated_expert_ids)
        },
    )

    torch.testing.assert_close(torch_output.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_t3k_router_feeds_planned_routed_expert_group(
    tiny_tt_preprocessed_checkpoint: Path,
    t3k_mesh,
) -> None:
    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    router_weights = load_router_weights(tiny_tt_preprocessed_checkpoint, manifest=manifest, layer=0)

    hidden_size = router_weights.gate_weight.shape[-1]
    seq_len = 32
    torch_input = torch.linspace(-0.25, 0.25, steps=seq_len * hidden_size, dtype=torch.float32).reshape(
        1, 1, seq_len, hidden_size
    )
    torch_input = torch_input.to(torch.bfloat16)
    input_ids = torch.zeros(1, seq_len, dtype=torch.int64)

    router_submesh = t3k_mesh.create_submesh(
        ttnn.MeshShape(1, 1),
        offset=ttnn.MeshCoordinate(0, 3),
    )
    tt_input = ttnn.from_torch(torch_input, device=router_submesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    router = TtRouter.from_preprocessed(tiny_tt_preprocessed_checkpoint, device=router_submesh, layer=0)
    route_weights, route_indices = router(tt_input, input_ids=input_ids)

    expected_route_weights, expected_route_indices = v4_router(
        torch_input[:, 0],
        router_weights.gate_weight,
        topk=int(manifest["config"]["num_experts_per_tok"]),
        route_scale=float(manifest["config"]["routed_scaling_factor"]),
        scoring_func=str(manifest["config"]["scoring_func"]),
        input_ids=input_ids,
        tid2eid=router_weights.tid2eid,
    )
    torch.testing.assert_close(route_indices, expected_route_indices)
    torch.testing.assert_close(route_weights.float(), expected_route_weights.float(), rtol=5e-2, atol=5e-2)

    active_expert_ids = unique_route_expert_ids(route_indices)
    assert len(active_expert_ids) == 2
    plan = plan_batch1_decode_expert_placements((2, 4), active_expert_ids, replicas_per_expert=1)
    weights_by_expert = {
        expert_id: _load_expert_weights(tiny_tt_preprocessed_checkpoint, expert_id) for expert_id in active_expert_ids
    }
    expected = combine_routed_experts(
        torch_input[:, 0],
        route_weights,
        route_indices,
        {expert_id: (weights["w1"], weights["w2"], weights["w3"]) for expert_id, weights in weights_by_expert.items()},
        swiglu_limit=float(manifest["config"]["swiglu_limit"]),
    ).reshape_as(torch_input)

    group = TtPlannedRoutedExpertGroup.from_preprocessed(tiny_tt_preprocessed_checkpoint, t3k_mesh, plan)
    torch_output = group.run_torch_host_combine(
        torch_input,
        route_weights_by_expert(route_weights, route_indices, active_expert_ids),
    )

    torch.testing.assert_close(torch_output.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_t3k_moe_ffn_block_routes_plans_shared_and_host_combines(
    tiny_tt_preprocessed_checkpoint: Path,
    t3k_mesh,
) -> None:
    """Single-module smoke; requires T3K because routed experts use planned disjoint 1x1 submeshes."""

    manifest = load_tt_manifest(tiny_tt_preprocessed_checkpoint)
    router_weights = load_router_weights(tiny_tt_preprocessed_checkpoint, manifest=manifest, layer=0)

    hidden_size = router_weights.gate_weight.shape[-1]
    seq_len = 32
    torch_input = torch.linspace(-0.2, 0.3, steps=seq_len * hidden_size, dtype=torch.float32).reshape(
        1, 1, seq_len, hidden_size
    )
    torch_input = torch_input.to(torch.bfloat16)
    input_ids = torch.zeros(1, seq_len, dtype=torch.int64)

    expected_route_weights, expected_route_indices = v4_router(
        torch_input[:, 0],
        router_weights.gate_weight,
        topk=int(manifest["config"]["num_experts_per_tok"]),
        route_scale=float(manifest["config"]["routed_scaling_factor"]),
        scoring_func=str(manifest["config"]["scoring_func"]),
        input_ids=input_ids,
        tid2eid=router_weights.tid2eid,
    )
    active_expert_ids = unique_route_expert_ids(expected_route_indices)
    weights_by_expert = {
        expert_id: _load_expert_weights(tiny_tt_preprocessed_checkpoint, expert_id) for expert_id in active_expert_ids
    }
    shared_expert = _load_shared_expert_weights(tiny_tt_preprocessed_checkpoint)
    expected = combine_routed_experts(
        torch_input[:, 0],
        expected_route_weights,
        expected_route_indices,
        {expert_id: (weights["w1"], weights["w2"], weights["w3"]) for expert_id, weights in weights_by_expert.items()},
        shared_expert=shared_expert,
        swiglu_limit=float(manifest["config"]["swiglu_limit"]),
    ).reshape_as(torch_input)

    block = TtMoEFeedForwardBlock.from_preprocessed(tiny_tt_preprocessed_checkpoint, mesh_device=t3k_mesh, layer=0)
    tt_input = ttnn.from_torch(
        torch_input,
        device=block.primary_submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    torch_output = block(tt_input, input_ids=input_ids)

    torch.testing.assert_close(torch_output.float(), expected.float(), rtol=8e-2, atol=8e-2)


def test_t3k_decoder_layer_scaffold_attention_moe_residuals_match_torch(
    tiny_three_layer_tt_preprocessed_checkpoint: Path,
    t3k_mesh,
) -> None:
    """Callable decoder layer smoke; hyperconnection mixing remains intentionally absent."""

    manifest = load_tt_manifest(tiny_three_layer_tt_preprocessed_checkpoint)
    layer = 2
    hidden_size = manifest["config"]["hidden_size"]
    seq_len = 32
    torch_input = torch.linspace(-0.14, 0.26, steps=seq_len * hidden_size, dtype=torch.float32).reshape(
        1, 1, seq_len, hidden_size
    )
    torch_input = torch_input.to(torch.bfloat16)
    input_ids = torch.zeros(1, seq_len, dtype=torch.int64)

    expected = _decoder_layer_reference(
        tiny_three_layer_tt_preprocessed_checkpoint,
        manifest,
        layer=layer,
        hidden_states=torch_input,
        input_ids=input_ids,
    )

    module = TtDecoderLayer.from_preprocessed(
        tiny_three_layer_tt_preprocessed_checkpoint,
        mesh_device=t3k_mesh,
        layer=layer,
    )
    torch_output = module(torch_input, input_ids=input_ids)

    assert torch_output.shape == expected.shape
    passing, pcc_message = comp_pcc(expected.float(), torch_output.float(), pcc=0.98)
    assert passing, f"Decoder layer output PCC below 0.98: {pcc_message}"


def test_t3k_tiny_model_scaffold_embedding_decoder_logits_match_torch(
    tiny_three_layer_tt_preprocessed_checkpoint: Path,
    t3k_mesh,
) -> None:
    """Tiny model-level scaffold: host embedding, one decoder layer, TTNN LM head."""

    manifest = load_tt_manifest(tiny_three_layer_tt_preprocessed_checkpoint)
    layer = 2
    seq_len = 32
    vocab_size = int(manifest["config"]["vocab_size"])
    input_ids = torch.zeros(1, seq_len, dtype=torch.int64)

    expected = _tiny_model_reference_logits(
        tiny_three_layer_tt_preprocessed_checkpoint,
        manifest,
        layer=layer,
        input_ids=input_ids,
    )

    module = TtDeepSeekV4FlashTinyModel.from_preprocessed(
        tiny_three_layer_tt_preprocessed_checkpoint,
        mesh_device=t3k_mesh,
        layer=layer,
    )
    torch_output = module(input_ids)

    assert torch_output.shape == (1, seq_len, vocab_size)
    passing, pcc_message = comp_pcc(expected.float(), torch_output.float(), pcc=0.98)
    assert passing, f"Tiny model logits PCC below 0.98: {pcc_message}"


def test_t3k_tiny_model_scaffold_two_decoder_layers_match_torch(
    tiny_compressed_stack_tt_preprocessed_checkpoint: Path,
    t3k_mesh,
) -> None:
    """Tiny model stack smoke: host embedding, two decoder layers, TTNN LM head."""

    manifest = load_tt_manifest(tiny_compressed_stack_tt_preprocessed_checkpoint)
    layer_ids = (0, 2)
    seq_len = 32
    vocab_size = int(manifest["config"]["vocab_size"])
    input_ids = torch.zeros(1, seq_len, dtype=torch.int64)

    expected = _tiny_model_reference_logits(
        tiny_compressed_stack_tt_preprocessed_checkpoint,
        manifest,
        layer_ids=layer_ids,
        input_ids=input_ids,
    )

    module = TtDeepSeekV4FlashTinyModel.from_preprocessed(
        tiny_compressed_stack_tt_preprocessed_checkpoint,
        mesh_device=t3k_mesh,
        layer_ids=layer_ids,
    )
    torch_output = module(input_ids)

    assert module.layer_ids == layer_ids
    assert torch_output.shape == (1, seq_len, vocab_size)
    passing, pcc_message = comp_pcc(expected.float(), torch_output.float(), pcc=0.75)
    assert passing, f"Tiny model stack logits PCC below 0.75: {pcc_message}"


def _tiny_model_reference_logits(
    preprocessed_path: Path,
    manifest: dict,
    *,
    input_ids: torch.Tensor,
    layer: int | None = None,
    layer_ids: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if layer_ids is None:
        if layer is None:
            raise ValueError("layer or layer_ids must be provided")
        layer_ids = (int(layer),)
    weights = load_model_embedding_head_weights(preprocessed_path, manifest=manifest)
    hidden_states = embed_input_ids_host(input_ids, weights.embed_weight).unsqueeze(1)
    for layer_id in layer_ids:
        hidden_states = _decoder_layer_reference(
            preprocessed_path,
            manifest,
            layer=layer_id,
            hidden_states=hidden_states,
            input_ids=input_ids,
        )
    return F.linear(hidden_states[:, 0].float(), weights.head_weight.float())


def _decoder_layer_reference(
    preprocessed_path: Path,
    manifest: dict,
    *,
    layer: int,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    norm_weights = load_decoder_layer_norm_weights(preprocessed_path, manifest=manifest, layer=layer)
    projection_weights = load_attention_projection_weights(
        preprocessed_path,
        manifest=manifest,
        layer=layer,
        include_output_projection=True,
    )
    compressor_weights = load_prefill_compressor_weights(preprocessed_path, manifest=manifest, layer=layer)
    indexer_weights = load_prefill_indexer_weights(preprocessed_path, manifest=manifest, layer=layer)
    attn_sink = load_attention_sink(preprocessed_path, manifest=manifest, layer=layer)

    num_heads = manifest["config"]["num_attention_heads"]
    head_dim = manifest["config"]["head_dim"]
    index_n_heads = manifest["config"]["index_n_heads"]
    index_head_dim = manifest["config"]["index_head_dim"]
    compress_ratio = manifest["config"]["compress_ratios"][layer]
    seq_len = hidden_states.shape[2]
    norm_eps = float(manifest["config"]["rms_norm_eps"])

    attn_input = rms_norm(hidden_states[:, 0], norm_weights.attn_norm, norm_eps)
    q_rank = F.linear(attn_input.float(), projection_weights.wq_a.to(torch.bfloat16).float()).to(torch.bfloat16)
    q_rank = rms_norm(q_rank, projection_weights.q_norm.to(torch.bfloat16), norm_eps)
    q = F.linear(q_rank.float(), projection_weights.wq_b.to(torch.bfloat16).float()).to(torch.bfloat16)
    compressed_kv = compressor_prefill(
        attn_input,
        compressor_weights.wkv.to(torch.bfloat16),
        compressor_weights.wgate.to(torch.bfloat16),
        compressor_weights.ape,
        compressor_weights.norm_weight,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
        norm_eps=norm_eps,
        overlap=True,
    ).to(torch.bfloat16)
    index_q = F.linear(q_rank.float(), indexer_weights.wq_b.to(torch.bfloat16).float()).to(torch.bfloat16)
    index_q = index_q.reshape(1, seq_len, index_n_heads, index_head_dim)
    index_kv = compressor_prefill(
        attn_input,
        indexer_weights.compressor.wkv.to(torch.bfloat16),
        indexer_weights.compressor.wgate.to(torch.bfloat16),
        indexer_weights.compressor.ape,
        indexer_weights.compressor.norm_weight,
        compress_ratio=compress_ratio,
        head_dim=index_head_dim,
        norm_eps=norm_eps,
        overlap=True,
    )
    index_weights = F.linear(attn_input.float(), indexer_weights.weights_proj.to(torch.bfloat16).float()).to(
        torch.bfloat16
    )
    index_weights = index_weights.float() * (index_head_dim**-0.5 * index_n_heads**-0.5)
    topk_idxs = indexer_topk(
        index_q,
        index_kv,
        index_weights,
        index_topk=int(manifest["config"]["index_topk"]),
        compress_ratio=compress_ratio,
        start_pos=0,
        offset=0,
    )
    attention_output = sparse_attention(
        q.reshape(1, seq_len, num_heads, head_dim),
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
    attention_output = F.linear(output_rank.float(), projection_weights.wo_b.to(torch.bfloat16).float()).unsqueeze(1)
    hidden_after_attention = (hidden_states.float() + attention_output.float()).to(hidden_states.dtype)

    ffn_input = rms_norm(hidden_after_attention[:, 0], norm_weights.ffn_norm, norm_eps)
    router_weights = load_router_weights(preprocessed_path, manifest=manifest, layer=layer)
    route_weights, route_indices = v4_router(
        ffn_input,
        router_weights.gate_weight,
        topk=int(manifest["config"]["num_experts_per_tok"]),
        route_scale=float(manifest["config"]["routed_scaling_factor"]),
        scoring_func=str(manifest["config"]["scoring_func"]),
        input_ids=input_ids,
        tid2eid=router_weights.tid2eid,
    )
    active_expert_ids = unique_route_expert_ids(route_indices)
    weights_by_expert = {
        expert_id: _load_expert_weights(preprocessed_path, expert_id, layer=layer) for expert_id in active_expert_ids
    }
    ffn_output = combine_routed_experts(
        ffn_input,
        route_weights,
        route_indices,
        {expert_id: (weights["w1"], weights["w2"], weights["w3"]) for expert_id, weights in weights_by_expert.items()},
        shared_expert=_load_shared_expert_weights(preprocessed_path, layer=layer),
        swiglu_limit=float(manifest["config"]["swiglu_limit"]),
    ).unsqueeze(1)
    return (hidden_after_attention.float() + ffn_output.float()).to(hidden_states.dtype)


def _load_expert_weights(preprocessed_path: Path, expert_id: int, *, layer: int = 0) -> dict[str, torch.Tensor]:
    return {
        projection: load_packed_expert_weight(
            preprocessed_path, layer=layer, expert=expert_id, projection=projection
        ).dequantize(dtype=torch.bfloat16)
        for projection in ("w1", "w2", "w3")
    }


def _load_shared_expert_weights(
    preprocessed_path: Path, *, layer: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    manifest = load_tt_manifest(preprocessed_path)
    non_expert_path = preprocessed_path / manifest["artifacts"]["non_expert_safetensors"][0]
    with safe_open(non_expert_path, framework="pt", device="cpu") as handle:
        return tuple(
            handle.get_tensor(f"layers.{layer}.ffn.shared_experts.{projection}.weight").to(torch.bfloat16)
            for projection in ("w1", "w2", "w3")
        )
