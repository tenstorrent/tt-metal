# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real-weight batch ownership against independently generated HF layer outputs."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.gpt_oss_120b_qb2.tt.model import HF_CONTEXT_LENGTH, MODEL_REVISION, StreamingCheckpoint
from models.demos.gpt_oss_120b_qb2.tt.multichip_decoder import MultichipDecoder
from models.demos.gpt_oss_120b_qb2.tt.precision import load_precision_config
from models.tt_transformers.tt.common import rope_scaling_model_factory
from models.tt_transformers.tt.rope import compute_gather_cos_sin

ACTIVE_BATCHES = {1: (1,), 4: (2, 3, 4), 8: (5, 7, 8), 32: (9, 15, 16, 17, 31, 32)}


def _reference(layer_index):
    root = Path(os.environ["GPT_OSS_120B_BATCH_REFERENCE"])
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["checkpoint_revision"] == MODEL_REVISION
    assert manifest["transformers"] == "5.12.1"
    assert manifest["layers"] == [0, 1]
    values = []
    for name in (f"layer{layer_index}-input.pt", f"layer{layer_index}-output.pt"):
        path = root / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][name]["sha256"]
        values.append(torch.load(path, map_location="cpu", weights_only=True))
    inputs, expected = values[0]["input"], values[1]["output"]
    assert inputs.shape == expected.shape == (32, 129, 2880)
    return inputs, expected


def _tensor(value, mesh_device, *, integer=False):
    return ttnn.from_torch(
        value,
        device=mesh_device,
        dtype=ttnn.int32 if integer else ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _rows_pcc(actual, expected):
    assert actual.shape == expected.shape
    assert actual.isfinite().all() and expected.isfinite().all()
    actual = actual.float().flatten(1)
    expected = expected.float().flatten(1)
    actual = actual - actual.mean(dim=1, keepdim=True)
    expected = expected - expected.mean(dim=1, keepdim=True)
    return ((actual * expected).sum(dim=1) / (actual.norm(dim=1) * expected.norm(dim=1))).tolist()


def _read_ranks(output, shape):
    ranks = [ttnn.to_torch(tensor).clone().reshape(shape) for tensor in ttnn.get_device_tensors(output)]
    assert len(ranks) == 4
    for rank in ranks[1:]:
        assert torch.equal(ranks[0], rank), "Tensor-parallel residual replicas disagree"
    return ranks[0]


def _decode_rope(cos, sin, positions, layer):
    # Decode RoPE pairs cosine/sine rows with the same physical user core grid
    # as the layer transformation matrix, including the 32-row bucket.
    memory = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, layer.hf_config.head_dim),
        core_grid=layer.self_attn.transformation_mats["decode"].memory_config().shard_spec.grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    result = []
    for value in (cos, sin):
        interleaved = _tensor(value[:, :, positions.clamp_min(0).long()].permute(0, 2, 1, 3), layer.mesh_device)
        result.append(ttnn.to_memory_config(interleaved, memory))
        interleaved.deallocate(True)
    return result


def _layer_and_rope(mesh_device, layer_index, width):
    config = AutoConfig.from_pretrained(os.environ["GPT_OSS_120B_SNAPSHOT"], local_files_only=True)
    assert config.max_position_embeddings == HF_CONTEXT_LENGTH
    checkpoint = StreamingCheckpoint(os.environ["GPT_OSS_120B_SNAPSHOT"])
    precision = load_precision_config()
    layer = MultichipDecoder.from_state_dict(
        checkpoint.layer_state_dict(layer_index),
        hf_config=config,
        layer_idx=layer_index,
        mesh_device=mesh_device,
        max_batch_size=width,
        max_context_length=HF_CONTEXT_LENGTH,
        page_size=64,
        tensor_cache_path=Path(os.environ["TT_METAL_CACHE"])
        / "layer_state"
        / f"layer_{layer_index}"
        / f"width_{width}",
        calibrated_checkpoint_revision=MODEL_REVISION,
        policy=precision.decoder_policy_for_layer(layer_index),
    )
    cos, sin = compute_gather_cos_sin(
        dhead=config.head_dim,
        end=2 * HF_CONTEXT_LENGTH,
        theta=getattr(config, "rope_theta", None) or getattr(config, "default_theta", 150000.0),
        rope_scaling=rope_scaling_model_factory(config.rope_scaling),
    )
    return layer, config, cos, sin


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("layer_index", [0, 1])
@pytest.mark.parametrize("width", [1, 4, 8, 32])
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "require_exact_physical_num_devices": True})],
    indirect=True,
)
def test_partial_batches_preserve_distinct_history(mesh_device, device_params, layer_index, width):
    """Reuse one paged cache across both request orderings at every current decode width."""
    del device_params
    inputs, expected = _reference(layer_index)
    layer, config, cos, sin = _layer_and_rope(mesh_device, layer_index, width)
    pages_per_user = HF_CONTEXT_LENGTH // 64
    pages = torch.randperm(width * pages_per_user, generator=torch.Generator().manual_seed(901 + layer_index))
    pages = pages.to(torch.int32).reshape(width, pages_per_user)
    result_root = Path(os.environ["GPT_OSS_120B_RESULTS"]) / "layer_state"
    result_root.mkdir(parents=True, exist_ok=True)
    result_path = result_root / f"layer{layer_index}-width{width}.json"
    hidden = _tensor(inputs[:width, :128].unsqueeze(0), mesh_device)
    rope = [_tensor(value[:, :, :128], mesh_device) for value in (cos, sin)]
    table = _tensor(pages, mesh_device, integer=True)
    output = layer.prefill_forward(
        hidden, position_embeddings=rope, page_table=table, batch_size=width, fill_seq_lens=[128] * width
    )
    prefill = _read_ranks(output, (width, 128, config.hidden_size))
    records = [{"mode": "prefill", "width": width, "row_pcc": _rows_pcc(prefill, expected[:width, :128])}]
    result_path.write_text(json.dumps({"layer": layer_index, "width": width, "cases": records}, indent=2) + "\n")
    assert min(records[0]["row_pcc"]) >= 0.99, records[0]
    for tensor in (output, hidden, table, *rope):
        tensor.deallocate(True)

    try:
        for active in ACTIVE_BATCHES[width]:
            for permuted in (False, True):
                order = torch.arange(width)
                if permuted:
                    order = order.roll(active // 2).flip(0)
                selected = order[:active]
                current = torch.full((width,), -1, dtype=torch.int32)
                current[:active] = 128
                host = torch.zeros((1, 1, width, config.hidden_size), dtype=torch.bfloat16)
                host[0, 0, :active] = inputs[selected, 128]
                hidden = _tensor(host, mesh_device)
                position = _tensor(current, mesh_device, integer=True)
                table = _tensor(pages[order], mesh_device, integer=True)
                rope = _decode_rope(cos, sin, current, layer)
                output = layer.decode_forward(
                    hidden,
                    position_embeddings=rope,
                    current_position=position,
                    page_table=table,
                    batch_size=width,
                )
                actual = _read_ranks(output, (width, config.hidden_size))[:active]
                row = {
                    "mode": "decode",
                    "active": active,
                    "width": width,
                    "permuted": permuted,
                    "request_order": selected.tolist(),
                    "row_pcc": _rows_pcc(actual, expected[selected, 128]),
                }
                records.append(row)
                for tensor in (output, hidden, position, table, *rope):
                    tensor.deallocate(True)
                assert min(row["row_pcc"]) >= 0.99, row
    finally:
        result_path.write_text(json.dumps({"layer": layer_index, "width": width, "cases": records}, indent=2) + "\n")


BOUNDARIES = (63, 64, 65, 511, 512, 513, 767, 768, 769, 8191, 8192, 8193)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("layer_index", [0, 1])
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "require_exact_physical_num_devices": True})],
    indirect=True,
)
def test_page_ring_and_chunk_boundaries(mesh_device, device_params, layer_index):
    """Compare paged decode and warm/cold chunk continuation to independent HF rows."""
    del device_params
    root = Path(os.environ["GPT_OSS_120B_BOUNDARY_REFERENCE"])
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["checkpoint_revision"] == MODEL_REVISION
    assert manifest["transformers"] == "5.12.1" and manifest["layers"] == 36
    assert manifest["boundaries"] == list(BOUNDARIES)
    tensors = []
    for name, key in ((f"layer{layer_index}-input.pt", "input"), (f"layer{layer_index}-output.pt", "output")):
        path = root / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][name]["sha256"]
        tensors.append(torch.load(path, map_location="cpu", weights_only=True)[key])
    inputs, expected = tensors
    assert inputs.shape == expected.shape == (1, max(BOUNDARIES), 2880)
    layer, config, cos, sin = _layer_and_rope(mesh_device, layer_index, 1)
    pages_per_user = HF_CONTEXT_LENGTH // 64
    if layer.self_attn.config.cache_position_modulo:
        # The serving ring owns a contiguous run of twelve physical blocks;
        # repeat those IDs across the logical table at a nonzero physical offset.
        pages = (torch.arange(pages_per_user) % 12 + 24).reshape(1, -1).to(torch.int32)
    else:
        pages = torch.randperm(pages_per_user, generator=torch.Generator().manual_seed(903)).reshape(1, -1)
        pages = pages.to(torch.int32)
    records = []
    result_path = Path(os.environ["GPT_OSS_120B_RESULTS"]) / "layer_state" / f"layer{layer_index}-boundaries.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    def prefill(start, stop, *, cold=False):
        hidden = _tensor(inputs[:, start:stop].unsqueeze(0), mesh_device)
        rope = [_tensor(value[:, :, start:stop], mesh_device) for value in (cos, sin)]
        table = _tensor(pages, mesh_device, integer=True)
        tail = None
        if start and layer.self_attn.config.cache_position_modulo:
            tail = -1 if cold else int(pages[0, start // 64 - 2])
        output = layer.prefill_forward(
            hidden,
            position_embeddings=rope,
            page_table=table,
            batch_size=1,
            fill_seq_lens=[stop - start],
            chunk_start_idx=start,
            ring_tail_block=tail,
        )
        actual = _read_ranks(output, (1, stop - start, config.hidden_size))[:, -1]
        for tensor in (output, hidden, table, *rope):
            tensor.deallocate(True)
        return _rows_pcc(actual, expected[:, stop - 1])[0]

    try:
        for boundary in BOUNDARIES:
            stop = boundary - 1
            resume = max(0, (stop - config.sliding_window) // 512 * 512)
            modes = ["whole"]
            if resume:
                modes.append("warm")
                if layer.self_attn.config.cache_position_modulo:
                    modes.append("cold_ring")
            for mode in modes:
                start = 0 if mode == "whole" else resume
                if mode == "warm":
                    prefill(0, start)
                prefill_pcc = prefill(start, stop, cold=mode == "cold_ring")
                hidden = _tensor(inputs[:, stop:boundary].reshape(1, 1, 1, -1), mesh_device)
                position = _tensor(torch.tensor([stop], dtype=torch.int32), mesh_device, integer=True)
                table = _tensor(pages, mesh_device, integer=True)
                rope = _decode_rope(cos, sin, torch.tensor([stop]), layer)
                output = layer.decode_forward(
                    hidden,
                    position_embeddings=rope,
                    current_position=position,
                    page_table=table,
                    batch_size=1,
                )
                actual = _read_ranks(output, (1, config.hidden_size))
                row = {
                    "boundary": boundary,
                    "mode": mode,
                    "resume": start,
                    "last_prefill_pcc": prefill_pcc,
                    "decode_pcc": _rows_pcc(actual, expected[:, stop])[0],
                }
                records.append(row)
                for tensor in (output, hidden, position, table, *rope):
                    tensor.deallocate(True)
                assert min(row["last_prefill_pcc"], row["decode_pcc"]) >= 0.99, row
    finally:
        result_path.write_text(json.dumps({"layer": layer_index, "cases": records}, indent=2) + "\n")


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [((1, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "require_exact_physical_num_devices": True})],
    indirect=True,
)
def test_padded_single_user_prefill_preserves_live_ring(mesh_device, device_params):
    """A 1024-token padded launch must retain a 128-token live prompt in its 768-token ring."""
    del device_params
    inputs, expected = _reference(0)
    layer, config, cos, sin = _layer_and_rope(mesh_device, 0, 1)
    padded = torch.zeros((1, 1, 1024, config.hidden_size), dtype=torch.bfloat16)
    padded[0, 0, :128] = inputs[0, :128]
    hidden = _tensor(padded, mesh_device)
    rope = [_tensor(value[:, :, :1024], mesh_device) for value in (cos, sin)]
    table = _tensor(torch.arange(HF_CONTEXT_LENGTH // 64, dtype=torch.int32).reshape(1, -1), mesh_device, integer=True)
    output = layer.prefill_forward(
        hidden,
        position_embeddings=rope,
        page_table=table,
        batch_size=1,
        fill_seq_lens=[128],
    )
    prefill = _read_ranks(output, (1, 1024, config.hidden_size))[:, :128]
    prefill_pcc = _rows_pcc(prefill, expected[:1, :128])[0]
    for tensor in (hidden, output, *rope):
        tensor.deallocate(True)
    hidden = _tensor(inputs[:1, 128].reshape(1, 1, 1, -1), mesh_device)
    position = _tensor(torch.tensor([128], dtype=torch.int32), mesh_device, integer=True)
    rope = _decode_rope(cos, sin, torch.tensor([128]), layer)
    output = layer.decode_forward(hidden, position_embeddings=rope, current_position=position, page_table=table)
    actual = _read_ranks(output, (1, config.hidden_size))
    result = {"prefill_pcc": prefill_pcc, "decode_pcc": _rows_pcc(actual, expected[:1, 128])[0]}
    path = Path(os.environ["GPT_OSS_120B_RESULTS"]) / "padded-ring.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    for tensor in (hidden, position, table, output, *rope):
        tensor.deallocate(True)
    assert min(result.values()) >= 0.99, result
