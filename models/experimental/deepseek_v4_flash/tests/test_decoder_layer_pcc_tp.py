# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""TP=4 PCC for the complete DeepSeek-V4-Flash decoder-layer decode path.

This reuses the real-weight HuggingFace reference bundle from
``test_decoder_layer_pcc.py`` and runs ``DeepSeekV4DecoderLayer.decode_static``
on a 1x4 submesh. It covers the same path used by the traced full-model demo:
hyperconnections, norms, tensor-parallel attention, tensor-parallel routed and
shared experts, their all-reduce, residual-stream mixes, and the DRISC weight
prefetcher (the shared 64-core decode GCB; sequential o_a and TP gate/up stay on
the DRAM->L1 copy).
"""

from __future__ import annotations

import contextlib
import types

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.deepseek_v4_flash.tests.test_decoder_layer_pcc import (
    DECODE_PCC_THRESHOLD,
    _DEFAULT_MODEL_DIR,
    _DECODE_STEPS,
    _build_layer_weights,
    _checkpoint_available,
    _expert_provider,
    _generate_reference,
    _reference_path,
    _weight_cache,
)
from models.experimental.deepseek_v4_flash.tt.attention import (
    build_static_layer_cache,
    decode_sdpa_bounds,
    int32_pos_tensor,
    make_rope_table,
)
from models.experimental.deepseek_v4_flash.tt.decoder_layer import DeepSeekV4DecoderLayer
from models.experimental.deepseek_v4_flash.tt.decode_prefetch import make_decode_prefetch_buffers
from models.experimental.deepseek_v4_flash.tt.moe import DeepSeekV4HashRouter, DeepSeekV4PreloadedExperts
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session

TP_SIZE = 4
PARENT_MESH = (8, 4)
SUBMESH_SHAPE = (1, TP_SIZE)


def _to_tt_replicated(tensor: torch.Tensor, device: ttnn.MeshDevice) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def _rope_rows(cos_half: torch.Tensor, sin_half: torch.Tensor, device: ttnn.MeshDevice) -> tuple:
    cos, sin = make_rope_table(cos_half, sin_half)
    return (
        _to_tt_replicated(cos, device),
        _to_tt_replicated(sin, device),
        _to_tt_replicated(-sin, device),
    )


def _token_row(input_ids: torch.Tensor, device: ttnn.MeshDevice) -> ttnn.Tensor:
    return ttnn.from_torch(
        input_ids.reshape(1, -1).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@torch.no_grad()
def test_tp4_pipeline_handoff_replicates_all_ranks(mesh_device, reset_seeds) -> None:
    """Repeated full-model socket handoffs preserve one residual copy per TP rank."""
    sender = mesh_device.create_submesh(ttnn.MeshShape(*SUBMESH_SHAPE), ttnn.MeshCoordinate(0, 0))
    receiver = mesh_device.create_submesh(ttnn.MeshShape(*SUBMESH_SHAPE), ttnn.MeshCoordinate(1, 0))
    connections = []
    for coord in ttnn.MeshCoordinateRange(sender.shape):
        for core in (ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)):
            mesh_core = ttnn.MeshCoreCoord(coord, core)
            connections.append(ttnn.SocketConnection(mesh_core, mesh_core))
    socket_config = ttnn.SocketConfig(
        connections,
        ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 16384),
    )
    send_socket, recv_socket = ttnn.create_socket_pair(sender, receiver, socket_config)

    for step in range(64):
        reference = torch.full((1, 1, 4, 4096), float(step), dtype=torch.bfloat16)
        source = _to_tt_replicated(reference, sender)
        source_rm = ttnn.to_layout(source, ttnn.ROW_MAJOR_LAYOUT)
        output = ttnn.allocate_tensor_on_device(source_rm.spec, receiver)
        ttnn.experimental.send_direct_async(source_rm, send_socket)
        ttnn.experimental.recv_direct_async(output, recv_socket)

        copies = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(receiver, dim=0))
        for rank, rank_output in enumerate(copies.reshape(TP_SIZE, *reference.shape)):
            torch.testing.assert_close(
                rank_output,
                reference,
                rtol=0,
                atol=0,
                msg=lambda msg: f"step {step}, TP rank {rank}: {msg}",
            )
        source.deallocate(True)
        source_rm.deallocate(True)
        output.deallocate(True)


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize(
    "layer_idx,seq_len",
    ((0, 128), (2, 128), (5, 128)),
    ids=["sliding_hash_layer0", "csa_hash_layer2", "hca_learned_layer5"],
)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_decoder_layer_decode_static_pcc_tp4(mesh_device, reset_seeds, tmp_path, layer_idx: int, seq_len: int) -> None:
    """Compare the traced decoder-layer path against the real HF layer."""
    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"need an {PARENT_MESH[0]}x{PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")

    submesh = mesh_device.create_submesh(ttnn.MeshShape(*SUBMESH_SHAPE), ttnn.MeshCoordinate(0, 0))
    batch = 1
    ref_path, need_gen = _reference_path(tmp_path, f"decoder_layer_tp{TP_SIZE}_{layer_idx}_{batch}_{seq_len}")
    if not need_gen:
        cached = torch.load(ref_path, weights_only=False)
        need_gen = "input_ids" not in cached or "mlp_layer_types" not in cached["config"]
    if need_gen and not _generate_reference(ref_path, layer_idx, batch, seq_len):
        pytest.skip(f"could not generate HF reference for layer {layer_idx}")

    bundle = torch.load(ref_path, weights_only=False)
    cfg = types.SimpleNamespace(**bundle["config"])
    layer_type = bundle["layer_type"]
    compress_rate = cfg.compress_rates[layer_type] if layer_type != "sliding_attention" else None

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    cache = _weight_cache(layer_idx)
    weights = _build_layer_weights(loader, layer_idx, layer_type)
    is_hash = cfg.mlp_layer_types[layer_idx] == "hash_moe"
    use_prefetcher = ttnn.experimental.is_tensor_prefetcher_supported(submesh)
    prefetch_buffers = None
    if use_prefetcher:
        # Same shallower ring as the TP1 decoder-layer PCC: the 256-token cache sits in
        # L1 beside the weights, and the production 16-page depth leaves SDPA nowhere to
        # sit. One mapping is passed into attention and MoE so they share the 64-core GCB.
        prefetch_buffers = make_decode_prefetch_buffers(submesh, ttnn.bfloat4_b, num_prefetch_pages=8)
    gate = None
    if is_hash:
        gate = DeepSeekV4HashRouter(
            cfg,
            {
                "gate.weight": weights["mlp.gate.weight"],
                "gate.tid2eid": loader.get_tensor(f"layers.{layer_idx}.mlp.gate.tid2eid").long(),
            },
            submesh,
            cache=cache.sub("mlp") if cache else None,
            use_prefetcher=use_prefetcher,
            prefetch_buffers=prefetch_buffers,
            weight_dtype=ttnn.bfloat4_b,
        )
    experts = DeepSeekV4PreloadedExperts(
        cfg,
        _expert_provider(loader, layer_idx),
        submesh,
        dtype=ttnn.bfloat4_b,
        cache=cache.sub("mlp") if cache else None,
        tp_size=TP_SIZE,
    )
    layer = DeepSeekV4DecoderLayer(
        cfg,
        layer_idx,
        weights,
        submesh,
        experts=experts,
        gate=gate,
        cache=cache,
        weight_dtype=ttnn.bfloat4_b,
        use_prefetcher=use_prefetcher,
        prefetch_buffers=prefetch_buffers,
        tp_size=TP_SIZE,
    )
    assert layer.self_attn.tp_size == TP_SIZE
    assert layer.mlp.tp_size == TP_SIZE
    assert layer.mlp.experts.tp_size == TP_SIZE

    streams = bundle["streams"]
    reference = bundle["output"].to(torch.float32)
    split = seq_len - 32
    assert split % 32 == 0 and split + _DECODE_STEPS <= seq_len
    kv_cache = build_static_layer_cache(
        submesh,
        cfg.sliding_window,
        layer_type,
        cfg.head_dim,
        seq_len,
        cfg.compress_rates,
        batch=batch,
    )

    with contextlib.ExitStack() as prefetcher:
        if use_prefetcher:
            prefetcher.enter_context(tensor_prefetcher_session(submesh))
            ttnn.experimental.wait_for_cq_on_tensor_prefetcher(submesh, cq_id=0)
        for pos in range(split + _DECODE_STEPS):
            layer.prefetch_weights()
            cos, sin, neg_sin = _rope_rows(
                bundle["cos_q"][pos : pos + 1],
                bundle["sin_q"][pos : pos + 1],
                submesh,
            )
            cos_win = sin_win = win_slot = win_row = None
            pool = False
            if compress_rate is not None:
                window = max((pos + 1) // compress_rate - 1, 0)
                pool = (pos + 1) % compress_rate == 0
                win_cos, win_sin = make_rope_table(
                    bundle["cos_win"][window : window + 1],
                    bundle["sin_win"][window : window + 1],
                )
                cos_win = _to_tt_replicated(win_cos, submesh)
                sin_win = _to_tt_replicated(win_sin, submesh)
                win_slot = int32_pos_tensor(pos % compress_rate, submesh, batch)
                win_row = int32_pos_tensor(cfg.sliding_window + window, submesh, batch)

            mask, sdpa_cur_pos = decode_sdpa_bounds(
                cfg.sliding_window, layer_type, compress_rate, pos, seq_len, submesh, batch
            )
            out_tt = layer.decode_static(
                _to_tt_replicated(streams[:, pos : pos + 1], submesh),
                cos,
                sin,
                neg_sin,
                cos_win,
                sin_win,
                mask,
                kv_cache,
                int32_pos_tensor(pos % cfg.sliding_window, submesh, batch),
                int32_pos_tensor(pos, submesh, batch),
                hash_token=_token_row(bundle["input_ids"][:, pos : pos + 1], submesh) if is_hash else None,
                pool_compressor=pool,
                win_slot=win_slot,
                win_row=win_row,
                sdpa_cur_pos=sdpa_cur_pos,
            )
            if pos < split:
                continue

            ref_row = reference[:, pos : pos + 1]
            out_torch = ttnn.to_torch(
                out_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
            )
            rank_outputs = out_torch.reshape(TP_SIZE, *ref_row.shape).to(torch.float32)
            for rank, rank_output in enumerate(rank_outputs):
                passing, pcc_message = comp_pcc(ref_row, rank_output, pcc=DECODE_PCC_THRESHOLD)
                logger.info(comp_allclose(ref_row, rank_output))
                logger.info(f"[decoder layer TP{TP_SIZE} rank {rank} layer {layer_idx} pos {pos}] PCC: {pcc_message}")
                assert passing, (
                    f"decoder layer TP{TP_SIZE} rank {rank} layer {layer_idx} pos {pos} "
                    f"PCC < {DECODE_PCC_THRESHOLD}: {pcc_message}"
                )
