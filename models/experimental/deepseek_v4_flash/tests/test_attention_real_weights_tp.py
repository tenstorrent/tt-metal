# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""TP=4 PCC for the complete DeepSeek-V4-Flash decode attention layer."""

from __future__ import annotations

import contextlib
import os
import types

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.deepseek_v4_flash.tests.test_attention_real_weights import (
    DECODE_PCC_THRESHOLD,
    _DEFAULT_MODEL_DIR,
    _DECODE_STEPS,
    _build_attn_weights,
    _checkpoint_available,
    _generate_reference,
    _reference_path,
    _weight_cache,
)
from models.experimental.deepseek_v4_flash.tt.attention import (
    DeepSeekV4Attention,
    build_static_layer_cache,
    decode_sdpa_bounds,
    int32_pos_tensor,
    make_rope_table,
)
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session

TP_SIZE = 4
PARENT_MESH = (8, 4)
SUBMESH_SHAPE = (1, TP_SIZE)


def _1x4_line_submesh(mesh_device):
    """Carve the TP ethernet line from the opened 8x4 Galaxy mesh."""
    return mesh_device.create_submesh(ttnn.MeshShape(*SUBMESH_SHAPE))


def _to_tt_replicated(tensor: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def _rope_rows(cos_half: torch.Tensor, sin_half: torch.Tensor, device) -> tuple:
    cos, sin = make_rope_table(cos_half, sin_half)
    return (
        _to_tt_replicated(cos, device),
        _to_tt_replicated(sin, device),
        _to_tt_replicated(-sin, device),
    )


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("layer_idx,seq_len", ((1, 2), (5, 128)))
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_attention_real_weights_decode_tp4(mesh_device, reset_seeds, tmp_path, layer_idx: int, seq_len: int) -> None:
    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"need an {PARENT_MESH[0]}x{PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")
    submesh = _1x4_line_submesh(mesh_device)

    batch = 1
    ref_path, need_gen = _reference_path(tmp_path, f"attn_pcc_tp{TP_SIZE}_{layer_idx}_{batch}_{seq_len}")
    if need_gen and not _generate_reference(ref_path, layer_idx, batch, seq_len):
        pytest.skip(f"could not generate HF reference for layer {layer_idx}")

    bundle = torch.load(ref_path, weights_only=False)
    cfg = types.SimpleNamespace(**bundle["config"])
    layer_type = bundle["layer_type"]
    compress_rate = cfg.compress_rates[layer_type] if layer_type != "sliding_attention" else None

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    qkv_tp_strategy = os.environ.get("DEEPSEEK_QKV_TP_STRATEGY", "fused_replicated_full")
    o_b_tp_strategy = os.environ.get("DEEPSEEK_OB_TP_STRATEGY", "row")
    o_a_tp_strategy = os.environ.get("DEEPSEEK_OA_TP_STRATEGY", "sequential")
    use_prefetcher = os.environ.get("DEEPSEEK_USE_PREFETCHER", "1") != "0" and (
        ttnn.experimental.is_tensor_prefetcher_supported(submesh)
    )
    attn = DeepSeekV4Attention(
        cfg,
        layer_idx,
        _build_attn_weights(loader, layer_idx, layer_type),
        submesh,
        cache=_weight_cache(layer_idx),
        weight_dtype=ttnn.bfloat4_b,
        use_prefetcher=use_prefetcher,
        tp_size=TP_SIZE,
        qkv_tp_strategy=qkv_tp_strategy,
        o_b_tp_strategy=o_b_tp_strategy,
        o_a_tp_strategy=o_a_tp_strategy,
    )
    expected_o_b_width = cfg.hidden_size if o_b_tp_strategy == "row" else cfg.hidden_size // TP_SIZE
    assert attn.o_b_proj.N == expected_o_b_width
    kv_cache = build_static_layer_cache(
        submesh,
        cfg.sliding_window,
        layer_type,
        cfg.head_dim,
        seq_len,
        cfg.compress_rates,
        batch=batch,
    )

    hidden = bundle["hidden"]
    reference = bundle["output"].float()
    with contextlib.ExitStack() as prefetcher:
        if use_prefetcher:
            prefetcher.enter_context(tensor_prefetcher_session(submesh))
            ttnn.experimental.wait_for_cq_on_tensor_prefetcher(submesh, cq_id=0)

        for pos in range(seq_len):
            attn.prefetch_weights()
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
            output = attn.decode(
                _to_tt_replicated(hidden[:, pos : pos + 1].reshape(batch, 1, 1, cfg.hidden_size), submesh),
                cos,
                sin,
                neg_sin,
                cos_win,
                sin_win,
                mask,
                kv_cache,
                int32_pos_tensor(pos % cfg.sliding_window, submesh, batch),
                int32_pos_tensor(pos, submesh, batch),
                pool_compressor=pool,
                win_slot=win_slot,
                win_row=win_row,
                sdpa_cur_pos=sdpa_cur_pos,
            )

            if pos < seq_len - min(_DECODE_STEPS, seq_len):
                continue

            # Row-parallel o_b's all-reduce leaves one identical full-hidden
            # copy per TP rank (as does column mode's output all-gather).
            output_torch = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
            output_torch = output_torch[:batch].reshape(batch, 1, cfg.hidden_size).float()
            ref_row = reference[:, pos : pos + 1]
            passing, pcc_message = comp_pcc(ref_row, output_torch, pcc=DECODE_PCC_THRESHOLD)
            logger.info(comp_allclose(ref_row, output_torch))
            logger.info(f"[attention TP{TP_SIZE} {qkv_tp_strategy=} layer {layer_idx} pos {pos}] PCC: {pcc_message}")
            assert passing, f"attention TP{TP_SIZE} PCC < {DECODE_PCC_THRESHOLD}: {pcc_message}"
