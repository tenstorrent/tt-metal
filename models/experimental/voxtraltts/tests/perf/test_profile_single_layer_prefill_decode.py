# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-perf: one text decoder layer (layer 0), 128-token prefill + 1 decode step.

Measured between Tracy ``start``/``stop`` signposts. Text stack only (AR KV cache);
acoustic/tokenizer stages use separate perf tests.
"""
from __future__ import annotations

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.voxtraltts.tests.common import create_real_voxtral_text_model_or_skip
from models.experimental.voxtraltts.tt.voxtral_tt_args import voxtral_text_logits_pcc_optimizations
from models.tt_transformers.tt.common import Mode

try:
    from tracy import signpost

    _USE_SIGNPOST = True
except ModuleNotFoundError:
    _USE_SIGNPOST = False

PREFILL_SEQ_LEN = 128  # must be multiple of 128 for prefill attention
DECODE_POS = PREFILL_SEQ_LEN
NUM_WARMUP_ITERS = 1


def _start() -> None:
    if _USE_SIGNPOST:
        signpost(header="start")


def _stop() -> None:
    if _USE_SIGNPOST:
        signpost(header="stop")


def _prefill_layer(model, layer, hidden, *, dim):
    """Run one layer prefill (fills KV cache)."""
    inner = model.inner
    args = inner.args
    seq_len = hidden.shape[1]
    dummy_tokens = torch.zeros(1, seq_len, dtype=torch.int64)
    _, rot_global, rot_local, _, _, _ = model.prepare_inputs_prefill(dummy_tokens, start_pos=0)
    prefill_cfg = args.get_residual_mem_config(Mode.PREFILL, inner.prefetcher)
    x_tt = ttnn.from_torch(
        hidden.reshape(1, 1, seq_len, dim).to(torch.bfloat16),
        device=inner.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=prefill_cfg,
        mesh_mapper=ttnn.ReplicateTensorToMesh(inner.mesh_device),
    )
    return layer(
        x_tt,
        None,
        rot_mats_global=rot_global,
        rot_mats_local=rot_local,
        user_id=0,
        mode=Mode.PREFILL,
        page_table=None,
        kv_cache=None,
    )


def _decode_layer(model, layer, new_hidden, pos_idx, *, dim):
    """Run one layer decode at *pos_idx* (reads KV cache)."""
    inner = model.inner
    args = inner.args
    current_pos_t = torch.tensor([pos_idx], dtype=torch.int64)
    rot_global_d = inner.rope_setup.get_rot_mats(current_pos_t)
    rot_local_d = inner.rope_local_setup.get_rot_mats(current_pos_t) if hasattr(inner, "rope_local_setup") else None
    current_pos_tt = ttnn.from_torch(
        current_pos_t,
        device=inner.mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(inner.mesh_device, dims=(None, None), mesh_shape=args.cluster_shape),
    )
    decode_cfg = args.get_residual_mem_config(Mode.DECODE, inner.prefetcher)
    x_decode = ttnn.from_torch(
        new_hidden.reshape(1, 1, 1, dim).to(torch.bfloat16),
        device=inner.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=decode_cfg,
        mesh_mapper=ttnn.ReplicateTensorToMesh(inner.mesh_device),
    )
    return layer(
        x_decode,
        current_pos_tt,
        rot_mats_global=rot_global_d,
        rot_mats_local=rot_local_d,
        user_id=0,
        mode=Mode.DECODE,
        page_table=None,
        kv_cache=None,
    )


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_profile_single_layer_prefill_decode(device, reset_seeds):
    """Tracy signposted prefill+decode on text layer 0."""
    model = create_real_voxtral_text_model_or_skip(
        device,
        max_seq_len=max(256, DECODE_POS + 1),
        dtype=ttnn.bfloat16,
        optimizations=voxtral_text_logits_pcc_optimizations,
    )
    inner = model.inner
    dim = inner.args.dim
    layer = inner.layers[0]

    hidden_full = (torch.randn(1, PREFILL_SEQ_LEN + 1, dim, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    prompt_hidden = hidden_full[:, :PREFILL_SEQ_LEN, :]
    new_hidden = hidden_full[:, PREFILL_SEQ_LEN, :]

    # Drain profiler after model load so signposted region is not dropped.
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)

    for _ in range(NUM_WARMUP_ITERS):
        pre = _prefill_layer(model, layer, prompt_hidden, dim=dim)
        out = _decode_layer(model, layer, new_hidden, DECODE_POS, dim=dim)
        pre.deallocate(True)
        out.deallocate(True)
        ttnn.synchronize_device(device)

    ttnn.synchronize_device(device)
    _start()
    pre = _prefill_layer(model, layer, prompt_hidden, dim=dim)
    out = _decode_layer(model, layer, new_hidden, DECODE_POS, dim=dim)
    ttnn.synchronize_device(device)
    _stop()
    pre.deallocate(True)
    out.deallocate(True)

    logger.info(
        f"single-layer profile complete: prefill_seq_len={PREFILL_SEQ_LEN}, decode_pos={DECODE_POS}, "
        f"signposts={'on' if _USE_SIGNPOST else 'off'}"
    )
