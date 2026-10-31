# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip GDN prefill: Qwen36GatedDeltaNet vs TPGatedDeltaNet at TP=1, then decode from the prefill state.

Prefills one real GDN layer with both modules at each length in LENGTHS (fresh state), checks the outputs
agree, then runs DECODE_STEPS decode steps on each module (TP=1 with the fused recurrence) from the state
the prefill captured and checks those agree too.
    MESH_DEVICE=P150 HF_MODEL=<HF checkpoint dir> pytest models/demos/blackhole/bonsai2/tp1/test_gdn_prefill_tp1.py -s
"""
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.bonsai2.tp1.test_gdn_decode_tp1 import DEVICE_PARAMS, MIN_PCC, _build, _to_dev
from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc

LENGTHS = [128, 2048]
DECODE_STEPS = 4


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", LENGTHS)
def test_gdn_prefill_tp1(mesh_device, seq_len):
    args, li, _, old, tp = _build(mesh_device, max_seq_len=4096)
    tp.use_fused_recurrent_decode = True
    # layer.py sets this for the real model (TP=1 has no mesh axis to all-gather along); the test
    # builds the module directly, so without this it exercises an AGMM path production never takes.
    tp._fuse_agmm = False
    # gdn_chunk_size is set by _init_tp_config, so it lives on the TP=1 args, not the single-device ones.
    chunk = args.tp1_args(mesh_device).gdn_chunk_size

    def host(out, rows):
        return ttnn.to_torch(out).float().reshape(-1, args.dim)[:rows]

    x = torch.randn(1, seq_len, args.dim, dtype=torch.bfloat16)
    x_old, x_tp = (
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device) for _ in range(2)
    )
    o_old = old.forward(x_old, mode="chunk", chunk_size=chunk, valid_len=None)
    # capture_state saves the final rec/conv state for the decode steps below, which requires the
    # persistent buffers: carry_state must be True (tp.py asserts this).
    o_tp = tp.forward_prefill(x_tp, chunk_size=chunk, capture_state=True, carry_state=True)
    prefill_pcc = compute_pcc(host(o_old, seq_len), host(o_tp, seq_len))
    ttnn.deallocate(o_old)
    ttnn.deallocate(o_tp)

    decode_pccs = []
    for _ in range(DECODE_STEPS):
        xd = torch.randn(1, 1, args.dim, dtype=torch.bfloat16)
        d_old = old.forward(_to_dev(mesh_device, xd), mode="recurrent")
        d_tp = tp.forward_decode(_to_dev(mesh_device, xd))
        decode_pccs.append(compute_pcc(host(d_old, 1), host(d_tp, 1)))
        ttnn.deallocate(d_old)
        ttnn.deallocate(d_tp)
    logger.info(
        f"layer {li}, T={seq_len}: prefill PCC {prefill_pcc:.5f}, decode PCC after prefill "
        f"{[round(p, 5) for p in decode_pccs]}"
    )
    assert prefill_pcc > MIN_PCC, f"prefill outputs disagree: {prefill_pcc}"
    assert min(decode_pccs) > MIN_PCC, f"decode from the prefill state disagrees: {decode_pccs}"
