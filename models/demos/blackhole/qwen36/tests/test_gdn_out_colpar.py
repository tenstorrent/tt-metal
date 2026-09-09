# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN prefill out-projection: column-parallel AG+matmul vs the row-parallel matmul+reduce-scatter.

Runs one real GDN layer's forward_prefill at the PRODUCTION chunk length (T=2048) down both
out-proj arms and PCCs them against each other. test_gdn_tp only covers T=128/256, which does not
exercise the 2048-row AGMM shape the traced prefill actually runs.
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import (
    load_gdn_layer,
    model_path,
    parametrize_mesh_tp,
    shard_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("T", [512, 2048], ids=["T512", "T2048"])
def test_gdn_out_colpar_vs_mmrs(mesh_device, T, reset_seeds, ensure_gc):
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    nd = mesh_device.get_num_devices()
    if nd == 1:
        pytest.skip("TP-only")
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device)
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    assert gdn._out_colpar_prefill, "column-parallel prefill out-proj not active"
    composer = tp_composer(mesh_device)

    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
    x_tt = shard_to_device(mesh_device, x, dim=-1)

    logger.info(f"[colpar] T={T} starting AGMM out-proj arm")
    gdn.reset_state()
    o = gdn.forward_prefill(x_tt, chunk_size=128)
    got = ttnn.to_torch(o, mesh_composer=composer).reshape(T, -1).float()
    ttnn.deallocate(o)
    logger.info(f"[colpar] T={T} AGMM arm OK, out shape {tuple(got.shape)}")

    logger.info(f"[colpar] T={T} starting MMRS reference arm")
    gdn._out_colpar_prefill = False
    gdn.reset_state()
    o2 = gdn.forward_prefill(x_tt, chunk_size=128)
    ref = ttnn.to_torch(o2, mesh_composer=composer).reshape(T, -1).float()
    ttnn.deallocate(o2)
    gdn._out_colpar_prefill = True
    logger.info(f"[colpar] T={T} MMRS arm OK")

    passing, pcc = comp_pcc(ref, got, 0.99)
    logger.info(f"GDN out-proj colpar vs MMRS PCC (T={T}) = {pcc}")
    assert passing, f"colpar/MMRS mismatch at T={T}: {pcc}"
