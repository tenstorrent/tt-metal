# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP=4 validation for Qwen3.5 GDN chunk-PREFILL on a (1,4) mesh.

Prefill (FIR conv + shared gated_delta_attn_seq chunk kernel) must agree with
the already-validated step-by-step decode over the same token sequence (zero
init state). This is an internal-consistency check across two independent code
paths — no hand-written chunked-recurrence reference needed.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=/home/ttuser/atupe/Qwen27b \
      pytest models/demos/blackhole/qwen3_5_9b/tests/test_gdn_tp_prefill.py -v -s
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen3_5_9b.tests.test_gdn_tp import _load_gdn_layer, _mp
from models.demos.blackhole.qwen3_5_9b.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 4)))],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_gdn_tp_prefill(mesh_device, reset_seeds, ensure_gc):
    mp = _mp()
    os.environ.setdefault("HF_MODEL", mp)
    T = 128
    args = Qwen35ModelArgs(mesh_device, max_batch_size=1, max_seq_len=256)
    nd = mesh_device.get_num_devices()
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(f"devices={nd} gdn layer={li} T={T}")

    sd = _load_gdn_layer(mp, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)

    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=3 if nd > 1 else 0)

    # ---- Prefill ----
    gdn.reset_state()
    out_pf = gdn.forward_prefill(x_tt, chunk_size=128)
    pf = ttnn.to_torch(out_pf, mesh_composer=composer)[0, 0].float()  # [T, dim]

    # ---- Decode the same tokens one at a time ----
    gdn.reset_state()
    dec_rows = []
    for t in range(T):
        xt = ttnn.from_torch(
            x[:, :, t : t + 1, :],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ot = gdn.forward_decode(xt)
        dec_rows.append(ttnn.to_torch(ot, mesh_composer=composer)[0, 0, 0].float())  # [dim]
    dec = torch.stack(dec_rows, dim=0)  # [T, dim]

    from models.common.utility_functions import comp_pcc

    passing, pcc = comp_pcc(dec, pf, 0.95)
    logger.info(f"GDN TP PREFILL vs DECODE PCC (T={T}) = {pcc}")
    assert passing, f"GDN prefill/decode mismatch PCC: {pcc}"
