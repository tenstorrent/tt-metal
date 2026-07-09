# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Component PCC: single-device Qwen3.5-MoE sparse MLP (layer 0) vs torch reference.

Only runs on a MoE checkpoint (``args.moe_num_experts > 0``) — auto-skips on the dense
9B/27B. Export the MoE checkpoint before running, e.g.:

    HF_MODEL=Qwen/Qwen3.6-35B-A3B MESH_DEVICE=P150 \
        pytest models/demos/blackhole/qwen36/tests/unit/test_moe.py -v -s

Loads just one layer's router + fused experts + shared-expert weights (RAM-light) so a
single 35B layer fits on one P150.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tests.test_factory import (
    compute_pcc,
    get_pcc_threshold,
    load_moe_layer,
    torch_moe_reference,
)
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

from .conftest import DEVICE_PARAMS

pytestmark = [run_for_blackhole(), pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)]


@torch.no_grad()
@pytest.mark.parametrize("seq_len", [1, 32], ids=["decode", "prefill"])
def test_moe_pcc(device, seq_len, request):
    args = Qwen36ModelArgs(mesh_device=device)
    if args.moe_num_experts <= 0:
        pytest.skip("not a MoE checkpoint (moe_num_experts == 0)")

    from models.demos.blackhole.qwen36.tt.moe import MoEConfig, Qwen36MoE

    moe_state = load_moe_layer(args.CKPT_DIR, 0)

    x = torch.randn(1, 1, seq_len, args.dim, dtype=torch.bfloat16)
    ref = torch_moe_reference(moe_state, x[0, 0].float(), args.moe_top_k, args.moe_norm_topk_prob)  # [S, dim]

    moe = Qwen36MoE(device, MoEConfig.from_args(args), moe_state, args=args)
    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(moe.forward(x_t))[0, 0].float()  # [S, dim]

    pcc = compute_pcc(ref, out)
    logger.info(f"MoE ({request.node.callspec.id}) PCC = {pcc:.6f}")
    logger.info(f"Ref range [{ref.min():.4f}, {ref.max():.4f}]  TTNN range [{out.min():.4f}, {out.max():.4f}]")
    # bf4 gate/up + top-k routing quantization -> relaxed threshold.
    assert pcc > get_pcc_threshold(request), f"MoE PCC too low: {pcc}"
