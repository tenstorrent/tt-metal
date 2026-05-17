"""SigLIP MLP sub-block composition: LN2 → FC1 → GELU → FC2 → residual.

End-to-end orchestrator gluing the K1.1 MLP primitives.

Shape:
  M = 256, D = 1152, intermediate = 4304 (padded to 4320 for tile alignment).
  FC1: (M, D) → (M, 4320)
  GELU: elementwise on (M, 4320)
  FC2: (M, 4320) → (M, D) — 2D K-parallel kernel

Validation: compare against torch fp32 reference of the same sub-block
(LN + FC1 + GELU-tanh + FC2 + residual). Tanh-approximation GELU matches
SigLIP-So400m's gelu_pytorch_tanh.
"""
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_fc1 import pcc  # noqa: E402

M = 256
D = 1152
INTERMEDIATE_TRUE = 4304
INTERMEDIATE_PADDED = 4320


def make_inputs(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, D, generator=g, dtype=torch.bfloat16) * 0.5
    ln_w = torch.ones(D, dtype=torch.bfloat16) + torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.1
    ln_b = torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.05
    # FC1 weight (K=D, N_logical=4304), to be padded to 4320 for device.
    fc1_w_logical = torch.randn(D, INTERMEDIATE_TRUE, generator=g, dtype=torch.bfloat16) * (1.0 / math.sqrt(D))
    # FC2 weight (K_logical=4304, N=D), to be padded to (4320, D) for device.
    fc2_w_logical = torch.randn(INTERMEDIATE_TRUE, D, generator=g, dtype=torch.bfloat16) * (
        1.0 / math.sqrt(INTERMEDIATE_TRUE)
    )
    return x, ln_w, ln_b, fc1_w_logical, fc2_w_logical


def torch_mlp_subblock(x, ln_w, ln_b, fc1_w_logical, fc2_w_logical):
    """Reference MLP sub-block in fp32 (using LOGICAL unpadded weights)."""
    xf = x.float()
    ln_out = F.layer_norm(xf, (D,), ln_w.float(), ln_b.float(), eps=1e-6)
    fc1_out = ln_out @ fc1_w_logical.float()  # (M, 4304)
    gelu_out = F.gelu(fc1_out, approximate="tanh")  # (M, 4304)
    fc2_out = gelu_out @ fc2_w_logical.float()  # (M, D)
    out = xf + fc2_out  # residual
    return out.to(torch.bfloat16)


def device_mlp_subblock(device, x, ln_w, ln_b, fc1_w_logical, fc2_w_logical):
    """Device path through K1.1 MLP primitives. Returns (M, D) bf16."""
    import ttnn
    from layernorm_op import SigLIPLayerNormOp, build_tensors_for_ln_test
    from fc1_op import SigLIPFC1MatmulOp, build_tensors_for_fc1_test
    from gelu_op import SigLIPGeluOp, build_tensors_for_gelu_test
    from fc2_op import SigLIPFC2MatmulOp, build_tensors_for_fc2_test
    from residual_op import SigLIPResidualAddOp, build_tensors_for_residual_test

    def _free(*tensors):
        for t in tensors:
            ttnn.deallocate(t)

    # Pad FC1 weight N axis (D, 4304) → (D, 4320), pad FC2 weight K axis (4304, D) → (4320, D).
    fc1_w_pad = torch.cat(
        [fc1_w_logical, torch.zeros(D, INTERMEDIATE_PADDED - INTERMEDIATE_TRUE, dtype=fc1_w_logical.dtype)],
        dim=1,
    ).contiguous()
    fc2_w_pad = torch.cat(
        [fc2_w_logical, torch.zeros(INTERMEDIATE_PADDED - INTERMEDIATE_TRUE, D, dtype=fc2_w_logical.dtype)],
        dim=0,
    ).contiguous()

    # ============ Stage 1: LN2 ============
    (
        act_tt,
        gamma_tt,
        beta_tt,
        scaler_tt,
        ones_tt,
        accum_tt,
        xmm_tt,
        xmm2_tt,
        mean_tt,
        var_tt,
        ivar_tt,
        ln_out_tt,
    ) = build_tensors_for_ln_test(device, ln_w, ln_b, x, num_cores=8)
    SigLIPLayerNormOp.op(
        act_tt,
        gamma_tt,
        beta_tt,
        scaler_tt,
        ones_tt,
        accum_tt,
        xmm_tt,
        xmm2_tt,
        mean_tt,
        var_tt,
        ivar_tt,
        ln_out_tt,
        num_cores=8,
        eps=1e-6,
    )
    ln_out_host = ttnn.to_torch(ln_out_tt).contiguous()
    _free(act_tt, gamma_tt, beta_tt, scaler_tt, ones_tt, accum_tt, xmm_tt, xmm2_tt, mean_tt, var_tt, ivar_tt, ln_out_tt)

    # ============ Stage 2: FC1 (M=256, K=1152, N=4320 padded) ============
    act_fc1, w_fc1, fc1_out_tt = build_tensors_for_fc1_test(device, fc1_w_pad, ln_out_host, num_cores=27)
    SigLIPFC1MatmulOp.op(act_fc1, w_fc1, fc1_out_tt, num_cores=27)
    fc1_out_host = ttnn.to_torch(fc1_out_tt).contiguous()  # (M, 4320)
    _free(act_fc1, w_fc1, fc1_out_tt)

    # ============ Stage 3: GELU ============
    gelu_in_tt, gelu_out_tt = build_tensors_for_gelu_test(device, fc1_out_host, num_cores=8)
    SigLIPGeluOp.op(gelu_in_tt, gelu_out_tt)
    gelu_out_host = ttnn.to_torch(gelu_out_tt).contiguous()  # (M, 4320)
    _free(gelu_in_tt, gelu_out_tt)

    # ============ Stage 4: FC2 (M=256, K=4320, N=1152) — 2D 9x3 K-parallel ============
    act_fc2, w_fc2, fc2_out_tt = build_tensors_for_fc2_test(device, fc2_w_pad, gelu_out_host)
    SigLIPFC2MatmulOp.op(act_fc2, w_fc2, fc2_out_tt, device)
    fc2_out_full = ttnn.to_torch(fc2_out_tt).contiguous()  # (M*3, D) — 3 K-row replicas stacked
    fc2_out_host = fc2_out_full[:M, :].contiguous()  # take row-0 replica
    _free(act_fc2, w_fc2, fc2_out_tt)

    # ============ Stage 5: Residual add (M, D) ============
    a_tt, b_tt, res_out_tt = build_tensors_for_residual_test(device, fc2_out_host, x, num_cores=8)
    SigLIPResidualAddOp.op(a_tt, b_tt, res_out_tt)
    result = ttnn.to_torch(res_out_tt).contiguous()
    _free(a_tt, b_tt, res_out_tt)
    return result


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_mlp_subblock_synthetic(device):
    """End-to-end MLP sub-block on synthetic weights."""
    x, ln_w, ln_b, fc1_w, fc2_w = make_inputs(seed=42)
    y_golden = torch_mlp_subblock(x, ln_w, ln_b, fc1_w, fc2_w)

    y_device = device_mlp_subblock(device, x, ln_w, ln_b, fc1_w, fc2_w)

    p = pcc(y_golden, y_device)
    print(f"\nPCC (MLP sub-block end-to-end) = {p:.6f}")
    print(f"  M={M}, D={D}, intermediate={INTERMEDIATE_TRUE}→{INTERMEDIATE_PADDED}")
    assert p >= 0.99, f"MLP sub-block PCC {p} below 0.99 gate"
