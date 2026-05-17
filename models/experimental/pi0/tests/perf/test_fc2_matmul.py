"""SigLIP MLP FC2 (intermediate→hidden) — device kernel parity vs torch fp32.

2D K×N parallel kernel on 27 cores (9×3). Shape: M=256, K=4304→pad 4320, N=1152.

K-padding: weight (1152, 4304) → transpose → pad K axis to (4320, 1152).
Activation: Gaussian (256, 4320). The K-padded zeros contribute nothing to
the matmul output (multiply by 0). PCC compares against torch reference
computed on logical (unpadded) operands.

Output is BLOCK_SHARDED 9×3 with the 3 K-rows replicated after the 3-way
N-col reduce. to_torch returns (M*3, N) stacked rows; we take rows [0..M].
"""
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_fc1 import pcc  # noqa: E402

PI05_WEIGHTS = "/storage/sdawle/pi05_weights/pi05_base/model.safetensors"
VP = "paligemma_with_expert.paligemma.model.vision_tower."
M = 256
K_LOGICAL = 4304
K_PADDED = 4320
N = 1152


def load_layer0_fc2_padded() -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (W_fc2_padded [K=4320, N=1152], b_fc2 [N=1152]).

    HF FC2 weight is (N=1152, K=4304); transpose to (K=4304, N=1152) then pad
    K-axis with 16 zero rows to reach K=4320.
    """
    sd = load_file(PI05_WEIGHTS)
    w = sd[f"{VP}vision_model.encoder.layers.0.mlp.fc2.weight"]  # (1152, 4304)
    b = sd[f"{VP}vision_model.encoder.layers.0.mlp.fc2.bias"]  # (1152,)
    w_kn = w.T.contiguous()  # (4304, 1152)
    pad_k = K_PADDED - K_LOGICAL  # 16
    w_padded = torch.cat([w_kn, torch.zeros(pad_k, N, dtype=w_kn.dtype)], dim=0).contiguous()  # (4320, 1152)
    return w_padded, b


def make_fc2_activation_padded(seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic FC2 input. Returns (x_logical [M, K_LOGICAL], x_padded [M, K_PADDED])."""
    g = torch.Generator().manual_seed(seed)
    x_logical = torch.randn(M, K_LOGICAL, generator=g, dtype=torch.bfloat16)
    pad_k = K_PADDED - K_LOGICAL
    x_padded = torch.cat([x_logical, torch.zeros(M, pad_k, dtype=x_logical.dtype)], dim=1).contiguous()
    return x_logical, x_padded


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_device_kernel_pcc(device):
    from fc2_op import SigLIPFC2MatmulOp, build_tensors_for_fc2_test

    w_padded, _ = load_layer0_fc2_padded()
    x_logical, x_padded = make_fc2_activation_padded(seed=42)

    # Torch golden on logical (unpadded) operands.
    w_logical = w_padded[:K_LOGICAL, :]  # (4304, 1152)
    y_golden = (x_logical.float() @ w_logical.float()).to(x_logical.dtype)

    activation_tt, weight_tt, output_tt = build_tensors_for_fc2_test(device, w_padded, x_padded)
    SigLIPFC2MatmulOp.op(activation_tt, weight_tt, output_tt, device)

    import ttnn as _ttnn

    y_full = _ttnn.to_torch(output_tt)  # (M*3, N) — 3 K-row replicas stacked
    # Take row 0's copy; rows [M..2M] and [2M..3M] should be identical.
    y_device = y_full[:M, :]
    p = pcc(y_golden, y_device)
    print(f"\nPCC (FC2, no bias, kernel vs torch) = {p:.6f}")

    # Sanity: confirm the 3 K-row replicas are bit-identical.
    p01 = pcc(y_full[:M, :], y_full[M : 2 * M, :])
    p02 = pcc(y_full[:M, :], y_full[2 * M : 3 * M, :])
    print(f"  K-row replica consistency: row0 vs row1 = {p01:.6f}, row0 vs row2 = {p02:.6f}")

    assert p >= 0.99, f"PCC {p} below 0.99 gate"
