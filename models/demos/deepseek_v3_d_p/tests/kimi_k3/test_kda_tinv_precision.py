# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Can any compute config make `prepare_chunk_recurrence`'s t_inv accurate on real Kimi-K3 data?

Attribution so far: with real layer-1 tensors, substituting the device's prepared protocol for the
torch one takes the recurrence from 1.0 to 0.0014, while substituting the device's scan for the
torch scan leaves it at 0.99999. Within prepare, six of seven outputs match at 1.00000 and `t_inv`
matches at 0.99508 — which sounds fine until you notice t_inv is `I + strictly-lower`, so whole-
tensor PCC is dominated by the diagonal. Scored on its strictly-lower part alone it is 0.935.

`invert_doubling` evaluates the exact nilpotent identity
`(I-N)^-1 = (I+N)(I+N^2)(I+N^4)(I+N^8)(I+N^16)`, whose kernel comment says the shorter dependency
chain "is also expected to improve fp32 stability, but that must be validated empirically." On the
real N it is exact in fp32 (max abs error 5e-4) and catastrophic with bf16 intermediates (12.99,
against a true inverse bounded by 1.0), because N^16 reaches O(1e2) while the answer is O(1).

So the question is whether the kernel can be made to hold fp32 through those five matmuls by
configuration alone. If it can, the fix is one line in `ttKDA`; if not, the inversion has to change.
"""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda.ops import causal_depthwise_conv_reference, kda_gate_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import load_kda_layer_state_dict
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.test_prepare_chunk_recurrence import _device_inputs, _run

SEQ = 1024
C = 32


@run_for_blackhole()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_tinv_precision_across_compute_configs(device, device_params):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")

    config = kimi_k3_kda_config()
    w = {k: v.float() for k, v in load_kda_layer_state_dict(Path(checkpoint), 1, config).items()}
    h = trace.rows("kda", "kda_input_layer_0", 0, SEQ).float().unsqueeze(0)
    zero = h.new_zeros(1, config.conv_kernel_size - 1, config.q_dim)

    def conv(n):
        out, _ = causal_depthwise_conv_reference(F.linear(h, w[f"{n}_proj.weight"]), w[f"{n}_conv1d.weight"], zero)
        return out

    q, k, v = conv("q"), conv("k"), conv("v")
    gate = kda_gate_reference(
        F.linear(F.linear(h, w["f_a_proj.weight"]), w["f_b_proj.weight"]).reshape(
            1, SEQ, config.num_heads, config.head_k_dim
        ),
        w["A_log"],
        w["dt_bias"],
        config.gate_lower_bound,
    ).reshape(1, SEQ, config.q_dim)
    beta_flat = torch.sigmoid(F.linear(h, w["b_proj.weight"]))
    beta = beta_flat.reshape(SEQ // C, C, config.num_heads).permute(2, 0, 1).unsqueeze(-1).contiguous()

    # Build N and the exact inverse directly, in fp64, so the reference owes nothing to the op.
    def heads_chunks(t, d):
        return t.reshape(SEQ, config.num_heads, d).permute(1, 0, 2).reshape(config.num_heads, SEQ // C, C, d)

    kh = heads_chunks(k, config.head_k_dim)
    gh = heads_chunks(gate, config.head_k_dim)
    kh = kh * torch.rsqrt(kh.square().sum(-1, keepdim=True) + 1e-6)
    cg = gh.cumsum(dim=2)
    akk = torch.matmul(beta * kh * cg.exp(), (kh * (-cg).exp()).transpose(-1, -2))
    neg_n = torch.tril(akk, diagonal=-1).reshape(-1, C, C)
    eye = torch.eye(C)
    exact = torch.linalg.inv((eye + neg_n).double()).float()
    lower = torch.tril(torch.ones(C, C), diagonal=-1).bool()
    norms = neg_n.abs().sum(-1).max(-1).values
    logger.info(
        f"  real N: ||N||inf median {float(norms.median()):6.3f} max {float(norms.max()):7.3f}; "
        f"exact |t_inv|max {float(exact.abs().max()):6.3f}"
    )

    inputs = (q, k, v, gate, beta)
    device_inputs = _device_inputs(inputs, device)
    configs = {
        "LoFi": ttnn.MathFidelity.LoFi,
        "HiFi2": ttnn.MathFidelity.HiFi2,
        "HiFi4": ttnn.MathFidelity.HiFi4,
    }
    for name, fidelity in configs.items():
        for fp32_acc in (False, True):
            cfg = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_acc,
                packer_l1_acc=False,
            )
            got = ttnn.to_torch(_run(device_inputs, config.num_heads, compute_kernel_config=cfg)[6]).float()
            got = got.reshape(-1, C, C)
            pcc = float(str(comp_pcc(exact[..., lower], got[..., lower], 0.99)[1]).split()[-1])
            logger.info(
                f"    {name:6s} fp32_dest_acc={str(fp32_acc):5s}  t_inv strictly-lower PCC {pcc:9.5f}"
                f"   max|err| {float((exact - got).abs().max()):10.4f}"
            )
