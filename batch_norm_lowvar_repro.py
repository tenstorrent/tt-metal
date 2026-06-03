# SPDX-License-Identifier: Apache-2.0
#
# Repro: ttnn.batch_norm(training=True) produces wrong output for inputs with a
# LARGE per-channel mean and SMALL per-channel variance. Root cause: the host
# computes the batch variance as E[x^2] - E[x]^2 (see
# ttnn/cpp/ttnn/operations/normalization/batch_norm/batch_norm.cpp, training
# branch). For large mean / small variance this cancels catastrophically in
# bf16: E[x^2] and E[x]^2 are both ~mean^2, so their difference loses all
# precision -> batch_var ~= 0 or slightly negative -> 1/sqrt(batch_var+eps)
# blows up (output std huge) or = NaN.
#
# Reaches Gemma-4 audio because nn.LayerNorm/RMSNorm lower to
# stablehlo.batch_norm_training -> ttnn.batch_norm, and the conv-subsample
# activations have exactly this large-mean / low-variance profile.
#
# Suggested fix: numerically-stable variance E[(x - mean)^2] instead of
# E[x^2] - E[x]^2.
#
# Run:
#   export TT_METAL_HOME=/home/sshon/tt-metal
#   source /home/sshon/tt-metal/python_env/bin/activate
#   python batch_norm_lowvar_repro.py
import torch
import torch.nn.functional as F
import ttnn
from models.common.utility_functions import comp_pcc  # tt-metal's standard PCC


def run_case(device, name, ch_mean_scale, ch_std):
    torch.manual_seed(0)
    N, C, H, W = 2, 64, 32, 32
    mean = torch.randn(1, C, 1, 1) * ch_mean_scale  # per-channel mean
    x = (mean + ch_std * torch.randn(N, C, H, W)).to(torch.bfloat16)

    def p(t):
        return ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    tt_x = p(x)
    # gamma=1, beta=0, running_mean=0, running_var=1 (the nn.LayerNorm lowering defaults)
    tt_out = ttnn.batch_norm(
        tt_x,
        running_mean=p(torch.zeros(1, C, 1, 1)),
        running_var=p(torch.ones(1, C, 1, 1)),
        weight=p(torch.ones(1, C, 1, 1)),
        bias=p(torch.zeros(1, C, 1, 1)),
        training=True,
        eps=1e-5,
    )
    out = ttnn.to_torch(tt_out).float()
    # CPU reference: per-channel batch_norm (gamma=1, beta=0) over (N, H, W).
    ref = F.batch_norm(x.float(), None, None, torch.ones(C), torch.zeros(C), training=True, eps=1e-5)

    passed, msg = comp_pcc(ref, out, pcc=0.99)  # tt-metal's standard PCC check
    print(
        f"[{name}] mean_scale={ch_mean_scale} ch_std={ch_std}\n"
        f"    comp_pcc passed={passed}  {msg}    (expected pass @ 0.99)\n"
        f"    std: tt={out.std():.3f}  cpu_bn={ref.std():.3f}   (cpu_bn ~= 1.0; tt should match)\n"
    )


def main():
    device = ttnn.open_device(device_id=0)
    try:
        # healthy: small mean, unit variance -> no cancellation -> correct
        run_case(device, "OK   mean~0  std=1.0", ch_mean_scale=0.0, ch_std=1.0)
        # BUG: large mean, small variance -> E[x^2]-E[x]^2 cancels -> wrong / NaN
        run_case(device, "BUG  mean~5  std=0.1", ch_mean_scale=5.0, ch_std=0.1)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
