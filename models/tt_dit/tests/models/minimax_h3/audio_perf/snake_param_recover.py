"""Is the remaining 4.4e-04 exactly the parameters being rounded to bf16?

After forcing the SrcA format reconfig the snake lands on every channel column, but the error against
the float64 golden sits at 4.4e-04 -- about 2^-9 relative, which is bf16 mantissa precision. If the
parameter tile is reaching DST through SrcA (16/19-bit) rather than unpack-to-dest fp32, then a golden
recomputed with bf16-rounded alpha/inv_beta should match the device to fp32 grade.

Two goldens, one run:
    exact params  -> expected ~4.4e-04 (what we see now)
    bf16 params   -> ~1e-07 if and only if truncation-on-the-parameters is the whole story
"""

import os

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _make_kaiser_sinc_kernel_1d

B, T, C, K = 1, 1024, 32, 7
EPS = 1e-9
TILE = 32


def param_rows(alpha, inv_beta, width):
    rows = []
    for vec in (alpha, inv_beta):
        p = torch.zeros(width, dtype=torch.float32)
        p[: len(vec)] = vec
        rows.append(p.unsqueeze(0).expand(TILE, width).contiguous())
    return torch.cat(rows, dim=0)


device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
try:
    torch.manual_seed(0)
    taps = _make_kaiser_sinc_kernel_1d(0.5 / 2, 0.6 / 2, K).tolist()
    x = torch.randn(B, T, C) * 0.3
    alpha = torch.rand(C) * 0.8 + 0.6
    beta = torch.rand(C) * 0.8 + 0.6
    inv_beta = 1.0 / (beta + EPS)

    wt = torch.tensor(taps, dtype=torch.float32).reshape(1, 1, K).expand(C, 1, K).contiguous()
    weight = ttnn.from_torch(
        wt, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    xr = ttnn.reshape(xd, (B, T, 1, C))
    cc = ttnn.Conv1dConfig(weights_dtype=ttnn.float32, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
    mc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
    )

    def run(w):
        out, _, wb = ttnn.conv1d(
            input_tensor=xr,
            weight_tensor=w,
            device=device,
            in_channels=C,
            out_channels=C,
            batch_size=B,
            input_length=T,
            kernel_size=K,
            stride=1,
            padding=0,
            dilation=1,
            groups=C,
            dtype=ttnn.float32,
            conv_config=cc,
            compute_config=mc,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        prep = wb[0] if isinstance(wb, (tuple, list)) else wb
        return ttnn.to_torch(out).float().reshape(-1, C), ttnn.to_torch(prep).float()

    conv_only, pw = run(weight)
    width = pw.shape[-1]
    pw2 = torch.cat([pw.reshape(-1, width), param_rows(alpha, inv_beta, width)], dim=0)
    pw2 = pw2.reshape(1, 1, pw2.shape[0], width)
    w = ttnn.from_torch(
        pw2, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    os.environ["TT_CONV1D_SNAKE_PARAMS"] = "1"
    try:
        fused, _ = run(w)
    finally:
        del os.environ["TT_CONV1D_SNAKE_PARAMS"]

    # golden conv in float64, then snake with exact vs bf16-rounded parameters
    T_out = conv_only.shape[0]
    gold = torch.zeros(T_out, C, dtype=torch.float64)
    xd64 = x[0].double()
    for n in range(T_out):
        acc = torch.zeros(C, dtype=torch.float64)
        for k in range(K):
            acc += float(taps[K - 1 - k]) * xd64[n + k]
        gold[n] = acc

    def bf16(t):
        return t.to(torch.bfloat16).to(torch.float64)

    # Recover what the kernel actually used, per channel column. With the written alpha, the ratio
    # delta / sin^2(alpha*x) must be a constant equal to the written inv_beta. If alpha is wrong the
    # ratio drifts with x (large std); if only inv_beta is wrong the ratio is constant but offset.
    xs = gold  # float64 golden conv, the exact x the snake should see
    delta = fused.double() - conv_only.double()
    print(
        f"{'col':>3} {'inv_b written':>13} {'ratio mean':>12} {'ratio std':>11} "
        f"{'alpha written':>13} {'rel err':>10}",
        flush=True,
    )
    worst = []
    for c in range(C):
        s2 = torch.sin(alpha[c].double() * xs[:, c]) ** 2
        m = s2 > 1e-3
        if int(m.sum()) < 50:
            continue
        ratio = delta[m, c] / s2[m]
        rel = abs(float(ratio.mean()) - float(inv_beta[c])) / float(inv_beta[c])
        worst.append((rel, c))
        if c < 12:
            print(
                f"{c:3d} {float(inv_beta[c]):13.6f} {float(ratio.mean()):12.6f} "
                f"{float(ratio.std()):11.2e} {float(alpha[c]):13.6f} {rel:10.2e}",
                flush=True,
            )
    worst.sort(reverse=True)
    print(f"\nworst columns by |ratio.mean - inv_beta|/inv_beta:", flush=True)
    for rel, c in worst[:6]:
        print(
            f"  col {c:2d}: rel err {rel:.3e}  alpha={float(alpha[c]):.6f} " f"inv_beta={float(inv_beta[c]):.6f}",
            flush=True,
        )
    print(f"median rel err across columns: {sorted(r for r,_ in worst)[len(worst)//2]:.3e}", flush=True)
finally:
    ttnn.close_mesh_device(device)
print("DIAG6 DONE", flush=True)
