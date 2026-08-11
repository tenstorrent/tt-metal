"""Which half of the fused snake is wrong: where the parameters are read, or what is done with them?

`sin^2` is bounded by 1, so `x + inv_beta*sin^2(alpha*x)` cannot be inf unless inv_beta itself is inf.
That means either the two tiles are fetched from the wrong place, or DST is being clobbered.

Bisect with parameter values whose answer is known without a golden:

    alpha=0, inv_beta=0   -> y == x   exactly (plain conv)
    alpha=1, inv_beta=0   -> y == x   exactly (alpha must not leak when inv_beta is 0)
    alpha=0, inv_beta=1   -> y == x   exactly (sin(0)=0)
    alpha=1, inv_beta=1   -> y == x + sin(x)^2, constant across channels
    per-channel random    -> the real case

If the all-zero case already comes back inf, the kernel is not reading the tiles we wrote.
If the zero cases are exact and only the constant case is wrong, the arithmetic or DST is at fault.
"""

import os

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _make_kaiser_sinc_kernel_1d

B, T, C, K = 1, 1024, 32, 7
TILE = 32


def param_rows(alpha, inv_beta, width):
    rows = []
    for vec in (alpha, inv_beta):
        padded = torch.zeros(width, dtype=torch.float32)
        padded[: len(vec)] = vec
        rows.append(padded.unsqueeze(0).expand(TILE, width).contiguous())
    return torch.cat(rows, dim=0)


def describe(name, got, expect):
    d = (got.double() - expect.double()).abs()
    finite = torch.isfinite(got)
    tag = f"{name:24s}"
    if int(finite.sum()) != got.numel():
        print(
            f"{tag} NON-FINITE  nan={int(torch.isnan(got).sum())} inf={int(torch.isinf(got).sum())}"
            f" / {got.numel()}",
            flush=True,
        )
        return
    print(
        f"{tag} maxdiff={float(d.max()):.6e}  rel_rmse=" f"{float(d.pow(2).mean().sqrt() / expect.double().std()):.6e}",
        flush=True,
    )


device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
try:
    torch.manual_seed(0)
    taps = _make_kaiser_sinc_kernel_1d(0.5 / 2, 0.6 / 2, K).tolist()
    x = torch.randn(B, T, C) * 0.3
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
    print(f"prepared {tuple(pw.shape)}  conv_only {tuple(conv_only.shape)}", flush=True)

    cases = [
        ("alpha=0 invb=0", torch.zeros(C), torch.zeros(C)),
        ("alpha=1 invb=0", torch.ones(C), torch.zeros(C)),
        ("alpha=0 invb=1", torch.zeros(C), torch.ones(C)),
        ("alpha=1 invb=1", torch.ones(C), torch.ones(C)),
        ("per-channel random", torch.rand(C) * 0.8 + 0.6, 1.0 / (torch.rand(C) * 0.8 + 0.6 + 1e-9)),
    ]

    for name, alpha, inv_beta in cases:
        pw2 = torch.cat([pw.reshape(-1, width), param_rows(alpha, inv_beta, width)], dim=0)
        pw2 = pw2.reshape(1, 1, pw2.shape[0], width)
        widened = ttnn.from_torch(
            pw2,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        os.environ["TT_CONV1D_SNAKE_PARAMS"] = "1"
        try:
            fused, prep_back = run(widened)
        finally:
            del os.environ["TT_CONV1D_SNAKE_PARAMS"]
        # If conv1d re-prepared (and so truncated) the widened weight, the kernel's page ids 7 and 8
        # point past the end of the buffer and it reads whatever DRAM holds -- which is how a bounded
        # sin^2 turns into inf. The shape coming back tells us directly.
        if tuple(prep_back.shape)[-2] != pw2.shape[-2]:
            print(
                f"  !! prepared came back {tuple(prep_back.shape)}, sent {tuple(pw2.shape)}"
                f" -- the widened rows did NOT survive",
                flush=True,
            )
            got_rows = prep_back.reshape(-1, width)
            print(
                f"  !! rows 224..225 of returned prepared: {got_rows[224:226, :4].tolist()}"
                if got_rows.shape[0] > 225
                else "  !! returned prepared has no row 224",
                flush=True,
            )
        expect = conv_only + inv_beta * torch.sin(alpha * conv_only) ** 2
        describe(name, fused, expect)
finally:
    ttnn.close_mesh_device(device)
print("BISECT DONE", flush=True)
