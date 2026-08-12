"""Does the fused per-channel snake actually compute `x + inv_beta*sin(alpha*x)^2`?

Final step of Step 2a. The kernel (`apply_snake_beta`), the dedicated SNAKE_PARAMS CB and the reader
that fills it are all in and built; nothing has yet fed them real parameters or checked the result.

How the parameters get there, with no op-signature change:

  * conv1d is called once normally to obtain the *prepared* weight tensor
  * two tile-rows are appended to it -- alpha, then inv_beta = 1/(beta+eps) -- each with the
    per-channel value replicated down all 32 rows so the kernel can use plain mul_binary_tile
  * conv1d is called again with that widened weight and TT_CONV1D_SNAKE_PARAMS set

Appending is safe because the weights reader's strides come from the matrix *width*
(`weight_stride_h = weight_matrix_width_ntiles`) and the block height, never from its total height,
and `weight_matrix_height` is read only by a `% TILE_HEIGHT` assertion. The reader pulls the two rows
into the dedicated CB, once for the whole op, and the compute kernel reads them without popping.

Bar: rel_rmse ~1e-07 against a float64 golden, matching what GELU reached through the scalar seam.
"""

import os

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _make_kaiser_sinc_kernel_1d
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

B, T, C, K = 1, 1024, 32, 7
EPS = 1e-9
TILE = 32


def build_conv(device, weight, activation_env: bool):
    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.float32,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    return conv_config, compute_config


def run_conv(device, xr, weight, conv_config, compute_config):
    out, _, w_and_b = ttnn.conv1d(
        input_tensor=xr,
        weight_tensor=weight,
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
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    prepared = w_and_b[0] if isinstance(w_and_b, (tuple, list)) else w_and_b
    return out, prepared


def param_rows(alpha, inv_beta, width_datums):
    """Two tile-rows, each (32, width): the per-channel value replicated down every row."""
    rows = []
    for vec in (alpha, inv_beta):
        padded = torch.zeros(width_datums, dtype=torch.float32)
        padded[: len(vec)] = vec
        rows.append(padded.unsqueeze(0).expand(TILE, width_datums).contiguous())
    return torch.cat(rows, dim=0)  # (64, width)


def main():
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
        cc, mc = build_conv(device, weight, False)

        # 1. plain conv, and capture the prepared weight
        plain_out, prepared = run_conv(device, xr, weight, cc, mc)
        conv_only = ttnn.to_torch(plain_out).float().reshape(-1, C)
        pw = ttnn.to_torch(prepared).float()
        print(f"prepared weight {tuple(pw.shape)}")

        # 2. float64 golden: the same conv, then snake
        T_out = conv_only.shape[0]
        gold = torch.zeros(T_out, C, dtype=torch.float64)
        xd64 = x[0].double()
        for n in range(T_out):
            acc = torch.zeros(C, dtype=torch.float64)
            for k in range(K):
                acc += float(taps[K - 1 - k]) * xd64[n + k]
            gold[n] = acc
        gold_snake = gold + inv_beta.double() * torch.sin(alpha.double() * gold) ** 2
        conv_err = float((conv_only.double() - gold).pow(2).mean().sqrt() / gold.std())
        print(f"conv alone vs golden: rel_rmse {conv_err:.3e}   (sanity: the conv itself must be exact)")

        # 3. widen the prepared weight with the two parameter rows
        width = pw.shape[-1]
        pw2 = torch.cat([pw.reshape(-1, width), param_rows(alpha, inv_beta, width)], dim=0)
        pw2 = pw2.reshape(1, 1, pw2.shape[0], width)
        # Must land on *device*: conv1d treats a host weight as unprepared and validates it must be
        # ROW_MAJOR (prepare_conv2d_weights.cpp:1057). A device tensor is taken as already prepared,
        # which is what we want -- the widened rows must survive untouched.
        widened = ttnn.from_torch(
            pw2,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        print(f"widened weight  {tuple(pw2.shape)}  (+2 tile-rows)")

        # 4. same conv with the fused snake enabled
        os.environ["TT_CONV1D_SNAKE_PARAMS"] = "1"
        try:
            fused_out, _ = run_conv(device, xr, widened, cc, mc)
            fused = ttnn.to_torch(fused_out).float().reshape(-1, C)
        except Exception as exc:  # noqa: BLE001
            print(f"FUSED RUN FAILED: {str(exc).splitlines()[0][:120]}")
            return
        finally:
            del os.environ["TT_CONV1D_SNAKE_PARAMS"]

        err = float((fused.double() - gold_snake).pow(2).mean().sqrt() / gold_snake.std())
        mx = float((fused.double() - gold_snake).abs().max())
        print(f"\nfused snake vs float64 golden: rel_rmse {err:.3e}   maxdiff {mx:.3e}")
        print("PASS (~1e-07 bar)" if err < 1e-6 else "FAIL -- fused output does not match the golden")

        # Is it doing anything at all? If the define never reached the kernel the output would equal
        # the plain conv, which is a distinct failure from computing the wrong thing.
        same_as_conv = float((fused - conv_only).abs().max())
        print(f"differs from plain conv by {same_as_conv:.3e}  (0.0 would mean the fusion never ran)")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
