"""Does the depthwise conv1d honour a fused activation, and is it free?

First increment of Step 2. The conv2d program factory already emits SFPU_OP_INIT_ACTIVATION /
SFPU_OP_FUNC_ACTIVATION from Conv2dConfig::activation, and conv_bmm_tilize.cpp consumes them -- but
compute_depthwise_conv1d.cpp never did, so a fused activation was silently dropped on the depthwise
path. Now wired in all four accumulate paths.

If it works, an activation rides along on the conv output instead of costing a separate op plus the
ROW_MAJOR->TILE->ROW_MAJOR round trip around it. That is the cheapest available piece of the band
fusion, and it needs no new op.

Checks three things:
  * no activation  -> unchanged against a float64 golden (the default path must not move)
  * relu fused     -> matches relu(golden), i.e. the define is actually reaching the kernel
  * cost           -> fused relu against plain conv, to see whether the activation is free
"""

import statistics
import time

import torch

import ttnn
from models.tt_dit.layers.audio_ops import _make_kaiser_sinc_kernel_1d
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

B, C, T_PAD, K, STRIDE = 2, 8, 165606, 7, 1
ITERS = 5


def run(device, x, taps, activation):
    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.float32,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        **({"activation": activation} if activation else {}),
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    wt = torch.tensor(taps, dtype=torch.float32).reshape(1, 1, K).expand(C, 1, K).contiguous()
    weight = ttnn.from_torch(
        wt, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out, _, _ = ttnn.conv1d(
        input_tensor=ttnn.reshape(xd, (B, T_PAD, 1, C)),
        weight_tensor=weight,
        device=device,
        in_channels=C,
        out_channels=C,
        batch_size=B,
        input_length=T_PAD,
        kernel_size=K,
        stride=STRIDE,
        padding=0,
        dilation=1,
        groups=C,
        dtype=ttnn.float32,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    return ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        torch.manual_seed(0)
        taps = _make_kaiser_sinc_kernel_1d(0.5 / 2, 0.6 / 2, K).tolist()
        x = torch.randn(B, T_PAD, C) * 0.3
        t_out = (T_PAD - K) // STRIDE + 1
        xd64 = x.double()
        golden = torch.zeros(B, t_out, C, dtype=torch.float64)
        for k, tap in enumerate(taps):
            golden += float(tap) * xd64[:, k : k + t_out, :]

        # RELU is special-cased by the factory to the packer (`pack_relu`), so it never emits
        # SFPU_OP_INIT_ACTIVATION and cannot exercise this seam. GELU does.
        gelu = ttnn.UnaryWithParam((ttnn.UnaryOpType.GELU, False))
        gelu_ref = torch.nn.functional.gelu(golden)
        for label, act, ref in (("none", None, golden), ("gelu", gelu, gelu_ref)):
            try:
                got = ttnn.to_torch(run(device, x, taps, act)).float().reshape(B, t_out, C)
                err = float((got.double() - ref).pow(2).mean().sqrt() / ref.std())
                fn = lambda: run(device, x, taps, act)
                fn()
                ttnn.synchronize_device(device)
                ts = []
                for _ in range(ITERS):
                    s = time.perf_counter()
                    fn()
                    ttnn.synchronize_device(device)
                    ts.append((time.perf_counter() - s) * 1e3)
                print(f"  activation={label:<5} rel_rmse={err:>10.3e}  {statistics.median(ts):>7.2f} ms")
            except Exception as exc:  # noqa: BLE001
                print(f"  activation={label:<5} FAILED {str(exc).splitlines()[0][:60]}")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
