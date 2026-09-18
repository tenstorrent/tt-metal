# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone repro of the Quasar conv2d DFB credit underflow on the LAYER convs.

Companion to test_conv_hang.py, which covers the STEM conv (4x4 over the folded 16-channel
input). This covers the same kernel failing on the ordinary ResNet-50 layer convs -- 3x3 with
stride 2, and 3x3 with stride 1 -- so the two together bracket the problem.

Symptom on ttsim (quasar-release-candidate):

    tensix_semget: sem=0 sem_max=... sem_index=1 sem_sel=0x... pipe=... tile_id=...

i.e. a dataflow-buffer credit popped with no matching post, sem_index=1, issued from the
packer pipe, inside
    ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/conv_bmm_tilize_metal2.cpp

test_conv_hang.py describes the stem failing as a 3-thread MATH<->PACK<->UNPACK cycle over the
matmul-partials compute self-loop DFB. Whether these are one root cause or two instances is
exactly the open question -- on ttsim this manifests as an assert (the credit underflow is
caught) rather than a spin, which is why it reads differently.

What separates pass from fail here: 1x1 stride-1 convs are clean; the failure needs kernel>1 or
stride-2, i.e. the halo / multi-block weight path. test_conv_1x1_control below is the control
and is expected to PASS -- if it fails too, the diagnosis above is wrong.

Setting TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 (tilize and matmul as separate programs) avoids the
failure on all three cases, which is how the ResNet-50 op set currently runs. Retiring that
workaround is the reason for this repro.

Run (emulator or ttsim; slow dispatch + forced JIT):

    TT_METAL_SIMULATOR=<sim.so or emu-config-dir> \
    TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
    TT_METAL_QUASAR_NOC_API_VERSION=1 \
    pytest test_conv_semget_layer.py -k stride2 -x

Expected: stride2 / stride1_3x3 fail in conv_bmm_tilize_metal2; 1x1_control passes.
With TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 all three pass.
"""
import math

import pytest
import torch
import ttnn

# (label, in_ch, out_ch, H, W, kernel, stride, expectation)
# Shapes are taken verbatim from the ResNet-50 op matrix rows noted in the comments.
CASES = [
    # conv2d_019: act 1x1x196x512 (14x14), weight 512x512x3x3, stride 2 -> out 7x7
    ("stride2", 512, 512, 14, 14, (3, 3), (2, 2), "fail"),
    # conv2d_023: act 1x1x49x512 (7x7), weight 512x512x3x3, stride 1
    ("stride1_3x3", 512, 512, 7, 7, (3, 3), (1, 1), "fail"),
    # conv2d_022: act 1x1x49x2048 (7x7), weight 512x2048x1x1, stride 1 -- CONTROL, should pass
    ("1x1_control", 2048, 512, 7, 7, (1, 1), (1, 1), "pass"),
]


def _run_layer_conv(device, in_ch, out_ch, h, w, kernel, stride, batch_size=1):
    padding = (kernel[0] // 2, kernel[1] // 2)

    # Quasar requires HEIGHT_SHARDED activations: quasar::conv2d sets in0_block_w = full_K to
    # contract K in one block, and width/block sharding split K across grid columns, so each
    # core would hold only full_K/ncols and quasar::matmul rejects the shape.
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = compute_grid.x * compute_grid.y
    tensor_height = batch_size * h * w
    shard_height = math.ceil(tensor_height / num_cores) if num_cores else tensor_height
    grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, in_ch),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch.manual_seed(0)
    act_torch = torch.rand((1, 1, tensor_height, in_ch), dtype=torch.bfloat16)
    act = ttnn.from_torch(act_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    act = act.to(device, mem_config)

    weight = ttnn.from_torch(
        torch.rand((out_ch, in_ch, *kernel), dtype=torch.bfloat16), dtype=ttnn.bfloat16
    )
    bias = ttnn.from_torch(torch.rand((1, 1, 1, out_ch), dtype=torch.bfloat16), dtype=ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),  # fused RELU -> fuse_bias path
        deallocate_activation=False,
        reallocate_halo_output=True,
        act_block_h_override=0,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        reshard_if_not_optimal=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        # NB: no fp32_dest_acc_en. The op matrix records it true, but on Quasar it makes the
        # compute kernel consume an FP32 'matmul_partials' DFB and program build refuses it
        # before the conv ever runs -- a different failure that masks this one.
        packer_l1_acc=True,  # packer_l1_acc + fused bias -> the matmul-partials CB
    )

    out, [out_h, out_w], [weight, bias] = ttnn.experimental.quasar.conv2d(
        input_tensor=act,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=in_ch,
        out_channels=out_ch,
        batch_size=batch_size,
        input_height=h,
        input_width=w,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=(1, 1),
        groups=1,
        device=device,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    ttnn.synchronize_device(device)  # the failure is before this returns
    return out, out_h, out_w


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("label,in_ch,out_ch,h,w,kernel,stride,expect", CASES, ids=[c[0] for c in CASES])
@pytest.mark.timeout(1800)  # emulation is slow; cap so a stall doesn't block the suite
def test_layer_conv_semget(mesh_device, label, in_ch, out_ch, h, w, kernel, stride, expect):
    out, out_h, out_w = _run_layer_conv(mesh_device, in_ch, out_ch, h, w, kernel, stride)
    assert out is not None
    assert (out_h, out_w) == (
        (h + 2 * (kernel[0] // 2) - kernel[0]) // stride[0] + 1,
        (w + 2 * (kernel[1] // 2) - kernel[1]) // stride[1] + 1,
    )


# Runnable without pytest as well: tt-metal's pytest.ini sets minversion 7.2, and some
# bring-up venvs are older, which would otherwise make this repro unusable exactly where it
# is needed. Same cases, same config, device opened directly.
#     python3 -u test_conv_semget_layer.py [stride2|stride1_3x3|1x1_control]
if __name__ == "__main__":
    import sys

    only = sys.argv[1] if len(sys.argv) > 1 else None
    # l1_small_size must match the pytest device_params fixture -- the conv allocates an
    # L1_SMALL buffer and the default device opens too small, which fails as an unrelated OOM
    # and hides the actual bug.
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    rc = 0
    try:
        for label, in_ch, out_ch, h, w, kernel, stride, expect in CASES:
            if only and only != label:
                continue
            print(f"[semget] {label}: {in_ch}->{out_ch} {h}x{w} k={kernel} s={stride} expect={expect}", flush=True)
            try:
                _run_layer_conv(dev, in_ch, out_ch, h, w, kernel, stride)
                got = "pass"
                print(f"[semget] {label}: completed", flush=True)
            except Exception as exc:  # noqa: BLE001 - the message is the whole datum
                got = "fail"
                print(f"[semget] {label}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            if got != expect:
                rc = 1
                print(f"[semget] {label}: UNEXPECTED (got {got}, expected {expect})", flush=True)
    finally:
        ttnn.close_device(dev)
    print(f"[semget] overall rc={rc}", flush=True)
    raise SystemExit(rc)
