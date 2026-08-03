# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.conv2d`` — every distinct conv shape YOLOv8 runs (yolov8l + yolov8s,
640x640, batch 1).

Imports NO YOLOv8 model code. The shapes below were extracted read-only from the
``sdawle/yolov8_bh`` branch and are reproduced here as raw conv parameters so the op can
be exercised in isolation on Quasar / Blackhole.

Shape source (branch ``origin/sdawle/yolov8_bh``)
-------------------------------------------------
* Conv wrapper — ``models/demos/yolov8l/tt/ttnn_yolov8l.py:231-262`` (``TtConv.__call__``):
  ``ttnn.conv2d(in_channels=x.shape[-1], out_channels=input_params[3],
  kernel_size=(input_params[0],)*2, stride=(input_params[1],)*2,
  padding=(input_params[2],)*2, batch_size=1, input_height/width from the tensor)``.
  So each config entry is ``input_params = [kernel, stride, pad, out_ch, in_ch]``.
* Numeric params — ``models/demos/yolov8{l,s}/tt/configs.json`` groups
  ``conv_config`` / ``c2f_configs`` / ``sppf_configs`` / ``detect_config``.
* Which conv runs where + model.5's inline params — ``ttnn_yolov8l.py:820-970`` (l) and
  ``ttnn_yolov8s.py:591-706`` (s) ``__init__``; forward wiring ``ttnn_yolov8l.py:973-1118``.

Resolution tracing (input H==W per conv)
----------------------------------------
The config does NOT store each conv's feature-map size, so it was traced through the
forward pass. At 640x640 the stride-2 convs halve the side: 640→320→160→80→40→20.
Backbone (each ``model.N``): model.0 @640 → model.1 @320 → c2f model.2 @160 →
model.3 @160 → c2f model.4 @80 → model.5 @80 → c2f model.6 @40 → model.7 @40 →
c2f model.8 @20 → sppf model.9 @20. Neck: upsample(20→40) → c2f model.12 @40 →
upsample(40→80) → c2f model.15 @80 → model.16 @80 (down to 40) → c2f model.18 @40 →
model.19 @40 (down to 20) → c2f model.21 @20. Detect head (``model.22``) runs its three
branches on P3/P4/P5 = {80, 40, 20} (ttnn_yolov8l.py:1109 ``ch=(320,640,640)``,
3 detection levels; standard YOLOv8 strides 8/16/32 at 640).

* A C2f (``TtC2f``, ttnn_yolov8l.py:333-505) is spatially stride-1 throughout, so all its
  sub-convs run at the block's resolution. Its real conv call sites are
  ``cv1_a=input_params[3]``, ``cv1_b=input_params[4]`` (identical params → collapse),
  ``cv2=input_params[1]`` and the bottleneck ``m.cv=input_params[2]`` (bottleneck cv1==cv2
  share params; ``input_params[0]`` is dead config, never passed to ttnn.conv2d — excluded).
* SPPF (``TtSppf``, :507) sub-convs cv1=input_params[0], cv2=input_params[1], both @20.
* Detect (``TtDetect``/``TtDetectCv2``, :550-757): cv2 (box) and cv3 (cls) branches each run
  3 convs (input_params[0..2]) at the branch resolution; the DFL conv (``TtDFL``, :618) is a
  1x1 over the decoded box distribution — see note on that param below.

First conv channel padding (ttnn_yolov8l.py:973-989): the 3-channel image is padded to
16 channels (``min_channels=16``) before ``permute``+conv, so ``model.0`` runs with
in_channels=16 (config's ``in_ch`` field — 3 for l, 8 for s — is the pre-pad value and is
ignored at runtime). ``raw_in_ch`` below carries that pre-pad count so the test zero-pads
exactly as the model does.

Deduplication: 60 conv call sites per variant (120 total) collapse to 69 distinct
``(in_ch, out_ch, kernel, stride, pad, input_h, input_w)`` tuples — 41 first-seen in
yolov8l, 28 more unique to yolov8s. Each param id is ``<variant>-<name>-<res>``.

Reference / tolerance
---------------------
Reference is ``torch.nn.functional.conv2d`` on the same random weights/bias/input (the
canonical conv test ``tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py``
``run_conv`` does exactly this). bf16 activations+weights with LoFi fidelity: PCC ~0.98,
relaxed to ~0.97 for very large reduction depth (in_ch*k*k > 10000), matching run_conv's
own thresholds.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.yolo_ops import op_utils as U

# fmt: off
# (hw, in_ch, out_ch, kernel, stride, padding, input_h, input_w, raw_in_ch)
#   hw       = feature-map side used by tests/yolo_ops/conftest.py for emulator gating.
#   raw_in_ch= pre-pad in_ch for the first conv (3 for l, 8 for s), else 0 (no pad).
# See module docstring for the full shape-source + resolution-trace citation.
CONV_CASES = [
    pytest.param(640, 16, 64, 3, 2, 1, 640, 640, 3, id="l-model.0-640"),
    pytest.param(320, 64, 128, 3, 2, 1, 320, 320, 0, id="l-model.1-320"),
    pytest.param(160, 128, 256, 3, 2, 1, 160, 160, 0, id="l-model.3-160"),
    pytest.param(80, 256, 512, 3, 2, 1, 80, 80, 0, id="l-model.5-80"),
    pytest.param(40, 512, 512, 3, 2, 1, 40, 40, 0, id="l-model.7-40"),
    pytest.param(80, 256, 256, 3, 2, 1, 80, 80, 0, id="l-model.16-80"),
    pytest.param(160, 128, 64, 1, 1, 0, 160, 160, 0, id="l-model.2.cv1_a-160"),
    pytest.param(160, 320, 128, 1, 1, 0, 160, 160, 0, id="l-model.2.cv2-160"),
    pytest.param(160, 64, 64, 3, 1, 1, 160, 160, 0, id="l-model.2.m-160"),
    pytest.param(80, 256, 128, 1, 1, 0, 80, 80, 0, id="l-model.4.cv1_a-80"),
    pytest.param(80, 1024, 256, 1, 1, 0, 80, 80, 0, id="l-model.4.cv2-80"),
    pytest.param(80, 128, 128, 3, 1, 1, 80, 80, 0, id="l-model.4.m-80"),
    pytest.param(40, 512, 256, 1, 1, 0, 40, 40, 0, id="l-model.6.cv1_a-40"),
    pytest.param(40, 2048, 512, 1, 1, 0, 40, 40, 0, id="l-model.6.cv2-40"),
    pytest.param(40, 256, 256, 3, 1, 1, 40, 40, 0, id="l-model.6.m-40"),
    pytest.param(20, 512, 256, 1, 1, 0, 20, 20, 0, id="l-model.8.cv1_a-20"),
    pytest.param(20, 1280, 512, 1, 1, 0, 20, 20, 0, id="l-model.8.cv2-20"),
    pytest.param(20, 256, 256, 3, 1, 1, 20, 20, 0, id="l-model.8.m-20"),
    pytest.param(40, 1024, 256, 1, 1, 0, 40, 40, 0, id="l-model.12.cv1_a-40"),
    pytest.param(40, 1280, 512, 1, 1, 0, 40, 40, 0, id="l-model.12.cv2-40"),
    pytest.param(80, 768, 128, 1, 1, 0, 80, 80, 0, id="l-model.15.cv1_a-80"),
    pytest.param(80, 640, 256, 1, 1, 0, 80, 80, 0, id="l-model.15.cv2-80"),
    pytest.param(40, 768, 256, 1, 1, 0, 40, 40, 0, id="l-model.18.cv1_a-40"),
    pytest.param(20, 1024, 256, 1, 1, 0, 20, 20, 0, id="l-model.21.cv1_a-20"),
    pytest.param(20, 1024, 512, 1, 1, 0, 20, 20, 0, id="l-model.9.cv2-20"),
    pytest.param(80, 256, 64, 3, 1, 1, 80, 80, 0, id="l-model.22.cv2.0.0-80"),
    pytest.param(80, 64, 64, 3, 1, 1, 80, 80, 0, id="l-model.22.cv2.0.1-80"),
    pytest.param(80, 64, 64, 1, 1, 0, 80, 80, 0, id="l-model.22.cv2.0.2-80"),
    pytest.param(80, 256, 256, 3, 1, 1, 80, 80, 0, id="l-model.22.cv3.0.0-80"),
    pytest.param(80, 256, 80, 1, 1, 0, 80, 80, 0, id="l-model.22.cv3.0.2-80"),
    pytest.param(40, 512, 64, 3, 1, 1, 40, 40, 0, id="l-model.22.cv2.1.0-40"),
    pytest.param(40, 64, 64, 3, 1, 1, 40, 40, 0, id="l-model.22.cv2.1.1-40"),
    pytest.param(40, 64, 64, 1, 1, 0, 40, 40, 0, id="l-model.22.cv2.1.2-40"),
    pytest.param(40, 512, 256, 3, 1, 1, 40, 40, 0, id="l-model.22.cv3.1.0-40"),
    pytest.param(40, 256, 80, 1, 1, 0, 40, 40, 0, id="l-model.22.cv3.1.2-40"),
    pytest.param(20, 512, 64, 3, 1, 1, 20, 20, 0, id="l-model.22.cv2.2.0-20"),
    pytest.param(20, 64, 64, 3, 1, 1, 20, 20, 0, id="l-model.22.cv2.2.1-20"),
    pytest.param(20, 64, 64, 1, 1, 0, 20, 20, 0, id="l-model.22.cv2.2.2-20"),
    pytest.param(20, 512, 256, 3, 1, 1, 20, 20, 0, id="l-model.22.cv3.2.0-20"),
    pytest.param(20, 256, 80, 1, 1, 0, 20, 20, 0, id="l-model.22.cv3.2.2-20"),
    # DFL: 1x1 conv over the box distribution. TtDFL (ttnn_yolov8l.py:642-654) reshapes to
    # (b,4,16,a), softmaxes, permutes to (b,4,a,16) => NHWC N=1,H=4,W=a,C=16, then conv2d
    # in=16 out=1 k1s1p0. a = #anchors = 80*80+40*40+20*20 = 8400 @640. Non-square (H=4).
    pytest.param(8400, 16, 1, 1, 1, 0, 4, 8400, 0, id="l-model.22.dfl-8400"),
    pytest.param(640, 16, 32, 3, 2, 1, 640, 640, 3, id="s-model.0-640"),
    pytest.param(320, 32, 64, 3, 2, 1, 320, 320, 0, id="s-model.1-320"),
    pytest.param(160, 64, 128, 3, 2, 1, 160, 160, 0, id="s-model.3-160"),
    pytest.param(80, 128, 256, 3, 2, 1, 80, 80, 0, id="s-model.5-80"),
    pytest.param(40, 256, 512, 3, 2, 1, 40, 40, 0, id="s-model.7-40"),
    pytest.param(80, 128, 128, 3, 2, 1, 80, 80, 0, id="s-model.16-80"),
    pytest.param(40, 256, 256, 3, 2, 1, 40, 40, 0, id="s-model.19-40"),
    pytest.param(160, 64, 32, 1, 1, 0, 160, 160, 0, id="s-model.2.cv1_a-160"),
    pytest.param(160, 96, 64, 1, 1, 0, 160, 160, 0, id="s-model.2.cv2-160"),
    pytest.param(160, 32, 32, 3, 1, 1, 160, 160, 0, id="s-model.2.m-160"),
    pytest.param(80, 128, 64, 1, 1, 0, 80, 80, 0, id="s-model.4.cv1_a-80"),
    pytest.param(40, 256, 128, 1, 1, 0, 40, 40, 0, id="s-model.6.cv1_a-40"),
    pytest.param(40, 128, 128, 3, 1, 1, 40, 40, 0, id="s-model.6.m-40"),
    pytest.param(20, 768, 512, 1, 1, 0, 20, 20, 0, id="s-model.8.cv2-20"),
    pytest.param(40, 768, 128, 1, 1, 0, 40, 40, 0, id="s-model.12.cv1_a-40"),
    pytest.param(40, 384, 256, 1, 1, 0, 40, 40, 0, id="s-model.12.cv2-40"),
    pytest.param(80, 384, 64, 1, 1, 0, 80, 80, 0, id="s-model.15.cv1_a-80"),
    pytest.param(80, 192, 128, 1, 1, 0, 80, 80, 0, id="s-model.15.cv2-80"),
    pytest.param(40, 384, 128, 1, 1, 0, 40, 40, 0, id="s-model.18.cv1_a-40"),
    pytest.param(20, 768, 256, 1, 1, 0, 20, 20, 0, id="s-model.21.cv1_a-20"),
    pytest.param(80, 128, 64, 3, 1, 1, 80, 80, 0, id="s-model.22.cv2.0.0-80"),
    pytest.param(80, 128, 80, 1, 1, 0, 80, 80, 0, id="s-model.22.cv3.0.2-80"),
    pytest.param(40, 256, 64, 3, 1, 1, 40, 40, 0, id="s-model.22.cv2.1.0-40"),
    pytest.param(40, 256, 128, 3, 1, 1, 40, 40, 0, id="s-model.22.cv3.1.0-40"),
    pytest.param(40, 128, 80, 1, 1, 0, 40, 40, 0, id="s-model.22.cv3.1.2-40"),
    pytest.param(20, 512, 128, 3, 1, 1, 20, 20, 0, id="s-model.22.cv3.2.0-20"),
    pytest.param(20, 128, 128, 3, 1, 1, 20, 20, 0, id="s-model.22.cv3.2.1-20"),
    pytest.param(20, 128, 80, 1, 1, 0, 20, 20, 0, id="s-model.22.cv3.2.2-20"),
]
# fmt: on


@U.with_mesh_l1small()  # conv2d allocates from the L1-small region (default l1_small_size=0 OOMs)
@pytest.mark.parametrize(
    "hw, in_ch, out_ch, kernel, stride, padding, input_h, input_w, raw_in_ch",
    CONV_CASES,
)
def test_conv2d(
    ttnn_mesh_device,
    reset_seeds,
    hw,
    in_ch,
    out_ch,
    kernel,
    stride,
    padding,
    input_h,
    input_w,
    raw_in_ch,
):
    mesh = ttnn_mesh_device

    # --- build torch input / weights / bias exactly like run_conv (canonical conv test) ---
    # First conv: create the raw (pre-pad) channels then zero-pad to `in_ch`, mirroring the
    # model's ttnn.pad on the channel dim (ttnn_yolov8l.py:979-986, value=0.0).
    src_ch = raw_in_ch if raw_in_ch else in_ch
    torch_input_nchw = torch.randn(U.BATCH, src_ch, input_h, input_w, dtype=torch.float32)
    if raw_in_ch:
        torch_input_nchw = torch.nn.functional.pad(
            torch_input_nchw, (0, 0, 0, 0, 0, in_ch - raw_in_ch), mode="constant", value=0.0
        )

    torch_weight = torch.randn(out_ch, in_ch, kernel, kernel, dtype=torch.float32)
    torch_bias = torch.randn(1, 1, 1, out_ch, dtype=torch.float32)

    # torch reference (NCHW), symmetric padding on H/W as the model uses (padding,)*2.
    ref_nchw = torch.nn.functional.conv2d(
        torch_input_nchw,
        torch_weight,
        bias=torch_bias.reshape(-1),
        stride=(stride, stride),
        padding=(padding, padding),
        dilation=(1, 1),
        groups=1,
    )
    out_height = ref_nchw.shape[2]
    out_width = ref_nchw.shape[3]
    ref_nhwc = ref_nchw.permute(0, 2, 3, 1).contiguous()  # NHWC, the layout conv2d returns

    # --- ttnn inputs: activation NHWC row-major; weights/bias on host (conv2d prepares them) ---
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1).contiguous()
    tt_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        # let conv2d auto-pick the shard layout (model varies HEIGHT/BLOCK per conv; this
        # op-level test only checks numerics, not the model's exact sharding choice).
        shard_layout=None,
    )
    # Mirror the model's compute config (ttnn_yolov8l.py:203-210): LoFi, no fp32 dest acc.
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    [tt_out, [tt_out_h, tt_out_w], [_, _]] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=in_ch,
        out_channels=out_ch,
        device=mesh,
        bias_tensor=tt_bias,
        kernel_size=(kernel, kernel),
        stride=(stride, stride),
        padding=(padding, padding),
        dilation=(1, 1),
        batch_size=U.BATCH,
        input_height=input_h,
        input_width=input_w,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=1,
        memory_config=None,
        return_weights_and_bias=True,
        return_output_dim=True,
        dtype=ttnn.bfloat16,
    )

    assert (tt_out_h, tt_out_w) == (
        out_height,
        out_width,
    ), f"conv output dims {(tt_out_h, tt_out_w)} != torch {(out_height, out_width)}"

    # conv2d returns a flattened row-major [1, 1, N*H_out*W_out, out_ch] (padded to tiles).
    # Bring to host and reshape to the reference NHWC for PCC.
    out = ttnn.from_device(tt_out)
    out = ttnn.to_torch(out).float()
    out = out.reshape(U.BATCH, out_height, out_width, out.shape[-1])
    out = out[:, :, :, :out_ch].contiguous()

    # bf16 conv @ LoFi: ~0.98; relax for very large reduction depth (matches run_conv).
    pcc = 0.97 if in_ch * kernel * kernel > 10000 else 0.98
    passing, msg = U.comp_pcc(ref_nhwc, out, pcc)
    assert passing, f"conv2d PCC below {pcc}: {msg}"
