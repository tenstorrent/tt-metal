# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Per-call-site PCC test for every ttnn.conv2d the tt-forge ResNet-50 graph issues, run on Quasar via
ttnn.experimental.quasar.conv2d.

53 cases -- one per conv in the graph, which is exactly the 53 convs of torchvision ResNet-50
(1 stem + 16 bottlenecks x 3 + 4 downsample). Every case uses the configuration the FORGE compiler
chose, not a hand-picked one: channels, spatial dims, kernel, stride, the fused activation, the
Conv2dConfig flags, the l1_full slice config, MathFidelity.HiFi2, and the FULL input + output memory
configs (buffer, page layout, shard shape, core ranges, ROW_MAJOR orientation). The tables below are
transcribed from the TTNN IR, cross-checked against the EmitPy render of the same module
(proof/emitpy/B_ttnn_route.py) and re-derived from the torchvision topology by the first test in
this file.

WHAT THE FORGE CONFIGS LOOK LIKE
  stem     7x7 s2 p3, 3 -> 64 @224x224, HEIGHT_SHARDED out over 56 cores, shard [224, 64].
           Its INPUT is L1 INTERLEAVED (the reshaped, tilized image) -- the only conv fed
           interleaved; every other conv is fed a sharded tensor.
  layer1   HEIGHT_SHARDED, 49 cores (8x6 + 1), shard [64, C] @56x56.
           conv1 and downsample take the max_pool2d output, which is ROW_MAJOR -- the only
           two convs with a row-major activation. Everything else is TILE.
  layer2   HEIGHT_SHARDED, 25 cores (8x3 + 1), shard [32, C] @28x28.
  layer3   BLOCK_SHARDED, 56 cores (8x7) @14x14 -- except layer3.0.conv1, which is still
           HEIGHT_SHARDED: the HS -> BS reshard happens in the two to_memory_config ops
           after it (see test_to_memory_config_forge.py), so layer3.0.conv2 and
           layer3.0.downsample are fed the [128, x] block-sharded reshard results.
  layer4   BLOCK_SHARDED, 56 cores @14x14 then 16 cores (8x2) @7x7.
  all 53   batch 1, bf16 in/out/weights, MathFidelity.HiFi2, output in L1,
           act_block_h_override=0, slice config l1_full/0, dilation 1, groups 1,
           symmetric padding = kernel // 2, enable_kernel_stride_folding=False.
  33 of the 53 convs carry a FUSED RELU. The other 20 are the 16 bottleneck conv3s and the 4
  downsamples, whose output feeds a residual add -- there the relu is a separate op after the add.

WEIGHTS
  Forge pre-prepares its weights in const-eval functions: prepare_conv2d_weights turns the OIHW
  host weight into [1, 1, in_ch*kh*kw, out_ch]. Here the RAW OIHW weight is handed to
  quasar.conv2d, which prepares it internally, and the test asserts the shape it produced equals
  the shape Forge's prepare_conv2d_weights produced -- so the two weight-prep paths are checked
  to agree. Quasar exposes neither prepare entry point.

LAYOUT CONVERSION
  Forge hands conv2d a channels-last flattened activation [1, 1, N*H*W, C]; the torch golden is
  NCHW. So the input is permuted NCHW -> NHWC and flattened, and the ttnn output is reshaped back
  to [N, OH, OW, C], channel padding sliced off, and permuted to NCHW before the PCC check.

RUN
  full sweep (53 call-sites):
    pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe/test_conv2d_forge.py
  one test per DISTINCT config (24 of the 53; the rest are byte-identical repeats):
    pytest -s ... test_conv2d_forge.py -k "not dup"
  one layer / just the table check:
    pytest -s ... test_conv2d_forge.py -k "L3"
    pytest ... test_conv2d_forge.py -k topology        # host-only, no device

KNOWN QUASAR ISSUES THESE WILL WALK INTO (pre-existing, see ../ops/)
  * the fused conv_bmm_tilize path -- every kernel > 1 conv, i.e. the 7x7 stem and all 3x3
    convs -- has deadlocked on Quasar in conv_bmm_tilize_metal2 (../test_conv_hang.py).
    Those cases can HANG rather than fail, hence the module timeout.
  * the layer3/layer4 BLOCK_SHARDED convs (512->1024, 1024->2048) have overflowed the
    uint16_t weights-DFB ring. Note Forge PINS its own block-sharded layout here rather than
    letting the op reshard, a different setup from ../ops/test_conv2d.py.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.97  # bf16 activations + weights, MathFidelity.HiFi2

# --- Forge's memory configs, verbatim from the TTNN IR --------------------------------------------
# Core-range sets, inclusive corners. Every Forge shard spec is L1 with ROW_MAJOR orientation.
CR56 = (((0, 0), (7, 6)),)  # 8x7     = 56 cores
CR49 = (((0, 0), (7, 5)), ((0, 6), (0, 6)))  # 8x6 + 1 = 49 cores
CR25 = (((0, 0), (7, 2)), ((0, 3), (0, 3)))  # 8x3 + 1 = 25 cores
CR16 = (((0, 0), (7, 1)),)  # 8x2     = 16 cores

# (memory_layout, buffer_type, core_ranges, shard_shape, page_layout)
# fmt: off
L1_IL         = ("INTERLEAVED",    "L1", None, None,       "TILE")
HS56_224x64   = ("HEIGHT_SHARDED", "L1", CR56, (224,  64), "TILE")
HS56_56x64_RM = ("HEIGHT_SHARDED", "L1", CR56, ( 56,  64), "ROW_MAJOR")  # the max_pool2d output
HS49_64x64    = ("HEIGHT_SHARDED", "L1", CR49, ( 64,  64), "TILE")
HS49_64x128   = ("HEIGHT_SHARDED", "L1", CR49, ( 64, 128), "TILE")
HS49_64x256   = ("HEIGHT_SHARDED", "L1", CR49, ( 64, 256), "TILE")
HS25_32x128   = ("HEIGHT_SHARDED", "L1", CR25, ( 32, 128), "TILE")
HS25_32x256   = ("HEIGHT_SHARDED", "L1", CR25, ( 32, 256), "TILE")
HS25_32x512   = ("HEIGHT_SHARDED", "L1", CR25, ( 32, 512), "TILE")
BS56_32x32    = ("BLOCK_SHARDED",  "L1", CR56, ( 32,  32), "TILE")
BS56_32x64    = ("BLOCK_SHARDED",  "L1", CR56, ( 32,  64), "TILE")
BS56_32x128   = ("BLOCK_SHARDED",  "L1", CR56, ( 32, 128), "TILE")
BS56_128x32   = ("BLOCK_SHARDED",  "L1", CR56, (128,  32), "TILE")  # layer2 -> layer3 reshard out
BS56_128x64   = ("BLOCK_SHARDED",  "L1", CR56, (128,  64), "TILE")  # layer2 -> layer3 reshard out
BS16_32x64    = ("BLOCK_SHARDED",  "L1", CR16, ( 32,  64), "TILE")
BS16_32x256   = ("BLOCK_SHARDED",  "L1", CR16, ( 32, 256), "TILE")

# (idx, tag,                       ic,   oc,  hw, k, s, relu,  deall, input_mem,      output_mem,    dup_of)
# hw is both input_height and input_width; padding is k // 2 on all four sides.
# `relu` is Conv2dConfig.activation, `deall` is Conv2dConfig.deallocate_activation.
CONV_CASES = [
    ( 0, "conv1",                   3,   64, 224, 7, 2, True,  True,  L1_IL,          HS56_224x64,   None ),
    ( 1, "layer1.0.conv1",         64,   64,  56, 1, 1, True,  False, HS56_56x64_RM,  HS49_64x64,    None ),
    ( 2, "layer1.0.conv2",         64,   64,  56, 3, 1, True,  True,  HS49_64x64,     HS49_64x64,    None ),
    ( 3, "layer1.0.conv3",         64,  256,  56, 1, 1, False, True,  HS49_64x64,     HS49_64x256,   None ),
    ( 4, "layer1.0.downsample",    64,  256,  56, 1, 1, False, False, HS56_56x64_RM,  HS49_64x256,   None ),
    ( 5, "layer1.1.conv1",        256,   64,  56, 1, 1, True,  False, HS49_64x256,    HS49_64x64,    None ),
    ( 6, "layer1.1.conv2",         64,   64,  56, 3, 1, True,  True,  HS49_64x64,     HS49_64x64,    2    ),
    ( 7, "layer1.1.conv3",         64,  256,  56, 1, 1, False, True,  HS49_64x64,     HS49_64x256,   3    ),
    ( 8, "layer1.2.conv1",        256,   64,  56, 1, 1, True,  False, HS49_64x256,    HS49_64x64,    5    ),
    ( 9, "layer1.2.conv2",         64,   64,  56, 3, 1, True,  True,  HS49_64x64,     HS49_64x64,    2    ),
    (10, "layer1.2.conv3",         64,  256,  56, 1, 1, False, True,  HS49_64x64,     HS49_64x256,   3    ),
    (11, "layer2.0.conv1",        256,  128,  56, 1, 1, True,  False, HS49_64x256,    HS49_64x128,   None ),
    (12, "layer2.0.conv2",        128,  128,  56, 3, 2, True,  True,  HS49_64x128,    HS25_32x128,   None ),
    (13, "layer2.0.conv3",        128,  512,  28, 1, 1, False, True,  HS25_32x128,    HS25_32x512,   None ),
    (14, "layer2.0.downsample",   256,  512,  56, 1, 2, False, False, HS49_64x256,    HS25_32x512,   None ),
    (15, "layer2.1.conv1",        512,  128,  28, 1, 1, True,  False, HS25_32x512,    HS25_32x128,   None ),
    (16, "layer2.1.conv2",        128,  128,  28, 3, 1, True,  True,  HS25_32x128,    HS25_32x128,   None ),
    (17, "layer2.1.conv3",        128,  512,  28, 1, 1, False, True,  HS25_32x128,    HS25_32x512,   13   ),
    (18, "layer2.2.conv1",        512,  128,  28, 1, 1, True,  False, HS25_32x512,    HS25_32x128,   15   ),
    (19, "layer2.2.conv2",        128,  128,  28, 3, 1, True,  True,  HS25_32x128,    HS25_32x128,   16   ),
    (20, "layer2.2.conv3",        128,  512,  28, 1, 1, False, True,  HS25_32x128,    HS25_32x512,   13   ),
    (21, "layer2.3.conv1",        512,  128,  28, 1, 1, True,  False, HS25_32x512,    HS25_32x128,   15   ),
    (22, "layer2.3.conv2",        128,  128,  28, 3, 1, True,  True,  HS25_32x128,    HS25_32x128,   16   ),
    (23, "layer2.3.conv3",        128,  512,  28, 1, 1, False, True,  HS25_32x128,    HS25_32x512,   13   ),
    (24, "layer3.0.conv1",        512,  256,  28, 1, 1, True,  False, HS25_32x512,    HS25_32x256,   None ),
    (25, "layer3.0.conv2",        256,  256,  28, 3, 2, True,  True,  BS56_128x32,    BS56_32x32,    None ),
    (26, "layer3.0.conv3",        256, 1024,  14, 1, 1, False, True,  BS56_32x32,     BS56_32x128,   None ),
    (27, "layer3.0.downsample",   512, 1024,  28, 1, 2, False, True,  BS56_128x64,    BS56_32x128,   None ),
    (28, "layer3.1.conv1",       1024,  256,  14, 1, 1, True,  False, BS56_32x128,    BS56_32x32,    None ),
    (29, "layer3.1.conv2",        256,  256,  14, 3, 1, True,  True,  BS56_32x32,     BS56_32x32,    None ),
    (30, "layer3.1.conv3",        256, 1024,  14, 1, 1, False, True,  BS56_32x32,     BS56_32x128,   26   ),
    (31, "layer3.2.conv1",       1024,  256,  14, 1, 1, True,  False, BS56_32x128,    BS56_32x32,    28   ),
    (32, "layer3.2.conv2",        256,  256,  14, 3, 1, True,  True,  BS56_32x32,     BS56_32x32,    29   ),
    (33, "layer3.2.conv3",        256, 1024,  14, 1, 1, False, True,  BS56_32x32,     BS56_32x128,   26   ),
    (34, "layer3.3.conv1",       1024,  256,  14, 1, 1, True,  False, BS56_32x128,    BS56_32x32,    28   ),
    (35, "layer3.3.conv2",        256,  256,  14, 3, 1, True,  True,  BS56_32x32,     BS56_32x32,    29   ),
    (36, "layer3.3.conv3",        256, 1024,  14, 1, 1, False, True,  BS56_32x32,     BS56_32x128,   26   ),
    (37, "layer3.4.conv1",       1024,  256,  14, 1, 1, True,  False, BS56_32x128,    BS56_32x32,    28   ),
    (38, "layer3.4.conv2",        256,  256,  14, 3, 1, True,  True,  BS56_32x32,     BS56_32x32,    29   ),
    (39, "layer3.4.conv3",        256, 1024,  14, 1, 1, False, True,  BS56_32x32,     BS56_32x128,   26   ),
    (40, "layer3.5.conv1",       1024,  256,  14, 1, 1, True,  False, BS56_32x128,    BS56_32x32,    28   ),
    (41, "layer3.5.conv2",        256,  256,  14, 3, 1, True,  True,  BS56_32x32,     BS56_32x32,    29   ),
    (42, "layer3.5.conv3",        256, 1024,  14, 1, 1, False, True,  BS56_32x32,     BS56_32x128,   26   ),
    (43, "layer4.0.conv1",       1024,  512,  14, 1, 1, True,  False, BS56_32x128,    BS56_32x64,    None ),
    (44, "layer4.0.conv2",        512,  512,  14, 3, 2, True,  True,  BS56_32x64,     BS16_32x64,    None ),
    (45, "layer4.0.conv3",        512, 2048,   7, 1, 1, False, True,  BS16_32x64,     BS16_32x256,   None ),
    (46, "layer4.0.downsample",  1024, 2048,  14, 1, 2, False, False, BS56_32x128,    BS16_32x256,   None ),
    (47, "layer4.1.conv1",       2048,  512,   7, 1, 1, True,  False, BS16_32x256,    BS16_32x64,    None ),
    (48, "layer4.1.conv2",        512,  512,   7, 3, 1, True,  True,  BS16_32x64,     BS16_32x64,    None ),
    (49, "layer4.1.conv3",        512, 2048,   7, 1, 1, False, True,  BS16_32x64,     BS16_32x256,   45   ),
    (50, "layer4.2.conv1",       2048,  512,   7, 1, 1, True,  False, BS16_32x256,    BS16_32x64,    47   ),
    (51, "layer4.2.conv2",        512,  512,   7, 3, 1, True,  True,  BS16_32x64,     BS16_32x64,    48   ),
    (52, "layer4.2.conv3",        512, 2048,   7, 1, 1, False, True,  BS16_32x64,     BS16_32x256,   45   ),
]
# fmt: on


def _mem(spec):
    """Frozen Forge memory-config tuple -> a real ttnn.MemoryConfig."""
    memory_layout, buffer_type, core_ranges, shard_shape, _page_layout = spec
    layout = getattr(ttnn.TensorMemoryLayout, memory_layout)
    buffer = getattr(ttnn.BufferType, buffer_type)
    if core_ranges is None:
        return ttnn.MemoryConfig(layout, buffer, None)
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*lo), ttnn.CoreCoord(*hi)) for lo, hi in core_ranges])
    spec = ttnn.ShardSpec(ranges, list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(layout, buffer, spec)


def _page(spec):
    return ttnn.TILE_LAYOUT if spec[4] == "TILE" else ttnn.ROW_MAJOR_LAYOUT


def _to_device(x, spec, device):
    """Host tensor -> device tensor in the exact Forge page layout + memory config."""
    tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=_page(spec))
    try:
        return tt.to(device, _mem(spec))
    except Exception as e:
        raise AssertionError(
            "could not place a %s tensor in the Forge memory config (%s/%s/%s, shard %s over %s): %s"
            % (tuple(x.shape), spec[1], spec[0], spec[4], spec[3], spec[2], e)
        ) from e


def _require_grid(device, *specs):
    """Skip unless the device compute grid can hold every Forge core range in play."""
    grid = device.compute_with_storage_grid_size()
    for spec in specs:
        if spec[2] is None:
            continue
        need_x = max(hi[0] for _lo, hi in spec[2]) + 1
        need_y = max(hi[1] for _lo, hi in spec[2]) + 1
        if need_x > grid.x or need_y > grid.y:
            pytest.skip(
                "Forge pins a %dx%d core grid but this device grid is %dx%d. These configs need a "
                "full Quasar part; ../ops/ covers the same kernels with device-derived sharding."
                % (need_x, need_y, grid.x, grid.y)
            )


def _flat_nhwc(x_nchw):
    """(N,C,H,W) -> the [1, 1, N*H*W, C] channels-last flat form every ttnn conv takes."""
    n, c, h, w = x_nchw.shape
    return x_nchw.permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c).contiguous()


def _id(case):
    idx, tag, ic, oc, hw, k, s, relu, _deall, _in_mem, _out_mem, dup_of = case
    return "%02d_%s_%dto%d_%dx%d_k%d_s%d%s%s" % (
        idx,
        tag.replace("layer", "L").replace(".", "_"),
        ic,
        oc,
        hw,
        hw,
        k,
        s,
        "_relu" if relu else "",
        "_dup%d" % dup_of if dup_of is not None else "",
    )


# --------------------------------------------------------------------------------------------------
# host-only: the table really is torchvision ResNet-50's, re-derived rather than trusted
# --------------------------------------------------------------------------------------------------
def test_forge_conv_table_matches_resnet50_topology():
    """
    Rebuild the 53-conv topology from first principles -- layers [3,4,6,3], widths
    [64,128,256,512], expansion 4, stride on the 3x3 -- and check the table against it.

    This is a table check, not a device test: it runs in milliseconds and fails loudly if the
    configs are ever edited into something that is no longer ResNet-50, instead of quietly
    PCC-testing the wrong numbers.
    """
    want = [("conv1", 3, 64, 224, 7, 2)]
    ch_in, spatial = 64, 56  # after the stem conv + max_pool2d
    for layer, (blocks, width) in enumerate(zip([3, 4, 6, 3], [64, 128, 256, 512]), start=1):
        for b in range(blocks):
            # stride 2 on the first block's 3x3 of layer2..4; the 1x1 conv1/conv3 never
            # stride, and the downsample mirrors the block's stride.
            stride = 2 if (b == 0 and layer > 1) else 1
            want.append(("layer%d.%d.conv1" % (layer, b), ch_in, width, spatial, 1, 1))
            want.append(("layer%d.%d.conv2" % (layer, b), width, width, spatial, 3, stride))
            out_hw = spatial // stride
            want.append(("layer%d.%d.conv3" % (layer, b), width, width * 4, out_hw, 1, 1))
            if b == 0:
                want.append(("layer%d.0.downsample" % layer, ch_in, width * 4, spatial, 1, stride))
            ch_in = width * 4
            spatial = out_hw

    assert len(CONV_CASES) == 53, "the Forge graph has 53 convs, table has %d" % len(CONV_CASES)
    assert len(want) == 53, "re-derived topology has %d convs, expected 53" % len(want)
    for pos, (case, exp) in enumerate(zip(CONV_CASES, want)):
        assert case[0] == pos, "CONV_CASES[%d] carries idx %d" % (pos, case[0])
        got = case[1:7]
        assert got == exp, "conv%d disagrees with the re-derived ResNet-50 topology:\n  table   %s\n  derived %s" % (
            pos,
            got,
            exp,
        )

    # fused-relu split: only the 16 conv3s and the 4 downsamples feed a residual add
    fused = [c for c in CONV_CASES if c[7]]
    bare = [c for c in CONV_CASES if not c[7]]
    assert len(fused) == 33 and len(bare) == 20, "expected 33 fused-relu / 20 bare, got %d / %d" % (
        len(fused),
        len(bare),
    )
    assert all(
        c[1].endswith("conv3") or c[1].endswith("downsample") for c in bare
    ), "a conv without a fused relu that is neither a conv3 nor a downsample: %s" % [
        c[1] for c in bare if not (c[1].endswith("conv3") or c[1].endswith("downsample"))
    ]

    # duplicate bookkeeping: every dup must point at an earlier case with an identical config
    distinct = 0
    for case in CONV_CASES:
        if case[11] is None:
            distinct += 1
            continue
        first = CONV_CASES[case[11]]
        assert first[11] is None and first[0] < case[0], "conv%d dup chain is not flat" % case[0]
        assert first[2:11] == case[2:11], "conv%d is marked a duplicate of conv%d but the configs differ" % (
            case[0],
            first[0],
        )
    assert distinct == 24, "expected 24 distinct conv configs, table says %d" % distinct


# --------------------------------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------------------------------
# The non-1x1 convs go through the fused tilize path that has hung on Quasar; cap the module so a
# hang surfaces as a timeout instead of blocking the whole sweep.
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("case", CONV_CASES, ids=[_id(c) for c in CONV_CASES])
def test_forge_conv2d(mesh_device, case):
    device = mesh_device
    torch.manual_seed(0)

    idx, tag, ic, oc, hw, k, s, fused_relu, dealloc_act, in_mem, out_mem, _dup_of = case
    _require_grid(device, in_mem, out_mem)

    batch, groups = 1, 1
    pad = k // 2  # symmetric on all four sides for every resnet conv

    # ---- torch golden (NCHW) --------------------------------------------------------------------
    x_nchw = torch.randn((batch, ic, hw, hw), dtype=torch.bfloat16).float()
    weight = torch.randn((oc, ic // groups, k, k), dtype=torch.bfloat16).float()
    bias = torch.randn((1, 1, 1, oc), dtype=torch.bfloat16).float()

    golden = torch.nn.functional.conv2d(
        x_nchw, weight, bias=bias.reshape(-1), stride=(s, s), padding=(pad, pad), dilation=(1, 1)
    )
    if fused_relu:
        golden = torch.relu(golden)
    exp_oh, exp_ow = golden.shape[2], golden.shape[3]
    assert (exp_oh, exp_ow) == (hw // s, hw // s), "conv%d torch output %dx%d disagrees with the IR's %dx%d" % (
        idx,
        exp_oh,
        exp_ow,
        hw // s,
        hw // s,
    )

    # ---- operands in Forge's exact layout -------------------------------------------------------
    tt_in = _to_device(_flat_nhwc(x_nchw.to(torch.bfloat16)), in_mem, device)
    tt_w = ttnn.from_torch(weight.to(torch.bfloat16), dtype=ttnn.bfloat16)
    tt_b = ttnn.from_torch(bias.to(torch.bfloat16), dtype=ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if fused_relu else None,
        shard_layout=getattr(ttnn.TensorMemoryLayout, out_mem[0]),
        act_block_h_override=0,
        deallocate_activation=dealloc_act,
        config_tensors_in_dram=True,
        enable_kernel_stride_folding=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2)

    out, [out_h, out_w], [prep_w, prep_b] = ttnn.experimental.quasar.conv2d(
        input_tensor=tt_in,
        weight_tensor=tt_w,
        bias_tensor=tt_b,
        device=device,
        in_channels=ic,
        out_channels=oc,
        batch_size=batch,
        input_height=hw,
        input_width=hw,
        kernel_size=(k, k),
        stride=(s, s),
        padding=(pad, pad, pad, pad),
        dilation=(1, 1),
        groups=groups,
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        memory_config=_mem(out_mem),
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    ttnn.synchronize_device(device)

    # ---- structural checks against the Forge ground truth ---------------------------------------
    assert (out_h, out_w) == (
        exp_oh,
        exp_ow,
    ), "conv%d output spatial: op returned %dx%d, Forge IR / torch say %dx%d" % (idx, out_h, out_w, exp_oh, exp_ow)
    # the op's internally-prepared weight must match what Forge's prepare_conv2d_weights made
    want_prep = (1, 1, ic * k * k, oc)
    assert tuple(prep_w.shape) == want_prep, "conv%d prepared weight %s, Forge's prepare_conv2d_weights makes %s" % (
        idx,
        tuple(prep_w.shape),
        want_prep,
    )
    assert tuple(prep_b.shape)[-1] >= oc, "prepared bias too narrow: %s" % (tuple(prep_b.shape),)
    assert out.shape[-1] >= oc, "conv output has %d channels, need >= %d" % (out.shape[-1], oc)
    assert out.shape[-2] == batch * exp_oh * exp_ow, "conv%d output rows: got %d, Forge IR says %d" % (
        idx,
        out.shape[-2],
        batch * exp_oh * exp_ow,
    )
    got_layout = out.memory_config().memory_layout
    assert got_layout == getattr(ttnn.TensorMemoryLayout, out_mem[0]), "conv%d landed in %s but Forge asked for %s" % (
        idx,
        got_layout,
        out_mem[0],
    )

    # ---- PCC ------------------------------------------------------------------------------------
    tt_out = ttnn.to_torch(ttnn.from_device(out)).reshape(batch, out_h, out_w, -1)[:, :, :, :oc]
    assert_with_pcc(golden, tt_out.permute(0, 3, 1, 2).float(), pcc=PCC)
