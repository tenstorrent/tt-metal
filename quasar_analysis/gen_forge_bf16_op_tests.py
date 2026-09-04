# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Generate ONE STANDALONE TEST FILE PER OP CALL-SITE for the bf16-only tt-forge ResNet-50 compile.

91 files, one per COMPUTE row of resnet50_forge_bf16_vs_quasar.xlsx sheet 1 (its layout-plumbing
rows are skipped -- see SKIP_OPS), written into
  models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/
in the style of the sibling ../ops/ suite: a flat directory of self-contained test files, each with
its own docstring (WHERE IT COMES FROM / WHAT IT VALIDATES / RUN), its own config constants and its
own operand builder. No conftest.py, no shared helper module, no config module.

Each file declares the same five module constants, which test_op_inventory_bf16.py parses back off
disk (with ast, no import, no device) and cross-checks against the ResNet-50 topology:
    SHEET_ROW, FORGE_OP, QUASAR_OP, OPERAND_SHAPES, OUTPUT_SHAPE

Run:
    python quasar_analysis/gen_forge_bf16_op_tests.py            # write the files
    python quasar_analysis/gen_forge_bf16_op_tests.py --check    # report what it would write
"""

import argparse
import os
import re
import textwrap
import xml.etree.ElementTree as ET
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(REPO, "resnet50_forge_bf16_vs_quasar.xlsx")
OUTDIR = os.path.join(REPO, "models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16")
RELDIR = "models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

HEADER = """# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""

COMPILE_NOTE = """THE COMPILE
-----------
CompilerConfig() with exactly enable_optimization_passes=True and default_df_override=Float16_b,
and nothing else -- no consteval, no opt_level=2, no HiFi2, no remove_dead_values, no
max_legal_layouts. Every tensor is bf16 and DRAM INTERLEAVED, so this file pins no core range and
nothing here depends on the device grid. The same op under the OPTIMISED compile (L1, sharded,
HiFi2, pinned core ranges) is in ../ResNet50_Forge_Fe/."""

RUN_NOTE = """RUN
---
  TT_METAL_SIMULATOR=<dir>/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 ARCH_NAME=quasar \\
  pytest -s {rel}/{fname}"""

IMPORTS = '''
import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# This compile pins <interleaved> #dram on every tensor -- no shard spec, no core ranges.
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _assert_quasar(device):
    """
    Refuse to report a pass unless this really ran on a Quasar part.

    Every op in this file is a ttnn.experimental.quasar op, which builds Gen2 kernels; on any other
    arch it would TT_FATAL rather than quietly produce a number, but asserting it here means a green
    tick in this file always means "green ON QUASAR" without having to go and read the run header.

    To prove the op also DISPATCHED (a device program was built and enqueued, not a host fallback),
    run the suite under the attestation plugin:
        pytest -p quasar_analysis.pytest_quasar_attest ...
    which captures the ttnn graph around every test and records the device operations underneath.
    """
    assert device.arch() == ttnn.Arch.QUASAR, (
        "this test ran on %s, not Arch.QUASAR. Open a Quasar device (TT_METAL_SIMULATOR=<dir>/"
        "libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 ARCH_NAME=quasar) -- see "
        "test_op_inventory_bf16.py::test_device_under_test_is_quasar." % device.arch()
    )
'''


DEVICE_PARAMS = '@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)'


# --------------------------------------------------------------------------------------------------
# sheet 1
# --------------------------------------------------------------------------------------------------
def read_sheet1():
    z = zipfile.ZipFile(XLSX)
    root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for r in root.iter(NS + "row"):
        cells, hi = {}, -1
        for c in r.findall(NS + "c"):
            idx = 0
            for ch in re.match(r"([A-Z]+)", c.get("r")).group(1):
                idx = idx * 26 + (ord(ch) - 64)
            idx -= 1
            is_el = c.find(NS + "is")
            if is_el is not None:
                v = "".join(t.text or "" for t in is_el.iter(NS + "t"))
            else:
                ve = c.find(NS + "v")
                v = ve.text if ve is not None and ve.text is not None else ""
            if v:
                cells[idx], hi = v, max(hi, idx)
        rows.append([cells.get(i, "") for i in range(hi + 1)])

    recs = []
    for r in rows[3:]:
        g = lambda i: r[i] if len(r) > i else ""
        ops = [dict(role=g(2 + 4 * k), shape=g(3 + 4 * k), dtype=g(4 + 4 * k), cfg=g(5 + 4 * k)) for k in range(4)]
        recs.append(
            dict(
                idx=int(g(0)),
                op=g(1),
                ops=[o for o in ops if o["role"]],
                out=dict(role=g(18), shape=g(19), dtype=g(20), cfg=g(21)),
                attrs=g(22),
                ir=g(23),
            )
        )
    assert len(recs) == 141 and [x["idx"] for x in recs] == list(range(141)), "sheet 1 is not 141 rows 0..140"
    return recs


def layout_of(cfg):
    if "system_memory" in cfg:
        return "host"
    if "device handle" in cfg:
        return "device handle"
    return "TILE" if "ttcore.tile" in cfg else "ROW_MAJOR"


def mem_of(cfg):
    if "system_memory" in cfg:
        return "#system_memory (host)"
    if "device handle" in cfg:
        return ""
    return "DRAM interleaved"


def tup(shape):
    return tuple(int(x) for x in shape.split("x"))


def short(shape):
    t = tup(shape)
    return "x".join(str(d) for d in (t[2:] if len(t) == 4 and t[0] == 1 and t[1] == 1 else t))


# --------------------------------------------------------------------------------------------------
# naming: one file per row
# --------------------------------------------------------------------------------------------------
def conv_tags():
    """The 53 conv call-sites, in @forward order, named after the torchvision module."""
    names = ["conv1"]
    for layer, (blocks, _w) in enumerate(zip([3, 4, 6, 3], [64, 128, 256, 512]), start=1):
        for b in range(blocks):
            names += ["layer%d_%d_conv1" % (layer, b), "layer%d_%d_conv2" % (layer, b), "layer%d_%d_conv3" % (layer, b)]
            if b == 0:
                names.append("layer%d_0_downsample" % layer)
    return names


def block_tags(n):
    """The 16 bottlenecks, in @forward order."""
    out = []
    for layer, blocks in enumerate([3, 4, 6, 3], start=1):
        for b in range(blocks):
            out.append("layer%d_%d" % (layer, b))
    assert len(out) == n
    return out


def name_rows(recs):
    """row index -> (file stem, short tag), for the rows that get a file."""
    convs = iter(conv_tags())
    adds = iter(block_tags(16))
    relus = iter(block_tags(16))
    names = {}
    for r in recs:
        i, op = r["idx"], r["op"]
        if op in SKIP_OPS:
            continue
        if op == "ttnn.conv2d":
            tag = next(convs)
        elif op == "ttnn.add":
            tag = next(adds)
        elif op == "ttnn.relu":
            tag = next(relus)
        elif op == "ttnn.reshape":
            tag = "stem_flatten" if i == 2 else "classifier_squeeze"
        elif op == "ttnn.permute":
            tag = "nchw2nhwc"
        elif op == "ttnn.max_pool2d":
            tag = "stem"
        elif op == "ttnn.mean":
            tag = "global_avgpool"
        elif op == "ttnn.linear":
            tag = "fc"
        else:
            raise SystemExit("unhandled op %s at row %d" % (op, i))
        names[i] = ("test_op%03d_%s_%s.py" % (i, op.split(".")[1], tag), tag)
    return names


# --------------------------------------------------------------------------------------------------
# docstring pieces
# --------------------------------------------------------------------------------------------------
def wrap_block(text, indent="    ", width=112):
    return "\n".join(textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=indent + "    "))


def operand_table(rec):
    lines = []
    for o in rec["ops"]:
        if o["role"] == "Device handle":
            lines.append("    %-34s %s" % (o["role"], "(!ttnn.device)"))
            continue
        lines.append(
            "    %-34s %-14s %-6s %-10s %s" % (o["role"], o["shape"], "bf16", layout_of(o["cfg"]), mem_of(o["cfg"]))
        )
    o = rec["out"]
    lines.append(
        "    %-34s %-14s %-6s %-10s %s" % ("-> " + o["role"], o["shape"], "bf16", layout_of(o["cfg"]), mem_of(o["cfg"]))
    )
    return "\n".join(lines)


def docstring(rec, names, prose, validates, status, fname):
    i = rec["idx"]
    stem_title = "Sheet 1 row %d of 141 -- %s%s" % (i, rec["op"], (", %s" % names[i][1]) if names[i][1] else "")
    parts = [
        stem_title,
        "",
        "One op, one file. Part of the per-call-site replay of the BF16-ONLY tt-forge ResNet-50 compile;",
        "%s/ holds one of these for every one of the 141 ops in @forward." % RELDIR.split("/")[-1],
        "",
        "WHERE IT COMES FROM",
        "-------------------",
        prose.strip(),
        "",
        'TTNN IR, verbatim from resnet50_forge_bf16_vs_quasar.xlsx sheet 1 ("Forge ops (bf16 only)"):',
        "",
        wrap_block(rec["ir"], indent="    "),
        "",
        "Operands, verbatim from the same row:",
        "",
        operand_table(rec),
        "",
        "Attributes:",
        "",
        wrap_block(rec["attrs"], indent="    "),
        "",
        "WHAT IT VALIDATES",
        "-----------------",
        validates.strip(),
        "",
        COMPILE_NOTE,
        "",
        RUN_NOTE.format(rel=RELDIR, fname=fname),
        "",
        "Status on 2026-09-04 (craq-sim, Arch.QUASAR, 8x4): %s" % status,
    ]
    body = "\n".join(parts).rstrip()
    assert '"""' not in body
    return '"""\n%s\n"""\n' % body


def consts(rec, quasar_op):
    ops = [tup(o["shape"]) for o in rec["ops"] if o["role"] not in ("Device handle",)]
    lines = [
        "# --- the five constants test_op_inventory_bf16.py parses back off disk ---",
        "SHEET_ROW = %d" % rec["idx"],
        'FORGE_OP = "%s"' % rec["op"],
        ('QUASAR_OP = "%s"' % quasar_op) if quasar_op else "QUASAR_OP = None  # no such op on Quasar",
        "OPERAND_SHAPES = (%s%s)" % (", ".join(repr(s) for s in ops), "," if len(ops) == 1 else ""),
        "OUTPUT_SHAPE = %r" % (tup(rec["out"]["shape"]),),
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------------
# per-op-kind emitters -- each returns (prose, validates, quasar_op, body)
# --------------------------------------------------------------------------------------------------
def em_permute(rec, names, consumer):
    prose = (
        "The NCHW -> NHWC conversion of the model input, once, before the stem. torchvision hands Forge an\n"
        "NCHW image; every ttnn conv wants channels-last, so Forge permutes first and flattens second\n"
        "(sheet 1 row 2).\n\n"
        "This is the hardest possible permute for a tiled layout: it moves the 3-element channel axis from\n"
        "position 1 to position 3, so a 224x224 tiled face becomes a 224x3 one and every tile is rebuilt."
    )
    validates = (
        "THE GAP: ttnn.experimental.quasar binds `transpose` (a two-axis swap) but NO general `permute`. The\n"
        "hand-written metal quasar model never needs one -- it uploads its input already channels-last and\n"
        "folds it -- so this op has no counterpart there to compare against.\n\n"
        "So there is no ttnn.experimental.quasar.permute to call, and NOTHING IN THIS SUITE XFAILS. What\n"
        "this file runs instead is the route that DOES exist: 0,2,3,1 decomposed into the two adjacent swaps\n"
        "quasar.transpose can express,\n"
        "        [1,3,224,224] --t(1,2)--> [1,224,3,224] --t(2,3)--> [1,224,224,3]\n"
        "which is what a Quasar-aware compiler would have to lower this permute into. That is a real device\n"
        "test with an exact-equality check, not a placeholder.\n\n"
        "The gap itself is watched in ONE place:\n"
        "    test_op_inventory_bf16.py::test_forge_ops_map_onto_the_live_quasar_build\n"
        "fails the day quasar binds a permute, which is the signal to add the direct test here.\n\n"
        "A permute moves data, it does not compute it, so the check is EXACT equality."
    )
    body = '''
IN_SHAPE = (1, 3, 224, 224)
PERMUTATION = (0, 2, 3, 1)
OUT_SHAPE = (1, 224, 224, 3)


{dp}
def test_forge_bf16_op001_permute_via_transpose(mesh_device):
    """
    Sheet 1 row 1's permutation 0,2,3,1 lowered to the two adjacent swaps quasar.transpose can express.

    Quasar has no permute, so this decomposition is the only route -- see the module docstring. It is a
    full device test: real operands, exact-equality check, no xfail.
    """
    device = mesh_device
    torch.manual_seed(0)

    host = torch.randn(IN_SHAPE, dtype=torch.bfloat16)
    golden = host.permute(*PERMUTATION).contiguous()

    tt = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    tt = ttnn.experimental.quasar.transpose(tt, 1, 2, memory_config=DRAM)  # [1,3,224,224] -> [1,224,3,224]
    assert tuple(tt.shape) == (1, 224, 3, 224), "first swap gave %s" % (tuple(tt.shape),)
    tt = ttnn.experimental.quasar.transpose(tt, 2, 3, memory_config=DRAM)  # -> [1,224,224,3]
    ttnn.synchronize_device(device)

    assert tuple(tt.shape) == OUT_SHAPE, "decomposed permute gave %s, sheet 1 row 1 says %s" % (
        tuple(tt.shape),
        OUT_SHAPE,
    )
    got = ttnn.to_torch(ttnn.from_device(tt))
    assert_with_pcc(golden.float(), got.float(), pcc=0.9999)
    assert torch.equal(got.to(torch.bfloat16), golden), "decomposed permute changed %d of %d elements" % (
        int((got.to(torch.bfloat16) != golden).sum()),
        golden.numel(),
    )
'''.format(dp=DEVICE_PARAMS)
    return prose, validates, None, body


def em_reshape(rec, names, consumer):
    src, dst = tup(rec["ops"][0]["shape"]), tup(rec["out"]["shape"])
    if rec["idx"] == 2:
        prose = (
            "Flatten the permuted image into the channels-last activation layout every ttnn conv wants:\n"
            "[1, 224, 224, 3] -> [1, 1, 50176, 3], feeding the 7x7 stem conv (sheet 1 row 4).\n\n"
            "The last dim is 3, so both the source and the result carry 29 columns of tile padding, and the\n"
            "row count changes from 224 (7 tiles) to 50176 (1568 tiles). In a tiled layout that is a real data\n"
            "movement, not a view."
        )
    else:
        prose = (
            "Drop the pooled spatial dims before the classifier: [1, 1, 1, 2048] -> [1, 2048], between the\n"
            "global average (sheet 1 row 138) and the fc (row 140).\n\n"
            "A rank change only -- both shapes are one 32x2048 padded tile row -- so this one should be a view."
        )
    validates = (
        "quasar.reshape is one of the four generic ops known to work unchanged on Quasar (reshape / clone /\n"
        "to_memory_config / reallocate -- all layout-and-alloc, no kernel), so this is expected to pass; it is\n"
        "here so the sheet's op list is covered end to end.\n\n"
        "A reshape moves data without computing, so the check is EXACT equality against the reshaped torch\n"
        "tensor, with a PCC assert alongside so a partial corruption reports a number."
    )
    body = """
IN_SHAPE = {src!r}
OUT_SHAPE = {dst!r}


{dp}
def test_forge_bf16_op{i:03d}_reshape(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    host = torch.randn(IN_SHAPE, dtype=torch.bfloat16)
    golden = host.reshape(OUT_SHAPE)

    tt_in = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    out = ttnn.experimental.quasar.reshape(tt_in, list(OUT_SHAPE), memory_config=DRAM)
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == OUT_SHAPE, "output shape %s, sheet 1 row {i} says %s" % (tuple(out.shape), OUT_SHAPE)
    assert out.layout == ttnn.TILE_LAYOUT, "output layout %s, the IR says tiled" % (out.layout,)
    mem = out.memory_config()
    assert mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, "not interleaved: %s" % (mem.memory_layout,)
    assert mem.buffer_type == ttnn.BufferType.DRAM, "not in DRAM: %s" % (mem.buffer_type,)

    got = ttnn.to_torch(ttnn.from_device(out))
    assert_with_pcc(golden.float(), got.float(), pcc=0.9999)
    assert torch.equal(got.to(torch.bfloat16), golden), "reshape changed %d of %d elements" % (
        int((got.to(torch.bfloat16) != golden).sum()),
        golden.numel(),
    )
""".format(src=src, dst=dst, i=rec["idx"], dp=DEVICE_PARAMS)
    return prose, validates, "quasar.reshape", body


def conv_params(rec):
    a = rec["attrs"]
    g = lambda p: re.search(p, a).group(1)
    ic, oc = int(g(r"in_channels = (\d+)")), int(g(r"out_channels = (\d+)"))
    hw = int(g(r"input_height = (\d+)"))
    k = int(g(r"kernel_size = array<i32: (\d+)"))
    s = int(g(r"stride = array<i32: (\d+)"))
    pad = int(g(r"padding = array<i32: (\d+)"))
    return ic, oc, hw, k, s, pad, "op_type = relu" in a


def em_conv2d(rec, names, consumer):
    ic, oc, hw, k, s, pad, fused = conv_params(rec)
    tag = names[rec["idx"]][1]
    oh = hw // s
    halo = k > 1 or s > 1
    prose = (
        "torchvision ResNet-50 `%s`: %d -> %d channels, %dx%d kernel, stride %d, padding %d, over a %dx%d\n"
        "feature map, producing %dx%d. One of the 53 convs in the graph.\n\n"
        "%s\n\n"
        "Forge hands conv2d a channels-last flattened activation [1, 1, N*H*W, C] in ROW_MAJOR and gets a TILE\n"
        "result back, so this test builds the operand row-major and asserts the result is tiled.%s The weight is\n"
        "the RAW OIHW tensor straight from host memory: this compile runs no prepare_conv2d_weights anywhere,\n"
        "so quasar.conv2d prepares it internally."
        % (
            tag.replace("_", "."),
            ic,
            oc,
            k,
            k,
            s,
            pad,
            hw,
            hw,
            oh,
            oh,
            (
                "It carries a FUSED RELU (Conv2dConfig.activation = <op_type = relu>). 33 of the 53 convs do;\nthe other 20 are the 16 bottleneck conv3s and the 4 downsamples, whose output feeds a residual add,\nwhere Forge emits relu as a separate op instead."
                if fused
                else "It carries NO fused activation: its output feeds a residual add, and the relu that follows is\nemitted by Forge as a SEPARATE ttnn.relu op -- which Quasar has no binding for. 20 of the 53 convs\nare like this (the 16 conv3s and the 4 downsamples)."
            ),
            "",
        )
    )
    validates = (
        "PCC >= 0.98 against torch.nn.functional.conv2d%s, plus four structural checks against the Forge\n"
        "ground truth before the numbers are even looked at: the returned (out_h, out_w) is %dx%d, the op's\n"
        "INTERNALLY-PREPARED weight has the shape prepare_conv2d_weights would have produced\n"
        "([1, 1, %d, %d]) so the two weight-prep paths are checked to agree, the output has %d rows, and it\n"
        "landed TILE / INTERLEAVED / DRAM as the IR says.\n\n"
        "Forge's compute config -- math_fidelity = hifi4 WITH fp32_dest_acc_en = true -- is passed through\n"
        "VERBATIM, because that is the configuration the sheet records. On Quasar that flag has needed an\n"
        "explicit per-DFB unpack_modes entry; if that is what fails, that is the finding, not a test defect.\n\n"
        "%s"
        % (
            " then torch.relu" if fused else "",
            oh,
            oh,
            ic * k * k,
            oc,
            oh * oh,
            (
                "This conv needs the HALO (kernel > 1 or stride > 1), whose gather path has been broken on\nQuasar; it can also HANG rather than fail, hence the timeout marker."
                if halo
                else "This is a stride-1 1x1 conv, so it needs no halo and lowers straight onto the matmul path."
            ),
        )
    )
    body = """
IN_CHANNELS = {ic}
OUT_CHANNELS = {oc}
INPUT_HW = {hw}          # both input_height and input_width
KERNEL = {k}
STRIDE = {s}
PADDING = {pad}          # symmetric on all four sides; == KERNEL // 2 for every resnet conv
FUSED_RELU = {fused}
BATCH, GROUPS, DILATION = 1, 1, (1, 1)


@pytest.mark.timeout(600)
{dp}
def test_forge_bf16_op{i:03d}_conv2d(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    # ---- torch golden (NCHW) --------------------------------------------------------------------
    x_nchw = torch.randn((BATCH, IN_CHANNELS, INPUT_HW, INPUT_HW), dtype=torch.bfloat16).float()
    weight = torch.randn((OUT_CHANNELS, IN_CHANNELS // GROUPS, KERNEL, KERNEL), dtype=torch.bfloat16).float()
    bias = torch.randn((1, 1, 1, OUT_CHANNELS), dtype=torch.bfloat16).float()

    golden = torch.nn.functional.conv2d(
        x_nchw, weight, bias=bias.reshape(-1), stride=(STRIDE, STRIDE), padding=(PADDING, PADDING), dilation=DILATION
    )
    if FUSED_RELU:
        golden = torch.relu(golden)
    exp_oh, exp_ow = golden.shape[2], golden.shape[3]
    assert (exp_oh, exp_ow) == ({oh}, {oh}), "torch says %dx%d, sheet 1 row {i} says {oh}x{oh}" % (exp_oh, exp_ow)

    # ---- operands in Forge's exact layout: activation ROW_MAJOR in DRAM, weights on host ---------
    flat = x_nchw.to(torch.bfloat16).permute(0, 2, 3, 1).reshape(1, 1, BATCH * INPUT_HW * INPUT_HW, IN_CHANNELS)
    tt_in = ttnn.from_torch(
        flat.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM
    )
    tt_w = ttnn.from_torch(weight.to(torch.bfloat16), dtype=ttnn.bfloat16)  # raw OIHW, #system_memory
    tt_b = ttnn.from_torch(bias.to(torch.bfloat16), dtype=ttnn.bfloat16)  # [1,1,1,oc], #system_memory

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if FUSED_RELU else None,
        enable_kernel_stride_folding=False,
        config_tensors_in_dram=True,
    )
    # compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )

    out, [out_h, out_w], [prep_w, prep_b] = ttnn.experimental.quasar.conv2d(
        input_tensor=tt_in,
        weight_tensor=tt_w,
        bias_tensor=tt_b,
        device=device,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        batch_size=BATCH,
        input_height=INPUT_HW,
        input_width=INPUT_HW,
        kernel_size=(KERNEL, KERNEL),
        stride=(STRIDE, STRIDE),
        padding=(PADDING, PADDING, PADDING, PADDING),
        dilation=DILATION,
        groups=GROUPS,
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        memory_config=DRAM,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    ttnn.synchronize_device(device)

    # ---- structural checks against the Forge ground truth ----------------------------------------
    assert (out_h, out_w) == (exp_oh, exp_ow), "op returned %dx%d, Forge IR / torch say %dx%d" % (
        out_h,
        out_w,
        exp_oh,
        exp_ow,
    )
    want_prep = (1, 1, IN_CHANNELS * KERNEL * KERNEL, OUT_CHANNELS)
    assert tuple(prep_w.shape) == want_prep, "prepared weight %s, prepare_conv2d_weights makes %s" % (
        tuple(prep_w.shape),
        want_prep,
    )
    assert tuple(prep_b.shape)[-1] >= OUT_CHANNELS, "prepared bias too narrow: %s" % (tuple(prep_b.shape),)
    assert out.shape[-1] >= OUT_CHANNELS, "output has %d channels, need >= %d" % (out.shape[-1], OUT_CHANNELS)
    assert out.shape[-2] == BATCH * exp_oh * exp_ow, "output rows %d, Forge IR says %d" % (
        out.shape[-2],
        BATCH * exp_oh * exp_ow,
    )
    mem = out.memory_config()
    assert mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, "not interleaved: %s" % (mem.memory_layout,)
    assert mem.buffer_type == ttnn.BufferType.DRAM, "not in DRAM: %s" % (mem.buffer_type,)
    assert out.layout == ttnn.TILE_LAYOUT, "output layout %s, the IR says the result is tiled" % (out.layout,)

    # ---- PCC -------------------------------------------------------------------------------------
    tt_out = ttnn.to_torch(ttnn.from_device(out)).reshape(BATCH, out_h, out_w, -1)[:, :, :, :OUT_CHANNELS]
    assert_with_pcc(golden, tt_out.permute(0, 3, 1, 2).float(), pcc=0.98)
""".format(ic=ic, oc=oc, hw=hw, k=k, s=s, pad=pad, fused=fused, i=rec["idx"], oh=oh, dp=DEVICE_PARAMS)
    return prose, validates, "quasar.conv2d", body


def em_max_pool2d(rec, names, consumer):
    prose = (
        "The stem max pool -- the only pooling op in ResNet-50 apart from the final average. 3x3 kernel,\n"
        "stride 2, padding 1 over the 112x112x64 stem-conv output, halving it to 56x56.\n\n"
        "Both the operand and the result are ROW_MAJOR here, unlike the optimised compile where the pool\n"
        "output is a height-sharded tensor feeding two convs directly.\n\n"
        "reallocate_halo_output = false is Forge's choice and is passed verbatim -- the ttnn default is True,\n"
        "so leaving it out would NOT be a faithful replay."
    )
    validates = (
        "PCC >= 0.999 against torch.nn.functional.max_pool2d. Max SELECTS a value, it does not accumulate one,\n"
        "so bf16 in gives bf16 out with no arithmetic error and the bound can be tight.\n\n"
        "A 3x3 pool needs the HALO, the same machinery every kernel > 1 conv needs. That makes this file the\n"
        "cheapest halo test in the directory, and a useful control: if the convs fail on the halo and this one\n"
        "passes, the fault is in the CONV halo path, not the halo as such."
    )
    body = """
BATCH, CHANNELS = 1, 64
IN_H, IN_W = 112, 112
KERNEL, STRIDE, PADDING, DILATION = (3, 3), (2, 2), (1, 1), (1, 1)
CEIL_MODE = False
REALLOCATE_HALO_OUTPUT = False  # Forge's choice; the ttnn default is True
OUT_H, OUT_W = 56, 56


@pytest.mark.timeout(600)
{dp}
def test_forge_bf16_op006_max_pool2d(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    x_nchw = torch.randn((BATCH, CHANNELS, IN_H, IN_W), dtype=torch.bfloat16).float()
    golden = torch.nn.functional.max_pool2d(
        x_nchw, kernel_size=KERNEL, stride=STRIDE, padding=PADDING, dilation=DILATION, ceil_mode=CEIL_MODE
    )
    assert tuple(golden.shape) == (BATCH, CHANNELS, OUT_H, OUT_W), tuple(golden.shape)

    flat = x_nchw.to(torch.bfloat16).permute(0, 2, 3, 1).reshape(1, 1, BATCH * IN_H * IN_W, CHANNELS)
    tt_in = ttnn.from_torch(
        flat.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM
    )

    out = ttnn.experimental.quasar.max_pool2d(
        input_tensor=tt_in,
        batch_size=BATCH,
        input_h=IN_H,
        input_w=IN_W,
        channels=CHANNELS,
        kernel_size=list(KERNEL),
        stride=list(STRIDE),
        padding=list(PADDING),
        dilation=list(DILATION),
        ceil_mode=CEIL_MODE,
        memory_config=DRAM,
        reallocate_halo_output=REALLOCATE_HALO_OUTPUT,
        dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn.synchronize_device(device)

    assert out.shape[-2] == BATCH * OUT_H * OUT_W, "output rows %d, sheet 1 row 6 says %d" % (
        out.shape[-2],
        BATCH * OUT_H * OUT_W,
    )
    assert out.shape[-1] >= CHANNELS, "output has %d channels, need >= %d" % (out.shape[-1], CHANNELS)
    mem = out.memory_config()
    assert mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, "not interleaved: %s" % (mem.memory_layout,)
    assert mem.buffer_type == ttnn.BufferType.DRAM, "not in DRAM: %s" % (mem.buffer_type,)

    got = ttnn.to_torch(ttnn.from_device(out)).reshape(BATCH, OUT_H, OUT_W, -1)[:, :, :, :CHANNELS]
    assert_with_pcc(golden, got.permute(0, 3, 1, 2).float(), pcc=0.999)
""".format(dp=DEVICE_PARAMS)
    return prose, validates, "quasar.max_pool2d", body


def em_add(rec, names, consumer):
    shape = tup(rec["ops"][0]["shape"])
    tag = names[rec["idx"]][1].replace("_", ".")
    prose = (
        "The residual add that closes bottleneck %s: conv3's output plus the skip branch. Sheet 1 resolves\n"
        "the two branches back to their common ancestor and labels the shorter path the residual/skip, so\n"
        "operand 1 is the main branch and operand 2 is the skip (the downsample output on the first block of\n"
        "each layer, the block input otherwise).\n\n"
        "The op carries NO ATTRIBUTES AT ALL -- no fused activation, no memory config, no compute config --\n"
        "and Forge emits the OUT-OF-PLACE ttnn.add, not add_. That is the difference that matters against the\n"
        "hand-written metal model, which fuses the following relu into the add\n"
        "(quasar.add_(out, ds_out, activations=[UnaryWithParam(RELU)]), see ../ops/test_add.py). Here the add\n"
        "is bare and the relu is left stranded on sheet 1 row %d." % (tag, rec["idx"] + 1)
    )
    aligned = shape[-2] % 32 == 0
    validates = (
        "PCC >= 0.99 against a plain torch add -- a bf16 elementwise add is near-exact -- plus the output\n"
        "shape, TILE, INTERLEAVED and DRAM.\n\n"
        "%s"
        % (
            "The height %d is tile-aligned, so there is no row padding for the op to step around." % shape[-2]
            if aligned
            else "The height %d is NOT tile-aligned, so both operands carry row padding and the add has to leave\nthe pad rows alone."
            % shape[-2]
        )
    )
    body = """
SHAPE = {shape!r}


{dp}
def test_forge_bf16_op{i:03d}_add(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    # operand 1 is the main branch (the conv3 output), operand 2 the residual / skip
    main = torch.randn(SHAPE, dtype=torch.bfloat16)
    skip = torch.randn(SHAPE, dtype=torch.bfloat16)
    golden = main.float() + skip.float()

    tt_main = ttnn.from_torch(main, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    tt_skip = ttnn.from_torch(skip, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

    # bare add: the IR row carries "(no attributes)" -- no fused activation, no memory config
    out = ttnn.experimental.quasar.add(tt_main, tt_skip)
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == SHAPE, "output shape %s, sheet 1 row {i} says %s" % (tuple(out.shape), SHAPE)
    assert out.layout == ttnn.TILE_LAYOUT, "output layout %s, the IR says tiled" % (out.layout,)
    mem = out.memory_config()
    assert mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, "not interleaved: %s" % (mem.memory_layout,)
    assert mem.buffer_type == ttnn.BufferType.DRAM, "not in DRAM: %s" % (mem.buffer_type,)

    got = ttnn.to_torch(ttnn.from_device(out)).float()
    assert_with_pcc(golden, got, pcc=0.99)
""".format(shape=shape, i=rec["idx"], dp=DEVICE_PARAMS)
    return prose, validates, "quasar.add", body


def em_relu(rec, names, consumer):
    shape = tup(rec["ops"][0]["shape"])
    tag = names[rec["idx"]][1].replace("_", ".")
    prose = (
        "The relu that follows the residual add of bottleneck %s (sheet 1 row %d). Forge places it as a\n"
        "SEPARATE op, with the same shape, layout and memory config as the add it consumes:\n\n"
        "    add(conv3_out, skip)      <-- sheet 1 row %d\n"
        "    relu(...)                 <-- THIS ROW, one of 16 with no Quasar equivalent\n\n"
        "Forge DOES fuse relu into 33 of the 53 convs via Conv2dConfig.activation, and that path works on\n"
        "Quasar. It is only these 16 post-add relus that have no home." % (tag, rec["idx"] - 1, rec["idx"] - 1)
    )
    validates = (
        "THE GAP: ttnn.experimental.quasar binds data movement, conv2d, the pools, the matmul family and a\n"
        "BINARY front-end. It binds NO plain unary activation -- no relu, sigmoid or gelu. (prelu, pow and\n"
        "polyval are the only unary-with-param ops bound, and none of them is relu.)\n\n"
        "So there is no ttnn.experimental.quasar.relu to call, and NOTHING IN THIS SUITE XFAILS. What this\n"
        "file runs instead is the route that DOES exist: the add and the relu collapsed into one\n"
        "quasar.add with a fused RELU activation -- exactly what the hand-written metal model already does\n"
        "(resnet50Bottleneck.__call__, see ../ops/test_add.py) and what a Quasar-aware compiler would have\n"
        "to emit for this pair. That is a real device test with a real PCC check, not a placeholder.\n\n"
        "The gap itself is watched in ONE place rather than sixteen:\n"
        "    test_op_inventory_bf16.py::test_forge_ops_map_onto_the_live_quasar_build\n"
        "fails the day quasar binds a standalone relu, which is the signal to add the direct test here.\n\n"
        "Inputs are torch.randn, so the clamp really clamps rather than being a no-op."
    )
    body = '''
SHAPE = {shape!r}


{dp}
def test_forge_bf16_op{i:03d}_relu_fused_add(mesh_device):
    """
    Sheet 1 rows {prev} and {i} collapsed into one quasar.add(a, b, activations=[UnaryWithParam(RELU)]).

    Quasar has no standalone relu, so this fusion is the only route for this pair -- see the module
    docstring. It is a full device test: real operands, real PCC bound, no xfail.
    """
    device = mesh_device
    torch.manual_seed(0)

    main = torch.randn(SHAPE, dtype=torch.bfloat16)
    skip = torch.randn(SHAPE, dtype=torch.bfloat16)
    golden = torch.relu(main.float() + skip.float())

    tt_main = ttnn.from_torch(main, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    tt_skip = ttnn.from_torch(skip, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

    out = ttnn.experimental.quasar.add(
        tt_main,
        tt_skip,
        activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
    )
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == SHAPE, "output shape %s, sheet 1 row {i} says %s" % (tuple(out.shape), SHAPE)
    got = ttnn.to_torch(ttnn.from_device(out)).float()
    assert_with_pcc(golden, got, pcc=0.99)
    assert (got < 0).sum() == 0, "%d negative values survived the fused RELU" % int((got < 0).sum())
'''.format(shape=shape, i=rec["idx"], prev=rec["idx"] - 1, dp=DEVICE_PARAMS)
    return prose, validates, None, body


def em_mean(rec, names, consumer):
    prose = (
        "The global average pool before the classifier. dim_arg = [-2] with keep_dim reduces the flattened\n"
        "spatial axis of the layer4 output -- the mean over the 49 positions of the 7x7 feature map, which is\n"
        "exactly torchvision's nn.AdaptiveAvgPool2d((1, 1)).\n\n"
        "        [1, 1, 49, 2048]  ->  [1, 1, 1, 2048]\n\n"
        "Forge attaches its compute config here too: math_fidelity = hifi4, fp32_dest_acc_en = true."
    )
    validates = (
        "THE GAP: the Quasar namespace binds NO reduction at all -- no sum, no mean, no max, no argmax, and\n"
        "no normalization built on one.\n\n"
        "So there is no ttnn.experimental.quasar.mean to call, and NOTHING IN THIS SUITE XFAILS. What this\n"
        "file runs instead is the route that DOES exist: quasar.avg_pool2d with a 7x7 kernel over a 7x7\n"
        "input, stride 1, no padding -- the same arithmetic, and the lowering a Quasar-aware compiler would\n"
        "use. Forge's compute config is carried verbatim, which is worth watching: conv2d and linear are\n"
        "REJECTED for fp32_dest_acc_en = true on Quasar and this op is not.\n\n"
        "The gap itself is watched in ONE place:\n"
        "    test_op_inventory_bf16.py::test_forge_ops_map_onto_the_live_quasar_build\n"
        "fails the day quasar binds a reduction, which is the signal to add the direct test here.\n\n"
        "PCC >= 0.99: a 49-term bf16 mean."
    )
    body = '''
IN_SHAPE = (1, 1, 49, 2048)
OUT_SHAPE = (1, 1, 1, 2048)
DIM_ARG = [-2]
KEEP_DIM = True
SPATIAL = 7  # 49 = 7 x 7
CHANNELS = 2048


@pytest.mark.timeout(600)
{dp}
def test_forge_bf16_op138_mean_via_avg_pool2d(mesh_device):
    """
    Sheet 1 row 138's mean over the flattened 7x7 spatial axis, lowered to a 7x7 avg_pool2d.

    Quasar binds no reduction, so this is the only route -- see the module docstring. It is a full
    device test: real operands, real PCC bound, Forge's compute config carried verbatim, no xfail.
    """
    device = mesh_device
    torch.manual_seed(0)

    host = torch.randn(IN_SHAPE, dtype=torch.bfloat16)
    golden = host.float().mean(dim=-2, keepdim=True)  # [1, 1, 1, 2048]

    # compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )

    tt_in = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)
    out = ttnn.experimental.quasar.avg_pool2d(
        input_tensor=tt_in,
        batch_size=1,
        input_h=SPATIAL,
        input_w=SPATIAL,
        channels=CHANNELS,
        kernel_size=[SPATIAL, SPATIAL],
        stride=[1, 1],
        padding=[0, 0],
        ceil_mode=False,
        count_include_pad=True,
        memory_config=DRAM,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_config,
    )
    ttnn.synchronize_device(device)

    assert out.shape[-2] == 1, "a %dx%d window left %d rows, expected 1" % (SPATIAL, SPATIAL, out.shape[-2])
    assert out.shape[-1] >= CHANNELS, "output has %d channels, need >= %d" % (out.shape[-1], CHANNELS)

    got = ttnn.to_torch(ttnn.from_device(out)).reshape(1, 1, 1, -1)[:, :, :, :CHANNELS].float()
    assert_with_pcc(golden, got, pcc=0.99)
'''.format(dp=DEVICE_PARAMS)
    return prose, validates, None, body


def em_linear(rec, names, consumer):
    prose = (
        "The 1000-way classifier, the last op in @forward. The weight is stored K x N (2048 x 1000), so both\n"
        "transposes are false, and the bias is the IR's rank-1 [1000] whose memref is 1x32 tiles -- that is\n"
        "the padded 2-D layout [1, 1000], which is how it is built here.\n\n"
        "No program_config and no core_grid: this compile leaves the matmul to pick its own. That is the\n"
        "difference from the optimised compile, which pins a MatmulMultiCoreReuseMultiCast1DProgramConfig."
    )
    validates = (
        "PCC >= 0.98 against act @ weight + bias -- a 2048-deep bf16 reduction -- plus the output shape,\n"
        "TILE, INTERLEAVED and DRAM.\n\n"
        "What is awkward about this case: M = 1 and N = 1000 are BOTH ragged. The activation is one row padded\n"
        "to a 32-row tile, and the output width pads 1000 -> 1024. So it exercises the matmul's handling of a\n"
        "single-tile-row activation with a non-tile-aligned N, which is where the fc has failed before\n"
        "(../ops/test_linear.py, ../ops/test_fc_kspill.py).\n\n"
        "Forge's fp32_dest_acc_en = true is passed through verbatim, because that is what the sheet records."
    )
    body = """
IN_FEATURES, OUT_FEATURES = 2048, 1000
ACT_SHAPE = (1, IN_FEATURES)
WEIGHT_SHAPE = (IN_FEATURES, OUT_FEATURES)
BIAS_SHAPE = (1, OUT_FEATURES)  # the IR's rank-1 [1000] in its padded 1x32-tile layout
OUT_SHAPE = (1, OUT_FEATURES)
TRANSPOSE_A = False
TRANSPOSE_B = False


@pytest.mark.timeout(600)
{dp}
def test_forge_bf16_op140_linear(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    act = torch.randn(ACT_SHAPE, dtype=torch.bfloat16)
    weight = torch.randn(WEIGHT_SHAPE, dtype=torch.bfloat16)
    bias = torch.randn(BIAS_SHAPE, dtype=torch.bfloat16)
    golden = act.float() @ weight.float() + bias.float()

    tt_act = ttnn.from_torch(act, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    tt_w = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    tt_b = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

    # compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )

    out = ttnn.experimental.quasar.linear(
        tt_act,
        tt_w,
        bias=tt_b,
        transpose_a=TRANSPOSE_A,
        transpose_b=TRANSPOSE_B,
        memory_config=DRAM,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_config,
    )
    ttnn.synchronize_device(device)

    assert tuple(out.shape) == OUT_SHAPE, "output shape %s, sheet 1 row 140 says %s" % (tuple(out.shape), OUT_SHAPE)
    assert out.layout == ttnn.TILE_LAYOUT, "output layout %s, the IR says tiled" % (out.layout,)
    mem = out.memory_config()
    assert mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, "not interleaved: %s" % (mem.memory_layout,)
    assert mem.buffer_type == ttnn.BufferType.DRAM, "not in DRAM: %s" % (mem.buffer_type,)

    got = ttnn.to_torch(ttnn.from_device(out)).float()
    assert_with_pcc(golden, got, pcc=0.98)
""".format(dp=DEVICE_PARAMS)
    return prose, validates, "quasar.linear", body


# Sheet-1 ops that get no test file: layout plumbing rather than compute -- they move a tensor
# between TILE and ROW_MAJOR without computing anything, which is the same reason the workbook's
# comparison sheets 3 and 4 leave them out. Rows whose op is listed here are skipped entirely: no
# file, no name, no emitter.
SKIP_OPS = {"ttnn.to_layout"}

EMITTERS = {
    "ttnn.permute": em_permute,
    "ttnn.reshape": em_reshape,
    "ttnn.conv2d": em_conv2d,
    "ttnn.max_pool2d": em_max_pool2d,
    "ttnn.add": em_add,
    "ttnn.relu": em_relu,
    "ttnn.mean": em_mean,
    "ttnn.linear": em_linear,
}


# --------------------------------------------------------------------------------------------------
# the observed status of each row, from the parametrised sweep in quasar_analysis/forge_fe_bf16_runs/
# --------------------------------------------------------------------------------------------------
def read_status(recs):
    """
    sheet row -> a one-line status, taken from the last logged run.

    Keyed on the row in the test file name (test_op<row>_...), which is the only thing that ties a
    log line to a sheet-1 row now that the suite is one file per op. Anything the log does not
    mention stays "(not run)" rather than silently claiming a result.
    """
    import glob

    outcome, tb_of = {}, {}
    for path in sorted(glob.glob(os.path.join(REPO, "quasar_analysis/forge_fe_bf16_runs/*.log"))):
        txt = open(path, errors="replace").read()
        for line in txt.split("\n"):
            m = re.match(r"^(PASSED|FAILED|XFAIL)\s+\S*/test_op(\d{3})_\S*\.py::", line)
            if m:
                outcome[int(m.group(2))] = m.group(1)
        # --tb=line prints one line per failure in execution order with no per-test header; the
        # FAILED lines of the short summary are in the same order, so they pair positionally.
        block = re.search(r"=+ FAILURES =+\n(.*?)\n=+ short test summary", txt, re.S)
        tbs = (
            [l.strip() for l in block.group(1).split("\n") if re.match(r"^/\S+\.py:\d+: ", l.strip())] if block else []
        )
        fails = [l.split()[1] for l in txt.split("\n") if l.startswith("FAILED models/")]
        if len(tbs) == len(fails):
            for nid, tb in zip(fails, tbs):
                m = re.search(r"/test_op(\d{3})_\S*\.py::", nid)
                if m:
                    tb_of[int(m.group(1))] = tb

    def cause(tb):
        if not tb:
            return ""
        if "1076" in tb or "unpack_modes" in tb:
            return (
                "cause A, fp32_dest_acc_en=true rejected (program_spec.cpp:1076, no unpack_modes entry for the "
                "FP32 DFB)"
            )
        if "1439" in tb:
            # --tb=line stops before the info: block that names the buffer, so do not guess one;
            # SUMMARY.txt has the per-buffer counts, taken from the full log.
            return "cause B, Gen2 forbids the self-looped halo scratch DFB (program_spec.cpp:1439)"
        m = re.search(r"AssertionError: ([0-9.]+)", tb)
        if m:
            return "NUMERIC failure -- the op ran and returned, but missed its bound at PCC %.4f" % float(m.group(1))
        return tb

    status = {}
    for row, oc in outcome.items():
        text = {"PASSED": "PASS", "FAILED": "FAIL", "XFAIL": "XFAIL"}[oc]
        if oc == "FAILED":
            text += " -- " + cause(tb_of.get(row, ""))
        status[row] = text
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report what would be written, write nothing")
    args = ap.parse_args()

    recs = read_sheet1()
    names = name_rows(recs)
    status = read_status(recs)

    by_idx = {r["idx"]: r for r in recs}
    written, counts, skipped = [], {}, 0
    for rec in recs:
        i = rec["idx"]
        if rec["op"] in SKIP_OPS:
            skipped += 1
            continue
        fname, _tag = names[i]
        nxt = by_idx.get(i + 1)
        consumer = (nxt["idx"], names[nxt["idx"]][1].replace("_", ".")) if nxt and nxt["op"] == "ttnn.conv2d" else None
        prose, validates, quasar_op, body = EMITTERS[rec["op"]](rec, names, consumer)
        text = HEADER + "\n" + docstring(rec, names, prose, validates, status.get(i, "(not run)"), fname)
        text += IMPORTS + "\n" + consts(rec, quasar_op) + body
        # one uniform edit rather than eight copies in the emitters: every test asserts the arch
        # before it touches the device.
        assert "    device = mesh_device\n" in text, "%s has a test that never binds the device" % fname
        text = text.replace("    device = mesh_device\n", "    device = mesh_device\n    _assert_quasar(device)\n")
        written.append((fname, text))
        counts[rec["op"]] = counts.get(rec["op"], 0) + 1

    print("%d files (%d sheet-1 rows skipped: %s)" % (len(written), skipped, ", ".join(sorted(SKIP_OPS))))
    for op, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print("  %-20s %3d" % (op, n))
    assert sum(counts.values()) + skipped == 141, "%d written + %d skipped != 141" % (sum(counts.values()), skipped)
    assert len({f for f, _ in written}) == len(written), "duplicate file names"
    assert not (set(counts) & SKIP_OPS), "a skipped op still got a file"

    if args.check:
        print("--check: nothing written")
        print("first file would be %s, last %s" % (written[0][0], written[-1][0]))
        return

    # remove the parametrised files these supersede
    for stale in os.listdir(OUTDIR):
        if re.match(r"^test_op\d{3}_.*\.py$", stale) and stale not in {f for f, _ in written}:
            os.remove(os.path.join(OUTDIR, stale))
            print("removed stale %s" % stale)

    for fname, text in written:
        with open(os.path.join(OUTDIR, fname), "w") as fh:
            fh.write(text)
    print("wrote %d files into %s" % (len(written), RELDIR))


if __name__ == "__main__":
    main()
