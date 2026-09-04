# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
The census: what the 91 per-op files in this directory collectively say, whether they still
describe ResNet-50, whether Quasar is registered in this build, and whether the device under test
is actually a Quasar part.

WHY THIS FILE EXISTS
--------------------
Every other file here tests exactly ONE op, which is what makes them easy to hand to the LLK team
-- but it also means nothing checks that the 91 of them, taken together, are still the ResNet-50
graph. That is this file's job, and it does it by READING THE FILES OFF DISK: each op file declares

    SHEET_ROW, FORGE_OP, QUASAR_OP, OPERAND_SHAPES, OUTPUT_SHAPE

and the tests below parse those five constants out with `ast` -- no import, no ttnn, no device --
then re-derive the ResNet-50 topology from first principles and check the files against it. So a
file that is renamed, deleted, duplicated or edited into something that is no longer ResNet-50
fails HERE, loudly, instead of quietly testing the wrong numbers 90 files away.

Everything about the build is likewise checked against the LIVE ttnn build rather than asserted
from a document, so this file fails both when a mapped quasar op vanishes and when one of the three
gaps closes.

SOURCE
------
resnet50_forge_bf16_vs_quasar.xlsx, sheet 1 ("Forge ops (bf16 only)"): 141 ops in @forward from
CompilerConfig() with exactly enable_optimization_passes=True and default_df_override=Float16_b.
Sheet 5 of the same workbook is the per-test ledger. The op files are generated from sheet 1 by
quasar_analysis/gen_forge_bf16_op_tests.py.

COVERAGE
--------
This suite covers the 91 COMPUTE ops of that graph -- one file each. Sheet 1's remaining rows are
layout-plumbing steps that move a tensor between TILE and ROW_MAJOR without computing anything; the
workbook's comparison sheets 3 and 4 leave them out for the same reason and so does this directory.
test_op_census_matches_sheet1 below asserts the 91.

THE THREE GAPS THIS COMPILE HITS
--------------------------------
  ttnn.relu    x16   no standalone unary op on Quasar at all      test_op0NN_relu_*.py
  ttnn.permute  x1   quasar binds transpose, not permute          test_op001_permute_nchw2nhwc.py
  ttnn.mean     x1   quasar binds no reduction op at all          test_op138_mean_global_avgpool.py

RUN
---
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/test_op_inventory_bf16.py
  pytest    ... test_op_inventory_bf16.py -k "not device"     # host-only, no device needed
"""

import ast
import os
import re

import pytest

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))

# The compute-op census of sheet 1, verbatim: 91 ops, one test file each.
# (ttnn.deallocate / ttnn.get_device produce no tensor and are not rows.)
FORGE_OP_COUNTS = {
    "ttnn.conv2d": 53,
    "ttnn.add": 16,
    "ttnn.relu": 16,
    "ttnn.reshape": 2,
    "ttnn.permute": 1,
    "ttnn.max_pool2d": 1,
    "ttnn.mean": 1,
    "ttnn.linear": 1,
}
FORGE_COMPUTE_OPS = 91

# Forge op -> the ttnn.experimental.quasar name that runs it, or None where there is no such op.
FORGE_TO_QUASAR = {
    "ttnn.conv2d": "conv2d",
    "ttnn.add": "add",
    "ttnn.relu": None,  # gap -- fused into conv2d/add instead
    "ttnn.reshape": "reshape",
    "ttnn.permute": None,  # gap -- decomposes into transpose
    "ttnn.max_pool2d": "max_pool2d",
    "ttnn.mean": None,  # gap -- lowers to avg_pool2d
    "ttnn.linear": "linear",
}

# The ops the three gaps have to be routed through instead.
WORKAROUND_OPS = ("transpose", "avg_pool2d", "tilize", "untilize_with_unpadding", "to_memory_config")

CONSTANTS = ("SHEET_ROW", "FORGE_OP", "QUASAR_OP", "OPERAND_SHAPES", "OUTPUT_SHAPE")


def _scan_op_files():
    """
    Parse the five declared constants out of every test_opNNN_*.py in this directory.

    Deliberately `ast` over the file text rather than importing: this stays host-only and fast, it
    cannot be fooled by an import side effect, and it works even when a file is broken enough that
    importing it would raise.
    """
    found = {}
    for fname in sorted(os.listdir(HERE)):
        m = re.match(r"^test_op(\d{3})_(.+)\.py$", fname)
        if not m:
            continue
        tree = ast.parse(open(os.path.join(HERE, fname)).read(), filename=fname)
        got = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in CONSTANTS:
                    got[name] = ast.literal_eval(node.value)
        missing = [c for c in CONSTANTS if c not in got]
        assert not missing, "%s declares no %s -- every op file must declare all of %s" % (
            fname,
            ", ".join(missing),
            ", ".join(CONSTANTS),
        )
        assert got["SHEET_ROW"] == int(m.group(1)), "%s declares SHEET_ROW %d, its name says %s" % (
            fname,
            got["SHEET_ROW"],
            m.group(1),
        )
        got["file"] = fname
        found[got["SHEET_ROW"]] = got
    return found


# --------------------------------------------------------------------------------------------------
# host-only: do the 141 files still describe ResNet-50?
# --------------------------------------------------------------------------------------------------
def test_one_file_per_compute_op():
    """Exactly one op file per compute op: 91 files, unique sheet rows, all inside sheet 1's range."""
    files = _scan_op_files()
    assert len(files) == FORGE_COMPUTE_OPS, "found %d test_opNNN_*.py files, expected %d" % (
        len(files),
        FORGE_COMPUTE_OPS,
    )
    # _scan_op_files keys on SHEET_ROW, so a duplicate row would silently collapse -- count the files too
    n_on_disk = len([f for f in os.listdir(HERE) if re.match(r"^test_op\d{3}_.+\.py$", f)])
    assert n_on_disk == len(files), "%d op files on disk but only %d distinct SHEET_ROWs -- two files share a row" % (
        n_on_disk,
        len(files),
    )
    out_of_range = [r for r in files if not 0 <= r <= 140]
    assert not out_of_range, "these files declare a SHEET_ROW outside sheet 1's 0..140: %s" % (out_of_range,)


def test_op_census_matches_sheet1():
    """Each compute-op kind must appear the number of times sheet 1 records."""
    files = _scan_op_files()
    counts = {}
    for rec in files.values():
        counts[rec["FORGE_OP"]] = counts.get(rec["FORGE_OP"], 0) + 1
    assert counts == FORGE_OP_COUNTS, "the op census off disk is %s, sheet 1 says %s" % (counts, FORGE_OP_COUNTS)
    assert sum(FORGE_OP_COUNTS.values()) == FORGE_COMPUTE_OPS
    assert set(FORGE_OP_COUNTS) == set(FORGE_TO_QUASAR), "the census and the op map name different ops: %s" % (
        set(FORGE_OP_COUNTS) ^ set(FORGE_TO_QUASAR),
    )
    # every file's QUASAR_OP must agree with the map
    for rec in files.values():
        want = FORGE_TO_QUASAR[rec["FORGE_OP"]]
        want = ("quasar.%s" % want) if want else None
        assert rec["QUASAR_OP"] == want, "%s declares QUASAR_OP %r, the map says %r" % (
            rec["file"],
            rec["QUASAR_OP"],
            want,
        )


def test_conv_files_match_resnet50_topology():
    """
    Rebuild the 53-conv topology from first principles -- layers [3,4,6,3], widths
    [64,128,256,512], expansion 4, stride on the 3x3 -- and check the conv files' declared operand
    and output shapes against it.

    Runs in milliseconds with no device, and fails loudly if the files are ever edited into
    something that is no longer ResNet-50.
    """
    files = _scan_op_files()
    convs = [files[k] for k in sorted(files) if files[k]["FORGE_OP"] == "ttnn.conv2d"]
    assert len(convs) == 53, "expected 53 conv files, found %d" % len(convs)

    want = [("conv1", 3, 64, 224, 7, 2)]
    ch_in, spatial = 64, 56  # after the stem conv + max_pool2d
    for layer, (blocks, width) in enumerate(zip([3, 4, 6, 3], [64, 128, 256, 512]), start=1):
        for b in range(blocks):
            # stride 2 on the first block's 3x3 of layer2..4; the 1x1 conv1/conv3 never stride, and
            # the downsample mirrors the block's stride.
            stride = 2 if (b == 0 and layer > 1) else 1
            want.append(("layer%d.%d.conv1" % (layer, b), ch_in, width, spatial, 1, 1))
            want.append(("layer%d.%d.conv2" % (layer, b), width, width, spatial, 3, stride))
            out_hw = spatial // stride
            want.append(("layer%d.%d.conv3" % (layer, b), width, width * 4, out_hw, 1, 1))
            if b == 0:
                want.append(("layer%d.0.downsample" % layer, ch_in, width * 4, spatial, 1, stride))
            ch_in = width * 4
            spatial = out_hw
    assert len(want) == 53

    for rec, (tag, ic, oc, hw, k, s) in zip(convs, want):
        act, weight, bias = rec["OPERAND_SHAPES"]
        oh = hw // s
        assert act == (1, 1, hw * hw, ic), "%s (%s): activation %s, topology says %s" % (
            rec["file"],
            tag,
            act,
            (1, 1, hw * hw, ic),
        )
        assert weight == (oc, ic, k, k), "%s (%s): weight %s, topology says %s" % (
            rec["file"],
            tag,
            weight,
            (oc, ic, k, k),
        )
        assert bias == (1, 1, 1, oc), "%s (%s): bias %s, topology says %s" % (rec["file"], tag, bias, (1, 1, 1, oc))
        assert rec["OUTPUT_SHAPE"] == (1, 1, oh * oh, oc), "%s (%s): output %s, topology says %s" % (
            rec["file"],
            tag,
            rec["OUTPUT_SHAPE"],
            (1, 1, oh * oh, oc),
        )
        # the file name must name the module it replays
        assert tag.replace(".", "_") in rec["file"], "%s does not name the conv it replays (%s)" % (rec["file"], tag)


def test_residual_files_match_resnet50_topology():
    """The 16 adds and the 16 relus follow the bottleneck widths, and each relu sits right after its add."""
    files = _scan_op_files()
    want, hw = [], 56
    for layer, (blocks, width) in enumerate(zip([3, 4, 6, 3], [256, 512, 1024, 2048]), start=1):
        if layer > 1:
            hw //= 2
        want += [(1, 1, hw * hw, width)] * blocks
    assert len(want) == 16

    adds = [files[k] for k in sorted(files) if files[k]["FORGE_OP"] == "ttnn.add"]
    relus = [files[k] for k in sorted(files) if files[k]["FORGE_OP"] == "ttnn.relu"]
    assert len(adds) == len(relus) == 16, "expected 16 adds and 16 relus, found %d and %d" % (len(adds), len(relus))

    for rec, shape in zip(adds, want):
        assert rec["OPERAND_SHAPES"] == (shape, shape), "%s: operands %s, residual topology says two of %s" % (
            rec["file"],
            rec["OPERAND_SHAPES"],
            shape,
        )
        assert rec["OUTPUT_SHAPE"] == shape, "%s: output %s, expected %s" % (rec["file"], rec["OUTPUT_SHAPE"], shape)
    for rec, shape in zip(relus, want):
        assert rec["OPERAND_SHAPES"] == (shape,), "%s: operand %s, expected %s" % (
            rec["file"],
            rec["OPERAND_SHAPES"],
            (shape,),
        )
    for add, relu in zip(adds, relus):
        assert relu["SHEET_ROW"] == add["SHEET_ROW"] + 1, "%s does not sit directly after %s" % (
            relu["file"],
            add["file"],
        )


def test_every_op_file_is_shape_consistent():
    """Cheap whole-directory sanity: shapes are non-empty tuples of positive ints."""
    files = _scan_op_files()
    for rec in files.values():
        shapes = list(rec["OPERAND_SHAPES"]) + [rec["OUTPUT_SHAPE"]]
        for s in shapes:
            assert isinstance(s, tuple) and s, "%s declares a bad shape %r" % (rec["file"], s)
            assert all(isinstance(d, int) and d > 0 for d in s), "%s declares a bad shape %r" % (rec["file"], s)


# --------------------------------------------------------------------------------------------------
# host-only: the live ttnn build
# --------------------------------------------------------------------------------------------------
def test_quasar_arch_is_registered():
    """Quasar is a registered architecture in this ttnn build and its op namespace is bound."""
    assert hasattr(ttnn, "Arch") and hasattr(ttnn.Arch, "QUASAR"), (
        "this ttnn build does not register Arch.QUASAR (it knows %s). Nothing in this directory can "
        "run against a Quasar part." % [m for m in dir(ttnn.Arch) if not m.startswith("_")]
    )
    assert hasattr(ttnn.experimental, "quasar"), "ttnn.experimental.quasar is not bound in this build"

    bound = sorted(n for n in dir(ttnn.experimental.quasar) if not n.startswith("_"))
    print("\nttnn.experimental.quasar binds %d names:\n  %s" % (len(bound), ", ".join(bound)))
    assert len(bound) > 50, "only %d names bound under ttnn.experimental.quasar -- the build looks truncated" % len(
        bound
    )


def test_forge_ops_map_onto_the_live_quasar_build():
    """Every Forge op either resolves to a live quasar op or is one of the three named gaps."""
    q = ttnn.experimental.quasar
    missing, closed = [], []
    for forge_op, quasar_name in sorted(FORGE_TO_QUASAR.items()):
        short = forge_op.split(".", 1)[1]
        if quasar_name is None:
            if hasattr(q, short):
                closed.append(forge_op)
            continue
        if not hasattr(q, quasar_name):
            missing.append("%s -> quasar.%s" % (forge_op, quasar_name))

    assert not missing, (
        "a quasar op this directory depends on has disappeared from the build: %s. The matching op "
        "files cannot run." % missing
    )
    assert not closed, (
        "a documented Quasar gap has CLOSED: %s now exists. This test is the ONE place that watches for "
        "that -- the op files carry no xfail and no probe, they just run the workaround route. Update "
        "FORGE_TO_QUASAR and add a direct test for the real op in the matching op file(s)." % closed
    )


def test_workaround_ops_exist():
    """The three gaps are routed through these; if one goes, the workaround tests are dead code."""
    q = ttnn.experimental.quasar
    gone = [n for n in WORKAROUND_OPS if not hasattr(q, n)]
    assert not gone, "the workaround route depends on quasar.%s, which is not bound" % gone


# --------------------------------------------------------------------------------------------------
# device
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_device_under_test_is_quasar(mesh_device):
    """
    The device this suite opened must be a Quasar part.

    This is a Quasar suite: every op it calls lives under ttnn.experimental.quasar, and those ops
    build Gen2 kernels that TT_FATAL on a Wormhole device. Running the sweeps against anything else
    reports failures that say nothing about Quasar, so the arch is asserted here rather than left
    to be discovered 141 files later.

    To open a Quasar device with the craq-sim functional simulator:
        TT_METAL_SIMULATOR=<dir>/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 ARCH_NAME=quasar pytest ...
    (libttsim.so must sit beside its soc_descriptor.yaml -- UMD reads the descriptor from the same
    directory.) To run one of the op files on Wormhole as a control instead, deselect this test with
    -k "not is_quasar" and expect the quasar ops themselves to fail.
    """
    device = mesh_device
    arch = device.arch()
    grid = device.compute_with_storage_grid_size()
    print("\ndevice under test: arch=%s compute_with_storage_grid=%sx%s" % (arch, grid.x, grid.y))

    assert arch == ttnn.Arch.QUASAR, (
        "the open device reports %s, not Arch.QUASAR. Every op in this directory is a "
        "ttnn.experimental.quasar op and will not run here -- see this test's docstring for how to "
        "open a Quasar device." % arch
    )
    assert grid.x >= 1 and grid.y >= 1, "the Quasar device reports an empty compute grid: %sx%s" % (grid.x, grid.y)

    # Nothing in this directory pins a core range -- the whole compile is DRAM interleaved -- so any
    # Quasar grid is usable and no test is ever skipped for grid size.
    print("no file in this directory pins a core range; all %d workers are available" % (grid.x * grid.y))
