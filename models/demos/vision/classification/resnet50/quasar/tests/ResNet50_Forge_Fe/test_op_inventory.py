# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
HOST-ONLY triage for the ResNet50_Forge_Fe suite. No device, no kernels -- runs in about a second
and answers the question this folder exists for, before anything is put on hardware:

    of the 10 distinct ops the tt-forge ResNet-50 graph issues, which does
    ttnn.experimental.quasar actually expose?

It resolves every entry of FORGE_TO_QUASAR against the LIVE ttnn build rather than against a
hardcoded list, so the mapping cannot silently rot when the quasar bindings change:

  * a Forge op mapped to a quasar op that has vanished  -> failure, the per-op test is now lying
  * a Forge op mapped to None whose quasar op has ARRIVED -> failure, the gap has closed and the
    mapping (and the corresponding *_forge.py gap test) needs updating

The graph's op counts are also frozen here, because the per-op files each cover one op and none of
them can see whether the whole graph is still accounted for.

RUN
    pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe/test_op_inventory.py
"""

import ttnn

# --- the tt-forge ResNet-50 graph, from the TTNN IR (@forward, 96 compute ops) --------------------
# Cross-checked against the EmitPy render of the same module, proof/emitpy/B_ttnn_route.py.
FORGE_OP_COUNTS = {
    "ttnn.conv2d": 53,
    "ttnn.add": 16,
    "ttnn.relu": 16,
    "ttnn.to_memory_config": 4,
    "ttnn.reshape": 2,
    "ttnn.to_layout": 1,
    "ttnn.permute": 1,
    "ttnn.max_pool2d": 1,
    "ttnn.mean": 1,
    "ttnn.linear": 1,
}
# Plus 276 ttnn.deallocate and, in the 106 const-eval functions, 53 prepare_conv2d_weights +
# 53 prepare_conv2d_bias. Quasar exposes neither prepare entry point; quasar.conv2d prepares
# internally and test_conv2d_forge.py checks the two agree on the prepared shape.
FORGE_DEALLOCATE_COUNT = 276
FORGE_PREPARE_WEIGHTS_COUNT = 53
FORGE_PREPARE_BIAS_COUNT = 53

# The Forge op -> quasar op mapping this folder asserts against. A None value is a REAL GAP: the
# Forge graph issues that op and ttnn.experimental.quasar has no equivalent.
FORGE_TO_QUASAR = {
    "ttnn.conv2d": "conv2d",
    "ttnn.add": "add",
    "ttnn.relu": None,  # no plain unary activation is bound under quasar
    "ttnn.to_memory_config": "to_memory_config",
    "ttnn.reshape": "reshape",
    "ttnn.to_layout": "to_layout",
    "ttnn.permute": None,  # only `transpose`, a 2-axis swap, is bound
    "ttnn.max_pool2d": "max_pool2d",
    "ttnn.mean": None,  # the quasar reduction has a device backend but no python binding
    "ttnn.linear": "linear",
}

# What the model does instead, for the three gaps. These are the routes the *_forge.py workaround
# tests exercise.
FORGE_GAP_WORKAROUND = {
    "ttnn.relu": "fuse it: quasar.add(..., activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)])"
    " -- test_relu_forge.py::test_forge_relu_fused_into_add",
    "ttnn.permute": "decompose it: quasar.transpose(1, 2) then quasar.transpose(2, 3) gives"
    " NCHW->NHWC -- test_permute_forge.py::test_forge_permute_via_transpose",
    "ttnn.mean": "call the generic ttnn.mean, which dispatches to the quasar reduce on a quasar"
    " device -- test_mean_forge.py::test_forge_mean_generic",
}

TOTAL_COMPUTE_OPS = 96


def _quasar_ns():
    return getattr(getattr(ttnn, "experimental", None), "quasar", None)


def test_quasar_namespace_exists():
    """Nothing in this folder can run without ttnn.experimental.quasar. Print what it holds."""
    ns = _quasar_ns()
    assert ns is not None, "this ttnn build has no ttnn.experimental.quasar namespace -- nothing in this folder can run"
    names = sorted(n for n in dir(ns) if not n.startswith("_"))
    print("\nttnn.experimental.quasar exposes %d names:" % len(names))
    for i in range(0, len(names), 6):
        print("    " + "  ".join("%-26s" % n for n in names[i : i + 6]))


def test_forge_graph_op_counts_are_accounted_for():
    """Every compute op in the graph must have a mapping decision, and the counts must add up."""
    assert (
        sum(FORGE_OP_COUNTS.values()) == TOTAL_COMPUTE_OPS
    ), "the op counts sum to %d, but @forward has %d compute ops" % (sum(FORGE_OP_COUNTS.values()), TOTAL_COMPUTE_OPS)
    assert set(FORGE_OP_COUNTS) == set(FORGE_TO_QUASAR), (
        "FORGE_TO_QUASAR is out of sync with the graph.\n"
        "  in graph, unmapped  : %s\n"
        "  mapped, not in graph: %s"
        % (
            sorted(set(FORGE_OP_COUNTS) - set(FORGE_TO_QUASAR)),
            sorted(set(FORGE_TO_QUASAR) - set(FORGE_OP_COUNTS)),
        )
    )
    assert set(FORGE_GAP_WORKAROUND) == {
        op for op, q in FORGE_TO_QUASAR.items() if q is None
    }, "every gap needs a documented workaround, and only gaps should have one"


def test_mapped_quasar_ops_still_exist():
    """A Forge op mapped to a quasar op that has vanished means the per-op test is now lying."""
    ns = _quasar_ns()
    missing = {
        forge_op: quasar_op
        for forge_op, quasar_op in FORGE_TO_QUASAR.items()
        if quasar_op is not None and getattr(ns, quasar_op, None) is None
    }
    assert not missing, (
        "FORGE_TO_QUASAR maps these Forge ops onto quasar ops that no longer exist: %s.\n"
        "The corresponding *_forge.py test cannot work until the mapping is fixed." % missing
    )


def test_known_gaps_are_still_gaps():
    """
    The three gaps, checked against the live build.

    A failure here is GOOD NEWS -- a quasar binding landed. Fix it by pointing FORGE_TO_QUASAR at
    the new op; the matching gap test in the per-op file then starts exercising it for real (each
    one resolves the op at runtime and only xfails when it is absent).
    """
    ns = _quasar_ns()
    closed = {}
    for forge_op, quasar_op in FORGE_TO_QUASAR.items():
        if quasar_op is not None:
            continue
        name = forge_op.split(".", 1)[1]
        if getattr(ns, name, None) is not None:
            closed[forge_op] = "ttnn.experimental.quasar.%s" % name
    assert not closed, (
        "these gaps have CLOSED -- quasar now exposes %s. Update FORGE_TO_QUASAR (and drop the "
        "workaround note); the gap tests will start running the real op." % closed
    )


def test_gap_summary():
    """Print the support matrix this folder produces, so a run is self-documenting."""
    ns = _quasar_ns()
    print("\nForge op                 count  quasar op            route")
    print("-" * 78)
    for forge_op in sorted(FORGE_OP_COUNTS, key=lambda k: -FORGE_OP_COUNTS[k]):
        quasar_op = FORGE_TO_QUASAR[forge_op]
        if quasar_op is None:
            route = "GAP -- %s" % FORGE_GAP_WORKAROUND[forge_op].split(" -- ")[0]
            shown = "-- none --"
        else:
            route = "direct" if getattr(ns, quasar_op, None) is not None else "MAPPED BUT MISSING"
            shown = quasar_op
        print("%-24s %5d  %-20s %s" % (forge_op, FORGE_OP_COUNTS[forge_op], shown, route))
    print("-" * 78)
    print(
        "%-24s %5d  (+ %d deallocate, %d prepare_conv2d_weights, %d prepare_conv2d_bias)"
        % (
            "TOTAL",
            TOTAL_COMPUTE_OPS,
            FORGE_DEALLOCATE_COUNT,
            FORGE_PREPARE_WEIGHTS_COUNT,
            FORGE_PREPARE_BIAS_COUNT,
        )
    )
