# SPDX-License-Identifier: Apache-2.0
"""check_lever_coverage: a lever (dtype knob / kernel swap) detected on the representative layer slice
must reach EVERY layer instance. The full-depth op-signature MULTISET tells us if all N instances
changed; the ordered sequence lets us segment the run into repeated blocks and name which blocks a
PARTIAL application missed (so the agent moves the edit to the shared block definition and reapplies)."""

import importlib.util
from collections import Counter
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "perf_mcp_cov", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py")
)
P = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(P)


def _seq(mm_dtypes):
    """A 4-block model: embed, then per block [norm, matmul<dtype>, attn], then lm_head."""
    s = ["embed(((1,4096),))"]
    for dt in mm_dtypes:
        s += ["norm(((1,4096),))", "matmul(((1,4096),),'%s')" % dt, "attn(((1,4096),))"]
    s += ["lmhead(((1,4096),))"]
    return s


def test_fully_applied_all_instances_changed():
    seq = _seq(["BFLOAT8_B"] * 4)
    r = P.compute_lever_coverage(Counter(seq), seq, "matmul(", "BFLOAT16", "BFLOAT8_B")
    assert r["fully_applied"] is True
    assert r["total_instances"] == 4 and r["applied"] == 4 and r["stale_remaining"] == 0
    assert r["missed_blocks"] == []


def test_partial_application_names_the_missed_block():
    # block index 2 was left on the old dtype (edit hit an instance-specific path, not the shared block)
    seq = _seq(["BFLOAT8_B", "BFLOAT8_B", "BFLOAT16", "BFLOAT8_B"])
    r = P.compute_lever_coverage(Counter(seq), seq, "matmul(", "BFLOAT16", "BFLOAT8_B")
    assert r["fully_applied"] is False
    assert r["stale_remaining"] == 1 and r["applied"] == 3 and r["total_instances"] == 4
    assert r["n_blocks"] == 4 and r["block_source"] == "inferred"  # no signposts here -> anchor fallback
    assert r["missed_blocks"] == [2]  # segmented from the ordered op stream, no model internals


def test_signposts_give_exact_block_attribution():
    # with REAL per-block signposts in the stream, segmentation is exact (not inferred): each block's
    # ops sit between its signpost and the next. Block 1 (0-based) left on the old dtype here.
    seq = ["embed(...)"]
    for i, dt in enumerate(["BFLOAT8_B", "BFLOAT16", "BFLOAT8_B"]):
        seq += ["PERF_BLOCK_SIGNPOST:%d" % i, "norm(...)", "matmul(((1,4096),),'%s')" % dt, "attn(...)"]
    r = P.compute_lever_coverage(Counter(seq), seq, "matmul(", "BFLOAT16", "BFLOAT8_B")
    assert r["block_source"] == "signposts" and r["n_blocks"] == 3
    assert r["fully_applied"] is False and r["missed_blocks"] == [1]


def test_decoupled_signposts_are_distrusted_and_fall_back_to_inferred():
    seq = ["embed(...)"]
    for dt in ["BFLOAT8_B", "BFLOAT16", "BFLOAT8_B"]:
        seq += ["norm(...)", "matmul(((1,4096),),'%s')" % dt, "attn(...)"]
    seq += ["PERF_BLOCK_SIGNPOST:%d" % i for i in range(20)]
    r = P.compute_lever_coverage(Counter(seq), seq, "matmul(", "BFLOAT16", "BFLOAT8_B")
    assert r["block_source"] == "inferred"
    assert r["fully_applied"] is False and r["missed_blocks"] == [1]


def test_op_not_found():
    seq = _seq(["BFLOAT8_B"] * 4)
    r = P.compute_lever_coverage(Counter(seq), seq, "conv2d(", "BFLOAT16", "BFLOAT8_B")
    assert r["status"] == "not_found"


def test_no_dtype_markers_is_inconclusive_not_false():
    # a grid/program_config lever is not tensor-visible -> can't verify by signature, must not claim failure
    seq = _seq(["BFLOAT8_B"] * 4)
    r = P.compute_lever_coverage(Counter(seq), seq, "matmul(", "", "")
    assert r["fully_applied"] is None and r["status"] == "ok"
