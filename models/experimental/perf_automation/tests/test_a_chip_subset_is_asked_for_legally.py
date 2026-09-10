"""A subset request must narrow VISIBILITY, and may only do so with the descriptor that legalises it.

tt-metal sizes a model with `{<labels>}.get(MESH_DEVICE, len(ttnn.get_device_ids()))`. The upstream
table (models/tt_transformers/conftest.py, from PR #39080) lists no Blackhole shape but the 8-chip
one, so on a Blackhole host every label this tool emits MISSES and falls through to the number of
VISIBLE chips. Visibility was deliberately left unrestricted -- pinning it makes fabric discovery
call the board a CUSTOM cluster and refuse to start without a mesh-graph descriptor -- so the
fallback always saw the whole host.

Measured 2026-08-29: `--devices 0 --mesh 1x1` against a demo declaring {"chips": 1, "tp": 1,
"mesh": [1, 1]} ran on FOUR chips at 85W each with the ethernet fabric up, reached 99-103C, and two
chips stopped answering. tt-metal ships 52 descriptors; pairing the matching one with the visibility
setting makes the subset legal, and the model then sizes itself correctly through the same fallback.
"""
from __future__ import annotations

from pathlib import Path

from models.experimental.perf_automation.agent.before_loop import _requested_chip_count
from models.experimental.perf_automation.agent.mesh_descriptor import find_descriptor

_TT_ROOT = Path(__file__).resolve().parents[4]


def test_what_devices_asks_for():
    assert _requested_chip_count("0") == 1
    assert _requested_chip_count("single") == 1
    assert _requested_chip_count("0,1") == 2
    assert _requested_chip_count("0,1,2,3") == 4


def test_a_request_for_the_whole_host_never_narrows_visibility():
    """`all` must keep today's behaviour exactly: no pinning, no descriptor, no CUSTOM cluster."""
    for text in ("all", "", None, "garbage"):
        assert _requested_chip_count(text) is None, "%r must not be read as a chip count" % (text,)


def test_a_descriptor_is_found_by_what_it_declares_not_by_its_name():
    """Filename matching would miss a new board; these are matched on arch + declared dims."""
    got = find_descriptor(_TT_ROOT, "Blackhole", 1)
    assert got is not None, "no single-chip Blackhole descriptor found; a subset request cannot be legalised"
    text = got.read_text(errors="replace")
    assert "BLACKHOLE" in text, "%s is not a Blackhole descriptor" % got.name
    assert "dims:" in text and "[ 1, 1 ]" in text.replace("[1,1]", "[ 1, 1 ]"), (
        "%s does not declare a 1x1 topology" % got.name
    )


def test_the_chip_count_asked_for_is_the_chip_count_declared():
    for chips in (1, 2, 4):
        got = find_descriptor(_TT_ROOT, "Blackhole", chips)
        if got is None:
            continue
        import re

        dims = re.search(r"device_topology\s*\{\s*dims:\s*\[([^\]]*)\]", got.read_text(errors="replace"))
        product = 1
        for n in re.findall(r"\d+", dims.group(1)):
            product *= int(n)
        assert product == chips, "%s declares %d chips, asked for %d" % (got.name, product, chips)


def test_an_unknown_arch_yields_no_descriptor_rather_than_a_wrong_one():
    """None is the safe answer -- the caller then leaves visibility alone instead of crashing."""
    assert find_descriptor(_TT_ROOT, "NotARealArch", 1) is None
    assert find_descriptor(_TT_ROOT, "Blackhole", 0) is None
    assert find_descriptor(_TT_ROOT, "", 1) is None
