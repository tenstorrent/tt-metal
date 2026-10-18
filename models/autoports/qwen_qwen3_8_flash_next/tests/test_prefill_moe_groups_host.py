# SPDX-License-Identifier: Apache-2.0
"""Host-only capacity/address and opt-in grouping contracts; no TTNN import."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tt/multichip_decoder.py"


def namespace():
    tree = ast.parse(SOURCE.read_text())
    wanted = {
        "_EXPERT_HEADER",
        "_EXPERT_DISPATCH_PACKED",
        "_EXPERT_COMPACT_PACKED",
        "_EXPERT_DISPATCH_PACKED_256",
        "_EXPERT_COMPACT_PACKED_256",
        "_EXPERT_DISPATCH_PACKED_512",
        "_EXPERT_COMPACT_PACKED_512",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets)
        or isinstance(node, ast.FunctionDef)
        and node.name == "_expert_capacity_source"
    ]
    result = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), result)
    return result


def method(name, **globals_):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ParallelMoE")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), globals_)
    return globals_[name]


def test_capacity256_changes_only_three_qualified_kernel_fragments():
    ns = namespace()
    assert ns["_EXPERT_DISPATCH_PACKED_256"].replace("e*256+count", "e*128+count") == ns["_EXPERT_DISPATCH_PACKED"]
    assert (
        ns["_EXPERT_COMPACT_PACKED_256"]
        .replace("ASSERT(packed<256)", "ASSERT(packed<128)")
        .replace("(e%128)*8+packed/32", "(e%128)*4+packed/32")
        == ns["_EXPERT_COMPACT_PACKED"]
    )
    # Every expert's worst-case256 input rows and8 output tile rows fit exactly,
    # and adjacent expert regions do not overlap at the old128 boundary.
    for expert in range(128):
        assert expert * 256 + 255 < (expert + 1) * 256
        assert (expert * 8 + 255 // 32) * 80 + 79 < (expert + 1) * 8 * 80
    for source in ("missing", "old old"):
        try:
            ns["_expert_capacity_source"](source, "old", "new")
        except ValueError:
            pass
        else:
            raise AssertionError("Source drift must be rejected")


def test_native_grouping_unchanged_and_view_opt_in_does_not_mutate_owner():
    call = method("__call__", ttnn=SimpleNamespace(concat=lambda xs, dim: torch.cat(xs, dim=dim)))
    observed = []
    owner = SimpleNamespace(
        ops=SimpleNamespace(scatter_reduce=lambda value: value),
        group_packed=lambda value: observed.append(value.shape[2]) or value,
        group_native=lambda value: observed.append("native") or value,
    )
    value = torch.zeros(1, 1, 512, 1)
    call(owner, value)
    assert observed == [128, 128, 128, 128]
    view = SimpleNamespace(**vars(owner), prefill_group_size=256)
    observed.clear()
    call(view, value)
    assert observed == [256, 256]
    view512 = SimpleNamespace(**vars(owner), prefill_group_size=512)
    observed.clear()
    call(view512, value)
    assert observed == [512]
    observed.clear()
    call(view512, value[:, :, :256])
    assert observed == [256]
    observed.clear()
    call(view, value[:, :, :128])
    call(owner, value[:, :, :1])
    assert observed == [128, "native"]
    assert not hasattr(owner, "prefill_group_size")


def test_capacity_rejects_unprepared_or_oversized_groups_before_work():
    call = method("group_packed")
    for owner, n in [
        (SimpleNamespace(), 129),
        (SimpleNamespace(prefill_group_size=256), 257),
        (SimpleNamespace(prefill_group_size=512), 513),
        (SimpleNamespace(), 0),
    ]:
        try:
            call(owner, SimpleNamespace(shape=(1, 1, n, 2560)))
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal capacity must be rejected before allocating or dispatching")


def test_capacity512_changes_only_qualified_strides_and_bounds():
    ns = namespace()
    assert ns["_EXPERT_DISPATCH_PACKED_512"].replace("e*512+count", "e*128+count") == ns["_EXPERT_DISPATCH_PACKED"]
    assert (
        ns["_EXPERT_COMPACT_PACKED_512"]
        .replace("ASSERT(packed<512)", "ASSERT(packed<128)")
        .replace("(e%128)*16+packed/32", "(e%128)*4+packed/32")
        == ns["_EXPERT_COMPACT_PACKED"]
    )
    for expert in range(128):
        assert expert * 512 + 511 < (expert + 1) * 512
        assert (expert * 16 + 511 // 32) * 80 + 79 < (expert + 1) * 16 * 80
