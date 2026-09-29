# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of the fused programs' meta (the census's legibility of ``GenericOp`` rows): every launch under
``ttnn/fused`` passes a ``program_meta`` built from its builder's shapes and constants, the byte and FLOP sums follow
the tensors the descriptors are built from, and the recording joins launches to device operation ids in call order
(off by default: the served process never pays for it)."""

import ast
from pathlib import Path
from types import SimpleNamespace

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import moe_post
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

FUSED = Path(fp.__file__).parent
MODULES = sorted(FUSED.glob("*/__init__.py"))
KERNELS = 23  # sub-packages under ttnn/fused
LAUNCHES = 65  # run_program call sites over them (every one passes its meta)
# a program's ``kernel`` is a registered name; the QSA block's mirrors and probes that are not a kernel of their own say
# ``qsa_block``; the names bound in the modules that hold a kernel name
KERNEL_NAMES = set(fused.kernels()) | {"qsa_block"}
KERNEL_NAME_BINDINGS = {"NAME", "ADVANCE_NAME", "CANDIDATE_ROW", "kernel"}


def _dotted(node) -> str:
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return node.id if isinstance(node, ast.Name) else ""


def _launches(tree):
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _dotted(n.func) in ("fp.run_program", "ttnn.generic_op")
    ]


def _variant_names(node) -> set[str]:
    """The string constants a ``variant`` argument can take (a constant or a conditional of constants)."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _variant_names(node.body) | _variant_names(node.orelse)
    raise AssertionError(f"variant must be a string constant or a conditional of them, got {ast.dump(node)}")


def test_every_fused_launch_passes_its_meta():
    assert len(MODULES) == KERNELS
    launches = 0
    for path in MODULES:
        tree = ast.parse(path.read_text())
        for call in _launches(tree):
            assert (
                _dotted(call.func) == "fp.run_program"
            ), f"{path.parent.name}: launch generic_op through fp.run_program"
            assert any(k.arg == "meta" for k in call.keywords), f"{path.parent.name}:{call.lineno} passes no meta"
            launches += 1
    assert launches == LAUNCHES


def test_every_meta_names_a_kernel_and_a_distinct_variant():
    seen: dict[str, set[str]] = {}
    for path in MODULES:
        tree = ast.parse(path.read_text())
        metas = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _dotted(n.func) == "fp.program_meta"]
        if not _launches(
            tree
        ):  # a family of ttnn ops and another package's programs (qsa_rows) launches nothing itself
            assert not metas, f"{path.parent.name} builds a program_meta without a launch"
            continue
        assert metas, f"{path.parent.name} builds no program_meta"
        variants: list[str] = []
        for call in metas:
            kernel, variant = call.args[0], call.args[1]
            if isinstance(kernel, ast.Constant):
                assert kernel.value in KERNEL_NAMES, f"{path.parent.name}: unknown kernel {kernel.value!r}"
            else:
                assert isinstance(kernel, ast.Name) and kernel.id in KERNEL_NAME_BINDINGS, ast.dump(kernel)
            variants.extend(sorted(_variant_names(variant)))
            assert all(
                k.arg in {"reads", "writes", "partial", "flops", "dram_bytes", "l1_bytes", "cores", "outputs"}
                for k in call.keywords
            )
        assert len(variants) == len(set(variants)), f"{path.parent.name}: a variant name is used twice: {variants}"
        seen[path.parent.name] = set(variants)
    # the served step's programs are legible by these names (the census table's rows)
    assert {"post", "post_sigmoid"} <= seen["moe_post"]
    assert {"stats_normalize_down_gather", "normalize_down_gather", "stats_gather", "gather_line"} <= seen["gr_fold"]
    assert seen["gr_recip_last"] == {"stats_recip_down_gather"}
    assert {"normalize_down", "low_rank_gate", "normalize", "down_project", "low_rank", "gate", "stats"} <= seen[
        "gr_read"
    ]
    assert {"index_tail", "main_tail", "post_attention", "widen_partial", "selection_row", "score_merge"} <= seen[
        "qsa_block"
    ]
    assert {"scan", "merge", "resolve"} <= seen["greedy_tail"] and {"derive", "derive_lanes", "advance"} <= seen[
        "position_derive"
    ]
    assert {"lanes", "single_core"} <= seen["router_tail"] and seen["gdn_step"] == {"step"}
    assert seen["moe_dense"] == {"composite", "composed_eltwise"}
    # the prefill slab's rows programs (opt-in): the slab and the one-tile-row verify form of each
    assert seen["gdn_pre_rows"] == {"slab", "verify_rows"}
    assert seen["gdn_post_rows"] == {"cast", "cast_verify", "norm", "norm_verify"}
    # the verify rows' fold (a default): the scan and its commit pick, and their lanes forms
    assert seen["gdn_rows_scan"] == {"verify_rows", "commit_pick", "verify_rows_lanes", "commit_pick_lanes"}
    assert seen["qsa_rows"] == {"score_pages", "post_attention_rows"}  # programs 1-3 launch through qsa_block


def _fake(shape, dtype, *, l1=False, padded=None):
    buffer_type = ttnn.BufferType.L1 if l1 else ttnn.BufferType.DRAM
    return SimpleNamespace(
        shape=tuple(shape),
        padded_shape=tuple(padded or shape),
        dtype=dtype,
        memory_config=lambda: SimpleNamespace(buffer_type=buffer_type),
    )


def test_tensor_bytes_follow_the_padded_shape_and_the_dtype():
    assert fp.tensor_bytes(_fake((1, 1, 1, 2560), ttnn.bfloat16, padded=(1, 1, 32, 2560))) == 32 * 2560 * 2
    assert fp.tensor_bytes(_fake((4, 1, 1, 384), ttnn.float32, padded=(4, 1, 32, 384))) == 4 * 32 * 384 * 4
    assert fp.tensor_bytes(_fake((1, 1, 1, 10), ttnn.uint16)) == 20  # a ROW_MAJOR routing row: no tile padding
    assert fp.tensor_bytes(_fake((1, 1, 32, 32), ttnn.bfloat8_b)) == fp.TILE_BYTES[ttnn.bfloat8_b]
    assert fp.tensor_bytes(_fake((1, 1, 64, 64), ttnn.bfloat4_b)) == 4 * fp.TILE_BYTES[ttnn.bfloat4_b]
    assert fp.in_l1(_fake((1,), ttnn.uint32, l1=True)) and not fp.in_l1(_fake((1,), ttnn.uint32))
    assert fp.in_l1(SimpleNamespace(shape=(1,), dtype=ttnn.uint32)) is False  # a host fake counts as DRAM


def test_program_meta_sums_the_tensors_by_where_they_live():
    dram_row = _fake((1, 1, 1, 2560), ttnn.bfloat16, padded=(1, 1, 32, 2560))  # 163840 B
    l1_shard = _fake((1, 1, 1, 3072), ttnn.bfloat16, padded=(1, 1, 32, 3072), l1=True)  # 196608 B
    out = _fake((1, 1, 1, 320), ttnn.float32, padded=(1, 1, 32, 320))  # 40960 B
    meta = fp.program_meta(
        "gr_read",
        "probe",
        3,
        reads=(dram_row, l1_shard),
        writes=(out,),
        partial=((l1_shard, 4096), (dram_row, 64)),
        flops=7,
        dram_bytes=1000,
        l1_bytes=10,
        cores=12,
    )
    assert meta == fp.FusedProgramMeta("gr_read", "probe", 3, 163840 + 40960 + 64 + 1000, 196608 + 4096 + 10, 7, 12)
    assert fp.program_meta("x", "y", 1).dram_bytes == 0 and fp.program_meta("x", "y", 1).flops == 0
    for bad in ({"rows": 0}, {"dram_bytes": -1}, {"flops": -1}):
        try:
            fp.program_meta("x", "y", **{"rows": 1, **bad}) if "rows" not in bad else fp.program_meta("x", "y", 0)
        except ValueError:
            continue
        raise AssertionError(f"program_meta accepted {bad}")


def test_moe_post_meta_by_construction():
    """The sums follow the descriptor's operands: the pages at their bound, the routing rows and the owner row once
    per core (to L1 when moe_compute's drain-core shard holds them), the shared partial, the sigmoid tile, the sum."""

    rows = 5
    pages = _fake((10, rows, 2560), ttnn.bfloat16)
    scores, indices = _fake((1, rows, 10), ttnn.bfloat16, l1=True), _fake((1, rows, 10), ttnn.uint16, l1=True)
    owner = _fake((1, 512), ttnn.uint16)
    shared = _fake((1, 1, rows, 2560), ttnn.bfloat16, padded=(1, 1, 32, 2560))
    sigmoid = _fake((1, 1, 32, 32), ttnn.bfloat16)
    out = _fake((1, 1, rows, 2560), ttnn.bfloat16, padded=(1, 1, 32, 2560))
    meta = moe_post.moe_post_meta(pages, scores, indices, owner, shared, sigmoid, out, rows=rows)
    routing_per_core = 2 * rows * 10 * 2 + 512 * 2
    assert meta.kernel == "moe_post" and meta.variant == "post_sigmoid" and meta.rows == rows and meta.cores == 80
    assert meta.dram_bytes == 10 * rows * 2560 * 2 + 2 * 32 * 2560 * 2 + 32 * 32 * 2
    assert meta.l1_bytes == 80 * routing_per_core
    assert meta.flops == rows * 2560 * (2 * 10 + 2)
    plain = moe_post.moe_post_meta(
        pages,
        _fake((1, 1, rows, 10), ttnn.bfloat16),
        _fake((1, 1, rows, 10), ttnn.uint16),
        owner,
        shared,
        None,
        out,
        rows=rows,
    )
    assert plain.variant == "post" and plain.l1_bytes == 0 and plain.flops == rows * 2560 * (2 * 10 + 1)
    assert plain.dram_bytes == 10 * rows * 2560 * 2 + 2 * 32 * 2560 * 2 + 80 * routing_per_core


def test_recording_joins_launches_to_device_operation_ids_in_call_order(monkeypatch):
    counter = [100]
    launched = []

    def generic_op(io, descriptor):
        launched.append((len(io), descriptor))
        counter[0] += 1  # the launch takes one device operation id
        return io[-1]

    monkeypatch.setattr(ttnn, "generic_op", generic_op)
    monkeypatch.setattr(fp, "_device_operation_id", lambda: counter[0])
    fp.reset_program_meta()
    assert fp.program_meta_recording() is False  # the default: no host cost in the served process
    first = fp.program_meta("gr_write", "write", 1, flops=3)
    assert fp.run_program(["a", "b"], "d1", meta=first) == "b" and fp.program_meta_records() == ()
    fp.record_program_meta(True)
    try:
        assert fp.program_meta_recording() is True
        fp.run_program(["a", "b"], "d2", meta=first)
        fp.run_program(["a"], "d3")  # no meta: launched, not recorded
        second = fp.program_meta("moe_post", "post", 2)
        fp.run_program(["a", "b", "c"], "d4", meta=second)
        records = fp.program_meta_records()
        assert [(r.first_id, r.end_id, r.meta) for r in records] == [(101, 102, first), (103, 104, second)]
        assert records[0].covers(101) and not records[0].covers(102) and records[1].covers(103)
        assert [d for _, d in launched] == ["d1", "d2", "d3", "d4"]
        fp.reset_program_meta()
        assert fp.program_meta_records() == ()
    finally:
        fp.record_program_meta(False)
    assert fp.program_meta_recording() is False
