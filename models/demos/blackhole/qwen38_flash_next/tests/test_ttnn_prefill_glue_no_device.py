# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The prefill slab's glue forms (``QWEN38_PREFILL_GLUE``) without a device: the policy's resolution (the bitwise
default set, ``today``, an exact list), the rule that every form is read under a slab condition only (so the decode
path and the 32/128-row bodies never see one), the previous forms' chains kept in place behind ``today``, the hoisted
QSA block masks against the per-position emulation with their admission (the byte cap, the free DRAM) and fallback,
and the flat q/k rows state as a slab option.
"""

from __future__ import annotations

import ast
import inspect
import logging
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from loguru import logger as loguru_logger

from models.demos.blackhole.qwen38_flash_next.tests import test_ttnn_qsa_chunk_no_device as qsa_test
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gr as gr_module
from models.demos.blackhole.qwen38_flash_next.ttnn import prefill_glue
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import DEFAULT_SLAB_ROWS

SLAB = DEFAULT_SLAB_ROWS
ENV = prefill_glue.ENV
MODULES = {"gr": gr_module, "qsa": qsa_module, "gdn": gdn_module}
SLAB_UINT32_TEMPLATES = (
    "page_offsets",
    "row_index_col",
    "row_index_slots",
    "arange_slots_rows",
    "all_ones_rows",
    "arange_blocks_row",
    "row_index_row",
)


def _policy(*names: str) -> prefill_glue.PrefillGluePolicy:
    return prefill_glue.PrefillGluePolicy.from_environ({ENV: ",".join(names)})


# --------------------------------------------------------------------------- the policy


def test_default_policy_is_the_bitwise_set_today_is_none_and_a_list_is_exact(expect_error, monkeypatch) -> None:
    # Unset or empty: the two forms measured bitwise the previous slab body (2026-09-25) run.
    default = prefill_glue.PrefillGluePolicy.from_environ({})
    assert default.names == prefill_glue.DEFAULT_ON == {"gr_gather_generic", "qsa_mask_hoist"}
    assert prefill_glue.PrefillGluePolicy.from_environ({ENV: " "}).names == prefill_glue.DEFAULT_ON
    assert all(prefill_glue.FORMS[name][0] == prefill_glue.BITWISE for name in prefill_glue.DEFAULT_ON)
    assert {cls for cls, _ in prefill_glue.FORMS.values()} == {prefill_glue.BITWISE, prefill_glue.TOLERANCE}
    # ``today``: the previous slab body, no form.
    today = _policy(prefill_glue.TODAY)
    assert today.names == frozenset() and all(not today.enabled(name) for name in prefill_glue.FORMS)
    # A list runs exactly the named forms: the measurement's arms keep their spellings (the gather alone is arm 1).
    on = _policy("qsa_mask_hoist", " gr_gather_tuned ", "gdn_qk_flat", "")
    assert on.names == {"qsa_mask_hoist", "gr_gather_tuned", "gdn_qk_flat"}
    assert on.enabled("gr_gather_tuned") and not on.enabled("gr_gather_generic")
    assert _policy("gr_gather_generic").names == {"gr_gather_generic"}
    assert {name for name in on.names if prefill_glue.FORMS[name][0] == prefill_glue.TOLERANCE} == {"gdn_qk_flat"}
    with expect_error(KeyError, match="no prefill glue form"):  # allow-pytest.raises: pure contract test
        on.enabled("ple_slab_pass")
    with expect_error(ValueError, match="unregistered"):  # allow-pytest.raises: pure contract test
        _policy("qsa_mask_hoist", "ple_slab_pass")
    with expect_error(ValueError, match="exclusive"):  # allow-pytest.raises: pure contract test
        _policy("gr_gather_generic", "gr_gather_tuned")
    with expect_error(ValueError, match="stands alone"):  # allow-pytest.raises: pure contract test
        _policy(prefill_glue.TODAY, "qsa_mask_hoist")
    # The default set is checked at import: registered, bitwise, no exclusive pair.
    prefill_glue._check_default_on()
    for bad, match in (
        ({"gr_gather_generic", "qsa_scores_rs_ag"}, "must be bitwise"),
        ({"gr_gather_generic", "ple_slab_pass"}, "unregistered"),
        ({"gr_gather_generic", "gr_gather_tuned"}, "exclusive"),
    ):
        monkeypatch.setattr(prefill_glue, "DEFAULT_ON", frozenset(bad))
        with expect_error(ValueError, match=match):  # allow-pytest.raises: pure contract test
            prefill_glue._check_default_on()


def test_process_policy_is_resolved_once(monkeypatch) -> None:
    prefill_glue.policy.cache_clear()
    try:
        monkeypatch.setenv(ENV, "qsa_mask_hoist")
        first = prefill_glue.policy()
        monkeypatch.setenv(ENV, "gdn_qk_flat")
        assert prefill_glue.policy() is first and first.names == {"qsa_mask_hoist"}
    finally:
        prefill_glue.policy.cache_clear()


# --------------------------------------------------------------------------- every read sits under a slab condition


def _slab_guarded(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """The call's nearest guard names the slab: an ``if``/conditional expression whose test mentions ``slab``, an
    ``and`` whose first operand does, or a function whose name does."""

    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.If, ast.IfExp)) and "slab" in ast.unparse(node.test):
            return True
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And) and "slab" in ast.unparse(node.values[0]):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return "slab" in node.name
    return False


def test_every_form_is_read_by_its_slab_branch_only() -> None:
    read_by: dict[str, set[str]] = {name: set() for name in prefill_glue.FORMS}
    for module_name, module in MODULES.items():
        tree = ast.parse(inspect.getsource(module))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "enabled"
            ):
                continue
            if ast.unparse(node.func.value) not in ("self.glue", "glue"):
                continue  # the fused kernel registry's switch, not a glue form
            assert len(node.args) == 1 and isinstance(node.args[0], ast.Constant), ast.unparse(node)
            name = node.args[0].value
            assert name in prefill_glue.FORMS, f"{module_name} reads the unregistered form {name!r}"
            assert _slab_guarded(node, parents), f"{module_name}: {ast.unparse(node)} is not under a slab condition"
            read_by[name].add(module_name)
    for name, (_, branch) in prefill_glue.FORMS.items():
        module_name, function = branch.split(".")
        assert read_by[name] == {module_name}, f"{name} is read by {sorted(read_by[name])}, registered for {branch}"
        owner = MODULES[module_name]
        for cls in (owner,) + tuple(v for v in vars(owner).values() if inspect.isclass(v)):
            if hasattr(cls, function):
                assert f'enabled("{name}")' in inspect.getsource(getattr(cls, function))
                break
        else:
            raise AssertionError(f"{branch} is not a function of {module_name}")


# --------------------------------------------------------------------------- the default forms keep their chains


def test_default_forms_keep_the_chains_and_the_switch_texts() -> None:
    read = inspect.getsource(gr_module.Qwen38TTNNGatedResidual.read_rows)
    for name in ("gr_partial_rs_ag", "gr_gather_generic", "gr_gather_tuned"):
        assert f'slab and self.glue.enabled("{name}")' in read
    # ``today`` is the async op at one worker and one chunk per sync (the tuned form changes those two settings);
    # the generic op (the default form) is the branch before it, so the previous chain stays in place.
    assert "chunks_per_sync=10 if tuned else 1" in read and "num_workers_per_link=2 if tuned else 1" in read
    assert read.index("ttnn.reduce_scatter(") < read.index("ttnn.experimental.all_gather_async(")
    assert read.index("ttnn.experimental.all_gather_async(") < read.index("ttnn.experimental.fast_reduce_nc(")
    select = inspect.getsource(qsa_module.Qwen38TTNNQSA._sparse_indices_slab)
    assert 'self.glue.enabled("qsa_scores_rs_ag")' in select and "ttnn.reduce_scatter(" in select
    assert "ttnn.all_gather(local_ids, dim=2" in select and "if scores_rs_ag:" in select
    make = inspect.getsource(gdn_module.Qwen38TTNNGDN._make_chunk_inputs)
    assert "if rows_state.flat_qk:" in make and "constants.qk_expand" in make
    assert "ttnn.rms_norm(heads_tensor, epsilon=QK_L2_NORM_EPS / HEAD_DIM)" in make
    chunk = inspect.getsource(gdn_module.Qwen38TTNNGDN._chunk_rows)
    assert "if rows_state.flat_qk else rows_state.q" in chunk and "if rows_state.flat_qk else rows_state.k" in chunk


def test_scores_rs_mask_value_is_exact_in_bf16_where_the_indexer_mask_overflows() -> None:
    value = torch.tensor([qsa_module.SLAB_SCORES_RS_MASK_VALUE], dtype=torch.bfloat16)
    assert float(value) == -(2.0**100)
    summed = value.clone()
    for _ in range(3):
        summed = summed + value
    assert float(summed) == -(2.0**102)
    assert float(torch.tensor([1234.5], dtype=torch.bfloat16) + value) == -(2.0**100)
    indexer = torch.tensor([qsa_module.INDEXER_MASK_VALUE], dtype=torch.bfloat16)
    assert float(indexer) == qsa_module.INDEXER_MASK_VALUE and math.isinf(float(indexer + indexer))
    assert qsa_module.slab_scores_mask_value(_policy()) == qsa_module.INDEXER_MASK_VALUE
    assert qsa_module.slab_scores_mask_value(_policy("qsa_scores_rs_ag")) == qsa_module.SLAB_SCORES_RS_MASK_VALUE


# --------------------------------------------------------------------------- the hoisted QSA block masks


def _slab_fake() -> SimpleNamespace:
    fake = qsa_test._integer_fake()

    def slice_(tensor, start, end, *, memory_config=None):
        window = tuple(slice(a, b) for a, b in zip(start, end))
        return qsa_test._IntTensor(tensor._read()[window].clone(), tensor.dtype, tensor.layout)

    def to_layout(tensor, layout, *, memory_config=None):
        return qsa_test._IntTensor(tensor._read().clone(), tensor.dtype, layout)

    fake.slice = slice_
    fake.to_layout = to_layout
    return fake


@pytest.fixture
def slab_fake(monkeypatch):
    fake = _slab_fake()
    monkeypatch.setattr(qsa_module, "ttnn", fake)
    return fake


def _slab_inputs(
    position: int,
    blocks: int,
    glue: prefill_glue.PrefillGluePolicy,
    admission: "qsa_module.Qwen38HoistedMaskAdmission | None | str" = "policy",
):
    """The slab's chunk inputs under ``glue``; ``admission`` is the chunk constants' hoist decision (by default the
    cap-only admission when the policy names the hoist, as the constants take it without a device view)."""

    host = qsa_module.qsa_chunk_constant_rows(blocks, SLAB)
    constants = SimpleNamespace(
        allocated_compressed_blocks=blocks,
        high27_mask=qsa_test._u32(torch.full((1, 1, 1, 1), qsa_module.KV_BLOCK_START_MASK)),
    )
    if admission == "policy":
        admission = qsa_module.admit_hoisted_masks(SLAB, blocks) if glue.enabled("qsa_mask_hoist") else None
    chunk = SimpleNamespace(
        allocated_compressed_blocks=blocks,
        rows=SLAB,
        block_tiles=host["block_start_lanes"].shape[2],
        hoist_masks=admission,
        **{name: qsa_test._u32(host[name]) for name in SLAB_UINT32_TEMPLATES},
    )
    return qsa_module.derive_qsa_chunk_inputs(
        qsa_test._u32(torch.full((1, 1, 1, 1), position)), constants, chunk, glue=glue
    )


@pytest.mark.parametrize("position", [0, 2048])
def test_hoisted_block_masks_are_the_per_position_masks(expect_error, slab_fake, position: int) -> None:
    blocks = 1024  # a 4k context keeps the emulation small; the mask rule does not depend on the block count
    plain = _slab_inputs(position, blocks, _policy(prefill_glue.TODAY))
    assert plain.block_masks == ()
    hoisted = _slab_inputs(position, blocks, _policy("qsa_mask_hoist"))
    assert len(hoisted.block_masks) == SLAB // qsa_module.SLAB_SCORE_BLOCK_ROWS
    assert all(
        mask.layout is slab_fake.ROW_MAJOR_LAYOUT and mask.dtype is slab_fake.bfloat16 for mask in hoisted.block_masks
    )
    rs_ag = _slab_inputs(position, blocks, _policy("qsa_mask_hoist", "qsa_scores_rs_ag"))
    for index, mask in enumerate(hoisted.block_masks):
        assert mask.shape == (1, 1, qsa_module.SLAB_SCORE_BLOCK_ROWS, blocks)
        for row in (0, 1, 3, 4, 255, 511):
            slab_row = index * qsa_module.SLAB_SCORE_BLOCK_ROWS + row
            one = qsa_module.emulate_qsa_position_inputs(position + slab_row, allocated_compressed_blocks=blocks)
            expected = one["indexer_neg_mask"].reshape(-1)
            assert torch.equal(mask._read()[0, 0, row], expected)
            hidden = (expected != 0).to(torch.bfloat16) * qsa_module.SLAB_SCORES_RS_MASK_VALUE
            assert torch.equal(rs_ag.block_masks[index]._read()[0, 0, row], hidden)
    # The layer admits the hoisted masks and refuses a wrong count.
    layer = qsa_module.Qwen38TTNNQSA.__new__(qsa_module.Qwen38TTNNQSA)
    layer.allocated_compressed_blocks = blocks
    layer._validate_chunk_inputs(hoisted)
    layer._validate_chunk_inputs(plain)
    with expect_error(ValueError, match="one per"):  # allow-pytest.raises: pure contract test
        layer._validate_chunk_inputs(replace(hoisted, block_masks=hoisted.block_masks[:1]))
    hoisted.deallocate()
    assert all(not mask.alive for mask in hoisted.block_masks)


def test_hoist_admission_takes_the_cap_and_the_free_dram_and_names_the_fallback(expect_error) -> None:
    working_set = qsa_module.slab_working_set_bytes(SLAB)
    assert working_set == 1_036_507_648  # the dense admission's term at 2048 rows (988.5 MiB)
    # The cap: the masks are 1 KiB per context token at 2048 rows, so 32k (32 MiB) and 64k (64 MiB, the cap itself)
    # hoist; 128k (128 MiB) and 256k fall back to the per-layer derivation.
    for context, admitted in ((32_768, True), (65_536, True), (131_072, False), (262_144, False)):
        blocks = context // qsa_module.COMPRESS_RATIO
        admission = qsa_module.admit_hoisted_masks(SLAB, blocks)
        assert admission.mask_bytes == qsa_module.hoisted_mask_bytes(SLAB, blocks) == context * 1024
        assert (admission.rows, admission.blocks, admission.limit_bytes) == (SLAB, blocks, 64 << 20)
        assert admission.hoist is admitted, (context, admission.reason)
        assert admission.free_bytes_per_device is None and admission.slab_working_set_bytes is None
        if not admitted:
            assert "HOISTED_MASK_BYTES_MAX" in admission.reason and "derived per layer" in admission.reason
    # The free DRAM after the build must hold the masks plus the slab's working set, byte for byte.
    blocks_32k = 32_768 // qsa_module.COMPRESS_RATIO
    needed = (32 << 20) + working_set
    admitted = qsa_module.admit_hoisted_masks(
        SLAB, blocks_32k, free_bytes_per_device=needed, slab_working_set=working_set
    )
    assert admitted.hoist and (admitted.free_bytes_per_device, admitted.slab_working_set_bytes) == (needed, working_set)
    assert "working set" in admitted.reason
    short = qsa_module.admit_hoisted_masks(
        SLAB, blocks_32k, free_bytes_per_device=needed - 1, slab_working_set=working_set
    )
    assert not short.hoist and "working set" in short.reason and "derived per layer" in short.reason
    assert "1020.5 MiB" in short.reason  # the need, so the log names the number that decided
    with expect_error(ValueError, match="working-set term"):  # allow-pytest.raises: pure contract test
        qsa_module.admit_hoisted_masks(SLAB, blocks_32k, free_bytes_per_device=needed)


class _PropagateToLogging(logging.Handler):
    """A loguru sink that hands each record to the standard logger of its name, so ``caplog`` sees loguru's lines."""

    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


@pytest.fixture
def loguru_caplog(caplog):
    sink = loguru_logger.add(_PropagateToLogging(), format="{message}", level="INFO")
    try:
        with caplog.at_level(logging.INFO):
            yield caplog
    finally:
        loguru_logger.remove(sink)


def _mask_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "prefill slab QSA block masks" in r.getMessage()]


def test_slab_hoist_admission_reads_the_allocator_for_a_slab_under_the_form_only(
    expect_error, slab_fake, loguru_caplog
) -> None:
    free_per_bank = [200_000_000]  # 1.6 GB per device over 8 banks: holds 32 MiB of masks + the 988.5 MiB working set
    reads: list[str] = []
    slab_fake.BufferType = SimpleNamespace(DRAM="dram")

    def get_memory_view(mesh_device, buffer_type):
        assert mesh_device == "mesh" and buffer_type == "dram"
        reads.append(buffer_type)
        return SimpleNamespace(total_bytes_free_per_bank=free_per_bank[0], num_banks=8)

    slab_fake.get_memory_view = get_memory_view
    blocks_32k = 32_768 // qsa_module.COMPRESS_RATIO
    # Not a slab, or the form not named (``today``): no decision and no allocator read.
    assert qsa_module.slab_hoisted_mask_admission("mesh", 128, blocks_32k, _policy("qsa_mask_hoist")) is None
    assert _mask_lines(loguru_caplog) == []  # a chunk form: no decision, no line
    assert qsa_module.slab_hoisted_mask_admission("mesh", SLAB, blocks_32k, _policy(prefill_glue.TODAY)) is None
    assert reads == []
    # The build logs the decision once, in the admission's own words (the served log carries it): ``today`` names
    # the per-layer form and the form's absence with the masks' size and the cap ...
    assert _mask_lines(loguru_caplog) == [
        "prefill slab QSA block masks derived per layer: qsa_mask_hoist is not named by QWEN38_PREFILL_GLUE "
        "(32.0 MiB of masks per slab (2048 rows, 8192 blocks), cap 64.0 MiB)"
    ]
    admitted = qsa_module.slab_hoisted_mask_admission("mesh", SLAB, blocks_32k, _policy("qsa_mask_hoist"))
    assert admitted.hoist and admitted.free_bytes_per_device == 1_600_000_000 and reads == ["dram"]
    assert admitted.slab_working_set_bytes == qsa_module.slab_working_set_bytes(SLAB)
    # ... the admitted hoist names the masks, the cap, the free MiB per device and the working-set term ...
    assert (
        _mask_lines(loguru_caplog)[-1]
        == "prefill slab QSA block masks hoisted: " + admitted.reason
        == (
            "prefill slab QSA block masks hoisted: 32.0 MiB of hoisted QSA block masks per slab (2048 rows, 8192 blocks) "
            "within the 64.0 MiB cap; 1525.9 MiB free per device hold them plus the slab's 988.5 MiB working set"
        )
    )
    free_per_bank[0] = 100_000_000  # 0.8 GB per device: the per-layer form, under the default policy too
    fallback = qsa_module.slab_hoisted_mask_admission(
        "mesh", SLAB, blocks_32k, prefill_glue.PrefillGluePolicy.from_environ({})
    )
    assert not fallback.hoist and fallback.free_bytes_per_device == 800_000_000 and len(reads) == 2
    # ... and the refusal names the need against the free DRAM that decided it.
    assert (
        _mask_lines(loguru_caplog)[-1]
        == "prefill slab QSA block masks derived per layer: " + fallback.reason
        == (
            "prefill slab QSA block masks derived per layer: 32.0 MiB of hoisted QSA block masks per slab (2048 rows, "
            "8192 blocks) plus the slab's 988.5 MiB working set = 1020.5 MiB exceed the 762.9 MiB of DRAM free per device "
            "after the build: the masks are derived per layer"
        )
    )
    assert len(_mask_lines(loguru_caplog)) == 3
    # The derivation follows the constants' decision: masks under an admitted one, none under the fallback, and a
    # decision taken for another slab shape is refused.
    blocks = 1024
    hoisted = _slab_inputs(0, blocks, _policy("qsa_mask_hoist"), admission=qsa_module.admit_hoisted_masks(SLAB, blocks))
    assert len(hoisted.block_masks) == SLAB // qsa_module.SLAB_SCORE_BLOCK_ROWS
    hoisted.deallocate()
    assert _slab_inputs(0, blocks, _policy("qsa_mask_hoist"), admission=fallback).block_masks == ()
    assert _slab_inputs(0, blocks, _policy("qsa_mask_hoist"), admission=None).block_masks == ()
    with expect_error(ValueError, match="hoist admission was taken for"):  # allow-pytest.raises: contract
        _slab_inputs(0, blocks, _policy("qsa_mask_hoist"), admission=qsa_module.admit_hoisted_masks(SLAB, blocks_32k))
    # The constants' build takes the decision for a slab and carries it.
    build = inspect.getsource(qsa_module.Qwen38TTNNQSAChunkConstants.build)
    assert "hoist_masks = slab_hoisted_mask_admission(mesh_device, rows, blocks, glue) if slab else None" in build
    derive = inspect.getsource(qsa_module.derive_qsa_chunk_inputs)
    assert "admission = chunk.hoist_masks" in derive and "if admission is not None and admission.hoist:" in derive


# --------------------------------------------------------------------------- the flat q/k rows state


def test_flat_qk_rows_state_is_a_slab_option(expect_error) -> None:
    heads = (1, 128, gdn_module.VALUE_HEADS_PER_DEVICE, gdn_module.HEAD_DIM)
    assert gdn_module.rows_qk_layout(128, False) == (heads, 2)
    assert gdn_module.rows_qk_layout(SLAB, True) == ((1, 1, SLAB, gdn_module.QK_WIDTH_PER_DEVICE), 3)
    contract = SimpleNamespace(validate_mesh=lambda mesh: None)
    chunk_constants = SimpleNamespace(mesh_contract=contract, rows=128, tile_rows=128)
    with expect_error(ValueError, match="slab option"):  # allow-pytest.raises: pure contract test
        gdn_module.Qwen38TTNNGDNRowsState.allocate(object(), contract, chunk_constants, layer_index=0, flat_qk=True)
    slab_constants = SimpleNamespace(mesh_contract=contract, rows=SLAB, tile_rows=SLAB)
    body = SimpleNamespace(constants=slab_constants, flat_qk=False)
    with expect_error(ValueError, match="disagree"):  # allow-pytest.raises: pure contract test
        gdn_module.Qwen38TTNNGDNRowsState.allocate(
            object(), contract, slab_constants, layer_index=0, body=body, flat_qk=True
        )
    allocate = inspect.getsource(gdn_module.Qwen38TTNNGDN.allocate_rows_state)
    assert 'is_slab_rows(constants.rows) and self.glue.enabled("gdn_qk_flat")' in allocate
