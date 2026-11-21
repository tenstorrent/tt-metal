# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The guards save and load run before touching a device or a file: JSON state, host shard sizes, optimizer param
groups, the fused-name lint and the module walk. Device-free."""

import os
import warnings

import numpy as np

import ttml.distributed_checkpointing as dcp

from ttml.distributed_checkpointing._errors import CheckpointError, LayoutError, LazyParameterError, MissingKeyError
from ttml.distributed_checkpointing._index import Entry
from ttml.distributed_checkpointing._layouts import Identity, fused_segments
from ttml.distributed_checkpointing._planner import TensorInfo
from ttml.distributed_checkpointing._save import _logical_bytes
from ttml.distributed_checkpointing._state_dict import (
    LiveTensor,
    _cpp_bound_params,
    _lint_fusion,
    _python_parameters,
    check_member_union,
    check_param_groups,
    check_undeclared_fusion,
    reshard_fsdp_modules,
)
from ttml.distributed_checkpointing._topology import Label, Placement, dist_coords, tp_split
from ttml.modules.parameter import Parameter, TensorMetadata


def _label(placements, dist_shape) -> Label:
    return Label(tuple(placements), tuple(dist_shape), tuple(dist_coords(dist_shape)))


def test_host_shards_must_be_exactly_their_logical_size(expect_error):
    view = np.arange(16 * 64, dtype=np.uint16).view(np.uint8)
    assert _logical_bytes(view, (1, 1, 16, 64), 2) is view
    padded = np.zeros(32 * 64 * 2, dtype=np.uint8)  # height-padded to 32 rows: refused, not sliced at a guessed stride
    with expect_error(CheckpointError, "is not the 2048 bytes"):
        _logical_bytes(padded, (1, 1, 16, 64), 2)


class _StubOptimizer:
    def __init__(self, groups):
        self.state = {"num_param_groups": len(groups)}
        for i, names in enumerate(groups):
            self.state[f"param_groups.{i}.params"] = "\n".join(sorted(names))

    def get_state_dict(self):
        return self.state


def test_param_groups_are_checked_before_anything_is_placed(expect_error):
    saved = {
        "optimizer/num_param_groups": 2,
        "optimizer/param_groups.0.params": "a\nb",
        "optimizer/param_groups.1.params": "c",
    }
    check_param_groups(_StubOptimizer([["a", "b"], ["c"]]), saved, strict=True)
    # The checkpoint's members are the union over the saving ranks, so a live group may hold fewer of them...
    check_param_groups(_StubOptimizer([["a"], ["c"]]), saved, strict=True)
    # ...but not a param the checkpoint's group does not have: swapped groups would get each other's options.
    with expect_error(CheckpointError, "group 0 holds params the checkpoint's group 0 does not"):
        check_param_groups(_StubOptimizer([["c"], ["a", "b"]]), saved, strict=True)
    with expect_error(CheckpointError, "2 optimizer param groups, the live optimizer 1"):
        check_param_groups(_StubOptimizer([["a", "b", "c"]]), saved, strict=True)


def test_strict_needs_the_ranks_to_hold_every_member_between_them(expect_error):
    """Under strict, the ranks' live members of each group, together, must be the checkpoint's: one rank may hold part
    of a group, but a param no rank holds is refused, since an optimizer without per-parameter state (SGD without
    momentum) would otherwise stop training it without a word."""
    saved = {
        "optimizer/num_param_groups": 2,
        "optimizer/param_groups.0.params": "a\nb",
        "optimizer/param_groups.1.params": "c",
    }
    check_member_union(saved, [{"0": ["a", "b"], "1": ["c"]}])  # one process holding everything
    check_member_union(saved, [{"0": ["a"], "1": ["c"]}, {"0": ["b"], "1": []}])  # two ranks, each holding part
    with expect_error(CheckpointError, r"^optimizer param group 0: the checkpoint's group holds params no rank's "):
        check_member_union(saved, [{"0": ["a"], "1": ["c"]}])
    with expect_error(CheckpointError, r"group 1: .* no rank's optimizer holds: \['c'\]"):
        check_member_union(saved, [{"0": ["a"], "1": []}, {"0": ["b"], "1": []}])
    check_member_union(saved, [{}])  # no live member lists: check_param_groups warned, nothing to compare


def test_strict_load_needs_optimizer_state(expect_error):
    """A strict load with an optimizer needs optimizer state in the checkpoint."""
    # A model-only checkpoint: an optimizer whose state is scalars only would restore nothing, so strict refuses.
    with expect_error(MissingKeyError, "the checkpoint holds no optimizer state"):
        check_param_groups(_StubOptimizer([["a"]]), {}, strict=True)
    check_param_groups(_StubOptimizer([["a"]]), {}, strict=False)  # keeps the live state, nothing to check


class _StubOptimizerWithoutMembers(_StubOptimizer):
    def get_state_dict(self):
        return {"num_param_groups": self.state["num_param_groups"]}


def test_param_groups_whose_members_cannot_be_compared_warn():
    saved = {"optimizer/num_param_groups": 1, "optimizer/param_groups.0.params": "a\nb"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_param_groups(_StubOptimizerWithoutMembers([["a", "b"]]), saved, strict=True)
        check_param_groups(_StubOptimizer([["a", "b"]]), {"optimizer/num_param_groups": 1}, strict=True)
        check_param_groups(_StubOptimizer([["a", "b"]]), saved, strict=True)  # comparable and equal: silent
    messages = [str(w.message) for w in caught]
    assert len(messages) == 2, messages
    assert "group 0: the live optimizer records no member list" in messages[0]
    assert "group 0: the checkpoint records no member list" in messages[1]


class _Tensor:
    def __init__(self, fsdp_axis=None):
        if fsdp_axis is not None:
            self._fsdp_managed = True
            self._fsdp_axis = fsdp_axis


def test_fused_name_lint_ignores_fsdp_rows_and_warns_for_tp_rows():
    key = "model/Llama/blocks/0/attention/qkv_linear/weight"
    fsdp_rows = _label([Placement.shard(2), Placement.replicate()], (4, 1))  # FSDP axis 0 on dim 2, tp 1
    tp_rows = _label([Placement.replicate(), Placement.shard(2)], (1, 4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _lint_fusion(key, fsdp_rows, {}, _Tensor(fsdp_axis=0))
        assert not caught, "FSDP splits rows that are already canonical"
        _lint_fusion(key, tp_rows, {}, _Tensor())
        assert len(caught) == 1 and "declares no fused_segments" in str(caught[0].message)
        _lint_fusion(key, tp_rows, {"fused": (2, (64, 32, 32))}, _Tensor())
        assert len(caught) == 1, "a declared fused weight is fine"


def test_fused_name_lint_matches_whole_name_parts():
    tp_rows = _label([Placement.replicate(), Placement.shard(2)], (1, 4))
    fused = ["kv_linear", "qkv_proj", "w_qkv", "gate_up_proj", "fused_qkv", "attention_kv"]
    head_major = ["wkv_a", "wkv_b", "q_linear", "in_proj", "kvass"]  # DeepSeek's wkv_b reshards head by head
    for name in fused + head_major:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _lint_fusion(f"model/Root/blocks/0/attention/{name}/weight", tp_rows, {}, _Tensor())
        assert len(caught) == (name in fused), name


class _CppChild:
    """A nanobind-bound module: no __dict__."""

    __slots__ = ()


class _Model:
    def named_modules(self):
        yield "", self
        yield "cpp_child", _CppChild()


def test_module_walk_skips_children_without_python_attributes():
    names, declarations = _python_parameters(_Model(), {})
    assert names == {} and declarations == {}


class _CppModel:
    """A nanobind-bound model: no named_modules and no __dict__."""

    __slots__ = ()


def test_cpp_bound_parameters_are_named():
    """Parameters of C++-bound modules are named, and the module walks pass over them."""
    params = {"llama/llama_block_0/attention/kv_linear/weight": 0, "llama/fc/weight": 1}
    assert _cpp_bound_params(_CppModel(), params) == list(params), "a C++-bound model: every parameter"
    assert _python_parameters(_CppModel(), params) == ({}, {})
    assert reshard_fsdp_modules(_CppModel()) == []
    mixed = {"Root/cpp_child/gamma": 0, "Root/cpp_childish/w": 1, "Root/w": 2}
    assert _cpp_bound_params(_Model(), mixed) == ["Root/cpp_child/gamma"], "only the C++-bound child's parameters"


def _entry(key, label, transform, split):
    """An index entry of a [1, 1, 128, 64] weight saved under ``label``, as save records it."""
    return Entry(
        key, "param", "BF16", "TILE", (1, 1, 128, 64), (1, 1, 128, 64), transform, label, (key,), [], tp_split=split
    )


def test_fused_looking_weight_needs_same_undescribed_split(expect_error):
    """Segments scramble when the per-shard row order one side leaves undeclared differs from the other's: compared
    through the TP split each side leaves undescribed, FSDP's split left out."""
    key = "model/Root/blocks/0/attention/qkv_linear/weight"
    single = _label([Placement.replicate(), Placement.replicate()], (1, 1))
    tp2 = _label([Placement.replicate(), Placement.shard(2)], (1, 2))
    tp4 = _label([Placement.replicate(), Placement.shard(2)], (1, 4))
    fsdp2 = _label([Placement.shard(2), Placement.replicate()], (2, 1))
    declared = {2: fused_segments(2, (64, 32, 32), 2), 4: fused_segments(2, (64, 32, 32), 4)}

    def live(label, transform=Identity(), fsdp_axis=None, fused_name=True):
        info = TensorInfo(
            key,
            "param",
            "BF16",
            "TILE",
            label,
            (1, 1, 128 // label.shard_count(2), 64),
            transform,
            (key,),
            tp_split=tp_split(label, fsdp_axis, 4),
        )
        return LiveTensor(key, None, info, fused_name=fused_name)

    saved_tp2 = _entry(key, tp2, Identity(), (1, 1, 2, 1))  # undeclared under TP2: its rows are in TP2's order
    check_undeclared_fusion(live(tp2), saved_tp2)  # the same TP size, undeclared on both sides: copied as it is
    with expect_error(
        LayoutError, r"leaves a TP split of \[1, 1, 2, 1\] per dim undescribed and the live model \[1, 1, 4, 1\]"
    ):
        check_undeclared_fusion(live(tp4), saved_tp2)
    with expect_error(LayoutError, r"undescribed and the live model \[1, 1, 1, 1\]"):
        check_undeclared_fusion(live(tp2, declared[2]), saved_tp2)  # declaring it now would reorder TP2's rows
    check_undeclared_fusion(live(tp4, fused_name=False), saved_tp2)  # not a fused-looking name: not checked

    saved_declared = _entry(key, tp2, declared[2], (1, 1, 2, 1))
    check_undeclared_fusion(live(single), saved_declared)  # one device: canonical rows, nothing undescribed
    check_undeclared_fusion(live(tp4, declared[4]), saved_declared)
    with expect_error(
        LayoutError, r"leaves a TP split of \[1, 1, 1, 1\] per dim undescribed and the live model \[1, 1, 4, 1\]"
    ):
        check_undeclared_fusion(live(tp4), saved_declared)

    # FSDP splits rows already in canonical order: an undeclared weight saved under FSDP loads anywhere TP leaves it
    # whole, where comparing the plain splits would have refused it.
    saved_fsdp = _entry(key, fsdp2, Identity(), tp_split(fsdp2, 0, 4))
    assert saved_fsdp.tp_split == (1, 1, 1, 1)
    check_undeclared_fusion(live(single), saved_fsdp)
    check_undeclared_fusion(live(tp2, declared[2]), saved_fsdp)


class _Declaring:
    """A Python module whose one parameter declares ``declaration``."""

    def __init__(self, declaration):
        self.w = Parameter(object())
        self.w.checkpoint_layout = declaration

    def named_modules(self):
        yield "", self


def test_an_unknown_layout_declaration_is_refused(expect_error):
    _, declarations = _python_parameters(_Declaring({"fused": (2, (64, 32, 32))}), {"Root/w": 0})
    assert list(declarations.values()) == [{"fused": (2, (64, 32, 32))}]
    # A misspelt key is refused: ignored, it would leave the weight with an identity layout that scrambles a cross-TP
    # load.
    with expect_error(LayoutError, r"Root/w: checkpoint_layout declares \['fused_segments'\]"):
        _python_parameters(_Declaring({"fused_segments": (2, (64, 32, 32))}), {"Root/w": 0})


class _Lazy:
    """A Python module with an unmaterialized parameter, at the root or in the child ``blocks.0``."""

    def __init__(self, nested: bool):
        self.nested = nested
        self.w = Parameter(TensorMetadata((1, 1, 32, 32), lambda shape, mapper: None))

    def named_modules(self):
        yield ("blocks.0", self) if self.nested else ("", self)


def test_a_lazy_parameter_is_named_by_its_key(expect_error):
    with expect_error(LazyParameterError, r"^Root/w has not been materialized; call ttml.materialize_module first"):
        _python_parameters(_Lazy(nested=False), {"Root/x": 0})
    with expect_error(LazyParameterError, r"^Root/blocks/0/w has not been materialized"):
        _python_parameters(_Lazy(nested=True), {"Root/x": 0})


def test_argument_errors_come_first(tmp_path, expect_error):
    """Argument errors are raised before anything is touched, as CheckpointErrors naming the step that failed, with
    the TypeError or ValueError as their cause, as they are on every rank of a multi-rank world."""
    path = str(tmp_path / "step_0000001")
    save, load = "collecting the live tensors", "checking the arguments"
    cases = [
        (lambda: dcp.save(path), save, TypeError, "save needs a model and/or an optimizer"),
        (lambda: dcp.load(path, optimizer=object()), load, TypeError, r"optimizer state .*; pass model= as well"),
        (lambda: dcp.save(path, model=object(), header=[1]), save, TypeError, "header must be a dict, not list"),
        (lambda: dcp.save(path, model=object(), threads=0), save, ValueError, "threads must be at least 1, got 0"),
        (lambda: dcp.save(path, model=object(), max_inflight_bytes=0), save, ValueError, "max_inflight_bytes must"),
        (lambda: dcp.load(path, model=object(), threads=-1), load, ValueError, "threads must be at least 1, got -1"),
    ]
    for call, step, cause, message in cases:
        with expect_error(dcp.CheckpointError, rf"^{step} failed: {cause.__name__}: {message}") as raised:
            call()
        assert type(raised.value) is dcp.CheckpointError and type(raised.value.__cause__) is cause, message
    assert os.listdir(tmp_path) == []
