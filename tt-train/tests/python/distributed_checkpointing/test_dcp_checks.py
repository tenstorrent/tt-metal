# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The guards save and load run before touching a device or a file: JSON state, host shard sizes, optimizer param
groups, the fused-name lint and the module walk. Device-free."""

import warnings

import numpy as np

from ttml.distributed_checkpointing._errors import CheckpointError
from ttml.distributed_checkpointing._index import check_json
from ttml.distributed_checkpointing._save import _logical_bytes
from ttml.distributed_checkpointing._state_dict import (
    _cpp_bound_params,
    _lint_fusion,
    _python_parameters,
    check_param_groups,
    fsdp_gate,
)
from ttml.distributed_checkpointing._topology import Label, Placement, dist_coords


def _label(placements, dist_shape) -> Label:
    return Label(tuple(placements), tuple(dist_shape), tuple(dist_coords(dist_shape)))


def test_check_json_returns_plain_json_and_names_what_would_not_read_back(expect_error):
    assert check_json("header", {"step": 3, "lr": 1e-3, "shape": (1, 2), "tags": ["a"], "none": None}) == {
        "step": 3,
        "lr": 1e-3,
        "shape": [1, 2],
        "tags": ["a"],
        "none": None,
    }
    with expect_error(CheckpointError, "dataloader at /per_shard: key 0 is a int"):
        check_json("dataloader", {"per_shard": {0: 7}})
    with expect_error(CheckpointError, r"scheduler at /eta_min: inf is not finite"):
        check_json("scheduler", {"eta_min": float("inf")})
    with expect_error(CheckpointError, r"header at /steps\[1\]: a int64 has no JSON form"):
        check_json("header", {"steps": [1, np.int64(2)]})


def test_host_shards_must_be_exactly_their_logical_size(expect_error):
    view = np.arange(16 * 64, dtype=np.uint16).view(np.uint8)
    assert _logical_bytes(view, (1, 1, 16, 64), 2) is view
    padded = np.zeros(32 * 64 * 2, dtype=np.uint8)  # height-padded: the old fallback sliced it at the wrong stride
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
    check_param_groups(_StubOptimizer([["a", "b"], ["c"]]), saved)
    check_param_groups(_StubOptimizer([["a"]]), {})  # a model-only checkpoint has nothing to check
    # The checkpoint's members are the union over the saving ranks, so a live group may hold fewer of them...
    check_param_groups(_StubOptimizer([["a"], ["c"]]), saved)
    # ...but not a param the checkpoint's group does not have: swapped groups would get each other's options.
    with expect_error(CheckpointError, "group 0 holds params the checkpoint's group 0 does not"):
        check_param_groups(_StubOptimizer([["c"], ["a", "b"]]), saved)
    with expect_error(CheckpointError, "2 optimizer param groups, the live optimizer 1"):
        check_param_groups(_StubOptimizer([["a", "b", "c"]]), saved)


class _StubOptimizerWithoutMembers(_StubOptimizer):
    def get_state_dict(self):
        return {"num_param_groups": self.state["num_param_groups"]}


def test_param_groups_whose_members_cannot_be_compared_warn():
    saved = {"optimizer/num_param_groups": 1, "optimizer/param_groups.0.params": "a\nb"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_param_groups(_StubOptimizerWithoutMembers([["a", "b"]]), saved)
        check_param_groups(_StubOptimizer([["a", "b"]]), {"optimizer/num_param_groups": 1})
        check_param_groups(_StubOptimizer([["a", "b"]]), saved)  # comparable and equal: silent
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


def test_parameters_of_cpp_bound_modules_are_named_and_the_walks_pass_over_them():
    params = {"llama/llama_block_0/attention/kv_linear/weight": 0, "llama/fc/weight": 1}
    assert _cpp_bound_params(_CppModel(), params) == list(params), "a C++-bound model: every parameter"
    assert _python_parameters(_CppModel(), params) == ({}, {})
    assert fsdp_gate(_CppModel(), True) == []
    mixed = {"Root/cpp_child/gamma": 0, "Root/cpp_childish/w": 1, "Root/w": 2}
    assert _cpp_bound_params(_Model(), mixed) == ["Root/cpp_child/gamma"], "only the C++-bound child's parameters"
