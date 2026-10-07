# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttml.checkpointing`` refuses to write or restore a tensor whose gathered shape contradicts its parameter's.

``save_checkpoint`` gathers each tensor by its own topology label. A moment labelled ``Replicate`` on a mesh axis its
parameter is sharded on gathers to one device's shard, and before this guard the file was written with that
truncated record silently. Every optimizer state tensor is created as ``zeros_like`` / ``typecast`` of its parameter,
so a shape mismatch between the two is always a wrong label. The guard is a sink check: it does not know or care
which op mislabelled the tensor, it only refuses to serialise the consequence.

Each test states its negative control in its docstring: what the same call did before the guard existed.

Runs on the module-scoped ``tp_mesh`` ([1, 2], conftest). Helpers are copied from
``test_optimizer_state_topology.py`` (``TPBlock``, ``_train_step``) rather than imported: the repo runs pytest with
``--import-mode=importlib``, under which sibling test modules are not importable.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

import ttml
import ttnn
from ttml import checkpointing
from ttml.checkpointing import CheckpointShapeError
from ttml.sharding import ReplicaMismatchError
from ttml.modules import AbstractModuleBase, ColumnParallelLinear, LinearLayer, RowParallelLinear

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

# DIM is chosen so that a record halved along its shard dim, then split 2-way by the live mapper, still lands on
# tile-aligned (32-multiple) per-device shapes: the per-device check is what fires rather than a tilize error.
DIM = 128
NATIVE = ttml.autograd.PreferredPrecision.NATIVE


# --- helpers (mirroring test_optimizer_state_topology.py) ------------------------------------------------------


class TPBlock(AbstractModuleBase):
    """Replicated linear -> column-parallel -> row-parallel: sharded weights on dims 2 and 3, a sharded bias
    (column-parallel, dim 3) and replicated params (the first linear, the row-parallel bias)."""

    def __init__(self) -> None:
        super().__init__()
        self.inp = LinearLayer(DIM, DIM)
        self.up = ColumnParallelLinear(DIM, DIM, has_bias=True)
        self.down = RowParallelLinear(DIM, DIM, has_bias=True, input_is_parallel=True)

    def forward(self, x):
        return self.down(self.up(self.inp(x)))


def _make_adamw(params):
    cfg = ttml.optimizers.AdamWConfig.make(lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)
    return ttml.optimizers.AdamW(params, cfg)


def _train_step(model, opt) -> None:
    """One forward/backward/optimizer step on a ``(1, 1, DIM, DIM)`` input, so every moment exists and is non-zero."""
    ctx = ttml.autograd.AutoContext.get_instance()
    x = ttml.autograd.Tensor.from_numpy(
        np.random.default_rng(0).standard_normal((1, 1, DIM, DIM), dtype=np.float32),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
    )
    opt.zero_grad()
    loss = ttml.ops.unary.mean(model(x))
    loss.backward(False)
    opt.step()
    ctx.reset_graph()


def _fresh_tp():
    """An untrained TPBlock and its AdamW: (model, params, opt)."""
    model = TPBlock()
    params = model.parameters()
    return model, params, _make_adamw(params)


def _trained_tp():
    model, params, opt = _fresh_tp()
    _train_step(model, opt)
    return params, opt


def _gathered_shape(tensor) -> tuple:
    """The shape the tensor gathers to by its label, trusting the label (forged labels are gathered here too)."""
    return tuple(ttml.Sharding.from_tensor(tensor).gather(tensor, verify_replicas=False).shape)


def _sharded_param_name(params) -> str:
    """Name of one parameter that is sharded on some mesh axis (the guard is vacuous for a replicated one)."""
    for name, t in params.items():
        placements = ttml.Sharding.from_tensor(t).placements
        if placements is not None and any(isinstance(p, ttnn.PlacementShard) for p in placements):
            return name
    raise AssertionError("TPBlock has no sharded parameter")


def _shard_dim(tensor) -> int:
    """The tensor dim its (single) Shard placement splits."""
    dims = [p.dim for p in ttml.Sharding.from_tensor(tensor).placements if isinstance(p, ttnn.PlacementShard)]
    assert len(dims) == 1, dims
    return dims[0]


def _forge_replicate(tensor) -> "ttnn.TensorTopology":
    """Relabel ``tensor`` as fully replicated without touching its data, returning the original label.

    Python ``get_value`` shares the C++ tensor's attributes, so the relabel sticks. The original topology is
    captured before the relabel (``tensor_topology()`` returns a copy) so the caller can restore it."""
    value = tensor.get_value(NATIVE)
    original = value.tensor_topology()
    forged = ttnn.TensorTopology(
        ttnn.MeshShape(list(original.distribution_shape())),
        [ttnn.PlacementReplicate()] * len(original.placements()),
        list(original.mesh_coords()),
    )
    value.update_tensor_topology(forged)
    assert ttml.Sharding.from_tensor(tensor).is_fully_replicated, "precondition: the forged label did not stick"
    return original


def _assert_nothing_written(path: str) -> None:
    assert not os.path.exists(path), "a rejected save must not produce a checkpoint"
    assert not os.path.exists(path + ".tmp"), "a rejected save must remove its temp file"


def _stream_records(path: str) -> list:
    """The header record followed by every tensor record of a checkpoint, in file order."""
    records = []
    with open(path, "rb") as f:
        while True:
            try:
                records.append(pickle.load(f))
            except EOFError:
                return records


def _record_keys(manifest: dict) -> list:
    """``(group, path, name)`` for every tensor record, in the stream order ``save_checkpoint`` uses (DFS)."""

    def walk(node, path):
        if not isinstance(node, dict):
            return
        if "named_parameters" in node:
            for name in node["named_parameters"]:
                yield (*path, name)
            return
        for key, sub in node.items():
            yield from walk(sub, path + (key,))

    return [key for group, skeleton in manifest.items() for key in walk(skeleton, (group,))]


def _write_with_truncated_record(src: str, dst: str, record_key: tuple, dim: int) -> tuple:
    """Copy checkpoint ``src`` to ``dst`` with the record at ``record_key`` halved along ``dim``.

    This is what a moment mislabelled ``Replicate`` on its parameter's sharded axis used to be saved as: one
    device's slice. Returns the truncated record's shape."""
    header, *records = _stream_records(src)
    keys = _record_keys(header["manifest"])
    assert len(keys) == len(records), (len(keys), len(records))
    index = keys.index(record_key)
    full = records[index]
    half = full.shape[dim] // 2
    records[index] = np.ascontiguousarray(np.take(full, range(half), axis=dim))
    with open(dst, "wb") as f:
        pickle.dump(header, f)
        for record in records:
            pickle.dump(record, f)
    return records[index].shape


# --- save --------------------------------------------------------------------------------------------------------


def test_save_rejects_state_whose_label_does_not_describe_its_data(tp_mesh, tmp_path, expect_error):
    """A moment relabelled ``Replicate`` on the axis its parameter is sharded on gathers to one shard; the save must
    refuse, name the leaf, both shapes and the layout, and leave no file behind. Restoring the label makes the same
    save succeed (positive control: the guard is not simply rejecting every TP checkpoint).

    The moment's copies differ, so the replica check refuses it first, by name; the shape cross-check is then
    exercised with ``verify_replicas=False``, as the second line.

    Negative control: before the guard, ``save_checkpoint`` wrote the file with the (1, 1, DIM/2, DIM)-or-similar
    record silently, so the ``expect_error`` here would have failed with "did not raise"."""
    params, opt = _trained_tp()
    name = _sharded_param_name(params)
    param_shape = _gathered_shape(params[name])
    moment = opt.get_state_dict()["exp_avg"][name]
    assert _gathered_shape(moment) == param_shape, "precondition: moment and parameter agree before the forgery"

    original = _forge_replicate(moment)
    forged_shape = _gathered_shape(moment)
    assert forged_shape != param_shape, "precondition: the forged label must make the moment gather to a shard"

    path = str(tmp_path / "forged.ckpt")
    with expect_error(ReplicaMismatchError, rf"optimizer/exp_avg\[{name}\] is labelled .* differs"):
        checkpointing.save_checkpoint(path, header={}, model_params=params, optimizer=opt)
    _assert_nothing_written(path)

    with expect_error(CheckpointShapeError, rf"optimizer/exp_avg\[{name}\] gathers to") as excinfo:
        checkpointing.save_checkpoint(path, header={}, model_params=params, optimizer=opt, verify_replicas=False)
    message = str(excinfo.value)
    for fragment in (str(forged_shape), str(param_shape), f"model[{name}]", "Replicate", "Nothing was written"):
        assert fragment in message, f"missing {fragment!r} in:\n{message}"
    _assert_nothing_written(path)

    moment.get_value(NATIVE).update_tensor_topology(original)
    assert _gathered_shape(moment) == param_shape
    checkpointing.save_checkpoint(path, header={}, model_params=params, optimizer=opt)
    assert os.path.exists(path) and not os.path.exists(path + ".tmp")


def test_save_rejects_param_whose_replicate_label_hides_different_data(tp_mesh, tmp_path, expect_error):
    """A parameter's own wrong label has no tensor to be cross-checked against: a sharded parameter relabelled
    ``Replicate`` gathers to one device's shard, and with its optimizer saved alongside, the shape cross-check would
    blame the (correct) moments. The replica check reads both copies, finds they differ and refuses the parameter
    itself, by name, before anything is written. Restoring the label makes the same save succeed.

    Negative control: with ``verify_replicas=False`` (the gather before this check) the file is written and its
    record for the parameter holds one device's shard."""
    params, opt = _trained_tp()
    name = _sharded_param_name(params)
    full = _gathered_shape(params[name])
    original = _forge_replicate(params[name])

    path = str(tmp_path / "forged_param.ckpt")
    with expect_error(ReplicaMismatchError, rf"model\[{name}\] is labelled .* differs from copy 0") as excinfo:
        checkpointing.save_checkpoint(path, header={}, model_params=params, optimizer=opt)
    for fragment in ("Replicate", "Nothing was written"):
        assert fragment in str(excinfo.value), str(excinfo.value)
    _assert_nothing_written(path)

    unchecked = str(tmp_path / "unchecked.ckpt")
    checkpointing.save_checkpoint(unchecked, header={}, model_params=params, verify_replicas=False)
    header, *records = _stream_records(unchecked)
    record = records[_record_keys(header["manifest"]).index(("model", name))]
    assert tuple(record.shape) != full, "negative control: the label-trusting gather saves one shard"

    params[name].get_value(NATIVE).update_tensor_topology(original)
    checkpointing.save_checkpoint(path, header={}, model_params=params, optimizer=opt)
    assert os.path.exists(path) and not os.path.exists(path + ".tmp")


def test_save_rejects_collapsed_replicate_label_over_different_data(tp_mesh, tmp_path, expect_error):
    """The default mappers' collapsed 1-D label (``[2] / (Replicate,)``) goes through the same check: a tensor whose
    two devices hold different halves under that label is refused; a truly replicated tensor under the same label
    (the default ``replicate_tensor_to_mesh_mapper``) saves.

    Negative control: before this check the first save wrote the first device's half silently."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    wide = np.random.default_rng(3).standard_normal((1, 1, 32, 64), dtype=np.float32)
    halves = ttml.autograd.Tensor.from_numpy(
        wide, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 3)
    )
    assert len(ttml.Sharding.from_tensor(halves).dist_shape) == 1, "precondition: a collapsed 1-D label"
    _forge_replicate(halves)
    params = ttml.NamedParameters()
    params["halves"] = halves

    path = str(tmp_path / "collapsed.ckpt")
    with expect_error(ReplicaMismatchError, r"model\[halves\] is labelled \[Replicate\] over mesh \[2\]"):
        checkpointing.save_checkpoint(path, header={}, model_params=params)
    _assert_nothing_written(path)

    same = ttml.autograd.Tensor.from_numpy(
        wide[..., :32],
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        ttml.core.distributed.replicate_tensor_to_mesh_mapper(device),
    )
    replicated = ttml.NamedParameters()
    replicated["same"] = same
    checkpointing.save_checkpoint(path, header={}, model_params=replicated)
    _, record = _stream_records(path)
    assert tuple(record.shape) == (1, 1, 32, 32)


def test_replica_check_can_be_disabled(tp_mesh, tmp_path, expect_error, monkeypatch):
    """The replica check is on by default; ``TTML_CHECKPOINT_VERIFY_REPLICAS=0`` turns it off for every save in the
    process (the label-trusting gather, which keeps one copy), and an explicit ``verify_replicas=`` overrides the
    variable either way. Checked on a parameter whose ``Replicate`` label hides different data, saved alone so the
    shape cross-check has nothing to compare it with."""
    params, _ = _trained_tp()
    name = _sharded_param_name(params)
    _forge_replicate(params[name])
    path = str(tmp_path / "toggle.ckpt")

    monkeypatch.delenv(checkpointing.VERIFY_REPLICAS_ENV, raising=False)
    with expect_error(ReplicaMismatchError, rf"model\[{name}\] is labelled"):
        checkpointing.save_checkpoint(path, header={}, model_params=params)
    _assert_nothing_written(path)

    for off in ("0", "false", "OFF"):
        monkeypatch.setenv(checkpointing.VERIFY_REPLICAS_ENV, off)
        checkpointing.save_checkpoint(path, header={}, model_params=params)
        assert os.path.exists(path), off
        os.remove(path)

    with expect_error(ReplicaMismatchError, rf"model\[{name}\] is labelled"):
        checkpointing.save_checkpoint(path, header={}, model_params=params, verify_replicas=True)
    _assert_nothing_written(path)

    monkeypatch.setenv(checkpointing.VERIFY_REPLICAS_ENV, "1")
    checkpointing.save_checkpoint(path, header={}, model_params=params, verify_replicas=False)
    assert os.path.exists(path)


def test_save_keeps_nd_replicated_tensor(tp_mesh, tmp_path):
    """A tensor replicated under an N-D label (``[1, 2] / (Replicate, Replicate)``, what an explicit N-D mapper makes)
    saves at its own shape with the replica check on and off. The check stacks one axis's copies on a tensor dim, and
    the composer needs a distinct dim per mesh axis ("dims must be unique"), so the other ``Replicate`` axis must not
    reuse it."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    mapper = ttnn.create_mesh_mapper(device, ttnn.MeshMapperConfig([ttnn.PlacementReplicate() for _ in tp_mesh.shape]))
    data = np.random.default_rng(4).standard_normal((1, 1, 32, 32), dtype=np.float32)
    t = ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
    assert len(ttml.Sharding.from_tensor(t).dist_shape) == 2, "precondition: an N-D label"
    params = ttml.NamedParameters()
    params["nd"] = t
    for verify in (True, False):
        path = str(tmp_path / f"nd_{verify}.ckpt")
        checkpointing.save_checkpoint(path, header={}, model_params=params, verify_replicas=verify)
        _, record = _stream_records(path)
        assert tuple(record.shape) == (1, 1, 32, 32), (verify, record.shape)


def test_save_rejects_param_not_at_expected_shape(tp_mesh, tmp_path, expect_error):
    """``expected_shapes`` holds a parameter to an absolute full shape: the one check that also catches a parameter
    whose own label is wrong (parameter and moments gathering to the same wrong shape). A wrong expectation is
    rejected with the leaf and both shapes named; the correct expectations pass; a name no tensor carries is an
    error rather than a vacuous pass.

    Negative control: ``expected_shapes`` did not exist before, so the first call raised ``TypeError`` (unknown
    kwarg) rather than ``CheckpointShapeError``, and no shape was ever held to an external reference."""
    params, opt = _trained_tp()
    name = _sharded_param_name(params)
    full = _gathered_shape(params[name])
    per_device = tuple(params[name].get_value(NATIVE).shape)
    assert per_device != full, "precondition: a sharded parameter's per-device shape differs from its full shape"

    path = str(tmp_path / "expected.ckpt")
    with expect_error(CheckpointShapeError, rf"model\[{name}\] gathers to .* expected_shapes says"):
        checkpointing.save_checkpoint(
            path, header={}, model_params=params, optimizer=opt, expected_shapes={name: per_device}
        )
    _assert_nothing_written(path)

    with expect_error(ValueError, "expected_shapes names tensors not in this checkpoint"):
        checkpointing.save_checkpoint(path, header={}, model_params=params, expected_shapes={"no/such/param": full})
    _assert_nothing_written(path)

    checkpointing.save_checkpoint(
        path,
        header={},
        model_params=params,
        optimizer=opt,
        expected_shapes={n: _gathered_shape(t) for n, t in params.items()},
    )
    assert os.path.exists(path)


def test_save_warns_and_skips_state_not_in_model_params(tp_mesh, tmp_path):
    """Optimizer state keyed by a name that is not in ``model_params`` has no parameter to compare against, so it is
    written as-is and reported in one summary warning (count plus the leaf names) instead of rejected
    (``MuonWithAdamW`` splits the params across two inner optimizers, and callers may checkpoint a subset).
    Everything else in the same save is still checked.

    Negative control: no warning was emitted before the guard (``pytest.warns`` would fail with "did not warn")."""
    params, opt = _trained_tp()
    dropped = _sharded_param_name(params)
    subset = ttml.NamedParameters()
    for n, t in params.items():
        if n != dropped:
            subset[n] = t
    assert len(subset) == len(params) - 1

    # Every moment the optimizer keeps for the dropped parameter (AdamW: exp_avg, exp_avg_sq, plus max_exp_avg_sq
    # under amsgrad) is one unchecked leaf. The summary names at most three, so here all of them must appear.
    unchecked = [
        f"optimizer/{key}[{dropped}]"
        for key, node in opt.get_state_dict().items()
        if isinstance(node, ttml.NamedParameters) and dropped in node
    ]
    assert 1 <= len(unchecked) <= 3, unchecked

    path = str(tmp_path / "subset.ckpt")
    with pytest.warns(UserWarning, match="not parameters of the model being saved") as rec:
        checkpointing.save_checkpoint(path, header={}, model_params=subset, optimizer=opt)
    summaries = [str(w.message) for w in rec if "not parameters of the model being saved" in str(w.message)]
    assert len(summaries) == 1, f"expected one summary warning, not one per leaf: {summaries}"
    for fragment in (f"{len(unchecked)} optimizer state tensor", "saved as-is", *unchecked):
        assert fragment in summaries[0], f"missing {fragment!r} in:\n{summaries[0]}"
    assert os.path.exists(path) and not os.path.exists(path + ".tmp")

    # The unchecked leaf was written at its own gathered shape; the rest of the file is intact.
    header, *records = _stream_records(path)
    keys = _record_keys(header["manifest"])
    assert ("model", dropped) not in keys
    assert tuple(records[keys.index(("optimizer", "exp_avg", dropped))].shape) == _gathered_shape(params[dropped])


def test_save_without_model_params_warns_once(tp_mesh, tmp_path):
    """An optimizer saved alone cannot be cross-checked at all; say so once rather than per leaf."""
    params, opt = _trained_tp()
    path = str(tmp_path / "opt_only.ckpt")
    with pytest.warns(UserWarning, match="saving an optimizer without model_params") as rec:
        checkpointing.save_checkpoint(path, header={}, optimizer=opt)
    assert sum("without model_params" in str(w.message) for w in rec) == 1
    assert sum("not parameters of the model being saved" in str(w.message) for w in rec) == 0
    assert os.path.exists(path)


# --- load --------------------------------------------------------------------------------------------------------


def test_load_rejects_truncated_optimizer_record(tp_mesh, tmp_path, expect_error):
    """A checkpoint already on disk with one ``exp_avg`` record saved as a single shard (what the pre-guard saver
    produced for a mislabelled moment) must be refused on resume, whether the model is restored alongside or the
    optimizer is loaded alone (the model group is read for its shapes either way). The parameter's own record is
    intact, so the refusal comes from the optimizer-vs-model cross-check, before anything is installed.

    Negative control: before the guard, ``load_checkpoint`` redistributed the half-size record by the live
    ``Shard`` mapper and ``set_state_dict`` installed a quarter-size moment without complaint."""
    params, opt = _trained_tp()
    name = _sharded_param_name(params)
    good = str(tmp_path / "good.ckpt")
    checkpointing.save_checkpoint(good, header={"step": 1}, model_params=params, optimizer=opt)

    bad = str(tmp_path / "bad.ckpt")
    truncated = _write_with_truncated_record(good, bad, ("optimizer", "exp_avg", name), _shard_dim(params[name]))
    full = _gathered_shape(params[name])
    assert truncated != full

    _, fresh_params, fresh_opt = _fresh_tp()
    with expect_error(CheckpointShapeError, rf"optimizer/exp_avg\[{name}\] record has shape") as excinfo:
        checkpointing.load_checkpoint(bad, model_params=fresh_params, optimizer=fresh_opt)
    message = str(excinfo.value)
    assert str(tuple(truncated)) in message and str(full) in message, message
    assert tuple(fresh_opt.get_state_dict()["exp_avg"][name].get_value(NATIVE).shape) == tuple(
        params[name].get_value(NATIVE).shape
    ), "the live moment must be untouched by a refused load"

    _, _, opt_only = _fresh_tp()
    with expect_error(CheckpointShapeError, rf"optimizer/exp_avg\[{name}\] record has shape"):
        checkpointing.load_checkpoint(bad, optimizer=opt_only)

    # Positive control: the untampered file restores into the same fresh pair.
    _, fresh_params, fresh_opt = _fresh_tp()
    assert checkpointing.load_checkpoint(good, model_params=fresh_params, optimizer=fresh_opt) == {"step": 1}


def test_load_rejects_truncated_record_without_model_group(tp_mesh, tmp_path, expect_error):
    """With no model group in the file there is nothing to cross-check against, so the per-device check is the last
    line: the half-size record redistributed by the live ``Shard(dim)`` mapper lands at a quarter of the parameter
    per device, which is not the live moment's shape.

    Negative control: ``set_state_dict`` has no shape check of its own, so before the guard this installed the
    wrong-size moment silently."""
    params, opt = _trained_tp()
    name = _sharded_param_name(params)
    good = str(tmp_path / "opt_good.ckpt")
    with pytest.warns(UserWarning, match="without model_params"):
        checkpointing.save_checkpoint(good, header={}, optimizer=opt)

    bad = str(tmp_path / "opt_bad.ckpt")
    _write_with_truncated_record(good, bad, ("optimizer", "exp_avg", name), _shard_dim(params[name]))

    _, _, fresh_opt = _fresh_tp()
    with expect_error(CheckpointShapeError, rf"optimizer/exp_avg\[{name}\] record of shape .* per-device"):
        checkpointing.load_checkpoint(bad, optimizer=fresh_opt)


def test_load_rejects_truncated_model_record(tp_mesh, tmp_path, expect_error):
    """A parameter record saved as one shard (a parameter whose own label was wrong) has nothing earlier in the file
    to be cross-checked against, so the per-device check in the model group is the last line: the half-size record
    redistributed by the live ``Shard(dim)`` mapper lands at a quarter of the parameter per device, and the load
    refuses before ``assign`` touches the live parameter.

    Negative control: ``assign`` (``AutocastTensor::set_tensor``) has no shape check of its own, so before the guard
    this installed the quarter-size parameter silently (the ``expect_error`` would have failed with "did not raise")."""
    params, opt = _trained_tp()
    name = _sharded_param_name(params)
    good = str(tmp_path / "model_good.ckpt")
    checkpointing.save_checkpoint(good, header={}, model_params=params, optimizer=opt)

    bad = str(tmp_path / "model_bad.ckpt")
    _write_with_truncated_record(good, bad, ("model", name), _shard_dim(params[name]))

    _, fresh_params, _ = _fresh_tp()
    live_shape = tuple(fresh_params[name].get_value(NATIVE).shape)
    with expect_error(CheckpointShapeError, rf"model\[{name}\] record of shape .* per-device"):
        checkpointing.load_checkpoint(bad, model_params=fresh_params)
    assert tuple(fresh_params[name].get_value(NATIVE).shape) == live_shape, "a refused load must not assign"
