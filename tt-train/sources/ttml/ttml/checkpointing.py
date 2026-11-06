# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint primitives: save/load a model and/or an optimizer with an opaque caller header.

A checkpoint file is one header record followed by one gathered-array record per tensor, so peak host
memory stays ~one tensor instead of the whole model + optimizer."""

from __future__ import annotations

import contextlib
import os
import pickle
import warnings

from tqdm import tqdm

import ttnn
import ttml

from .sharding import ReplicaMismatchError, Sharding

FORMAT_VERSION = 1

# What to do about a refused tensor, shared by the refusals of a save and of a load respectively.
_SAVE_REMEDY = (
    "Nothing was written. Fix the label where the tensor was produced (the op that relabelled it), not the checkpoint."
)
_NOTHING_RESTORED = "Nothing was restored past this point"
_LOAD_REMEDY = f"{_NOTHING_RESTORED}; re-save from a run whose tensors gather to their full shape."


class CheckpointShapeError(ValueError):
    """A tensor would be saved or restored at a shape that contradicts the shape it must have.

    Invariant guarded: every optimizer state tensor keyed by a parameter name (AdamW ``exp_avg`` /
    ``exp_avg_sq`` / ``max_exp_avg_sq``, SGD ``theta``, Muon's momentum buffer, the fp32 master copy, ...)
    is created as ``zeros_like(param)`` or ``typecast(param)`` and therefore gathers to the same full shape as
    its parameter. Only shapes are compared -- never dtype (``AdamWFullPrecision`` keeps a float32 master of a
    bf16 parameter).

    ``save_checkpoint`` gathers each tensor by its own topology label, so a label that does not describe the
    data (a moment reporting ``Replicate`` on a mesh axis its parameter is sharded on, say) gathers to one
    device's shard and the file would hold truncated state. Raising here keeps such a file from ever being
    written, and refuses to resume from one that already exists.
    """


# Bars render one indent level under the caller's human bracket lines; descs are left-padded to a
# fixed width so model/optimizer bars line up. {desc} renders raw under a custom bar_format (no ": ").
_BAR_FORMAT = "    {desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
_DESC_WIDTH = 18


def _progress(iterable, *, total: int, desc: str, enabled: bool):
    """Wrap `iterable` in a tqdm bar, rendered only when `enabled`. Off by default."""
    return tqdm(iterable, total=total, desc=desc.ljust(_DESC_WIDTH), disable=not enabled, bar_format=_BAR_FORMAT)


def _tensor_meta(tensor: ttml.autograd.Tensor) -> dict:
    """Per-tensor header metadata (layout + dtype); the gathered data is streamed as a separate record."""
    val = tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE)
    if val.dtype == ttnn.DataType.FLOAT32:
        dtype = "FLOAT32"
    elif val.dtype == ttnn.DataType.BFLOAT16:
        dtype = "BFLOAT16"
    else:
        raise ValueError(f"checkpointing: unsupported tensor dtype {val.dtype} (only FLOAT32 and BFLOAT16)")
    return {"layout": "ROW_MAJOR" if val.get_layout() == ttnn.Layout.ROW_MAJOR else "TILE", "dtype": dtype}


def _load_into(path: tuple, name: str, live: ttml.autograd.Tensor, meta: dict, data) -> None:
    """Load header `meta` + its streamed `data` into the `live` tensor in place. The live tensor decides the sharding
    (its mapper distributes the data; see `Sharding.derive_mapper`) and the dtype (see `assign_numpy`).

    A record that does not land at the live tensor's per-device shape is refused as `CheckpointShapeError`, the live
    tensor untouched. Two causes look the same here: a checkpoint saved from a model of another configuration, and a
    record that holds one device's shard saved as if it were the full tensor."""
    layout = ttnn.Layout.ROW_MAJOR if meta["layout"] == "ROW_MAJOR" else ttnn.Layout.TILE
    sharding = Sharding.from_tensor(live)
    try:
        ttml.autograd.assign_numpy(live, data, layout=layout, mapper=sharding.derive_mapper())
    except ttml.autograd.ShapeMismatchError as e:
        raise CheckpointShapeError(
            f"checkpointing: {_leaf_label(path, name)} record of shape {tuple(data.shape)} does not redistribute to "
            f"the live per-device shape ({e}) of the layout {sharding.describe()}. Either the checkpoint was saved "
            f"from a model of another configuration (a different hidden or vocab size, say), and has to be loaded "
            f"into a model built with that configuration, or the record holds one device's shard saved through a "
            f"wrong topology label, and the checkpoint has to be re-saved from a run whose tensors gather to their "
            f"full shape. {_NOTHING_RESTORED}."
        ) from e


def _leaf_label(path: tuple, name: str) -> str:
    """`("optimizer", "exp_avg"), "Model/fc/weight"` -> `optimizer/exp_avg[Model/fc/weight]`."""
    return f"{'/'.join(path)}[{name}]"


def _walk(value, path: tuple = ()) -> tuple:
    """A state value -> (skeleton, ordered `(path, name, tensor)` list); tensors stream in this DFS order.

    `NamedParameters` is a leaf encoded as `{"named_parameters": {name: meta}}`; a `dict` recurses
    (composite optimizers, e.g. MuonWithAdamW, nest sub-state dicts); scalars pass through into the skeleton.
    `path` is the tuple of dict keys leading to the leaf (empty for a bare `NamedParameters`), and `name` the
    leaf's key -- for optimizer state, the parameter the tensor belongs to.
    """
    if isinstance(value, ttml.NamedParameters):
        skeleton = {"named_parameters": {}}
        tensors = []
        for name, tensor in value.items():
            skeleton["named_parameters"][name] = _tensor_meta(tensor)
            tensors.append((path, name, tensor))
        return skeleton, tensors
    if isinstance(value, dict):
        skeleton = {}
        tensors = []
        for key, sub in value.items():
            skeleton[key], sub_tensors = _walk(sub, path + (key,))
            tensors.extend(sub_tensors)
        return skeleton, tensors
    if isinstance(value, (bool, int, float, str)):
        return value, []
    raise ValueError(f"checkpointing: unsupported state value of type {type(value).__name__}")


def _rebuild(
    node, live_node, f, model_shapes: dict | None, display_progress: bool = False, path: tuple = ("optimizer",)
):
    """Reconstruct a skeleton `node` from stream `f`, loading each tensor into its live counterpart in `live_node`.

    `path` names the current sub-state (e.g. AdamW's `exp_avg`/`exp_avg_sq`) so each leaf's progress bar is
    distinguishable rather than a string of identical "Loading optimizer" bars, and so errors can name the leaf.
    `model_shapes` maps parameter name -> the shape of that parameter's record in this file (None if the file has
    no model group ahead of the optimizer): a state record keyed by a parameter must have that parameter's
    shape, or the checkpoint holds truncated state (see `CheckpointShapeError`)."""
    if not isinstance(node, dict):
        return node  # scalar
    if "named_parameters" in node:
        named = ttml.NamedParameters()
        leaf = node["named_parameters"]
        for name, meta in _progress(
            leaf.items(), total=len(leaf), desc=f"Loading {path[-1]}", enabled=display_progress
        ):
            data = pickle.load(f)
            if model_shapes is not None and name in model_shapes and tuple(data.shape) != model_shapes[name]:
                raise CheckpointShapeError(
                    f"checkpointing: {_leaf_label(path, name)} record has shape {tuple(data.shape)} but the parameter "
                    f"{name!r} was saved at {model_shapes[name]} in the same file. Optimizer state always has its "
                    f"parameter's shape, so this record holds truncated state (saved as one device's shard through a "
                    f"wrong topology label). {_LOAD_REMEDY}"
                )
            _load_into(path, name, live_node[name], meta, data)
            named[name] = live_node[name]
        return named
    return {
        key: _rebuild(v, live_node[key], f, model_shapes, display_progress, path + (key,)) for key, v in node.items()
    }


def _skip(node, f) -> dict:
    """Read and discard the tensor records a skeleton `node` describes, keeping the stream aligned.

    Returns `{name: shape}` of the records read when `node` is a bare `NamedParameters` leaf -- the model
    group -- so a later optimizer group can still be cross-checked against it; `{}` for anything else."""
    if not isinstance(node, dict):
        return {}
    if "named_parameters" in node:
        return {name: tuple(pickle.load(f).shape) for name in node["named_parameters"]}
    for sub in node.values():
        _skip(sub, f)
    return {}


def _load_params(params: ttml.NamedParameters, skeleton: dict, f, display_progress: bool = False) -> dict:
    """Stream a model's params back into live `params` in place, validating coverage.

    Returns `{name: record shape}` for every record in the group (restored or not), for the optimizer cross-check."""
    leaf = skeleton["named_parameters"]
    restored = set()
    shapes = {}
    for name, meta in _progress(leaf.items(), total=len(leaf), desc="Loading model", enabled=display_progress):
        data = pickle.load(f)  # read every record to keep the stream aligned, even when skipping
        shapes[name] = tuple(data.shape)
        if name not in params:
            continue
        _load_into(("model",), name, params[name], meta, data)
        restored.add(name)

    missing = set(params) - restored  # in model, not restored → left at init (dangerous)
    unexpected = set(leaf) - set(params)  # in checkpoint, not in model → ignored
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint restore mismatch: "
            f"{len(missing)} param(s) left at init "
            f"({sorted(missing)[:3]}{'...' if len(missing) > 3 else ''}), "
            f"{len(unexpected)} checkpoint param(s) ignored"
        )
    return shapes


def _load_optimizer(
    optimizer: ttml.optimizers.OptimizerBase,
    skeleton: dict,
    f,
    model_shapes: dict | None,
    display_progress: bool = False,
) -> None:
    """Stream optimizer state back, resharding each moment per the live state dict, then set it."""
    live = optimizer.get_state_dict()
    optimizer.set_state_dict(
        {
            key: _rebuild(node, live[key], f, model_shapes, display_progress, ("optimizer", key))
            for key, node in skeleton.items()
        }
    )


def _check_state_shape(
    path: tuple, name: str, tensor: ttml.autograd.Tensor, shape: tuple, parameter_shape: tuple
) -> None:
    """Hold one gathered optimizer leaf to the gathered shape of its parameter before it is written (see
    `CheckpointShapeError`)."""
    if shape == tuple(parameter_shape):
        return
    raise CheckpointShapeError(
        f"checkpointing: {_leaf_label(('optimizer', *path), name)} gathers to {shape} but its parameter model[{name}] "
        f"gathers to {tuple(parameter_shape)}. The tensor is laid out as {Sharding.from_tensor(tensor).describe()}; if "
        f"that label does not describe its data (e.g. Replicate on a mesh axis the parameter is sharded on), gathering "
        f"by it keeps one device's copy and the checkpoint would hold truncated state. {_SAVE_REMEDY}"
    )


def _gather_record(group: str, path: tuple, name: str, tensor: ttml.autograd.Tensor, verify_replicas: bool):
    """Gather one tensor to the host by its own label, naming the leaf if the copies its label calls replicas differ."""
    try:
        return Sharding.from_tensor(tensor).gather(tensor, verify_replicas=verify_replicas)
    except ReplicaMismatchError as e:
        raise ReplicaMismatchError(f"checkpointing: {_leaf_label((group, *path), name)} is {e}. {_SAVE_REMEDY}") from e


def save_checkpoint(
    path: str,
    *,
    header: dict | None = None,
    model_params=None,
    optimizer=None,
    verify_replicas: bool = True,
    display_progress: bool = False,
) -> None:
    """Write `header` (opaque) plus `model_params` and/or the `optimizer`'s state to `path`.

    `model_params` is a `NamedParameters` (e.g. `module.parameters()`); `optimizer` is an `OptimizerBase`.
    Tensors are gathered to the host one at a time (peak host mem ~one tensor) and streamed after the header
    record. Writes a temp file then atomically renames, so a crash mid-write leaves a previous checkpoint intact.

    Every optimizer state tensor keyed by a parameter in `model_params` must gather to that parameter's shape,
    else `CheckpointShapeError` is raised and no file is written (the `.tmp` is removed): a mismatch means one
    side's topology label does not describe its data and the file would hold truncated state. State keyed by a
    name that is not in `model_params` is written as-is, and one summary warning, issued before the file is
    committed, counts such leaves and names up to three; an optimizer saved without `model_params` warns once,
    since nothing can cross-check its state. Any failure, the final rename included, removes the `.tmp`.

    `verify_replicas` (on by default) compares the values of every copy a tensor's label calls a replica (tile padding
    is not compared) before one copy is saved, for model parameters and optimizer state alike, and raises
    `ReplicaMismatchError` naming the tensor if they differ or the label does not cover the tensor's shards: a
    `Replicate` label over data that differs would otherwise be saved as one device's copy. This is the check that
    covers a parameter whose own label is wrong in that direction. `False` restores the label-trusting gather.
    """
    manifest = {}
    records = []  # (group, path, name, tensor) in stream order
    if model_params is not None:
        manifest["model"], model_tensors = _walk(model_params)
        records.extend(("model", *entry) for entry in model_tensors)
    if optimizer is not None:
        manifest["optimizer"], optimizer_tensors = _walk(optimizer.get_state_dict())
        records.extend(("optimizer", *entry) for entry in optimizer_tensors)
        if model_params is None:
            warnings.warn(
                "checkpointing: saving an optimizer without model_params, so its state tensors cannot be checked "
                "against their parameters' shapes (a mislabelled moment would be saved truncated)",
                stacklevel=2,
            )

    model_shapes: dict = {}  # parameter name -> gathered shape, recorded as the model group streams
    unchecked = []  # optimizer leaves with no parameter in this checkpoint to be held to
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump({"format": FORMAT_VERSION, "header": header or {}, "manifest": manifest}, f)
            for group, sub_path, name, tensor in _progress(
                records, total=len(records), desc="Saving checkpoint", enabled=display_progress
            ):
                data = _gather_record(group, sub_path, name, tensor, verify_replicas)  # one at a time; freed after dump
                shape = tuple(data.shape)
                if group == "model":
                    model_shapes[name] = shape
                elif name in model_shapes:
                    _check_state_shape(sub_path, name, tensor, shape, model_shapes[name])
                else:
                    unchecked.append(_leaf_label((group, *sub_path), name))
                pickle.dump(data, f)
        if unchecked and model_params is not None:
            # Before the rename: with warnings configured as errors this raises inside the transaction, so the
            # previous checkpoint stays in place and the .tmp is removed, not a committed file plus an exception.
            warnings.warn(
                f"checkpointing: {len(unchecked)} optimizer state tensor(s) are keyed by names that are not parameters "
                f"of the model being saved, so their gathered shapes could not be cross-checked; saved as-is: "
                f"{', '.join(unchecked[:3])}{' ...' if len(unchecked) > 3 else ''}",
                stacklevel=2,
            )
        os.replace(tmp_path, path)  # atomic; inside the try so a failed rename leaves no .tmp behind either
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)  # a rejected, interrupted or uncommitted save leaves no half-written file behind
        raise


def _read_record0(f) -> dict:
    """Read + validate the header record written by `save_checkpoint`."""
    try:
        record = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, AttributeError) as e:
        raise ValueError(f"checkpointing: could not read checkpoint header: {e}") from e
    if not isinstance(record, dict) or record.get("format") != FORMAT_VERSION:
        got = record.get("format") if isinstance(record, dict) else type(record).__name__
        raise ValueError(
            f"checkpointing: not a ttml checkpoint or unsupported format (expected {FORMAT_VERSION}, got {got})"
        )
    return record


def read_header(path: str) -> dict:
    """Return the opaque caller header — reads the first record only, no tensor data."""
    with open(path, "rb") as f:
        return _read_record0(f)["header"]


def load_checkpoint(path: str, *, model_params=None, optimizer=None, display_progress: bool = False) -> dict:
    """Restore `model_params` (assigned in place) and/or the `optimizer`'s state from `path`.

    `model_params` is the live `NamedParameters`; `optimizer` is the live `OptimizerBase` (restored via
    `set_state_dict`), each resharded onto its current mesh layout. A group present in the file but not
    requested here is skipped (e.g. loading only the model for inference); requesting a group the file
    lacks is an error. Returns the opaque header.

    Raises `CheckpointShapeError` before installing anything at the wrong size: an optimizer record must have
    the shape its parameter was saved at in the same file (the model group precedes the optimizer group in files
    `save_checkpoint` writes, and is read for its shapes even when not requested), and every record must
    redistribute to the live tensor's per-device shape. A file without a model group gets the per-device check only.
    """
    with open(path, "rb") as f:
        record = _read_record0(f)
        manifest = record["manifest"]
        requested = {name for name, target in (("model", model_params), ("optimizer", optimizer)) if target is not None}
        absent = requested - set(manifest)
        if absent:
            raise ValueError(f"checkpointing: requested group(s) {sorted(absent)} not in checkpoint {sorted(manifest)}")
        model_shapes = None  # parameter name -> record shape, once the model group has streamed past
        for name, skeleton in manifest.items():  # file order owns the stream order
            if name == "model":
                model_shapes = (
                    _load_params(model_params, skeleton, f, display_progress)
                    if model_params is not None
                    else _skip(skeleton, f)
                )
            elif name == "optimizer" and optimizer is not None:
                _load_optimizer(optimizer, skeleton, f, model_shapes, display_progress)
            else:
                _skip(skeleton, f)
    return record["header"]
