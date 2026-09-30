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

import ml_dtypes
from tqdm import tqdm

import ttnn
import ttml

from .sharding import Sharding

FORMAT_VERSION = 1

NATIVE = ttml.autograd.PreferredPrecision.NATIVE


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
    val = tensor.get_value(NATIVE)
    if val.dtype == ttnn.DataType.FLOAT32:
        dtype = "FLOAT32"
    elif val.dtype == ttnn.DataType.BFLOAT16:
        dtype = "BFLOAT16"
    else:
        raise ValueError(f"checkpointing: unsupported tensor dtype {val.dtype} (only FLOAT32 and BFLOAT16)")
    return {"layout": "ROW_MAJOR" if val.get_layout() == ttnn.Layout.ROW_MAJOR else "TILE", "dtype": dtype}


def _tensor_from_record(meta: dict, data, mapper) -> ttml.autograd.Tensor:
    """Rebuild a Tensor from header `meta` + its streamed `data`, distributed onto the mesh via `mapper`
    (None on a unit mesh; see `Sharding.derive_mapper`)."""
    layout = ttnn.Layout.ROW_MAJOR if meta["layout"] == "ROW_MAJOR" else ttnn.Layout.TILE
    if meta["dtype"] == "FLOAT32":
        arr, new_type = data, ttnn.DataType.FLOAT32
    else:
        arr, new_type = data.astype(ml_dtypes.bfloat16), ttnn.DataType.BFLOAT16
    return ttml.autograd.Tensor.from_numpy(arr, layout=layout, new_type=new_type, mapper=mapper)


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
    if isinstance(value, (bool, int, float)):
        return value, []
    raise ValueError(f"checkpointing: unsupported state value of type {type(value).__name__}")


def _check_device_shape(path: tuple, name: str, new: ttml.autograd.Tensor, live: ttml.autograd.Tensor, record_shape):
    """Refuse to install `new` over `live` unless the per-device shapes agree.

    `assign` / `set_state_dict` replace the underlying tensor without any shape check, so a record that does
    not describe a tensor of the live layout (one shard saved as if it were the full tensor, redistributed by
    the live mapper) would otherwise land silently at the wrong size."""
    got = tuple(new.get_value(NATIVE).shape)
    want = tuple(live.get_value(NATIVE).shape)
    if got != want:
        raise CheckpointShapeError(
            f"checkpointing: {_leaf_label(path, name)} record of shape {tuple(record_shape)} redistributes to a "
            f"per-device shape {got}, but the live tensor is {want} per device "
            f"({Sharding.from_tensor(live).describe()}). The record does not hold a full tensor of this layout -- "
            f"most likely it was saved as one device's shard through a wrong topology label. Nothing was restored "
            f"past this point; re-save from a run whose tensors gather to their full shape."
        )


def _rebuild(
    node, live_node, f, model_shapes: dict | None, display_progress: bool = False, path: tuple = ("optimizer",)
):
    """Reconstruct a skeleton `node` from stream `f`, resharding each tensor per the live `live_node`.

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
                    f"wrong topology label). Refusing to restore it; re-save from a run whose state gathers to the "
                    f"full shape."
                )
            live = live_node[name]
            new = _tensor_from_record(meta, data, Sharding.from_tensor(live).derive_mapper())
            _check_device_shape(path, name, new, live, data.shape)
            named[name] = new
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
        new = _tensor_from_record(meta, data, Sharding.from_tensor(params[name]).derive_mapper())
        _check_device_shape(("model",), name, new, params[name], data.shape)
        params[name].assign(new)
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


def _check_gathered_shape(
    group: str,
    path: tuple,
    name: str,
    tensor: ttml.autograd.Tensor,
    shape: tuple,
    model_shapes: dict,
    expected_shapes: dict | None,
    have_model: bool,
) -> None:
    """Hold one gathered tensor to the shape it must have before it is written (see `CheckpointShapeError`).

    Model params are recorded in `model_shapes` and held to `expected_shapes` when given. Optimizer leaves are held
    to their parameter's gathered shape when the parameter is in this checkpoint, else to `expected_shapes`; a leaf
    whose name is no model param at all is skipped with a warning (an optimizer over a subset of the params)."""
    label = _leaf_label((group, *path), name)
    if group == "model":
        model_shapes[name] = shape
        reference, source = (expected_shapes or {}).get(name), "expected_shapes"
    elif name in model_shapes:
        reference, source = model_shapes[name], f"its parameter model[{name}]"
    elif expected_shapes is not None and name in expected_shapes:
        reference, source = expected_shapes[name], "expected_shapes"
    else:
        if have_model:
            warnings.warn(
                f"checkpointing: {label} is not a parameter of the model being saved, so its gathered shape {shape} "
                f"cannot be cross-checked; saving it as-is",
                stacklevel=3,
            )
        return
    if reference is None or tuple(reference) == shape:
        return
    raise CheckpointShapeError(
        f"checkpointing: {label} gathers to {shape} but {source} says {tuple(reference)}. The tensor is laid out as "
        f"{Sharding.from_tensor(tensor).describe()}; if that label does not describe its data (e.g. Replicate on a "
        f"mesh axis the parameter is sharded on), gathering by it keeps one device's copy and the checkpoint would "
        f"hold truncated state. Nothing was written. Fix the label where the tensor was produced (the op that "
        f"relabelled it), not the checkpoint."
    )


def save_checkpoint(
    path: str,
    *,
    header: dict | None = None,
    model_params=None,
    optimizer=None,
    expected_shapes: dict[str, tuple] | None = None,
    display_progress: bool = False,
) -> None:
    """Write `header` (opaque) plus `model_params` and/or the `optimizer`'s state to `path`.

    `model_params` is a `NamedParameters` (e.g. `module.parameters()`); `optimizer` is an `OptimizerBase`.
    Tensors are gathered to the host one at a time (peak host mem ~one tensor) and streamed after the header
    record. Writes a temp file then atomically renames, so a crash mid-write leaves a previous checkpoint intact.

    Every optimizer state tensor keyed by a parameter in `model_params` must gather to that parameter's shape,
    else `CheckpointShapeError` is raised and no file is written (the `.tmp` is removed): a mismatch means one
    side's topology label does not describe its data and the file would hold truncated state. State keyed by a
    name that is not in `model_params` is written with a warning; an optimizer saved without `model_params`
    cannot be cross-checked (warned once).

    `expected_shapes` maps parameter name -> full (gathered) shape and is held against the matching model params
    and optimizer leaves. It is the only check that also catches a parameter whose own label is wrong (both the
    parameter and its moments gathering to the same wrong shape). TEST-FACING for now: trainers cannot populate it
    yet, because the global shape of a sharded parameter is not retained on the `Parameter` (only TP-aware
    modules know it, FSDP stores none, and lazy init discards it at materialize); a follow-up keeps it there.
    Names in `expected_shapes` that no written tensor carries are an error, so a check cannot pass vacuously.
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

    model_shapes: dict = {}
    seen = set()
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump({"format": FORMAT_VERSION, "header": header or {}, "manifest": manifest}, f)
            for group, sub_path, name, tensor in _progress(
                records, total=len(records), desc="Saving checkpoint", enabled=display_progress
            ):
                data = Sharding.from_tensor(tensor).gather(tensor)  # gather one at a time; freed after dump
                _check_gathered_shape(
                    group,
                    sub_path,
                    name,
                    tensor,
                    tuple(data.shape),
                    model_shapes,
                    expected_shapes,
                    model_params is not None,
                )
                seen.add(name)
                pickle.dump(data, f)
        if expected_shapes is not None and (unknown := set(expected_shapes) - seen):
            raise ValueError(f"checkpointing: expected_shapes names tensors not in this checkpoint: {sorted(unknown)}")
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)  # a rejected or interrupted save leaves no half-written file behind
        raise
    os.replace(tmp_path, path)


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
