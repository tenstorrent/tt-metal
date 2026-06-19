# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint primitives: save/load a model and/or an optimizer with an opaque caller header.

A checkpoint file is one header record followed by one gathered-array record per tensor, so peak host
memory stays ~one tensor instead of the whole model + optimizer."""

from __future__ import annotations

import os
import pickle

import ml_dtypes
from tqdm import tqdm

import ttnn
import ttml

from .sharding import Sharding

FORMAT_VERSION = 1

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


def _tensor_from_record(meta: dict, data, mapper) -> ttml.autograd.Tensor:
    """Rebuild a Tensor from header `meta` + its streamed `data`, distributed onto the mesh via `mapper`
    (None on a unit mesh; see `Sharding.derive_mapper`)."""
    layout = ttnn.Layout.ROW_MAJOR if meta["layout"] == "ROW_MAJOR" else ttnn.Layout.TILE
    if meta["dtype"] == "FLOAT32":
        arr, new_type = data, ttnn.DataType.FLOAT32
    else:
        arr, new_type = data.astype(ml_dtypes.bfloat16), ttnn.DataType.BFLOAT16
    return ttml.autograd.Tensor.from_numpy(arr, layout=layout, new_type=new_type, mapper=mapper)


def _walk(value) -> tuple:
    """A state value -> (skeleton, ordered tensor list); tensors stream in this DFS order.

    `NamedParameters` is a leaf encoded as `{"named_parameters": {name: meta}}`; a `dict` recurses
    (composite optimizers, e.g. MuonWithAdamW, nest sub-state dicts); scalars pass through into the skeleton.
    """
    if isinstance(value, ttml.NamedParameters):
        skeleton = {"named_parameters": {}}
        tensors = []
        for name, tensor in value.items():
            skeleton["named_parameters"][name] = _tensor_meta(tensor)
            tensors.append(tensor)
        return skeleton, tensors
    if isinstance(value, dict):
        skeleton = {}
        tensors = []
        for key, sub in value.items():
            skeleton[key], sub_tensors = _walk(sub)
            tensors.extend(sub_tensors)
        return skeleton, tensors
    if isinstance(value, (bool, int, float)):
        return value, []
    raise ValueError(f"checkpointing: unsupported state value of type {type(value).__name__}")


def _rebuild(node, live_node, f, display_progress: bool = False, label: str = "optimizer"):
    """Reconstruct a skeleton `node` from stream `f`, resharding each tensor per the live `live_node`.

    `label` names the current sub-state (e.g. AdamW's `exp_avg`/`exp_avg_sq`) so each leaf's progress bar
    is distinguishable rather than a string of identical "Loading optimizer" bars."""
    if not isinstance(node, dict):
        return node  # scalar
    if "named_parameters" in node:
        named = ttml.NamedParameters()
        leaf = node["named_parameters"]
        for name, meta in _progress(leaf.items(), total=len(leaf), desc=f"Loading {label}", enabled=display_progress):
            data = pickle.load(f)
            mapper = Sharding.from_tensor(live_node[name]).derive_mapper()
            named[name] = _tensor_from_record(meta, data, mapper)
        return named
    return {key: _rebuild(v, live_node[key], f, display_progress, key) for key, v in node.items()}


def _skip(node, f) -> None:
    """Read and discard the tensor records a skeleton `node` describes, keeping the stream aligned."""
    if not isinstance(node, dict):
        return
    if "named_parameters" in node:
        for _ in node["named_parameters"]:
            pickle.load(f)
        return
    for sub in node.values():
        _skip(sub, f)


def _load_params(params: ttml.NamedParameters, skeleton: dict, f, display_progress: bool = False) -> None:
    """Stream a model's params back into live `params` in place, validating coverage."""
    leaf = skeleton["named_parameters"]
    restored = set()
    for name, meta in _progress(leaf.items(), total=len(leaf), desc="Loading model", enabled=display_progress):
        data = pickle.load(f)  # read every record to keep the stream aligned, even when skipping
        if name not in params:
            continue
        mapper = Sharding.from_tensor(params[name]).derive_mapper()
        params[name].assign(_tensor_from_record(meta, data, mapper))
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


def _load_optimizer(
    optimizer: ttml.optimizers.OptimizerBase, skeleton: dict, f, display_progress: bool = False
) -> None:
    """Stream optimizer state back, resharding each moment per the live state dict, then set it."""
    live = optimizer.get_state_dict()
    optimizer.set_state_dict(
        {key: _rebuild(node, live[key], f, display_progress, key) for key, node in skeleton.items()}
    )


def save_checkpoint(
    path: str, *, header: dict | None = None, model_params=None, optimizer=None, display_progress: bool = False
) -> None:
    """Write `header` (opaque) plus `model_params` and/or the `optimizer`'s state to `path`.

    `model_params` is a `NamedParameters` (e.g. `module.parameters()`); `optimizer` is an `OptimizerBase`.
    Tensors are gathered to the host one at a time (peak host mem ~one tensor) and streamed after the header
    record. Writes a temp file then atomically renames, so a crash mid-write leaves a previous checkpoint intact.
    """
    manifest = {}
    tensors = []
    if model_params is not None:
        manifest["model"], model_tensors = _walk(model_params)
        tensors.extend(model_tensors)
    if optimizer is not None:
        manifest["optimizer"], optimizer_tensors = _walk(optimizer.get_state_dict())
        tensors.extend(optimizer_tensors)

    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({"format": FORMAT_VERSION, "header": header or {}, "manifest": manifest}, f)
        for tensor in _progress(tensors, total=len(tensors), desc="Saving checkpoint", enabled=display_progress):
            pickle.dump(Sharding.from_tensor(tensor).gather(tensor), f)  # gather one at a time; freed after dump
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
    """
    with open(path, "rb") as f:
        record = _read_record0(f)
        manifest = record["manifest"]
        requested = {name for name, target in (("model", model_params), ("optimizer", optimizer)) if target is not None}
        absent = requested - set(manifest)
        if absent:
            raise ValueError(f"checkpointing: requested group(s) {sorted(absent)} not in checkpoint {sorted(manifest)}")
        for name, skeleton in manifest.items():  # file order owns the stream order
            if name == "model" and model_params is not None:
                _load_params(model_params, skeleton, f, display_progress)
            elif name == "optimizer" and optimizer is not None:
                _load_optimizer(optimizer, skeleton, f, display_progress)
            else:
                _skip(skeleton, f)
    return record["header"]
