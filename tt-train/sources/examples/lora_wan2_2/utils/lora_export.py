# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA adapter save/load in PEFT/diffusers key format. See README.md for the format."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import ttnn
from safetensors.numpy import load_file, save_file

import ttml

# ttml leaf module name -> diffusers leaf path
_LEAF_RENAMES = [
    (re.compile(r"(attn[12])\.to_out$"), r"\1.to_out.0"),
    (re.compile(r"ffn\.ff1$"), "ffn.net.0.proj"),
    (re.compile(r"ffn\.ff2$"), "ffn.net.2"),
]

_LORA_SLOT_RE = re.compile(r"^(?P<base>.+)\.(?P<slot>lora_A|lora_B)$")

# Under TP, column-parallel projections shard lora_B; row-parallel shard lora_A.
_COL_PARALLEL = ("to_q", "to_k", "to_v", "ff1")
_ROW_PARALLEL = ("to_out", "ff2")


def _to_diffusers_key(ttml_param_name: str) -> str:
    name = ttml_param_name
    if name.startswith("model."):
        name = name[len("model.") :]

    m = _LORA_SLOT_RE.match(name)
    if m is None:
        raise ValueError(f"not a LoRA parameter name: {ttml_param_name!r}")
    base, slot = m.group("base"), m.group("slot")

    for pattern, repl in _LEAF_RENAMES:
        base, n = pattern.subn(repl, base)
        if n:
            break

    return f"transformer.{base}.{slot}.weight"


def _is_col_parallel_lora_B(name: str) -> bool:
    return any(f".{proj}.lora_B" in name for proj in _COL_PARALLEL)


def _is_row_parallel_lora_A(name: str) -> bool:
    return any(f".{proj}.lora_A" in name for proj in _ROW_PARALLEL)


def _iter_lora_params(model):
    for name, tensor in model.named_parameters():
        if name.endswith(("lora_A", "lora_B")):
            yield name, tensor


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _gather(tensor, name: str, mesh_shape) -> np.ndarray:
    dp_size, tp_size = tuple(mesh_shape)
    if tp_size == 1 and dp_size == 1:
        return np.asarray(tensor.to_numpy(), dtype=np.float32)

    device, val_tt = _device(), tensor.get_value()
    if _is_col_parallel_lora_B(name):
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 2)
        arr = np.asarray(ttnn.to_torch(val_tt, mesh_composer=composer).float().numpy(), dtype=np.float32)
        if dp_size > 1:
            arr = arr[:, :, : arr.shape[2] // dp_size, :]
    elif _is_row_parallel_lora_A(name):
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 3)
        arr = np.asarray(ttnn.to_torch(val_tt, mesh_composer=composer).float().numpy(), dtype=np.float32)
        if dp_size > 1:
            arr = arr[:, :, :, : arr.shape[3] // dp_size]
    else:
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        arr = np.asarray(ttnn.to_torch(val_tt, mesh_composer=composer).float().numpy(), dtype=np.float32)
        arr = arr[:1]
    return arr


def _scatter(w_np: np.ndarray, name: str, mesh_shape):
    _, tp_size = tuple(mesh_shape)
    mapper = None
    if tp_size > 1:
        device = _device()
        if _is_col_parallel_lora_B(name):
            mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 2, 1)
        elif _is_row_parallel_lora_A(name):
            mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 3, 1)
    if mapper is not None:
        return ttml.autograd.Tensor.from_numpy(w_np, ttnn.Layout.TILE, ttnn.bfloat16, mapper)
    return ttml.autograd.Tensor.from_numpy(w_np, ttnn.Layout.TILE, ttnn.bfloat16)


def lora_state_dict(model, mesh_shape=(1, 1)) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    for name, tensor in _iter_lora_params(model):
        arr = _gather(tensor, name, mesh_shape)
        # ttml is 4-D (1, 1, out, in); PEFT/diffusers store 2-D.
        state[_to_diffusers_key(name)] = np.ascontiguousarray(arr.reshape(arr.shape[-2], arr.shape[-1]))
    if not state:
        raise RuntimeError("no LoRA parameters found — was the adapter injected before training?")
    return state


def save_lora_expert(model, path: str, mesh_shape=(1, 1)) -> None:
    state = lora_state_dict(model, mesh_shape)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path))
    print(f"[save] wrote {len(state)} LoRA tensors -> {path}")


def save_all(experts: dict, cfg, suffix: str = "") -> None:
    for role, model in experts.items():
        p = cfg.expert_path(role)
        if suffix:
            p = str(Path(p).with_name(Path(p).stem + suffix + ".safetensors"))
        save_lora_expert(model, p, cfg.MESH_SHAPE)


def load_lora_expert(model, path: str, mesh_shape=(1, 1)) -> int:
    state = load_file(str(path))
    restored = 0
    for name, tensor in _iter_lora_params(model):
        key = _to_diffusers_key(name)
        if key not in state:
            raise KeyError(f"checkpoint {path} has no entry for {key!r} (from ttml param {name!r})")
        w_np = np.ascontiguousarray(state[key], dtype=np.float32)[None, None]
        tensor.set_value(_scatter(w_np, name, mesh_shape).get_value())
        restored += 1
    return restored
