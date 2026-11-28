# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA adapter save/load in PEFT/diffusers key format. """

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
import ttnn
from safetensors.numpy import load_file, save_file

import ttml

if TYPE_CHECKING:
    from pipeline_config import Config

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

# Parameters live in bf16, and ttml's autocast cache never refreshes the FULL/float32
# view after an in-place optimizer write (ttml/autograd/autocast_tensor.cpp, tracking
# #41657). A default to_numpy() therefore returns whatever the first read cached, so
# every checkpoint after the first is a stale snapshot. NATIVE reads the stored value.
_NATIVE = ttml.autograd.PreferredPrecision.NATIVE


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


def _iter_lora_params(model: ttml.modules.ModuleBase) -> Iterator[tuple[str, ttml.autograd.Tensor]]:
    for name, tensor in model.named_parameters():
        if name.endswith(("lora_A", "lora_B")):
            yield name, tensor


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _gather(tensor: ttml.autograd.Tensor, name: str, mesh_shape: Sequence[int]) -> np.ndarray:
    dp_size, tp_size = tuple(mesh_shape)
    if tp_size == 1 and dp_size == 1:
        return np.asarray(tensor.to_numpy(precision=_NATIVE), dtype=np.float32)

    device = _device()
    if _is_col_parallel_lora_B(name):
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 2)
        arr = np.asarray(tensor.to_numpy(composer=composer, precision=_NATIVE), dtype=np.float32)
        if dp_size > 1:
            arr = arr[:, :, : arr.shape[2] // dp_size, :]
    elif _is_row_parallel_lora_A(name):
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 3)
        arr = np.asarray(tensor.to_numpy(composer=composer, precision=_NATIVE), dtype=np.float32)
        if dp_size > 1:
            arr = arr[:, :, :, : arr.shape[3] // dp_size]
    else:
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        arr = np.asarray(tensor.to_numpy(composer=composer, precision=_NATIVE), dtype=np.float32)
        arr = arr[:1]
    return arr


def _scatter(w_np: np.ndarray, name: str, mesh_shape: Sequence[int]) -> ttml.autograd.Tensor:
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


def init_lora_A_gaussian(
    model: ttml.modules.ModuleBase, rank: int, mesh_shape: Sequence[int] = (1, 1), seed: int = 0
) -> int:
    """Overwrite every lora_A with N(0, 1/rank), matching PEFT's "gaussian" init.

    ttml initializes lora_A kaiming-uniform, ~4x smaller than PEFT at rank 32.
    dW = B @ A and dL/dB scales with A, so the smaller init yields a weaker
    adapter for the same step count. lora_B is left at its zero init.
    """
    rng = np.random.default_rng(seed)
    std = 1.0 / float(rank)
    initialized = 0
    for name, tensor in _iter_lora_params(model):
        if not name.endswith("lora_A"):
            continue
        # _gather returns the logical 4-D shape _scatter expects back.
        shape = _gather(tensor, name, mesh_shape).shape
        w_np = np.ascontiguousarray(rng.normal(0.0, std, size=shape), dtype=np.float32)
        tensor.set_value(_scatter(w_np, name, mesh_shape).get_value())
        initialized += 1
    if not initialized:
        raise RuntimeError("no lora_A parameters found — was the adapter injected?")
    return initialized


def lora_state_dict(model: ttml.modules.ModuleBase, mesh_shape: Sequence[int] = (1, 1)) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    for name, tensor in _iter_lora_params(model):
        arr = _gather(tensor, name, mesh_shape)
        # ttml is 4-D (1, 1, out, in); PEFT/diffusers store 2-D.
        state[_to_diffusers_key(name)] = np.ascontiguousarray(arr.reshape(arr.shape[-2], arr.shape[-1]))
    if not state:
        raise RuntimeError("no LoRA parameters found — was the adapter injected before training?")
    return state


def save_lora_expert(model: ttml.modules.ModuleBase, path: str, mesh_shape: Sequence[int] = (1, 1)) -> int:
    """Write one expert's adapter. Returns the tensor count so the caller reports it."""
    state = lora_state_dict(model, mesh_shape)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path))
    return len(state)


def load_lora_expert(model: ttml.modules.ModuleBase, path: str, mesh_shape: Sequence[int] = (1, 1)) -> int:
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


def _with_suffix(path: str, suffix: str) -> str:
    if not suffix:
        return path
    return str(Path(path).with_name(Path(path).stem + suffix + ".safetensors"))


def save_all(experts: dict[str, ttml.modules.ModuleBase], cfg: Config, suffix: str = "") -> None:
    for role, model in experts.items():
        path = _with_suffix(cfg.expert_path(role), suffix)
        n = save_lora_expert(model, path, cfg.MESH_SHAPE)
        print(f"[save] wrote {n} LoRA tensors -> {path}")


def load_all(experts: dict[str, ttml.modules.ModuleBase], cfg: Config, suffix: str = "") -> None:
    """Load adapters back into live experts. Not a training resume: the optimizer moments,
    step counter and data order are not restored, so this is a warm start, not a continuation.
    """
    for role, model in experts.items():
        path = _with_suffix(cfg.expert_path(role), suffix)
        n = load_lora_expert(model, path, cfg.MESH_SHAPE)
        print(f"[load] restored {n} LoRA tensors <- {path}")
