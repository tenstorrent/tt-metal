# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Training state that is not a tensor: the LR scheduler, the dataloader and the random number generators.

``save`` stores it in ``header.json`` under ``"train_state"``, next to the caller's header and never inside it, and
``load`` restores it into the objects passed. The scheduler and the dataloader go through their own
``get_state_dict`` / ``set_state_dict`` (the ttml schedulers and ``TTMLDataloader``, or anything with those two
methods) and must be JSON. The RNG block holds one entry per world rank, since every generator is per process: the
ttml generator, the mt19937 in ``AutoContext`` that seeds every ttml random op (dropout, rand, randn) and AdamW's
stochastic rounding, as the text ``get_generator_state`` emits; python's and numpy's global generators as hex of
their state words; and torch's CPU generator when torch is imported.

Restoring on a world with another number of hosts: a rank takes its own entry when the checkpoint has one,
otherwise it keeps its current generators and warns.
"""

from __future__ import annotations

import json
import random
import sys
import warnings
from typing import Any

import numpy as np

import ttml

from ._errors import CheckpointError, VersionError

TRAIN_STATE_KEY = "train_state"
RNG_FORMAT = 1


def _json(name: str, value: Any) -> Any:
    """``value`` as plain JSON types (tuples become lists), or a ``CheckpointError`` naming the component."""
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as err:
        raise CheckpointError(f"{name} state is not JSON: {err}") from None


def _hex_u32(words) -> str:
    return np.asarray(words, dtype="<u4").tobytes().hex()


def _u32_from_hex(text: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(text), dtype="<u4")


def capture_rng(world_rank: int = 0) -> dict:
    """The state of every generator this process draws from, as JSON, under ``world_rank``'s entry: ttml, python,
    numpy and torch when it is imported. A multi-rank save merges every rank's entry into one block."""
    return {"format": RNG_FORMAT, "ranks": {str(int(world_rank)): capture_process_rng()}}


def capture_process_rng() -> dict:
    """This process's generators: ``ttml`` (text), ``python`` and ``numpy`` (hex words), ``torch`` when imported."""
    ctx = ttml.autograd.AutoContext.get_instance()
    version, words, gauss_next = random.getstate()
    state: dict[str, Any] = {
        "ttml": ctx.get_generator_state(),
        "python": {"version": int(version), "state": _hex_u32(words), "gauss_next": gauss_next},
    }
    numpy_state = np.random.get_state()
    if isinstance(numpy_state, tuple) and numpy_state[0] == "MT19937":
        _, keys, pos, has_gauss, cached_gaussian = numpy_state
        state["numpy"] = {
            "bit_generator": "MT19937",
            "state": _hex_u32(keys),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        }
    else:
        warnings.warn("numpy's global generator is not the legacy MT19937; its state is not saved", stacklevel=2)
    torch = sys.modules.get("torch")
    if torch is not None:
        state["torch"] = torch.get_rng_state().numpy().tobytes().hex()
    return state


def restore_rng(state: dict, world_rank: int = 0) -> list[str]:
    """Restore the generators ``capture_rng`` recorded for ``world_rank``; returns the names restored (``rng.ttml``,
    ``rng.python``, ``rng.numpy``, ``rng.torch``). A rank without an entry keeps its generators, with a warning;
    torch's is restored only when torch is imported."""
    if int(state.get("format", RNG_FORMAT)) > RNG_FORMAT:
        raise VersionError(f"RNG state format {state['format']} is newer than this reader ({RNG_FORMAT})")
    ranks = state.get("ranks", {})
    own = ranks.get(str(int(world_rank)))
    if own is None:
        warnings.warn(
            f"the checkpoint has no RNG state for world rank {world_rank} (it has ranks {sorted(ranks)}); keeping "
            "the current generators",
            stacklevel=2,
        )
        return []
    return restore_process_rng(own)


def restore_process_rng(state: dict) -> list[str]:
    """Restore this process's generators from a ``capture_process_rng`` block; returns the names restored."""
    restored: list[str] = []
    ttml_state = state.get("ttml")
    if ttml_state is not None:
        ttml.autograd.AutoContext.get_instance().set_generator_state(ttml_state)
        restored.append("rng.ttml")
    python_state = state.get("python")
    if python_state is not None:
        words = tuple(int(w) for w in _u32_from_hex(python_state["state"]))
        random.setstate((int(python_state["version"]), words, python_state["gauss_next"]))
        restored.append("rng.python")
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                _u32_from_hex(numpy_state["state"]).copy(),
                int(numpy_state["pos"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        restored.append("rng.numpy")
    torch_state = state.get("torch")
    torch = sys.modules.get("torch")
    if torch_state is not None and torch is not None:
        torch.set_rng_state(torch.from_numpy(np.frombuffer(bytes.fromhex(torch_state), dtype=np.uint8).copy()))
        restored.append("rng.torch")
    return restored


def capture_train_state(*, scheduler=None, dataloader=None, rng: bool = True, world_rank: int = 0) -> dict:
    """The ``train_state`` block ``save`` stores: whichever of the scheduler, the dataloader and the generators
    are asked for."""
    state: dict[str, Any] = {}
    if scheduler is not None:
        state["scheduler"] = _json("scheduler", scheduler.get_state_dict())
    if dataloader is not None:
        state["dataloader"] = _json("dataloader", dataloader.get_state_dict())
    if rng:
        state["rng"] = capture_rng(world_rank)
    return state


def missing_train_state(state: dict, *, scheduler=None, dataloader=None, rng: bool = False) -> list[str]:
    """``train_state/<component>`` for every component asked for that the checkpoint does not hold."""
    wanted = (("scheduler", scheduler is not None), ("dataloader", dataloader is not None), ("rng", rng))
    return [f"{TRAIN_STATE_KEY}/{name}" for name, asked in wanted if asked and name not in state]


def restore_train_state(
    state: dict, *, scheduler=None, dataloader=None, rng: bool = False, world_rank: int = 0
) -> list[str]:
    """Restore the components asked for that ``state`` holds; returns their names in the order applied."""
    restored: list[str] = []
    if dataloader is not None and "dataloader" in state:
        dataloader.set_state_dict(state["dataloader"])
        restored.append("dataloader")
    if scheduler is not None and "scheduler" in state:
        scheduler.set_state_dict(state["scheduler"])
        restored.append("scheduler")
    if rng and "rng" in state:
        restored.extend(restore_rng(state["rng"], world_rank))
    return restored


def read_train_state(ckpt_dir: str) -> dict:
    """The ``train_state`` block ``save`` stored (scheduler, dataloader, rng), without touching any device."""
    from ._index import read_header

    return read_header(ckpt_dir).get(TRAIN_STATE_KEY, {})
