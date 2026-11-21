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
their state words; ``ttml.init``'s own numpy generator, which draws the initial weights of every layer built
(a model grown after a resume, LoRA adapters added later); and torch's CPU generator when torch is imported.

Restoring on a world with another number of hosts: a rank takes its own entry when the checkpoint has one,
otherwise it keeps its current generators and warns.

The dataloader state is stored once (``"dataloader"``) when every rank's is the same, and per world rank
(``"dataloader_ranks"``) when the ranks' differ, as data-parallel loaders that keep a seed or a shard of their own do.
A rank then restores its own; one the checkpoint holds none for (a world with more ranks than the save, or a rank
that saved without a dataloader) is missing it, which a strict load refuses: a loader that starts over while the
others resume would repeat data without a word.

``load`` checks what it can before it touches anything (the RNG block's format, and that this rank's entry decodes)
and restores in an order that keeps a refusal cheap: the scheduler first (its ``set_state_dict`` leaves it unchanged
when it refuses a state), then the dataloader, then the generators. A component that refuses its state raises a
``CheckpointError`` naming it.
"""

from __future__ import annotations

import os
import random
import sys
import warnings
from typing import Any

import numpy as np
import ttnn

import ttml

from ._errors import CheckpointError, IncompleteCheckpointError, MissingKeyError, VersionError
from ._index import HEADER_NAME, check_json, read_checked_header

TRAIN_STATE_KEY = "train_state"
RNG_FORMAT = 1
# A std::mt19937's state words. get_generator_state writes them as text, the C++ standard library's form: libstdc++
# writes the 624 words and the position in [0, 624], libc++ the 624 words alone, starting at the position.
_MT19937_WORDS = 624


def _hex_u32(words) -> str:
    return np.asarray(words, dtype="<u4").tobytes().hex()


def _u32_from_hex(text: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(text), dtype="<u4")


def this_world_rank() -> int:
    """This process's rank in the MPI world; 0 when no distributed context is initialized."""
    if ttnn.distributed_context_is_initialized():
        return int(ttnn.distributed_context_world_rank())
    return 0


def _mt19937_text_tokens() -> int:
    """How many tokens this build's ttml generator text has: 625 with libstdc++, 624 with libc++."""
    return len(ttml.autograd.AutoContext.get_instance().get_generator_state().split())


def capture_rng(world_rank: int | None = None) -> dict:
    """The state of every generator this process draws from, as JSON, under ``world_rank``'s entry (by default this
    process's world rank): ttml, python, numpy, ttml.init and torch when it is imported. A multi-rank save merges every
    rank's entry into one block."""
    rank = this_world_rank() if world_rank is None else int(world_rank)
    return {"format": RNG_FORMAT, "ranks": {str(rank): capture_process_rng()}}


def capture_process_rng() -> dict:
    """This process's generators: ``ttml`` (text), ``python`` and ``numpy`` (hex words), ``ttml_init`` (its PCG64
    state, the 128-bit integers in hex), ``torch`` when imported."""
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
    init_state = ttml.init.get_rng_state()
    if init_state["bit_generator"] == "PCG64":
        state["ttml_init"] = {
            "bit_generator": "PCG64",
            "state": format(init_state["state"]["state"], "x"),
            "inc": format(init_state["state"]["inc"], "x"),
            "has_uint32": int(init_state["has_uint32"]),
            "uinteger": int(init_state["uinteger"]),
        }
    else:
        warnings.warn(
            f"ttml.init's generator is a {init_state['bit_generator']}, not the PCG64 manual_seed makes; its state is "
            "not saved",
            stacklevel=2,
        )
    torch = sys.modules.get("torch")
    if torch is not None:
        state["torch"] = torch.get_rng_state().numpy().tobytes().hex()
    return state


def _check_rng_format(state: dict) -> None:
    """``state`` must be an RNG block, as ``capture_rng`` makes it, of a format this reader knows."""
    if not isinstance(state, dict) or "format" not in state or not isinstance(state.get("ranks"), dict):
        found = sorted(state) if isinstance(state, dict) else type(state).__name__
        raise CheckpointError(
            f"not an RNG block (found {found}); it needs 'format' and 'ranks': pass the train state's 'rng' entry"
        )
    try:
        version = int(state["format"])
    except (TypeError, ValueError):
        raise CheckpointError(f"RNG state format {state['format']!r} is not a version number") from None
    if version > RNG_FORMAT:
        raise VersionError(f"RNG state format {version} is newer than this reader ({RNG_FORMAT})")


def restore_rng(state: dict, world_rank: int | None = None) -> list[str]:
    """Restore the generators ``capture_rng`` recorded for ``world_rank`` (by default this process's world rank);
    returns the names restored (``rng.ttml``, ``rng.python``, ``rng.numpy``, ``rng.ttml_init``, ``rng.torch``). A rank
    without an entry keeps its generators, with a warning; torch's is restored only when torch is imported."""
    world_rank = this_world_rank() if world_rank is None else int(world_rank)
    _check_rng_format(state)
    ranks = state["ranks"]
    own = ranks.get(str(int(world_rank)))
    if own is None:
        warnings.warn(
            f"the checkpoint has no RNG state for world rank {world_rank} (it has ranks {sorted(ranks)}); keeping "
            "the current generators",
            stacklevel=2,
        )
        return []
    return restore_process_rng(own)


def _decode_process_rng(state: dict) -> dict[str, Any]:
    """A ``capture_process_rng`` block decoded into what each generator's setter takes, or a ``CheckpointError``:
    every generator is decoded, and set on a throwaway generator of its kind, before any live one is set, so a
    malformed block changes nothing."""
    decoded: dict[str, Any] = {}
    try:
        if state.get("ttml") is not None:
            tokens = str(state["ttml"]).split()
            # The C++ setter streams the text into the live generator and stops partway at anything it cannot read,
            # leaving the generator in neither state, and no throwaway ttml generator exists to try it on. So the text
            # is checked to be what this build's mt19937 writes: as many ASCII-decimal tokens as its own text, the
            # first 624 of them 32-bit words, and a position in [0, 624] after them when its form has one. A text in
            # the other standard library's form would be misread, so it is refused too.
            expected = _mt19937_text_tokens()
            if len(tokens) != expected:
                raise ValueError(
                    f"the ttml generator state has {len(tokens)} tokens, not the {expected} this build writes for an "
                    "mt19937 (a build with the other C++ standard library writes another form)"
                )
            bad = next((i for i, token in enumerate(tokens) if not (token.isascii() and token.isdigit())), None)
            if bad is not None:
                raise ValueError(
                    f"token {bad} of the ttml generator state, {tokens[bad]!r}, is not an ASCII decimal number"
                )
            bad = next((i for i, token in enumerate(tokens[:_MT19937_WORDS]) if int(token) > 0xFFFFFFFF), None)
            if bad is not None:
                raise ValueError(f"word {bad} of the ttml generator state, {tokens[bad]}, does not fit in 32 bits")
            if expected > _MT19937_WORDS and int(tokens[-1]) > _MT19937_WORDS:
                raise ValueError(f"the ttml generator state's position, {tokens[-1]}, is past {_MT19937_WORDS}")
            decoded["ttml"] = " ".join(tokens)
        python_state = state.get("python")
        if python_state is not None:
            words = tuple(int(w) for w in _u32_from_hex(python_state["state"]))
            gauss_next = python_state["gauss_next"]
            # setstate stores the cached Gaussian as given; the next random.gauss would fail on anything but a number.
            if gauss_next is not None and type(gauss_next) not in (int, float):
                raise TypeError(f"python's cached Gaussian is a {type(gauss_next).__name__}, not a number or null")
            decoded["python"] = (int(python_state["version"]), words, gauss_next)
            random.Random().setstate(decoded["python"])
        numpy_state = state.get("numpy")
        if numpy_state is not None:
            decoded["numpy"] = (
                numpy_state["bit_generator"],
                _u32_from_hex(numpy_state["state"]).copy(),
                int(numpy_state["pos"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
            # set_state takes any position, and the next draw then reads past the 624 words: zeros, or a crash.
            if not 0 <= decoded["numpy"][2] <= _MT19937_WORDS:
                raise ValueError(f"numpy's position {decoded['numpy'][2]} is outside [0, {_MT19937_WORDS}]")
            np.random.RandomState().set_state(decoded["numpy"])
        init_state = state.get("ttml_init")
        if init_state is not None:
            decoded["ttml_init"] = {
                "bit_generator": init_state["bit_generator"],
                "state": {"state": int(init_state["state"], 16), "inc": int(init_state["inc"], 16)},
                "has_uint32": int(init_state["has_uint32"]),
                "uinteger": int(init_state["uinteger"]),
            }
            # PCG64's state and increment are 128-bit; it would wrap a negative one rather than refuse it.
            for name, value in decoded["ttml_init"]["state"].items():
                if not 0 <= value < 2**128:
                    raise ValueError(f"ttml.init's PCG64 {name} {value:#x} is not a 128-bit value")
            np.random.PCG64().state = decoded["ttml_init"]
        if state.get("torch") is not None:
            decoded["torch"] = np.frombuffer(bytes.fromhex(state["torch"]), dtype=np.uint8).copy()
            torch = sys.modules.get("torch")
            if torch is not None:
                torch.Generator().set_state(torch.from_numpy(decoded["torch"]))
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, RuntimeError) as err:
        raise CheckpointError(
            f"the stored RNG state is malformed ({type(err).__name__}: {err}); pass rng=False to load to restore "
            "everything else and keep the current generators"
        ) from err
    return decoded


def restore_process_rng(state: dict) -> list[str]:
    """Restore this process's generators from a ``capture_process_rng`` block; returns the names restored. A generator
    the block holds no state for (numpy's when it was not MT19937 at save, torch's when torch was imported only now)
    keeps its state, with a warning; so does torch's stored state when torch is not imported."""
    decoded = _decode_process_rng(state)
    torch = sys.modules.get("torch")
    live = ["ttml", "python", "numpy", "ttml_init"] + (["torch"] if torch is not None else [])
    unstored = [name for name in live if name not in decoded]
    if unstored:
        warnings.warn(
            f"the checkpoint holds no state for these generators, which keep their current state: {unstored}",
            stacklevel=2,
        )
    if "torch" in decoded and torch is None:
        warnings.warn(
            "the checkpoint holds torch's generator state, but torch is not imported: not restored", stacklevel=2
        )
    restored: list[str] = []
    if "ttml" in decoded:
        ttml.autograd.AutoContext.get_instance().set_generator_state(decoded["ttml"])
        restored.append("rng.ttml")
    if "python" in decoded:
        random.setstate(decoded["python"])
        restored.append("rng.python")
    if "numpy" in decoded:
        np.random.set_state(decoded["numpy"])
        restored.append("rng.numpy")
    if "ttml_init" in decoded:
        ttml.init.set_rng_state(decoded["ttml_init"])
        restored.append("rng.ttml_init")
    if "torch" in decoded and torch is not None:
        torch.set_rng_state(torch.from_numpy(decoded["torch"]))
        restored.append("rng.torch")
    return restored


DATALOADER_RANKS = "dataloader_ranks"  # the per-rank dataloader states: {world rank: state}


def dataloader_state(state: dict, world_rank: int) -> tuple[bool, Any]:
    """``(found, dataloader state)`` for this world rank: the one copy every rank shares, or its own when the states
    were stored per rank."""
    if "dataloader" in state:
        return True, state["dataloader"]
    own = state.get(DATALOADER_RANKS, {})
    return str(int(world_rank)) in own, own.get(str(int(world_rank)))


def capture_train_state(*, scheduler=None, dataloader=None, rng: bool = True, world_rank: int = 0) -> dict:
    """The ``train_state`` block ``save`` stores: whichever of the scheduler, the dataloader and the generators
    are asked for."""
    state: dict[str, Any] = {}
    if scheduler is not None:
        state["scheduler"] = check_json("scheduler state", scheduler.get_state_dict())
    if dataloader is not None:
        state["dataloader"] = check_json("dataloader state", dataloader.get_state_dict())
    if rng:
        state["rng"] = capture_rng(world_rank)
    return state


def missing_train_state(
    state: dict, *, scheduler=None, dataloader=None, rng: bool = False, world_rank: int = 0
) -> list[str]:
    """``train_state/<component>`` for every component asked for that the checkpoint does not hold (a dataloader
    stored per rank: none for this world rank)."""
    held = {
        "scheduler": "scheduler" in state,
        "dataloader": dataloader_state(state, world_rank)[0],
        "rng": "rng" in state,
    }
    wanted = (("scheduler", scheduler is not None), ("dataloader", dataloader is not None), ("rng", rng))
    return [f"{TRAIN_STATE_KEY}/{name}" for name, asked in wanted if asked and not held[name]]


def missing_train_state_error(missing: list[str], *, rng_by_default: bool) -> MissingKeyError:
    """A strict load's refusal of components asked for that the checkpoint lacks. When the RNG block is one of them and
    the caller left ``rng`` at its default, which asks for it on a resume, it says how to load without it."""
    message = f"the checkpoint holds no state for {missing}"
    if rng_by_default and f"{TRAIN_STATE_KEY}/rng" in missing:
        message += (
            "; rng was not passed, and a load that restores a scheduler or dataloader restores the generators too: "
            "pass rng=False to keep the current ones (a checkpoint saved with rng=False has none)"
        )
    return MissingKeyError(message)


def check_train_state(state: dict, *, scheduler=None, dataloader=None, rng: bool = False, world_rank: int = 0) -> None:
    """What can be checked before ``load`` touches anything: that a scheduler or dataloader passed can take a state
    (a ``set_state_dict``), the RNG block's format, and that this rank's entry decodes. The scheduler and dataloader
    states themselves can only be judged by their own ``set_state_dict``, after the tensors are placed."""
    held = {"scheduler": "scheduler" in state, "dataloader": dataloader_state(state, world_rank)[0]}
    for name, component in (("scheduler", scheduler), ("dataloader", dataloader)):
        if component is not None and held[name] and not callable(getattr(component, "set_state_dict", None)):
            raise TypeError(f"the {name} passed ({type(component).__name__}) has no set_state_dict to restore it with")
    if rng and "rng" in state:
        _check_rng_format(state["rng"])
        own = state["rng"]["ranks"].get(str(int(world_rank)))
        if own is not None:
            _decode_process_rng(own)


def _apply(name: str, set_state_dict, value) -> None:
    try:
        set_state_dict(value)
    except CheckpointError:
        raise
    except Exception as err:  # noqa: BLE001  (a component's own refusal: KeyError, ValueError, ...)
        raise CheckpointError(
            f"the live {name} refused the stored {name} state ({type(err).__name__}: {err}); is it configured as when "
            f"the checkpoint was saved, and does it take its state back as JSON types (a tuple comes back as a list)?"
        ) from err


def restore_train_state(
    state: dict, *, scheduler=None, dataloader=None, rng: bool = False, world_rank: int = 0
) -> list[str]:
    """Restore the components asked for that ``state`` holds; returns their names in the order applied. The scheduler
    goes first: its ``set_state_dict`` leaves it unchanged when it refuses, so a refused scheduler state stops the
    restore before the dataloader moves or a generator is set."""
    restored: list[str] = []
    if scheduler is not None and "scheduler" in state:
        _apply("scheduler", scheduler.set_state_dict, state["scheduler"])
        restored.append("scheduler")
    found, loader_state = dataloader_state(state, world_rank)
    if dataloader is not None and found:
        _apply("dataloader", dataloader.set_state_dict, loader_state)
        restored.append("dataloader")
    if rng and "rng" in state:
        restored.extend(restore_rng(state["rng"], world_rank))
    return restored


def train_state_of(header: dict, ckpt_dir: str) -> dict:
    """The ``train_state`` block of a header read with ``read_checked_header``. Every save writes one, so a header
    without it, or with something other than an object there, is refused as unreadable."""
    state = header.get(TRAIN_STATE_KEY)
    if not isinstance(state, dict):
        raise IncompleteCheckpointError(
            f"{os.path.join(ckpt_dir, HEADER_NAME)} has no {TRAIN_STATE_KEY} object (found {type(state).__name__}); "
            "the checkpoint is corrupt or truncated"
        )
    return state


def read_train_state(ckpt_dir: str) -> dict:
    """The ``train_state`` block ``save`` stored (scheduler, dataloader, rng), without touching any device."""
    return train_state_of(read_checked_header(ckpt_dir), ckpt_dir)
