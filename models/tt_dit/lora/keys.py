# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parse an adapter file into classified, layout-neutral entries.

One adapter carries up to four kinds of payload, and a loader that recognises only the first
silently ships a wrong model:

``lora_A`` / ``lora_B``
    The low-rank pair. Spelled ``lora_A``/``lora_B`` by PEFT (with the adapter name interposed
    when it is not ``default``), ``lora_down``/``lora_up`` by kohya and ComfyUI.

``.diff`` / ``.diff_b``
    An exact additive delta for a weight or bias the base model already has. Distillation
    adapters use it where a rank-``r`` factorization buys nothing or cannot be formed -- RMSNorm
    vectors, biases, and matrices whose smaller dimension is already at or below ``r``.

``.set_weight``
    A whole parameter the base model does not carry, so no delta is expressible. Not a delta and
    not loadable against an unmodified architecture; recognised here so it can be rejected with a
    reason instead of being dropped as unparseable.

``.alpha``
    The scale numerator. Absent means the adapter's own scale is 1, **not** ``alpha == rank``:
    synthesising ``alpha = rank`` would silently rescale every adapter that publishes a plain
    ``W + B @ A`` contract.

A key that matches none of these is an error, never a warning. Adapters are published against a
named base revision; an unrecognised key means either the wrong file or a convention this parser
does not yet read, and both are worth stopping for. A key that is simply *absent*, by contrast, is
routine -- publishers drop tensors that came out identical to the base.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

Kind = Literal["lora", "diff", "diff_b", "set_weight"]

# Longest first: ``.diff_b`` must be tested before ``.diff`` would match its truncation.
_DENSE_SUFFIXES: tuple[tuple[str, Kind, str], ...] = (
    (".diff_b", "diff_b", "bias"),
    (".diff", "diff", "weight"),
    (".set_weight", "set_weight", "weight"),
)

_STRIP_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.", "unet.", "model.")

# PEFT interposes the adapter's name when it is not the default: ``.lora_A.myname.weight``.
_ADAPTER_NAME_INFIX = re.compile(r"\.(lora_[AB])\.[^.]+\.weight$")
_SLOT_ALIASES = ((".lora_down", ".lora_A"), (".lora_up", ".lora_B"))
_LOW_RANK_RE = re.compile(r"^(?P<base>.*)\.lora_(?P<slot>A|B)\.weight$")


@dataclass(frozen=True)
class AdapterEntry:
    """One classified payload, addressed by the module path it targets.

    ``path`` is in the *adapter's* keyspace, not the model's -- resolving it to a parameter is
    :mod:`.route`'s job, using the model's own preparation pipeline.
    """

    path: str
    kind: Kind
    A: torch.Tensor | None = None
    B: torch.Tensor | None = None
    delta: torch.Tensor | None = None
    alpha: float | None = None

    @property
    def rank(self) -> int | None:
        return None if self.A is None else self.A.shape[0]

    @property
    def scale(self) -> float:
        """The adapter's own scale, before any caller-supplied strength."""
        if self.alpha is None or self.rank is None:
            return 1.0
        return self.alpha / self.rank


@dataclass
class AdapterStats:
    """What the file contained, for asserting against its own metadata."""

    tensors: int = 0
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    metadata: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
        return f"{self.tensors} tensors ({kinds})"


def normalize_path(raw: str) -> str:
    """Strip publisher prefixes and collapse slot spellings onto ``lora_A``/``lora_B``."""
    for prefix in _STRIP_PREFIXES:
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    raw = _ADAPTER_NAME_INFIX.sub(r".\1.weight", raw)
    for alias, canonical in _SLOT_ALIASES:
        raw = raw.replace(f"{alias}.", f"{canonical}.")
        if raw.endswith(alias):
            raw = f"{raw[: -len(alias)]}{canonical}"
    return raw


def parse_adapter(source: str | Path | Mapping[str, torch.Tensor]) -> tuple[list[AdapterEntry], AdapterStats]:
    """Classify every tensor in an adapter.

    Raises on an unrecognised key, on a half low-rank pair, and on a rank disagreement within one
    pair -- each of those means the delta cannot be reconstructed, and reconstructing it partly is
    worse than not at all.
    """
    stats = AdapterStats()
    raw, stats.metadata = _read(source)
    stats.tensors = len(raw)

    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    alphas: dict[str, float] = {}
    entries: list[AdapterEntry] = []
    unrecognized: list[str] = []

    for raw_key, tensor in raw.items():
        key = normalize_path(raw_key)

        dense = next(((suffix, kind) for suffix, kind, _ in _DENSE_SUFFIXES if key.endswith(suffix)), None)
        if dense is not None:
            suffix, kind = dense
            entries.append(AdapterEntry(path=key[: -len(suffix)], kind=kind, delta=tensor))
            stats.counts[kind] += 1
            continue

        match = _LOW_RANK_RE.match(key)
        if match:
            pairs[match.group("base")][match.group("slot")] = tensor
            continue

        if key.endswith(".alpha"):
            alphas[key[: -len(".alpha")]] = float(tensor.item())
            stats.counts["alpha"] += 1
            continue

        unrecognized.append(raw_key)

    if unrecognized:
        sample = ", ".join(unrecognized[:5])
        more = "" if len(unrecognized) <= 5 else f" (+{len(unrecognized) - 5} more)"
        msg = f"{len(unrecognized)} adapter key(s) matched no known convention: {sample}{more}"
        raise ValueError(msg)

    for path, slots in sorted(pairs.items()):
        missing = {"A", "B"} - set(slots)
        if missing:
            msg = f"{path}: low-rank pair is missing lora_{'/'.join(sorted(missing))}"
            raise ValueError(msg)
        a, b = slots["A"], slots["B"]
        if a.shape[0] != b.shape[1]:
            msg = f"{path}: rank disagrees between A {tuple(a.shape)} and B {tuple(b.shape)}"
            raise ValueError(msg)
        entries.append(AdapterEntry(path=path, kind="lora", A=a, B=b, alpha=alphas.get(path)))
        stats.counts["lora"] += 2

    return entries, stats


def _read(source: str | Path | Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if not isinstance(source, (str, Path)):
        return dict(source), {}
    from safetensors import safe_open

    with safe_open(str(source), framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}, dict(handle.metadata() or {})
