# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Running blocks in series.

A :class:`Chain` is an ordered list of :class:`Step`, run one after another over
a shared :class:`Registers` namespace. That is the whole model. It knows nothing
about the hardware — the data-transfer blocks are turned into steps by
:class:`~.golden.Golden` — so a chain is free to be any shape:

    unpack, unpack, math, pack                  a binary op
    unpack, math, pack, unpack, math, pack      two passes
    Chain().repeat(8, [unpack, math, pack])     a loop
    unpack, math, pack, pack                    one result to two buffers

Steps declare what they read and write, so a chain can be traced stage by stage
and a mistyped register name fails where it is used rather than silently reading
something stale.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import torch


class Registers:
    """The named slots a chain passes between steps.

    Holds whatever a step wants to put there — L1 byte buffers, src-register
    tensors, Dest, loop counters. No slot is special.
    """

    def __init__(self, **initial: Any):
        self._slots: Dict[str, Any] = dict(initial)

    def __getitem__(self, name: str) -> Any:
        if name not in self._slots:
            raise KeyError(
                f"register {name!r} has not been written; "
                f"available: {sorted(self._slots) or 'none'}"
            )
        return self._slots[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self._slots[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self._slots

    def get(self, name: str, default: Any = None) -> Any:
        return self._slots.get(name, default)

    def names(self) -> List[str]:
        return sorted(self._slots)

    def __repr__(self) -> str:
        return f"Registers({', '.join(self.names())})"


@dataclass
class Step:
    """One unit of work over the registers.

    Args:
        name: shown in a trace.
        run: does the work, reading and writing `registers`.
        reads / writes: register names, for the trace and for a dry-run check.
    """

    name: str
    run: Callable[[Registers], None]
    reads: Sequence[str] = ()
    writes: Sequence[str] = ()


@dataclass
class StageRecord:
    """What one step did, when running with ``trace=True``."""

    index: int
    name: str
    writes: Sequence[str]
    summary: str

    def __str__(self) -> str:
        wrote = ",".join(self.writes) or "-"
        return f"{self.index:>2}. {self.name:24} -> {wrote:10} {self.summary}"


def summarise(value: Any) -> str:
    """One-line description of a register's contents, for traces."""
    if isinstance(value, torch.Tensor):
        finite = value.float()[torch.isfinite(value.float())]
        rng = f"[{finite.min():.4g}, {finite.max():.4g}]" if finite.numel() else "[]"
        return f"{tuple(value.shape)} {value.dtype} {rng}"
    if isinstance(value, (list, bytes, bytearray)):
        return f"{len(value)} B"
    return repr(value)


class Chain:
    """An ordered list of steps run over shared registers.

    Build one by construction or with :meth:`then` / :meth:`repeat`; chains also
    concatenate with ``+``.
    """

    def __init__(self, steps: Iterable[Step] = ()):
        self.steps: List[Step] = list(steps)

    # ---- building -----------------------------------------------------

    def then(self, *steps: Step) -> "Chain":
        """Append steps. Returns self, so calls read left to right."""
        self.steps.extend(steps)
        return self

    def repeat(self, times: int, steps: Iterable[Step]) -> "Chain":
        """Append `steps` `times` over — a loop, unrolled.

        Steps are re-run against the same registers, so a loop body that wants
        distinct inputs per iteration should be built with a comprehension
        instead, giving each iteration its own step objects.
        """
        body = list(steps)
        for _ in range(times):
            self.steps.extend(body)
        return self

    def __add__(self, other: "Chain") -> "Chain":
        return Chain(self.steps + other.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __repr__(self) -> str:
        return f"Chain({' -> '.join(s.name for s in self.steps) or 'empty'})"

    # ---- running ------------------------------------------------------

    def run(
        self,
        registers: Optional[Registers] = None,
        *,
        result: Optional[str] = None,
        trace: Optional[List[StageRecord]] = None,
    ) -> Any:
        """Run every step in order.

        Returns the whole :class:`Registers` namespace, or just one slot when
        `result` names it. Pass a list as `trace` to collect a per-step record.
        """
        regs = registers if registers is not None else Registers()
        for index, step in enumerate(self.steps):
            step.run(regs)
            if trace is not None:
                trace.append(
                    StageRecord(
                        index,
                        step.name,
                        tuple(step.writes),
                        " | ".join(summarise(regs.get(w)) for w in step.writes)
                        or summarise(None),
                    )
                )
        return regs[result] if result is not None else regs

    def dry_run(self, available: Iterable[str] = ()) -> List[str]:
        """Report steps that read a register nothing has written yet.

        Catches a mistyped or out-of-order register name without running any
        arithmetic. Returns the problems found, empty if the chain is sound.
        """
        known = set(available)
        problems = []
        for index, step in enumerate(self.steps):
            missing = [name for name in step.reads if name not in known]
            if missing:
                problems.append(f"step {index} ({step.name}) reads unwritten {missing}")
            known.update(step.writes)
        return problems
