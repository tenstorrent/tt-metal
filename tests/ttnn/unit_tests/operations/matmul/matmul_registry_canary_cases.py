# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure lock-to-canary case selection for the populated matmul registry."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True)
class CanarySemanticCase:
    domain: str
    beta_bits: int | None = None

    @property
    def beta(self) -> float | None:
        if self.beta_bits is None:
            return None
        return struct.unpack("!f", struct.pack("!I", self.beta_bits))[0]

    @property
    def test_id(self) -> str:
        if self.beta_bits is None:
            return self.domain
        return f"{self.domain}-beta-{self.beta_bits:08x}"


class CanaryLockError(ValueError):
    """The checked lock cannot define a safe populated-canary case set."""


_DOMAIN_ORDER = {"dense.matmul": 0, "dense.linear": 1, "dense.addmm": 2}
_ONE_F32_BITS = 0x3F800000
_ZERO_F32_BITS = frozenset({0, 0x80000000})


def represented_semantic_cases(
    lock: dict[str, Any], *, require_populated: bool = False
) -> tuple[CanarySemanticCase, ...]:
    """Return each public operation/scalar semantic represented by the lock.

    Full lock validation remains owned by the build-time emitter. This selector
    repeats the small operation boundary it uses to decide which silicon calls
    to make, and rejects malformed values instead of silently skipping them.
    """

    entries = lock.get("entries")
    if not isinstance(entries, list):
        raise CanaryLockError("matmul registry lock entries must be an array")
    cases: set[CanarySemanticCase] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CanaryLockError(f"matmul registry entry {index} must be an object")
        domain = entry.get("domain")
        key = entry.get("key")
        if domain not in _DOMAIN_ORDER or not isinstance(key, dict):
            raise CanaryLockError(f"matmul registry entry {index} has an unsupported dense domain or key")
        alpha_bits = key.get("alpha_f32_bits")
        beta_bits = key.get("beta_f32_bits")
        if domain == "dense.addmm":
            if (
                isinstance(alpha_bits, bool)
                or not isinstance(alpha_bits, int)
                or alpha_bits != _ONE_F32_BITS
                or isinstance(beta_bits, bool)
                or not isinstance(beta_bits, int)
                or beta_bits not in _ZERO_F32_BITS
            ):
                raise CanaryLockError(f"matmul registry entry {index} has unsupported addmm scalar semantics")
            cases.add(CanarySemanticCase(domain=domain, beta_bits=beta_bits))
        else:
            if alpha_bits is not None or beta_bits is not None:
                raise CanaryLockError(f"matmul registry entry {index} leaks scalar semantics across domains")
            cases.add(CanarySemanticCase(domain=domain))
    result = tuple(sorted(cases, key=lambda case: (_DOMAIN_ORDER[case.domain], case.beta_bits or 0)))
    if require_populated and not result:
        raise CanaryLockError("matmul registry lock has no valid dense entry")
    return result
