# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Small target-composition numerical models shared by compiler and scorer.

No compiler, schedule or artifact loading is needed to evaluate a declared
composition. Coefficients are supplied by its typed declaration/sidecar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ttpoly.precision.fma import fma_bh

if TYPE_CHECKING:
    from ttpoly.spec.target_class_merge import TargetSelectedClassMergeRelation


@dataclass(frozen=True)
class NativeEvenPolyvalExteriorComposite:
    coefficients: tuple[float, ...]
    class_merge: TargetSelectedClassMergeRelation | None = None
    kind: str = "native_even_polyval_exterior"


def _bf16_rne_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) >> np.uint32(16)).astype(np.uint16)


def native_even_polyval_exterior_output(
    raw_bf16: np.ndarray,
    composite: NativeEvenPolyvalExteriorComposite,
) -> np.ndarray:
    """Replay the declared public SFPU expression for every raw BF16 word.

    The polynomial is ``1 + u*P10(u)``, where ``u=x*x``.  In particular, the
    outer multiplication is part of the public expression; omitting it would
    silently change the degree from 22 to 20 as a polynomial in ``x``.
    """
    raw = np.asarray(raw_bf16, dtype=np.uint16)
    values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    result = np.empty(raw.size, dtype=np.float32)
    coefficients = tuple(np.float32(value) for value in composite.coefficients)
    for index, value in enumerate(values):
        coordinate = fma_bh(value, value, np.float32(0.0))
        polynomial = coefficients[-1]
        for coefficient in reversed(coefficients[:-1]):
            polynomial = fma_bh(polynomial, coordinate, coefficient)
        result[index] = fma_bh(polynomial, coordinate, np.float32(1.0))
    return _bf16_rne_bits(result)


def native_even_polyval_exterior_finite_mask(
    raw_bf16: np.ndarray,
    composite: NativeEvenPolyvalExteriorComposite,
    *,
    open_interval_bound: float,
) -> np.ndarray:
    """Return finite exterior rows derived from the expression and BF16 egress."""
    raw = np.asarray(raw_bf16, dtype=np.uint16)
    values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    exponent = raw & np.uint16(0x7F80)
    finite_normal = (exponent != 0) & (exponent != np.uint16(0x7F80))
    exterior = finite_normal & (np.abs(values) >= np.float32(open_interval_bound))
    output = native_even_polyval_exterior_output(raw, composite)
    return exterior & ((output & np.uint16(0x7F80)) != np.uint16(0x7F80))
