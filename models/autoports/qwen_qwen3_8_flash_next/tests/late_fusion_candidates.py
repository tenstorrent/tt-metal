# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent late graph-fusion experiments; never selected by production.

Call ``install(name)`` before constructing the decoder. ``CANDIDATE_NAMES``
contains the CLI names for a benchmark driver. Each installation restores the
three original factory symbols before replacing exactly one component.
These candidates require measured correctness, trace, and latency validation.
"""

import inspect
import textwrap

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused

_BASE_HC = fused.FusedHyperConnection
_BASE_GDN = fused.FusedGatedDeltaNet
_BASE_QSA = fused.FusedQSA


def _rewrite(method, replacements):
    """Copy one existing method and fail loudly if its expected text changes."""
    source = textwrap.dedent(inspect.getsource(method))
    for old, new in replacements:
        count = source.count(old)
        if count == 0 and source.count(new) == 1:
            continue  # The selected production path may already contain this exact rewrite.
        if count != 1:
            raise RuntimeError(f"Expected one replacement in {method.__qualname__}, found {count}: {old!r}")
        source = source.replace(old, new, 1)
    namespace = dict(method.__globals__)
    exec(compile(source, f"{__file__}:{method.__qualname__}", "exec"), namespace)
    rewritten = namespace[method.__name__]
    rewritten.__module__ = __name__
    rewritten._rewritten_source = source
    return rewritten


class AddcmulWriteHC(_BASE_HC):
    """Fuse both short vector and multi-token split residual writes.

    addcmul rounds its final BF16 result, whereas the control also rounds the
    product. That numerical difference is intentional and must be evaluated.
    """

    def write(self, residual, update, inject):
        if residual.shape[2] > 1:
            return ttnn.concat(
                [
                    ttnn.addcmul(self._part(residual, i), update, inject[..., i : i + 1], value=1.0)
                    for i in range(self.count)
                ],
                dim=-1,
            )
        b, seq = tuple(residual.shape)[1:3]
        streams = ttnn.reshape(residual, (1, b * seq, self.count, self.width))
        update = ttnn.reshape(update, (1, b * seq, 1, self.width))
        inject = ttnn.reshape(inject, (1, b * seq, self.count, 1))
        result = ttnn.addcmul(streams, update, inject, value=1.0)
        return ttnn.reshape(result, residual.shape)


class AddcmulReadHC(_BASE_HC):
    """Fuse only the three prefill accumulation steps; short read is unchanged."""

    _read_prefill = _rewrite(
        _BASE_HC._read_prefill,
        [
            (
                "mixed = ttnn.add(mixed, ttnn.multiply(self._part(mix, i), streams[i]))",
                "mixed = ttnn.addcmul(mixed, self._part(mix, i), streams[i], value=1.0)",
            )
        ],
    )


class L2ActivationGDN(_BASE_GDN):
    """Keep FP32 square/sum/epsilon and fold rsqrt into the final multiply."""

    @staticmethod
    def _l2norm(x):
        x = ttnn.typecast(x, ttnn.float32)
        squared_norm = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
        return ttnn.multiply(
            x,
            ttnn.add(squared_norm, 1e-6),
            input_tensor_b_activations=[ttnn.UnaryOpType.RSQRT],
        )


class SoftplusActivationGDN(_BASE_GDN):
    """Fold the existing default-beta/threshold softplus into decay multiply."""

    _chunk = _rewrite(
        _BASE_GDN._chunk,
        [
            (
                "g = ttnn.multiply(self.a_neg, ttnn.softplus(ttnn.add(a, self.dt_bias)))",
                "g = ttnn.multiply(self.a_neg, ttnn.add(a, self.dt_bias), "
                "input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)])",
            )
        ],
    )


class PositionMACQSA(_BASE_QSA):
    """Reuse next position and fuse complete-block subtraction into native MAC."""

    _chunk_forward = _rewrite(
        _BASE_QSA._chunk_forward,
        [
            (
                "complete = ttnn.floor(ttnn.multiply(ttnn.add(positions, 1.0), 0.25))",
                "next_position = ttnn.add(positions, 1.0)\n"
                "    complete = ttnn.floor(ttnn.multiply(next_position, 0.25))",
            ),
            (
                "tail_count = ttnn.subtract(ttnn.add(positions, 1.0), ttnn.multiply(complete, 4.0))",
                "tail_count = ttnn.mac(complete, -4.0, next_position)",
            ),
        ],
    )


_CANDIDATES = {
    "addcmul_write": ("FusedHyperConnection", AddcmulWriteHC),
    "addcmul_read": ("FusedHyperConnection", AddcmulReadHC),
    "l2_activation": ("FusedGatedDeltaNet", L2ActivationGDN),
    "softplus_activation": ("FusedGatedDeltaNet", SoftplusActivationGDN),
    "position_mac": ("FusedQSA", PositionMACQSA),
}
CANDIDATE_NAMES = tuple(_CANDIDATES)


def install(name):
    """Select one experiment for subsequent decoder construction; return its class."""
    if name not in _CANDIDATES:
        raise ValueError(f"Unknown candidate {name!r}; choose one of {CANDIDATE_NAMES}")
    fused.FusedHyperConnection = _BASE_HC
    fused.FusedGatedDeltaNet = _BASE_GDN
    fused.FusedQSA = _BASE_QSA
    symbol, candidate = _CANDIDATES[name]
    setattr(fused, symbol, candidate)
    return candidate
