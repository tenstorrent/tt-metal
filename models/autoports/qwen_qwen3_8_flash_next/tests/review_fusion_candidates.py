# SPDX-License-Identifier: Apache-2.0
"""Isolate additional input/output activation folds identified in stage review."""
import inspect
import textwrap

from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused


def rewrite(method, replacements):
    source = textwrap.dedent(inspect.getsource(method))
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"Expected one occurrence of {old!r} in {method.__qualname__}")
        source = source.replace(old, new)
    namespace = dict(method.__globals__)
    exec(compile(source, __file__, "exec"), namespace)
    return namespace[method.__name__]


class SignedSqrtPLE(fused.FusedPLE):
    forward = rewrite(
        fused.FusedPLE.forward,
        [
            (
                "gate = ttnn.multiply(ttnn.sqrt(ttnn.maximum(ttnn.abs(gate), 1e-6)), ttnn.sign(gate))",
                "gate = ttnn.multiply(ttnn.maximum(ttnn.abs(gate), 1e-6), gate, "
                "input_tensor_a_activations=[ttnn.UnaryOpType.SQRT], input_tensor_b_activations=[ttnn.UnaryOpType.SIGN])",
            )
        ],
    )


class FloorQSA(fused.FusedQSA):
    _chunk_forward = rewrite(
        fused.FusedQSA._chunk_forward,
        [
            (
                "block_start = ttnn.multiply(ttnn.floor(ttnn.multiply(positions, 0.25)), 4.0)",
                "block_start = ttnn.multiply(ttnn.multiply(positions, 0.25), 4.0, input_tensor_a_activations=[ttnn.UnaryOpType.FLOOR])",
            ),
            (
                "complete = ttnn.floor(ttnn.multiply(next_position, 0.25))",
                "complete = ttnn.multiply(next_position, 0.25, activations=[ttnn.UnaryOpType.FLOOR])",
            ),
        ],
    )


def install(name):
    if name == "signed_sqrt":
        fused.FusedPLE = SignedSqrtPLE
    elif name == "floor_position":
        fused.FusedQSA = FloorQSA
    else:
        raise ValueError(name)
