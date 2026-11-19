# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-only FP32 arithmetic for the selection-sensitive QSA projection.

The target indexer maps BF16 [1,1,T,2560] to BF16 [1,1,T,640], T=1..32.
Its discrete top-k amplifies small projection errors: the full-context
diagnostic loses a key with almost unit attention mass. FP32-output HiFi4
matmul still differs before output rounding; the measured SFPU product and
addition control restores near-exact HF BF16 results. See
doc/functional_decoder/qsa_indexer_causal_summary.json for evidence.

Weights are expanded to FP32 in setup. A pass uses FP32 elementwise products
and balanced additions, then rounds once to BF16. In particular, it does not
use FPU sum reduction, which would pass FP32 products through narrower Src
registers. Static shape loops and device allocations are trace-capturable;
there is no host conversion, tensor readback, or data-dependent host branch.
"""

import ttnn


def precise_indexer_rope(value, cos, sin):
    """Match HF's three BF16 boundaries for a 128-wide indexer tensor.

    The two products round separately, then their sum rounds. Exact-input
    controls over 242 chunks establish bitwise agreement; native BF16 arithmetic
    agrees on only about 95% of those elements. The nonrotary half is unchanged.
    """
    rotated = ttnn.concat((ttnn.neg(value[..., 32:64]), value[..., :32]), -1)
    first = ttnn.typecast(
        ttnn.multiply(ttnn.typecast(value[..., :64], ttnn.float32), ttnn.typecast(cos, ttnn.float32)),
        ttnn.bfloat16,
    )
    second = ttnn.typecast(
        ttnn.multiply(ttnn.typecast(rotated, ttnn.float32), ttnn.typecast(sin, ttnn.float32)), ttnn.bfloat16
    )
    result = ttnn.typecast(
        ttnn.add(ttnn.typecast(first, ttnn.float32), ttnn.typecast(second, ttnn.float32)), ttnn.bfloat16
    )
    return ttnn.concat((result, value[..., 64:]), -1)


class PreciseProjection:
    def __init__(self, loader, name, output_width):
        """Setup only: retain exact [output_width,2560] weights in 32-row blocks."""
        weight = loader.state_dict[name]
        if output_width % 32 or tuple(weight.shape) != (output_width, 2560):
            raise ValueError(f"Expected [{output_width},2560] projection weight, width divisible by 32")
        self.weights = [
            loader.tensor(weight[start : start + 32].reshape(1, 1, 32, 2560), ttnn.float32)
            for start in range(0, output_width, 32)
        ]

    def __call__(self, hidden_states):
        """Project 1..32 valid rows entirely on device, preserving BF16 output."""
        count = hidden_states.shape[-2]
        if tuple(hidden_states.shape) != (1, 1, count, 2560) or not 1 <= count <= 32:
            raise ValueError("Expected [1,1,1..32,2560] projection input")
        expanded = ttnn.reshape(ttnn.typecast(hidden_states, ttnn.float32), (1, count, 1, 2560))
        columns = []
        for weight in self.weights:
            products = ttnn.multiply(expanded, weight)
            tails = []
            width = 2560
            while width > 1:
                half = width // 2
                if width % 2:
                    tails.append(products[..., -1:])
                products = ttnn.add(products[..., :half], products[..., half : 2 * half])
                width = half
            for tail in tails:
                products = ttnn.add(products, tail)
            columns.append(ttnn.reshape(products, (1, 1, count, 32)))
        return ttnn.typecast(ttnn.concat(columns, dim=-1), ttnn.bfloat16)


class PreciseIndexerProjection(PreciseProjection):
    def __init__(self, loader, name="self_attn.indexer.index_qk_proj.weight"):
        super().__init__(loader, name, 640)


class PreciseRouterProjection(PreciseProjection):
    def __init__(self, loader, name="mlp.gate.weight"):
        """Return the target router's BF16 logits before its FP32 softmax/top-k."""
        super().__init__(loader, name, 512)
