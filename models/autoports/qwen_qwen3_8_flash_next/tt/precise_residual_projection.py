# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP32 SFPU projections for the target gated residual's BF16 boundaries.

Supported real checkpoint matrices are down [320,10240], up [10240,320], and
optional injection [4,10240]. The exact-operand device control restores MLP
mixed-input BF16 equality to100% for requests16/23 and99.987% for29, fixing
router-sensitive failures that persist with exact post-attention input.
See doc/functional_decoder/gr_projection_intermediates.json for evidence.

Setup expands weights to FP32. Forward accepts device BF16 TILE [1,1,Q,K],
1<=Q<=32, and returns device BF16 TILE [1,1,Q,N]. FP32 products and balanced
SFPU additions preserve arithmetic before the single HF BF16 output boundary.
No FPU reduction or host tensor operation occurs inside a pass. Warm the
static physical signatures before capture. Existing caller-owned state and
trace input/output lifetime rules continue to apply.

The validated minimal repair uses down/up; accurate injection is available
for explicit controls but changed none of the tested full-path results.
"""

import ttnn


class PreciseResidualProjection:
    def __init__(self, loader, name):
        """Setup only: load one actual GR checkpoint matrix into bounded blocks."""
        weight = loader.state_dict[name]
        self.output_width, self.input_width = weight.shape
        if (self.output_width, self.input_width) not in ((320, 10240), (10240, 320), (4, 10240)):
            raise ValueError("Expected a target GR down, up, or injection matrix")
        self.block = 256 if self.input_width == 320 else 32
        self.weights = []
        for start in range(0, self.output_width, self.block):
            part = weight[start : start + self.block]
            if part.shape[0] < self.block:
                # Only the optional four-output injection matrix requires
                # setup padding. Its padded rows never become logical output.
                padded = part.new_zeros((self.block, self.input_width))
                padded[: part.shape[0]] = part
                part = padded
            self.weights.append(loader.tensor(part.reshape(1, 1, self.block, self.input_width), ttnn.float32))

    def __call__(self, value):
        count = value.shape[-2]
        if tuple(value.shape) != (1, 1, count, self.input_width) or not 1 <= count <= 32:
            raise ValueError(f"Expected device GR input [1,1,1..32,{self.input_width}]")
        expanded = ttnn.reshape(ttnn.typecast(value, ttnn.float32), (1, count, 1, self.input_width))
        columns = []
        for weight in self.weights:
            products = ttnn.multiply(expanded, weight)
            width, tails = self.input_width, []
            while width > 1:
                half = width // 2
                if width % 2:
                    tails.append(products[..., -1:])
                left, right = products[..., :half], products[..., half : 2 * half]
                # Slices own their buffers. Drop the consumed full product
                # before allocating the next reduction level (40 MiB at Q32).
                del products
                products = ttnn.add(left, right)
                del left, right
                width = half
            for tail in tails:
                products = ttnn.add(products, tail)
            columns.append(ttnn.reshape(products, (1, 1, count, self.block)))
        result = ttnn.concat(columns, dim=-1) if len(columns) > 1 else columns[0]
        return ttnn.typecast(result[..., : self.output_width], ttnn.bfloat16)
