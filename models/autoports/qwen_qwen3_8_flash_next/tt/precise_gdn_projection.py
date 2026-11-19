# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""BF16-resident, FP32-arithmetic GDN QKV/output projections.

Exact-input controls show native HiFi4 projections disagree with HF on about
15% of BF16 outputs, including with FP32 packing or K32/K128 partial sums.
The SFPU product/addition tree restores >99.96% equality and repairs the
request3 decode failure with only QKV and output projections replaced.
See request3_device_projection_precision.json for the controlled experiment.

Weights remain BF16 in 256-output parent blocks; no second matrix exists.
Single-token calls use all256 outputs per block to stay below the shared
trace command-size ceiling. Prefill slices32 outputs on device to bound
workspace. Both use the same K reduction tree.
Each forward casts one block to FP32 on device, multiplies and reduces with
balanced SFPU additions, and casts once at the HF BF16 output boundary.
Input/output are TILE [1,1,Q,K/N], Q>=1; larger calls use token blocks<=32.
All loop bounds are static shapes;
there are no host tensor conversions or data-dependent host branches.
"""

import ttnn


class PreciseGDNProjection:
    def __init__(self, loader, name):
        weight = loader.state_dict[name]
        self.output_width, self.input_width = weight.shape
        if (self.output_width, self.input_width) not in ((10240, 2560), (2560, 6144)):
            raise ValueError("Expected target GDN QKV [10240,2560] or output [2560,6144] weight")
        self.parent_block = 256
        self.weights = [
            loader.tensor(weight[start : start + self.parent_block].reshape(1, 1, self.parent_block, self.input_width))
            for start in range(0, self.output_width, self.parent_block)
        ]

    def __call__(self, value):
        count = value.shape[-2]
        if tuple(value.shape) != (1, 1, count, self.input_width) or count < 1:
            raise ValueError(f"Expected device GDN input [1,1,Q>=1,{self.input_width}]")
        if count > 32:
            return ttnn.concat(
                [self._project_block(value[:, :, start : min(start + 32, count), :]) for start in range(0, count, 32)],
                dim=2,
            )
        return self._project_block(value)

    def _project_block(self, value):
        count = value.shape[-2]
        block = 256 if count == 1 else 32
        expanded = ttnn.reshape(ttnn.typecast(value, ttnn.float32), (1, count, 1, self.input_width))
        columns = []
        for parent in self.weights:
            for offset in range(0, self.parent_block, block):
                weight = parent if block == self.parent_block else parent[:, :, offset : offset + block, :]
                products = ttnn.multiply(expanded, ttnn.typecast(weight, ttnn.float32))
                width, tails = self.input_width, []
                while width > 1:
                    half = width // 2
                    if width % 2:
                        tails.append(products[..., -1:])
                    left, right = products[..., :half], products[..., half : 2 * half]
                    del products
                    products = ttnn.add(left, right)
                    del left, right
                    width = half
                for tail in tails:
                    products = ttnn.add(products, tail)
                columns.append(ttnn.reshape(products, (1, 1, count, block)))
        return ttnn.typecast(ttnn.concat(columns, dim=-1), ttnn.bfloat16)
