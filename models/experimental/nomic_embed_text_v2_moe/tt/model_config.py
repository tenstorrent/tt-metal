# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-side settings every TTNN op in this port runs under.

One compute kernel config for everything: HiFi4 with fp32 destination accumulation. Both
halves were measured on a p300c at T=1024, against the router's 5e-3 max-abs budget:

  math fidelity        softmax max-abs is 2.7e-2 to 3.0e-2 at the stock config, 5.4e-3 to
                       6.8e-3 with HiFi4 alone, 1.4e-3 to 1.9e-3 with both. The router's
                       budget is 5e-3, so HiFi4 alone misses it, but not by much.
  destination accum    a bf16 accumulator costs 6x to 15x max-abs on every matmul. fc2, a
                       3072-deep reduction, degrades from 1.35e-1 to 2.08e+00; the router's
                       768-deep matmul reroutes 28 tokens in 1024 rather than 1.

fp32_dest_acc_en halves the destination register budget, trading throughput for accuracy.
Phase 1 is gated on correctness; Phase 2 can lower it per op group once there is a throughput
number to weigh against. No sharding, no program configs, everything DRAM-interleaved.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn

ACTIVATION_DTYPE = ttnn.bfloat16
WEIGHT_DTYPE = ttnn.bfloat16

# The one path kept at higher precision: the router's softmax feeds a top-2 selection, so a
# near-tie decided by rounding changes which experts a token visits. Rounding the probabilities
# to bfloat16 before the topk reroutes 0.34% to 0.59% of tokens. ttnn.scatter rejects float32 so a cast
# is unavoidable, but ttnn.topk takes fp32 and its uint32 index feeds the scatter unchanged, so
# only the two selected weights are cast, after the selection.
ROUTER_DTYPE = ttnn.float32

LAYOUT = ttnn.TILE_LAYOUT
MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG


@dataclass(frozen=True)
class TtModelConfig:
    """The settings that depend on the device, plus the dtypes and layout they go with.

    Model dimensions are not here; they come from the vendored config.json via
    reference.configuration_nomic_moe, the one source both sides read.
    """

    core_grid: ttnn.CoreCoord
    compute_kernel_config: ttnn.DeviceComputeKernelConfig

    activation_dtype: ttnn.DataType = ACTIVATION_DTYPE
    weight_dtype: ttnn.DataType = WEIGHT_DTYPE
    router_dtype: ttnn.DataType = ROUTER_DTYPE
    layout: ttnn.Layout = LAYOUT

    @classmethod
    def from_device(cls, device) -> "TtModelConfig":
        """Build the config from an open ttnn device or single-device mesh.

        The grid is queried, never hardcoded: this p300c reports 11x10, not the (8, 10) that
        models/tt_transformers/tt/model_config.py implies.

        math_approx_mode is off because it selects cheaper SFPU polynomials.
        """
        return cls(
            core_grid=device.compute_with_storage_grid_size(),
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
