# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the NLLB-style feed-forward block used across
SeamlessM4T-v2 (text encoder / decoder layers and T2U encoder / decoder layers).

The PyTorch reference is
`models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::seamless_ffn_forward`,
which computes::

    h = Linear(hidden, ffn_dim)(x) + b1
    h = ReLU(h)
    y = Linear(ffn_dim, hidden)(h) + b2

For the ``-large`` configuration: ``hidden = 1024``, ``ffn_dim = 8192``,
``activation = relu`` and biases enabled on both projections.

This block follows the structure of the BERT MLP at
``models/demos/bert/tt/ttnn_optimized_bert.py`` (two ``ttnn.linear`` calls
with an activation between them), simplified to the unsharded interleaved
DRAM path matched to the saved golden tensor.
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule


class SeamlessFfn(LightweightModule):
    """Two-projection feed-forward block with a ReLU between projections.

    Args:
        device: ttnn device or mesh device.
        fc1_weight: torch.Tensor of shape ``(ffn_dim, hidden)`` -- first
            projection weight in PyTorch convention.
        fc1_bias:   torch.Tensor of shape ``(ffn_dim,)``.
        fc2_weight: torch.Tensor of shape ``(hidden, ffn_dim)`` -- second
            projection weight in PyTorch convention.
        fc2_bias:   torch.Tensor of shape ``(hidden,)``.
        weight_dtype: storage dtype for weights / biases on device.
        weight_memory_config: where to store weights / biases (default DRAM).
    """

    def __init__(
        self,
        device,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device

        ffn_dim, hidden = fc1_weight.shape
        self.hidden = int(hidden)
        self.ffn_dim = int(ffn_dim)

        # Perf note (tracy on AR-loop traced text_decoder_layer): the fc2
        # matmul (K=ffn_dim=8192 -> N=hidden=1024, M=1) is the single
        # hottest op at ~117 us per call (~25% of one decoder-layer
        # replay) and is DRAM-bandwidth bound (HiFi2 had zero effect).
        # Storing the 8 MB fc2 weight in bfloat8_b halves the DRAM read
        # bandwidth on the K-major reduction. The matmul still accumulates
        # in fp32_dest and outputs bfloat16, so PCC stays within the 0.99
        # band (verified per-block + e2e). fc1 weight stays bf16 because
        # the fc1 N expansion fans out to 8192 outputs and is much less
        # sensitive to weight quantisation; gain there is small (DRAM
        # read is interleaved across 128 cores) and not worth the dtype
        # change.
        fc2_weight_dtype = ttnn.bfloat8_b if weight_dtype == ttnn.bfloat16 else weight_dtype

        # ttnn.linear expects weights laid out as ``(in_features, out_features)``,
        # i.e. the transpose of the PyTorch ``nn.Linear.weight``.
        self.fc1_weight = ttnn.from_torch(
            fc1_weight.transpose(0, 1).contiguous(),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )
        self.fc1_bias = ttnn.from_torch(
            fc1_bias.reshape(1, -1),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )
        self.fc2_weight = ttnn.from_torch(
            fc2_weight.transpose(0, 1).contiguous(),
            device=device,
            dtype=fc2_weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )
        self.fc2_bias = ttnn.from_torch(
            fc2_bias.reshape(1, -1),
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )

        # packer_l1_acc=True is the SKILL-recommended default; lets the matmul
        # kernel pack into its L1 accumulator and reduce dst-register pressure
        # on the wide-K (8192) fc2 reduction. Verified PCC-neutral on the FFN
        # standalone (0.9999) and on the text_decoder_layer composite.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Run the FFN forward pass on a TILE_LAYOUT ttnn tensor.

        ``x`` has shape ``[..., hidden]``. Output has the same shape.
        """
        hidden = ttnn.linear(
            x,
            self.fc1_weight,
            bias=self.fc1_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        hidden = ttnn.relu(hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.linear(
            hidden,
            self.fc2_weight,
            bias=self.fc2_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(hidden)
        return output
