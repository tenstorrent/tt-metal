# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Expert selection, the TTNN form of reference.NomicRouter.

Routing is a discrete decision: an error large enough to reorder two near-tied experts does not
degrade a token's output, it sends the token through a different pair of 4.7M-parameter experts.
That is why this is the one path held at float32, and why the selection happens before the cast
rather than after.
"""

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import to_device, transpose_linear_weight


class TtNomicRouter(LightweightModule):
    """Softmax over all experts in fp32, top-k, then scatter back to a dense per-expert row.

    The top-k weights are deliberately NOT renormalized: moe_normalize_expert_weights is false
    in this checkpoint, so they sum to less than 1 and the MoE branch is attenuated relative to
    the residual. Dividing by the top-k sum, which Mixtral and Switch both do and which is the
    reflex to copy, still scores about 0.993 PCC.

    fp32 is carried as far as ttnn.scatter allows. scatter rejects float32 in both layouts, so a
    cast is unavoidable, but ttnn.topk accepts fp32 and its uint32 index feeds scatter unchanged;
    only the destination and the two selected weights are cast, after the selection. Casting the
    probabilities first instead reroutes 0.34% to 0.59% of tokens, because bfloat16 quantizes an
    8-wide row coarsely enough to turn near-ties into exact ties.

    The softmax needs both halves of the port's compute kernel config. HiFi4 alone measures
    5.4e-3 to 6.8e-3 max-abs against a 5e-3 budget; with fp32 destination accumulation it is
    1.4e-3 to 1.9e-3.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix):
        super().__init__()
        self.tt_config = tt_config
        self.top_k = config.moe_top_k

        # The one bias-free projection in the model, and the only weight kept in fp32.
        self.weight = to_device(
            transpose_linear_weight(state_dict[f"{state_dict_prefix}layer.weight"]),
            device,
            dtype=tt_config.router_dtype,
        )

    def select(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Score every expert and take the top-k, all in float32.

        Split out of forward so the per-module test can assert on the selection itself: index
        agreement against torch is the gate, and PCC on the dense output cannot express it.

        Args:
            x: (1, 1, T, H) flat token activations.

        Returns:
            tuple: probabilities (1, 1, T, E) fp32, values (1, 1, T, K) fp32 unrenormalized, and
            indices (1, 1, T, K) uint32.
        """
        logits = ttnn.linear(
            ttnn.typecast(x, self.tt_config.router_dtype),
            self.weight,
            compute_kernel_config=self.tt_config.compute_kernel_config,
        )
        probabilities = ttnn.softmax(logits, dim=-1, compute_kernel_config=self.tt_config.compute_kernel_config)
        ttnn.deallocate(logits)

        values, indices = ttnn.topk(probabilities, k=self.top_k, dim=-1)
        return probabilities, values, indices

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Produce the dense routing weights the expert gate multiply consumes.

        Args:
            x: (1, 1, T, H) flat token activations.

        Returns:
            ttnn.Tensor: (1, 1, T, E) in the activation dtype, carrying the routed weight at the
            top-k positions and zero everywhere else.
        """
        probabilities, values, indices = self.select(x)

        # ttnn.zeros rather than zeros_like(typecast(probabilities)): the destination only needs
        # the shape, and the cast would convert a (1, 1, T, E) tensor to throw its values away.
        dense = ttnn.scatter(
            ttnn.zeros(
                probabilities.shape,
                dtype=self.tt_config.activation_dtype,
                layout=self.tt_config.layout,
                device=probabilities.device(),
            ),
            dim=-1,
            index=indices,
            src=ttnn.typecast(values, self.tt_config.activation_dtype),
        )
        for tensor in (probabilities, values, indices):
            ttnn.deallocate(tensor)
        return dense
