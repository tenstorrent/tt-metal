# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal scaffold — bounty tt-metal#49307 track)
# SPDX-License-Identifier: Apache-2.0
#
# CohereLayerNorm for Command-R (c4ai-command-r-v01).
#
# HF reference (transformers v4.39.3, models/cohere/modeling_cohere.py lines 78-94):
#   - fp32 internal compute: mean, variance, (x - mean) * rsqrt(var + eps)
#   - affine: * weight (fp32), optional + bias (Command-R v01 norms are weight-only)
#   - cast back to input dtype
#   - eps = config.layer_norm_eps = 1e-5
#
# This is a MEAN-CENTERING LayerNorm — NOT the RMSNorm used across tt_transformers.
# TTNN mapping: ttnn.layer_norm is a proven op; fp32-internal semantics must be
# matched for the P4 PCC >= 0.99 gate (tt-rd scripts/command-r/ harness).

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtCohereLayerNorm(LightweightModule):
    """Scaffold: LayerNorm (mean-centering) wrapper for Command-R norms.

    Used for: per-layer input_layernorm (x40) + final model norm (x1).
    NOTE: weight-only (no bias) for c4ai-command-r-v01 (attention_bias=false and the
    HF CohereLayerNorm defaults bias=None; checkpoint tensors are *.weight only).

    TODO(PCC): validate ttnn.layer_norm vs HF CohereLayerNorm fp32 compute at
    PCC >= 0.99 on captured CPU-reference activations before Stage-1 bring-up.
    TODO(P5): decide DistributedNorm strategy for TP=4 mesh (1,4) — RMSNorm path in
    tt_transformers uses DistributedNorm(RMSNorm(...)); LayerNorm equivalent wiring
    to be confirmed in the P5 survey (models/tt_transformers/tt/distributed_norm.py).
    """

    def __init__(
        self,
        device,
        dim,
        eps,
        state_dict=None,
        state_dict_prefix=None,
        weight_cache_path=None,
        weight_dtype=ttnn.bfloat16,
        weight_key="input_layernorm",
        tt_ccl=None,
    ):
        super().__init__()
        self.device = device
        self.dim = dim
        self.eps = eps
        self.tt_ccl = tt_ccl
        # TODO(P5): load state_dict[f"{state_dict_prefix}{weight_key}.weight"] via
        # ttnn.as_tensor (mirror models/common/rmsnorm.py RMSNorm.load_*) — deferred
        # until ModelArgs gains the cohere model_type (P5 survey).
        self.weight = None  # scaffold placeholder

    def forward(self, x: ttnn.Tensor, mode="decode", norm_config=None) -> ttnn.Tensor:
        # TODO(PCC): exact memory_config / program_config / fp32 compute_kernel_config
        # to match HF fp32-internal semantics; validated against CPU reference dumps.
        return ttnn.layer_norm(x, epsilon=self.eps, weight=self.weight)
