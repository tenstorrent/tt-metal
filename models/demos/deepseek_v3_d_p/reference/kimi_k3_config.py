# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Kimi K3 routed-expert dimensions.

Scoped deliberately to the MoE routed-expert path (issue #51351) -- the only part
of K3 that is brought up so far. Unlike the sibling per-model configs this is NOT
a full model config; add the attention / RoPE / vocab constants alongside the code
that first needs them, so nothing here is guessed ahead of use.

K3 down-projects the token embedding before the routed experts, so the expert FFN
sees EMB_SIZE -> MOE_LATENT_SIZE. The fused routed-expert op consumes that already
projected input, i.e. its K axis is MOE_LATENT_SIZE (not EMB_SIZE) and its weights
are shaped against the same.
"""


class KimiK3Config:
    """Kimi K3 MoE routed-expert dimensions."""

    EMB_SIZE = 7168  # model embedding dimension (pre-projection)
    MOE_LATENT_SIZE = 3584  # routed-expert input dimension (post down-projection)
    MOE_INTERMEDIATE_SIZE = 3072  # routed-expert FFN hidden dimension

    # The SiTU-GLU betas deliberately do NOT live here. The device kernel bakes them from
    # SituGluConfigKimi (situ_glu_sfpu.h) and the torch reference reads SITU_BETA_GATE /
    # SITU_BETA_UP from reference/tt/moe/expert.py; a third copy on this config would be
    # read by nothing, so editing it would silently fail to change the activation.
