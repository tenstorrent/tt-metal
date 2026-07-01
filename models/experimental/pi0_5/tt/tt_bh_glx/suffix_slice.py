# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Replicated suffix MLP on the mesh.

The suffix weights (~8 MB) are replicated on every chip, which avoids a
per-step host-bounce of adarms_cond. SuffixSlice wraps the replicated
Pi0_5SuffixEmbeddingTTNN; the 1×8 pipeline constructs one on its mesh.
"""

from __future__ import annotations

from typing import Dict

import torch

from models.experimental.pi0_5.common.configs import SuffixConfig
from models.experimental.pi0_5.tt.ttnn_suffix import (
    Pi0_5SuffixEmbeddingTTNN,
    convert_pi0_5_suffix_weights_to_ttnn,
)


class SuffixSlice:
    """One copy of the pi0.5 suffix MLP on a single-chip submesh.

    Exposes embed_actions(noisy_actions) and embed_adarms_cond(timestep)
    from the underlying Pi0_5SuffixEmbeddingTTNN. The internal embedding
    object owns its weights on this submesh.
    """

    def __init__(self, suffix_config: SuffixConfig, pi0_projections: Dict[str, torch.Tensor], submesh):
        self.config = suffix_config
        self.submesh = submesh
        ttnn_weights = convert_pi0_5_suffix_weights_to_ttnn(pi0_projections, submesh)
        self.suffix = Pi0_5SuffixEmbeddingTTNN(suffix_config, ttnn_weights, submesh)

    def embed_actions(self, noisy_actions):
        return self.suffix.embed_actions(noisy_actions)

    def embed_adarms_cond(self, timestep):
        return self.suffix.embed_adarms_cond(timestep)

    def project_output(self, expert_hidden):
        """action_out_proj: (B, action_horizon, expert_width) -> (B, action_horizon, action_dim)."""
        return self.suffix.project_output(expert_hidden)
