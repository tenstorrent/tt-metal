# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 configuration.

Reuses PI0's GemmaConfig / SigLIPConfig / SuffixConfig / PrefixConfig /
PaliGemmaConfig / DenoiseConfig. Only PI0ModelConfig is overridden to flip
pi0.5 defaults (pi05=True, larger max_token_len, discrete state).
"""

from dataclasses import dataclass, field

from models.experimental.pi0.common.configs import (
    GemmaConfig,
    SigLIPConfig,
    PI0ModelConfig,
)


@dataclass
class Pi0_5ModelConfig(PI0ModelConfig):
    """
    PI0.5 model configuration.

    Differences vs PI0:
      - pi05 = True (drives adaRMS path in suffix + expert)
      - max_token_len = 200 (pi0.5 default in openpi)
      - discrete_state_input = True (state encoded as language tokens)
    """

    pi05: bool = True
    max_token_len: int = 200
    discrete_state_input: bool = True

    vlm_config: GemmaConfig = field(default_factory=GemmaConfig.gemma_2b)
    expert_config: GemmaConfig = field(default_factory=GemmaConfig.gemma_300m)
    siglip_config: SigLIPConfig = field(default_factory=SigLIPConfig)

    def __post_init__(self):
        super().__post_init__()
