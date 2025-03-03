# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_pix_art_alpha_text_projection import (
    ttnn_PixArtAlphaTextProjection,
)
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_time_step_embeddings import ttnn_TimestepEmbedding
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_time_steps import ttnn_Timesteps


class ttnn_CombinedTimestepTextProjEmbeddings:
    def __init__(self, embedding_dim, pooled_projection_dim, parameters):
        self.time_proj = ttnn_Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = ttnn_TimestepEmbedding(parameters.timestep_embedder)
        self.text_embedder = ttnn_PixArtAlphaTextProjection(parameters.text_embedder)

    def __call__(self, timestep, pooled_projection, device):
        timesteps_proj = self.time_proj(timestep, device)
        timesteps_emb = self.timestep_embedder(timesteps_proj, device)
        pooled_projections = self.text_embedder(pooled_projection, device)
        conditioning = ttnn.add(timesteps_emb, pooled_projections)
        return conditioning
