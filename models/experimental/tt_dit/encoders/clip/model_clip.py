# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from ...layers.embeddings import TextEmbedding
from ...models.transformers.transformer_encoders import CLIPTextEncoderTransformer


# TODO: add module docstring. finalize docstrings/comments for classes and methods


class CLIPTextModel:
    def __init__(self, config, mesh_device=None, with_projection=False, init=False, parallel_manager=None):
        self.config = config
        self.with_projection = with_projection
        self.mesh_device = mesh_device
        self.parallel_manager = parallel_manager

        self.embeddings = TextEmbedding(
            config=config,
            mesh_device=mesh_device,
            init=init,
            with_projection=with_projection,
        )

        self.encoder = CLIPTextEncoderTransformer(
            config=config,
            mesh_device=mesh_device,
            parallel_manager=parallel_manager,
        )


class hidden_states:
    hidden_states: list[ttnn.Tensor]

    def __getitem__(self, index):
        return self.hidden_states[index]
