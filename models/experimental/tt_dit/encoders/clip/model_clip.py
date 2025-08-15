# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from ...utils.substate import substate
from ...layers.embeddings import TextEmbedding
from ...models.transformers.transformer_encoders import CLIPTextEncoderTransformer


# TODO: add module docstring. finalize docstrings/comments for classes and methods


class CLIPTextModel:
    def __init__(
        self,
        config,
        mesh_device=None,
        with_projection=False,
        init=False,
        parallel_manager=None,
        ccl_manager=None,
        parallel_config=None,
    ):
        self.config = config
        self.with_projection = with_projection
        self.mesh_device = mesh_device
        self.parallel_manager = parallel_manager
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.embeddings = TextEmbedding(
            config=config,
            mesh_device=mesh_device,
            init=init,
            with_projection=with_projection,
        )

        self.encoder = CLIPTextEncoderTransformer(
            config=config,
            mesh_device=mesh_device,
            parallel_manager=self.parallel_manager,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
        )

    def load_state_dict(self, state_dict):
        """Load weights from HuggingFace state dictionary and distribute to components"""

        text_model_state = substate(state_dict, "text_model")

        embeddings_state = substate(text_model_state, "embeddings")
        if embeddings_state:
            self.embeddings.load_state_dict(embeddings_state)

        encoder_state = substate(text_model_state, "encoder")
        if encoder_state:
            self.encoder.load_state_dict(
                encoder_state,
                ccl_manager=self.ccl_manager,
                parallel_config=self.parallel_config,
            )


class hidden_states:
    hidden_states: list[ttnn.Tensor]

    def __getitem__(self, index):
        return self.hidden_states[index]
