# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from models.common.lightweightmodule import LightweightModule
from models.experimental.devstarl2_small.tt.tt_pixtral_attention_layer import TtPixtralAttentionLayer


class TtPixtralTransformer(LightweightModule):
    """TT stack matching HF ``PixtralTransformer``: repeated ``PixtralAttentionLayer`` blocks."""

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        configuration,
        weight_cache_path,
        dtype,
        n_layers: int | None = None,
        vision_prefix: str = "vision_tower.transformer.layers.",
    ):
        super().__init__()
        self._mesh_device = mesh_device
        self._tt_ccl = tt_ccl
        self._state_dict = state_dict
        self._configuration = configuration
        self._weight_cache_path = weight_cache_path
        self._dtype = dtype
        self._vision_prefix = vision_prefix
        self._n_layers = int(n_layers if n_layers is not None else configuration.vision_n_layers)
        self._layers: dict[int, TtPixtralAttentionLayer] = {}

    def _get_layer(self, layer_index: int) -> TtPixtralAttentionLayer:
        layer = self._layers.get(layer_index)
        if layer is None:
            layer = TtPixtralAttentionLayer(
                mesh_device=self._mesh_device,
                tt_ccl=self._tt_ccl,
                state_dict=self._state_dict,
                state_dict_prefix=f"{self._vision_prefix}{layer_index}.",
                weight_cache_path=self._weight_cache_path,
                dtype=self._dtype,
                configuration=self._configuration,
            )
            self._layers[layer_index] = layer
        return layer

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        h = hidden_states
        for layer_index in range(self._n_layers):
            h = self._get_layer(layer_index)(h, attention_mask=attention_mask, position_embeddings=position_embeddings)
        return h


__all__ = ["TtPixtralTransformer"]
