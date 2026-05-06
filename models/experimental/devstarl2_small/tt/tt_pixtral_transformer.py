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
        nl = int(n_layers if n_layers is not None else configuration.vision_n_layers)
        self.layers = [
            TtPixtralAttentionLayer(
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                state_dict=state_dict,
                state_dict_prefix=f"{vision_prefix}{i}.",
                weight_cache_path=weight_cache_path,
                dtype=dtype,
                configuration=configuration,
            )
            for i in range(nl)
        ]

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        h = hidden_states
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask, position_embeddings=position_embeddings)
        return h


__all__ = ["TtPixtralTransformer"]
