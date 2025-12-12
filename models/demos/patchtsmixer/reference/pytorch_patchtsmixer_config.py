from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class PatchTSMixerConfig:
    # Time series
    context_length: int = 32
    patch_length: int = 8
    num_input_channels: int = 1
    patch_stride: int = 8

    # Model
    d_model: int = 8
    expansion_factor: int = 2
    num_layers: int = 3
    dropout: float = 0.2
    mode: str = "common_channel"          # or "mix_channel"
    gated_attn: bool = True
    norm_eps: float = 1e-5

    # Scaling / misc
    scaling: Optional[Union[str, bool]] = "std"

    # Head
    prediction_length: int = 16
    head_dropout: float = 0.2

    def __post_init__(self):
        # same formula as HF config
        self.num_patches = (
            (max(self.context_length, self.patch_length) - self.patch_length)
            // self.patch_stride
            + 1
        )
