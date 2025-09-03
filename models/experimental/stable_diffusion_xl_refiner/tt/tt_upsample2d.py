import ttnn
from models.experimental.stable_diffusion_xl_refiner.tt.components.convolution_layer import ConvolutionLayer
from models.experimental.stable_diffusion_xl_refiner.tt.tt_config import get_upsample_config


class TtUpsample2D:
    def __init__(
        self,
        device,
        state_dict,
        module_path,
    ):
        super().__init__()

        self.device = device
        self.module_path = module_path

        self.scale_factor = 2

        # Configuration setup
        self.block_config = get_upsample_config(module_path)

        # Initialize components
        self._initialize_components(state_dict)

    def _initialize_components(self, state_dict):
        # Order of layers:
        # 1. conv_layer

        self.conv_layer = ConvolutionLayer(
            self.device,
            state_dict[f"{self.module_path}.conv.weight"],
            state_dict[f"{self.module_path}.conv.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0),
            self.block_config.conv,
        )

    def forward(self, input_tensor):
        hidden_states = input_tensor

        # TILE_LAYOUT Fails on one of the cases bcs 16x16 is not tile alligned
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        print(f"Shape before upsample: {hidden_states.shape}")
        hidden_states = ttnn.upsample(hidden_states, (self.scale_factor, self.scale_factor))
        print(f"Shape after upsample: {hidden_states.shape}")
        B, H, W, C = list(hidden_states.shape)
        print(f"B, C, H, W after upsample: {B, C, H, W}")
        hidden_states, [C, H, W] = self.conv_layer.apply(hidden_states, B, C, H, W)
        print(f"B, C, H, W after conv: {B, C, H, W}")
        return hidden_states, [C, H, W]
