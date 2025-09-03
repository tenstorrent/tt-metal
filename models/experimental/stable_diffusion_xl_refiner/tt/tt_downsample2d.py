from models.experimental.stable_diffusion_xl_refiner.tt.components.convolution_layer import ConvolutionLayer
from models.experimental.stable_diffusion_xl_refiner.tt.tt_config import get_downsample_config


class TtDownsample2D:
    def __init__(
        self,
        device,
        state_dict,
        module_path,
    ):
        super().__init__()

        self.device = device
        self.module_path = module_path

        # Configuration setup
        self.block_config = get_downsample_config(module_path)

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

    def forward(self, hidden_states, input_shape):
        B, C, H, W = input_shape

        hidden_states, [C, H, W] = self.conv_layer.apply(hidden_states, B, C, H, W)

        return hidden_states, [C, H, W]
