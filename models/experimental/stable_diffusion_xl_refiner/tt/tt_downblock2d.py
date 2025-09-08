from models.experimental.stable_diffusion_xl_refiner.tt.tt_config import get_downblock_config
from models.experimental.stable_diffusion_xl_refiner.tt.tt_resnetblock2d import TtResnetBlock2D
from models.experimental.stable_diffusion_xl_refiner.tt.tt_downsample2d import TtDownsample2D


class TtDownBlock2D:
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
        self.block_config = get_downblock_config(module_path)

        self._initialize_components(state_dict)

    def _initialize_components(self, state_dict):
        # Order of layers:
        # 1. resnets (list of three)
        # 2. downsample (optional)

        self.resnets = []
        for i in range(self.block_config.num_resnets):
            resnet_module_path = f"{self.module_path}.resnets.{i}"
            # Maybe will use in a refactor, to pass as an argument to TtResnetBlock2D
            # resnet_config = self.block_config.resnet_configs.get(i)

            self.resnets.append(
                TtResnetBlock2D(
                    device=self.device,
                    state_dict=state_dict,
                    module_path=resnet_module_path,
                )
            )

        if self.block_config.has_downsample:
            self.downsample = TtDownsample2D(
                device=self.device,
                state_dict=state_dict,
                module_path=f"{self.module_path}.downsamplers.0",
                # config=self.block_config.downsample_config
            )
        else:
            self.downsample = None

    def forward(self, hidden_states, input_shape, temb):
        B, C, H, W = input_shape
        residuals = ()

        for resnet in self.resnets:
            hidden_states, [C, H, W] = resnet.forward(hidden_states, temb, [B, C, H, W])
            residuals = residuals + (hidden_states,)

        if self.downsample is not None:
            hidden_states, [C, H, W] = self.downsample.forward(hidden_states, [B, C, H, W])
            residuals = residuals + (hidden_states,)

        return hidden_states, [C, H, W], residuals
