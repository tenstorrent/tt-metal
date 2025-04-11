import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_attention import Attention
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_resnet import ResnetBlock


class MidBlock:
    def __init__(
        self,
        torch_midblock,
        device,
        in_channels,
        input_height,
        input_width,
    ):
        self.resnets = []
        self.resnets.append(
            ResnetBlock(
                torch_midblock.resnets[0],
                device,
                in_channels,
                input_height,
                input_width,
                in_channels,
                1,
                1,
                (1, 1),
                (1, 1),
            )
        )
        self.attention = Attention(
            torch_midblock.attentions[0],
            device,
            in_channels,
        )

        self.resnets.append(
            ResnetBlock(
                torch_midblock.resnets[1],
                device,
                in_channels,
                input_height,
                input_width,
                in_channels,
                1,
                1,
                (1, 1),
                (1, 1),
            )
        )

    def __call__(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)
        hidden_states = ttnn.reshape(
            hidden_states, [1, 1, hidden_states.shape[1] * hidden_states.shape[2], hidden_states.shape[3]]
        )
        hidden_states = self.attention(hidden_states)
        hidden_states = self.resnets[1](hidden_states)
        return hidden_states
