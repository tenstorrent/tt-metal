from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_upsample import UpsampleBlock
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_resnet import ResnetBlock


class UpDecoderBlock:
    def __init__(
        self,
        torch_upblock,
        device,
        in_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        resnet_norm_blocks,
        resnet_conv1_channel_split_factors,
        resnet_conv2_channel_split_factors,
        upsample_conv_channel_split_factors,
    ):
        breakpoint()
        print("Initializing UpDecoderBlock")
        self.resnets = []
        for i in range(2):
            self.resnets.append(
                ResnetBlock(
                    torch_upblock.resnets[i],
                    device,
                    in_channels if i == 0 else out_channels,
                    input_height,
                    input_width,
                    out_channels,
                    resnet_norm_blocks[i][0],
                    resnet_norm_blocks[i][1],
                    resnet_conv1_channel_split_factors[i],
                    resnet_conv2_channel_split_factors[i],
                )
            )

        self.upsample = None
        if torch_upblock.upsamplers:
            self.upsample = UpsampleBlock(
                torch_upblock.upsamplers[0],
                device,
                out_channels,
                out_channels,
                output_height,
                output_width,
                upsample_conv_channel_split_factors[0],
                upsample_conv_channel_split_factors[1],
            )

        print("UpDecoderBlock initialized")

    def __call__(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.upsample:
            hidden_states = self.upsample(hidden_states)

        return hidden_states
