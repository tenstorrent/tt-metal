import torch
from typing import List

from python_api_testing.models.EfficientNet.tt.efficientnet_mbconv import (
    TtEfficientnetMbConv,
    MBConvConfig,
)

from python_api_testing.models.EfficientNet.tt.efficientnet_fused_mbconv import (
    TtEfficientnetFusedMBConv,
    FusedMBConvConfig,
)


class TtEfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[MBConvConfig]"
            )

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                )

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = (
            last_channel if last_channel is not None else 4 * lastconv_input_channels
        )
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _efficientnet(
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
    dropout: float,
    last_channel: Optional[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> EfficientNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = EfficientNet(
        inverted_residual_setting, dropout, last_channel=last_channel, **kwargs
    )

    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)
        )

    return model


def _efficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
    if arch.startswith("efficientnet_b"):
        bneck_conf = partial(
            MBConvConfig,
            width_mult=kwargs.pop("width_mult"),
            depth_mult=kwargs.pop("depth_mult"),
        )
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_m"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_l"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel
