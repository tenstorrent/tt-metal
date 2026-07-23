from models.demos.vision.detection.rtdetr.tt.backbone import TtRTDetrConvEncoder, TtRTDetrResNetConvLayer


class TtRTDetrModel:
    def __init__(self, config, parameters, device, dtype):
        self.backbone = TtRTDetrConvEncoder(config, parameters=parameters.backbone, device=device, dtype=dtype)

        self.encoder_in_channels = config.encoder_in_channels
        self.encoder_hidden_dim = config.encoder_hidden_dim

        self.encoder_input_proj = [
            TtRTDetrResNetConvLayer(
                config=config,
                parameters=parameters.encoder_input_proj[index],
                device=device,
                dtype=dtype,
                in_channels=in_channels,
                out_channels=self.encoder_hidden_dim,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation="identity",
            )
            for index, in_channels in enumerate(self.encoder_in_channels)
        ]

    def _encoder_input_projection(self, features, batch_size):
        projected_features = []

        for feature, projection in zip(features, self.encoder_input_proj):
            x, height, width = feature

            x, height, width = projection(
                x,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
            )

            projected_features.append((x, height, width))

        return projected_features

    def __call__(self, pixel_values):
        batch_size = pixel_values.shape[0]

        features = self.backbone(pixel_values)
        projected_features = self._encoder_input_projection(features, batch_size)

        return projected_features
