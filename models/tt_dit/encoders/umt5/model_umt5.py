from ..t5.model_t5 import T5Config, T5Encoder

UMT5Encoder = T5Encoder


class UMT5Config(T5Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_relative_position_bias = [True] * (self.num_hidden_layers)
