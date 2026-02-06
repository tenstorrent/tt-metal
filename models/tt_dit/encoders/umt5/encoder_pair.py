from transformers.models.umt5.modeling_umt5 import UMT5EncoderModel

from ..t5.encoder_pair import T5TokenizerEncoderPair


class UMT5TokenizerEncoderPair(T5TokenizerEncoderPair):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_torch_model(self, checkpoint: str) -> UMT5EncoderModel:
        return UMT5EncoderModel.from_pretrained(checkpoint)
