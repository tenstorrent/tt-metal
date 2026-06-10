"""Phase 0 harness: static config + reference weight access for the TTNN port.

`XttsGptConfig` captures the GPT2 backbone hyper-parameters read off the
reference architecture (see xtts_model_arch dump). `load_reference_state_dict`
returns the upstream model's parameters so the TTNN layers can pull their
weights by name — start by porting one module, slice its weights here.
"""

from dataclasses import dataclass

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


@dataclass(frozen=True)
class XttsGptConfig:
    # GPT2 backbone (the Phase 1 target)
    hidden_size: int = 1024
    num_layers: int = 30
    num_heads: int = 16  # 1024 / 64
    head_dim: int = 64
    ffn_size: int = 4096
    layer_norm_eps: float = 1e-5

    # vocab / embeddings
    text_vocab_size: int = 6681
    mel_vocab_size: int = 1026
    max_text_pos: int = 404
    max_mel_pos: int = 608

    # NOTE: GPT2 uses Conv1D (weight stored [nx, nf], i.e. transposed vs nn.Linear).
    # The TTNN port must transpose c_attn/c_proj/c_fc weights at load time.
    gpt2_uses_conv1d: bool = True


def load_reference_state_dict(device: str = "cpu"):
    """Load the upstream XTTS-v2 and return (model, state_dict).

    Heavy (downloads ~1.9 GB on first call). Kept out of the dataclass so the
    config can be imported cheaply.
    """
    from TTS.api import TTS

    tts = TTS(MODEL_NAME).to(device)
    model = tts.synthesizer.tts_model
    assert model is not None, "tts.synthesizer.tts_model is None — model failed to load"
    model.eval()
    return model, model.state_dict()
