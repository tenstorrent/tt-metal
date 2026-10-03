"""Build the S2 Pro codec (modded DAC) exactly as fish-speech's hydra config `modded_dac_vq.yaml` does,
without hydra/omegaconf. The numbers below ARE that yaml; `check_yaml()` asserts the vendored copy agrees."""
from __future__ import annotations

import functools
from pathlib import Path

import torch

from models.autoports.fishaudio_s2_pro.reference.modded_dac import DAC, ModelArgs, WindowLimitedTransformer
from models.autoports.fishaudio_s2_pro.reference.rvq import DownsampleResidualVectorQuantize

TRANSFORMER_CFG = dict(
    block_size=2048,
    n_layer=8,
    n_head=16,
    dim=1024,
    intermediate_size=3072,
    n_local_heads=-1,
    head_dim=64,
    rope_base=10000,
    norm_eps=1e-5,
    dropout_rate=0.1,
    attn_dropout_rate=0.1,
    channels_first=True,
)
GENERAL_CFG = dict(
    block_size=8192,
    n_local_heads=-1,
    head_dim=64,
    rope_base=10000,
    norm_eps=1e-5,
    dropout_rate=0.1,
    attn_dropout_rate=0.1,
    channels_first=True,
)
DAC_CFG = dict(
    sample_rate=44100,
    encoder_dim=64,
    encoder_rates=[2, 4, 8, 8],
    decoder_dim=1536,
    decoder_rates=[8, 8, 4, 2],
    encoder_transformer_layers=[0, 0, 0, 4],
    decoder_transformer_layers=[4, 0, 0, 0],
)
QUANT_CFG = dict(
    input_dim=1024,
    n_codebooks=9,
    codebook_size=1024,
    codebook_dim=8,
    quantizer_dropout=0.5,
    downsample_factor=[2, 2],
    semantic_codebook_size=4096,
)
WINDOW = dict(causal=True, window_size=128, input_dim=1024)


def _window_transformer():
    return WindowLimitedTransformer(config=ModelArgs(**TRANSFORMER_CFG), **WINDOW)


def build_codec() -> DAC:
    quant = DownsampleResidualVectorQuantize(
        pre_module=_window_transformer(), post_module=_window_transformer(), **QUANT_CFG
    )
    general = functools.partial(ModelArgs, **GENERAL_CFG)
    return DAC(quantizer=quant, transformer_general_config=general, **DAC_CFG)


def check_yaml(path=None) -> bool:
    """True if the vendored yaml still matches the constants (needs pyyaml; returns None if unavailable)."""
    try:
        import yaml
    except ImportError:
        return None
    y = yaml.safe_load(open(path or Path(__file__).with_name("modded_dac_vq.yaml")))

    def same(a, b):  # YAML 1.1 reads "1e-5" as a string
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return a == b

    ok = all(same(y[k], v) for k, v in DAC_CFG.items())
    q = y["quantizer"]
    ok &= all(same(q[k], v) for k, v in QUANT_CFG.items())
    t = q["post_module"]["config"]
    ok &= all(same(t[k], v) for k, v in TRANSFORMER_CFG.items()) and q["post_module"]["window_size"] == 128
    return bool(ok)


def load_codec(snapshot, device="cpu", dtype=torch.float32) -> DAC:
    from models.autoports.fishaudio_s2_pro.tt.weights import load_codec_state

    model = build_codec()
    sd = load_codec_state(snapshot)
    result = model.load_state_dict(sd, strict=False, assign=True)
    unexpected = [k for k in result.unexpected_keys]
    assert not unexpected, f"unexpected codec keys: {unexpected[:5]}"
    missing = [k for k in result.missing_keys if not k.endswith(("causal_mask", "freqs_cis"))]
    assert not missing, f"missing codec weights: {missing[:5]}"
    return model.to(device=device, dtype=dtype).eval()
