# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint access for Qwen3-TTS.

The Base checkpoint is a single `model.safetensors` holding exactly two top-level
prefixes: `speaker_encoder.` (76 tensors, 12.0M parameters) and `talker.` (the 1.7B
decoder). Every reader here opens the file lazily and pulls only the keys it names, so
speaker-encoder work never materialises the talker.

Where the checkpoint comes from, first match wins:

    $QWEN3_TTS_CKPT   a local directory holding config.json + model.safetensors
    $HF_MODEL         a hub id, or a local directory (the tiered-CI convention)
    DEFAULT_REPO      Qwen/Qwen3-TTS-12Hz-1.7B-Base at PINNED_REVISION

`expected_speaker_shapes()` derives every speaker-encoder tensor name and shape from the
config alone, mirroring how `Qwen3TTSSpeakerEncoder.__init__` builds its modules. Nothing
is copied from the file, so a checkpoint whose weights no longer match its own config
fails loudly instead of loading crooked.

Architecture references point into qwen-tts 0.1.1 (see the pins in the model-bringup
notes); `core/models/modeling_qwen3_tts.py:311` is the encoder, `:95-308` its blocks.
"""

import functools
import json
import os

import torch
from safetensors import safe_open

DEFAULT_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PINNED_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
SPEAKER_PREFIX = "speaker_encoder."
TALKER_PREFIX = "talker."

# What a fresh download needs: the config, the weights, and the BPE vocab the text
# front-end reads. The 25 Hz tokenizer under speech_tokenizer/ is not pulled here.
HUB_PATTERNS = [
    CONFIG_FILE,
    WEIGHTS_FILE,
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
]

# Qwen3TTSSpeakerEncoderConfig's own defaults (configuration_qwen3_tts.py:47). The
# published config.json carries only the fields that differ from these, so the rest has
# to live somewhere; keep this in sync with the upstream dataclass, not with one file.
SPEAKER_ENCODER_DEFAULTS = {
    "mel_dim": 128,
    "enc_dim": 1024,
    "enc_channels": [512, 512, 512, 512, 1536],
    "enc_kernel_sizes": [5, 3, 3, 3, 1],
    "enc_dilations": [1, 2, 3, 4, 1],
    "enc_attention_channels": 128,
    "enc_res2net_scale": 8,
    "enc_se_channels": 128,
    "sample_rate": 24000,
}

# The mel front-end that feeds the speaker encoder, from the call site in
# `Qwen3TTSForConditionalGeneration.extract_speaker_embedding` (modeling:1943). These are
# call-site literals rather than config fields, which is why they are pinned here.
SPEAKER_MEL = {
    "n_fft": 1024,
    "num_mels": 128,
    "sampling_rate": 24000,
    "hop_size": 256,
    "win_size": 1024,
    "fmin": 0,
    "fmax": 12000,
}


def _hub_revision(repo):
    """Pin the default repo; leave any other repo on whatever the caller asked for."""
    override = os.environ.get("QWEN3_TTS_REVISION", "").strip()
    if override:
        return override
    return PINNED_REVISION if repo == DEFAULT_REPO else None


@functools.lru_cache(maxsize=None)
def checkpoint_dir(allow_download=True):
    """Local directory holding config.json and model.safetensors.

    Set `allow_download=False` to fail rather than reach the network, which is what a
    test wants when it is meant to be measuring a warm cache.
    """
    local = os.environ.get("QWEN3_TTS_CKPT", "").strip()
    if local:
        if not os.path.isdir(local):
            raise FileNotFoundError(f"$QWEN3_TTS_CKPT is not a directory: {local}")
        return local

    repo = os.environ.get("HF_MODEL", "").strip() or DEFAULT_REPO
    if os.path.isdir(repo):  # tiered CI on LFC runners passes a path, not a hub id
        return repo

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo,
        revision=_hub_revision(repo),
        allow_patterns=HUB_PATTERNS,
        local_files_only=not allow_download,
    )


def weights_path(allow_download=True):
    path = os.path.join(checkpoint_dir(allow_download), WEIGHTS_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no {WEIGHTS_FILE} in {checkpoint_dir(allow_download)}")
    return path


@functools.lru_cache(maxsize=None)
def _model_config_json(allow_download=True):
    with open(os.path.join(checkpoint_dir(allow_download), CONFIG_FILE)) as f:
        return json.load(f)


def model_config(allow_download=True):
    """The whole config.json, as a fresh dict the caller may mutate."""
    return json.loads(json.dumps(_model_config_json(allow_download)))


def speaker_encoder_config(allow_download=True):
    """Upstream defaults overlaid with whatever this checkpoint overrides."""
    cfg = dict(SPEAKER_ENCODER_DEFAULTS)
    cfg.update(model_config(allow_download).get("speaker_encoder_config") or {})
    return cfg


def expected_speaker_shapes(cfg=None):
    """Every `speaker_encoder.*` tensor this config implies, keyed without the prefix.

    Mirrors `Qwen3TTSSpeakerEncoder.__init__`: one TDNN block, then a SE-Res2Net block per
    intermediate channel entry, then multi-layer feature aggregation, attentive statistics
    pooling, and a 1x1 output convolution. Conv1d weights are (out, in, kernel).
    """
    cfg = dict(cfg or speaker_encoder_config())
    channels = cfg["enc_channels"]
    kernels = cfg["enc_kernel_sizes"]
    dilations = cfg["enc_dilations"]
    if not len(channels) == len(kernels) == len(dilations):
        raise ValueError("enc_channels, enc_kernel_sizes and enc_dilations must have the same length")

    scale = cfg["enc_res2net_scale"]
    se_channels = cfg["enc_se_channels"]
    attention = cfg["enc_attention_channels"]
    shapes = {}

    def conv(name, out_channels, in_channels, kernel):
        shapes[f"{name}.weight"] = (out_channels, in_channels, kernel)
        shapes[f"{name}.bias"] = (out_channels,)

    # The initial TDNN layer reads the mel bins.
    conv("blocks.0.conv", channels[0], cfg["mel_dim"], kernels[0])

    # SE-Res2Net layers. The last channel entry sizes the aggregation, not a block.
    for i in range(1, len(channels) - 1):
        block = f"blocks.{i}"
        width = channels[i] // scale
        if width * scale != channels[i]:
            raise ValueError(f"enc_channels[{i}]={channels[i]} is not divisible by enc_res2net_scale={scale}")
        conv(f"{block}.tdnn1.conv", channels[i], channels[i - 1], 1)
        # Res2Net splits into `scale` groups and convolves all but the first.
        for j in range(scale - 1):
            conv(f"{block}.res2net_block.blocks.{j}.conv", width, width, kernels[i])
        conv(f"{block}.tdnn2.conv", channels[i], channels[i], 1)
        conv(f"{block}.se_block.conv1", se_channels, channels[i], 1)
        conv(f"{block}.se_block.conv2", channels[i], se_channels, 1)

    # Aggregation runs over the concatenated SE-Res2Net outputs. The encoder sizes this
    # layer from enc_channels[-1], which only works while that equals the concatenation
    # width (3 x 512 = 1536 here); a config that breaks the coincidence breaks the model.
    conv("mfa.conv", channels[-1], channels[-1], kernels[-1])

    # Attentive statistics pooling sees the features plus a broadcast mean and std.
    conv("asp.tdnn.conv", attention, channels[-1] * 3, 1)
    conv("asp.conv", channels[-1], attention, 1)

    # Output projection over the concatenated pooled mean and std.
    conv("fc", cfg["enc_dim"], channels[-1] * 2, 1)
    return shapes


def speaker_shapes_in_file(allow_download=True):
    """Shapes of the `speaker_encoder.*` tensors, read from the header only."""
    with safe_open(weights_path(allow_download), framework="pt") as f:
        return {
            key[len(SPEAKER_PREFIX) :]: tuple(f.get_slice(key).get_shape())
            for key in f.keys()
            if key.startswith(SPEAKER_PREFIX)
        }


def prefixes_in_file(allow_download=True):
    """Top-level prefixes present in the checkpoint, for sanity checks."""
    with safe_open(weights_path(allow_download), framework="pt") as f:
        return {key.split(".", 1)[0] for key in f.keys()}


def load_speaker_state(dtype=torch.float32, allow_download=True):
    """The speaker encoder's weights, keyed without the `speaker_encoder.` prefix.

    Reads only the tensors the config asks for, so the talker's 1.7B of weights stay on
    disk. Defaults to fp32 because the CPU reference runs in fp32; pass `dtype=None` to
    keep the checkpoint's own bf16.
    """
    shapes = expected_speaker_shapes(speaker_encoder_config(allow_download))
    state = {}
    with safe_open(weights_path(allow_download), framework="pt") as f:
        available = set(f.keys())
        missing = sorted(name for name in shapes if SPEAKER_PREFIX + name not in available)
        if missing:
            raise KeyError(f"checkpoint is missing {len(missing)} speaker-encoder tensors, first: {missing[0]}")
        for name, shape in shapes.items():
            tensor = f.get_tensor(SPEAKER_PREFIX + name)
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{SPEAKER_PREFIX}{name}: checkpoint has {tuple(tensor.shape)}, config implies {shape}"
                )
            state[name] = tensor if dtype is None else tensor.to(dtype)
    return state
