# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

MISSING: Any = "???"


EXTRACTOR_MODE_CHOICES = Literal[
    "default",
    "layer_norm",
]

LAYER_TYPE_CHOICES = Literal[
    "transformer",
    "conformer",
]


AVAILABLE_ACT_FNS = Literal[
    "relu",
    "gelu",
    "gelu_approximate",
    "tanh",
    "linear",
]


class Config:
    def __init__(self):
        self.device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.instead: str | None = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def params_config(self) -> tuple:
        # 5G GPU_RAM conf
        # note: keep x_query < x_center/ 2 to voice degradation
        x_pad = 1
        x_query = 3
        x_center = 12
        x_max = 16

        # x_pad = 1
        # x_query = 6
        # x_center = 24
        # x_max = 16

        # x_pad = 1
        # x_query = 6
        # x_center = 38
        # x_max = 41

        # x_pad = 1
        # x_query = 2
        # x_center = 6
        # x_max = 8

        # x_pad = 1
        # x_query = 1
        # x_center = 3
        # x_max = 4

        return x_pad, x_query, x_center, x_max

    def use_cpu(self) -> None:
        self.device = self.instead = "cpu"
        self.params_config()

    def device_config(self) -> tuple:
        return self.params_config()


def _get_configs_dir() -> str:
    configs_dir = os.getenv("RVC_CONFIGS_DIR")
    if not configs_dir:
        raise OSError("RVC_CONFIGS_DIR is not set. Set it to the directory containing v1/ and v2/ config folders.")
    if not os.path.isdir(configs_dir):
        raise FileNotFoundError(f"RVC_CONFIGS_DIR does not exist: {configs_dir}")
    return configs_dir


def _get_assets_dir() -> str:
    assets_dir = os.getenv("RVC_ASSETS_DIR")
    if not assets_dir:
        raise OSError("RVC_ASSETS_DIR is not set. Set it to the directory containing pretrained model weights.")
    if not os.path.isdir(assets_dir):
        raise FileNotFoundError(f"RVC_ASSETS_DIR does not exist: {assets_dir}")
    return assets_dir


def _build_path_mapping() -> dict[tuple[str, str, bool], tuple[str, str]]:
    configs_dir = _get_configs_dir()
    assets_dir = _get_assets_dir()
    pretrained_dir = os.path.join(assets_dir, "pretrained")
    return {
        ("v1", "32k", True): (
            os.path.join(pretrained_dir, "f0G32k.safetensors"),
            os.path.join(configs_dir, "v1/32k.json"),
        ),
        ("v1", "40k", True): (
            os.path.join(pretrained_dir, "f0G40k.safetensors"),
            os.path.join(configs_dir, "v1/40k.json"),
        ),
        ("v1", "48k", True): (
            os.path.join(pretrained_dir, "f0G48k.safetensors"),
            os.path.join(configs_dir, "v1/48k.json"),
        ),
        ("v1", "48k", False): (
            os.path.join(pretrained_dir, "G48k.safetensors"),
            os.path.join(configs_dir, "v1/48k.json"),
        ),
    }


def get_hubert_paths():
    configs_dir = _get_configs_dir()
    assets_dir = _get_assets_dir()
    return (
        os.path.join(configs_dir, "hubert_cfg.json"),
        os.path.join(assets_dir, "hubert.safetensors"),
    )


path_mapping = _build_path_mapping()


def get_model_and_config_paths(version: str, num: str, if_f0: bool) -> tuple[str, str]:
    key = (version, num, if_f0)
    if key not in path_mapping:
        raise KeyError(f"Unsupported path mapping key: {key}")
    return path_mapping[key]


@dataclass
class HubertPretrainingConfig:
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: list[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: str | None = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: float = field(
        default=-1.0,
        metadata={"help": "label frame rate. -1.0 for sequence label"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={"help": "target sample rate. audio files will be up/down sampled to this rate"},
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: int | None = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: int | None = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: int | None = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    single_target: bool | None = field(
        default=False,
        metadata={"help": "if set, AddTargetDatasets outputs same keys as AddTargetDataset"},
    )
    pad_audio: bool | None = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )


class HubertPretrainingTask:
    cfg: HubertPretrainingConfig

    def __init__(
        self,
        cfg: HubertPretrainingConfig,
    ):
        super().__init__()
        self.cfg = cfg

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size = criterion(model, sample)
        return loss, sample_size


@dataclass
class HubertConfig:
    label_rate: float = field(default=50, metadata={"help": "label rate"})

    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(default=12, metadata={"help": "num encoder layers in the transformer"})
    encoder_embed_dim: int = field(default=768, metadata={"help": "encoder embedding dimension"})
    encoder_ffn_embed_dim: int = field(default=3072, metadata={"help": "encoder embedding dimension for FFN"})
    encoder_attention_heads: int = field(default=12, metadata={"help": "num encoder attention heads"})
    activation_fn: AVAILABLE_ACT_FNS = field(default="gelu", metadata={"help": "activation function to use"})
    layer_type: LAYER_TYPE_CHOICES = field(default="transformer", metadata={"help": "layer type in encoder"})

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(default=False, metadata={"help": "include bias in conv encoder"})
    logit_temp: float = field(default=0.1, metadata={"help": "temperature to divide logits by"})
    target_glu: bool = field(default=False, metadata={"help": "adds projection + glu to targets"})
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    conv_pos_batch_norm: bool = field(
        default=False,
        metadata={"help": "use batch norm instead of weight norm in conv_pos (for bf16 models)"},
    )

    latent_temp: tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )
    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={"help": "pad the input to encoder such that the sequence length is divisible by multiple"},
    )

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={"help": "depthwise-conv-kernel-size for convolution in conformer layer"},
    )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
