import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

MISSING: Any = "???"

AVAILABLE_ACT_FNS = Literal[
    "relu",
    "gelu",
    "gelu_approximate",
    "tanh",
    "linear",
]

EXTRACTOR_MODE_CHOICES = Literal[
    "default",
    "layer_norm",
]
LAYER_TYPE_CHOICES = Literal[
    "transformer",
    "conformer",
]

MASKING_DISTRIBUTION_CHOICES = Literal[
    "static",
    "uniform",
    "normal",
    "poisson",
]


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
    random_crop: bool | None = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
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


def get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])


def relu_squared(x: torch.Tensor):
    return F.relu(x).pow(2)


def index_put(tensor, indices, value):
    tensor[indices] = value
    return tensor


def gelu_approximate(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x.float()).type_as(x)


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    if activation == "relu":
        return F.relu
    elif activation == "relu_squared":
        return relu_squared
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_approximate":
        return gelu_approximate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return nn.SiLU
    else:
        raise RuntimeError(f"--activation-fn {activation} not supported")


def pad_to_multiple(x, multiple, dim=-1, value=0):
    # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if m.is_integer():
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2

    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder


class TransposeLast(nn.Module):
    def __init__(self):
        super().__init__()
        self.tranpose_dim = -2

    def forward(self, x):
        return x.transpose(-2, -1)


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


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

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(default="static", metadata={"help": "how to choose mask length"})
    mask_other: float = field(
        default=0,
        metadata={"help": "secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh"},
    )
    no_mask_overlap: bool = field(default=False, metadata={"help": "whether to allow masks to overlap"})
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={"help": "secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh"},
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
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

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
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
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})


@dataclass
class Wav2Vec2Config:
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
            "groups in the first conv block, whereas layer_norm has layer norms in "
            "every block (meant to use with normalize=True)"
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
        metadata={"help": "project final representations and targets to this many dimensions.set to encoder_embed_dim is <= 0"},
    )
    layer_norm_first: bool = field(default=False, metadata={"help": "apply layernorm first in the transformer"})
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(default=False, metadata={"help": "include bias in conv encoder"})
    logit_temp: float = field(default=0.1, metadata={"help": "temperature to divide logits by"})
    target_glu: bool = field(default=False, metadata={"help": "adds projection + glu to targets"})
    feature_grad_mult: float = field(default=1.0, metadata={"help": "multiply feature extractor var grads by this"})
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )
    latent_groups: int = field(
        default=2,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )
    latent_dim: int = field(
        default=0,
        metadata={"help": "if > 0, uses this dimensionality for latent variables. otherwise uses final_dim / latent_groups"},
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(default=0.65, metadata={"help": "probability of replacing a token with mask"})
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(default="static", metadata={"help": "how to choose mask length"})
    mask_other: float = field(
        default=0,
        metadata={"help": "secondary mask argument (used for more complex distributions), see help in compute_mask_indices"},
    )
    no_mask_overlap: bool = field(default=False, metadata={"help": "whether to allow masks to overlap"})
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    require_same_masks: bool = field(
        default=True,
        metadata={"help": "whether to number of masked timesteps must be the same across all examples in a batch"},
    )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )

    # channel masking
    mask_channel_length: int = field(default=10, metadata={"help": "length of the mask for features (channels)"})
    mask_channel_prob: float = field(default=0.0, metadata={"help": "probability of replacing a feature with 0"})
    mask_channel_before: bool = False
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={"help": "secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh"},
    )
    no_mask_channel_overlap: bool = field(default=False, metadata={"help": "whether to allow channel masks to overlap"})
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # negative selection
    num_negatives: int = field(
        default=100,
        metadata={"help": "number of negative examples from the same sample"},
    )
    negatives_from_everywhere: bool = field(
        default=False,
        metadata={"help": "sample negatives from everywhere, not just masked states"},
    )
    cross_sample_negatives: int = field(default=0, metadata={"help": "number of negative examples from the any sample"})
    codebook_negatives: int = field(default=0, metadata={"help": "number of negative examples codebook"})

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    pos_conv_depth: int = field(
        default=1,
        metadata={"help": "depth of positional encoder network"},
    )

    latent_temp: tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "temperature for latent variable sampling. can be tuple of 3 values (start, end, decay)"},
    )
    max_positions: int = field(default=100000, metadata={"help": "Max positions"})
    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={"help": "pad the input to encoder such that the sequence length is divisible by multiple"},
    )
    crop_seq_to_multiple: int = field(
        default=1,
        metadata={"help": "crop convolutional feature extractor output such that the sequence length is divisible by multiple"},
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
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        """Rotary positional embedding
        Reference : https://blog.eleuther.ai/rotary-embeddings/
        Paper: https://arxiv.org/pdf/2104.09864.pdf
        Args:
            dim: Dimension of embedding
            base: Base value for exponential
            precision: precision to use for numerical values
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = 0
        self.cos_cached = torch.empty(self.seq_len_cached, 1, 1, dim)
        self.sin_cached = torch.empty(self.seq_len_cached, 1, 1, dim)

    def forward(self, x, seq_len: int = 0):
        """
        Args:
            x: Input x with T X B X C
            seq_len: Sequence length of input x
        """
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos().view(emb.size(0), 1, 1, emb.size(1))
            self.sin_cached = emb.sin().view(emb.size(0), 1, 1, emb.size(1))
        return self.cos_cached, self.sin_cached


class ESPNETMultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    Args:
        n_head: The number of heads.
        n_feat: The number of features.
    """

    def __init__(self, n_feat, n_head):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

    def forward_qkv(self, query, key, value, **kwargs):
        """Transform query, key and value.
        Args:
            query: Query tensor  B X T1 X C
            key: Key tensor B X T2 X C
            value: Value tensor  B X T2 X C
        Returns:
            torch.Tensor: Transformed query tensor  B X n_head X T1 X d_k
            torch.Tensor: Transformed key tensor B X n_head X T2 X d_k
            torch.Tensor: Transformed value tensor  B X n_head X T2 X d_k
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value: Transformed value B X n_head X T2 X d_k.
            scores: Attention score  B X n_head X T1 X T2
            mask: Mask  T2 X B
        Returns:
            torch.Tensor: Transformed value  B X T1 X d_model
                weighted by the attention score  B X T1 X T2
        """
        n_batch = value.size(0)
        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2).to(bool),
                float("-inf"),  # (batch, head, time1, time2)
            )
            p_attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        else:
            p_attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):
        """Compute scaled dot product attention.
        Args:
            query (torch.Tensor): Query tensor T X B X C
            key (torch.Tensor): Key tensor T X B X C
            value (torch.Tensor): Value tensor T X B X C
            mask (torch.Tensor): Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X D.
        """
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self.forward_attention(v, scores, key_padding_mask)
        scores = scores.transpose(0, 1)
        return scores, None


class RelPositionMultiHeadedAttention(ESPNETMultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head: The number of heads.
        n_feat: The number of features.
        zero_triu: Whether to zero the upper triangular part of attention matrix.
    """

    def __init__(self, n_feat, n_head, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_feat, n_head)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = Parameter(torch.zeros(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x: Input tensor B X n_head X T X 2T-1
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, key_padding_mask=None, **kwargs):
        """Compute scaled dot product attention.
        Args:
            query: Query tensor T X B X C
            key: Key tensor T X B X C
            value: Value tensor T X B X C
            pos_emb: Positional embedding tensor B X 2T-1 X C
            key_padding_mask: Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X C.
        """
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        pos_emb = pos_emb.transpose(0, 1)
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        scores = self.forward_attention(v, scores, key_padding_mask)
        scores = scores.transpose(0, 1)
        return scores, None


class RotaryPositionMultiHeadedAttention(ESPNETMultiHeadedAttention):
    def __init__(
        self,
        n_feat,
        n_head,
        precision,
    ):
        """Construct an RotaryPositionMultiHeadedAttention object."""
        super().__init__(n_feat, n_head)
        rotary_emd_base = 10000
        precision = torch.float
        self.rotary_ndims = self.d_k  # also try self.d_k//2
        if precision == "fp16":
            precision = torch.half

        self.rotary_emb = RotaryPositionalEmbedding(self.rotary_ndims, base=rotary_emd_base, precision=precision)

    def forward(self, query, key, value, key_padding_mask=None, **kwargs):
        """Compute rotary position attention.
        Args:
            query: Query tensor T X B X C
            key: Key tensor T X B X C
            value: Value tensor T X B X C
            key_padding_mask: Mask tensor T X B
        Returns:
            torch.Tensor: Output tensor T X B X D.
        Notes:
            Assumes self attn
        """

        T, B, C = value.size()
        query = query.view(T, B, self.h, self.d_k)
        key = key.view(T, B, self.h, self.d_k)
        value = value.view(T, B, self.h, self.d_k)
        cos, sin = self.rotary_emb(value, seq_len=T)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, offset=0)  # offset is based on layer_past

        query = query.view(T, B, self.h * self.d_k)
        key = key.view(T, B, self.h * self.d_k)
        value = value.view(T, B, self.h * self.d_k)

        # TBD to BTD
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = self.forward_attention(v, scores, key_padding_mask)
        scores = scores.transpose(0, 1)
        return scores, None


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:
        super().__init__()
        # Initialize parameters
        self.embedding_dim = embed_dim

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            attention_heads,
            self_attention=True,
        )

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_padding_mask: torch.Tensor | None = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
            )
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.fc2(x)

            x = residual + x
        else:
            x = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
            )

            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.fc2(x)

            x = residual + x
            x = self.final_layer_norm(x)

        return x


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: list[tuple[int, int, int]],
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert not (is_layer_norm and is_group_norm), "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=0.0),
                    nn.Sequential(
                        TransposeLast(),
                        nn.LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=0.0),
                    nn.GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=0.0), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x


class FeedForwardModule(nn.Module):
    """Positionwise feed forward layer used in conformer"""

    def __init__(
        self,
        input_feat,
        hidden_units,
        activation_fn="swish",
        bias=True,
    ):
        """
        Args:
            input_feat: Input feature dimension
            hidden_units: Hidden unit dimension
            activation_fn: Name of activation function
            bias: If linear layers should have bias
        """

        super().__init__()
        self.layer_norm = nn.LayerNorm(input_feat)
        self.w_1 = nn.Linear(input_feat, hidden_units, bias=bias)
        self.w_2 = nn.Linear(hidden_units, input_feat, bias=bias)
        self.activation = get_activation_fn(activation_fn)(hidden_units)

    def forward(self, x):
        """
        Args:
            x: Input Tensor of shape  T X B X C
        Returns:
            Tensor of shape T X B X C
        """
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = self.activation(x)
        return self.w_2(x)


class ConvolutionModule(nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(
        self,
        embed_dim,
        channels,
        depthwise_kernel_size,
        activation_fn="swish",
        bias=False,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            channels: Number of channels in depthwise conv layers
            depthwise_kernel_size: Depthwise conv layer kernel size
            activation_fn: Activation function to use after depthwise convolution kernel
            bias: If bias should be added to conv layers
        """
        super().__init__()
        assert (depthwise_kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.pointwise_conv1 = nn.Conv1d(
            embed_dim,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            depthwise_kernel_size,
            stride=1,
            padding=(depthwise_kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.batch_norm = nn.BatchNorm1d(channels)
        self.activation = get_activation_fn(activation_fn)(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x: Input of shape B X T X C
        Returns:
          Tensor of shape B X T X C
        """
        x = self.layer_norm(x)
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = self.glu(x)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class ConformerWav2Vec2EncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100.
    We currently don't support relative positional encoding in MHA"""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        use_fp16,
        depthwise_conv_kernel_size=31,
        activation_fn="swish",
        attn_type=None,
        pos_enc_type="abs",
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            ffn_embed_dim: FFN layer dimension
            attention_heads: Number of attention heads in MHA
            depthwise_conv_kernel_size: Size of kernel in depthwise conv layer in convolution module
            activation_fn: Activation function name to use in convulation block and feed forward block
            attn_type: MHA implementation type
            pos_enc_type: Positional encoding type - abs, rope, rel_pos
        """
        self.pos_enc_type = pos_enc_type
        super().__init__()

        self.ffn1 = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
        )

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        if attn_type == "espnet":
            if self.pos_enc_type == "rel_pos":
                self.self_attn = RelPositionMultiHeadedAttention(
                    embed_dim,
                    attention_heads,
                )
            elif self.pos_enc_type == "rope":
                self.self_attn = RotaryPositionMultiHeadedAttention(embed_dim, attention_heads, precision=use_fp16)
            elif self.pos_enc_type == "abs":
                self.self_attn = ESPNETMultiHeadedAttention(
                    embed_dim,
                    attention_heads,
                )
            else:
                raise Exception(f"Unsupported attention type {self.pos_enc_type}")
        else:
            # Default to MHA
            self.self_attn = MultiheadAttention(
                embed_dim,
                attention_heads,
            )

        self.conv_module = ConvolutionModule(
            embed_dim=embed_dim,
            channels=embed_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            activation_fn=activation_fn,
        )

        self.ffn2 = FeedForwardModule(
            embed_dim,
            ffn_embed_dim,
            activation_fn=activation_fn,
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_padding_mask: torch.Tensor | None = None,
        position_emb=None,
    ):
        """
        Args:
            x: Tensor of shape T X B X C
            encoder_padding_mask: Optional mask tensor
            positions:
        Returns:
            Tensor of shape T X B X C
        """
        residual = x
        x = self.ffn1(x)
        x = x * 0.5 + residual
        residual = x
        x = self.self_attn_layer_norm(x)
        if self.pos_enc_type == "rel_pos":
            x = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                pos_emb=position_emb,
            )
        else:
            x = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
            )
        x = x + residual

        residual = x
        # TBC to BTC
        x = x.transpose(0, 1)
        x = self.conv_module(x)
        # BTC to TBC
        x = x.transpose(0, 1)
        x = residual + x

        residual = x
        x = self.ffn2(x)

        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        self_attention=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=True)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _pad_masks(
        self,
        key_padding_mask: Tensor | None,
    ) -> Tensor | None:
        if key_padding_mask is not None:
            shape = key_padding_mask.size()[:-1] + torch.Size([1])
            key_padding_mask = torch.cat(
                [
                    key_padding_mask,
                    key_padding_mask.new_zeros(shape),
                ],
                dim=-1,
            )
        return key_padding_mask

    def forward(
        self,
        query: Tensor,
        key: Tensor | None,
        value: Tensor | None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            assert value is not None
            assert src_len, key_bsz == value.shape[:2]

        if (
            # The Multihead attention implemented in pytorch forces strong dimension check
            # for input embedding dimention and K,Q,V projection dimension.
            # Since pruning will break the dimension check and it is not easy to modify the pytorch API,
            # it is preferred to bypass the pytorch MHA when we need to skip embed_dim_check
            True
        ):
            assert key is not None and value is not None

            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                None,
                None,
                False,
                0.0,
                self.out_proj.weight,
                self.out_proj.bias,
                False,
                key_padding_mask.bool() if key_padding_mask is not None else None,
                False,
                None,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )[0]

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        kv_bsz = bsz  # need default value for scripting
        if k is not None:
            kv_bsz = k.size(1)
            k = k.contiguous().view(-1, kv_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, kv_bsz * self.num_heads, self.head_dim).transpose(0, 1)

        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == kv_bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(kv_bsz, -1, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = attn_weights

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)
        return attn


def make_conv_pos(e, k, g, is_batch_norm=False):
    pos_conv = nn.Conv1d(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
    )
    std = math.sqrt(4 / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    if not is_batch_norm:
        pos_conv = nn.utils.parametrizations.weight_norm(pos_conv, name="weight", dim=2)
        pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())
    else:
        batch_norm = nn.BatchNorm1d(e)
        pos_conv = nn.Sequential(batch_norm, pos_conv, SamePad(k), nn.GELU())

    return pos_conv


class TransformerEncoder(nn.Module):
    def build_encoder_layer(self, args: dict[str, Any]):
        if args.get("layer_type", "transformer") == "transformer":
            layer = TransformerSentenceEncoderLayer(
                embed_dim=self.embedding_dim,
                ffn_embed_dim=args["encoder_ffn_embed_dim"],
                attention_heads=args["encoder_attention_heads"],
                activation_fn=args["activation_fn"],
                layer_norm_first=args["layer_norm_first"],
            )
        elif args.get("layer_type", "transformer") == "conformer":
            layer = ConformerWav2Vec2EncoderLayer(
                embed_dim=self.embedding_dim,
                ffn_embed_dim=args["encoder_ffn_embed_dim"],
                attention_heads=args["encoder_attention_heads"],
                depthwise_conv_kernel_size=args["depthwise_conv_kernel_size"],
                activation_fn="swish",
                attn_type=args["attn_type"],
                use_fp16=args["fp16"],
                pos_enc_type="abs",
            )
        return layer

    def __init__(
        self,
        args: dict[str, Any],
    ):
        super().__init__()

        self.embedding_dim = args["encoder_embed_dim"]
        self.required_seq_len_multiple = args.get("required_seq_len_multiple", 2)

        pos_conv_depth = getattr(args, "pos_conv_depth", 1)
        if pos_conv_depth > 1:
            num_layers = args["pos_conv_depth"]
            k = max(3, args["conv_pos"] // num_layers)

            def make_conv_block(e, k, g, n_layers):
                return nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv1d(
                                e,
                                e,
                                kernel_size=k,
                                padding=k // 2,
                                groups=g,
                            ),
                            SamePad(k),
                            TransposeLast(),
                            nn.LayerNorm(e, elementwise_affine=False),
                            TransposeLast(),
                            nn.GELU(),
                        )
                        for _ in range(n_layers)
                    ]
                )

            self.pos_conv = make_conv_block(self.embedding_dim, k, args["conv_pos_groups"], num_layers)
        else:
            self.pos_conv = make_conv_pos(
                self.embedding_dim,
                args["conv_pos"],
                args["conv_pos_groups"],
                is_batch_norm=args.get("conv_pos_batch_norm", False),
            )

        encoder_layers = args["encoder_layers"]

        self.layers = nn.ModuleList([self.build_encoder_layer(args) for _ in range(encoder_layers)])
        self.layer_norm_first = args["layer_norm_first"]
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x,
        padding_mask,
        tgt_layer,
    ):
        x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(x, self.required_seq_len_multiple, dim=-2, value=0)

        padding_mask, _ = pad_to_multiple(padding_mask, self.required_seq_len_multiple, dim=-1, value=True)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        for i, layer in enumerate(self.layers):
            x = layer(x, self_attn_padding_mask=padding_mask)
            if i == tgt_layer:
                break

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # undo paddding
        if pad_length > 0:
            x = x[:, :-pad_length]
        return x


class HubertModel(nn.Module):
    def __init__(
        self,
        cfg,
        task_cfg: HubertPretrainingConfig,
    ):
        super().__init__()
        self._is_generation_fast = False
        feature_enc_layers = eval(cfg["conv_feature_layers"])  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            mode=cfg["extractor_mode"],
            conv_bias=cfg["conv_bias"],
        )
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = cfg["label_rate"] * feature_ds_rate / task_cfg.sample_rate

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg["encoder_embed_dim"]) if self.embed != cfg["encoder_embed_dim"] else None
        )

        self.feature_grad_mult = cfg["feature_grad_mult"]
        self.logit_temp = cfg["logit_temp"]

        final_dim = cfg["final_dim"] if cfg["final_dim"] > 0 else cfg["encoder_embed_dim"]

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = nn.LayerNorm(self.embed)

        self.untie_final_proj = cfg["untie_final_proj"]
        self.final_proj = nn.Linear(cfg["encoder_embed_dim"], final_dim)

    @classmethod
    def build_model(cls, cfg, task: HubertPretrainingTask):
        """Build a new model instance."""
        return HubertModel(cfg, task.cfg)

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor,
        output_layer: int,
    ) -> dict[str, torch.Tensor]:
        """output layer is 1-based"""
        features = self.feature_extractor(source)

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x = features

        x = self.encoder(
            x,
            padding_mask=padding_mask,
            tgt_layer=output_layer - 1,
        )

        return {"x": x, "features": features}

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor,
        output_layer: int,
    ) -> torch.Tensor:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            output_layer=output_layer,
        )
        return res["x"]

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list
