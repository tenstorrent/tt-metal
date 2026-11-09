# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import json
import math
import os

import torch
import torch.nn as nn


class PatchTSMixerGatedAttention(nn.Module):
    """
    Module that applies gated attention to input data.
    Args:
        d_model: dimension of the model i.e hidden features vector size.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attn_layer = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        w = self.softmax(self.attn_layer(x))
        return x * w


class PatchTSMixerBatchNorm(nn.Module):
    """
    Module that applies batch normalization over the sequence length (time) dimension
    Args:
        d_model: dimension of the model.
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(d_model, eps)

    def forward(self, x):
        return self.batchnorm(x.transpose(1, 2)).transpose(1, 2)


class PatchTSMixerPositionalEncoding(nn.Module):
    """
    Class for positional encoding.
    """

    def __init__(self, num_patches, d_model, use_pe=True, pe_type="sincos"):
        super().__init__()
        if not use_pe:
            self.pe = nn.Parameter(torch.zeros(num_patches, d_model))
        else:
            if pe_type == "random":
                self.pe = nn.Parameter(torch.randn(num_patches, d_model))
            elif pe_type == "sincos":
                pos = torch.arange(0, num_patches).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
                pe = torch.zeros(num_patches, d_model)
                pe[:, 0::2] = torch.sin(pos * div_term)
                pe[:, 1::2] = torch.cos(pos * div_term)
                pe = (pe - pe.mean()) / (pe.std() * 10)
                self.register_buffer("pe", pe)  # non-learnable buffer
            else:
                raise ValueError("pe_type must be 'random' or 'sincos'")

    def forward(self, x):
        # x: (B, C, N_p, D)
        return x + self.pe


class PatchTSMixerScaler(nn.Module):
    """
    Per-window instance scaling over the time dimension, as HF's std and mean scalers.
    """

    def __init__(self, scaling="std"):
        super().__init__()
        self.scaling = "std" if scaling is True else scaling
        if self.scaling not in ("std", "mean"):
            raise ValueError("scaling must be 'std', 'mean' or True")

    def forward(self, x):
        # x: (B, L, C) -> scaled x, loc (B, 1, C), scale (B, 1, C)
        if self.scaling == "std":
            loc = x.mean(dim=1, keepdim=True)
            scale = torch.sqrt(((x - loc) ** 2).mean(dim=1, keepdim=True) + 1e-5)
        else:
            loc = torch.zeros_like(x[:, :1])
            scale = x.abs().mean(dim=1, keepdim=True).clamp(min=1e-10)
        return (x - loc) / scale, loc, scale


def patch_mask(
    mask_type,
    batch_size,
    num_channels,
    num_patches,
    *,
    random_mask_ratio=0.5,
    num_forecast_mask_patches=(2,),
    channel_consistent_masking=True,
):
    """
    Bool (B, C, N_p) mask of the patches to hide, drawn as HF's random_masking / forecast_masking do.
    """
    if mask_type == "random":
        len_keep = int(num_patches * (1 - random_mask_ratio))
        if channel_consistent_masking:
            noise = torch.rand(batch_size, 1, num_patches).repeat(1, num_channels, 1)
        else:
            noise = torch.rand(batch_size, num_channels, num_patches)
        mask = torch.ones(batch_size, num_channels, num_patches)
        mask[:, :, :len_keep] = 0
        ids_restore = torch.argsort(torch.argsort(noise, dim=-1), dim=-1)
        return torch.gather(mask, dim=-1, index=ids_restore).bool()
    if mask_type == "forecast":
        if isinstance(num_forecast_mask_patches, int):
            num_forecast_mask_patches = [num_forecast_mask_patches]
        counts = [batch_size // len(num_forecast_mask_patches)] * len(num_forecast_mask_patches)
        counts[0] += batch_size - sum(counts)
        mask = torch.zeros(batch_size, num_channels, num_patches)
        start = 0
        for patch_len, count in zip(num_forecast_mask_patches, counts):
            if patch_len <= 0 or patch_len >= num_patches:
                raise ValueError(f"num_forecast_mask_patches {patch_len} must be in (0, {num_patches})")
            mask[start : start + count, :, -patch_len:] = 1
            start += count
        return mask[torch.randperm(batch_size)].bool()
    raise ValueError(f"Invalid mask type {mask_type}.")


class PatchTSMixerNormLayer(nn.Module):
    """
    Normalization block.
    """

    def __init__(self, d_model, norm_type="LayerNorm", eps=1e-5):
        super().__init__()
        self.norm_type = norm_type.lower()
        if "batch" in self.norm_type:
            self.norm = nn.BatchNorm1d(d_model, eps=eps)
        else:
            self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        # x: (B, C, N_p, D)
        if "batch" in self.norm_type:
            B, C, N_p, D = x.shape
            x_reshaped = x.view(B * C, N_p, D)
            x_reshaped = self.norm(x_reshaped.transpose(1, 2)).transpose(1, 2)
            return x_reshaped.view(B, C, N_p, D)
        else:
            # LayerNorm over last dim
            return self.norm(x)


class PatchTSMixerMLP(nn.Module):
    def __init__(self, in_features, out_features, expansion=2, dropout=0.1):
        super().__init__()
        hidden = in_features * expansion
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(torch.nn.functional.gelu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class PatchTSMixerChannelFeatureMixerBlock(nn.Module):
    """
    this module mixes the features in the channel dimension.
    """

    def __init__(
        self, num_channels, d_model, norm_type="LayerNorm", expansion=2, dropout=0.1, use_gated_attn=False, eps=1e-5
    ):
        super().__init__()
        self.norm = PatchTSMixerNormLayer(d_model, norm_type=norm_type, eps=eps)
        self.use_gated_attn = use_gated_attn
        self.mlp = PatchTSMixerMLP(
            in_features=num_channels, out_features=num_channels, expansion=expansion, dropout=dropout
        )
        if use_gated_attn:
            self.gate = PatchTSMixerGatedAttention(d_model=num_channels)

    def forward(self, x):
        # x: (B, C, N_p, D)
        residual = x
        x = self.norm(x)  # (B, C, N_p, D)
        x = x.permute(0, 3, 2, 1)  # (B, D, N_p, C)

        if self.use_gated_attn:
            x = self.gate(x)  # gate over channels

        x = self.mlp(x)  # mix over channels
        x = x.permute(0, 3, 2, 1)  # (B, C, N_p, D)
        return x + residual


class FeatureMixerBlock(nn.Module):
    """
    Module that mixes the hidden feature dimension d_model for each patch.
    """

    def __init__(
        self, d_model: int, expansion: int = 2, dropout: float = 0.1, use_gated_attn: bool = False, eps: float = 1e-5
    ):
        super().__init__()
        self.norm = PatchTSMixerNormLayer(d_model=d_model, eps=eps)
        self.mlp = PatchTSMixerMLP(
            in_features=d_model,
            out_features=d_model,
            expansion=expansion,
            dropout=dropout,
        )
        self.use_gated_attn = use_gated_attn
        if use_gated_attn:
            self.gate = PatchTSMixerGatedAttention(d_model=d_model)

    def forward(self, hidden):
        """

        :hidden: (B, N_p, D)
        :returns: (B, N_p, D)
        """
        residual = hidden

        # Normalize over feature dimension D
        hidden = self.norm(hidden)

        # MLP over the last dim (D)
        hidden = self.mlp(hidden)

        # Optional gated attention over features
        if self.use_gated_attn:
            hidden = self.gate(hidden)

        return hidden + residual


class PatchMixerBlock(nn.Module):
    def __init__(
        self,
        num_patches: int,
        d_model: int,
        expansion: int = 2,
        dropout: float = 0.1,
        use_gated_attn: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.norm = PatchTSMixerNormLayer(d_model=d_model, eps=eps)
        self.mlp = PatchTSMixerMLP(
            in_features=num_patches,
            out_features=num_patches,
            expansion=expansion,
            dropout=dropout,
        )
        self.use_gated_attn = use_gated_attn
        if use_gated_attn:
            self.gate = PatchTSMixerGatedAttention(d_model=num_patches)

    def forward(self, x):
        """

        :x: (B, C, N_p, D)
        :Returns: (B, C, N_p, D)
        """
        residual = x  # (B, C, N_p, D)

        # Normalize over the last dim D
        x = self.norm(x)  # (B, C, N_p, D)

        # We want to mix along the patch dimension N_p, so move it the last axis
        x = x.transpose(2, 3)

        # MLP over patches (last dim = N_p)
        x = self.mlp(x)  # (B, C, D, N_p)

        # Optional gated attention over patches
        if self.use_gated_attn:
            x = self.gate(x)  # (B, C, D, N_p)

        # Put patches back as axis 2: (B, C, D, N_p) -> (B, C, N_p, D)
        x = x.transpose(2, 3)

        # Residual connection
        return x + residual


class PatchTSMixerLayer(nn.Module):
    """
    the PatchTSMixer layer that does all three kinds of mixing.
    """

    def __init__(
        self,
        num_patches: int,
        d_model: int,
        num_channels: int,
        mode: str = "common_channel",  # or "mix_channel"
        expansion: int = 2,
        dropout: float = 0.1,
        use_gated_attn: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.mode = mode

        if mode == "mix_channel":
            self.channel_mixer = PatchTSMixerChannelFeatureMixerBlock(
                num_channels=num_channels,
                d_model=d_model,
                expansion=expansion,
                dropout=dropout,
                use_gated_attn=use_gated_attn,
                eps=eps,
            )
        self.patch_mixer = PatchMixerBlock(
            num_patches=num_patches,
            d_model=d_model,
            expansion=expansion,
            dropout=dropout,
            use_gated_attn=use_gated_attn,
            eps=eps,
        )

        self.feature_mixer = FeatureMixerBlock(
            d_model=d_model,
            expansion=expansion,
            dropout=dropout,
            use_gated_attn=use_gated_attn,
            eps=eps,
        )

    def forward(self, hidden):
        """
        hidden: (B, C, N_p, D)
        """
        if self.mode == "mix_channel":
            hidden = self.channel_mixer(hidden)
        hidden = self.patch_mixer(hidden)
        hidden = self.feature_mixer(hidden)
        return hidden


class PatchTSMixerBlock(nn.Module):
    """
    Simplified backbone: a stack of SimplePatchTSMixerLayer modules.

    Each layer:
      - (optionally) mixes channels
      - mixes patches (time)
      - mixes features (d_model)

    Expected input: hidden of shape (B, C, N_p, D)
    """

    def __init__(self, num_layers: int, layer_kwargs: dict):
        """
        Args:
           num_layers: how many mixer layers to stack.
           layer_kwargs: kwargs passed to each SimplePatchTSMixerLayer, e.g.:
               {
                   "num_patches": 64,
                   "d_model": 16,
                   "num_channels": 7,
                   "mode": "common_channel",
                   "expansion": 2,
                   "dropout": 0.1,
                   "use_gated_attn": False,
                   "eps": 1e-5,
               }
        """
        super().__init__()
        self.layers = nn.ModuleList([PatchTSMixerLayer(**layer_kwargs) for _ in range(num_layers)])

    def forward(self, hidden: torch.Tensor, output_hidden_states: bool = False):
        """
         Args:
            hidden: (B, C, N_p, D)
            output_hidden_states: if True, also return list of layer outputs.

        Returns:
            embedding: final output (B, C, N_p, D)
            all_hidden_states: list of intermediate outputs (or None)
        """
        all_hidden_states = [] if output_hidden_states else None
        embedding = hidden

        for layer in self.layers:
            embedding = layer(embedding)
            if output_hidden_states:
                all_hidden_states.append(embedding)

        return embedding, all_hidden_states


class PatchTSMixerForecastHead(nn.Module):
    """
    Simple forecasting head:
        Input: (B, C, N_p, D)
        Output: (B, prediction_length, C)
    """

    def __init__(self, num_patches: int, d_model: int, prediction_length: int, head_dropout: float = 0.1):
        super().__init__()
        self.prediction_length = prediction_length
        self.dropout = nn.Dropout(head_dropout)
        self.flatten = nn.Flatten(start_dim=-2)  # flatten (N_p, D) -> (N_p*D)
        self.proj = nn.Linear(num_patches * d_model, prediction_length)

    def forward(self, hidden_features: torch.Tensor) -> torch.Tensor:
        # hidden_features: (B, C, N_p, D)
        x = self.flatten(hidden_features)  # (B, C, N_p*D)
        x = self.dropout(x)
        x = self.proj(x)  # (B, C, H)
        x = x.transpose(-1, -2)  # (B, H, C) to match HF convention.
        return x


class PatchTSMixerPatchify(nn.Module):
    """
    Patchify input (B, L, C) into Output (B, C, N_patches, patch_length)
    """

    def __init__(self, context_length, patch_length, patch_stride):
        super().__init__()
        self.context_length = context_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride

        # number of patches
        self.num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1

        # compute where patching should start if misaligned.
        new_len = patch_length + patch_stride * (self.num_patches - 1)
        self.sequence_start = context_length - new_len

    def forward(self, x):
        """
        x: (B, L, C)
        """
        B, L, C = x.shape

        if L != self.context_length:
            raise ValueError(f"Expected sequence length {self.context_length}, got {L}")

        # crop left side if needed
        x = x[:, self.sequence_start :, :]  # (B, new_len, C)

        # unfold along time dimension
        patches = x.unfold(dimension=1, size=self.patch_length, step=self.patch_stride)

        # shapes:
        # x: (B, new_len, C)
        # patches: (B, N_patches, C, patch_length)

        # transpose dims to match HF: (B, C, N_patches, patch_length)
        patches = patches.transpose(1, 2)

        return patches


class PatchTSMixerEmbedding(nn.Module):
    def __init__(self, context_length, patch_length, patch_stride, d_model):
        super().__init__()

        # patchify HF-style
        self.patchify = PatchTSMixerPatchify(
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
        )

        # linear projection: patch_length -> d_model
        self.proj = nn.Linear(patch_length, d_model)

    def forward(self, x, mask=None, mask_value=0):
        """
        x: (B, C, L)
        HF expects (B, L, C) so adjust
        mask: optional bool (B, C, N_p); masked patches are filled with mask_value before the projection
        """
        x = x.transpose(1, 2)
        patches = self.patchify(x)  # (B, C, N_p, patch_len)

        if mask is not None:
            patches = patches.masked_fill(mask.unsqueeze(-1), mask_value)

        # project last dimension
        patches = self.proj(patches)  # (B, C, N_p, d_model)

        return patches


class PatchTSMixerModelForForecasting(nn.Module):
    """
    Minimal PatchTSMixer-style forecasting model.

    Expects past_values of shape (B, L, C) and returns predictions of shape (B, H, C),
    where:
        B = batchs_size
        L = context_length
        C = num_input_channels (variables)
        H = prediction_length
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        patch_length: int,
        patch_stride: int,
        num_channels: int,
        d_model: int,
        num_layers: int,
        mode: str = "common_channel",  # or "mix_channel"
        expansion: int = 2,
        dropout: float = 0.1,
        use_gated_attn: bool = False,
        head_dropout: float = 0.1,
        eps: float = 1e-5,
        scaling=None,  # None, "std", "mean" or True (HF's default "std")
        use_positional_encoding: bool = True,
    ):
        super().__init__()

        self.scaler = PatchTSMixerScaler(scaling) if scaling else None

        # 1) Patchify + linear projection to d_model
        self.patch_embed = PatchTSMixerEmbedding(
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
        )

        # number of patches must match what HF uses
        num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1

        # 2) Positional encoding over patches
        self.pos_enc = (
            PatchTSMixerPositionalEncoding(num_patches=num_patches, d_model=d_model, use_pe=True, pe_type="sincos")
            if use_positional_encoding
            else None
        )

        # 3) Mixer stack (time + feature [+ channel] mixing)
        layer_kwargs = dict(
            num_patches=num_patches,
            d_model=d_model,
            num_channels=num_channels,
            mode=mode,
            expansion=expansion,
            dropout=dropout,
            use_gated_attn=use_gated_attn,
            eps=eps,
        )
        self.mixer_block = PatchTSMixerBlock(
            num_layers=num_layers,
            layer_kwargs=layer_kwargs,
        )

        # 4) forecasting head (B, C, N_p, D) -> (B, H, C)
        self.head = PatchTSMixerForecastHead(
            num_patches=num_patches,
            d_model=d_model,
            prediction_length=prediction_length,
            head_dropout=head_dropout,
        )

    def forward(self, past_values):
        """
        past values: (B, L, C) from ForecastDFDataset batch
        returns: (B, H, C)
        """
        if self.scaler is not None:
            past_values, loc, scale = self.scaler(past_values)

        # (B, L, C) -> (B, C, L)
        x = past_values.transpose(1, 2)

        # 1) patchify & embed: (B, C, L) -> (B, C, N_p, D)
        x = self.patch_embed(x)

        # 2) positional encoding
        if self.pos_enc is not None:
            x = self.pos_enc(x)

        # 3) mixer stack: (B, C, N_p, D) -> (B, C, N_p, D)
        x, _ = self.mixer_block(x, output_hidden_states=False)

        # 4) head: (B, C, N_p, D) -> (B, H, C)
        preds = self.head(x)
        if self.scaler is not None:
            preds = preds * scale + loc
        return preds


class PatchTSMixerLinearHead(nn.Module):
    """
    Linear head for Classification and Regression.
    """

    def __init__(
        self,
        d_model: int,
        num_channels: int,
        num_patches: int,
        num_targets: int,
        head_aggregation: str = None,  # None, "use_last", "max_pool", "avg_pool"
        output_range: tuple = None,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.head_aggregation = head_aggregation
        self.output_range = output_range

        # Calculate input features based on aggregation
        if head_aggregation is None:
            mul_factor = num_patches
        else:
            mul_factor = 1

        in_features = d_model * num_channels * mul_factor

        self.projection = nn.Linear(in_features, num_targets)
        self.dropout = nn.Dropout(head_dropout)

        if head_aggregation is None:
            self.flatten = nn.Flatten(start_dim=-3)
        else:
            self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, hidden_features):
        # hidden_features: (B, C, Np, D)

        # Transpose: (B, C, Np, D) -> (B, C, D, Np)
        x = hidden_features.transpose(-1, -2)

        # Apply aggregation over patch dimension
        if self.head_aggregation == "use_last":
            x = x[..., -1]  # (B, C, D)
        elif self.head_aggregation == "max_pool":
            x = x.max(dim=-1).values  # (B, C, D)
        elif self.head_aggregation == "avg_pool":
            x = x.mean(dim=-1)  # (B, C, D)
        # else: keep all patches (B, C, D, Np)

        # Flatten
        x = self.flatten(x)  # (B, features)
        x = self.dropout(x)
        x = self.projection(x)  # (B, num_targets)

        # Optional output range scaling
        if self.output_range is not None:
            min_val, max_val = self.output_range
            x = torch.sigmoid(x) * (max_val - min_val) + min_val

        return x


class PatchTSMixerForTimeSeriesClassification(nn.Module):
    """Classification variant."""

    def __init__(
        self,
        context_length: int,
        patch_length: int,
        patch_stride: int,
        num_channels: int,
        d_model: int,
        num_layers: int,
        num_classes: int,  # Number of classes
        mode: str = "common_channel",
        expansion: int = 2,
        dropout: float = 0.0,
        use_gated_attn: bool = False,
        head_aggregation: str = "avg_pool",
        head_dropout: float = 0.0,
        eps: float = 1e-5,
    ):
        super().__init__()

        # Calculate number of patches
        num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1

        # Embedding
        self.embedding = PatchTSMixerEmbedding(
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
        )

        # Positional encoding
        self.pos_encoder = PatchTSMixerPositionalEncoding(
            num_patches=num_patches,
            d_model=d_model,
        )

        # Encoder
        layer_kwargs = {
            "num_patches": num_patches,
            "d_model": d_model,
            "num_channels": num_channels,
            "mode": mode,
            "expansion": expansion,
            "dropout": dropout,
            "use_gated_attn": use_gated_attn,
            "eps": eps,
        }
        self.encoder = PatchTSMixerBlock(num_layers=num_layers, layer_kwargs=layer_kwargs)

        # Classification head
        self.head = PatchTSMixerLinearHead(
            d_model=d_model,
            num_channels=num_channels,
            num_patches=num_patches,
            num_targets=num_classes,
            head_aggregation=head_aggregation,
            head_dropout=head_dropout,
        )

    def forward(self, past_values):
        # (B, L, C) -> (B, C, L)
        x = past_values.transpose(1, 2)

        x = self.embedding(x)  # (B, C, Np, D)
        x = self.pos_encoder(x)  # (B, C, Np, D)
        x, _ = self.encoder(x)  # Unpack tuple
        logits = self.head(x)  # (B, num_classes)
        return logits


class PatchTSMixerForRegression(nn.Module):
    """Regression variant."""

    def __init__(
        self,
        context_length: int,
        patch_length: int,
        patch_stride: int,
        num_channels: int,
        d_model: int,
        num_layers: int,
        num_targets: int,  # Number of regression targets
        output_range: tuple = None,  # (min, max)
        mode: str = "common_channel",
        expansion: int = 2,
        dropout: float = 0.0,
        use_gated_attn: bool = False,
        head_aggregation: str = "avg_pool",
        head_dropout: float = 0.0,
        eps: float = 1e-5,
    ):
        super().__init__()

        num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1

        self.embedding = PatchTSMixerEmbedding(
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
        )

        self.pos_encoder = PatchTSMixerPositionalEncoding(
            num_patches=num_patches,
            d_model=d_model,
        )

        layer_kwargs = {
            "num_patches": num_patches,
            "d_model": d_model,
            "num_channels": num_channels,
            "mode": mode,
            "expansion": expansion,
            "dropout": dropout,
            "use_gated_attn": use_gated_attn,
            "eps": eps,
        }
        self.encoder = PatchTSMixerBlock(num_layers=num_layers, layer_kwargs=layer_kwargs)

        self.head = PatchTSMixerLinearHead(
            d_model=d_model,
            num_channels=num_channels,
            num_patches=num_patches,
            num_targets=num_targets,
            head_aggregation=head_aggregation,
            output_range=output_range,
            head_dropout=head_dropout,
        )

    def forward(self, past_values):
        # (B, L, C) -> (B, C, L)
        x = past_values.transpose(1, 2)

        x = self.embedding(x)
        x = self.pos_encoder(x)
        x, _ = self.encoder(x)  # Unpack tuple
        predictions = self.head(x)  # (B, num_targets)
        return predictions


class PatchTSMixerPretrainHead(nn.Module):
    """Head for self-supervised pre-training via masked patch prediction."""

    def __init__(
        self,
        d_model: int,
        patch_length: int,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        # Project from d_model back to patch_length to reconstruct patches
        self.projection = nn.Linear(d_model, patch_length)

    def forward(self, hidden_features):
        # hidden_features: (B, C, Np, D)
        x = self.dropout(hidden_features)
        x = self.projection(x)  # (B, C, Np, patch_length)
        return x


class PatchTSMixerForPretraining(nn.Module):
    """Pre-training variant for masked patch prediction."""

    def __init__(
        self,
        context_length: int,
        patch_length: int,
        patch_stride: int,
        num_channels: int,
        d_model: int,
        num_layers: int,
        mode: str = "common_channel",
        expansion: int = 2,
        dropout: float = 0.0,
        use_gated_attn: bool = False,
        head_dropout: float = 0.0,
        eps: float = 1e-5,
        scaling=None,  # None, "std", "mean" or True (HF's default "std")
        use_positional_encoding: bool = True,
        mask_type: str = None,  # None, "random" or "forecast" (HF's default "random")
        random_mask_ratio: float = 0.5,
        num_forecast_mask_patches=(2,),
        channel_consistent_masking: bool = True,
        mask_value: float = 0,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1
        self.scaler = PatchTSMixerScaler(scaling) if scaling else None
        self.mask_type = mask_type
        self.random_mask_ratio = random_mask_ratio
        self.num_forecast_mask_patches = num_forecast_mask_patches
        self.channel_consistent_masking = channel_consistent_masking
        self.mask_value = mask_value

        self.embedding = PatchTSMixerEmbedding(
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
        )

        self.pos_encoder = (
            PatchTSMixerPositionalEncoding(num_patches=self.num_patches, d_model=d_model)
            if use_positional_encoding
            else None
        )

        layer_kwargs = {
            "num_patches": self.num_patches,
            "d_model": d_model,
            "num_channels": num_channels,
            "mode": mode,
            "expansion": expansion,
            "dropout": dropout,
            "use_gated_attn": use_gated_attn,
            "eps": eps,
        }
        self.encoder = PatchTSMixerBlock(num_layers=num_layers, layer_kwargs=layer_kwargs)

        # Pre-training head reconstructs patches
        self.head = PatchTSMixerPretrainHead(
            d_model=d_model,
            patch_length=patch_length,
            head_dropout=head_dropout,
        )

    def forward(self, past_values, mask=None):
        """
        past_values: (B, L, C)
        mask: bool (B, C, Np) of the patches to hide; drawn from mask_type when None
        returns: reconstruction (B, C, Np, patch_length) in the scaled units, mask
        """
        if self.scaler is not None:
            past_values, _, _ = self.scaler(past_values)
        if mask is None and self.mask_type is not None:
            mask = patch_mask(
                self.mask_type,
                past_values.shape[0],
                self.num_channels,
                self.num_patches,
                random_mask_ratio=self.random_mask_ratio,
                num_forecast_mask_patches=self.num_forecast_mask_patches,
                channel_consistent_masking=self.channel_consistent_masking,
            )

        # (B, L, C) -> (B, C, L)
        x = past_values.transpose(1, 2)

        x = self.embedding(x, mask=mask, mask_value=self.mask_value)  # (B, C, Np, D)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x, _ = self.encoder(x)  # Unpack tuple
        reconstructed = self.head(x)  # (B, C, Np, patch_length)
        return reconstructed, mask


def hf_state_dict_to_port(state_dict, task="forecasting"):
    """
    Rename a Hugging Face PatchTSMixerForPrediction / PatchTSMixerForPretraining state dict to this module's keys.
    """
    embed, pos, enc, hf_head, head = {
        "forecasting": ("patch_embed", "pos_enc", "mixer_block", "base_forecast_block", "proj"),
        "pretraining": ("embedding", "pos_encoder", "encoder", "base_pt_block", "projection"),
    }[task]
    renames = (
        ("model.encoder.patcher.", f"{embed}.proj."),
        ("model.encoder.positional_encoder.position_enc", f"{pos}.pe"),
        ("model.encoder.mlp_mixer_encoder.mixers.", f"{enc}.layers."),
        (".channel_feature_mixer.", ".channel_mixer."),
        (".gating_block.", ".gate."),
        (f"head.{hf_head}.", f"head.{head}."),
    )
    out = {}
    for key, value in state_dict.items():
        for old, new in renames:
            key = key.replace(old, new)
        out[key] = value
    return out


def from_hf_config(config, state_dict=None, task="forecasting"):
    """
    Build the forecasting or pretraining model from a Hugging Face PatchTSMixerConfig dict
    (PatchTSMixerConfig.to_dict() or config.json) and load its state dict when given.
    """
    if config.get("self_attn") or config.get("norm_mlp", "LayerNorm") != "LayerNorm":
        raise ValueError("self_attn and norm_mlp other than LayerNorm are not supported")
    kwargs = dict(
        context_length=config["context_length"],
        patch_length=config["patch_length"],
        patch_stride=config["patch_stride"],
        num_channels=config["num_input_channels"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        mode=config.get("mode", "common_channel"),
        expansion=config.get("expansion_factor", 2),
        dropout=config.get("dropout", 0.2),
        use_gated_attn=config.get("gated_attn", True),
        head_dropout=config.get("head_dropout", 0.2),
        eps=config.get("norm_eps", 1e-5),
        scaling=config.get("scaling", "std"),
        use_positional_encoding=config.get("use_positional_encoding", False),
    )
    if task == "forecasting":
        if config.get("loss", "mse") != "mse" or config.get("prediction_channel_indices") is not None:
            raise ValueError("only loss='mse' with all channels predicted is supported")
        model = PatchTSMixerModelForForecasting(prediction_length=config["prediction_length"], **kwargs)
    elif task == "pretraining":
        if config.get("unmasked_channel_indices") is not None:
            raise ValueError("unmasked_channel_indices is not supported")
        model = PatchTSMixerForPretraining(
            mask_type=config.get("mask_type", "random"),
            random_mask_ratio=config.get("random_mask_ratio", 0.5),
            num_forecast_mask_patches=config.get("num_forecast_mask_patches", (2,)),
            channel_consistent_masking=config.get("channel_consistent_masking", True),
            mask_value=config.get("mask_value", 0),
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported task {task}")
    if state_dict is not None:
        model.load_state_dict(hf_state_dict_to_port(state_dict, task))
    return model.eval()


def load_hf_checkpoint(name_or_path):
    """
    Read config.json and model.safetensors of a Hugging Face checkpoint from a local directory or the hub.
    Returns (config dict, state dict) for from_hf_config.
    """
    from safetensors.torch import load_file

    path = name_or_path
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download

        path = snapshot_download(name_or_path, allow_patterns=["config.json", "model.safetensors"])
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    return config, load_file(os.path.join(path, "model.safetensors"))
