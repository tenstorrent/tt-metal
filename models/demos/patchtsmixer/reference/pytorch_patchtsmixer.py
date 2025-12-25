import math

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
                div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)))
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
        self, num_channels, d_model, norm_type="LayerNorm", expansion=2, dropout=0.1, gated_attn=False, eps=1e-5
    ):
        super().__init__()
        self.norm = PatchTSMixerNormLayer(d_model, norm_type=norm_type, eps=eps)
        self.gated_attn = gated_attn
        self.mlp = PatchTSMixerMLP(
            in_features=num_channels, out_features=num_channels, expansion=expansion, dropout=dropout
        )
        if gated_attn:
            self.gate = PatchTSMixerGatedAttention(d_model=num_channels)

    def forward(self, x):
        # x: (B, C, N_p, D)
        residual = x
        x = self.norm(x)  # (B, C, N_p, D)
        x = x.permute(0, 3, 2, 1)  # (B, D, N_p, C)

        if self.gated_attn:
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
                d_model=num_channels,
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

    def forward(self, x):
        """
        x: (B, C, L)
        HF expects (B, L, C) so adjust
        """
        x = x.transpose(1, 2)
        patches = self.patchify(x)  # (B, C, N_p, patch_len)

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
    ):
        super().__init__()

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
        self.pos_enc = PatchTSMixerPositionalEncoding(
            num_patches=num_patches,
            d_model=d_model,
            use_pe=True,
            pe_type="sincos",
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
        # (B, L, C) -> (B, C, L)
        x = past_values.transpose(1, 2)

        # 1) patchify & embed: (B, C, L) -> (B, C, N_p, D)
        x = self.patch_embed(x)

        # 2) positional encoding
        x = self.pos_enc(x)

        # 3) mixer stack: (B, C, N_p, D) -> (B, C, N_p, D)
        x, _ = self.mixer_block(x, output_hidden_states=False)

        # 4) head: (B, C, N_p, D) -> (B, H, C)
        preds = self.head(x)
        return preds
