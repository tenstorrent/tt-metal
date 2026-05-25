"""
TTNN-ported versions of modules.py (LinearNorm, LayerNorm, AdaLayerNorm,
TextEncoder, DurationEncoder, ProsodyPredictor, CustomAlbert).

TTNN ops used:
  nn.Linear       → ttnn.linear  (via tt_utils.tt_linear)
  nn.LayerNorm    → ttnn.layer_norm (via tt_utils.tt_layer_norm)
  Activation fns  → ttnn.leaky_relu / ttnn.gelu / ttnn.tanh / ttnn.sigmoid
  nn.Embedding    → ttnn.embedding  (or torch fallback — see TTTextEncoder)

Torch fallbacks (with reason):
  nn.LSTM         → torch: TTNN has no bidirectional LSTM op
  nn.Conv1d       → torch: dynamic seq-len makes ttnn.conv2d batch_size/H awkward;
                    1×1 convs handled via tt_linear instead
  F.interpolate   → torch: no TTNN interpolation op
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


from .tt_utils import (
    load_tt_linear,
    load_tt_weight,
    tt_layer_norm,
    tt_linear,
)
from .tt_istftnet import TTAdainResBlk1d


# ─────────────────────────────────────────────
# TTLinearNorm
# ─────────────────────────────────────────────


class TTLinearNorm(nn.Module):
    """Port of LinearNorm — nn.Linear with xavier init (init done in ref only)."""

    def __init__(self, linear: nn.Linear, device):
        super().__init__()
        self.device = device
        self.out_features = linear.out_features
        self.weight_tt, self._bias, _ = load_tt_linear(linear, device)  # _bias is torch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        return tt_linear(x, self.weight_tt, self._bias, self.out_features, self.device)


# ─────────────────────────────────────────────
# TTLayerNorm
# ─────────────────────────────────────────────


class TTLayerNorm(nn.Module):
    """
    Port of LayerNorm from modules.py.

    Reference forward:
        x = x.transpose(1, -1)          # (B, C, T) → (B, T, C)
        x = F.layer_norm(x, (C,), γ, β, ε)
        return x.transpose(1, -1)       # (B, T, C) → (B, C, T)
    """

    def __init__(self, channels: int, eps: float, gamma: torch.Tensor, beta: torch.Tensor, device):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.device = device
        # Stored as plain torch tensors — tt_layer_norm applies them in torch
        # to avoid TTNN tile-padding shape mismatches.
        self.register_buffer("gamma", gamma.detach().float())
        self.register_buffer("beta", beta.detach().float())

    @classmethod
    def from_ref(cls, ref_ln, device):
        return cls(ref_ln.channels, ref_ln.eps, ref_ln.gamma, ref_ln.beta, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = x.transpose(1, -1)  # (B, T, C)
        x = tt_layer_norm(x, self.gamma, self.beta, self.eps, self.device)
        return x.transpose(1, -1)  # (B, C, T)


# ─────────────────────────────────────────────
# TTAdaLayerNorm
# ─────────────────────────────────────────────


class TTAdaLayerNorm(nn.Module):
    """
    Port of AdaLayerNorm from modules.py.

    Reference:
        h = self.fc(s)                    # style → 2*channels
        h = h.view(B, 2*C, 1)
        gamma, beta = chunk(h, 2, dim=1)
        gamma = gamma.transpose(1,-1)     # (B,1,C)
        beta  = beta.transpose(1,-1)
        x = x.transpose(-1,-2).transpose(1,-1)   # (B,C,T) → (B,T,C)
        x = F.layer_norm(x, (C,), eps=ε)  # no affine params in norm itself
        x = (1 + gamma) * x + beta
        return x.transpose(1,-1).transpose(-1,-2)
    """

    def __init__(self, style_dim: int, channels: int, eps: float, fc: nn.Linear, device):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.device = device
        self.fc_w, self.fc_b, self.fc_out = load_tt_linear(fc, device)

    @classmethod
    def from_ref(cls, ref_ada, device):
        return cls(
            ref_ada.fc.in_features,
            ref_ada.channels,
            ref_ada.eps,
            ref_ada.fc,
            device,
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) — caller does x.transpose(-1,-2) before passing here
        # (see DurationEncoder: block(x.transpose(-1,-2), style).transpose(-1,-2))
        # The two transposes in the reference forward are a no-op on (B,T,C) input.
        h = tt_linear(s, self.fc_w, self.fc_b, self.fc_out, self.device)  # (B, 2*C)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma = gamma.transpose(1, -1)  # (B, 1, C)
        beta = beta.transpose(1, -1)
        # x is already (B, T, C); double-transpose from reference is a no-op.
        x_t = x.float()
        # ttnn.layer_norm normalises over last dim C
        x_t = tt_layer_norm(x_t, None, None, self.eps, self.device)
        x_t = (1.0 + gamma) * x_t + beta
        # Match reference return: (B, T, C)
        return x_t


# ─────────────────────────────────────────────
# TTTextEncoder
# ─────────────────────────────────────────────


class TTTextEncoder(nn.Module):
    """
    Port of TextEncoder from modules.py.

    Conv1d stack: torch fallback (dynamic L, kernel_size=5, dilation=1).
      TODO: replace with ttnn.conv2d(kernel=(5,1), stride=(1,1), padding=(2,0))
    LSTM: torch fallback — TTNN has no bidirectional LSTM.
    Embedding + output linear → torch (embedding table stays in CPU memory).
    """

    def __init__(self, ref_te, device):
        super().__init__()
        self.device = device
        # Embedding stays as torch (ttnn.embedding requires uint32 input_ids
        # and has non-trivial layout requirements; torch is sufficient here)
        self.embedding = ref_te.embedding
        # CNN blocks — keep as torch modules (Conv1d with dilation & LayerNorm)
        # TODO: replace Conv1d with ttnn.conv2d and TTLayerNorm
        self.cnn = ref_te.cnn
        # LSTM — torch fallback (TTNN has no bidirectional LSTM op)
        self.lstm = ref_te.lstm

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)  # (B, T, C)
        lengths = input_lengths if input_lengths.device == torch.device("cpu") else input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
        x_pad[:, :, : x.shape[-1]] = x
        x = x_pad
        x.masked_fill_(m, 0.0)
        return x


# ─────────────────────────────────────────────
# TTDurationEncoder
# ─────────────────────────────────────────────


class TTDurationEncoder(nn.Module):
    """
    Port of DurationEncoder from modules.py.

    Each LSTM uses torch fallback.
    Each AdaLayerNorm uses TTAdaLayerNorm (ttnn.layer_norm + ttnn.linear).
    """

    def __init__(self, ref_de, device):
        super().__init__()
        self.device = device
        self.d_model = ref_de.d_model
        self.sty_dim = ref_de.sty_dim
        self.dropout = ref_de.dropout

        self.lstms = nn.ModuleList()
        for block in ref_de.lstms:
            if isinstance(block, nn.LSTM):
                self.lstms.append(block)  # torch fallback
            else:
                # AdaLayerNorm → TTNN
                self.lstms.append(TTAdaLayerNorm.from_ref(block, device))

    def forward(
        self, x: torch.Tensor, style: torch.Tensor, text_lengths: torch.Tensor, m: torch.Tensor
    ) -> torch.Tensor:
        masks = m
        x = x.permute(2, 0, 1)  # (C, B, T) → (T, B, C)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)  # (B, T, C+sty)
        x = x.transpose(-1, -2)  # (B, C+sty, T)

        for block in self.lstms:
            if isinstance(block, TTAdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                lengths = text_lengths if text_lengths.device == torch.device("cpu") else text_lengths.cpu()
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, : x.shape[-1]] = x
                x = x_pad

        return x.transpose(-1, -2)


# ─────────────────────────────────────────────
# TTProsodyPredictor
# ─────────────────────────────────────────────


class TTProsodyPredictor(nn.Module):
    """
    Port of ProsodyPredictor from modules.py.

    LSTM sections: torch fallback.
    duration_proj (LinearNorm): ttnn.linear.
    F0/N resblocks: TTAdainResBlk1d.
    F0_proj / N_proj (Conv1d 1×1): tt_linear (equivalent math).
    """

    def __init__(self, ref_pp, device):
        super().__init__()
        self.device = device

        self.text_encoder = TTDurationEncoder(ref_pp.text_encoder, device)

        # torch fallback — TTNN has no bidirectional LSTM op
        self.lstm = ref_pp.lstm
        self.shared = ref_pp.shared

        self.duration_proj = TTLinearNorm(ref_pp.duration_proj.linear_layer, device)

        self.F0 = nn.ModuleList([TTAdainResBlk1d(b, device) for b in ref_pp.F0])
        self.N = nn.ModuleList([TTAdainResBlk1d(b, device) for b in ref_pp.N])

        # Conv1d(C, 1, 1) → equivalent to linear projection over channel dim
        self._build_proj(ref_pp.F0_proj, "F0_proj")
        self._build_proj(ref_pp.N_proj, "N_proj")

    def _build_proj(self, conv1d, name: str):
        """Conv1d(in_c, 1, kernel=1) stored as linear weight (in_c, 1)."""
        # conv1d.weight: (1, in_c, 1)
        w = conv1d.weight.detach().float().squeeze()  # (in_c,)
        w_tt = load_tt_weight(w.unsqueeze(1), self.device)  # (in_c, 1)
        b_tensor = conv1d.bias
        # Plain torch — tt_linear adds bias in torch
        b_tt = b_tensor.detach().float() if b_tensor is not None else None
        # Store as (weight_tt, bias_tt, out=1)
        setattr(self, f"_{name}_w", w_tt)
        setattr(self, f"_{name}_b", b_tt)
        setattr(self, f"_{name}_in", conv1d.in_channels)

    def _apply_proj(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Apply stored 1×1 conv projection via tt_linear."""
        # x: (B, C, T) — project C → 1
        w_tt = getattr(self, f"_{name}_w")
        b_tt = getattr(self, f"_{name}_b")
        x_t = x.transpose(1, 2)  # (B, T, C)
        out = tt_linear(x_t, w_tt, b_tt, 1, self.device)  # (B, T, 1)
        return out.transpose(1, 2)  # (B, 1, T)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        m = m.unsqueeze(1)
        lengths = text_lengths if text_lengths.device == torch.device("cpu") else text_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(d, lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]], device=x.device)
        x_pad[:, : x.shape[1], :] = x
        x = x_pad
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=False))
        en = d.transpose(-1, -2) @ alignment
        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self._apply_proj(F0, "F0_proj")
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self._apply_proj(N, "N_proj")
        return F0.squeeze(1), N.squeeze(1)


# ─────────────────────────────────────────────
# TTCustomAlbert
# ─────────────────────────────────────────────


class TTCustomAlbert(nn.Module):
    """
    Port of CustomAlbert from modules.py.

    The full Albert transformer runs in torch (CPU/CUDA).
    Porting the ~12-layer Albert attention stack to TTNN is feasible
    but out of scope for this initial port — it would require porting
    ttnn.transformer.attention for each layer.

    TODO: replace with TTNN multi-head attention blocks once verified.
    """

    def __init__(self, ref_albert, device):
        super().__init__()
        self.device = device
        # Keep the full HuggingFace AlbertModel as-is (torch)
        self.albert = ref_albert

    def forward(self, *args, **kwargs):
        # torch fallback — TTNN Albert not yet ported
        return self.albert(*args, **kwargs)
