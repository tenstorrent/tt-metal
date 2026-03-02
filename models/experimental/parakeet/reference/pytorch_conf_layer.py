import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
except ImportError:
    CausalConv1D = None  # fallback if NeMo not available


# -------------------------------------------------
# Helper: Swish activation
# -------------------------------------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# -------------------------------------------------
# Relative positional encoding (simplified)
# -------------------------------------------------
import math
import torch
import torch.nn as nn


class RelPositionalEncoding(nn.Module):
    """Relative positional encoding for RelPositionMultiHeadAttention (inference-friendly).

    Args:
        d_model (int): embedding dimension
        max_len (int): maximum sequence length (will allocate 2*max_len-1 positions)

    Forward:
        Args:
            x (torch.Tensor): (batch, time, d_model)
        Returns:
            pos_emb (torch.Tensor): (2*time-1, 1, d_model) by default.
                If return_two_values=True, returns (x, pos_emb) to match NeMo API.
    """

    def __init__(self, d_model: int, max_len: int = 5000, return_two_values: bool = False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.return_two_values = return_two_values
        self._build_pe()

    def _build_pe(self):
        """Build sinusoidal buffer for relative positions: -(max_len-1) .. (max_len-1)."""
        pe_len = 2 * self.max_len - 1
        pe = torch.zeros(pe_len, self.d_model)
        # Relative positions from -(max_len-1) to (max_len-1)
        position = torch.arange(-(self.max_len - 1), self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # shape: (2*max_len-1, d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, time, d_model)
        Returns:
            pos_emb: (2*time-1, 1, d_model) by default; or (x, pos_emb) if return_two_values.
        """
        T = x.size(1)
        # Center slice for current sequence length: central 2*T-1 positions
        start = max(0, self.max_len - T)
        end = min(self.pe.size(0), start + 2 * T - 1)  # ensure we don't exceed pe length
        pos_emb = self.pe[start:end].unsqueeze(0)  # (1, 2*T-1, d_model)  ← correct

        if self.return_two_values:
            # Mimic NeMo API: return (x, pos_emb)
            return x, pos_emb
        else:
            return pos_emb


# -------------------------------------------------
# Simplified RelPositionMultiHeadAttention (inference)
# -------------------------------------------------
class RelPositionMultiHeadAttention(nn.Module):
    """Multi-Head Attention with relative positional encoding (Transformer-XL style).
    Matches NeMo RelPositionMultiHeadAttention: pos_bias_u/v, linear_pos, rel_shift.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, pos_bias_u=None, pos_bias_v=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.s_d_k = math.sqrt(self.head_dim)

        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)

        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.head_dim))
            self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x):
        """Relative shift. Input: (B, H, T, 2*T-1). Output: (B, H, T, 2*T-1) then trim to (B, H, T, T)."""
        b, h, qlen, pos_len = x.size()
        x = torch.nn.functional.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        x = x[:, :, 1:].view(b, h, qlen, pos_len)
        return x

    def forward(self, x, pos_emb, mask=None):
        """Args:
            x: (B, T, d_model)
            pos_emb: (B, 2*T-1, d_model) or (1, 2*T-1, d_model) from RelPositionalEncoding
            mask: (B, T, T) bool (True = mask out)
        Returns:
            out: (B, T, d_model)
        """
        B, T, _ = x.size()
        q = self.linear_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.linear_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # pos_emb length must be 2*T-1 for NeMo/Transformer-XL
        p = self.linear_pos(pos_emb)
        if p.dim() == 3:
            p = p.squeeze(0)  # (2*T-1, d_model) -> then view
        n_pos = p.size(0)
        p = p.view(n_pos, self.n_heads, self.head_dim).transpose(0, 1)  # (H, 2*T-1, D)
        p = p.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, 2*T-1, D)

        q_with_bias_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)  # (B, H, T, D)
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))  # (B, H, T, T)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))  # (B, H, T, 2*T-1)
        matrix_bd = self.rel_shift(matrix_bd)  # (B, H, T, 2*T-1)
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]  # (B, H, T, T)

        scores = (matrix_ac + matrix_bd) / self.s_d_k

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -10000.0)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.linear_out(out)
        return out


# -------------------------------------------------
# ConformerFeedForward (inference-only)
# -------------------------------------------------
class ConformerFeedForward(nn.Module):
    """Feed-forward module: Linear -> Swish -> Linear (no dropout)."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.activation = Swish()
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


# -------------------------------------------------
# ConformerConvolution (inference-only, matches NeMo)
# -------------------------------------------------
class ConformerConvolution(nn.Module):
    """Convolution module: pointwise -> GLU -> depthwise -> BN -> Swish -> pointwise.
    Matches NeMo: CausalConv1D for depthwise (with conv_context_size for padding), use_bias on convs.
    """

    def __init__(self, d_model, kernel_size=31, use_bias=True):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        conv_context_size = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1, stride=1, bias=use_bias)
        # After GLU we have d_model channels; depthwise matches NeMo (CausalConv1D with conv_context_size)
        if CausalConv1D is not None:
            self.depthwise_conv = CausalConv1D(
                d_model,
                d_model,
                kernel_size=kernel_size,
                stride=1,
                padding=conv_context_size,
                groups=d_model,
                bias=use_bias,
            )
        else:
            self.depthwise_conv = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=kernel_size,
                stride=1,
                padding=conv_context_size,
                groups=d_model,
                bias=use_bias,
            )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, bias=use_bias)

    def forward(self, x, pad_mask=None):
        """
        Args:
            x: (B, T, d_model)
            pad_mask: (B, T) bool mask (True for padding)
        Returns:
            out: (B, T, d_model)
        """
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)  # (B, d_model, T)

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)  # (B, T, d_model)
        return x


# -------------------------------------------------
# ConformerLayer (inference-only)
# -------------------------------------------------
class ConformerLayer(nn.Module):
    """Single Conformer block: FF1 -> MHA -> Conv -> FF2 (Macaron style)."""

    def __init__(self, d_model=1024, d_ff=4096, n_heads=8, conv_kernel_size=31):
        super().__init__()
        self.fc_factor = 0.5

        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model, d_ff)

        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(d_model, n_heads)

        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(d_model, conv_kernel_size)

        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model, d_ff)

        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, att_mask=None, pad_mask=None):
        """
        Args:
            x: (B, T, d_model)
            pos_emb: (T, 1, d_model) from RelPositionalEncoding
            att_mask: (B, T, T) bool mask for attention
            pad_mask: (B, T) bool mask for padding
        Returns:
            out: (B, T, d_model)
        """
        # FF1
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + x * self.fc_factor

        # MHA
        x = self.norm_self_att(residual)
        x = self.self_attn(x, pos_emb, mask=att_mask)
        residual = residual + x

        # Conv
        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask)
        residual = residual + x

        # FF2
        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + x * self.fc_factor

        # Output norm
        x = self.norm_out(residual)
        return x


# -------------------------------------------------
# ConformerEncoder (stack of layers, inference-only)
# -------------------------------------------------
class ConformerEncoder(nn.Module):
    """Stack of ConformerLayer with positional encoding."""

    def __init__(self, d_model=1024, d_ff=4096, n_heads=8, conv_kernel_size=31, n_layers=24):
        super().__init__()
        self.pos_enc = RelPositionalEncoding(d_model)
        self.layers = nn.ModuleList([ConformerLayer(d_model, d_ff, n_heads, conv_kernel_size) for _ in range(n_layers)])

    def forward(self, x, lengths=None):
        """
        Args:
            x: (B, T, d_model) e.g., output from ConvSubsampling
            lengths: (B,) optional lengths for mask creation
        Returns:
            out: (B, T, d_model)
        """
        # Create masks if lengths provided
        pad_mask = None
        att_mask = None
        if lengths is not None:
            B, T, _ = x.size()
            pad_mask = torch.arange(T, device=x.device).expand(B, T) >= lengths.unsqueeze(1)
            # Attention mask: allow valid positions to attend; mask padding
            att_mask = pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)

        pos_emb = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, pos_emb, att_mask=att_mask, pad_mask=pad_mask)
        return x


# -------------------------------------------------
# Example usage (inference)
# -------------------------------------------------
if __name__ == "__main__":
    from models.experimental.parakeet.reference.pytorch_pre_enc import ConvSubsampling

    device = torch.device("cpu")
    # Example shapes matching your earlier debug
    batch, time_in, feat_in = 1, 744, 128
    x = torch.randn(batch, time_in, feat_in, device=device)
    lengths = torch.tensor([743], device=device)

    # Pre-encode (your ConvSubsampling)
    pre_encode = ConvSubsampling(feat_in=feat_in, feat_out=1024, conv_channels=256).to(device)
    x, lengths = pre_encode(x, lengths)  # -> (B, T//8, 1024)

    # Conformer encoder stack (24 layers)
    encoder = ConformerEncoder(d_model=1024, d_ff=4096, n_heads=8, conv_kernel_size=31, n_layers=24).to(device)
    encoder.eval()
    with torch.no_grad():
        out = encoder(x, lengths)
    print("Encoded output shape:", out.shape)  # (B, T//8, 1024)
