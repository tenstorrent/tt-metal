import math
import torch
import ttnn


class RelPositionalEncodingTTNN:
    """TTNN relative positional encoding for RelPositionMultiHeadAttention (inference-friendly).

    Args:
        device: TTNN device.
        d_model (int): embedding dimension
        max_len (int): maximum sequence length (will allocate 2*max_len-1 positions)

    Forward:
        Args:
            x (torch.Tensor): (batch, time, d_model)
        Returns:
            pos_emb (ttnn.Tensor): (2*time-1, 1, d_model) by default.
                If return_two_values=True, returns (x, pos_emb) to match NeMo API.
    """

    def __init__(self, device, d_model: int, max_len: int = 5000, return_two_values: bool = False):
        self.device = device
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
        self.pe = pe  # shape: (2*max_len-1, d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, time, d_model)
        Returns:
            pos_emb: (2*time-1, 1, d_model) by default; or (x, pos_emb) if return_two_values.
        """
        T = x.shape[1] if hasattr(x, "shape") else x.size(1)
        B = x.shape[0] if hasattr(x, "shape") else x.size(0)
        # Center slice for current sequence length: central 2*T-1 positions
        start = max(0, self.max_len - T)
        end = min(self.pe.size(0), start + 2 * T - 1)  # ensure we don't exceed pe length
        # (2*T-1, d_model) -> (1, 2*T-1, d_model) for batch=1
        pos_emb_torch = self.pe[start:end].unsqueeze(0)  # (1, 2*T-1, d_model)

        # Move to TTNN device with TILE layout for downstream ops
        pos_emb = ttnn.from_torch(
            pos_emb_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        if self.return_two_values:
            # Mimic NeMo API: return (x, pos_emb)
            return x, pos_emb
        else:
            return pos_emb

    def __call__(self, x):
        return self.forward(x)
