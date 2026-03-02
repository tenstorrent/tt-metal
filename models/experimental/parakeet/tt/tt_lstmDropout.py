import ttnn
import torch
from typing import Optional, Tuple, List, Dict
from models.experimental.parakeet.tt.tt_lstm import TtLSTMCell


class TT_LSTMDropout:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        device=None,
        memory_config=None,
        dtype=ttnn.float32,
        weights: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        ttnn-based LSTMDropout. Dropout uses ttnn.experimental.dropout.
        LSTM core is not yet available in ttnn; keep a placeholder or custom implementation.
        """
        self.device = device
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.dtype = dtype
        # Placeholder: ttnn does not provide an LSTM primitive yet.
        # You could replace this with a custom ttnn LSTM implementation or keep torch.nn.LSTM for now.
        self.lstm = TtLSTMCell(input_size, hidden_size, device, num_layers, dtype, memory_config, weights)

    def forward(
        self,
        x: ttnn.Tensor,
        h: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        c: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
    ):
        output, (h_tt, c_tt) = self.lstm(x, h)
        return output, (h_tt, c_tt)
