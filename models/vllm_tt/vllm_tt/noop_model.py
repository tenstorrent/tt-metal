import torch
import torch.nn as nn
from vllm.config import VllmConfig


class NoOpModel(nn.Module):
    """No-op model for testing the TT plugin stack.

    Produces random logits without any real computation.
    Used to validate the plugin infrastructure end-to-end.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        # Dummy parameter so PyTorch considers this a valid module
        self.dummy = nn.Parameter(torch.zeros(1))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        return torch.randn(batch_size, self.vocab_size, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # forward() already returns logits directly
        return hidden_states

    def load_weights(self, weights):
        pass
