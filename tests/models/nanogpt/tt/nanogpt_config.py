from torch import nn


class GPTConfig(nn.Module):

    model_type = "gpt2"

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304, # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
