import torch
from libs import tt_lib as ttm
import python_api_testing.models.bloom_new.bloom_utils as bloom_utils
from libs.tt_lib.fallback_ops import fallback_ops


def merge_heads(x: torch.Tensor, num_heads, hidden_size, num_attention_heads) -> torch.Tensor:

    head_dim = hidden_size // num_attention_heads

    batch_size_and_num_heads, seq_length, _ = x.shape
    batch_size = batch_size_and_num_heads // num_heads

    x = x.view(batch_size, num_heads, seq_length, head_dim)
    x = x.permute(0, 2, 1, 3)

    return x.reshape(1, batch_size, seq_length, num_heads * head_dim)

def tt_merge_heads(x: torch.Tensor, num_heads, hidden_size, num_attention_heads, device) -> torch.Tensor:

    head_dim = hidden_size // num_attention_heads

    batch_size_and_num_heads, seq_length, _ = x.shape
    batch_size = batch_size_and_num_heads // num_heads

    tt_x = bloom_utils.torch2tt_tensor(x, device)
    tt_reshaped = fallback_ops.reshape(tt_x, batch_size, num_heads, seq_length, head_dim)

    # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
    tt_permuted = ttm.tensor.permute(tt_reshaped, 0, 2, 1, 3)

    # reshape - fallback
    reshaped_2 = fallback_ops.reshape(tt_permuted, 1, batch_size, seq_length, num_heads*head_dim)

    return reshaped_2
