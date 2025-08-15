import ttnn
import torch


# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
def create_4d_causal_attention_mask(
    input_shape: tuple[int, int], device: ttnn.Device, dtype: ttnn.DataType
) -> ttnn.Tensor:
    """Create a 4D causal attention mask for the given input shape."""
    batch_size, tgt_len = input_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len)
    return ttnn.from_torch(mask, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
