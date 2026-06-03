# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch

TILE = 32

def pad_to_tile(x):
    """Pad last dim of a TTNN tensor to multiple of 32, return (padded, original_last_dim)."""
    last_dim = x.shape[-1]
    if last_dim % TILE == 0:
        return x, last_dim
    pad = TILE - (last_dim % TILE)
    # Convert to torch, pad, convert back
    t = ttnn.to_torch(x).float()
    t = torch.nn.functional.pad(t, (0, pad))
    padded = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=x.device())
    return padded, last_dim

def unpad(x, original_last_dim):
    """Slice last dim back to original size."""
    if x.shape[-1] == original_last_dim:
        return x
    t = ttnn.to_torch(x).float()[..., :original_last_dim]
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=x.device())

def layer_norm_padded(x, weight, bias):
    """Run ttnn.layer_norm handling non-tile-multiple last dims."""
    x_padded, orig_dim = pad_to_tile(x)
    out = ttnn.layer_norm(x_padded, weight=weight, bias=bias)
    return unpad(out, orig_dim)
