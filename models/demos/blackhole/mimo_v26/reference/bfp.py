"""Host emulation of ttnn bfloat8_b (bfp8_b): blocks of 16 consecutive values along the LAST dim of the stored (tile) tensor share one exponent,
sign + 7-bit magnitude mantissa. value = sign * m/64 * 2**e, m in [0,127], e = floor(log2(max|x|)) of the block; round to nearest (even), saturate at 127.
Device weights are stored transposed ([K,N] = W.T), so the 16-groups run along the OUTPUT dimension of an [N,K] torch weight: use `quant_weight_bfp8`.
Validated against a real ttnn host roundtrip in tests/test_bfp.py when a saved reference (/tmp/.../bfp8_ref.pt) is available (see LOG)."""
import torch


def bfp8_roundtrip(x, group=16):
    shp = x.shape
    xf = x.float().reshape(-1, group) if shp[-1] % group == 0 else None
    assert xf is not None, "last dim must be a multiple of 16"
    mx = xf.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(mx.clamp_min(1e-38)))
    scale = torch.pow(2.0, e - 6)  # one mantissa step
    m = torch.round(xf / scale).clamp(-127, 127)
    out = torch.where(mx > 0, m * scale, torch.zeros_like(xf))
    return out.reshape(shp)


def quant_weight_bfp8(w):
    """w [N,K] torch Linear weight -> bfp8 roundtrip with groups of 16 along N (device stores W.T tiles; group = tile row = 16 consecutive N)."""
    return bfp8_roundtrip(w.float().T.contiguous()).T.contiguous().to(w.dtype)
