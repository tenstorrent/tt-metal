import torch


def bfp_quantize(w_kn, mant_bits):
    """TT BFP{8,4}_b emulation. w_kn [K,N] (K,N multiples of 32). Blocks = 16 contiguous along N (face row).
    Mirrors blockfloat_common.cpp convert_u32_to_bfp (non-truncating, RNE after truncating-align shift)."""
    shape = w_kn.shape
    x = w_kn.float().contiguous().reshape(-1, 16)
    b = x.view(torch.int32)
    exp = (b >> 23) & 0xFF
    sign = (b >> 31) & 1
    man = (b & 0x7FFFFF) | (1 << 23)
    man = torch.where(exp == 0, torch.zeros_like(man), man)  # zero/denormal -> 0 (and excluded below)
    smax = exp.max(dim=1, keepdim=True).values
    d = (smax - exp).clamp(max=31)
    man = man >> d  # truncating align (int32 shift, d<=31)
    man = torch.where((smax - exp) > 31, torch.zeros_like(man), man)
    S = 24 - mant_bits
    rv = man & ((1 << S) - 1)
    m = man >> S
    tie = 1 << (S - 1)
    up = (rv > tie) | ((rv == tie) & ((m & 1) == 1))
    m = torch.clamp(m + up.to(m.dtype), max=(1 << mant_bits) - 1)
    # decode: value = m * 2^(smax-127) * 2^-(mant_bits-1)
    scale = torch.ldexp(torch.ones_like(x), (smax - 127 - (mant_bits - 1)).to(torch.int32))
    out = m.float() * scale
    out = torch.where(sign.bool() & (m != 0), -out, out)
    out = torch.where(smax == 0, torch.zeros_like(out), out)
    return out.reshape(shape)
