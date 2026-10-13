# The input classes of ab_dump.py, importable without a device.
import torch


def bf16_from_bits(bits):
    return bits.to(torch.int16).view(torch.bfloat16)


def all_patterns(perm_seed):
    bits = torch.arange(65536, dtype=torch.int32)
    if perm_seed == "rev":
        bits = bits.flip(0)
    elif perm_seed is not None:
        g = torch.Generator().manual_seed(perm_seed)
        bits = bits[torch.randperm(65536, generator=g)]
    return bf16_from_bits(bits).reshape(1, 1, 32, 2048)


def classes():
    g = torch.Generator().manual_seed(7)
    out = {
        "allpat": all_patterns(None),
        "allpat_rev": all_patterns("rev"),
        "allpat_perm": all_patterns(11),
        "ties4": torch.randint(-1, 3, (1, 1, 32, 2048), generator=g).to(torch.bfloat16),
        "randn": torch.randn((1, 1, 32, 2048), generator=g).to(torch.bfloat16),
    }
    z = torch.zeros((1, 1, 32, 2048), dtype=torch.bfloat16)
    z[..., 1::2] = -0.0
    z[..., ::7] = 1.0
    z[..., ::11] = float("inf")
    z[..., ::13] = -float("inf")
    out["signed_zero_inf"] = z
    return out


