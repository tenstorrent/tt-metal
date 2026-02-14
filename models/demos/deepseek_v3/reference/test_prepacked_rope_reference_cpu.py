import math
from dataclasses import dataclass

import torch


@dataclass
class ModelArgs:
    original_seq_len: int = 32
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    qk_rope_head_dim: int = 4
    max_seq_len: int = 64  # 128K bank


def precompute_freqs_cis_original(args: ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


def apply_rotary_emb_packed(x: torch.Tensor, freqs_cis_bank: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    x: [batch, total_seq_len, n_heads, dim]
    freqs_cis_bank: [128k, dim // 2]
    positions: [batch, total_seq_len] -> contains local indices (e.g. 0, 1, 2, 0, 1...)
    """
    dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))

    selected_freqs = freqs_cis_bank[positions].unsqueeze(2)

    y = torch.view_as_real(x_complex * selected_freqs).flatten(3)
    return y.to(dtype)


def run_test():
    args = ModelArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Precomputing 128K bank...")
    global_bank = precompute_freqs_cis_original(args).to(device)

    lengths = [8, 16, 40]
    seqs = [torch.randn(1, l, 1, args.qk_rope_head_dim, device=device) for l in lengths]

    ground_truth_outputs = []
    for s in seqs:
        slen = s.size(1)
        freqs_slice = global_bank[0:slen]
        ground_truth_outputs.append(apply_rotary_emb(s, freqs_slice))

    packed_x = torch.cat(seqs, dim=1)  # [1, 8192, 1, 64]
    pos_map = torch.cat([torch.arange(l) for l in lengths]).unsqueeze(0).to(device)
    packed_output = apply_rotary_emb_packed(packed_x, global_bank, pos_map)

    print("\nVerification:")
    curr_ptr = 0
    for i, l in enumerate(lengths):
        truth = ground_truth_outputs[i]
        test_slice = packed_output[:, curr_ptr : curr_ptr + l, :, :]

        max_err = torch.max(torch.abs(truth - test_slice)).item()
        print(f"Sub-sequence {i} (len {l}): Max Error = {max_err:.2e}")
        assert max_err < 1e-6
        curr_ptr += l

    print("\nSuccess! The packed implementation matches the original bank-slicing logic.")


if __name__ == "__main__":
    run_test()
