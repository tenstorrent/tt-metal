# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.t3000.llama2_70b.reference.llama.llama.model import apply_rotary_emb
from models.tt_transformers.tt.common import get_rot_transformation_mat
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

MAX_SEQ_LEN = 128 * 1024


@dataclass
class ModelArgs:
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    qk_rope_head_dim: int = 128
    max_seq_len: int = MAX_SEQ_LEN


def precompute_freqs_cis_bank(dim: int, args: ModelArgs):
    """Precomputes the 128K bank with DeepSeek V3 YaRN scaling."""
    seqlen = args.max_seq_len
    base = args.rope_theta
    factor = args.rope_factor

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # Apply YaRN
    if seqlen > args.original_seq_len:

        def find_correction_dim(num_rotations):
            return dim * math.log(args.original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

        low = math.floor(find_correction_dim(args.beta_fast))
        high = math.ceil(find_correction_dim(args.beta_slow))
        ramp = torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low if high > low else 0.001), 0, 1
        )
        smooth = 1.0 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return freqs  # Returning raw freqs to allow cos/sin derivation


def compute_packed_cos_sin(freqs_bank, position_ids):
    """
    Gather pre-selected frequencies based on packed position_ids
    and format them for ttnn.experimental.rotary_embedding_llama.
    """
    # Gather frequencies: [total_seq_len, head_dim // 2]
    selected_freqs = freqs_bank[position_ids]

    cos = selected_freqs.cos()
    sin = selected_freqs.sin()

    # TTNN doubled format: [c0, c0, c1, c1, ...]
    # Shape: [1, 1, total_seq_len, head_dim]
    cos_ttnn = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin_ttnn = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)

    return cos_ttnn, sin_ttnn


def run_test_rotary_packed(
    device,
    lengths,  # List of subsequence lengths, e.g., [1024, 2048]
    n_heads,
    head_dim,
    pcc,
    datatype=ttnn.bfloat16,
):
    args = ModelArgs(qk_rope_head_dim=head_dim)
    total_seq_len = sum(lengths)
    freqs_bank = precompute_freqs_cis_bank(head_dim, args)

    torch_input = torch.randn(1, n_heads, total_seq_len, head_dim)
    position_ids = torch.cat([torch.arange(l) for l in lengths])

    cos_torch, sin_torch = compute_packed_cos_sin(freqs_bank, position_ids)

    pytorch_outs = []
    curr = 0
    for l in lengths:
        sub_x = torch_input[:, :, curr : curr + l, :].transpose(1, 2)
        f_cis = torch.polar(torch.ones_like(freqs_bank[:l]), freqs_bank[:l])
        out_q, _ = apply_rotary_emb(sub_x, sub_x, freqs_cis=f_cis)
        pytorch_outs.append(out_q.transpose(1, 2))
        curr += l
    pytorch_gt = torch.cat(pytorch_outs, dim=2)

    transformation_mat = ttnn.from_torch(
        get_rot_transformation_mat(dhead=ttnn.TILE_SIZE), device=device, layout=ttnn.TILE_LAYOUT, dtype=datatype
    )

    tt_x = ttnn.from_torch(torch_input, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)
    tt_cos = ttnn.from_torch(cos_torch, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)
    tt_sin = ttnn.from_torch(sin_torch, device=device, dtype=datatype, layout=ttnn.TILE_LAYOUT)

    tt_out = ttnn.experimental.rotary_embedding_llama(tt_x, tt_cos, tt_sin, transformation_mat, is_decode_mode=False)

    tt_out_torch = ttnn.to_torch(tt_out)

    # 7. Comparison
    passing, output_pcc = comp_pcc(pytorch_gt, tt_out_torch, pcc)
    logger.info(f"PCC: {output_pcc}")
    assert passing


@pytest.mark.parametrize("lengths", ([1024, 2048, 1024], [32, 64], [4096]))
@pytest.mark.parametrize("head_dim", (64, 128))
def test_packed_rotary(device, lengths, head_dim):
    run_test_rotary_packed(device=device, lengths=lengths, n_heads=8, head_dim=head_dim, pcc=0.999)
