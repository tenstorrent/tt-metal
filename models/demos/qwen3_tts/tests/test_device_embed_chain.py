# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validate the on-device chain logits → argmax → ttnn.embedding → bf16 TILE
against the CPU reference (argmax + F.embedding). PCC must be > 0.99 to confirm
we can fuse this chain inside the CP decode trace without breaking the model.

    pytest -q models/demos/qwen3_tts/tests/test_device_embed_chain.py
"""
import pytest
import torch

import ttnn


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a_c = a - a.mean()
    b_c = b - b.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-12)).item()


def test_device_embed_chain(device):
    """logits → ttnn.argmax → ttnn.embedding == CPU torch.argmax + F.embedding."""
    torch.manual_seed(0)
    vocab = 2048
    hidden = 2048  # matches CodePredictor codec_embedding output dim

    # Random embedding table.
    embed_table = torch.randn(vocab, hidden, dtype=torch.bfloat16) * 0.05
    # Random logits with one strongly-biased token to make the argmax deterministic.
    logits = torch.randn(1, 1, 1, vocab, dtype=torch.bfloat16) * 0.1
    target_idx = 1234
    logits[0, 0, 0, target_idx] = 100.0

    # CPU reference
    ref_token = int(logits.float().squeeze().argmax().item())
    ref_embed = embed_table[ref_token].float()  # [hidden]

    # Device chain — logits come out of matmul as TILE; use untilize_with_unpadding
    # to drop the tile padding on the Y dim (real Y=1, padded to 32).
    logits_tt = ttnn.from_torch(
        logits, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    logits_rm = ttnn.untilize_with_unpadding(logits_tt, output_tensor_end=[0, 0, 0, vocab - 1], use_multicore=True)
    out_tok_tt = ttnn.argmax(logits_rm, dim=-1, keepdim=False, use_multicore=True)

    # Embedding table on device — match CodePredictor's storage shape [1, 1, vocab, hidden].
    embed_table_tt = ttnn.from_torch(
        embed_table.unsqueeze(0).unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    embed_table_4d = ttnn.reshape(embed_table_tt, [1, 1, vocab, hidden], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # ttnn.embedding wants the input as a 1D or 2D index tensor.
    # out_tok_tt shape is [1, 1, 1] — pass as-is and ttnn.embedding produces
    # [1, 1, 1, hidden] in the requested layout.
    embed_out_tt = ttnn.embedding(
        out_tok_tt,
        embed_table_4d,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    print(f"out_tok shape={out_tok_tt.shape}, embed_out shape={embed_out_tt.shape} dtype={embed_out_tt.dtype}")

    ttnn.synchronize_device(device)
    # Verify token id matches CPU argmax.
    out_tok_torch = ttnn.to_torch(out_tok_tt)
    print(f"out_tok torch: {out_tok_torch} (dtype={out_tok_torch.dtype}, shape={out_tok_torch.shape})")
    sampled = int(out_tok_torch.flatten()[0].item())
    assert sampled == ref_token == target_idx, f"argmax mismatch: device={sampled}, ref={ref_token}"

    # Verify embedding row matches.
    embed_out = ttnn.to_torch(embed_out_tt).squeeze().float()  # [hidden]
    pcc = pearson(ref_embed, embed_out)
    print(f"Embed PCC: {pcc:.6f}")
    assert pcc > 0.99, f"Embed PCC too low: {pcc}"
