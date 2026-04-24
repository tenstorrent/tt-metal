# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for: SDPA paged decode hang when worker nodes skip causal mask
on the partial last KV-cache page.

Bug (fixed in gpt-oss-hang-debug branch):
    In sdpa_flash_decode.cpp, the causal mask was applied only when
    `do_reduce && is_causal`. Worker nodes in the tree reduction have
    do_reduce=False, so they NEVER applied the causal mask.

    When cur_pos falls inside a KV-cache page (not at the end of a page),
    the remaining tokens in that page are zero-initialized. Without the causal
    mask, those zero-padded tokens contribute exp(Q*0)=exp(0)=1 to the softmax,
    producing incorrect (and potentially NaN) attention output.

    With 128 concurrent users all at the same cur_pos, ALL worker cores
    simultaneously encounter this, causing a coordinated device hang.

Fix:
    Change the loop condition from `k_chunk == k_chunk_end - 1` (this core's
    last chunk) to `k_chunk == k_num_chunks - 1` (the actual last chunk of
    the full KV sequence). This ensures only the core owning the partial last
    page applies the mask; other cores' valid intermediate pages are untouched.

Why this test is deterministic:
    - Uses a specific cur_pos that creates a partial last page:
        cur_pos = 2 * block_size + 3 = 131  (4 valid tokens in last page)
    - Forces num_cores_per_head > 1 by using a small sequence (3 pages) on
      a device with many cores — the op distributes one page per core,
      creating worker nodes (do_reduce=False) in the tree reduction.
    - The zero-padded tokens (positions 132-191) get non-zero attention weight
      without the fix, producing a measurably wrong output vs the CPU reference.
    - Single-device: the bug is in the compute kernel, not multi-chip fabric.

See: https://github.com/tenstorrent/tt-metal/issues/42917
"""

import pytest
import torch
import ttnn
from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import comp_pcc


def get_cpu_paged_sdpa_reference(Q, K_paged, V_paged, page_table, cur_pos, block_size, scale):
    """
    CPU reference for paged SDPA decode with correct causal masking.

    Reconstructs the full KV cache from paged format, computes attention with
    explicit masking of positions > cur_pos (including zero-padded page tail).
    """
    B, nkv, num_pages, blk, d = K_paged.shape
    max_seq = num_pages * block_size
    nh = Q.shape[2]

    # Reconstruct contiguous KV from pages
    K_cont = torch.zeros(B, nkv, max_seq, d)
    V_cont = torch.zeros(B, nkv, max_seq, d)
    for b in range(B):
        for pg, phys_pg in enumerate(page_table[b]):
            src_start = pg * block_size
            K_cont[b, :, src_start : src_start + block_size, :] = K_paged[b, :, phys_pg, :, :]
            V_cont[b, :, src_start : src_start + block_size, :] = V_paged[b, :, phys_pg, :, :]

    # Expand KV heads to match Q heads (GQA)
    heads_per_kv = nh // nkv
    K_exp = K_cont.repeat_interleave(heads_per_kv, dim=1)  # [B, nh, S, d]
    V_exp = V_cont.repeat_interleave(heads_per_kv, dim=1)  # [B, nh, S, d]

    # Q: [1, B, nh, d] → [B, nh, 1, d]
    q = Q.permute(1, 2, 0, 3)  # [B, nh, 1, d]

    attn_scores = torch.matmul(q * scale, K_exp.transpose(-1, -2))  # [B, nh, 1, S]

    # Causal mask: positions > cur_pos get -inf (zero-padded tokens included)
    mask = torch.full((1, 1, 1, max_seq), float("-inf"))
    mask[:, :, :, : cur_pos + 1] = 0.0
    attn_scores = attn_scores + mask

    attn_weights = torch.softmax(attn_scores, dim=-1)
    out = torch.matmul(attn_weights, V_exp)  # [B, nh, 1, d]
    return out.permute(2, 0, 1, 3)  # [1, B, nh, d]


@pytest.mark.timeout(120)
def test_sdpa_decode_paged_partial_mask_worker_node_regression(device):
    """
    Reproduces the bug where worker nodes (do_reduce=False) in the SDPA tree
    reduction skip the causal mask on the partial last KV-cache page.

    Trigger conditions (all required):
      1. Paged attention (block_size=64)
      2. cur_pos is NOT at a page boundary — partial last page has few valid tokens
      3. num_cores_per_head >= 2 — tree reduction active, worker nodes exist
      4. The core assigned to the partial last page is a worker (do_reduce=False)

    With the bug: worker skips mask → zero-padded tokens (positions 132–191)
                  get exp(0)=1 attention → wrong output, PCC < 0.99
    With the fix: all cores apply mask at k_chunk == k_num_chunks-1 → correct
    """
    torch.manual_seed(1234)

    # Small config that deterministically triggers the bug
    B, nkv, nh, d = 1, 1, 8, 128
    block_size = 64  # matches the failing GPT-OSS config
    #
    # cur_pos = 2*block_size + 3 = 131
    #   Page 0: tokens  0- 63  (full)
    #   Page 1: tokens 64-127  (full)
    #   Page 2: tokens 128-191 (partial: 4 valid = 128,129,130,131; 60 zeros)
    #
    # k_num_chunks = 3 pages → op assigns 1 page per core on WH 8x8 (64 cores)
    # Tree reduction: 1 root + 2 workers
    # Worker owning page 2 has do_reduce=False → misses the causal mask (the bug)
    cur_pos_val = 2 * block_size + 3  # = 131
    num_pages_per_seq = 4  # slightly more than needed, avoids edge-case sizing
    max_seq_len = num_pages_per_seq * block_size  # = 256

    scale = d**-0.5

    # Limit to a small (2,2) grid so the op uses 3 cores (one per KV page),
    # giving a tree with 1 root + 2 workers. Without this, on Galaxy all 32
    # chips × 64 cores = 2048 cores would be used, exceeding
    # MAX_TREE_REDUCTION_ROUNDS=6 (supports max 64 cores) and causing TT_FATAL.
    # Use a (8,4)=32 core grid with max_cores_per_head_batch=3.
    # This gives exactly 3 cores for k_num_chunks=3 pages:
    #   Core 0 (root, do_reduce=True):  page 0 (valid)
    #   Core 1 (worker, do_reduce=False): page 1 (valid)
    #   Core 2 (worker, do_reduce=False): page 2 (PARTIAL — 4 valid, 60 zeros)
    # grid_size.x=8 >= num_cores_per_head=3 satisfies the row constraint.
    # max_cores_per_head_batch=3 caps the cores so Galaxy's 2048 cores don't
    # exceed MAX_TREE_REDUCTION_ROUNDS=6.
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
        q_chunk_size=nh,
        k_chunk_size=block_size,
        max_cores_per_head_batch=3,
    )

    # === Random Q, K, V (ALL positions including padding) ===
    Q = torch.randn(1, B, nh, d)
    # Fill the ENTIRE page with random K/V — including the zero-padded tokens
    # (positions 132-191). The CPU reference masks them to -inf so they don't
    # contribute. The buggy kernel skips the mask, so those non-zero V values
    # corrupt the output — making PCC < 0.99 detectable.
    K_data = torch.randn(B, nkv, num_pages_per_seq * block_size, d)
    V_data = torch.randn(B, nkv, num_pages_per_seq * block_size, d)

    # === Build paged KV cache ===
    # Page table: sequential physical page allocation
    page_table = torch.arange(num_pages_per_seq, dtype=torch.int32).unsqueeze(0).expand(B, -1)

    # K/V paged: fill ALL positions with random data (no zeroing of padding)
    K_paged = K_data.view(B, nkv, num_pages_per_seq, block_size, d)
    V_paged = V_data.view(B, nkv, num_pages_per_seq, block_size, d)

    # === CPU reference (correct causal masking) ===
    ref = get_cpu_paged_sdpa_reference(Q, K_paged, V_paged, page_table, cur_pos_val, block_size, scale)

    # === TT device ===
    dram = ttnn.DRAM_MEMORY_CONFIG

    # Reshape for ttnn paged SDPA: K/V = [num_pages*B, nkv, block_size, d]
    K_tt_in = K_paged.squeeze(0)  # [nkv, num_pages, block_size, d] → needs [pages, nkv, blk, d]
    K_tt_in = K_paged.permute(0, 2, 1, 3, 4).reshape(B * num_pages_per_seq, nkv, block_size, d)
    V_tt_in = V_paged.permute(0, 2, 1, 3, 4).reshape(B * num_pages_per_seq, nkv, block_size, d)

    tt_Q = ttnn.as_tensor(Q, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_K = ttnn.as_tensor(K_tt_in, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_V = ttnn.as_tensor(V_tt_in, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_page_table = ttnn.as_tensor(
        page_table, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=dram
    )
    tt_cur_pos = ttnn.Tensor(torch.tensor([cur_pos_val] * B, dtype=torch.int32), ttnn.int32).to(device)

    tt_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_Q,
        tt_K,
        tt_V,
        cur_pos_tensor=tt_cur_pos,
        page_table_tensor=tt_page_table,
        scale=scale,
        program_config=program_config,
        memory_config=dram,
    )

    tt_out_torch = ttnn.to_torch(tt_out)

    # === Assertions ===
    # 1. No NaN — bug can produce NaN from exp(0)=1 dominating the softmax
    assert not torch.isnan(tt_out_torch).any(), (
        f"NaN in output at cur_pos={cur_pos_val} with block_size={block_size}. "
        f"Worker node skipped causal mask on partial last page "
        f"(page 2 has {(cur_pos_val+1) % block_size} valid tokens, "
        f"{block_size - (cur_pos_val+1) % block_size} zeros)."
    )

    # 2. PCC >= 0.99 vs CPU reference — bug causes measurable output corruption
    passing, pcc_val = comp_pcc(ref, tt_out_torch, pcc=0.99)
    assert passing, (
        f"PCC={pcc_val:.4f} < 0.99 at cur_pos={cur_pos_val} (block_size={block_size}). "
        f"Zero-padded tokens in the partial last page (positions "
        f"{cur_pos_val+1}-{(cur_pos_val//block_size+1)*block_size-1}) "
        f"received non-zero attention weight because the causal mask was not "
        f"applied on the worker node owning that page. "
        f"Fix: use k_chunk == k_num_chunks-1 instead of k_chunk == k_chunk_end-1 "
        f"in sdpa_flash_decode.cpp."
    )


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "cur_pos, block_size, description",
    [
        # Partial last page: 4 valid tokens out of 64
        (131, 64, "4_valid_of_64_in_last_page"),
        # Partial last page: 1 valid token out of 64 (most extreme)
        (128, 64, "1_valid_of_64_in_last_page"),
        # Partial last page: exactly at tile boundary (32 valid out of 64)
        (159, 64, "32_valid_of_64_last_page_tile_boundary"),
        # Original GPT-OSS-120B hang position
        (643, 64, "gpt_oss_120b_exact_hang_position"),
    ],
    ids=lambda x: x if isinstance(x, str) else str(x),
)
def test_sdpa_decode_paged_partial_mask_parametric(device, cur_pos, block_size, description):
    """
    Parametric version testing multiple cur_pos values that expose the partial
    last page masking bug. Each case has a different number of valid tokens
    in the last page.

    The GPT-OSS-120B production case (cur_pos=643) is included explicitly as
    the exact position where the hang was observed in CI.
    """
    torch.manual_seed(42 + cur_pos)

    B, nkv, nh, d = 1, 1, 8, 128
    scale = d**-0.5

    # Ensure enough pages
    num_pages = (cur_pos // block_size) + 2
    max_seq = num_pages * block_size

    Q = torch.randn(1, B, nh, d)
    K_data = torch.randn(B, nkv, cur_pos + 1, d)
    V_data = torch.randn(B, nkv, cur_pos + 1, d)

    page_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0).expand(B, -1)

    K_paged = torch.zeros(B, nkv, num_pages, block_size, d)
    V_paged = torch.zeros(B, nkv, num_pages, block_size, d)
    for tok in range(cur_pos + 1):
        pg, off = tok // block_size, tok % block_size
        K_paged[0, 0, pg, off, :] = K_data[0, 0, tok, :]
        V_paged[0, 0, pg, off, :] = V_data[0, 0, tok, :]

    ref = get_cpu_paged_sdpa_reference(Q, K_paged, V_paged, page_table, cur_pos, block_size, scale)

    dram = ttnn.DRAM_MEMORY_CONFIG
    K_tt_in = K_paged.permute(0, 2, 1, 3, 4).reshape(B * num_pages, nkv, block_size, d)
    V_tt_in = V_paged.permute(0, 2, 1, 3, 4).reshape(B * num_pages, nkv, block_size, d)

    tt_Q = ttnn.as_tensor(Q, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_K = ttnn.as_tensor(K_tt_in, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_V = ttnn.as_tensor(V_tt_in, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_page_table = ttnn.as_tensor(
        page_table, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=dram
    )
    tt_cur_pos = ttnn.Tensor(torch.tensor([cur_pos] * B, dtype=torch.int32), ttnn.int32).to(device)

    # Cap cores per head to 3 so num_tree_reduction_rounds stays within
    # MAX_TREE_REDUCTION_ROUNDS=6 on Galaxy (72 effective cores per chip).
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
        q_chunk_size=nh,
        k_chunk_size=block_size,
        max_cores_per_head_batch=3,
    )

    tt_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_Q,
        tt_K,
        tt_V,
        cur_pos_tensor=tt_cur_pos,
        page_table_tensor=tt_page_table,
        scale=scale,
        program_config=program_config,
        memory_config=dram,
    )
    tt_out_torch = ttnn.to_torch(tt_out)

    valid_tokens_in_last_page = (cur_pos + 1) % block_size or block_size
    zero_padded = block_size - valid_tokens_in_last_page

    assert not torch.isnan(tt_out_torch).any(), (
        f"NaN at cur_pos={cur_pos} ({description}): "
        f"{zero_padded} zero-padded tokens in last page received exp(0)=1 weight"
    )

    passing, pcc_val = comp_pcc(ref, tt_out_torch, pcc=0.99)
    assert passing, (
        f"PCC={pcc_val:.4f} at cur_pos={cur_pos} ({description}): "
        f"last page has {valid_tokens_in_last_page}/{block_size} valid tokens, "
        f"{zero_padded} zeros should be masked to -inf but weren't"
    )
