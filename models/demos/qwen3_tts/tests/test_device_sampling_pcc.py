"""
PCC test for the device-side sampling chain used in qwen3_tts CP_decode.

Tests three independent properties:

1. ttnn.topk values vs torch.topk values         (numerical accuracy of top-k)
2. ttnn.topk indices vs torch.topk indices       (set equality, k=64, sorted)
3. ttnn.sampling determinism                     (same seed → same token)
4. ttnn.sampling distribution sanity             (samples concentrate in top-k)

Run:
    pytest models/demos/qwen3_tts/tests/test_device_sampling_pcc.py -s
"""

import pytest
import torch

import ttnn

# Match the CP decode params used in the demo's _DeviceSampler.
CP_VOCAB = 2048
TOP_K = 50
SAMPLING_K_BUCKET = 64  # _SAMPLING_MAX_TOP_K — the topk-bucket size baked into trace
NUM_USERS = 32  # the kernel requires 32-user-wide tensors
TEMP = 0.9
TOP_P = 1.0
SEED = 42


def _torch_topk_reference(logits: torch.Tensor, k: int):
    """Return top-k values + indices, sorted by value desc."""
    return torch.topk(logits, k=k, dim=-1, largest=True, sorted=True)


def _build_device_tensors(device, k_int: int, p_float: float, temp_float: float):
    k_t = ttnn.from_torch(
        torch.full((NUM_USERS,), k_int, dtype=torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    p_t = ttnn.from_torch(
        torch.full((NUM_USERS,), p_float, dtype=torch.float32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    t_t = ttnn.from_torch(
        torch.full((NUM_USERS,), temp_float, dtype=torch.float32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    return k_t, p_t, t_t


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    if a.numel() == 0:
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


def test_topk_values_pcc(device):
    """ttnn.topk values vs torch.topk values: PCC > 0.99."""
    torch.manual_seed(0)
    logits_torch = torch.randn(1, 1, 1, CP_VOCAB, dtype=torch.bfloat16) * 5.0

    # Ref
    ref_values, ref_indices = _torch_topk_reference(logits_torch, k=SAMPLING_K_BUCKET)

    # Device
    logits_tt = ttnn.from_torch(
        logits_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    logits_32 = ttnn.repeat(logits_tt, ttnn.Shape([1, 1, NUM_USERS, 1]))
    tt_values, tt_indices = ttnn.topk(logits_32, k=SAMPLING_K_BUCKET, dim=-1, largest=True, sorted=True)
    tt_values_torch = ttnn.to_torch(tt_values).float()  # [1,1,32,K]
    tt_indices_torch = ttnn.to_torch(ttnn.to_layout(tt_indices, ttnn.ROW_MAJOR_LAYOUT)).long()
    ttnn.deallocate(logits_tt)
    ttnn.deallocate(logits_32)
    ttnn.deallocate(tt_values)
    ttnn.deallocate(tt_indices)

    # All 32 user rows should be identical (replicated input).
    user0_values = tt_values_torch[0, 0, 0, :]
    user0_indices = tt_indices_torch[0, 0, 0, :]

    pcc_v = _pcc(ref_values[0, 0, 0, :].float(), user0_values)
    set_match = len(set(ref_indices[0, 0, 0, :].tolist()) & set(user0_indices.tolist())) / SAMPLING_K_BUCKET

    print(f"\n[topk_values_pcc] PCC={pcc_v:.6f}  index set overlap={set_match:.4f}")
    assert pcc_v > 0.99, f"topk values PCC={pcc_v:.4f} < 0.99"
    assert set_match >= 0.95, f"topk index overlap={set_match:.4f} < 0.95"


def test_sampling_determinism(device):
    """Same logits + same seed → same sampled token (across two ttnn.sampling calls)."""
    torch.manual_seed(1)
    logits_torch = torch.randn(1, 1, 1, CP_VOCAB, dtype=torch.bfloat16) * 5.0

    logits_tt = ttnn.from_torch(
        logits_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    k_t, p_t, t_t = _build_device_tensors(device, TOP_K, TOP_P, TEMP)

    def _one_sample():
        l32 = ttnn.repeat(logits_tt, ttnn.Shape([1, 1, NUM_USERS, 1]))
        v, i = ttnn.topk(l32, k=SAMPLING_K_BUCKET, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(l32)
        i_rm = ttnn.to_layout(i, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(i)
        i_int32 = ttnn.typecast(i_rm, ttnn.int32)
        ttnn.deallocate(i_rm)
        out = ttnn.sampling(v, i_int32, k=k_t, p=p_t, temp=t_t, seed=SEED)
        ttnn.deallocate(v)
        ttnn.deallocate(i_int32)
        token = ttnn.to_torch(ttnn.from_device(out))[0, 0, 0, 0].item()
        ttnn.deallocate(out)
        return int(token)

    tok_a = _one_sample()
    tok_b = _one_sample()
    print(f"\n[sampling_determinism] seed={SEED}  tok_a={tok_a}  tok_b={tok_b}")
    assert tok_a == tok_b, f"non-deterministic: {tok_a} != {tok_b}"
    assert 0 <= tok_a < CP_VOCAB, f"token {tok_a} out of vocab"


def test_sampling_in_topk(device):
    """Sampled token must be one of the top-k indices (top_p=1.0 keeps them all)."""
    torch.manual_seed(2)
    logits_torch = torch.randn(1, 1, 1, CP_VOCAB, dtype=torch.bfloat16) * 5.0

    ref_values, ref_indices = _torch_topk_reference(logits_torch, k=TOP_K)
    allowed = set(ref_indices[0, 0, 0, :].tolist())

    logits_tt = ttnn.from_torch(
        logits_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    k_t, p_t, t_t = _build_device_tensors(device, TOP_K, TOP_P, TEMP)
    sampled = []
    for s in range(8):
        l32 = ttnn.repeat(logits_tt, ttnn.Shape([1, 1, NUM_USERS, 1]))
        v, i = ttnn.topk(l32, k=SAMPLING_K_BUCKET, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(l32)
        i_rm = ttnn.to_layout(i, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(i)
        i_int32 = ttnn.typecast(i_rm, ttnn.int32)
        ttnn.deallocate(i_rm)
        out = ttnn.sampling(v, i_int32, k=k_t, p=p_t, temp=t_t, seed=SEED + s)
        ttnn.deallocate(v)
        ttnn.deallocate(i_int32)
        tok = int(ttnn.to_torch(ttnn.from_device(out))[0, 0, 0, 0].item())
        ttnn.deallocate(out)
        sampled.append(tok)

    in_topk = sum(1 for t in sampled if t in allowed)
    print(f"\n[sampling_in_topk] sampled tokens: {sampled}  in_top{TOP_K}: {in_topk}/8")
    assert in_topk == 8, f"only {in_topk}/8 samples in top-{TOP_K}; samples={sampled}"


def test_chain_embedding_lookup(device):
    """Verify the chain's ttnn.embedding call returns the right row.

    The chain does: token_id (uint32 RM [1,1,1,1]) ⨯ embed_table [1,1,vocab,dim]
    → embed [1,1,1,dim]. Compare against torch.nn.functional.embedding for
    several specific token ids.
    """

    EMB_DIM = 2048
    VOCAB = 2048

    torch.manual_seed(0)
    table_torch = torch.randn(VOCAB, EMB_DIM, dtype=torch.bfloat16) * 0.5
    # Code-predictor reshapes the table to [1, 1, vocab, emb_dim] before lookup.
    table_4d_torch = table_torch.reshape(1, 1, VOCAB, EMB_DIM)
    table_tt = ttnn.from_torch(
        table_4d_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for tok_id in [0, 7, 137, 1023, 2047]:
        # Simulate the chain's [1,1,1,32] sample output, sliced to [1,1,1,1].
        sample_out = ttnn.from_torch(
            torch.full((1, 1, 1, NUM_USERS), tok_id, dtype=torch.int32),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        tok_slice = ttnn.slice(sample_out, [0, 0, 0, 0], [1, 1, 1, 1])
        embed_out = ttnn.embedding(
            tok_slice,
            table_tt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        embed_torch = ttnn.to_torch(embed_out).float()  # tile-padded
        # Reshape to [1,1,1,EMB_DIM] (drop tile padding by slicing logical width).
        # Actual logical shape might be [1,1,1,EMB_DIM] but tilelayout pads; use shape attr.
        logical_shape = tuple(int(d) for d in embed_out.shape)
        # take last EMB_DIM elements along last axis
        if embed_torch.dim() == 4 and embed_torch.shape[-1] >= EMB_DIM:
            embed_flat = embed_torch[..., :EMB_DIM].flatten()
        else:
            embed_flat = embed_torch.flatten()[:EMB_DIM]
        ref = table_torch[tok_id].float()

        pcc = _pcc(ref, embed_flat[:EMB_DIM])
        max_abs = (ref - embed_flat[:EMB_DIM]).abs().max().item()
        print(f"\n[chain_embedding] tok={tok_id} logical_shape={logical_shape} PCC={pcc:.6f} max|Δ|={max_abs:.6f}")
        assert pcc > 0.999, f"embedding tok={tok_id}: PCC={pcc:.4f}"
        assert max_abs < 1e-2, f"embedding tok={tok_id}: max|Δ|={max_abs:.4f}"

        ttnn.deallocate(sample_out)
        ttnn.deallocate(tok_slice)
        ttnn.deallocate(embed_out)
    ttnn.deallocate(table_tt)


def test_topk_padded_vocab_pcc(device):
    """topk(pad-to-8192(logits)) must equal topk(logits) — padding values are -inf
    so they never enter top-K. This is the multicore-enabling trick used by
    _DeviceSampler.
    """
    PAD_W = 8192
    torch.manual_seed(4)
    logits_torch = torch.randn(1, 1, 1, CP_VOCAB, dtype=torch.bfloat16) * 5.0

    # Reference: top-K on raw logits.
    ref_values, ref_indices = _torch_topk_reference(logits_torch, k=SAMPLING_K_BUCKET)

    # Device: replicate, pad with -inf, topk multicore.
    logits_tt = ttnn.from_torch(logits_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    logits_32 = ttnn.repeat(logits_tt, ttnn.Shape([1, 1, NUM_USERS, 1]))
    logits_padded = ttnn.pad(
        logits_32,
        [(0, 0), (0, 0), (0, 0), (0, PAD_W - CP_VOCAB)],
        value=-1e30,
    )
    ttnn.deallocate(logits_32)
    tt_values, tt_indices = ttnn.topk(logits_padded, k=SAMPLING_K_BUCKET, dim=-1, largest=True, sorted=True)
    tt_values_torch = ttnn.to_torch(tt_values).float()
    tt_indices_torch = ttnn.to_torch(ttnn.to_layout(tt_indices, ttnn.ROW_MAJOR_LAYOUT)).long()
    ttnn.deallocate(logits_padded)
    ttnn.deallocate(tt_values)
    ttnn.deallocate(tt_indices)

    user0_values = tt_values_torch[0, 0, 0, :]
    user0_indices = tt_indices_torch[0, 0, 0, :]

    pcc = _pcc(ref_values[0, 0, 0, :].float(), user0_values)
    set_match = len(set(ref_indices[0, 0, 0, :].tolist()) & set(user0_indices.tolist())) / SAMPLING_K_BUCKET
    # Critical: padded positions (>= CP_VOCAB) must NEVER appear in top-K.
    pad_leak = sum(1 for idx in user0_indices.tolist() if idx >= CP_VOCAB)

    print(f"\n[topk_padded] PCC={pcc:.6f}  set_match={set_match:.4f}  pad_leak={pad_leak}/{SAMPLING_K_BUCKET}")
    assert pcc > 0.99, f"PCC={pcc:.4f}"
    assert set_match >= 0.95, f"set_match={set_match:.4f}"
    assert pad_leak == 0, f"{pad_leak} padded positions leaked into top-K"


def test_argmax_vs_top1(device):
    """ttnn.sampling with k=1 should always return the argmax token."""
    torch.manual_seed(3)
    logits_torch = torch.randn(1, 1, 1, CP_VOCAB, dtype=torch.bfloat16) * 5.0

    expected = int(torch.argmax(logits_torch[0, 0, 0]).item())

    logits_tt = ttnn.from_torch(
        logits_torch,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    k_t, p_t, t_t = _build_device_tensors(device, k_int=1, p_float=1.0, temp_float=1.0)

    l32 = ttnn.repeat(logits_tt, ttnn.Shape([1, 1, NUM_USERS, 1]))
    v, i = ttnn.topk(l32, k=SAMPLING_K_BUCKET, dim=-1, largest=True, sorted=True)
    ttnn.deallocate(l32)
    i_rm = ttnn.to_layout(i, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(i)
    i_int32 = ttnn.typecast(i_rm, ttnn.int32)
    ttnn.deallocate(i_rm)
    out = ttnn.sampling(v, i_int32, k=k_t, p=p_t, temp=t_t, seed=SEED)
    ttnn.deallocate(v)
    ttnn.deallocate(i_int32)
    got = int(ttnn.to_torch(ttnn.from_device(out))[0, 0, 0, 0].item())
    ttnn.deallocate(out)

    print(f"\n[argmax_vs_top1] expected={expected}  got={got}")
    assert got == expected, f"k=1 sampling returned {got} but argmax is {expected}"
