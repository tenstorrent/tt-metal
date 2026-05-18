"""Fused attention_block kernel — tests.

Validates the multi-Op chaining + receiver-pull mcast + QKV matmul:

  test_attention_block_fused_ln_plus_residual
    Single TRISC dispatch that calls LN1 then residual_add in sequence, with
    L1 CB chaining (no host round-trip). Math: out = LN1(x; gamma, beta) + x.
    Loose PCC ≥ 0.999 gate (typically lands at 0.999990+).

  test_attention_block_fused_ln_mcast_probe
    Same dispatch, but reads back qkv_act_cb from each of the 36 QKV
    receivers and checks that every receiver holds LN1(x). Confirms the
    NCRISC sender→receiver semaphore wait + 8× noc_async_read pipeline
    reproduces the LN1 output bit-stably on every receiver.

  test_attention_block_fused_qkv_matmul
    Reads back qkv_out_cb assembled from 36 width-sharded slices and PCCs
    against the torch golden LN1(x) @ W_qkv. bfp8 weights drop the PCC bar
    from the LN1 floor (~0.99999) to the matmul floor (~0.999).
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "perf"))
from golden_fc1 import pcc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attention_block"))
from op import (  # noqa: E402
    SigLIPAttentionBlockFused,
    build_tensors_for_fused_attention_block,
)

M, D = 256, 1152
N_QKV_UNPADDED = 3 * D  # 3456 — natural Q/K/V concat at head_dim=72
NUM_HEADS = 16
HEAD_DIM_TRUE = 72
HEAD_DIM_PADDED = 96
N_QKV_PADDED = 3 * NUM_HEADS * HEAD_DIM_PADDED  # 4608 — device-side, head-aligned
EPS = 1e-6


def unpad_qkv_output(out_padded):
    """Slice (M, 4608) device output back to (M, 3456) by taking the first
    HEAD_DIM_TRUE cols of each head's HEAD_DIM_PADDED block. Inverse of the
    weight padding in build_tensors_for_fused_attention_block."""
    out_M = out_padded.shape[0]
    out = torch.zeros(out_M, N_QKV_UNPADDED, dtype=out_padded.dtype)
    for qkv_idx in range(3):
        for h in range(NUM_HEADS):
            src_start = qkv_idx * NUM_HEADS * HEAD_DIM_PADDED + h * HEAD_DIM_PADDED
            dst_start = qkv_idx * D + h * HEAD_DIM_TRUE
            out[:, dst_start : dst_start + HEAD_DIM_TRUE] = out_padded[:, src_start : src_start + HEAD_DIM_TRUE]
    return out


def make_inputs(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, D, generator=g, dtype=torch.bfloat16) * 0.5
    gamma = torch.ones(D, dtype=torch.bfloat16) + torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.1
    beta = torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.05
    return x, gamma, beta


def make_qkv_weight(seed: int = 7):
    """Match build_tensors_for_fused_attention_block's default w_qkv generator
    so the test golden uses the same weight matrix the device sees.

    Returns the UNPADDED (D, 3D=3456) weight — build_tensors does the per-head
    zero-padding internally.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(D, N_QKV_UNPADDED, generator=g, dtype=torch.bfloat16) * 0.05


def unpad_sdpa_assembled(assembled_padded):
    """Strip 24 zero cols per head from a (M, 16 * HEAD_DIM_PADDED = 1536)
    SDPA assembled output to (M, D = 16 * HEAD_DIM_TRUE = 1152). Mirror of
    unpad_qkv_output but for the LN1-row assembled SDPA representation."""
    assert assembled_padded.shape == (M, NUM_HEADS * HEAD_DIM_PADDED)
    out = torch.zeros(M, D, dtype=assembled_padded.dtype)
    for h in range(NUM_HEADS):
        src_start = h * HEAD_DIM_PADDED
        dst_start = h * HEAD_DIM_TRUE
        out[:, dst_start : dst_start + HEAD_DIM_TRUE] = assembled_padded[:, src_start : src_start + HEAD_DIM_TRUE]
    return out


def make_oproj_weight(seed: int = 11):
    """Deterministic (D=1152, D=1152) O-proj weight for the fused+oproj
    composition test."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(D, D, generator=g, dtype=torch.bfloat16) * 0.04


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_attention_block_fused_ln_plus_residual(device):
    x, gamma, beta = make_inputs(seed=42)

    # Golden: LN1(x) + x in fp32, output bf16.
    ln_out_golden = F.layer_norm(x.float(), (D,), gamma.float(), beta.float(), eps=EPS)
    y_golden = (ln_out_golden + x.float()).to(torch.bfloat16)

    tensors = build_tensors_for_fused_attention_block(device, x, gamma, beta)
    SigLIPAttentionBlockFused.op(*tensors, eps=EPS)

    import ttnn as _ttnn

    # Positional layout (positive indices for resilience to future growth):
    #   0..5: ln_in, gamma, beta, scaler, ones, accum
    #   6:   fused_scratch  (aliases xmm, xmm2 on LN1; qk_scores, attn_out,
    #         v_partial on SDPA — Commits 8/9/10)
    #   7..9: mean, var, ivar
    #   10..12: ln_out, x_residual, final_out
    #   13..14: qkv_act, ln_done_trigger
    #   15..17: qkv_w, qkv_out, qkv_done_trigger
    #   18..19: sdpa_q, sdpa_k_partial
    #   20..24: softmax max, exp, sum, isum, scaler  (Commit 9)
    final_out_tt = tensors[12]
    y_device = _ttnn.to_torch(final_out_tt)

    p = pcc(y_golden, y_device)
    print(f"\nPCC (fused LN1+residual vs torch fp32) = {p:.6f}")
    print(f"  shape={tuple(y_device.shape)}, dtype={y_device.dtype}")
    assert p >= 0.999, f"PCC {p} below 0.999 gate"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_attention_block_fused_ln_mcast_probe(device):
    """Probe the LN1→QKV receiver-pull on the 36-core grid.

    After SigLIPAttentionBlockFused.op runs, each of the 36 QKV receivers
    should hold a private copy of LN1(x). We read qkv_act_tt back as a
    (36 × 256, 1152) tensor, reshape to (36, 256, 1152), and PCC each
    receiver against the same LN1(x) golden.
    """
    x, gamma, beta = make_inputs(seed=42)

    # Golden: LN1(x) in fp32, cast to bf16. The 36 receivers each store this
    # full (256, 1152) tensor verbatim — per-receiver QKV slicing happens in
    # the matmul phase (consumed by the qkv_matmul test below).
    ln_out_golden = F.layer_norm(x.float(), (D,), gamma.float(), beta.float(), eps=EPS).to(torch.bfloat16)

    tensors = build_tensors_for_fused_attention_block(device, x, gamma, beta)
    SigLIPAttentionBlockFused.op(*tensors, eps=EPS)

    import ttnn as _ttnn

    # qkv_act_tt is at positional index 13 (see test_attention_block_fused_ln_plus_residual).
    qkv_act_tt = tensors[13]
    qkv_act_torch = _ttnn.to_torch(qkv_act_tt)

    num_receivers = SigLIPAttentionBlockFused.QKV_NUM_CORES
    assert qkv_act_torch.shape == (
        num_receivers * M,
        D,
    ), f"qkv_act tensor shape {tuple(qkv_act_torch.shape)} != expected {(num_receivers * M, D)}"

    # Each receiver's slice is (M, D) along the height axis.
    qkv_per_receiver = qkv_act_torch.reshape(num_receivers, M, D)

    # PCC per receiver against the same LN1(x) golden. The minimum PCC across
    # receivers gates the test — a single sender → receiver edge failure would
    # otherwise be masked by the average.
    min_pcc = float("inf")
    for r in range(num_receivers):
        p = pcc(ln_out_golden, qkv_per_receiver[r])
        if p < min_pcc:
            min_pcc = p

    print(f"\nPCC (LN1 mcast probe, min across {num_receivers} receivers) = {min_pcc:.6f}")
    print(f"  per-receiver shape={(M, D)}, dtype={qkv_per_receiver.dtype}")
    assert min_pcc >= 0.999988, f"min receiver PCC {min_pcc} below 0.999988 gate"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_attention_block_fused_qkv_matmul(device):
    """Validate the QKV matmul output.

    Each of the 36 QKV receivers computes a (256, 96) slice of LN1(x) @ W_qkv.
    Width-sharded across cores ⇒ ttnn.to_torch returns a (256, 3456) tensor
    with the N-dimension concatenated in the grid's row-major order.
    """
    x, gamma, beta = make_inputs(seed=42)
    w_qkv = make_qkv_weight(seed=7)  # unpadded (D, 3D=3456)

    # Golden: LN1(x) @ W_qkv in fp32 against the UNPADDED weight, bf16 output.
    # bfp8 quantization happens on the device side when from_torch loads the
    # weight; we keep the golden in bf16 against the bf16 weight so the PCC
    # includes the bfp8 noise.
    ln_out_golden = F.layer_norm(x.float(), (D,), gamma.float(), beta.float(), eps=EPS)
    qkv_golden_unpadded = (ln_out_golden.to(torch.float32) @ w_qkv.to(torch.float32)).to(torch.bfloat16)

    tensors = build_tensors_for_fused_attention_block(device, x, gamma, beta, w_qkv_torch=w_qkv)
    SigLIPAttentionBlockFused.op(*tensors, eps=EPS)

    import ttnn as _ttnn

    # qkv_out_tt is at positional index 16.
    # Device shape is the padded (M, 4608); we slice each head's first
    # head_dim_true cols to recover the natural (M, 3456) layout before PCC.
    qkv_out_tt = tensors[16]
    qkv_out_device_padded = _ttnn.to_torch(qkv_out_tt)
    assert qkv_out_device_padded.shape == (
        M,
        N_QKV_PADDED,
    ), f"qkv_out (padded) shape {tuple(qkv_out_device_padded.shape)} != {(M, N_QKV_PADDED)}"
    qkv_out_device = unpad_qkv_output(qkv_out_device_padded)
    assert qkv_out_device.shape == (M, N_QKV_UNPADDED)

    p = pcc(qkv_golden_unpadded, qkv_out_device)
    print(f"\nPCC (QKV matmul vs torch LN1(x) @ W_qkv, head-unpadded) = {p:.6f}")
    print(
        f"  device padded shape={tuple(qkv_out_device_padded.shape)}, "
        f"unpadded shape={tuple(qkv_out_device.shape)}, dtype={qkv_out_device.dtype}"
    )
    # bfp8 weight + HiFi4 accumulator typically lands ≥ 0.999 on this shape.
    assert p >= 0.999, f"PCC {p} below 0.999 gate"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_attention_block_fused_sdpa_q_probe(device):
    """Probe the QKV → SDPA per-head Q delivery on the 64-core SDPA grid.

    Each SDPA worker for head h pulls its M-slice (64 rows × 96 cols padded)
    of Q head h from the QKV core holding that head's slice. We read
    sdpa_q_tt back as (64*64, 96), reshape to (64 workers, 64, 96), and
    check each worker matches the expected M-slice of the head's padded Q.

    PCC gate matches the QKV matmul probe (0.999) since the data is just
    a NoC copy of QKV's bfp8/HiFi4 output — no additional compute.
    """
    x, gamma, beta = make_inputs(seed=42)
    w_qkv = make_qkv_weight(seed=7)

    # Build golden: padded QKV output sliced into per-head Q (16 × (M, 96)).
    ln_out_golden = F.layer_norm(x.float(), (D,), gamma.float(), beta.float(), eps=EPS)
    qkv_golden_unpadded = (ln_out_golden.to(torch.float32) @ w_qkv.to(torch.float32)).to(torch.bfloat16)
    # Reconstruct the device-side padded layout per-head (16 padded Q heads
    # of shape (M, HEAD_DIM_PADDED), with last 24 cols zero).
    q_heads_padded = torch.zeros(NUM_HEADS, M, HEAD_DIM_PADDED, dtype=torch.bfloat16)
    for h in range(NUM_HEADS):
        src_start = h * HEAD_DIM_TRUE
        q_heads_padded[h, :, :HEAD_DIM_TRUE] = qkv_golden_unpadded[:, src_start : src_start + HEAD_DIM_TRUE]

    tensors = build_tensors_for_fused_attention_block(device, x, gamma, beta, w_qkv_torch=w_qkv)
    SigLIPAttentionBlockFused.op(*tensors, eps=EPS)

    import ttnn as _ttnn

    # sdpa_q_tt is at positional index 18.
    sdpa_q_tt = tensors[18]
    sdpa_q_torch = _ttnn.to_torch(sdpa_q_tt)

    sdpa_num_workers = SigLIPAttentionBlockFused.SDPA_NUM_CORES  # 32 (#11 Commit 4: relocated SDPA)
    rows_per_worker = M // SigLIPAttentionBlockFused.NUM_SDPA_WORKERS_PER_HEAD  # 128 (2 workers/head)
    assert sdpa_q_torch.shape == (
        sdpa_num_workers * rows_per_worker,
        HEAD_DIM_PADDED,
    ), f"sdpa_q shape {tuple(sdpa_q_torch.shape)} != expected"

    # Reshape to (workers, rows, cols). Workers laid out row-major across the
    # 4×8 SDPA grid (logical x=8..11, y=0..7): linear shard idx maps to
    # relative (rel_x, rel_y) = (idx % SDPA_GRID_X, idx // SDPA_GRID_X). Head
    # mapping uses those relative coords (the absolute x_offset cancels out):
    #   head_idx   = (rel_y // num_workers_per_head) * SDPA_GRID_X + rel_x
    #   worker_idx = rel_y % num_workers_per_head
    sdpa_q_workers = sdpa_q_torch.reshape(sdpa_num_workers, rows_per_worker, HEAD_DIM_PADDED)
    sdpa_grid_x = SigLIPAttentionBlockFused.SDPA_GRID_X
    num_workers_per_head = SigLIPAttentionBlockFused.NUM_SDPA_WORKERS_PER_HEAD

    min_pcc = float("inf")
    worst = None
    for worker_linear in range(sdpa_num_workers):
        y = worker_linear // sdpa_grid_x
        x = worker_linear % sdpa_grid_x
        head_idx = (y // num_workers_per_head) * sdpa_grid_x + x
        worker_idx = y % num_workers_per_head
        expected = q_heads_padded[head_idx, worker_idx * rows_per_worker : (worker_idx + 1) * rows_per_worker, :]
        p = pcc(expected, sdpa_q_workers[worker_linear])
        if p < min_pcc:
            min_pcc = p
            worst = (worker_linear, x, y, head_idx, worker_idx)

    print(f"\nPCC (SDPA Q probe, min across {sdpa_num_workers} workers) = {min_pcc:.6f}")
    print(
        f"  worst worker: linear={worst[0]}, (x,y)=({worst[1]},{worst[2]}), " f"head={worst[3]}, worker_idx={worst[4]}"
    )
    print(f"  per-worker shape={(rows_per_worker, HEAD_DIM_PADDED)}, dtype={sdpa_q_workers.dtype}")
    assert min_pcc >= 0.999, f"min worker PCC {min_pcc} below 0.999 gate"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_attention_block_fused_sdpa_softmax_probe(device):
    """Per-head row-wise softmax on QK^T (Commit 9).

    Each of the 32 SDPA workers computes Q @ K^T then runs softmax(.., dim=-1)
    row-wise over the (128, 256) per-worker scores via pi05_siglip_ops::
    Softmax::Op. softmax_out_cb is aliased to sdpa_qk_scores_cb's L1 region
    (in-place via fused_scratch_tt) — after the dispatch, fused_scratch
    holds softmax values on SDPA cores' shards.

    TRISC iterates n_out (K-row) ∈ [0..7]: waits on NCRISC-streamed K-row,
    then for each m_out (Q-row) ∈ [0..3] matmuls Q[m_out, :] @ K[n_out, :]^T
    and packs to qk_scores tile slot (m_out * M_KV_TILES + n_out). Each
    K-row streams once and serves all 4 M-rows in sequence.

    Read-back: ttnn.to_torch(fused_scratch_tt) returns the full
    (LN1+SDPA) buffer; we slice each SDPA core's shard and pull the first
    32 tiles (the qk_scores region).

    Golden: torch Q_padded[h, worker_M_slice, :] @ K_padded[h, :, :].T
    = (128, 256), where _padded has head_dim_padded=96 cols with
    head_dim_true=72 real values + 24 zero cols.
    """
    x, gamma, beta = make_inputs(seed=42)
    w_qkv = make_qkv_weight(seed=7)

    ln_out_golden = F.layer_norm(x.float(), (D,), gamma.float(), beta.float(), eps=EPS)
    qkv_golden_unpadded = (ln_out_golden.to(torch.float32) @ w_qkv.to(torch.float32)).to(torch.bfloat16)

    # Build per-head padded Q and K matrices to match the device's layout
    # (head_dim_true=72 real data, head_dim_padded=96 with zeros at [72:96]).
    q_padded_per_head = torch.zeros(NUM_HEADS, M, HEAD_DIM_PADDED, dtype=torch.bfloat16)
    k_padded_per_head = torch.zeros(NUM_HEADS, M, HEAD_DIM_PADDED, dtype=torch.bfloat16)
    for h in range(NUM_HEADS):
        q_start = h * HEAD_DIM_TRUE
        k_start = D + h * HEAD_DIM_TRUE
        q_padded_per_head[h, :, :HEAD_DIM_TRUE] = qkv_golden_unpadded[:, q_start : q_start + HEAD_DIM_TRUE]
        k_padded_per_head[h, :, :HEAD_DIM_TRUE] = qkv_golden_unpadded[:, k_start : k_start + HEAD_DIM_TRUE]

    tensors = build_tensors_for_fused_attention_block(device, x, gamma, beta, w_qkv_torch=w_qkv)
    SigLIPAttentionBlockFused.op(*tensors, eps=EPS)

    import ttnn as _ttnn

    # fused_scratch_tt is at index 6 in build_tensors order.
    # Shape: (40 cores × 1152 rows, 32 cols). First 8 shards = LN1 cores
    # (holding xmm data). Next 32 shards = SDPA workers (holding qk_scores
    # at the front of each shard).
    fused_scratch_tt = tensors[6]
    fused_scratch_torch = _ttnn.to_torch(fused_scratch_tt)

    sdpa_num_workers = SigLIPAttentionBlockFused.SDPA_NUM_CORES  # 32
    sdpa_grid_x = SigLIPAttentionBlockFused.SDPA_GRID_X  # 4
    num_workers_per_head = SigLIPAttentionBlockFused.NUM_SDPA_WORKERS_PER_HEAD  # 2
    rows_per_worker = M // num_workers_per_head  # 128
    m_kv = M  # 256
    ln1_num_cores = SigLIPAttentionBlockFused.LN1_NUM_CORES  # 8
    D_TILES = SigLIPAttentionBlockFused.D_TILES  # 36

    # #11 Commit 10: fused_scratch shard grew from 36 → 72 tiles. shard_h = 2 * D_TILES * TILE.
    shard_h = 2 * D_TILES * SigLIPAttentionBlockFused.TILE  # 2304
    expected_shape = ((ln1_num_cores + sdpa_num_workers) * shard_h, SigLIPAttentionBlockFused.TILE)
    assert (
        fused_scratch_torch.shape == expected_shape
    ), f"fused_scratch shape {tuple(fused_scratch_torch.shape)} != {expected_shape}"

    # Extract each SDPA worker's qk_scores from its shard.
    # Per-worker shard: rows [ln1_num_cores + worker_idx] × shard_h .. +shard_h.
    # qk_scores occupies the FIRST 32 tiles = 32 row-tiles (since shard width is
    # one tile wide) = 32 × 32 = 1024 rows of the shard.
    # Each row-tile in the shard is one packed tile in CB-slot order; the CB's
    # pack_tile call wrote slot (m_out * M_KV_TILES + n_out) for the (m_out,
    # n_out) output tile, so:
    #   tile_idx → (m_out=tile_idx // M_KV_TILES, n_out=tile_idx % M_KV_TILES)
    m_kv_tiles = m_kv // SigLIPAttentionBlockFused.TILE  # 8
    m_per_worker_tiles = rows_per_worker // SigLIPAttentionBlockFused.TILE  # 4
    qk_scores_tiles_per_worker = m_per_worker_tiles * m_kv_tiles  # 32
    qk_scores_rows_per_worker = qk_scores_tiles_per_worker * SigLIPAttentionBlockFused.TILE  # 1024

    min_pcc = float("inf")
    worst = None
    for worker_linear in range(sdpa_num_workers):
        shard_start = (ln1_num_cores + worker_linear) * shard_h
        worker_shard = fused_scratch_torch[shard_start : shard_start + qk_scores_rows_per_worker, :]
        # Reshape to (qk_scores_tiles_per_worker, TILE, TILE).
        tile_stack = worker_shard.reshape(
            qk_scores_tiles_per_worker, SigLIPAttentionBlockFused.TILE, SigLIPAttentionBlockFused.TILE
        )
        # Re-arrange into the natural (M_per_worker, M_KV) layout.
        qk_worker = torch.zeros(rows_per_worker, m_kv, dtype=worker_shard.dtype)
        for tile_idx in range(qk_scores_tiles_per_worker):
            m_out = tile_idx // m_kv_tiles
            n_out = tile_idx % m_kv_tiles
            qk_worker[
                m_out * SigLIPAttentionBlockFused.TILE : (m_out + 1) * SigLIPAttentionBlockFused.TILE,
                n_out * SigLIPAttentionBlockFused.TILE : (n_out + 1) * SigLIPAttentionBlockFused.TILE,
            ] = tile_stack[tile_idx]

        # Head/worker mapping (same as in sdpa_q_probe).
        y = worker_linear // sdpa_grid_x
        x_rel = worker_linear % sdpa_grid_x
        head_idx = (y // num_workers_per_head) * sdpa_grid_x + x_rel
        worker_idx = y % num_workers_per_head
        m_start = worker_idx * rows_per_worker
        q_slice = q_padded_per_head[head_idx, m_start : m_start + rows_per_worker, :].to(torch.float32)
        k_slice = k_padded_per_head[head_idx, :, :].to(torch.float32)  # full (M_KV, head_dim_padded)
        scores = q_slice @ k_slice.T  # (rows_per_worker, M_KV) fp32
        # Pure softmax(scores, dim=-1) — no 1/sqrt(d_k) scaling in this commit.
        expected = torch.softmax(scores, dim=-1).to(torch.bfloat16)

        p = pcc(expected, qk_worker)
        if p < min_pcc:
            min_pcc = p
            worst = (worker_linear, head_idx, worker_idx)

    print(
        f"\nPCC (SDPA softmax(Q @ K^T) probe, min/32 workers) = "
        f"{min_pcc:.6f} (worst: worker={worst[0]}, head={worst[1]}, w_idx={worst[2]})"
    )
    print(f"  per-worker shape={(rows_per_worker, m_kv)}, dtype={qk_worker.dtype}")
    # Softmax stacks SFPU exp_tile + recip_tile on top of the bfp8/HiFi4
    # QK^T base — compounded approximation error. With pure bf16 weights
    # the same kernel lands ~0.998; with our bfp8 weights it lands ~0.996.
    # The pi0_base standalone SigLIP SDPA test runs at the same precision
    # regime.
    assert min_pcc >= 0.996, f"min softmax probe PCC {min_pcc} below 0.996"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_attention_block_fused_sdpa_attn_v_probe(device):
    """Streaming-V Attn @ V compute (Commit 10).

    Each SDPA worker computes (softmax(Q @ K^T) @ V) for its M-slice of the
    head, producing a (M_per_worker=128, head_dim_padded=96) bf16 = 12-tile
    output in sdpa_attn_out_cb (aliased into fused_scratch at byte offset
    65536 on SDPA cores).

    TRISC accumulates over the M_KV inner dimension by streaming V tile-by-tile
    via sdpa_v_partial_cb (3 tiles per V-row). 12 dst register slots hold
    one (m_out, d_out) output tile each; after all 8 K iterations, all 12
    tiles pack to attn_out in (m_out × 3 + d_out) order.

    Golden: per-head torch.softmax(Q @ K^T, dim=-1) @ V (no scaling).
    """
    x, gamma, beta = make_inputs(seed=42)
    w_qkv = make_qkv_weight(seed=7)

    ln_out_golden = F.layer_norm(x.float(), (D,), gamma.float(), beta.float(), eps=EPS)
    qkv_golden_unpadded = (ln_out_golden.to(torch.float32) @ w_qkv.to(torch.float32)).to(torch.bfloat16)

    q_padded_per_head = torch.zeros(NUM_HEADS, M, HEAD_DIM_PADDED, dtype=torch.bfloat16)
    k_padded_per_head = torch.zeros(NUM_HEADS, M, HEAD_DIM_PADDED, dtype=torch.bfloat16)
    v_padded_per_head = torch.zeros(NUM_HEADS, M, HEAD_DIM_PADDED, dtype=torch.bfloat16)
    for h in range(NUM_HEADS):
        q_start = h * HEAD_DIM_TRUE
        k_start = D + h * HEAD_DIM_TRUE
        v_start = 2 * D + h * HEAD_DIM_TRUE
        q_padded_per_head[h, :, :HEAD_DIM_TRUE] = qkv_golden_unpadded[:, q_start : q_start + HEAD_DIM_TRUE]
        k_padded_per_head[h, :, :HEAD_DIM_TRUE] = qkv_golden_unpadded[:, k_start : k_start + HEAD_DIM_TRUE]
        v_padded_per_head[h, :, :HEAD_DIM_TRUE] = qkv_golden_unpadded[:, v_start : v_start + HEAD_DIM_TRUE]

    tensors = build_tensors_for_fused_attention_block(device, x, gamma, beta, w_qkv_torch=w_qkv)
    SigLIPAttentionBlockFused.op(*tensors, eps=EPS)

    import ttnn as _ttnn

    fused_scratch_tt = tensors[6]
    fused_scratch_torch = _ttnn.to_torch(fused_scratch_tt)

    sdpa_num_workers = SigLIPAttentionBlockFused.SDPA_NUM_CORES  # 32
    sdpa_grid_x = SigLIPAttentionBlockFused.SDPA_GRID_X  # 4
    num_workers_per_head = SigLIPAttentionBlockFused.NUM_SDPA_WORKERS_PER_HEAD  # 2
    rows_per_worker = M // num_workers_per_head  # 128
    ln1_num_cores = SigLIPAttentionBlockFused.LN1_NUM_CORES  # 8
    head_dim_padded = HEAD_DIM_PADDED  # 96
    head_dim_n_tiles = SigLIPAttentionBlockFused.QKV_N_TILES_PER_CORE  # 3
    m_per_worker_tiles = rows_per_worker // SigLIPAttentionBlockFused.TILE  # 4
    attn_out_tiles_per_worker = m_per_worker_tiles * head_dim_n_tiles  # 12

    # fused_scratch shard layout (Commit 10): 72 tiles per worker.
    #   tiles 0..31 (rows 0..1023):    qk_scores / softmax_out
    #   tiles 32..43 (rows 1024..1407): attn_out  ← 12 tiles, this commit's output
    #   tiles 44..46 (rows 1408..1503): v_partial (transient; final state holds the last V row)
    shard_h = 2 * SigLIPAttentionBlockFused.D_TILES * SigLIPAttentionBlockFused.TILE  # 72 * 32 = 2304
    attn_out_start_tile = m_per_worker_tiles * (M // SigLIPAttentionBlockFused.TILE)  # 4*8 = 32
    attn_out_start_row = attn_out_start_tile * SigLIPAttentionBlockFused.TILE  # 1024
    attn_out_rows = attn_out_tiles_per_worker * SigLIPAttentionBlockFused.TILE  # 384

    expected_shape = ((ln1_num_cores + sdpa_num_workers) * shard_h, SigLIPAttentionBlockFused.TILE)
    assert (
        fused_scratch_torch.shape == expected_shape
    ), f"fused_scratch shape {tuple(fused_scratch_torch.shape)} != {expected_shape}"

    min_pcc = float("inf")
    worst = None
    for worker_linear in range(sdpa_num_workers):
        shard_start = (ln1_num_cores + worker_linear) * shard_h
        attn_region_start = shard_start + attn_out_start_row
        attn_region = fused_scratch_torch[attn_region_start : attn_region_start + attn_out_rows, :]
        tile_stack = attn_region.reshape(
            attn_out_tiles_per_worker, SigLIPAttentionBlockFused.TILE, SigLIPAttentionBlockFused.TILE
        )
        attn_worker = torch.zeros(rows_per_worker, head_dim_padded, dtype=attn_region.dtype)
        for tile_idx in range(attn_out_tiles_per_worker):
            m_out = tile_idx // head_dim_n_tiles
            d_out = tile_idx % head_dim_n_tiles
            attn_worker[
                m_out * SigLIPAttentionBlockFused.TILE : (m_out + 1) * SigLIPAttentionBlockFused.TILE,
                d_out * SigLIPAttentionBlockFused.TILE : (d_out + 1) * SigLIPAttentionBlockFused.TILE,
            ] = tile_stack[tile_idx]

        y = worker_linear // sdpa_grid_x
        x_rel = worker_linear % sdpa_grid_x
        head_idx = (y // num_workers_per_head) * sdpa_grid_x + x_rel
        worker_idx = y % num_workers_per_head
        m_start = worker_idx * rows_per_worker

        q_slice = q_padded_per_head[head_idx, m_start : m_start + rows_per_worker, :].to(torch.float32)
        k_full = k_padded_per_head[head_idx, :, :].to(torch.float32)
        v_full = v_padded_per_head[head_idx, :, :].to(torch.float32)
        scores = q_slice @ k_full.T
        softmax_scores = torch.softmax(scores, dim=-1)
        expected = (softmax_scores @ v_full).to(torch.bfloat16)

        p = pcc(expected, attn_worker)
        if p < min_pcc:
            min_pcc = p
            worst = (worker_linear, head_idx, worker_idx)

    print(
        f"\nPCC (SDPA Attn @ V probe, min/32 workers) = "
        f"{min_pcc:.6f} (worst: worker={worst[0]}, head={worst[1]}, w_idx={worst[2]})"
    )
    print(f"  per-worker shape={(rows_per_worker, head_dim_padded)}, dtype={attn_worker.dtype}")
    # Attn @ V stacks another bfp8/HiFi4 matmul on top of softmax — same
    # precision regime, gate stays at 0.996.
    assert min_pcc >= 0.996, f"min Attn @ V probe PCC {min_pcc} below 0.996"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_attention_block_fused_sdpa_assembled_probe(device):
    """Per-head output assembly on the LN1 row (Commit 11).

    After all 32 SDPA workers finish Attn @ V, each atomic_incs the 4 LN1
    cores it serves on sdpa_done_sem. Each LN1 core waits for the sem to
    reach NUM_HEADS=16, then fan-in-reads its M-slice of every head's
    attn_out (32 rows × 96 cols = 3 tiles per head). The 16 head slices
    concatenate horizontally into a (32, 16 * 96 = 1536) per-LN1-core
    block, totaling (256, 1536) across the LN1 row.

    Destination: sdpa_assembled_out_cb aliased into fused_scratch on LN1
    cores at byte offset 0 — overlapping (and overwriting) xmm/xmm2's L1
    region, which is dead by this phase. 48 tiles per LN1 core.

    Golden: per-head torch.softmax(Q @ K^T, dim=-1) @ V (padded to 96 cols),
    concatenated horizontally → (256, 1536). Each LN1 core's expected slice
    is rows [c*32 : (c+1)*32] of the concat.
    """
    x, gamma, beta = make_inputs(seed=42)
    w_qkv = make_qkv_weight(seed=7)

    ln_out_golden = F.layer_norm(x.float(), (D,), gamma.float(), beta.float(), eps=EPS)
    qkv_golden_unpadded = (ln_out_golden.to(torch.float32) @ w_qkv.to(torch.float32)).to(torch.bfloat16)

    q_padded_per_head = torch.zeros(NUM_HEADS, M, HEAD_DIM_PADDED, dtype=torch.bfloat16)
    k_padded_per_head = torch.zeros(NUM_HEADS, M, HEAD_DIM_PADDED, dtype=torch.bfloat16)
    v_padded_per_head = torch.zeros(NUM_HEADS, M, HEAD_DIM_PADDED, dtype=torch.bfloat16)
    for h in range(NUM_HEADS):
        q_start = h * HEAD_DIM_TRUE
        k_start = D + h * HEAD_DIM_TRUE
        v_start = 2 * D + h * HEAD_DIM_TRUE
        q_padded_per_head[h, :, :HEAD_DIM_TRUE] = qkv_golden_unpadded[:, q_start : q_start + HEAD_DIM_TRUE]
        k_padded_per_head[h, :, :HEAD_DIM_TRUE] = qkv_golden_unpadded[:, k_start : k_start + HEAD_DIM_TRUE]
        v_padded_per_head[h, :, :HEAD_DIM_TRUE] = qkv_golden_unpadded[:, v_start : v_start + HEAD_DIM_TRUE]

    # Build per-head torch SDPA outputs → concat to (M, NUM_HEADS * HEAD_DIM_PADDED).
    sdpa_per_head = torch.zeros(NUM_HEADS, M, HEAD_DIM_PADDED, dtype=torch.bfloat16)
    for h in range(NUM_HEADS):
        q = q_padded_per_head[h].to(torch.float32)
        k = k_padded_per_head[h].to(torch.float32)
        v = v_padded_per_head[h].to(torch.float32)
        scores = q @ k.T
        softmax_scores = torch.softmax(scores, dim=-1)
        sdpa_per_head[h] = (softmax_scores @ v).to(torch.bfloat16)
    # Concat heads horizontally: (M, NUM_HEADS * HEAD_DIM_PADDED) = (256, 1536).
    D_padded = NUM_HEADS * HEAD_DIM_PADDED  # 1536
    sdpa_concat = torch.zeros(M, D_padded, dtype=torch.bfloat16)
    for h in range(NUM_HEADS):
        sdpa_concat[:, h * HEAD_DIM_PADDED : (h + 1) * HEAD_DIM_PADDED] = sdpa_per_head[h]

    tensors = build_tensors_for_fused_attention_block(device, x, gamma, beta, w_qkv_torch=w_qkv)
    SigLIPAttentionBlockFused.op(*tensors, eps=EPS)

    import ttnn as _ttnn

    fused_scratch_tt = tensors[6]
    fused_scratch_torch = _ttnn.to_torch(fused_scratch_tt)

    ln1_num_cores = SigLIPAttentionBlockFused.LN1_NUM_CORES  # 8
    head_dim_padded = HEAD_DIM_PADDED  # 96
    head_dim_n_tiles = SigLIPAttentionBlockFused.QKV_N_TILES_PER_CORE  # 3
    m_per_ln1_core = M // ln1_num_cores  # 32 (one tile-row)
    shard_h = 2 * SigLIPAttentionBlockFused.D_TILES * SigLIPAttentionBlockFused.TILE  # 2304

    # Each LN1 core's shard contains 48 tiles of sdpa_assembled_out at the
    # first 1536 rows (= 48 row-tiles × 32). Tile h*3 + d holds (32, 32) at
    # cols [d*32 : (d+1)*32] of head h.
    assembled_tiles_per_core = SigLIPAttentionBlockFused.NUM_HEADS * head_dim_n_tiles  # 48
    assembled_rows = assembled_tiles_per_core * SigLIPAttentionBlockFused.TILE  # 1536

    min_pcc = float("inf")
    worst = None
    for c in range(ln1_num_cores):
        shard_start = c * shard_h  # LN1 shards 0..7 at rows 0..(8*2304-1)
        assembled_region = fused_scratch_torch[shard_start : shard_start + assembled_rows, :]
        tile_stack = assembled_region.reshape(
            assembled_tiles_per_core, SigLIPAttentionBlockFused.TILE, SigLIPAttentionBlockFused.TILE
        )
        # Re-assemble into (m_per_ln1_core=32, D_padded=1536).
        per_core = torch.zeros(m_per_ln1_core, D_padded, dtype=assembled_region.dtype)
        for h in range(SigLIPAttentionBlockFused.NUM_HEADS):
            for d in range(head_dim_n_tiles):
                tile_idx = h * head_dim_n_tiles + d
                col_start = h * head_dim_padded + d * SigLIPAttentionBlockFused.TILE
                per_core[:, col_start : col_start + SigLIPAttentionBlockFused.TILE] = tile_stack[tile_idx]

        expected = sdpa_concat[c * m_per_ln1_core : (c + 1) * m_per_ln1_core, :]
        p = pcc(expected, per_core)
        if p < min_pcc:
            min_pcc = p
            worst = c

    print(
        f"\nPCC (SDPA assembled output on LN1 row, min/{ln1_num_cores} LN1 cores) = "
        f"{min_pcc:.6f} (worst LN1 core={worst})"
    )
    print(f"  per-core shape={(m_per_ln1_core, D_padded)}, dtype={per_core.dtype}")
    # Assembly is a pure NoC copy of Attn @ V's output — no new compute, so
    # PCC should match Commit 10's gate.
    assert min_pcc >= 0.996, f"min assembled probe PCC {min_pcc} below 0.996"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_attention_block_fused_with_oproj(device):
    """Full attention block: fused (LN1+QKV+SDPA+assembly) → O-proj (Commit 12).

    Pi0 fused attention block doesn't have spare L1 to host O-proj's
    (1536, 1536) weight as one more in-kernel stage. Instead this commit
    composes the fused-kernel output with the standalone SigLIPOprojMatmulOp
    (separate dispatch on a 36-core grid, K=1152 unpadded).

    Pipeline:
      1. Run SigLIPAttentionBlockFused.op(). Read assembled_out from
         fused_scratch on LN1 row → (256, 1536) padded.
      2. Strip head_dim padding → (256, 1152) unpadded.
      3. Deallocate fused tensors to free L1 for O-proj.
      4. Run SigLIPOprojMatmulOp.op() with W_o (1152, 1152) bfp8.
      5. PCC against full torch golden: assembled @ W_o.
    """
    import importlib.util

    # Lazy-import standalone oproj op (lives under tests/perf, no proper module
    # path). We sys.path-injected the dir earlier for golden_fc1.
    spec_path = Path(__file__).resolve().parents[2] / "tests" / "perf" / "oproj_op.py"
    spec = importlib.util.spec_from_file_location("oproj_op", spec_path)
    oproj_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oproj_module)
    SigLIPOprojMatmulOp = oproj_module.SigLIPOprojMatmulOp
    build_tensors_for_oproj_test = oproj_module.build_tensors_for_oproj_test

    import ttnn as _ttnn

    x, gamma, beta = make_inputs(seed=42)
    w_qkv = make_qkv_weight(seed=7)
    w_oproj = make_oproj_weight(seed=11)

    # Full torch golden.
    ln_out_golden = F.layer_norm(x.float(), (D,), gamma.float(), beta.float(), eps=EPS)
    qkv_golden_unpadded = (ln_out_golden.to(torch.float32) @ w_qkv.to(torch.float32)).to(torch.bfloat16)
    sdpa_concat_unpadded = torch.zeros(M, D, dtype=torch.bfloat16)
    for h in range(NUM_HEADS):
        q = qkv_golden_unpadded[:, h * HEAD_DIM_TRUE : (h + 1) * HEAD_DIM_TRUE].to(torch.float32)
        k = qkv_golden_unpadded[:, D + h * HEAD_DIM_TRUE : D + (h + 1) * HEAD_DIM_TRUE].to(torch.float32)
        v = qkv_golden_unpadded[:, 2 * D + h * HEAD_DIM_TRUE : 2 * D + (h + 1) * HEAD_DIM_TRUE].to(torch.float32)
        scores = q @ k.T
        attn = torch.softmax(scores, dim=-1) @ v
        sdpa_concat_unpadded[:, h * HEAD_DIM_TRUE : (h + 1) * HEAD_DIM_TRUE] = attn.to(torch.bfloat16)
    oproj_golden = (sdpa_concat_unpadded.to(torch.float32) @ w_oproj.to(torch.float32)).to(torch.bfloat16)

    # ---- Step 1+2: run fused, read assembled, strip padding -----------------
    tensors = build_tensors_for_fused_attention_block(device, x, gamma, beta, w_qkv_torch=w_qkv)
    SigLIPAttentionBlockFused.op(*tensors, eps=EPS)

    fused_scratch_tt = tensors[6]
    fused_scratch_torch = _ttnn.to_torch(fused_scratch_tt)

    ln1_num_cores = SigLIPAttentionBlockFused.LN1_NUM_CORES
    head_dim_n_tiles = SigLIPAttentionBlockFused.QKV_N_TILES_PER_CORE
    shard_h = 2 * SigLIPAttentionBlockFused.D_TILES * SigLIPAttentionBlockFused.TILE  # 2304
    assembled_tiles_per_core = NUM_HEADS * head_dim_n_tiles  # 48
    assembled_rows_per_core = assembled_tiles_per_core * SigLIPAttentionBlockFused.TILE  # 1536
    m_per_ln1_core = M // ln1_num_cores  # 32
    D_padded = NUM_HEADS * HEAD_DIM_PADDED  # 1536

    # Re-assemble (256, 1536) from the 8 LN1 cores' 48-tile blocks.
    assembled_padded = torch.zeros(M, D_padded, dtype=fused_scratch_torch.dtype)
    for c in range(ln1_num_cores):
        shard_start = c * shard_h
        region = fused_scratch_torch[shard_start : shard_start + assembled_rows_per_core, :]
        tile_stack = region.reshape(
            assembled_tiles_per_core, SigLIPAttentionBlockFused.TILE, SigLIPAttentionBlockFused.TILE
        )
        per_core = torch.zeros(m_per_ln1_core, D_padded, dtype=region.dtype)
        for h in range(NUM_HEADS):
            for d in range(head_dim_n_tiles):
                tile_idx = h * head_dim_n_tiles + d
                col_start = h * HEAD_DIM_PADDED + d * SigLIPAttentionBlockFused.TILE
                per_core[:, col_start : col_start + SigLIPAttentionBlockFused.TILE] = tile_stack[tile_idx]
        assembled_padded[c * m_per_ln1_core : (c + 1) * m_per_ln1_core, :] = per_core

    # Strip per-head padding.
    assembled_unpadded = unpad_sdpa_assembled(assembled_padded)
    assert assembled_unpadded.shape == (M, D)

    # ---- Step 3: deallocate fused tensors to free L1 for O-proj ----------
    for tt_tensor in tensors:
        _ttnn.deallocate(tt_tensor)

    # ---- Step 4: run standalone O-proj -----------------------------------
    activation_tt, weight_tt, output_tt = build_tensors_for_oproj_test(
        device, w_oproj, assembled_unpadded, num_cores=36
    )
    SigLIPOprojMatmulOp.op(activation_tt, weight_tt, output_tt, num_cores=36)

    # O-proj output is (M=256, N=1152) width-sharded on 36 cores. to_torch
    # returns the un-sharded (M, N) view.
    oproj_device = _ttnn.to_torch(output_tt)
    assert oproj_device.shape == (M, D), f"O-proj output shape {tuple(oproj_device.shape)} != {(M, D)}"

    p = pcc(oproj_golden, oproj_device)
    print(f"\nPCC (fused attention block + O-proj, vs full torch) = {p:.6f}")
    print(f"  output shape={tuple(oproj_device.shape)}, dtype={oproj_device.dtype}")
    # O-proj stacks another bfp8/HiFi2 matmul on top of softmax + Attn @ V.
    # Same precision regime as the underlying probes; gate stays at 0.996.
    assert p >= 0.996, f"fused+O-proj PCC {p} below 0.996"


# pi05_base weights live in the HuggingFace cache; the older /storage path
# referenced by test_encoder_block_real.py is no longer accessible from this
# host. Resolve dynamically to be portable across cache snapshots.
_PI05_HF_CACHE_DIR = Path.home() / ".cache/huggingface/hub/models--lerobot--pi05_base/snapshots"
_PI05_WEIGHTS_PATHS = sorted(_PI05_HF_CACHE_DIR.glob("*/model.safetensors")) if _PI05_HF_CACHE_DIR.exists() else []
PI05_WEIGHTS_PATH = str(_PI05_WEIGHTS_PATHS[0]) if _PI05_WEIGHTS_PATHS else "<not-available>"
VP = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0."


def _load_real_layer0_attention_weights():
    """Load layer-0 SigLIP attention weights from the pi0_base checkpoint.

    Returns (ln1_w, ln1_b, qkv_w_unpadded, qkv_b, o_w, o_b) all bfloat16.
    HF stores weights as (out, in); we transpose to (in, out) for x @ W form.
    """
    from safetensors.torch import load_file

    sd = load_file(PI05_WEIGHTS_PATH)

    def g(name):
        return sd[f"{VP}{name}"].to(torch.bfloat16)

    ln1_w = g("layer_norm1.weight")
    ln1_b = g("layer_norm1.bias")
    wq = g("self_attn.q_proj.weight").T.contiguous()  # (D, D)
    wk = g("self_attn.k_proj.weight").T.contiguous()
    wv = g("self_attn.v_proj.weight").T.contiguous()
    bq = g("self_attn.q_proj.bias")  # (D,)
    bk = g("self_attn.k_proj.bias")
    bv = g("self_attn.v_proj.bias")
    qkv_w_unpadded = torch.cat([wq, wk, wv], dim=1).contiguous()  # (D, 3D=3456)
    qkv_b = torch.cat([bq, bk, bv], dim=0).contiguous()  # (3D,)
    o_w = g("self_attn.out_proj.weight").T.contiguous()  # (D, D)
    o_b = g("self_attn.out_proj.bias")  # (D,)
    return ln1_w, ln1_b, qkv_w_unpadded, qkv_b, o_w, o_b


def _make_real_input_activation(seed: int = 42) -> torch.Tensor:
    """Real patch_embed + pos_embed output for a deterministic synthetic image.
    Shape (M=256, D=1152) bf16 — distribution-realistic SigLIP layer-0 input.

    Inlined here (not using golden_fc1.make_real_activation) because that
    helper hard-codes the /storage path which isn't available on this host.
    """
    from safetensors.torch import load_file

    sd = load_file(PI05_WEIGHTS_PATH)
    vision_prefix = "paligemma_with_expert.paligemma.model.vision_tower."
    pe_w = sd[f"{vision_prefix}vision_model.embeddings.patch_embedding.weight"].float()
    pe_b = sd[f"{vision_prefix}vision_model.embeddings.patch_embedding.bias"].float()
    pos_emb = sd[f"{vision_prefix}vision_model.embeddings.position_embedding.weight"].float()

    IMAGE_SIZE = 224
    PATCH_SIZE = 14
    g = torch.Generator().manual_seed(seed)
    pixel = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g, dtype=torch.float32)
    patches = F.conv2d(pixel, pe_w, bias=pe_b, stride=PATCH_SIZE)
    embeds = patches.flatten(2).transpose(1, 2) + pos_emb.unsqueeze(0)
    return embeds.squeeze(0).to(torch.bfloat16)


@pytest.mark.skipif(
    not Path(PI05_WEIGHTS_PATH).exists(),
    reason=f"pi0_base weights not found at {PI05_WEIGHTS_PATH}",
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_attention_block_fused_real_weights(device):
    """Real-weights acceptance for the fused attention block + O-proj (#5).

    Loads pi0_base SigLIP layer-0 weights (LN1, Q/K/V projections, O-proj)
    and runs the full attention block (fused kernel + standalone O-proj) on
    a real patch_embed + pos_embed activation.

    Limitations vs the full SigLIP attention math:
      * No 1/sqrt(head_dim) softmax scaling. The fused kernel applies pure
        softmax(Q @ K^T), so the torch reference here also omits the scale.
      * No QKV / O-proj biases. The kernel's matmul Op-struct doesn't
        currently support output bias, so the torch reference here also
        omits them.
      * No attention residual. We compare against the bare attention(LN1(x))
        output, not (attention(LN1(x)) + x).

    These are documented gaps that don't reflect kernel-precision issues;
    they're tracked for follow-up commits (bias support is the bigger ask
    — likely needs a matmul Op-struct update). PCC here validates that the
    REAL-WEIGHT DYNAMIC RANGE (vs synthetic random weights) doesn't trigger
    precision blow-ups in the bfp8/HiFi4 matmul + softmax + Attn @ V pipeline.
    """
    import importlib.util

    spec_path = Path(__file__).resolve().parents[2] / "tests" / "perf" / "oproj_op.py"
    spec = importlib.util.spec_from_file_location("oproj_op", spec_path)
    oproj_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oproj_module)
    SigLIPOprojMatmulOp = oproj_module.SigLIPOprojMatmulOp
    build_tensors_for_oproj_test = oproj_module.build_tensors_for_oproj_test

    import ttnn as _ttnn

    ln1_w, ln1_b, qkv_w_unpadded, qkv_b, o_w, o_b = _load_real_layer0_attention_weights()
    x = _make_real_input_activation(seed=42)
    assert x.shape == (M, D)

    # ---- Torch golden (no scaling, no bias, no residual) ------------------
    ln_out = F.layer_norm(x.float(), (D,), ln1_w.float(), ln1_b.float(), eps=EPS)
    qkv_golden_unpadded = (ln_out @ qkv_w_unpadded.float()).to(torch.bfloat16)
    sdpa_concat = torch.zeros(M, D, dtype=torch.bfloat16)
    for h in range(NUM_HEADS):
        q = qkv_golden_unpadded[:, h * HEAD_DIM_TRUE : (h + 1) * HEAD_DIM_TRUE].to(torch.float32)
        k = qkv_golden_unpadded[:, D + h * HEAD_DIM_TRUE : D + (h + 1) * HEAD_DIM_TRUE].to(torch.float32)
        v = qkv_golden_unpadded[:, 2 * D + h * HEAD_DIM_TRUE : 2 * D + (h + 1) * HEAD_DIM_TRUE].to(torch.float32)
        scores = q @ k.T  # NO 1/sqrt(d_k) scaling
        attn = torch.softmax(scores, dim=-1) @ v
        sdpa_concat[:, h * HEAD_DIM_TRUE : (h + 1) * HEAD_DIM_TRUE] = attn.to(torch.bfloat16)
    oproj_golden = (sdpa_concat.float() @ o_w.float()).to(torch.bfloat16)  # NO O-proj bias

    # ---- Device pipeline --------------------------------------------------
    tensors = build_tensors_for_fused_attention_block(
        device,
        x,
        ln1_w,  # gamma (LN1's learned scale)
        ln1_b,  # beta (LN1's learned shift)
        w_qkv_torch=qkv_w_unpadded,
    )
    SigLIPAttentionBlockFused.op(*tensors, eps=EPS)

    fused_scratch_tt = tensors[6]
    fused_scratch_torch = _ttnn.to_torch(fused_scratch_tt)

    # Reassemble (256, 1536) from LN1 shards (same logic as the
    # fused+oproj test above).
    ln1_num_cores = SigLIPAttentionBlockFused.LN1_NUM_CORES
    head_dim_n_tiles = SigLIPAttentionBlockFused.QKV_N_TILES_PER_CORE
    shard_h = 2 * SigLIPAttentionBlockFused.D_TILES * SigLIPAttentionBlockFused.TILE
    assembled_tiles_per_core = NUM_HEADS * head_dim_n_tiles
    assembled_rows_per_core = assembled_tiles_per_core * SigLIPAttentionBlockFused.TILE
    m_per_ln1_core = M // ln1_num_cores
    D_padded = NUM_HEADS * HEAD_DIM_PADDED

    assembled_padded = torch.zeros(M, D_padded, dtype=fused_scratch_torch.dtype)
    for c in range(ln1_num_cores):
        shard_start = c * shard_h
        region = fused_scratch_torch[shard_start : shard_start + assembled_rows_per_core, :]
        tile_stack = region.reshape(
            assembled_tiles_per_core, SigLIPAttentionBlockFused.TILE, SigLIPAttentionBlockFused.TILE
        )
        per_core = torch.zeros(m_per_ln1_core, D_padded, dtype=region.dtype)
        for h in range(NUM_HEADS):
            for d in range(head_dim_n_tiles):
                tile_idx = h * head_dim_n_tiles + d
                col_start = h * HEAD_DIM_PADDED + d * SigLIPAttentionBlockFused.TILE
                per_core[:, col_start : col_start + SigLIPAttentionBlockFused.TILE] = tile_stack[tile_idx]
        assembled_padded[c * m_per_ln1_core : (c + 1) * m_per_ln1_core, :] = per_core

    assembled_unpadded = unpad_sdpa_assembled(assembled_padded)

    for tt_tensor in tensors:
        _ttnn.deallocate(tt_tensor)

    activation_tt, weight_tt, output_tt = build_tensors_for_oproj_test(device, o_w, assembled_unpadded, num_cores=36)
    SigLIPOprojMatmulOp.op(activation_tt, weight_tt, output_tt, num_cores=36)
    oproj_device = _ttnn.to_torch(output_tt)

    p = pcc(oproj_golden, oproj_device)
    print(f"\nPCC (fused+O-proj on REAL pi0_base layer-0 weights, vs no-scale no-bias torch) = {p:.6f}")
    print(f"  output shape={tuple(oproj_device.shape)}, dtype={oproj_device.dtype}")
    # Real-weight dynamic range vs synthetic. The bfp8 quantization on real
    # weights can be slightly worse than on small random weights (real layer-
    # 0 Q/K/V have wider spread per channel). Gate at 0.99 to admit this.
    assert p >= 0.99, f"real-weights fused+O-proj PCC {p} below 0.99"
