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
    #   6:   fused_scratch
    #   7..10: xmm2, mean, var, ivar
    #   11..13: ln_out, x_residual, final_out
    #   14..15: qkv_act, ln_done_trigger
    #   16..18: qkv_w, qkv_out, qkv_done_trigger
    #   19..20: sdpa_q, sdpa_k_partial
    #   21..25: softmax max, exp, sum, isum, scaler  (Commit 9)
    final_out_tt = tensors[13]
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

    # qkv_act_tt is at positional index 14 (see test_attention_block_fused_ln_plus_residual).
    qkv_act_tt = tensors[14]
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

    # qkv_out_tt is at positional index 17.
    # Device shape is the padded (M, 4608); we slice each head's first
    # head_dim_true cols to recover the natural (M, 3456) layout before PCC.
    qkv_out_tt = tensors[17]
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

    # sdpa_q_tt is at positional index 19.
    sdpa_q_tt = tensors[19]
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

    shard_h = D_TILES * SigLIPAttentionBlockFused.TILE  # 1152
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
