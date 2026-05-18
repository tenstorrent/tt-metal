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

    # build_tensors order tail: ..., final_out_tt, qkv_act_tt,
    # ln_done_trigger_tt, qkv_w_tt, qkv_out_tt, qkv_done_trigger_tt,
    # sdpa_q_tt, sdpa_k_probe_tt, sdpa_v_probe_tt. final_out_tt is -9.
    final_out_tt = tensors[-9]
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

    # qkv_act_tt is index -8 in the (... qkv_act_tt, ln_done_trigger_tt,
    # qkv_w_tt, qkv_out_tt, qkv_done_trigger_tt, sdpa_q_tt,
    # sdpa_k_probe_tt, sdpa_v_probe_tt) ordering.
    qkv_act_tt = tensors[-8]
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

    # qkv_out_tt is now at index -5 in build_tensors order (followed by
    # qkv_done_trigger_tt, sdpa_q_tt, sdpa_k_probe_tt, sdpa_v_probe_tt).
    # Device shape is the padded (M, 4608); we slice each head's first
    # head_dim_true cols to recover the natural (M, 3456) layout before PCC.
    qkv_out_tt = tensors[-5]
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

    # sdpa_q_tt is at index -3 in build_tensors order (followed by
    # sdpa_k_probe_tt and sdpa_v_probe_tt as of #11 Commit 5).
    sdpa_q_tt = tensors[-3]
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
def test_attention_block_fused_sdpa_kv_probe(device):
    """Validate K+V streaming probe — production NoC-read pattern.

    Each of the 32 SDPA workers reads ONE tile (32, 32) of its head's K from
    the head's K source core (QKV linear shard num_heads + h), and the same
    for V. The probe tiles are written to per-worker 1-tile CBs. Compares
    each worker's K tile against torch K head h, first 32 rows and first 32
    cols (the (0,0) tile of the (M=256, head_dim_padded=96) per-head matrix).
    Same for V.

    This is the same noc_async_read pattern that future SDPA compute commits
    will inline tile-by-tile during QK^T and Attn@V matmul iterations —
    no L1-resident K/V buffers needed.
    """
    x, gamma, beta = make_inputs(seed=42)
    w_qkv = make_qkv_weight(seed=7)

    ln_out_golden = F.layer_norm(x.float(), (D,), gamma.float(), beta.float(), eps=EPS)
    qkv_golden_unpadded = (ln_out_golden.to(torch.float32) @ w_qkv.to(torch.float32)).to(torch.bfloat16)
    # Build per-head padded K and V matrices, then take each head's (0,0) tile.
    k_tile_per_head = torch.zeros(NUM_HEADS, 32, 32, dtype=torch.bfloat16)
    v_tile_per_head = torch.zeros(NUM_HEADS, 32, 32, dtype=torch.bfloat16)
    for h in range(NUM_HEADS):
        k_start = D + h * HEAD_DIM_TRUE  # K-region: cols [D..2D)
        v_start = 2 * D + h * HEAD_DIM_TRUE  # V-region: cols [2D..3D)
        # The padded (256, 96) per-head matrix has real data in [:, :72]; the
        # (0, 0) tile is rows 0..31, cols 0..31 — all 32 cols are real (since
        # 32 < HEAD_DIM_TRUE=72).
        k_tile_per_head[h] = qkv_golden_unpadded[:32, k_start : k_start + 32]
        v_tile_per_head[h] = qkv_golden_unpadded[:32, v_start : v_start + 32]

    tensors = build_tensors_for_fused_attention_block(device, x, gamma, beta, w_qkv_torch=w_qkv)
    SigLIPAttentionBlockFused.op(*tensors, eps=EPS)

    import ttnn as _ttnn

    sdpa_k_probe_tt = tensors[-2]
    sdpa_v_probe_tt = tensors[-1]
    sdpa_k_probe_torch = _ttnn.to_torch(sdpa_k_probe_tt)
    sdpa_v_probe_torch = _ttnn.to_torch(sdpa_v_probe_tt)

    sdpa_num_workers = SigLIPAttentionBlockFused.SDPA_NUM_CORES
    sdpa_grid_x = SigLIPAttentionBlockFused.SDPA_GRID_X
    num_workers_per_head = SigLIPAttentionBlockFused.NUM_SDPA_WORKERS_PER_HEAD

    # Each per-worker probe is (32, 32) tile → tensor shape (32*32, 32) = (1024, 32).
    expected_shape = (sdpa_num_workers * 32, 32)
    assert (
        sdpa_k_probe_torch.shape == expected_shape
    ), f"K probe shape {tuple(sdpa_k_probe_torch.shape)} != {expected_shape}"
    assert (
        sdpa_v_probe_torch.shape == expected_shape
    ), f"V probe shape {tuple(sdpa_v_probe_torch.shape)} != {expected_shape}"

    k_per_worker = sdpa_k_probe_torch.reshape(sdpa_num_workers, 32, 32)
    v_per_worker = sdpa_v_probe_torch.reshape(sdpa_num_workers, 32, 32)

    min_k_pcc = float("inf")
    min_v_pcc = float("inf")
    worst_k = None
    worst_v = None
    for worker_linear in range(sdpa_num_workers):
        y = worker_linear // sdpa_grid_x
        x_rel = worker_linear % sdpa_grid_x
        head_idx = (y // num_workers_per_head) * sdpa_grid_x + x_rel
        pk = pcc(k_tile_per_head[head_idx], k_per_worker[worker_linear])
        pv = pcc(v_tile_per_head[head_idx], v_per_worker[worker_linear])
        if pk < min_k_pcc:
            min_k_pcc = pk
            worst_k = (worker_linear, head_idx)
        if pv < min_v_pcc:
            min_v_pcc = pv
            worst_v = (worker_linear, head_idx)

    print(
        f"\nPCC (SDPA K streaming probe, min/32 workers) = {min_k_pcc:.6f} (worst worker={worst_k[0]}, head={worst_k[1]})"
    )
    print(
        f"PCC (SDPA V streaming probe, min/32 workers) = {min_v_pcc:.6f} (worst worker={worst_v[0]}, head={worst_v[1]})"
    )
    # Probe gate is 0.998 (lower than QKV matmul's 0.999): bfp8 + HiFi4 on a
    # single (32, 32) tile has higher per-tile variance than the full (256, 96)
    # comparison. Math is bit-identical to the QKV matmul output — only the
    # comparison window differs.
    assert min_k_pcc >= 0.998, f"min K probe PCC {min_k_pcc} below 0.998"
    assert min_v_pcc >= 0.998, f"min V probe PCC {min_v_pcc} below 0.998"
