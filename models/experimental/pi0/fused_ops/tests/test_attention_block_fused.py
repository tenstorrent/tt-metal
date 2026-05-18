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
N_QKV = 3 * D  # 3456
EPS = 1e-6


def make_inputs(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, D, generator=g, dtype=torch.bfloat16) * 0.5
    gamma = torch.ones(D, dtype=torch.bfloat16) + torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.1
    beta = torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.05
    return x, gamma, beta


def make_qkv_weight(seed: int = 7):
    """Match build_tensors_for_fused_attention_block's default w_qkv generator
    so the test golden uses the same weight matrix the device sees."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(D, N_QKV, generator=g, dtype=torch.bfloat16) * 0.05


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_attention_block_fused_ln_plus_residual(device):
    x, gamma, beta = make_inputs(seed=42)

    # Golden: LN1(x) + x in fp32, output bf16.
    ln_out_golden = F.layer_norm(x.float(), (D,), gamma.float(), beta.float(), eps=EPS)
    y_golden = (ln_out_golden + x.float()).to(torch.bfloat16)

    tensors = build_tensors_for_fused_attention_block(device, x, gamma, beta)
    SigLIPAttentionBlockFused.op(*tensors, eps=EPS)

    import ttnn as _ttnn

    # build_tensors order: ..., final_out_tt, qkv_act_tt, ln_done_trigger_tt,
    # qkv_w_tt, qkv_out_tt.  Indexing from the end: final_out_tt is -5.
    final_out_tt = tensors[-5]
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

    # qkv_act_tt is index -4 in the (... qkv_act_tt, ln_done_trigger_tt,
    # qkv_w_tt, qkv_out_tt) ordering.
    qkv_act_tt = tensors[-4]
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
    w_qkv = make_qkv_weight(seed=7)

    # Golden: LN1(x) @ W_qkv in fp32, output bf16. bfp8 quantization happens
    # on the device side when from_torch loads the weight; we keep the golden
    # in bf16 against the *bf16* weight so the PCC includes the bfp8 noise.
    ln_out_golden = F.layer_norm(x.float(), (D,), gamma.float(), beta.float(), eps=EPS)
    qkv_golden = (ln_out_golden.to(torch.float32) @ w_qkv.to(torch.float32)).to(torch.bfloat16)

    tensors = build_tensors_for_fused_attention_block(device, x, gamma, beta, w_qkv_torch=w_qkv)
    SigLIPAttentionBlockFused.op(*tensors, eps=EPS)

    import ttnn as _ttnn

    # qkv_out_tt is the last tensor in the build_tensors ordering.
    qkv_out_tt = tensors[-1]
    qkv_out_device = _ttnn.to_torch(qkv_out_tt)

    assert qkv_out_device.shape == (M, N_QKV), f"qkv_out shape {tuple(qkv_out_device.shape)} != {(M, N_QKV)}"

    p = pcc(qkv_golden, qkv_out_device)
    print(f"\nPCC (QKV matmul vs torch LN1(x) @ W_qkv) = {p:.6f}")
    print(f"  shape={tuple(qkv_out_device.shape)}, dtype={qkv_out_device.dtype}")
    # bfp8 weight + HiFi4 accumulator typically lands ≥ 0.999 on this shape.
    assert p >= 0.999, f"PCC {p} below 0.999 gate"
