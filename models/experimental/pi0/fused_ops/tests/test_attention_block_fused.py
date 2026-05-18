"""Fused attention_block kernel — tests.

Validates the multi-Op chaining mechanic + Commit 2's LN1→QKV receiver-pull
mcast:

  test_attention_block_fused_ln_plus_residual
    Single TRISC dispatch that calls LN1 then residual_add in sequence, with
    L1 CB chaining (no host round-trip). Math: out = LN1(x; gamma, beta) + x.
    Loose PCC ≥ 0.999 gate (typically lands at 0.999990+).

  test_attention_block_fused_ln_mcast_probe
    Same dispatch, but reads back qkv_act_cb from each of the 36 QKV
    receivers and checks that every receiver holds LN1(x). Confirms the
    NCRISC sender→receiver semaphore wait + 8× noc_async_read pipeline
    reproduces the LN1 output bit-stably on every receiver.

PCC target ≥ 0.999988 for the mcast probe (LN1's per-test floor).
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
EPS = 1e-6


def make_inputs(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, D, generator=g, dtype=torch.bfloat16) * 0.5
    gamma = torch.ones(D, dtype=torch.bfloat16) + torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.1
    beta = torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.05
    return x, gamma, beta


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_attention_block_fused_ln_plus_residual(device):
    x, gamma, beta = make_inputs(seed=42)

    # Golden: LN1(x) + x in fp32, output bf16.
    ln_out_golden = F.layer_norm(x.float(), (D,), gamma.float(), beta.float(), eps=EPS)
    y_golden = (ln_out_golden + x.float()).to(torch.bfloat16)

    tensors = build_tensors_for_fused_attention_block(device, x, gamma, beta)
    SigLIPAttentionBlockFused.op(*tensors, eps=EPS)

    import ttnn as _ttnn

    # build_tensors order: ..., final_out_tt, qkv_act_tt, ln_done_trigger_tt.
    final_out_tt = tensors[-3]
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
    # full (256, 1152) tensor verbatim — no per-receiver shuffling at this
    # stage; that comes when the QKV matmul lands in Commit 3.
    ln_out_golden = F.layer_norm(x.float(), (D,), gamma.float(), beta.float(), eps=EPS).to(torch.bfloat16)

    tensors = build_tensors_for_fused_attention_block(device, x, gamma, beta)
    SigLIPAttentionBlockFused.op(*tensors, eps=EPS)

    import ttnn as _ttnn

    # build_tensors order: ..., final_out_tt, qkv_act_tt, ln_done_trigger_tt.
    qkv_act_tt = tensors[-2]
    qkv_act_torch = _ttnn.to_torch(qkv_act_tt)

    num_receivers = SigLIPAttentionBlockFused.QKV_NUM_CORES
    assert qkv_act_torch.shape == (num_receivers * M, D), (
        f"qkv_act tensor shape {tuple(qkv_act_torch.shape)} != expected " f"{(num_receivers * M, D)}"
    )

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
