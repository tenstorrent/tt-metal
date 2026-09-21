# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tt/sp_handoff.py: the host round-trip that injects sequence-parallel (SP)
prefill's full-layout caches into the TP=4 decode model.

Pure-torch mapping tests (no device) validate the full<->per-device-shard reshaping in
isolation. ``test_tp_cache_roundtrip_decode`` is the device acceptance test: decode must
produce IDENTICAL tokens/logits whether it continues straight from `prefill_tp`'s own
caches, or from caches that were snapshotted to full layout and re-injected.
"""
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.sp_handoff import (
    gdn_conv_full_to_tp_host,
    gdn_conv_tp_host_to_full,
    gdn_rec_full_to_tp_host,
    gdn_rec_tp_host_to_full,
    inject_into_tp_model,
    kv_full_to_tp_host,
    kv_tp_host_to_full,
    snapshot_tp_model_to_full_host,
)

# --------------------------------------------------------------------------- #
# Pure-torch mapping tests (no device; run before/without the marker file).
# --------------------------------------------------------------------------- #


def test_kv_full_to_tp_host_roundtrip():
    torch.manual_seed(0)
    num_devices, n_kv_heads, S, HD = 4, 2, 16, 256
    K = torch.randn(1, n_kv_heads, S, HD)
    V = torch.randn(1, n_kv_heads, S, HD)

    k_stacked, v_stacked = kv_full_to_tp_host(K, V, num_devices=num_devices, n_kv_heads=n_kv_heads)
    assert k_stacked.shape == (num_devices, 1, S, HD)
    assert v_stacked.shape == (num_devices, 1, S, HD)
    # devices 0,1 share head 0; devices 2,3 share head 1 (replicate_kv_weight order)
    assert torch.equal(k_stacked[0], k_stacked[1])
    assert torch.equal(k_stacked[2], k_stacked[3])
    assert not torch.equal(k_stacked[0], k_stacked[2])

    K2, V2 = kv_tp_host_to_full(k_stacked, v_stacked, num_devices=num_devices, n_kv_heads=n_kv_heads)
    assert torch.equal(K, K2)
    assert torch.equal(V, V2)


def test_gdn_rec_full_to_tp_host_roundtrip():
    torch.manual_seed(1)
    num_devices, Nv, Dk, Dv = 4, 16, 128, 128
    rec = torch.randn(1, Nv, Dk, Dv)

    stacked = gdn_rec_full_to_tp_host(rec, num_devices=num_devices)
    assert stacked.shape == (num_devices, Nv // num_devices, Dk, Dv)
    # device 1 holds value heads [4,8)
    assert torch.equal(stacked[1], rec[0, 4:8])

    rec2 = gdn_rec_tp_host_to_full(stacked, num_devices=num_devices)
    assert torch.equal(rec, rec2)


def test_gdn_conv_full_to_tp_host_roundtrip():
    torch.manual_seed(2)
    num_devices, K, key_dim, value_dim = 4, 4, 2048, 2048
    conv = torch.randn(1, K - 1, 2 * key_dim + value_dim)

    conv_list = gdn_conv_full_to_tp_host(conv, num_devices=num_devices, key_dim=key_dim, value_dim=value_dim)
    assert len(conv_list) == K
    D_tp = 2 * (key_dim // num_devices) + value_dim // num_devices
    for m, cm in enumerate(conv_list):
        assert cm.shape == (num_devices, 1, D_tp)
    assert torch.equal(conv_list[0], torch.zeros_like(conv_list[0]))  # m=0 is always zero

    conv2 = gdn_conv_tp_host_to_full(conv_list, num_devices=num_devices, key_dim=key_dim, value_dim=value_dim)
    assert torch.equal(conv, conv2)


def test_gdn_conv_full_to_tp_host_device1_placement():
    """conv_states[m] is a 3-slice [q|k|v] gather per device, NOT a contiguous 1536-wide cut
    (prepare_gdn_qkv-style indexing) -- build device 1's expected row by hand."""
    num_devices, key_dim, value_dim = 4, 2048, 2048
    kp, vp = key_dim // num_devices, value_dim // num_devices  # 512, 512
    # Unique value per channel so a wrong slice offset is detectable.
    conv = torch.arange(2 * key_dim + value_dim, dtype=torch.float32).reshape(1, 1, -1)

    conv_list = gdn_conv_full_to_tp_host(conv, num_devices=num_devices, key_dim=key_dim, value_dim=value_dim)
    d = 1
    row = conv[0, 0]
    expected = torch.cat(
        [
            row[d * kp : (d + 1) * kp],  # q slice: [512:1024)
            row[key_dim + d * kp : key_dim + (d + 1) * kp],  # k slice: [2560:3072)
            row[2 * key_dim + d * vp : 2 * key_dim + (d + 1) * vp],  # v slice: [4608:5120)
        ]
    )
    got = conv_list[1][d, 0]
    assert torch.equal(got, expected)

    # A naive contiguous cut at the same width would be WRONG -- confirms the gather is
    # not a plain slice.
    D_tp = kp + kp + vp
    contiguous_cut = row[d * D_tp : (d + 1) * D_tp]
    assert not torch.equal(got, contiguous_cut)


# --------------------------------------------------------------------------- #
# Device acceptance test.
# --------------------------------------------------------------------------- #


@torch.no_grad()
@parametrize_mesh_tp()
def test_tp_cache_roundtrip_decode(mesh_device, reset_seeds, ensure_gc):
    """decode_tp after inject_into_tp_model(snapshot_tp_model_to_full_host(...)) must match
    decode_tp continuing directly off prefill_tp's own caches: same greedy tokens, PCC>0.999
    logits per step."""
    nd = mesh_device.get_num_devices()
    assert nd > 1, "this test exercises the TP (num_devices>1) handoff path"

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=1024, n_layers=8)
    args = model.args
    vocab = args.vocab_size
    T, N_DEC = 512, 4
    torch.manual_seed(0)
    prompt = torch.randint(0, vocab, (T,)).tolist()

    # ---- prefill, then IMMEDIATELY snapshot (state at position T, before any decode step
    # mutates it further -- both paths below branch from this exact savepoint). ----
    model.reset_tp()
    logits0 = model.prefill_tp(torch.tensor([prompt], dtype=torch.long), valid_len=T)
    nxt = int(torch.argmax(logits0))
    kv_snap, gdn_snap = snapshot_tp_model_to_full_host(model)

    # ---- path A: decode_tp straight off prefill's own (in-place-mutating) caches ----
    tokens_A, logits_A = [nxt], []
    pos = T
    for _ in range(N_DEC):
        lg = model.decode_tp(tokens_A[-1], pos)
        logits_A.append(lg)
        tokens_A.append(int(torch.argmax(lg)))
        pos += 1
    logger.info(f"path A (direct) tokens: {tokens_A}")

    # ---- path B: reset, re-inject the position-T snapshot, decode from the SAME seed token ----
    model.reset_tp()
    inject_into_tp_model(model, kv_snap, gdn_snap)

    tokens_B, logits_B = [nxt], []
    pos = T
    for _ in range(N_DEC):
        lg = model.decode_tp(tokens_B[-1], pos)
        logits_B.append(lg)
        tokens_B.append(int(torch.argmax(lg)))
        pos += 1
    logger.info(f"path B (snapshot+inject) tokens: {tokens_B}")

    assert tokens_A[1:] == tokens_B[1:], f"greedy tokens diverged: A={tokens_A} B={tokens_B}"

    worst = 1.0
    for i in range(N_DEC):
        _, pcc = comp_pcc(logits_A[i].reshape(-1), logits_B[i].reshape(-1), 0.999)
        worst = min(worst, float(pcc))
        logger.info(f"decode step {i} logits PCC = {pcc}")
        assert float(pcc) >= 0.999, f"decode step {i} logits PCC {pcc} < 0.999"
    logger.info(f"PASSED: TP cache handoff round-trip matches direct decode; worst PCC = {worst:.6f}")
