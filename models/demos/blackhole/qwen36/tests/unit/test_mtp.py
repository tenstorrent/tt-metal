# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Component PCC: single-device MTP drafter head vs the torch reference.

Runs against a checkpoint that carries mtp.* weights (the 3.8 family) — set
HF_MODEL to it; skips otherwise. Only the MTP tensors are loaded (no backbone),
and the shared embedding / LM head are replaced with small random stand-ins fed
identically to both implementations, so the test fits any single device.

    HF_MODEL=/path/to/qwen38-27b pytest models/demos/blackhole/qwen36/tests/unit/test_mtp.py -v
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc, get_pcc_threshold
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.weight_mapping import checkpoint_has_mtp, load_qwen36_mtp_state_dict

from .conftest import DEVICE_PARAMS

pytestmark = [run_for_blackhole(), pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)]

_STUB_VOCAB = 1024  # random stand-in embedding table height (token ids stay below this)
_STUB_LOGITS = 512  # random stand-in LM head width


class _EmbeddingStub:
    """Host-lookup embedding: same random table feeds TT and the torch reference."""

    def __init__(self, table, device):
        self.table = table
        self.device = device

    def __call__(self, tok_tt):
        tid = int(ttnn.to_torch(tok_tt).reshape(-1)[0])
        return ttnn.from_torch(
            self.table[tid].reshape(1, 1, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )


@pytest.fixture
def mtp_setup(device):
    """(args, tt_head, torch_ref) built on shared random embedding/LM-head."""
    from models.demos.blackhole.qwen36.reference.mtp_torch import MTPTorchReference
    from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTPHead

    args = Qwen36ModelArgs(mesh_device=device)
    if not checkpoint_has_mtp(args.CKPT_DIR):
        pytest.skip(f"checkpoint {args.CKPT_DIR} has no mtp.* weights")
    mtp_sd = load_qwen36_mtp_state_dict(args.CKPT_DIR)

    torch.manual_seed(1234)
    embed_table = torch.randn(_STUB_VOCAB, args.dim, dtype=torch.float32) * 0.02
    lm_head = torch.randn(_STUB_LOGITS, args.dim, dtype=torch.float32) * 0.02

    from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup

    rope = Qwen36RoPESetup(device, args)
    lm_head_tt = ttnn.from_torch(
        lm_head.T.contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    head = Qwen36MTPHead(
        device,
        args,
        mtp_sd,
        embedding=_EmbeddingStub(embed_table, device),
        lm_head_weight=lm_head_tt,
        rope=rope,
    )
    head.allocate_kv_cache(num_blocks=8)

    ref = MTPTorchReference(
        mtp_sd,
        embed_weight=embed_table,
        lm_head_weight=lm_head,
        num_heads=args.n_heads,
        num_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        rope_head_dim=args.rope_head_dim,
        rope_theta=args.rope_theta,
        norm_eps=args.norm_eps,
        max_seq_len=args.max_seq_len,
    )
    yield args, head, ref
    head.free_kv_cache()


def test_mtp_step_pcc(device, mtp_setup, request):
    """Sequential drafter steps with true-hidden inputs match the torch reference."""
    args, head, ref = mtp_setup
    torch.manual_seed(42)
    threshold = get_pcc_threshold(request)

    for pos in range(4):
        token = int(torch.randint(0, _STUB_VOCAB, (1,)))
        hidden = torch.randn(args.dim)
        tt_logits, tt_hidden = head.step(token, hidden, pos)
        ref_logits, ref_hidden = ref.step(token, hidden, pos)

        pcc_l = compute_pcc(ref_logits, tt_logits)
        pcc_h = compute_pcc(ref_hidden, tt_hidden)
        logger.info(f"MTP step pos={pos}: logits PCC={pcc_l:.6f} hidden PCC={pcc_h:.6f}")
        assert pcc_l > threshold, f"MTP logits PCC too low at pos {pos}: {pcc_l}"
        assert pcc_h > threshold, f"MTP hidden PCC too low at pos {pos}: {pcc_h}"


def test_mtp_chained_draft_pcc(device, mtp_setup, request):
    """Chained drafting (feeding the drafter's own hidden back) stays on-reference."""
    args, head, ref = mtp_setup
    torch.manual_seed(43)
    threshold = get_pcc_threshold(request)

    # Two catch-up steps with true hiddens.
    tt_hidden = ref_hidden = None
    for pos in range(2):
        token = int(torch.randint(0, _STUB_VOCAB, (1,)))
        hidden = torch.randn(args.dim)
        tt_logits, tt_hidden = head.step(token, hidden, pos)
        ref_logits, ref_hidden = ref.step(token, hidden, pos)

    # Chain 3 draft steps on the drafter's own hidden. Both sides consume the
    # REFERENCE greedy pick (teacher-forced) so a bf8-vs-fp32 near-tie argmax
    # flip cannot fork the trajectories and mask the PCC comparison.
    for j in range(3):
        pos = 2 + j
        ref_token = int(ref_logits.argmax())
        if int(tt_logits.argmax()) != ref_token:
            logger.warning(f"greedy near-tie at chain step {j}: tt={int(tt_logits.argmax())} ref={ref_token}")
        tt_logits, tt_hidden = head.step(ref_token, tt_hidden, pos)
        ref_logits, ref_hidden = ref.step(ref_token, ref_hidden, pos)
        pcc_l = compute_pcc(ref_logits, tt_logits)
        pcc_h = compute_pcc(ref_hidden, tt_hidden)
        logger.info(f"MTP chain step {j} (pos={pos}): logits PCC={pcc_l:.6f} hidden PCC={pcc_h:.6f}")
        assert pcc_l > threshold, f"chained logits PCC too low at step {j}: {pcc_l}"
        assert pcc_h > threshold, f"chained hidden PCC too low at step {j}: {pcc_h}"


def test_mtp_kv_slot_overwrite(device, mtp_setup, request):
    """Rejected-draft drafter KV is overwritten by the next catch-up pass.

    Write garbage at slots 2..3 (a rejected chain), then replay slots 2..3 with
    the reference inputs — the final step must match a fresh reference run,
    proving position-addressed KV overwrite (the drafter rollback mechanism).
    """
    args, head, ref = mtp_setup
    torch.manual_seed(44)
    threshold = get_pcc_threshold(request)

    seeds = [(int(torch.randint(0, _STUB_VOCAB, (1,))), torch.randn(args.dim)) for _ in range(4)]
    for pos in range(2):
        head.step(*seeds[pos], pos)
        ref.step(*seeds[pos], pos)

    # Garbage chain at slots 2..3 (as if drafted then rejected). The reference
    # skips it — the TT head must erase the difference by slot overwrite.
    head.step(int(torch.randint(0, _STUB_VOCAB, (1,))), torch.randn(args.dim), 2)
    head.step(int(torch.randint(0, _STUB_VOCAB, (1,))), torch.randn(args.dim), 3)

    # Catch-up replay of slots 2..3 with the real inputs.
    for pos in (2, 3):
        tt_logits, _ = head.step(*seeds[pos], pos)
        ref_logits, _ = ref.step(*seeds[pos], pos)

    pcc = compute_pcc(ref_logits, tt_logits)
    logger.info(f"MTP KV overwrite: final logits PCC={pcc:.6f}")
    assert pcc > threshold, f"KV overwrite PCC too low: {pcc}"
