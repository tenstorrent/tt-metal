# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DFlash speculation with the **device** Qwen3.6-27B target and the host drafter.

Same loop as ``test_dflash_host.py``; only the target is swapped, from
:class:`~...reference.dflash.targets.HFTarget` to :class:`~...reference.dflash.targets.TtTarget`.
That swap is the whole point of the abstraction, and these tests are what proves it holds.

    MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_device.py

The load-bearing claim is unchanged and backend-independent: greedy speculation must emit exactly
the tokens the *same target* emits autoregressively. Here that also exercises the device rollback —
GDN state restored through ``_save_deltanet_states`` / ``_restore_deltanet_states``, paged KV left
to be overwritten.

Two tests are cheap and one is not:

* ``test_tt_taps_*`` and ``test_tt_block_forward_matches_stepwise`` truncate the model to a few
  layers and check the new device primitives (taps, all-position block logits) in isolation.
* ``test_tt_speculation_matches_autoregressive`` loads the full 64-layer 27B and generates. That is
  minutes of device time, so it is opt-in via ``DFLASH_RUN_TARGET=1``.
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.reference.dflash.drafters import TtDrafter
from models.demos.blackhole.qwen36.reference.dflash.generate import dflash_generate
from models.demos.blackhole.qwen36.reference.dflash.loader import (
    DFlashDrafterConfig,
    load_drafter,
    resolve_drafter_path,
    resolve_target_path,
)
from models.demos.blackhole.qwen36.reference.dflash.targets import TtTarget
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.dflash.config import load_drafter_state_dict
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64  # 64 * 64 = 4096 tokens, well past prompt + generation + one 128 bucket


def _page_table():
    return torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)


def _build(mesh_device, n_layers=None):
    """Load the model and bind paged KV + GDN state. ``n_layers=None`` loads the full stack."""
    model = Qwen36Model.from_pretrained(
        mesh_device,
        max_batch_size=1,
        max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE,
        n_layers=n_layers,
    )
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    return model


@torch.no_grad()
@parametrize_mesh_tp()
def test_tt_taps_have_the_right_shape_and_order(mesh_device, reset_seeds, ensure_gc):
    """Taps come back as one ``[1, S, n*dim]`` feature, concatenated in the order armed."""
    model = _build(mesh_device, n_layers=8)
    taps = [1, 5, 7]
    model.set_residual_taps(taps)

    tokens = torch.randint(0, model.args.vocab_size, (1, 40), dtype=torch.long)
    model.prefill_block_all_logits(tokens, _page_table(), actual_len=40, chunk_start=0)
    feature = model.take_taps(40)

    assert feature.shape == (1, 40, len(taps) * model.args.dim)
    assert torch.isfinite(feature).all()

    # Order is load-bearing (the drafter's `fc` was trained on one concat order), so a reversed
    # arming must produce a different — and correspondingly permuted — feature.
    model.set_residual_taps(list(reversed(taps)))
    model.prefill_block_all_logits(tokens, _page_table(), actual_len=40, chunk_start=0)
    reversed_feature = model.take_taps(40)
    dim = model.args.dim
    assert torch.equal(feature[..., :dim], reversed_feature[..., 2 * dim :])


@torch.no_grad()
@parametrize_mesh_tp()
def test_tt_taps_are_off_by_default(mesh_device, expect_error, reset_seeds, ensure_gc):
    """A model that was never armed records nothing — the production path is untouched."""
    model = _build(mesh_device, n_layers=8)
    tokens = torch.randint(0, model.args.vocab_size, (1, 40), dtype=torch.long)
    model.prefill_block_all_logits(tokens, _page_table(), actual_len=40, chunk_start=0)
    assert model._taps == {}
    with expect_error(AssertionError, "no taps armed"):
        model.take_taps(40)


@torch.no_grad()
@pytest.mark.parametrize("prefix", [128, 256, 64, 192], ids=lambda p: f"offset{p}")
@parametrize_mesh_tp()
def test_tt_block_forward_needs_bucket_alignment(mesh_device, prefix, reset_seeds, ensure_gc):
    """The raw device constraint: a block forward is exact only at a bucket-aligned ``chunk_start``.

    ``paged_fill_cache`` writes the whole padded bucket starting at block ``chunk_start // 64``, so
    consecutive segments are spaced by the BUCKET (128), not by their ``valid_len`` and not by the
    64-token paged block. Measured on T3K, aligned offsets come back at exactly 1.0 while unaligned
    ones do not. How BADLY an unaligned offset degrades depends on the token sample (0.16 and 0.27
    on one seed, 0.99 on another), so the assertion below is only that it is not exact.

    This is why :class:`TtTarget` anchors instead of using ``start`` directly. If a future device
    change makes an arbitrary offset exact, the unaligned cases here start passing and the anchoring
    can be dropped.
    """
    from models.common.utility_functions import comp_pcc

    model = _build(mesh_device, n_layers=8)
    page_table = _page_table()
    torch.manual_seed(0)
    total = prefix + 16
    tokens = torch.randint(0, model.args.vocab_size, (1, total), dtype=torch.long)

    whole = model.prefill_block_all_logits(tokens, page_table, actual_len=total, chunk_start=0)
    model.prefill_block_all_logits(tokens[:, :prefix], page_table, actual_len=prefix, chunk_start=0)
    split = model.prefill_block_all_logits(tokens[:, prefix:], page_table, actual_len=16, chunk_start=prefix)

    aligned = prefix % TtTarget.ANCHOR == 0
    _, pcc = comp_pcc(whole[:, prefix:], split, 0.99)
    logger.info(f"chunk_start={prefix} ({'aligned' if aligned else 'UNALIGNED'}): {pcc}")
    value = float(str(pcc).split()[-1]) if not isinstance(pcc, float) else pcc

    if aligned:
        assert value > 0.999, f"bucket-aligned chunk_start={prefix} should be exact, got {pcc}"
    else:
        assert value < 0.999, (
            f"chunk_start={prefix} is NOT bucket-aligned yet came back exact ({pcc}) — if the device "
            "now supports arbitrary offsets, TtTarget's anchoring is dead weight"
        )


@torch.no_grad()
@pytest.mark.parametrize("start", [40, 100, 128, 130], ids=lambda s: f"start{s}")
@parametrize_mesh_tp()
def test_tt_target_forward_matches_one_shot(mesh_device, start, reset_seeds, ensure_gc):
    """:class:`TtTarget` must be exact at ARBITRARY starts, which is what anchoring buys.

    Same comparison as the test above, but through the target instead of the raw model — including
    ``start`` values (40, 100, 130) where the raw path is badly wrong.
    """
    from models.common.utility_functions import comp_pcc

    model = _build(mesh_device, n_layers=8)
    torch.manual_seed(0)
    blk = min(16, TtTarget.ANCHOR - (start % TtTarget.ANCHOR))
    total = start + blk
    tokens = torch.randint(0, model.args.vocab_size, (1, total), dtype=torch.long)

    golden = model.prefill_block_all_logits(tokens, _page_table(), actual_len=total, chunk_start=0)

    target = TtTarget(model, [1, 5, 7], _page_table())
    target.reset()
    target.forward(tokens[:, :start], 0)
    logits, taps = target.forward(tokens[:, start:], start)

    assert logits.shape[1] == blk and taps.shape[1] == blk
    _, pcc = comp_pcc(golden[:, start:], logits, 0.99)
    logger.info(f"TtTarget block at start={start}: {pcc}")
    value = float(str(pcc).split()[-1]) if not isinstance(pcc, float) else pcc
    assert value > 0.99, f"TtTarget diverged from the one-shot prefill at start={start}: {pcc}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_tt_rejected_block_leaves_no_trace(mesh_device, reset_seeds, ensure_gc):
    """A rejected speculative block must not perturb what follows.

    On device this needs no explicit rollback at all: every :meth:`TtTarget.forward` restores GDN to
    the anchor and rewrites the whole bucket's KV, so the rejected tokens are simply overwritten.
    That is the device counterpart of the host's snapshot-and-replay, and it is why
    ``TtTarget.replays_after_rollback`` is False.
    """
    from models.common.utility_functions import comp_pcc

    model = _build(mesh_device, n_layers=8)
    vocab = model.args.vocab_size
    torch.manual_seed(0)
    prompt = torch.randint(0, vocab, (1, 40), dtype=torch.long)
    accepted = torch.randint(0, vocab, (1, 4), dtype=torch.long)
    rejected = torch.randint(0, vocab, (1, 16), dtype=torch.long)
    tail = torch.randint(0, vocab, (1, 8), dtype=torch.long)

    def run(with_rejected_block):
        target = TtTarget(model, [1, 5, 7], _page_table())
        target.reset()
        target.forward(prompt, 0)
        if with_rejected_block:
            snap = target.snapshot()
            target.forward(rejected, 40)  # 16 drafted, only 4 will survive
            target.restore(snap, 40)
        target.forward(accepted, 40)
        return target.forward(tail, 44)[0]

    golden = run(False)
    recovered = run(True)

    _, pcc = comp_pcc(golden, recovered, 0.99)
    logger.info(f"after a rejected block vs a clean run: {pcc}")
    value = float(str(pcc).split()[-1]) if not isinstance(pcc, float) else pcc
    assert value > 0.99, f"a rejected block perturbed the sequence that followed it: {pcc}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_tt_speculation_matches_autoregressive(mesh_device, reset_seeds, ensure_gc):
    """Full 27B on device + host drafter: speculation must not change the tokens.

    Opt-in — this loads the whole 64-layer stack and generates, which is minutes of device time.
    """
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B speculative generation on device")

    from transformers import AutoTokenizer

    drafter_path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(drafter_path)
    drafter = load_drafter(drafter_path)

    model = _build(mesh_device)  # full depth — taps need real checkpoint layer indices
    assert len(model.layers) == cfg.num_target_layers, (
        f"taps address checkpoint layers {cfg.target_layer_ids}; this model holds " f"{len(model.layers)} layers"
    )
    assert cfg.hidden_size == model.args.dim, f"drafter hidden {cfg.hidden_size} != target dim {model.args.dim}"
    assert cfg.vocab_size == model.args.vocab_size, "drafter and target must share a vocab (it borrows both heads)"

    target = TtTarget(model, cfg.target_layer_ids, _page_table(), checkpoint_path=resolve_target_path())

    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompt = tokenizer("The capital of France is", return_tensors="pt").input_ids
    max_new_tokens = 24

    stats = dflash_generate(drafter, target, prompt, max_new_tokens=max_new_tokens, return_stats=True)
    baseline = dflash_generate(drafter, target, prompt, max_new_tokens=max_new_tokens, block_size=1)

    logger.info(f"device generated: {tokenizer.decode(stats.output_ids[0, stats.num_input_tokens:])!r}")
    logger.info(
        f"device acceptance: mean {stats.mean_acceptance_length:.2f} tok/step over "
        f"{len(stats.acceptance_lengths)} steps (per-step {stats.acceptance_lengths}, "
        f"rollbacks {stats.num_rollbacks})"
    )

    assert torch.equal(stats.output_ids, baseline), (
        "device speculative output diverged from device autoregressive decoding:\n"
        f"  baseline    {baseline.tolist()}\n  speculative {stats.output_ids.tolist()}"
    )
    assert stats.mean_acceptance_length > 1.0, (
        f"drafter accepted {stats.mean_acceptance_length:.2f} tok/step against the device target — "
        "no better than no speculation; suspect the taps (order, layer ids, or TP gather)"
    )


@pytest.mark.timeout(0)
@torch.no_grad()
@parametrize_mesh_tp()
def test_tt_drafter_end_to_end(mesh_device, reset_seeds, ensure_gc):
    """The whole thing on device: ttnn drafter + ttnn target, nothing but token ids on host.

    This is the configuration the port exists for. The drafter's per-module PCC against the host
    reference is covered by ``tests/test_dflash_drafter_tp.py``; what this adds is that the two
    device halves compose — the target's taps stay on the mesh, feed the drafter's ``fc``, and the
    draft logits come back off the target's own resident LM head.

    The claim is the same one every configuration must satisfy: greedy speculation emits exactly
    what the same target emits autoregressively.
    """
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B with the device drafter")

    from transformers import AutoTokenizer

    drafter_path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(drafter_path)
    model = _build(mesh_device)  # full depth — the taps address real checkpoint layer indices
    assert len(model.layers) == cfg.num_target_layers

    target = TtTarget(model, cfg.target_layer_ids, _page_table(), device_taps=True)
    drafter = TtDrafter(
        TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(drafter_path), tt_ccl=model.tt_ccl),
        target,
    )

    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompt = tokenizer("The capital of France is", return_tensors="pt").input_ids
    max_new_tokens = 24

    stats = dflash_generate(drafter, target, prompt, max_new_tokens=max_new_tokens, return_stats=True)
    baseline = dflash_generate(drafter, target, prompt, max_new_tokens=max_new_tokens, block_size=1)

    logger.info(f"device drafter generated: {tokenizer.decode(stats.output_ids[0, stats.num_input_tokens:])!r}")
    logger.info(
        f"device drafter acceptance: mean {stats.mean_acceptance_length:.2f} tok/step over "
        f"{len(stats.acceptance_lengths)} steps (per-step {stats.acceptance_lengths})"
    )

    assert torch.equal(stats.output_ids, baseline), (
        "device-drafter speculation diverged from autoregressive decoding:\n"
        f"  baseline    {baseline.tolist()}\n  speculative {stats.output_ids.tolist()}"
    )
    assert stats.mean_acceptance_length > 1.0, (
        f"device drafter accepted {stats.mean_acceptance_length:.2f} tok/step — no better than no "
        "speculation; suspect the tap order into fc, or the all-reduce"
    )


@pytest.mark.timeout(0)
@torch.no_grad()
@parametrize_mesh_tp()
def test_tt_drafter_agrees_with_host_on_real_taps(mesh_device, reset_seeds, ensure_gc):
    """The rung between per-module PCC and end-to-end: same REAL taps, both drafters.

    ``tests/test_dflash_drafter_tp.py`` feeds the ttnn drafter synthetic taps it builds itself, so
    it validates the arithmetic but not the hand-off. This takes the target's actual residual
    stream and runs it through the host drafter and the device drafter. Two claims:

    1. the target's **device** taps equal its **host** taps (an exact-equality-class claim: same
       tensor, one read back and one kept on the mesh);
    2. the two drafters produce the same **hidden states** from them.

    The comparison is on hidden states, not tokens. Drafted tokens are an argmax over slots the
    drafter is predicting up to 15 positions ahead from mask tokens, and a bf16-vs-fp32 difference
    flips the winner in the far slots routinely (measured: 9/15 agreement, all disagreements past
    slot 0). Slot 0 is asserted because that is the one that decides whether any token is accepted;
    beyond that, draft quality is the end-to-end acceptance test's job.
    """
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    from models.common.utility_functions import comp_pcc
    from models.demos.blackhole.qwen36.reference.dflash.loader import load_drafter

    drafter_path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(drafter_path)
    sd = load_drafter_state_dict(drafter_path)
    model = _build(mesh_device)

    prompt = torch.tensor([[760, 6511, 314, 9338, 369]])  # "The capital of France is"
    start = prompt.shape[1]
    q_len = cfg.block_size
    block = torch.full((1, q_len), cfg.mask_token_id, dtype=torch.long)
    block[0, 0] = 11751  # a plausible anchor token

    def _to_host(t, width):
        host = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
        return host.reshape(-1, host.shape[-1])[:, :width].float().unsqueeze(0)

    # ---- host drafter on HOST taps ----
    host_target = TtTarget(model, cfg.target_layer_ids, _page_table(), device_taps=False)
    host_target.reset()
    _, host_taps = host_target.forward(prompt, 0, all_logits=False)

    ref = load_drafter(drafter_path)
    host_hidden = ref(
        target_hidden=host_taps.to(ref.dtype),
        noise_embedding=host_target.embed(block).to(ref.dtype),
        position_ids=torch.arange(start - host_taps.shape[1], start + q_len)[None],
    )[:, 1 - q_len :, :].float()

    # ---- device drafter on DEVICE taps, same target forward ----
    dev_target = TtTarget(model, cfg.target_layer_ids, _page_table(), device_taps=True)
    dev_target.reset()
    _, dev_taps = dev_target.forward(prompt, 0, all_logits=False)

    tt = TtDFlashDrafter(mesh_device, cfg, sd, tt_ccl=model.tt_ccl)
    tt_out = tt.forward(tt.project_taps(dev_taps), dev_target.embed_device(block), start)
    tt_hidden = _to_host(tt_out, cfg.hidden_size)[:, 1:, :]

    # 1. the taps must be the same tensor either way
    gathered = torch.cat(
        [
            ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)).reshape(1, -1, cfg.hidden_size)
            for t in dev_taps
        ],
        dim=-1,
    ).float()
    tap_ok, tap_out = comp_pcc(host_taps, gathered, 0.999)
    logger.info(f"device taps vs host taps: {tap_out}")
    assert tap_ok, f"the target's device taps do not match its host taps: {tap_out}"

    # 2. the two drafters must agree on the hidden states they produce from them
    ok, out = comp_pcc(host_hidden, tt_hidden, 0.99)
    logger.info(f"device drafter hidden vs host drafter hidden: {out}")
    assert ok, f"device drafter diverged from the host drafter on real taps: {out}"

    host_tok = torch.argmax(host_target.lm_head(host_hidden), dim=-1)
    dev_tok = torch.argmax(dev_target.lm_head(tt_hidden), dim=-1)
    agree = int((host_tok == dev_tok).sum())
    logger.info(f"host tokens:   {host_tok.tolist()}")
    logger.info(f"device tokens: {dev_tok.tolist()}  ({agree}/{q_len - 1} agree)")
    assert host_tok[0, 0] == dev_tok[0, 0], (
        f"the drafters disagree on slot 0 ({host_tok[0, 0].item()} vs {dev_tok[0, 0].item()}), which "
        "is the slot that decides whether speculation accepts anything at all"
    )
