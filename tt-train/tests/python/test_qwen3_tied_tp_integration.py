# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Integration regression for the qwen3 tied tensor-parallel path.

``test_vocab_parallel_embedding.py`` proves the VocabParallelEmbedding *module* is
correct against a replicated reference. What it cannot cover is the Qwen-specific
wiring layered on top of it: the embedding sharing the ColumnParallel LM-head
tensor, the parameter-name deduplication that keeps the shared tensor reported once,
construction under ``empty_init``, and the HF loader writing pretrained weights into
it. Those four together are what the 0.6B/1.7B tied TP path depends on, and they can
crash or silently map vocabulary rows wrong while every module test stays green.

The reference is the model's own **untied** path holding the same table in both of
its tensors -- that is what "tied" means, expressed with two tensors instead of one.
It is a genuinely independent implementation: the untied embedding is a
FeatureParallelEmbedding sharded on the *hidden* dim, so it shares no row-ownership
arithmetic with the vocab-sharded tied path. Both legs are shipped code, so nothing
here is a reimplementation that could be wrong in the same way.

  tied leg     VocabParallelEmbedding(padded_V) sharing lm_head's ColumnParallel
               weight -> one tensor, one gradient
  untied leg   FeatureParallelEmbedding(padded_V) + separate lm_head, both loaded
               with the same table -> two tensors, two gradients

  forward   logits must match
  backward  grad_tied == grad_embed + grad_lm_head   (the tying identity)

The vocab is chosen so a per-TP shard needs padding: 96 % 32 == 0, so the naive
alignment pads nothing, but 96 / 2 = 48 is not tile-aligned. That is the arithmetic
that silently shifted vocabulary rows on the 1.7B path at TP=8.
"""

import os
import sys

import numpy as np
import pytest
import torch
import ttml
import ttnn

# The fixture builds two models, loads pretrained weights into both, and runs a
# forward + loss + backward each. On a cold JIT cache -- which is what happens when
# a sibling module closed the mesh and this one has to reopen it -- that exceeds
# pytest.ini's global 300 s per-test timeout, so raise it for this module.
pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

TP_AXIS_SIZE = 2

# 96 % 32 == 0 but 96 / 2 = 48 is not a multiple of 32 -- a shard that needs padding.
VOCAB = 96
HIDDEN = 64
HEADS = KV_HEADS = TP_AXIS_SIZE
HEAD_DIM = 32
INTERMEDIATE = 128
SEQ = 32

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_QWEN3_EXAMPLE = os.path.abspath(os.path.join(_REPO_ROOT, "sources", "examples", "qwen3"))
# Force the qwen3 example dir to the front of sys.path so ``import utils`` resolves
# here even if a sibling example dir is already on the path.
if _QWEN3_EXAMPLE in sys.path:
    sys.path.remove(_QWEN3_EXAMPLE)
sys.path.insert(0, _QWEN3_EXAMPLE)


# The qwen3 example ships a top-level ``utils`` package, and so do sibling example
# dirs (e.g. examples/grpo, imported by test_grpo_trainer). In a single pytest session
# another test module may have already imported its own ``utils`` first, caching it in
# sys.modules and shadowing ours (sys.path.insert cannot override an already-imported
# module). Evict any cached ``utils*`` so the imports below resolve against
# _QWEN3_EXAMPLE, then restore the sibling's modules so we do not break whichever test
# imported them.
#
# Everything the tests need from the example is imported *here*, not lazily inside the
# helpers: after the restore below, a runtime ``from utils...`` would resolve against
# the sibling package again and fail.
def _qwen3_cached():
    """Cached modules that belong to the qwen3 example (``utils*`` plus its top-level
    modules), identified by where they were loaded from."""
    names = []
    for name, mod in list(sys.modules.items()):
        if name == "utils" or name.startswith("utils."):
            names.append(name)
        elif (getattr(mod, "__file__", None) or "").startswith(_QWEN3_EXAMPLE + os.sep):
            names.append(name)
    return names


# Evicting only ``utils*`` is not enough. ``utils.context_managers`` holds the module
# level ``_empty_init`` flag that ``empty_init()`` toggles and ``make_sharded_weight``
# reads. If a sibling qwen3 test module already imported ``model_qwen3_distributed``
# during *its* eviction window, that cached module keeps a reference to *its* copy of
# ``utils.context_managers`` -- so ``with empty_init()`` here would set one copy's flag
# while the model checks another's, silently skipping the tile-padded allocation. Evict
# the example's own modules too so this window is self-consistent.
_saved_utils = {k: sys.modules.pop(k) for k in _qwen3_cached()}
try:
    from ttml.models.qwen3 import Qwen3Config  # noqa: E402
    from utils.context_managers import empty_init  # noqa: E402
    from utils.tensor_utils import create_input_tensor_from_torch  # noqa: E402
    from model_qwen3_distributed import (  # noqa: E402
        DistributedQwen3ForCausalLM,
        load_weights_from_hf_distributed,
        tp_padded_vocab_size,
    )
    from generate import create_causal_mask_tensor  # noqa: E402
finally:
    for _k in _qwen3_cached():
        del sys.modules[_k]
    sys.modules.update(_saved_utils)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qwen3_config():
    return Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ,
    )


def _hf_state_dict():
    """A full pretrained-shaped state dict, from transformers' own Qwen3.

    Built from a config, so nothing is downloaded. Taking the names and shapes from
    the reference implementation rather than hand-writing them keeps this correct
    across upstream changes -- K and V are fused into one ttml tensor, for instance,
    which a hand-built dict would have to know about.
    """
    from transformers import Qwen3Config as HFConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    torch.manual_seed(0)
    hf = Qwen3ForCausalLM(
        HFConfig(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_hidden_layers=1,
            num_attention_heads=HEADS,
            num_key_value_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            max_position_embeddings=SEQ,
            tie_word_embeddings=False,
        )
    )
    sd = {k: v.detach().clone().float() for k, v in hf.state_dict().items()}
    # The untied leg only references the tied leg if both its tensors hold the same
    # table. transformers gives them independent random values when untied.
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()
    return sd


def _spanning_ids():
    """``SEQ`` ids covering every TP rank's vocab window, plus the window stride.

    This is load-bearing, not incidental. A row shift only displaces ranks *after* the
    first -- rank 0's offset is 0 whether the stride is right or wrong -- so ids
    confined to low values cannot detect one.
    """
    stride = tp_padded_vocab_size(VOCAB) // TP_AXIS_SIZE
    per_rank = SEQ // TP_AXIS_SIZE
    ids = []
    for rank in range(TP_AXIS_SIZE):
        lo = rank * stride + 1  # +1 keeps id 0 out, so no row is hit by padding alone
        ids += list(range(lo, min(lo + per_rank, VOCAB)))
    assert len(ids) == SEQ, f"could not fill {SEQ} ids across {TP_AXIS_SIZE} windows: {ids}"
    assert len({i // stride for i in ids}) == TP_AXIS_SIZE, f"ids do not span every rank: {ids}"
    return np.array(ids, dtype=np.int32), stride


def _gather(tensor, dim):
    """Reassemble a TP-sharded tensor along ``dim``; on a [1, TP] mesh a plain concat
    of every device in mesh order is the full tensor."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, dim)
    return tensor.to_numpy(ttnn.DataType.FLOAT32, composer=composer)


def _run_leg(tie, state_dict, ids_torch, targets_np, mask):
    """Build one leg under empty_init, load the table, forward, loss, backward."""
    ctx = ttml.autograd.AutoContext.get_instance()
    device = ctx.get_device()
    cfg = _qwen3_config()

    with empty_init():
        model = DistributedQwen3ForCausalLM(cfg, tie_word_embeddings=tie, shard_dim=1)
    load_weights_from_hf_distributed(model, state_dict, cfg, tie_word_embeddings=tie, shard_dim=1)

    ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    sharded_logits = model(create_input_tensor_from_torch(ids_torch, device), mask)
    logits = _gather(ttml.ops.distributed.all_gather(sharded_logits, dim=3, cluster_axis=1), 0)[:1]

    targets = ttml.autograd.Tensor.from_numpy(targets_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
    loss = ttml.ops.distributed.vocab_parallel_cross_entropy_loss(sharded_logits, targets, cluster_axis=1)
    loss_value = float(_gather(loss, 0)[0].mean())
    loss.backward(False)

    params = model.parameters()
    lm_name = next(n for n in params if n.endswith("lm_head/weight"))
    emb_name = next((n for n in params if n.endswith("embed_tokens/weight")), None)

    out = {
        "model": model,
        "params": params,
        "logits": logits,
        "loss": loss_value,
        # lm_head is ColumnParallel: sharded on dim 2 (vocab rows).
        "lm_weight": _gather(params[lm_name], 2)[0, 0],
        "lm_grad": _gather(params[lm_name].get_grad_tensor(), 2)[0, 0],
        "vocab_param_names": sorted(n for n in params if "embed_tokens" in n or "lm_head" in n),
    }
    if emb_name is not None:
        # FeatureParallelEmbedding is sharded on dim 3 (hidden), not dim 2.
        out["emb_grad"] = _gather(params[emb_name].get_grad_tensor(), 3)[0, 0]
    ctx.reset_graph()
    return out


@pytest.fixture(scope="module")
def legs(tp_mesh):
    """Both legs, built and run once -- each is a model build, HF load, and backward."""
    pytest.importorskip("transformers", reason="needs transformers for reference weight shapes")
    state_dict = _hf_state_dict()

    ids, stride = _spanning_ids()
    ids_np = ids.reshape(1, 1, 1, SEQ)
    targets_np = np.roll(ids, -1).reshape(1, SEQ).astype(np.uint32)
    mask = create_causal_mask_tensor(SEQ, ttml.autograd.AutoContext.get_instance().get_device())

    return {
        "table": state_dict["model.embed_tokens.weight"].numpy(),
        "ids": ids,
        "stride": stride,
        "tied": _run_leg(True, state_dict, torch.from_numpy(ids_np), targets_np, mask),
        "untied": _run_leg(False, state_dict, torch.from_numpy(ids_np), targets_np, mask),
    }


# ---------------------------------------------------------------------------
# 1. The embedding and the LM head are one parameter
# ---------------------------------------------------------------------------


def test_embedding_and_lm_head_are_one_parameter(legs):
    """The tied path must share a single tensor, not two tensors of equal shape.

    Shape equality is not enough: two independently allocated tables of the same
    shape would satisfy it while training two halves of a split weight.
    """
    model = legs["tied"]["model"]
    embed = model.model.embed_tokens.weight
    head = model.lm_head.weight

    assert embed is head, "embed_tokens.weight and lm_head.weight must be the same Parameter"
    assert embed.tensor is head.tensor, "the two roles must reference one device tensor"


def test_tied_weight_is_reported_once(legs):
    """Parameter-name deduplication: one tensor, one entry, under the LM-head name.

    ``AbstractModuleBase.__setattr__`` would register the shared tensor under a
    second (``model/embed_tokens/weight``) or third (``model/tied_embed_weight``)
    name, which would optimise it more than once per step.
    """
    tied_names = legs["tied"]["vocab_param_names"]
    untied_names = legs["untied"]["vocab_param_names"]

    assert len(tied_names) == 1, f"tied path must expose exactly one vocab parameter, got {tied_names}"
    assert tied_names[0].endswith("lm_head/weight"), tied_names
    assert not any("tied_embed" in n for n in legs["tied"]["params"]), "the tying helper leaked a parameter name"
    # The untied path is the contrast: genuinely two parameters.
    assert len(untied_names) == 2, untied_names


# ---------------------------------------------------------------------------
# 2. Pretrained weights land in the tied parameter
# ---------------------------------------------------------------------------


def test_pretrained_table_loads_into_the_tied_parameter(legs):
    """Every row and column of the checkpoint must arrive, with padding zeroed.

    Covers the loader end of the path: the tied weight is allocated at the padded
    width while the checkpoint is ``vocab_size`` rows, so the loader has to place the
    real rows and zero the remainder. A row shift or a transposed load shows up here.
    """
    table = legs["table"]  # [VOCAB, HIDDEN] float32, pre-bf16
    loaded = legs["tied"]["lm_weight"]  # [padded, HIDDEN] as stored on device

    padded = loaded.shape[0]
    assert padded % (32 * TP_AXIS_SIZE) == 0, f"padded width {padded} is not a multiple of 32*tp"
    assert padded >= VOCAB and (padded // TP_AXIS_SIZE) % 32 == 0

    # bf16 storage: compare against the checkpoint rounded the same way.
    expected = torch.from_numpy(table).bfloat16().float().numpy()
    np.testing.assert_allclose(loaded[:VOCAB], expected, rtol=0, atol=0)
    assert not loaded[VOCAB:].any(), "rows past vocab_size must be zero padding"


# ---------------------------------------------------------------------------
# 3. Forward and backward against the untied reference
# ---------------------------------------------------------------------------


def test_tied_forward_matches_untied_reference(legs):
    """Same table, same ids -> same logits, whichever way the table is sharded.

    The untied embedding shards the hidden dim and shares no row-ownership
    arithmetic with the vocab-sharded tied path, so agreement here exercises the
    vocab sharding against an independent placement.
    """
    ids, stride = legs["ids"], legs["stride"]
    assert (
        len({int(i) // stride for i in ids}) == TP_AXIS_SIZE
    ), f"ids must span every rank's vocab window or a row shift is invisible: {ids}"

    tied, untied = legs["tied"]["logits"], legs["untied"]["logits"]
    assert tied.shape == untied.shape, (tied.shape, untied.shape)
    np.testing.assert_allclose(tied, untied, rtol=1e-2, atol=1e-3)
    assert abs(legs["tied"]["loss"] - legs["untied"]["loss"]) < 1e-2, (
        legs["tied"]["loss"],
        legs["untied"]["loss"],
    )


def test_tied_backward_matches_untied_reference(legs):
    """grad(tied) == grad(embedding) + grad(LM head): the tying identity.

    This is the only check that exercises *both* of the shared tensor's roles. If the
    embedding half were dropped, misplaced, or double-counted, the sum would diverge
    -- and the embedding half is the majority of the signal, asserted below so the
    comparison cannot quietly become vacuous.
    """
    grad_tied = legs["tied"]["lm_grad"]
    grad_head = legs["untied"]["lm_grad"]
    grad_embed = legs["untied"]["emb_grad"]
    assert grad_tied.shape == grad_head.shape == grad_embed.shape

    peak = np.abs(grad_tied).max()
    assert peak > 0, "the tied weight received no gradient"

    # Without the embedding contribution the LM-head gradient alone is far off, so
    # this comparison is sensitive to losing or misplacing either half.
    head_only_error = np.abs(grad_tied - grad_head).max() / peak
    assert head_only_error > 0.1, (
        f"LM-head gradient alone is within {head_only_error:.1%} of the tied gradient; "
        "the embedding contribution is too small for this test to mean anything"
    )

    np.testing.assert_allclose(grad_tied, grad_embed + grad_head, rtol=2e-2, atol=1e-4)
