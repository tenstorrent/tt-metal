# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side (numpy only, no device) checks of the DFlash drafter port (TT_PORT_PLAN D2).

Every derived input the device consumes is checked against a speculators golden step
(``scripts/tt/dump_dflash_golden.py`` in the speculators repo), using numpy only:

* RoPE cos/sin (YaRN / DeepSeek-YaRN) vs the reference rotary module output.
* Noise-block token ids, via the reference embedding output.
* Teacher rows (roll-by-one + block gather), via the reference teacher logits.
* Slot loss weights + KL formula, by recomputing the reference loss from its logits.
* Weight-name mapping round trip (k/v fused into kv_proj and back).
* Fixed-budget anchor sampling invariants.

Set ``DFLASH_GOLDEN`` to a golden ``.npz`` (or a directory of them); tests that need
it skip otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ttml.models.dflash.host import (
    DFlashDrafterConfig,
    anchor_candidates,
    build_host_batch,
    required_anchor_count,
    rope_cos_sin,
    sample_anchors,
)
from ttml.models.dflash.weights import speculators_to_ttml, ttml_to_speculators


def _golden_paths() -> list[Path]:
    env = os.environ.get("DFLASH_GOLDEN")
    if not env:
        return []
    p = Path(env)
    return sorted(p.glob("*.npz")) if p.is_dir() else [p]


GOLDENS = _golden_paths()
needs_golden = pytest.mark.skipif(not GOLDENS, reason="set DFLASH_GOLDEN to a speculators golden .npz")


class Golden:
    def __init__(self, path: Path):
        self.path = path
        self.z = np.load(path)
        self.meta = json.loads(bytes(self.z["meta"]).decode())
        self.cfg = DFlashDrafterConfig.from_golden_meta(self.meta)

    def __getitem__(self, key):
        return self.z[key]

    def weights(self, step: int) -> dict[str, np.ndarray]:
        prefix = f"w{step}/"
        return {k[len(prefix) :]: self.z[k] for k in self.z.files if k.startswith(prefix)}

    def host_batch(self, step: int):
        get = lambda k: self.z[f"in{step}/{k}"]  # noqa: E731
        return build_host_batch(
            self.cfg,
            hidden_states=get("hidden_states"),
            verifier_last_hidden_states=get("verifier_last_hidden_states"),
            input_ids=get("input_ids"),
            loss_mask=get("loss_mask"),
            document_ids=get("document_ids"),
            position_ids=get("position_ids"),
            anchor_positions=get("anchor_positions"),
            anchor_valid=get("anchor_valid"),
            gamma=self.meta["loss"]["gamma"],
        )


@pytest.fixture(params=GOLDENS, ids=lambda p: p.stem)
def golden(request) -> Golden:
    return Golden(request.param)


def _rms_norm(x, w, eps):
    x = x.astype(np.float64)
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w


def _log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


@needs_golden
def test_rope_tables_match_reference(golden):
    # numpy and torch float32 pow() can differ by one ulp in inv_freq; times a position of
    # a few thousand that moves the angle by ~1e-5 rad. 1e-4 is still ~40x below bf16
    # resolution of the device tables.
    for step in range(golden.meta["steps"]):
        hb = golden.host_batch(step)
        ref_cos = golden[f"act{step}/rope_cos"]
        ref_sin = golden[f"act{step}/rope_sin"]
        assert ref_cos.shape == hb.cos.shape
        np.testing.assert_allclose(hb.cos, ref_cos, atol=1e-4, rtol=0)
        cos, sin = rope_cos_sin(hb.positions, golden.cfg.head_dim, golden.cfg.rope_parameters)
        np.testing.assert_allclose(sin, ref_sin, atol=1e-4, rtol=0)


@needs_golden
def test_noise_tokens_match_reference_embedding(golden):
    embed = golden.weights(0)["embed_tokens.weight"]
    for step in range(golden.meta["steps"]):
        hb = golden.host_batch(step)
        np.testing.assert_array_equal(embed[hb.noise_token_ids.astype(np.int64)], golden[f"act{step}/embed"])


@needs_golden
def test_teacher_rows_match_reference_teacher_logits(golden):
    w = golden.weights(0)
    for step in range(golden.meta["steps"]):
        hb = golden.host_batch(step)
        teacher = _rms_norm(hb.teacher_hidden, w["verifier_norm.weight"], golden.cfg.rms_norm_eps) @ w[
            "verifier_lm_head.weight"
        ].T.astype(np.float64)
        np.testing.assert_allclose(teacher, golden[f"act{step}/teacher_logits"], rtol=1e-4, atol=1e-4)


@needs_golden
def test_slot_weights_reproduce_reference_loss(golden):
    """loss = sum_slots w * KL(p || q) and the square-form -(N*V) mean(...) + C identity."""
    for step in range(golden.meta["steps"]):
        hb = golden.host_batch(step)
        log_q = _log_softmax(golden[f"act{step}/logits"])
        log_p = _log_softmax(golden[f"act{step}/teacher_logits"])
        p = np.exp(log_p)
        kl = (p * (log_p - log_q)).sum(-1)
        loss = float((hb.slot_weights * kl).sum())
        np.testing.assert_allclose(loss, float(golden[f"loss{step}"]), rtol=1e-5)

        n, v = log_q.shape
        wp = p * hb.slot_weights[:, None]
        square_form = -(n * v) * (log_q * wp).mean() + (n * v) * (wp * log_p).mean()
        np.testing.assert_allclose(square_form, loss, rtol=1e-9)

        # Anchor slots and padded anchors carry no weight.
        assert not hb.slot_weights[:: golden.cfg.block_size].any()
        assert not hb.slot_weights[~np.repeat(hb.anchor_valid, golden.cfg.block_size)].any()


@needs_golden
def test_weight_mapping_round_trip(golden):
    spec = golden.weights(0)
    mapped = speculators_to_ttml(spec)
    assert all(v.ndim == 4 for v in mapped.values())
    assert not any(".k_proj." in k or ".v_proj." in k for k in mapped)
    shapes = {k: v.shape for k, v in spec.items()}
    back = ttml_to_speculators(mapped, shapes)
    expected = {k: v for k, v in spec.items() if k != "verifier_lm_head.weight"}
    assert set(back) == set(expected)
    for k, v in expected.items():
        np.testing.assert_array_equal(back[k], v, err_msg=k)


@needs_golden
def test_masks_are_square_and_layer_typed(golden):
    hb = golden.host_batch(0)
    S = hb.context_len + hb.num_slots
    assert set(hb.masks) == set(golden.cfg.layer_types)
    for m in hb.masks.values():
        assert m.shape == (S, S) and m.any(axis=1).all()
    if len(hb.masks) == 2:
        # The sliding mask is a strict subset of the full one (window + causal block).
        full, sliding = hb.masks["full_attention"], hb.masks["sliding_attention"]
        assert not (sliding & ~full).any() and (full & ~sliding).any()


def test_sample_anchors_fixed_budget(expect_error):
    rng = np.random.default_rng(0)
    T, B = 256, 8
    docs = np.full(T, -1)
    docs[:100], docs[100:190], docs[190:240] = 0, 1, 2
    loss_mask = (docs >= 0).astype(np.int64)
    loss_mask[:30] = 0
    positions, valid = sample_anchors(rng, loss_mask, docs, B, max_anchors_per_seq=5, anchor_budget=16)
    assert positions.shape == valid.shape == (16,)
    assert valid.sum() == 15
    eligible, run_ids = anchor_candidates(loss_mask, docs, B)
    assert set(positions[valid]) <= set(eligible)
    for run in np.unique(run_ids):
        in_run = set(eligible[run_ids == run])
        assert sum(p in in_run for p in positions[valid]) == 5
    with expect_error(ValueError, "budget"):
        sample_anchors(rng, loss_mask, docs, B, max_anchors_per_seq=6, anchor_budget=16)
    positions, valid = sample_anchors(
        rng, loss_mask, docs, B, max_anchors_per_seq=6, anchor_budget=16, overflow="subsample"
    )
    assert valid.all() and len(set(positions)) == 16 and set(positions) <= set(eligible)
    assert required_anchor_count(loss_mask, docs, B, 6) == 18


def test_config_validation(expect_error):
    base = dict(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=64,
        num_aux_layers=2,
        block_size=8,
        mask_token_id=62,
    )
    with expect_error(ValueError, "sliding_window"):
        DFlashDrafterConfig(layer_types=("sliding_attention",), sliding_window=0, **base)
    with expect_error(ValueError, "layer types"):
        DFlashDrafterConfig(layer_types=("chunked",), **base)


def test_packing_and_collate_match_speculators():
    """data.py vs the speculators sampler/collate (skips where speculators is absent)."""
    torch = pytest.importorskip("torch")
    sampler_mod = pytest.importorskip("speculators.train.distributed_batch_sampler")
    data_mod = pytest.importorskip("speculators.train.data")
    from ttml.models.dflash.data import collate, packed_batches

    rng = np.random.default_rng(1)
    lengths = rng.integers(20, 700, size=200)
    lengths[[3, 50]] = 1500  # longer than max_len: truncated, not dropped
    max_len, seed, H, L = 1024, 5, 8, 3
    for epoch in range(3):
        ref = sampler_mod.MultipackDistributedBatchSamplerV2(max_len, lengths, 1, 0, seed=seed)
        ref.set_epoch(epoch)
        ours = packed_batches(lengths, max_len, seed, epoch)
        ref_batches = list(ref)
        assert len(ours) == len(ref_batches)
        for a, b in zip(ours, ref_batches):
            np.testing.assert_array_equal(a, b)

    ref_collate = data_mod.create_collate_fn(max_len, H, L, dtype=torch.float32)
    for batch in packed_batches(lengths, max_len, seed, 0)[:5] + [np.array([3, 7])]:
        samples = []
        for i in batch:
            s = int(lengths[i])
            samples.append(
                {
                    "hidden_states": rng.standard_normal((s, L * H)).astype(np.float32),
                    "verifier_last_hidden_states": rng.standard_normal((s, H)).astype(np.float32),
                    "input_ids": rng.integers(0, 100, s),
                    "loss_mask": (rng.random(s) > 0.5).astype(np.float32),
                }
            )
        ours = collate(samples, max_len)
        ref = ref_collate(
            [
                {
                    **{k: torch.from_numpy(v) for k, v in smp.items()},
                    "lengths": torch.tensor([smp["input_ids"].shape[0]]),
                    "position_ids": torch.arange(smp["input_ids"].shape[0]),
                }
                for smp in samples
            ]
        )
        for key in (
            "hidden_states",
            "verifier_last_hidden_states",
            "input_ids",
            "loss_mask",
            "position_ids",
            "document_ids",
        ):
            np.testing.assert_array_equal(ours[key], ref[key][0].numpy(), err_msg=key)
