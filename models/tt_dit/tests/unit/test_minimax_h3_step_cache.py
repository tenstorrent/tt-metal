# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-free checks of the H3 step cache (First Block Cache): its env switch, which steps it reuses, what a
reused step returns, and how the transformer's block loop hands the stack to it. torch tensors stand in for the
device tensors."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.tt_dit.models.transformers.minimax_h3 import step_cache as sc
from models.tt_dit.models.transformers.minimax_h3 import transformer_minimax_h3 as tm

NUM_BLOCKS = 4


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("0", None),
        ("0.08", (0.08, 0, -1)),
        ("0.1,4", (0.1, 4, -1)),
        (" 0.1,4,2 ", (0.1, 4, 2)),
    ],
)
def test_config_from_env(monkeypatch, value, expected):
    """Unset or a zero threshold is off; the fields are the threshold, the warmup steps and the cap on reused
    steps in a row, over one first block and no tail blocks."""
    monkeypatch.delenv(sc.STEP_CACHE_ENV, raising=False) if value is None else monkeypatch.setenv(
        sc.STEP_CACHE_ENV, value
    )
    config = sc.step_cache_config_from_env()
    if expected is None:
        assert config is None
        return
    got = (config.residual_diff_threshold, config.max_warmup_steps, config.max_continuous_cached_steps)
    assert got == expected
    assert (config.Fn_compute_blocks, config.Bn_compute_blocks) == (1, 0)


@pytest.mark.parametrize("value", ["on", "-0.1", "0.1,-1", "0.1,2,3,4", "0.1,x"])
def test_config_from_env_rejects(monkeypatch, expect_error, value):
    monkeypatch.setenv(sc.STEP_CACHE_ENV, value)
    with expect_error(ValueError, sc.STEP_CACHE_ENV):
        sc.step_cache_config_from_env()


def _cache(monkeypatch, value, valid_rows=None):
    """A cache over torch tensors, and the list its frees land in."""
    freed = []
    fake = SimpleNamespace(
        subtract=torch.subtract,
        multiply=torch.multiply,
        add=torch.add,
        abs=torch.abs,
        clone=torch.clone,
        sum=lambda x, dim, keepdim: torch.sum(x, dim=tuple(dim), keepdim=keepdim),
        to_torch=lambda x, mesh_composer: x,
        ConcatMeshToTensor=lambda device, dim: None,
        deallocate=lambda x: freed.append(x),
        using_distributed_env=lambda: False,
    )
    monkeypatch.setattr(sc, "ttnn", fake)
    monkeypatch.setenv(sc.STEP_CACHE_ENV, value)
    return sc.FirstBlockCache(sc.step_cache_config_from_env(), mesh_device=None, valid_rows=valid_rows), freed


def _run(cache, first_changes, pad_change=0.0):
    """One step per entry: block 0 adds that entry to rows 0-1 (and `pad_change` to row 2), each later block adds
    10 * (step + 1). Returns per step (the later blocks that ran, the output)."""
    steps = []
    for step, first_change in enumerate(first_changes):
        ran = []

        def run_block(index, hidden):
            if index == 0:
                return hidden + torch.tensor([first_change, first_change, pad_change]).reshape(1, 1, 3, 1)
            ran.append(index)
            return hidden + 10.0 * (step + 1)

        out = cache.run_stack(torch.full((1, 1, 3, 2), float(step)), run_block, NUM_BLOCKS)
        steps.append((ran, out[0, 0, 0, 0].item()))
    return steps


def test_reuse_sequence(monkeypatch):
    """A step is reused while block 0's change stays within the threshold of the last *computed* step's, so a slow
    drift ends the reuse: 1.03 is within 5% of 1.00, 1.06 is not although it is within 5% of 1.03. A reused step
    runs block 0 only and adds what blocks 1.. added at the last computed step."""
    cache, _ = _cache(monkeypatch, "0.05")
    steps = _run(cache, [1.00, 1.03, 1.06, 1.07])
    rest = [1, 2, 3]
    assert [ran for ran, _ in steps] == [rest, [], rest, []]
    # step s starts at s: computed steps add 3 blocks of 10 * (s + 1); step 1 adds step 0's 30, step 3 step 2's 90
    assert [out for _, out in steps] == pytest.approx([0 + 1.00 + 30, 1 + 1.03 + 30, 2 + 1.06 + 90, 3 + 1.07 + 90])
    assert cache.context.summary()["cached_steps"] == [[1, 3]]
    assert "reused 2 of 4 steps" in cache.summary()


def test_warmup_and_cap(monkeypatch):
    """Warmup steps always compute, and the cap bounds the reused steps in a row."""
    cache, _ = _cache(monkeypatch, "0.05,2,2")
    steps = _run(cache, [1.0] * 7)
    assert [bool(ran) for ran, _ in steps] == [True, True, False, False, True, False, False]


def test_padding_stays_out_of_the_change_test(monkeypatch):
    """`valid_rows` masks the pad row, whose change would otherwise end the reuse."""
    valid_rows = torch.tensor([1.0, 1.0, 0.0]).reshape(1, 1, 3, 1)
    for mask, reused in ((valid_rows, True), (None, False)):
        cache, _ = _cache(monkeypatch, "0.05", valid_rows=mask)
        cache.run_stack(torch.zeros(1, 1, 3, 2), lambda index, hidden: hidden + 1.0, NUM_BLOCKS)
        pad_moves = torch.tensor([1.0, 1.0, 5.0]).reshape(1, 1, 3, 1)
        cache.run_stack(torch.zeros(1, 1, 3, 2), lambda index, hidden: hidden + pad_moves, NUM_BLOCKS)
        assert cache.context.num_cached_steps() == int(reused)


def test_release(monkeypatch):
    """`release` frees the two tensors a computed step keeps, and a second call frees nothing."""
    cache, freed = _cache(monkeypatch, "0.05")
    _run(cache, [1.0])
    kept = [cache._first_change, cache._rest_change]
    freed.clear()
    cache.release()
    assert [id(t) for t in freed] == [id(t) for t in kept]
    cache.release()
    assert len(freed) == 2


@pytest.mark.parametrize("cached", [False, True])
def test_run_blocks_hands_the_stack_to_the_cache(monkeypatch, cached):
    """Without a cache `run_blocks` runs every block in order; with one it hands `run_stack` the input, a
    one-block runner with the same arguments, and the block count."""
    calls = []

    class Block:
        def __init__(self, index):
            self.index = index

        onehot_table = tilerow_tables = staticmethod(lambda *a: None)

        def __call__(self, hidden, logical_n, **kwargs):
            calls.append((self.index, logical_n, kwargs["rope_cos"]))
            return hidden + 1

    monkeypatch.setattr(tm.ttnn, "deallocate", lambda tensor: None)
    model = SimpleNamespace(transformer_blocks=[Block(i) for i in range(NUM_BLOCKS)])
    handed = []

    def run_stack(hidden, run_block, num_blocks):
        handed.append((hidden, num_blocks))
        return run_block(2, hidden)

    kwargs = {"step_cache": SimpleNamespace(run_stack=run_stack)} if cached else {}
    temb = SimpleNamespace(shape=[1, 1, 1, 8])
    out = tm.MiniMaxH3Transformer3DModel.run_blocks(model, 0, "n", temb, None, "cos", "sin", **kwargs)
    if cached:
        assert handed == [(0, NUM_BLOCKS)]
        assert (out, calls) == (1, [(2, "n", "cos")])
    else:
        assert (out, calls) == (NUM_BLOCKS, [(i, "n", "cos") for i in range(NUM_BLOCKS)])
