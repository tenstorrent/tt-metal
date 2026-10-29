# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Correctness gates for the FUSED topk_xl SFPLOADMACRO step groups (Blackhole).

The fused local sort, merge and rebuild use SFPLOADMACRO by default (opt-out:
``DISABLE_TOPK_XL_SFPLOADMACRO``). A schedule that is wrong only in timing
passes a golden check on some runs, so this file adds what ``test_topk_xl.py``
cannot express, in the shape of ``test_topk_xl_unfused_macro.py``:

  1. DIFFERENTIAL: the default build's packed output is byte-identical to the
     opt-out build's on identical stimuli, for the fused end-to-end row and for
     the column body that runs ``_topk_xl_rebuild_columns_`` straight after a
     fused merge.
  2. MUTATION CONTROL: ``TOPK_XL_FUSED_MACRO_MUTATE`` zeroes the fused Sequence
     words, so every SFPLOADMACRO becomes a plain SFPLOAD. The mutated build
     must diverge, or the differential proves nothing.
"""

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.llk_params import TopKXLSortMode
from helpers.param_config import parametrize
from test_topk_xl import _check, _variant
from test_topk_xl_unfused_macro import TOPK_XL_MACRO_KNOBS

pytestmark = [skip_for_wormhole, skip_for_quasar]


def _e2e(K, **knobs):
    return _variant(
        K,
        num_chunks=4,
        num_rows=2,
        fused_e2e=True,
        extra_templates=[TOPK_XL_MACRO_KNOBS(**knobs)],
    )


def _columns(**knobs):
    return _variant(
        1024,
        num_chunks=4,
        sort_mode=TopKXLSortMode.Columns,
        extra_templates=[TOPK_XL_MACRO_KNOBS(**knobs)],
    )


@parametrize(K=[1024, 2048])
def test_topk_xl_fused_macro_equals_opt_out(K):
    (K,) = K
    good_cfg, rows = _e2e(K)
    base_cfg, _ = _e2e(K, opt_out=True)
    # Build both before running either: under --compile-producer, run() skips
    # as soon as the first variant is built.
    good_cfg.prepare()
    base_cfg.prepare()
    good, base = good_cfg.run().result, base_cfg.run().result

    _check(good, K, rows, compare_index_set=True)
    assert torch.equal(
        good, base
    ), f"K={K}: the fused macro build's packed output differs from the opt-out build"


def test_topk_xl_fused_macro_columns_equals_opt_out():
    good_cfg, _ = _columns()
    base_cfg, _ = _columns(opt_out=True)
    good_cfg.prepare()
    base_cfg.prepare()
    good, base = good_cfg.run().result, base_cfg.run().result
    assert torch.equal(
        good, base
    ), "the fused macro build's column body differs from the opt-out build"


@parametrize(K=[1024, 2048])
def test_topk_xl_fused_macro_mutation(K):
    (K,) = K
    good_cfg, rows = _e2e(K)
    bad_cfg, _ = _e2e(K, mutate_fused=True)
    good_cfg.prepare()
    bad_cfg.prepare()
    good, bad = good_cfg.run().result, bad_cfg.run().result

    _check(good, K, rows, compare_index_set=True)
    assert not torch.equal(good, bad), (
        f"K={K}: MUTATION NOT DETECTED, schedule-nothing fused macros packed the same "
        f"output as the real ones, so the differential is blind to them"
    )


def test_topk_xl_fused_macro_columns_mutation():
    good_cfg, _ = _columns()
    bad_cfg, _ = _columns(mutate_fused=True)
    good_cfg.prepare()
    bad_cfg.prepare()
    good, bad = good_cfg.run().result, bad_cfg.run().result
    assert not torch.equal(
        good, bad
    ), "MUTATION NOT DETECTED: the column body does not run the fused macros"
