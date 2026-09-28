# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Correctness gates for the UNFUSED topk_xl SFPLOADMACRO port (Blackhole).

``ckernel_sfpu_topk_xl.h`` now macro-schedules the unfused merge body and the
unfused rebuild's single-level stride-64/32/16 bodies BY DEFAULT (opt-out:
``DISABLE_TOPK_XL_SFPLOADMACRO``). The full ``test_topk_xl.py`` matrix already
exercises the macro path implicitly (it compiles the shipping header); this
file adds the three checks that matrix cannot express:

  1. CHAINED golden at num_chunks = 2 and 4 on the exact op path (row-major,
     unfused). The branch's documented trap: a single merge+rebuild pair
     cannot see a mis-ordered rebuild (it only PERMUTES its K survivors);
     >= 3 chained pairs (num_chunks=4) are required for the error to reach
     the next merge and corrupt the returned SET.
  2. DIFFERENTIAL: the default (macro) build's packed output must be
     byte-identical to the ``DISABLE_TOPK_XL_SFPLOADMACRO`` (shipping-body)
     build on identical stimuli. This is the same macro==software evidence
     shape the silicon probe (test_topk_unfused_macro_probe.py) validated.
  3. MUTATION CONTROL: ``TOPK_XL_MACRO_MUTATE`` zeroes the macro Sequence
     words, degenerating every SFPLOADMACRO into a plain SFPLOAD — the
     documented "schedule nothing" failure mode that issue-rate timing
     CANNOT see (the scheduled SFPSWAP and SFPSTORE ride otherwise-idle
     sub-units either way). It silently drops the compare-exchanges AND the
     macroVD stores, so the mutated build must diverge from the good build;
     if it did not, the green runs above would prove nothing.
     (Do NOT mutate by clearing the Simple byte's 0x80 bit: SFPSWAP(VC==VD)
     is a same-register 2-cycle read-modify-write that silicon resolves as
     garbage, not a modelable no-op — measured 2026-08-16.)

Reuses ``sources/topk_xl_test.cpp`` (which compiles the shipping header) and
the stimulus/golden machinery of ``test_topk_xl.py``.
"""

from dataclasses import dataclass

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    TopKSortDirection,
    TopKXLChunkBaseMode,
    TopKXLIndexOp,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, TOPK_XL, TemplateParameter
from test_topk_xl import (
    ELEMENTS_PER_TILE,
    FORMATS,
    GROUP_SHIFT,
    _build_input,
    _check,
    _tiles_per_sequence,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]


@dataclass
class TOPK_XL_MACRO_KNOBS(TemplateParameter):
    """Build-header knobs for the unfused SFPLOADMACRO path.

    ``opt_out`` emits ``#define DISABLE_TOPK_XL_SFPLOADMACRO 1`` — the header
    rebuilds the byte-identical pre-macro bodies. ``mutate`` emits
    ``#define TOPK_XL_MACRO_MUTATE 1`` — the header zeroes the macro Sequence
    words (schedule nothing). Defined here rather than in
    helpers/test_variant_parameters.py because exactly one header consumes
    them (same rationale as perf_topk_merge_macro.py's parameter classes).
    """

    opt_out: bool = False
    mutate: bool = False

    def convert_to_cpp(self) -> str:
        lines = ["// topk_xl unfused SFPLOADMACRO knobs"]
        if self.opt_out:
            lines.append("#define DISABLE_TOPK_XL_SFPLOADMACRO 1")
        if self.mutate:
            lines.append("#define TOPK_XL_MACRO_MUTATE 1")
        return "\n".join(lines)


def _variant(
    K, num_chunks, opt_out=False, mutate=False, num_rows=2, reinit_after_copy=False
):
    """The op path: row-major index split, UNFUSED merge/rebuild, descending.
    Mirrors test_topk_xl._variant with the macro knobs added."""
    tail_elements = K
    tiles_per_seq = _tiles_per_sequence(K)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=format_dict[FORMATS.input_format])
    src_A, rows = _build_input(
        K, num_chunks, tail_elements, num_rows, "positive", as_float32=False
    )
    config = TestConfig(
        test_name="sources/topk_xl_test.cpp",
        formats=FORMATS,
        templates=[
            DEST_SYNC(DestSync.Full),
            TOPK_XL(
                k=K,
                num_chunks=num_chunks,
                tail_elements=tail_elements,
                num_rows=num_rows,
                index_op=TopKXLIndexOp.RowMajor,
                group_id=0,
                group_shift=GROUP_SHIFT,
                core_id=0,
                sort_direction=TopKSortDirection.Descending,
                fused_reduce=False,
                chunk_base_mode=TopKXLChunkBaseMode.Static,
                chunk_base=0,
                reinit_after_copy=reinit_after_copy,
            ),
            TOPK_XL_MACRO_KNOBS(opt_out=opt_out, mutate=mutate),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            FORMATS.input_format,
            src_B,
            FORMATS.input_format,
            FORMATS.output_format,
            tile_count_A=num_rows * num_chunks * tiles_per_seq,
            tile_count_B=1,
            tile_count_res=num_rows * 2 * tiles_per_seq,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=False,
    )
    return config, rows


@parametrize(K=[512, 1024, 2048], num_chunks=[2, 4])
def test_topk_xl_unfused_macro_chained(K, num_chunks):
    """Golden gate on the default (macro-on) build. num_chunks=4 chains three
    merge+rebuild pairs — the only configuration that catches a mis-ordered
    rebuild (interleave/drain bugs present as EXACTLY that failure)."""
    config, rows = _variant(K, num_chunks=num_chunks)
    result = config.run().result
    _check(result, K, rows, compare_index_set=True)


@parametrize(K=[512, 1024, 2048])
def test_topk_xl_unfused_macro_equals_opt_out(K):
    """DIFFERENTIAL: default (macro) build == DISABLE_TOPK_XL_SFPLOADMACRO
    (shipping-body) build, byte-for-byte on the packed result, at
    num_chunks=4. Both must also pass the golden — equality of two wrong
    outputs would otherwise slip through."""
    (K,) = K
    good_cfg, rows = _variant(K, num_chunks=4)
    base_cfg, _ = _variant(K, num_chunks=4, opt_out=True)
    # Build both before running either: under --compile-producer, run() skips
    # as soon as the first variant is built (same reason as
    # test_topk_rebuild_full_macro's mutation test).
    good_cfg.prepare()
    base_cfg.prepare()
    good, base = good_cfg.run().result, base_cfg.run().result

    _check(good, K, rows, compare_index_set=True)
    assert torch.equal(good, base), (
        f"K={K}: macro build's packed output differs from the opt-out "
        f"(shipping-body) build — the unfused SFPLOADMACRO bodies are not "
        f"bit-identical to the software bodies"
    )


@parametrize(K=[512, 1024, 2048])
def test_topk_xl_unfused_macro_mutation(K):
    """MUTATION CONTROL. TOPK_XL_MACRO_MUTATE zeroes the Sequence words:
    every SFPLOADMACRO in the unfused merge/rebuild degenerates into a plain
    SFPLOAD, dropping the compare-exchanges and the macroVD stores. The
    mutated build must diverge from the good build at num_chunks=4; a green
    mutated arm would mean every other test in this file is blind to whether
    the macros do any work."""
    (K,) = K
    good_cfg, rows = _variant(K, num_chunks=4)
    bad_cfg, _ = _variant(K, num_chunks=4, mutate=True)
    good_cfg.prepare()
    bad_cfg.prepare()
    good, bad = good_cfg.run().result, bad_cfg.run().result

    # Sanity: the unmutated build of this exact variant agrees with the
    # golden, otherwise a red mutation arm is evidence of nothing.
    _check(good, K, rows, compare_index_set=True)

    assert not torch.equal(good, bad), (
        f"K={K}: MUTATION NOT DETECTED — schedule-nothing macros produced "
        f"the same packed output as the real ones. Every PASS in this file "
        f"is void; the macro path is either not being compiled in or not "
        f"being exercised."
    )


@parametrize(K=[512, 1024, 2048], opt_out=[False, True])
def test_topk_xl_unfused_macro_reinit_after_copy(K, opt_out):
    """The post-copy unfused reinit reaches the MOP Expander through
    ``topk_mop_config<false>``, whose recording window is the macro body (16
    instructions) on the default build and the shipping body (18) under
    ``DISABLE_TOPK_XL_SFPLOADMACRO``. A reinit that programs the other build's
    length replays a truncated or overlong merge body, so run both arms at
    num_chunks = 4 for the same reason the chained gate does."""
    config, rows = _variant(K, num_chunks=4, opt_out=opt_out, reinit_after_copy=True)
    result = config.run().result
    _check(result, K, rows, compare_index_set=True)
