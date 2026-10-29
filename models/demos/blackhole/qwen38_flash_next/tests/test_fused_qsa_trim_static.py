# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_QSA_TRIM without a device: off by default, the option plumbing, the qsa.py wiring, and the reduce compute's
text against row_reduce's (the same tree, the widen and the narrow folded in)."""

from __future__ import annotations

import inspect
import re

from models.demos.blackhole.qwen38_flash_next.ttnn import qsa
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_trim

RR_BASE = (
    fp.REPO_ROOT / "models/demos/blackhole/qwen38_flash_next/ttnn/fused/row_reduce/kernels/compute.cpp"
).read_text()
RR_NEW = (fp.REPO_ROOT / qsa_trim.RR_COMPUTE).read_text()


def test_off_by_default_and_options():
    assert qsa_trim.options({}) == qsa_trim.DEFAULT_OPTS and not qsa_trim.enabled({qsa_trim.ENV: "0"})  # on by default
    assert qsa_trim.options({qsa_trim.ENV: "1"}) == qsa_trim.DEFAULT_OPTS == ("pages", "post", "reduce")
    assert qsa_trim.options({qsa_trim.ENV: "1", qsa_trim.OPTS_ENV: "pages"}) == ("pages",)
    assert (
        qsa_trim.options({qsa_trim.OPTS_ENV: "pages"}) == ("pages",)
        and qsa_trim.options({qsa_trim.ENV: "0", qsa_trim.OPTS_ENV: "pages"}) == ()
    )


def test_qsa_wiring_is_behind_the_switch_with_the_chain_kept():
    score = inspect.getsource(qsa.Qwen38TTNNQSA._score_blocks_fused)
    assert "self._trim_score_pages and self._trim_score_pages.admits(" in score and "ttnn.slice(" in score
    post = inspect.getsource(qsa.Qwen38TTNNQSA._sparse_value_attention_fused)
    assert "self._trim_post_attention.admits(" in post and "self._post_attention_fused(" in post
    out = inspect.getsource(qsa.Qwen38TTNNQSA._project_output_fused)
    assert "self._trim_reduce is not None and rows == 1 and self._trim_reduce.admits(local_partial_ws)" in out
    assert "_widen_partial_fused(local_partial_ws)" in out and "ttnn.typecast(output_fp32" in out
    assert qsa.Qwen38TTNNQSA._trim_score_pages is None and qsa.Qwen38TTNNQSA._trim_reduce is None


def _code(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S))


def test_reduce_compute_keeps_the_summation_tree():
    # the same add order: the loops of row_reduce's compute, with `last` marking the two writes into CB_OUT
    for piece in (
        "for (uint32_t k = 1; k < L; ++k) {",
        "acc = CB_F;",
        "for (uint32_t k = RING - 2; k > L; --k) {",
        "acc = CB_B;",
        "own_sum = CB_G;",
    ):
        assert _code(piece) in _code(RR_BASE) and _code(piece) in _code(RR_NEW), piece
    assert "add_pop<true>(CB_IN, acc, CB_OUT)" in RR_NEW and "add_pop<true>(own_sum, acc, CB_OUT)" in RR_NEW
    assert RR_NEW.count("add_pop<true>") == 2 and RR_NEW.count("add_pop<false>") == 3
    assert (
        "typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0)"
        in RR_NEW
    )
    assert "add_tiles(a, b, 0, 0, 0)" in RR_NEW and "reconfig_data_format(a, b)" in RR_NEW


def test_reduce_uses_row_reduce_v2_transport_unchanged():
    source = inspect.getsource(qsa_trim.reduce_w2n)
    assert "kernel=rr.TRANSPORT2_KERNEL" in source and "rr.prepare2(mesh, tiles)" in source
    assert "fp32_dest=True" in source and "stamp_topology(out, reference, 3)" in source
