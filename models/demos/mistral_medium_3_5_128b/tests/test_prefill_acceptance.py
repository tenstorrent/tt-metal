# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The pipeline's acceptance test (``method/ACCEPTANCE.md``): full depth, full width, real weights.

A real forward on the 8x4 Blackhole Galaxy over the prepared CPU golden trace's exact token ids,
comparing **every** layer's K and V against the trace. The verifier runs it twice — once with
``PREFILL_CHUNKED=0``, once with ``=1`` — against the same trace, and reads the JSON report this
test writes to ``PREFILL_ACCEPTANCE_OUT``.

**It measures nothing of its own.** The run is ``tests/galaxy_prefill_kv_pcc.py``'s
:func:`~.galaxy_prefill_kv_pcc.run_prefill_kv_pcc`, which is the recipe's P1/P2 Testing-table row.
One implementation means the report cannot disagree with the stage it is supposed to certify, and
there is no second code path where a number could come from somewhere other than a device.

What this file adds on top of that row is the contract:

* **the dimensions are pinned before the run**, from three independent places — the checkpoint's
  own ``config.json``, the snapshotted spec, and the trace's metadata. A reduced run is the one
  failure mode a PCC number cannot reveal (a 2-layer model reports two beautiful rows), so it is
  checked structurally rather than trusted.
* **PCC is asserted before the report is written.** A report that exists is a report whose every
  layer cleared ``pcc_lower_bound``.
* **the report is built from the measurement object**, field by field, with no expected value
  anywhere in this file to copy from.

The diagnostic ``MISTRAL_PREFILL_LAYERS`` / ``MISTRAL_PREFILL_SEQ`` knobs that
``galaxy_prefill_kv_pcc.py`` honours are deliberately not read here: this test has one size.

Run::

    PREFILL_CHUNKED=0 PREFILL_ACCEPTANCE_OUT=/tmp/acceptance_one_shot.json \\
        scripts/run_safe_pytest.sh \\
        models/demos/mistral_medium_3_5_128b/tests/test_prefill_acceptance.py -q -s
"""

import json
import math
import os
import sys
from pathlib import Path

import pytest
from loguru import logger

from models.demos.mistral_medium_3_5_128b.reference.golden import GoldenTrace
from models.demos.mistral_medium_3_5_128b.tests.galaxy_prefill_kv_pcc import (
    FULL_RUN_TIMEOUT,
    chunked_from_env,
    report_lines,
    run_prefill_kv_pcc,
)

#: Where the verifier wants the report. Unset (a developer running this by hand) => no file.
OUT_ENV = "PREFILL_ACCEPTANCE_OUT"


@pytest.fixture(scope="module")
def trace():
    """The prepared full-depth CPU trace (``PREFILL_TRACE_DIR``). Both modes use the same one."""
    return GoldenTrace.from_env()


def _assert_full_dimensions(cfg, spec, trace, chunked):
    """Fail before spending an hour on a run that would not have been acceptable.

    Three sources have to agree: ``cfg`` is read from the checkpoint's ``config.json`` (the
    verifier re-checks the report against it), ``spec`` is the snapshotted binding spec, and
    ``trace`` is the golden trace's own metadata. Any pair agreeing while the third differs means
    the run would measure something other than what the report would claim.
    """
    assert cfg.num_hidden_layers == trace.num_layers, (
        f"the trace covers {trace.num_layers} layers but the model has {cfg.num_hidden_layers}; "
        f"acceptance needs a full-depth trace"
    )
    assert cfg.num_key_value_heads == trace.num_kv_heads and cfg.head_dim == trace.head_dim, (
        f"KV geometry differs: model {cfg.num_key_value_heads}x{cfg.head_dim}, "
        f"trace {trace.num_kv_heads}x{trace.head_dim}"
    )
    assert trace.n_tokens > spec.chunk_size, (
        f"the trace's {trace.n_tokens} tokens do not span more than one {spec.chunk_size}-token "
        f"chunk, so a chunked run would be a one-shot run"
    )
    assert trace.n_tokens <= spec.max_seq_len, f"trace {trace.n_tokens} tokens exceeds spec max {spec.max_seq_len}"
    if chunked:
        assert trace.n_tokens % spec.chunk_size == 0, (
            f"this package's chunk loop writes whole chunks, and {trace.n_tokens} tokens is not a "
            f"multiple of the spec's chunk size {spec.chunk_size}"
        )
    assert spec.sp * spec.tp == 32 and (spec.sp, spec.tp) == (8, 4), f"spec parallelism is {spec.sp}x{spec.tp}"


@pytest.mark.timeout(FULL_RUN_TIMEOUT)
def test_prefill_kv(galaxy, cfg, mesh_config, ccl, spec, trace):
    """Real weights, full depth and width, every layer's K/V against the golden trace."""
    chunked = chunked_from_env()
    _assert_full_dimensions(cfg, spec, trace, chunked)

    m = run_prefill_kv_pcc(
        galaxy,
        cfg,
        mesh_config,
        ccl,
        trace,
        chunked=chunked,
        seq_len=trace.n_tokens,
        chunk_size=spec.chunk_size,
        num_layers=cfg.num_hidden_layers,
    )
    for line in report_lines(m, lower=spec.pcc_lower_bound, target=spec.pcc_target):
        logger.info(line)

    # The measurement has to be the shape the report promises before any of it is believed.
    assert len(m.layer_pcc) == cfg.num_hidden_layers, f"{len(m.layer_pcc)} PCC rows for {cfg.num_hidden_layers} layers"
    assert [r["layer"] for r in m.layer_pcc] == list(range(cfg.num_hidden_layers)), "rows out of order"
    assert m.seq_len == trace.n_tokens and m.num_layers == cfg.num_hidden_layers

    nonfinite = [r for r in m.layer_pcc if not (math.isfinite(r["k"]) and math.isfinite(r["v"]))]
    assert not nonfinite, f"non-finite PCC on {len(nonfinite)} layers: {nonfinite[:3]}"
    below = [r for r in m.layer_pcc if min(r["k"], r["v"]) < spec.pcc_lower_bound]
    assert not below, (
        f"{len(below)} of {cfg.num_hidden_layers} layers below pcc_lower_bound " f"{spec.pcc_lower_bound}: {below[:5]}"
    )

    _write_report(m, cfg=cfg, spec=spec)

    worst_k, worst_v = m.worst()
    if min(worst_k, worst_v) < spec.pcc_target:
        logger.warning(
            f"[{m.mode}] worst k {worst_k:.6f} v {worst_v:.6f} is below pcc_target {spec.pcc_target} "
            f"(above the asserted lower bound {spec.pcc_lower_bound}) — see README.md"
        )


def _write_report(m, *, cfg, spec):
    """The ``ACCEPTANCE.md`` report, every field taken from the run that just happened.

    ``chunk_size`` is the spec's configured chunk size in **both** modes, as the contract states.
    One-shot's cache period is ``seq_len``, which is a property of how this package addresses the
    cache rather than a configuration value, and it is in the log line rather than here.
    """
    path = os.environ.get(OUT_ENV)
    report = {
        "mode": m.mode,
        "python": sys.executable,
        "num_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "seq_len": m.seq_len,
        "chunk_size": spec.chunk_size,
        "sp": spec.sp,
        "tp": spec.tp,
        "target_hw": spec.target_hw,
        "layer_pcc": m.layer_pcc,
    }
    if not path:
        logger.warning(f"{OUT_ENV} is unset; the report is only in this log")
        logger.info(json.dumps(report))
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    logger.info(f"[{m.mode}] wrote {out} ({len(m.layer_pcc)} layers)")
