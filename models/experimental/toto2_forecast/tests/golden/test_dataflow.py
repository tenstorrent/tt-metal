# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks (python_env, no device): the folded fp32 dataflow against the reference goldens.

For each golden case this checks, against the reference captures:
- the host half (``tt/preprocessing.py``): per-patch loc / scale after the backfill, the patch tokens, and the
  masks: with one group per row the reference's causal + group-id mask is exactly ``tril``, so the device may run
  ``is_causal``;
- ``tests/golden/folded.py`` on the full sequence: the embedding and every captured layer;
- the device's decomposition (rows padded to whole tiles, the cut to the read-out tokens after the last time
  layer, the V = 1 variate shortcut): the head output, relative error <= 1e-5 (CONTRACT section 3), and the final
  quantiles after ``tt/readout.py``.

    TOTO2_DATAFLOW_JSON=<path>  appends one record per case
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from models.experimental.toto2_forecast.tests.golden.folded import FoldedToto2
from models.experimental.toto2_forecast.tests.golden.goldens import have_golden, load_golden, rel_err
from models.experimental.toto2_forecast.tt.preprocessing import pad_to_tile, prepare_inputs, time_mask_from_groups
from models.experimental.toto2_forecast.tt.readout import postprocess
from models.experimental.toto2_forecast.tt.weights import Toto2Weights

SMALL = ("uni_4x2048", "missing_5x2048", "mv_2x4x2048", "long_2x8192")
CASES = [
    *(("4m", c) for c in SMALL),
    *(("22m", c) for c in SMALL),
    *(("313m", c) for c in ("uni_16x2048", "missing_16x2048", "mv_4x8x2048")),
    *(("2.5B-FT", c) for c in ("uni_16x2048", "missing_16x2048", "mv_4x8x2048")),
]
HEAD_TOL = 1e-5
LAYER_TOL = 1e-5


@lru_cache(maxsize=1)
def _weights(ckpt: str) -> Toto2Weights:
    return Toto2Weights.from_checkpoint(ckpt)


def _record(entry: dict):
    path = os.environ.get("TOTO2_DATAFLOW_JSON")
    if path:
        out = Path(path)
        data = json.loads(out.read_text()) if out.is_file() else []
        out.write_text(json.dumps([*data, entry], indent=2) + "\n")


@pytest.mark.parametrize("ckpt,case", CASES, ids=[f"{c}-{n}" for c, n in CASES])
def test_dataflow(ckpt, case):
    if not have_golden(ckpt, case):
        pytest.skip(f"golden {ckpt}/{case} not generated yet")
    torch.set_num_threads(8)
    g = load_golden(ckpt, case)
    inp, cap = g["inputs"], g["captures"]
    host = prepare_inputs(inp["target"], inp["target_mask"], inp["series_ids"], inp["horizon"])
    p, s = host.num_context_patches, host.seq_len
    errs = {}

    # Host half.
    errs["loc"] = rel_err(host.loc, cap["loc_bf"][..., p:])
    errs["scale"] = rel_err(host.scale, cap["scale_bf"][..., p:])
    assert errs["loc"] <= 1e-6 and errs["scale"] <= 1e-6, errs
    if "tokens_context" in cap:
        tokens = torch.cat([cap["tokens_context"], cap["tokens_horizon"]], dim=2).reshape(host.rows, s, -1)
        errs["tokens"] = rel_err(host.tokens, tokens)
        assert errs["tokens"] <= 1e-6, errs
    reference_mask = time_mask_from_groups(host.group_ids.reshape(host.rows, s))
    clean = g["case"]["pattern"] == "clean"
    if clean:
        assert host.time_mask is None
        assert torch.equal(reference_mask, torch.tril(torch.ones(s, s, dtype=torch.bool)).expand_as(reference_mask))
    else:
        assert host.time_mask is not None and not torch.equal(
            reference_mask, torch.tril(torch.ones_like(reference_mask))
        )

    model = FoldedToto2(_weights(ckpt))
    # Full sequence, reference layouts.
    if "embed_context" in cap:
        captured = {}
        head = model.forward(host, cut=False, capture=captured)
        embed = torch.cat([cap["embed_context"], cap["embed_horizon"]], dim=2).reshape(host.rows, s, -1)
        errs["embed"] = rel_err(captured["embed"], embed)
        for key in sorted(k for k in cap if k.startswith("layer_")):
            errs[key] = rel_err(captured[key], cap[key])
        errs["head_full"] = rel_err(head, cap["head_out"])
        bad = {k: v for k, v in errs.items() if k.startswith(("layer_", "embed")) and v > LAYER_TOL}
        assert not bad, bad

    # The device decomposition: padded rows, the read-out cut, the variate shortcut.
    head = model.forward(host, cut=True, padded_seq=pad_to_tile(s))
    errs["head"] = rel_err(head, cap["head_out"])
    q = postprocess(head, host.loc, host.scale, host.caps, host.horizon)
    errs["q"] = rel_err(q, cap["q"])
    _record({"ckpt": ckpt, "case": case, **errs})
    print(f"\n[DATAFLOW] {ckpt}/{case}: " + " ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    assert errs["head"] <= HEAD_TOL and errs["q"] <= HEAD_TOL, errs


@pytest.mark.parametrize("ckpt,case", CASES, ids=[f"{c}-{n}" for c, n in CASES])
def test_dataflow_folded_betas(ckpt, case):
    """R2-1: the beta-folded weights (plain residual adds, eps / c^2) give the reference head and quantiles."""
    if not have_golden(ckpt, case):
        pytest.skip(f"golden {ckpt}/{case} not generated yet")
    torch.set_num_threads(8)
    g = load_golden(ckpt, case)
    inp, cap = g["inputs"], g["captures"]
    host = prepare_inputs(inp["target"], inp["target_mask"], inp["series_ids"], inp["horizon"])
    w = _weights(ckpt)
    folded = w.fold_residual_betas()
    assert all(lw.beta_attn == 1.0 and lw.beta_mlp == 1.0 for lw in folded.layers)
    head = FoldedToto2(w, fold_betas=True).forward(host, cut=True, padded_seq=pad_to_tile(host.seq_len))
    errs = {"head": rel_err(head, cap["head_out"])}
    q = postprocess(head, host.loc, host.scale, host.caps, host.horizon)
    errs["q"] = rel_err(q, cap["q"])
    errs["c_final"] = math.sqrt(w.config.norm_eps / folded.head_eps)
    print(f"\n[DATAFLOW-FOLD] {ckpt}/{case}: " + " ".join(f"{k}={v:.2e}" for k, v in errs.items()))
    assert errs["head"] <= HEAD_TOL and errs["q"] <= HEAD_TOL, errs
