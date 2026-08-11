#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate ``doc/functional_decoder/weight_stats.json`` from the real checkpoint.

The committed stats file records name/shape/dtype/mean/std/absmax for every tensor of one
representative layer of each decoder-layer kind. Tests build synthetic weights from it, so
the fast suite runs with the real geometry and scales but without the 60 GB download.

    python models/autoports/meta_models_muse_glimmer_30b/scripts/dump_weight_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path

from models.autoports.meta_models_muse_glimmer_30b.reference import hf_reference as R

OUTPUT = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder" / "weight_stats.json"


def main() -> int:
    text_config = R.load_text_config()
    payload = {
        "hf_model_id": R.HF_MODEL_ID,
        "note": ("per-tensor stats of the real checkpoint; used to build synthetic weights " "with real shapes/scales"),
        "layers": {},
    }
    for kind_id, kind in R.layer_kinds(text_config).items():
        state_dict = R.load_real_layer_state_dict(kind.layer_idx)
        payload["layers"][str(kind.layer_idx)] = {
            "kind_id": kind_id,
            "layer_type": kind.layer_type,
            "rope_theta": kind.rope_theta,
            "sliding_window": kind.sliding_window,
            "tensors": R.weight_stats(state_dict),
        }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
