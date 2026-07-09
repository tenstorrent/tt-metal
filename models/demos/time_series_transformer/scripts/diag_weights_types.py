# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Diagnostic: print the type of every leaf in the weights dict returned by
load_weights(), flagging any leaf that lacks .shape (the thing that broke
measure_weight_footprint.py). No assumptions -- just walks the real dict.
"""
import sys
from pathlib import Path

from transformers import TimeSeriesTransformerForPrediction

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tt.tst_model import load_weights  # noqa: E402

MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
MODEL_REVISION = "2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35"


def walk(name, obj, offenders):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(f"{name}.{k}", v, offenders)
    else:
        has_shape = hasattr(obj, "shape")
        flag = "" if has_shape else "  <-- NOT A TENSOR (no .shape)"
        print(f"{name:<55}{type(obj).__name__:<20}{flag}")
        if not has_shape:
            offenders.append((name, obj))


def main():
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    hf_model.eval()
    device = ttnn.open_device(device_id=0)
    try:
        weights = load_weights(hf_model, device)
        offenders = []
        walk("weights", weights, offenders)
        print("\n" + "=" * 70)
        if offenders:
            print(f"Found {len(offenders)} non-tensor leaf/leaves:")
            for name, obj in offenders:
                preview = repr(obj)
                if len(preview) > 80:
                    preview = preview[:80] + "..."
                print(f"  {name} = {preview}")
        else:
            print("No non-tensor leaves found (unexpected, given the earlier traceback).")
        print("=" * 70)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
