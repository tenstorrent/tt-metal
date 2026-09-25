# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only diagnostic: how far is the bf16 golden trace from a more precise forward of the same model?

Runs the full-depth torch reference over the trace's prompt twice per layer, on the real checkpoint:
  * bf16 compute (the golden's own arithmetic)  -> should reproduce the trace's K/V (~1.0)
  * fp32 compute (same bf16-dequantized weights) -> a more precise forward, no device involved
and reports each one's per-layer K/V PCC against the trace. If the fp32 forward decays against the
trace the same way the device does, the gap is the model's depth-wise sensitivity to rounding — the
bf16 golden is one rounding path among many — not a device defect.

    python models/demos/mistral_medium_3_5_128b/scripts/golden_precision_ceiling.py \\
        --checkpoint $PREFILL_HF_MODEL --trace $PREFILL_TRACE_DIR --out /tmp/golden_ceiling.json
"""

import argparse
import json
import time
from pathlib import Path

import torch

from models.common.utility_functions import comp_pcc
from models.demos.mistral_medium_3_5_128b.config import MistralMediumConfig
from models.demos.mistral_medium_3_5_128b.reference.checkpoint import CheckpointReader
from models.demos.mistral_medium_3_5_128b.reference.model import ReferenceDecoderLayer, rope_cos_sin
from models.demos.mistral_medium_3_5_128b.tt.kv_validation import golden_layer_kv, trace_token_ids


def _layer(cfg, sd, dtype):
    with torch.device("meta"):
        layer = ReferenceDecoderLayer(cfg)
    layer.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, assign=True)
    return layer.eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None, help="stop after this many layers")
    args = ap.parse_args()

    cfg = MistralMediumConfig.from_json(Path(args.checkpoint) / "config.json")
    reader = CheckpointReader(args.checkpoint)
    tokens = torch.tensor(trace_token_ids(args.trace)[0])
    n = tokens.numel()
    pos = torch.arange(n)
    rope = {dt: rope_cos_sin(cfg, pos, dtype=dt) for dt in (torch.bfloat16, torch.float32)}
    emb = reader.embedding()[tokens][None]
    h = {torch.bfloat16: emb, torch.float32: emb.float()}
    rows = []
    for i, sd in reader.iter_layers(range(args.layers or cfg.num_hidden_layers)):
        t0 = time.perf_counter()
        gk, gv = golden_layer_kv(args.trace, i, n)
        row = {"layer": i}
        for dt, tag in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
            h[dt], k, v = _layer(cfg, sd, dt)(h[dt], *rope[dt], pos)
            row[f"{tag}_k"] = float(comp_pcc(gk.float(), k[0].float(), 0.0)[1])
            row[f"{tag}_v"] = float(comp_pcc(gv.float(), v[0].float(), 0.0)[1])
        row["bf16_vs_fp32_hidden"] = float(comp_pcc(h[torch.float32], h[torch.bfloat16].float(), 0.0)[1])
        rows.append(row)
        print(json.dumps(row), f"({time.perf_counter() - t0:.0f}s)", flush=True)
        Path(args.out).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
