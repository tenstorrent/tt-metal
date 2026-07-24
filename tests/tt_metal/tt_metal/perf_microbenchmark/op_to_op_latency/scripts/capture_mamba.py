#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Capture ONE Mamba decode step (state-space model) under ttnn graph capture, dump JSON for
raw_hazard_analyzer.py. Single Wormhole chip, 5x8=40-core grid (fits fast dispatch).

Mamba is an SSM -- no attention -- so this probes whether the RAW/reorderable pattern differs for a
selective-scan recurrence vs the attention/conv models. RANDOM weights via Mamba.from_random (only
the ~2KB HF config json is fetched; hazard structure is weight-value independent). Uses the smallest
variant. Mirrors models/demos/wormhole/mamba/tests/test_mamba_model.py construction.
"""
import json
import sys

import torch
import ttnn

from models.demos.wormhole.mamba.reference.args import ModelMode
from models.demos.wormhole.mamba.reference.prefill_decode_model import Mamba
from models.demos.wormhole.mamba.tt import model_config
from models.demos.wormhole.mamba.tt.mamba_model import MambaTT

MODEL = "state-spaces/mamba-2.8b"  # config sharding is hardcoded for d_model=2560 (this variant)
BATCH = 32  # DECODE mode requires batch 32
OUT = "/tmp/mamba_capture.json"


def main():
    ref = Mamba.from_random(MODEL, batch_size=BATCH)  # random weights, config-only download
    print(f"Mamba.from_random {MODEL}: d_model={ref.args.d_model} n_layer={ref.args.n_layer}")

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        cfg = model_config.create_model_config(BATCH, ref.args.d_model, mode=ModelMode.DECODE, seq_len=1)
        tt_model = MambaTT(ref, device, cfg, tt_cache_path=None)

        input_ids = torch.randint(0, ref.args.vocab_size, (BATCH, 1), dtype=torch.long)

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out = tt_model(input_ids)
        ttnn.synchronize_device(device)
        captured = ttnn.graph.end_graph_capture()

        json.dump(captured, open(OUT, "w"))
        print(f"captured {len(captured)} nodes -> {OUT}  (out shape={tuple(out.shape)})")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
