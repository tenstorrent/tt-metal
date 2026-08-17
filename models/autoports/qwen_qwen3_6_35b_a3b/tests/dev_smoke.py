# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Development smoke driver: build the layer, run prefill/decode, print PCC vs HF.

Kept out of the pytest suite so it can be iterated quickly on one device open:

    python models/autoports/qwen_qwen3_6_35b_a3b/tests/dev_smoke.py --kind full --seq 256
"""

import argparse
import traceback

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.harness import (
    build_layer_pair,
    decode_and_compare,
    prefill_and_compare,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["full", "linear"], default="full")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--real-weights", action="store_true")
    ap.add_argument("--decode-steps", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(16)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        pair = build_layer_pair(
            device,
            kind=args.kind,
            max_batch_size=args.batch,
            supported_context=args.context,
            real_weights=args.real_weights,
        )
        res = prefill_and_compare(pair, seq_len=args.seq, user_id=0)
        print(f"SMOKE prefill kind={args.kind} seq={args.seq} PCC={res.pcc:.6f} maxabs={res.maxabs:.3e}")
        if args.decode_steps:
            dres = decode_and_compare(pair, prefill_len=args.seq, steps=args.decode_steps)
            for i, r in enumerate(dres):
                print(f"SMOKE decode step={i} PCC={r.pcc:.6f} maxabs={r.maxabs:.3e}")
    except Exception:
        traceback.print_exc()
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
