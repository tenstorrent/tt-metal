# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage A — decode SDPA config ACCURACY sweep (single-chip, fast).

Opens a 1x1 mesh ONCE, builds the two full-attention layers (0 FULL_DENSE, 4 FULL_MOE — the SDPA-config-
sensitive kinds; the sliding layer is window-bounded), prefills once per (layer, pos), then swaps
``dec._sdpa_pc_decode`` across config variants IN-PROCESS and measures decode PCC vs the layer-only HF
reference. Includes a k_chunk=128 NEGATIVE CONTROL that should reproduce the recorded lossiness
(layer-4 pos-2048 PCC ~ -0.016) — if it does while k64 passes, the shipped k64 default is validated on
current code.  Speed is intentionally NOT measured here (single-layer short-ctx timing is not the signal);
the long-context decode-speed win is measured served in Stage D (=1 vs =0 at ISL 32k/128k).

Run:
  cd /tmp && TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal \
    PYTHONPATH=/home/ttuser/dev/tt-metal \
    /home/ttuser/.tenstorrent-venv/bin/python \
    /home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/doc/vllm_integration/scripts/decode_config_sweep.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

import ttnn

MODEL_DIR = Path("/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1")
sys.path.insert(0, "/home/ttuser/dev/tt-metal")
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R  # noqa: E402
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W  # noqa: E402
from models.autoports.poolside_laguna_xs_2_1.tt.optimized_decoder import OptimizedDecoder  # noqa: E402

HIDDEN = 2048
PCC_BAR = 0.995
LAYERS = [0, 4]  # FULL_DENSE, FULL_MOE (full-attention; SDPA-config-sensitive)
POSITIONS = [513, 2048]  # 513 = non-128-aligned; 2048 = the recorded k128-failure position

# (k_chunk, exp_approx_mode, label). max_cores left unset (ttnn default 16) for all — matches shipped.
CONFIGS = [
    (128, False, "k128 (NEG CONTROL, expect FAIL)"),
    (64, False, "k64  (SHIPPED default)"),
    (32, False, "k32"),
    (64, True, "k64 + exp_approx"),
    (32, True, "k32 + exp_approx"),
]

OUT = MODEL_DIR / "doc/vllm_integration/decode_config_sweep"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "sweep.log"
RESULTS = OUT / "results.md"


def _pcc(a, b):
    a = a.flatten().float().numpy()
    b = b.flatten().float().numpy()
    return float(np.corrcoef(a, b)[0, 1])


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def _tt(x, device):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def main():
    LOG.write_text("")
    log("=== Stage A: decode SDPA config accuracy sweep (single-chip 1x1) ===")
    log(f"configs={[c[2] for c in CONFIGS]}  layers={LAYERS}  positions={POSITIONS}  PCC_BAR={PCC_BAR}")
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        hf_config = R.build_config()
        grid = dev.compute_with_storage_grid_size()

        def make_pc(k, exp):
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
                q_chunk_size=32,
                k_chunk_size=k,
                exp_approx_mode=exp,
            )

        # rows[(layer,pos)][label] = pcc
        rows = {}
        for layer in LAYERS:
            raw = W.load_layer_tensors(layer)
            ctx = R.make_context(
                hf_config, layer, state_dict=W.to_hf_layer_state_dict(raw, hf_config, layer), dtype=torch.float32
            )
            dec = OptimizedDecoder.from_state_dict(
                raw,
                hf_config=hf_config,
                layer_idx=layer,
                mesh_device=dev,
                max_seq_len=hf_config.max_position_embeddings,
            )
            assert dec._decode_use_sdpa_pc, "decode must use the _sdpa_pc_decode path for this sweep"
            for pos in POSITIONS:
                log(f"--- layer {layer} pos {pos}: prefill + reference ---")
                kv = dec.alloc_kv_cache(max_users=1, max_seq_len=pos + 8, block_size=32)
                pt = dec.make_page_table(1, kv["blocks_per_user"])
                torch.manual_seed(pos)
                x = torch.randn(1, pos, HIDDEN) * 0.5
                _, pkv = R.reference_forward(ctx, x)
                dec.prefill_forward(_tt(x, dev), kv, pt, user_id=0, start_pos=0)
                xd = torch.randn(1, 1, HIDDEN) * 0.5
                ref, _ = R.reference_forward(ctx, xd, past_key_values=pkv)
                cur = ttnn.from_torch(
                    torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
                )
                ridx = ttnn.from_torch(
                    torch.tensor([[pos]], dtype=torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=dev,
                )
                rows[(layer, pos)] = {}
                for k, exp, label in CONFIGS:
                    dec._sdpa_pc_decode = make_pc(k, exp)
                    out = dec.decode_forward(_tt(xd.reshape(1, 1, 1, HIDDEN), dev), cur, ridx, pt, kv)
                    got = ttnn.to_torch(out).float().reshape(1, 1, HIDDEN)
                    pcc = _pcc(got, ref)
                    rows[(layer, pos)][label] = pcc
                    verdict = "PASS" if pcc >= PCC_BAR else "FAIL"
                    log(f"  layer {layer} pos {pos:5d}  {label:34s}  PCC {pcc:+.5f}  {verdict}")

        # results.md
        lines = [
            "# Stage A — decode SDPA config accuracy sweep",
            "",
            f"Single-chip (1x1), full-attention layers {LAYERS}, PCC bar {PCC_BAR}. Decode PCC vs layer-only HF.",
            "",
            "| layer | pos | " + " | ".join(c[2] for c in CONFIGS) + " |",
            "|---|---|" + "|".join(["---"] * len(CONFIGS)) + "|",
        ]
        for (layer, pos), d in rows.items():
            cells = [f"{d[c[2]]:+.5f}" for c in CONFIGS]
            lines.append(f"| {layer} | {pos} | " + " | ".join(cells) + " |")

        # winner = fastest-known PCC-safe: prefer k64 (exp off) if it passes everywhere.
        def passes(label):
            return all(rows[key][label] >= PCC_BAR for key in rows)

        safe = [c[2] for c in CONFIGS if passes(c[2])]
        lines += [
            "",
            f"**PCC-safe configs (pass at all layer/pos):** {safe or 'NONE'}",
            "",
            "**Winner:** k64 (exp off) if PCC-safe — it is the shipped default and keeps the "
            "max_cores=16 long-context speed. k32 is PCC-safe but does more inner-loop iterations "
            "(slower at long ctx); exp_approx trades precision for marginal speed. k128 is the "
            "negative control (should FAIL, reproducing the recorded lossiness).",
            "",
            "Long-context decode SPEED for the winner vs the accurate-but-slow ttnn default "
            "(TT_LAGUNA_DECODE_SDPA_PC=0) is measured served in Stage D.",
        ]
        RESULTS.write_text("\n".join(lines) + "\n")
        log(f"=== DONE. PCC-safe: {safe}. results -> {RESULTS} ===")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
