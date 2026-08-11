# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-capacity probe for the Muse-Glimmer-30B functional decoder.

Runs one paged prefill (and optionally one decode at the far end of the
context) at a requested sequence length on a 1x1 mesh and reports whether it
fits.  Used to produce the byte/failure evidence recorded in
``doc/context_contract.json``.

Usage::

    python models/autoports/meta_models_muse_glimmer_30b/tests/functional_decoder_capacity_probe.py \
        --seq-len 131072 --layer 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

import ttnn

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script entry point
    sys.path.insert(0, str(REPO_ROOT))

from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R  # noqa: E402
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import FunctionalDecoder  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--decode", action="store_true", help="also decode one token at the end of the context")
    parser.add_argument(
        "--probe-above-supported-expecting-dram-oom",
        action="store_true",
        help="treat an out-of-memory failure as the expected outcome (records the first failing length)",
    )
    args = parser.parse_args()

    max_seq_len = args.max_seq_len or max(args.seq_len, 1)
    hf_config = R.hf_config()
    hidden_size = hf_config.text_config.hidden_size
    state_dict = R.synthetic_state_dict(args.layer)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = FunctionalDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=args.layer,
            mesh_device=mesh,
            max_batch_size=args.batch,
            max_seq_len=max_seq_len,
            page_block_size=args.block,
            prefill_chunk_size=args.chunk,
        )
        blocks_per_seq = (max_seq_len + args.block - 1) // args.block
        page_table = ttnn.from_torch(
            torch.arange(args.batch * blocks_per_seq, dtype=torch.int32).reshape(args.batch, blocks_per_seq),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden = R.synthetic_hidden_states(1, args.seq_len).reshape(1, 1, args.seq_len, hidden_size)
        tt_hidden = ttnn.from_torch(
            hidden,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        del hidden
        start = time.time()
        out = decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0)
        ttnn.synchronize_device(mesh)
        elapsed = time.time() - start
        shape = tuple(out.shape)
        tail = ttnn.to_torch(ttnn.slice(out, [0, 0, args.seq_len - 32, 0], [1, 1, args.seq_len, hidden_size]))
        finite = bool(torch.isfinite(tail.float()).all())
        ttnn.deallocate(out)
        ttnn.deallocate(tt_hidden)
        print(
            f"CAPACITY_PROBE_PASS mode=prefill seq_len={args.seq_len} output_shape={shape} "
            f"tail_finite={finite} seconds={elapsed:.1f}"
        )

        if args.decode:
            pos = args.seq_len - 1 if args.seq_len <= max_seq_len else max_seq_len - 1
            hidden = R.synthetic_hidden_states(args.batch, 1).reshape(1, 1, args.batch, hidden_size)
            tt_hidden = ttnn.from_torch(
                hidden,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            current_pos = ttnn.from_torch(
                torch.full((args.batch,), pos, dtype=torch.int32),
                device=mesh,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            rope_pos_ids = ttnn.from_torch(
                torch.full((1, args.batch), pos, dtype=torch.int32),
                device=mesh,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            start = time.time()
            out = decoder.decode_forward(
                tt_hidden, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids
            )
            ttnn.synchronize_device(mesh)
            elapsed = time.time() - start
            finite = bool(torch.isfinite(ttnn.to_torch(out).float()).all())
            print(
                f"CAPACITY_PROBE_PASS mode=decode cur_pos={pos} output_shape={tuple(out.shape)} "
                f"finite={finite} seconds={elapsed:.1f}"
            )
    except RuntimeError as error:
        message = str(error)
        oom = "Out of Memory" in message or "out of memory" in message.lower()
        if args.probe_above_supported_expecting_dram_oom and oom:
            print(f"CAPACITY_PROBE_EXPECTED_DRAM_OOM seq_len={args.seq_len}")
            print(message[:2000])
            return 0
        print(f"CAPACITY_PROBE_FAIL seq_len={args.seq_len} oom={oom}")
        print(message[:4000])
        return 1
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
