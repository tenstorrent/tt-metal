#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize contents of an MTP reference .pt without loading multi-GB tensors into RAM.

Uses the zip layout of torch.save: reads only ``data.pkl`` (small) and prints tensor keys
from the pickle structure. For full shapes/dtypes, run with PyTorch::

  python inspect_mtp_record.py /path/to/mtp_full_model_seq128.pt --full
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.demos.speculative_deepseek_r1_broad.default_paths import DEFAULT_MTP_RECORD_PATH


def _pkl_tensor_keys_from_zip(pt_path: Path) -> list[str]:
    with zipfile.ZipFile(pt_path, "r") as zf:
        pkl_names = [n for n in zf.namelist() if n.endswith("/data.pkl")]
        if not pkl_names:
            raise ValueError(f"No data.pkl inside zip: {pt_path}")
        raw = zf.read(pkl_names[0])
    # PyTorch pickles are mostly binary; top-level dict key names appear as ASCII substrings.
    known = (
        "metadata",
        "mesh_shape",
        "hidden_states",
        "logits",
        "next_tokens",
        "start_tokens",
        "prefill_token_ids",
        "prompt_token_ids",
        "input_ids",
        "input_token_ids",
        "num_prefill_tokens",
        "prompt_len",
        "prefill_len",
        "num_steps",
        "batch_size",
        "hidden_size",
        "vocab_size",
    )
    return sorted(k for k in known if k.encode("ascii") in raw)


def _full_report(pt_path: Path) -> None:
    import torch

    print("Loading (needs RAM ~ file size):", pt_path)
    payload = torch.load(str(pt_path), map_location="cpu")
    if not isinstance(payload, dict):
        print(type(payload))
        return
    print("Keys:", sorted(payload.keys()))
    for k in sorted(payload.keys()):
        v = payload[k]
        if torch.is_tensor(v):
            print(f"  {k}: tensor {tuple(v.shape)} {v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k}: dict keys {list(v.keys())}")
        else:
            print(f"  {k}: {type(v).__name__}")


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect MTP reference .pt structure")
    p.add_argument("record", type=Path, nargs="?", default=None, help="Path to .pt file")
    p.add_argument(
        "--full",
        action="store_true",
        help="torch.load entire file and print shapes (heavy)",
    )
    args = p.parse_args()
    path = args.record or DEFAULT_MTP_RECORD_PATH
    if not path.is_file():
        print("File not found:", path, file=sys.stderr)
        sys.exit(1)
    print("File:", path)
    print("Size bytes:", path.stat().st_size)
    if args.full:
        _full_report(path)
        return
    keys = _pkl_tensor_keys_from_zip(path)
    print("Detected top-level names (from data.pkl, no tensor load):", keys)
    print()
    print("Expected layout (MTP reference):")
    print("  hidden_states  [num_steps, batch, hidden_size]")
    print("  next_tokens    [num_steps, batch]")
    print("  start_tokens   [batch]")
    print("  prefill_token_ids / input_ids / input_token_ids  optional full prompt for replay+decode")
    print("  logits         [num_steps, batch, vocab_size]  (often present; eagle3 replay may ignore)")
    print("  metadata       dict (model_id, num_steps, batch_size, hidden_size, vocab_size, ...)")
    print()
    print("For exact shapes:  python", Path(__file__).name, str(path), "--full")


if __name__ == "__main__":
    main()
