#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Verify record tensor widths / token ids vs ``lmsys/DeepSeek-R1-NextN`` config (no weight download)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.demos.speculative_deepseek_r1_broad.record_nextn_alignment import check_record_nextn_alignment


def main() -> None:
    p = argparse.ArgumentParser(description="Check record vs DeepSeek-R1-NextN config dimensions")
    p.add_argument("--record", type=Path, required=True, help="Path to .pt trace or MTP reference")
    p.add_argument("--batch-index", type=int, default=0)
    p.add_argument(
        "--nextn-model-id",
        type=str,
        default="lmsys/DeepSeek-R1-NextN",
        help="HF repo id used only for AutoConfig",
    )
    p.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trust remote code when loading NextN AutoConfig (default: true)",
    )
    args = p.parse_args()

    rep = check_record_nextn_alignment(
        args.record,
        batch_index=args.batch_index,
        nextn_model_id=args.nextn_model_id,
        trust_remote_code=args.trust_remote_code,
    )
    for line in rep.summary_lines():
        print(line)
    if not rep.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
