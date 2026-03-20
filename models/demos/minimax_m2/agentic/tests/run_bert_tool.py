#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Quick standalone test: BERTTool on N300."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from loguru import logger

import ttnn

CONTEXT = (
    "The Tenstorrent N300 is an AI accelerator board featuring two Wormhole B0 chips "
    "connected via a high-bandwidth Ethernet link. Each chip has 80 Tensix cores and "
    "12 GB of GDDR6 memory, giving the board 24 GB total. "
    "The N300 is designed for inference workloads at the edge and in the data centre."
)


def main():
    logger.info("Opening single device (chip 0) for BERT...")
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    device.enable_program_cache()

    try:
        from models.demos.minimax_m2.agentic.tool_wrappers.bert_tool import BERTTool

        logger.info("Loading BERTTool...")
        bert = BERTTool(mesh_device=device)

        questions = [
            ("How many chips does the N300 have?", "two"),
            ("How many Tensix cores per chip?", "80"),
            ("How much total memory does the N300 have?", "24"),
        ]

        for q, expected in questions:
            ans = bert.qa(q, CONTEXT)
            ok = expected.lower() in ans.lower()
            status = "PASS" if ok else "FAIL"
            logger.info(f"[{status}] Q: {q!r} => {ans!r}")

        logger.info("BERT tool: ALL DONE")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
