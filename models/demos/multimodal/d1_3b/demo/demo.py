# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""d1-3B decision demo on a single Wormhole chip.

MESH_DEVICE=N150 python models/demos/multimodal/d1_3b/demo/demo.py
"""

import json

from loguru import logger

import ttnn
from models.demos.multimodal.d1_3b.tt.d1 import TtD1
from models.demos.multimodal.d1_3b.tt.model import L1_SMALL_SIZE, TRACE_REGION_SIZE

REQUESTS = [
    (
        "I was charged twice this month, please refund one of them.",
        {
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Charges, refunds, invoices",
                    "technical": "App or site faults",
                    "fraud": "Suspected unauthorised use",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["Can wait", "Today", "Blocking the customer now"],
            },
        },
    ),
    (
        "The app crashes every time I open settings on Android.",
        {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Charges, refunds, invoices",
                    "technical": "App or site faults",
                    "fraud": "Suspected unauthorised use",
                },
            }
        },
    ),
]


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=L1_SMALL_SIZE, trace_region_size=TRACE_REGION_SIZE)
    try:
        d1 = TtD1.from_pretrained(mesh)
        for state, questions in REQUESTS:
            out = d1.system_one(state, questions)
            logger.info(f"STATE: {state}")
            logger.info(json.dumps(out, indent=2, ensure_ascii=False))
        d1.release()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
