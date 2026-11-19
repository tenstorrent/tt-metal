# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run recorded-HF gates using the actual default multichip decoder."""

import hashlib
import json
import os
import sys
from pathlib import Path

from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_recorded_batch as harness
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder

if __name__ == "__main__":
    harness.GdnFidelityCandidate = MultichipDecoder
    harness.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    data = json.loads(output.read_text())
    source = Path(__file__).resolve().parents[1] / "tt/multichip_decoder.py"
    data.update(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
        skip_program_cache_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
        final_default=True,
    )
    output.write_text(json.dumps(data, indent=2) + "\n")
