# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure the immutable completed stage-four decoder with the current harness."""

import hashlib
import json
import sys
from pathlib import Path

from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_probe as harness
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_stage4_reference as frozen


if __name__ == "__main__":
    harness.MultichipDecoder = frozen.MultichipDecoder
    harness.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    result = json.loads(output.read_text())
    result.update(
        implementation_class="immutable completed stage-four MultichipDecoder",
        implementation_source=str(Path(frozen.__file__)),
        source_sha256=hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest(),
        measurement_role="before stage-five optimization, TP4",
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
