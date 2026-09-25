#!/usr/bin/env python3
"""Send the SHUTDOWN sentinel to the live runner (H2D service). Needs the same PREFILL_* env as the producer."""

import os
import struct
import sys

sys.path.insert(
    0, os.environ.get("TT_METAL_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5)))
)
import ttnn
from models.demos.common.prefill.runners import prefill_producer as pp

svc = ttnn.H2DStreamService.connect(os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill"), timeout_ms=60000)
svc.forward_to_tensor_bytes(pp._chunk_to_host_array([1] * pp.CHUNK_SIZE), metadata=struct.pack("<iii", -1, -1, -1))
svc.barrier()
print("[shutdown] sentinel sent")
