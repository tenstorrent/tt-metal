# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY: validated full-CB half-columns plus FP32 unary ROW broadcast."""

import hashlib
import types
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_half_columns_candidate as qualified

PINS = {
    "benchmark_gdn_half_columns_candidate.py": "217aa4e160c70ebd2528b4c7c3c323f6eff0637a65093ab1a96c0870e01d302f",
    "benchmark_gdn_half_columns_compute.hpp": "34da1b85b632d9712891dbb37cea1b045927f6578e457eb722ae2a24704de74b",
    "benchmark_gdn_half_columns_probe.py": "50139735fd5be0575d4b964e61aa4aafa027c5dcde9561eca9d5cc7b0393dcc8",
}
for name, expected in PINS.items():
    actual = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Qualified half-column source changed: {name}: {actual}")

baseline = qualified.baseline
HEADER_PATH = qualified.HEADER_PATH
READER = qualified.READER
COMPUTE = qualified.replace(
    qualified.COMPUTE,
    """    } else {
        copy_init(cb);
        copy_tile(cb,tile,dst);
    }""",
    """    } else if(cb==11) {
        unary_bcast_init<BroadcastType::ROW>(cb);
        unary_bcast<BroadcastType::ROW>(cb,tile,dst);
        unary_bcast_uninit<BroadcastType::ROW>(cb);
    } else {
        copy_init(cb);
        copy_tile(cb,tile,dst);
    }""",
)
COMPUTE = qualified.replace(
    COMPUTE,
    "sfpu_mul_bcast_row_init(); sfpu_mul_bcast_row(0,1);",
    "mul_binary_tile_init(); half_mul(0,1,0);",
)
# Clone the qualified launcher namespace; never replace production/module globals.
RECURRENCE = types.FunctionType(
    qualified.RECURRENCE.__code__,
    dict(qualified.RECURRENCE.__globals__, COMPUTE=COMPUTE),
    "half_row_bcast_recurrence",
)
gdn_method = qualified.gdn_method
