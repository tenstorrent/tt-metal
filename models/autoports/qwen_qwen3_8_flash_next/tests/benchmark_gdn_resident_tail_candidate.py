# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY: prune unused delta rows and unused broadcast data faces."""
import hashlib
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_resident_candidate as resident

PINS = {
    "benchmark_gdn_resident_candidate.py": "f6be388cf6da89452127b8b4bf11918783b0e1d07dd89fa0c362045032f65cfc",
    "benchmark_gdn_resident_compute.cpp": "dddc593e27aac0d554ae4f5cb5cc37a8bf0d110b42f65e3e8e555c4c8dd14bb2",
    "benchmark_gdn_resident_probe.py": "deda247ad3f41389b8073f466412768c16ee07f962b399a86ec06bd63da95870",
}
for name, expected in PINS.items():
    if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Qualified resident source changed: {name}")
REPO = resident.REPO
NATIVE_PINS = dict(resident.NATIVE_PINS)
NATIVE_PINS[
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary_bcast.h"
] = "28bf9237ad07c5e71ad587a488c0c5927085474d9186c08f441062bfc0f85584"
for name, expected in NATIVE_PINS.items():
    if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Native SFPU contract changed: {name}")
BASELINE_PINS = resident.PINS
baseline = resident.baseline
HEADER_PATH = resident.HEADER_PATH
BODY_PATH = Path(__file__).with_name("benchmark_gdn_resident_tail_compute.hpp")
replace = resident.qualified.replace
COMPUTE = replace(resident.COMPUTE, "ALWI void reduce_dst4()", BODY_PATH.read_text() + "\nALWI void reduce_dst4()")
COMPUTE = replace(COMPUTE, "half_sub(5, 4, 4);", "resident_row_delta_sub(5, 4, 4);")
COMPUTE = replace(COMPUTE, "half_mul(4, 6, 4);", "resident_row_delta_mul(4, 6, 4);")
COMPUTE = replace(COMPUTE, "sfpu_mul_bcast_row(6, 4);", "resident_left_mul_bcast_row(6, 4);")
READER = resident.READER
LAUNCHER_SOURCE = resident.LAUNCHER_SOURCE
_namespace = dict(vars(baseline), READER=READER, COMPUTE=COMPUTE)
exec(compile(LAUNCHER_SOURCE, __file__, "exec"), _namespace)
RECURRENCE = _namespace["resident_recurrence"]
gdn_method = resident.gdn_method


def recurrence(name):
    if name == "current_rm":
        return baseline._rm_recurrence
    if name == "resident_control":
        return resident.RECURRENCE
    return RECURRENCE
