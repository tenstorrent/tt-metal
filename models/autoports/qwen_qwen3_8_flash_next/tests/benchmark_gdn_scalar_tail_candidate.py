# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY: validated scalar-cache reader/launcher plus validated tail math."""
import hashlib
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_resident_tail_candidate as tail
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_scalar_cache_candidate as scalar

PINS = {
    **scalar.PINS,
    **tail.PINS,
    **{
        "benchmark_gdn_resident_tail_candidate.py": "6e806b9c8af8c57681c2f5fa40c72a0fee35152b3b29d5de2f7031dae00be67b",
        "benchmark_gdn_resident_tail_compute.hpp": "c04687297842295f051ecb84fa735de7511a7232eb2da317753a8b0adceea3ec",
        "benchmark_gdn_resident_tail_probe.py": "d48110fb89f01806ce2a5358db397cf0b6ee8b39c37cdfd19b957289cc4b371d",
        "benchmark_gdn_scalar_cache_candidate.py": "f9094c50dcb83ddaddd18a0e17bb070fa84cb33de4dc5bd08fda7795f185ce4b",
        "benchmark_gdn_scalar_cache_probe.py": "9d1338c112515ca63ec07d0841e0c4268f9e6ccc469a4fb82d10b33fe1921ae7",
    },
}
for name, expected in PINS.items():
    if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Qualified component source changed: {name}")
REPO = scalar.REPO
NATIVE_PINS = {**scalar.NATIVE_PINS, **tail.NATIVE_PINS}
resident = tail.resident
assert resident is scalar.narrow.resident
baseline = resident.baseline
HEADER_PATH = resident.HEADER_PATH
BODY_PATH = tail.BODY_PATH
READER = scalar.READER
COMPUTE = tail.COMPUTE
LAUNCHER_SOURCE = scalar.LAUNCHER_SOURCE
_namespace = dict(vars(baseline), READER=READER, COMPUTE=COMPUTE)
exec(compile(LAUNCHER_SOURCE, __file__, "exec"), _namespace)
RECURRENCE = _namespace["resident_recurrence"]
gdn_method = resident.gdn_method


def recurrence(name):
    if name == "current_rm":
        return baseline._rm_recurrence
    if name == "resident_control":
        return resident.RECURRENCE
    if name == "scalar_control":
        return scalar.RECURRENCE
    return RECURRENCE
