# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY: exact validated tail compute plus validated narrow reader."""
import hashlib
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_narrow_reader_candidate as narrow
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_resident_tail_candidate as tail

PINS = {
    **narrow.PINS,
    **tail.PINS,
    **{
        "benchmark_gdn_resident_tail_candidate.py": "6e806b9c8af8c57681c2f5fa40c72a0fee35152b3b29d5de2f7031dae00be67b",
        "benchmark_gdn_resident_tail_compute.hpp": "c04687297842295f051ecb84fa735de7511a7232eb2da317753a8b0adceea3ec",
        "benchmark_gdn_resident_tail_probe.py": "d48110fb89f01806ce2a5358db397cf0b6ee8b39c37cdfd19b957289cc4b371d",
        "benchmark_gdn_narrow_reader_candidate.py": "b38528618db38a086c5050fce575e0dce6b9a006ddf26c1b6805e978eade6fb2",
        "benchmark_gdn_narrow_reader_probe.py": "c5cc0e0acf4e41c97a86f4cfedca82d96b96b3729e8809ddccc94c6005a64cad",
    },
}
for name, expected in PINS.items():
    if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Qualified component source changed: {name}")
REPO = narrow.REPO
NATIVE_PINS = {**narrow.NATIVE_PINS, **tail.NATIVE_PINS}
resident = tail.resident
assert resident is narrow.resident
assert tail.LAUNCHER_SOURCE == narrow.LAUNCHER_SOURCE == resident.LAUNCHER_SOURCE
baseline = resident.baseline
HEADER_PATH = resident.HEADER_PATH
BODY_PATH = tail.BODY_PATH
READER = narrow.READER
COMPUTE = tail.COMPUTE
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
