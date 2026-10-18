# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY: exact64B V/decay/beta reads; qualified resident math unchanged."""

import hashlib
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_resident_candidate as resident

PINS = {
    **resident.PINS,
    "benchmark_gdn_resident_candidate.py": "f6be388cf6da89452127b8b4bf11918783b0e1d07dd89fa0c362045032f65cfc",
    "benchmark_gdn_resident_compute.cpp": "dddc593e27aac0d554ae4f5cb5cc37a8bf0d110b42f65e3e8e555c4c8dd14bb2",
    "benchmark_gdn_resident_probe.py": "deda247ad3f41389b8073f466412768c16ee07f962b399a86ec06bd63da95870",
}
for name, expected in PINS.items():
    if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Qualified resident source changed: {name}")
REPO = resident.REPO
NATIVE_PINS = {
    **resident.NATIVE_PINS,
    "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h": "47fbf6f1be529f34f261d6c3567f4e8c83763fc0b50b5ebf9678f12973705829",
    "tt_metal/hw/inc/internal/debug/sanitize.h": "f71d92dba723b34101966eda13fc784f47b83af9d3606426d47dc6a0171773e9",
}
for name, expected in NATIVE_PINS.items():
    if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Native reader/alignment contract changed: {name}")


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one qualified reader fragment: {old}")
    return source.replace(old, new)


def transform_reader(source):
    source = replace_once(
        source,
        "if (which==3) noc.async_read(v,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});",
        "if (which==3) noc.async_read(v,CoreLocalMem<uint32_t>(scratch),64,"
        "{.page_id=page,.offset_bytes=4*off(h%32,half*16)},{});",
    )
    source = replace_once(source, "src[off(h%32,c+half*16)]", "src[c]")
    for prefix, tensor in (("if(which==4)", "decay"), ("else", "beta")):
        source = replace_once(
            source,
            f"{prefix} noc.async_read({tensor},CoreLocalMem<uint32_t>(scratch),4096,"
            "{.page_id=(b*st+token/32)+h/32},{});",
            f"{prefix} noc.async_read({tensor},CoreLocalMem<uint32_t>(scratch),64,"
            "{.page_id=(b*st+token/32)+h/32,.offset_bytes=4*off(token%32,0)},{});",
        )
    return replace_once(source, "const uint32_t value=src[off(token%32,h%32)];", "const uint32_t value=src[h%32];")


baseline = resident.baseline
HEADER_PATH = resident.HEADER_PATH
BODY_PATH = resident.BODY_PATH
READER = transform_reader(resident.READER)
COMPUTE = resident.COMPUTE
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
