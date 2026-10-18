# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY: FP32 state resident in full-sync DST, native no-clear release."""

import hashlib
import inspect
import textwrap
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

REPO = Path(__file__).resolve().parents[4]
NATIVE_PINS = {
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h": "933caf779e7deb9476450deba04553ab900b35222357a5f52294b3bc395b18b4",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h": "343195b140e5d1eabcba970b5e8f9dc0c9107a51b0dfaa53920ca4e4efa1283e",
    "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp": "581b21aa6830f1d2d0429a33aa9af81639d95f005e4d9ee38912363d6dcc72c8",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h": "b8f85a9151d88409aefb0643d37291f7d330c9f1385ac0e4a8fb4334c4dfcd39",
}
for name, expected in NATIVE_PINS.items():
    if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Native DST/reduction contract changed: {name}")

baseline = qualified.baseline
HEADER_PATH = qualified.HEADER_PATH
BODY_PATH = Path(__file__).with_name("benchmark_gdn_resident_compute.cpp")
COMPUTE = qualified.COMPUTE[: qualified.COMPUTE.index("\nvoid product(")] + BODY_PATH.read_text()
READER = qualified.replace(
    qualified.READER,
    """        cb_reserve_back(0,4);
        for (uint32_t i=0;i<4;++i) {""",
    """        if(token==0) {
        cb_reserve_back(0,4);
        for (uint32_t i=0;i<4;++i) {""",
)
READER = qualified.replace(READER, "        cb_push_back(0,4);", "        cb_push_back(0,4);\n        }")
READER = qualified.replace(READER, "        cb_wait_front(12,4);", "        if(token+1==length) cb_wait_front(12,4);")
READER = qualified.replace(READER, "        cb_pop_front(12,4);", "        if(token+1==length) cb_pop_front(12,4);")


def launcher_source():
    source = textwrap.dedent(inspect.getsource(baseline._rm_recurrence))
    source = qualified.replace(
        source, "def _rm_recurrence(self, q, k, v, g, beta):", "def resident_recurrence(self, q, k, v, g, beta):"
    )
    source = qualified.replace(source, "jobs = batch * 48", "jobs = batch * 96")
    source = qualified.replace(source, "reader = _RM_QK_READER", "reader = READER")
    source = qualified.replace(source, "kernel_source=_BROADCAST_DECODE_COMPUTE,", "kernel_source=COMPUTE,")
    source = qualified.replace(source, "*job, length])", "*job, length, self._half_column_poison])")
    source = qualified.replace(source, "dst_full_sync_en=False", "dst_full_sync_en=True")
    source = qualified.replace(source, "[job[1] * length]", "[job[1], length]")
    return source


LAUNCHER_SOURCE = launcher_source()
_namespace = dict(vars(baseline), READER=READER, COMPUTE=COMPUTE)
exec(compile(LAUNCHER_SOURCE, __file__, "exec"), _namespace)
RECURRENCE = _namespace["resident_recurrence"]
gdn_method = qualified.gdn_method
