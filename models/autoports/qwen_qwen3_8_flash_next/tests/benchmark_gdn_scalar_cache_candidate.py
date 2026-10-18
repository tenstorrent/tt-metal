# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY: cache scalar tiles32 tokens; prune empty Q/K reader barriers."""

import hashlib
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_narrow_reader_candidate as narrow

PINS = {
    **narrow.PINS,
    "benchmark_gdn_narrow_reader_candidate.py": "b38528618db38a086c5050fce575e0dce6b9a006ddf26c1b6805e978eade6fb2",
    "benchmark_gdn_narrow_reader_probe.py": "c5cc0e0acf4e41c97a86f4cfedca82d96b96b3729e8809ddccc94c6005a64cad",
}
for name, expected in PINS.items():
    if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Qualified narrow-reader source changed: {name}")
REPO = narrow.REPO
NATIVE_PINS = narrow.NATIVE_PINS
baseline = narrow.baseline
HEADER_PATH = narrow.HEADER_PATH
BODY_PATH = narrow.BODY_PATH
replace_once = narrow.replace_once


def transform_reader(source):
    source = replace_once(
        source,
        "    const uint32_t scratch = get_write_ptr(14);",
        "    const uint32_t scratch = get_write_ptr(14);\n"
        "    cb_reserve_back(16,2);\n"
        "    const uint32_t scalar_cache = get_write_ptr(16);",
    )
    source = replace_once(
        source,
        """                if (which==3) noc.async_read(v,CoreLocalMem<uint32_t>(scratch),64,{.page_id=page,.offset_bytes=4*off(h%32,half*16)},{});
                noc.async_read_barrier();""",
        """                if (which==3) noc.async_read(v,CoreLocalMem<uint32_t>(scratch),64,{.page_id=page,.offset_bytes=4*off(h%32,half*16)},{});
                if(which==3) noc.async_read_barrier();""",
    )
    old = """        for(uint32_t which=4;which<=5;++which) {
            cb_reserve_back(which,1);
            if(which==4) noc.async_read(decay,CoreLocalMem<uint32_t>(scratch),64,{.page_id=(b*st+token/32)+h/32,.offset_bytes=4*off(token%32,0)},{});
            else noc.async_read(beta,CoreLocalMem<uint32_t>(scratch),64,{.page_id=(b*st+token/32)+h/32,.offset_bytes=4*off(token%32,0)},{});
            noc.async_read_barrier();
            const uint32_t value=src[h%32];"""
    new = """        if(token%32==0) {
            noc.async_read(decay,CoreLocalMem<uint32_t>(scalar_cache),4096,{.page_id=(b*st+token/32)+h/32},{});
            noc.async_read(beta,CoreLocalMem<uint32_t>(scalar_cache+4096),4096,{.page_id=(b*st+token/32)+h/32},{});
            noc.async_read_barrier();
        }
        for(uint32_t which=4;which<=5;++which) {
            cb_reserve_back(which,1);
            auto scalar_src=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scalar_cache+(which-4)*4096);
            const uint32_t value=scalar_src[off(token%32,h%32)];"""
    return replace_once(source, old, new)


def transform_launcher(source):
    return replace_once(
        source,
        "    cbs = [",
        "    if len(sizes) != 16:\n"
        "        raise ValueError('Scalar-cache CB16 requires qualified prefill geometry')\n"
        "    sizes.append(2)  # CB16 owns8192 bytes, reserved by reader only.\n"
        "    cbs = [",
    )


READER = transform_reader(narrow.READER)
COMPUTE = narrow.COMPUTE
LAUNCHER_SOURCE = transform_launcher(narrow.LAUNCHER_SOURCE)
_namespace = dict(vars(baseline), READER=READER, COMPUTE=COMPUTE)
exec(compile(LAUNCHER_SOURCE, __file__, "exec"), _namespace)
RECURRENCE = _namespace["resident_recurrence"]
gdn_method = narrow.gdn_method


def recurrence(name):
    if name == "current_rm":
        return baseline._rm_recurrence
    if name == "resident_control":
        return narrow.resident.RECURRENCE
    if name == "narrow_control":
        return narrow.RECURRENCE
    return RECURRENCE
