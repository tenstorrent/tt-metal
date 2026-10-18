# SPDX-License-Identifier: Apache-2.0
"""Tests-only half-column recurrence; no production module/global mutation."""

import inspect
import textwrap
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tt import gdn_prefill_qk as baseline


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one source fragment: {old}")
    return source.replace(old, new)


def reader_source():
    source = baseline._RM_QK_READER
    source = replace(
        source, "const uint32_t head=job/4, column=job%4;", "const uint32_t head=job/8, column=(job%8)/2, half=job%2;"
    )
    source = replace(
        source,
        "const uint32_t length=get_arg_val<uint32_t>(10), st=(length+31)/32;",
        "const uint32_t length=get_arg_val<uint32_t>(10), st=(length+31)/32;\n    const uint32_t poison=get_arg_val<uint32_t>(11);",
    )
    source = replace(
        source,
        """            if(token==0) noc.async_read(state, CoreLocalMem<uint32_t>(get_write_ptr(0)+i*4096),4096,
                {.page_id=head*16+i*4+column},{});""",
        """            if(token==0) {
                auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+i*4096);
                for(uint32_t f=0;f<2;++f) {
                    for(uint32_t j=0;j<256;++j) dst[f*512+256+j]=poison;
                    noc.async_read(state,CoreLocalMem<uint32_t>(get_write_ptr(0)+i*4096+f*2048),1024,
                        {.page_id=head*16+i*4+column,.offset_bytes=f*2048+half*1024},{});
                }
            }""",
    )
    source = replace(
        source,
        "for(uint32_t c=0;c<32;++c) dst[tile*1024+off(0,c)]=src[off(h%32,c)];",
        """for(uint32_t c=0;c<16;++c) {
                        dst[tile*1024+off(0,c)]=src[off(h%32,c+half*16)];
                        dst[tile*1024+off(0,c+16)]=poison;
                    }""",
    )
    source = replace(
        source,
        """        if(token+1==length) for(uint32_t i=0;i<4;++i)
            noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(12)+i*4096),final,4096,{},
                {.page_id=head*16+i*4+column});""",
        """        if(token+1==length) for(uint32_t i=0;i<4;++i) for(uint32_t f=0;f<2;++f)
            noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(12)+i*4096+f*2048),final,1024,{},
                {.page_id=head*16+i*4+column,.offset_bytes=f*2048+half*1024});""",
    )
    source = replace(
        source,
        "for(uint32_t t=0;t<1;++t) for(uint32_t c=0;c<32;++c)",
        "for(uint32_t t=0;t<1;++t) for(uint32_t c=0;c<16;++c)",
    )
    return replace(
        source,
        """            for(uint32_t t=0;t<1;++t)
                noc.async_write(CoreLocalMem<uint32_t>(get_write_ptr(15)+t*4096),output,4096,{},
                    {.page_id=(head*st+token/32)*4+column});""",
        """            for(uint32_t f=0;f<2;++f)
                noc.async_write(CoreLocalMem<uint32_t>(get_write_ptr(15)+f*2048),output,1024,{},
                    {.page_id=(head*st+token/32)*4+column,.offset_bytes=f*2048+half*1024});""",
    )


READER = reader_source()
HEADER_PATH = Path(__file__).with_name("benchmark_gdn_half_columns_compute.hpp")
HEADER = HEADER_PATH.read_text()
COMPUTE = baseline._BROADCAST_DECODE_COMPUTE
# Keep ordinary includes/load/finish and all original recurrence sequencing.
anchor = "void load(uint32_t cb,uint32_t tile,uint32_t dst) {"
COMPUTE = replace(COMPUTE, anchor, HEADER + "\n" + anchor)
start = COMPUTE.index("template<uint32_t Out> void reduce_product() {")
end = COMPUTE.index("\nvoid product(", start)
COMPUTE = COMPUTE[:start] + COMPUTE[end:]
for old, new in (
    ("mul_binary_tile(", "half_mul("),
    ("add_binary_tile(", "half_add("),
    ("sub_binary_tile(", "half_sub("),
    ("reduce_product<", "half_reduce_product<"),
):
    # The helper header is deliberately not rewritten recursively.
    boundary = COMPUTE.index(anchor)
    COMPUTE = COMPUTE[:boundary] + COMPUTE[boundary:].replace(old, new)


def make_recurrence():
    source = textwrap.dedent(inspect.getsource(baseline._rm_recurrence))
    source = replace(
        source, "def _rm_recurrence(self, q, k, v, g, beta):", "def half_recurrence(self, q, k, v, g, beta):"
    )
    source = replace(source, "jobs = batch * 48", "jobs = batch * 96")
    source = replace(source, "reader = _RM_QK_READER", "reader = READER")
    source = replace(source, "kernel_source=_BROADCAST_DECODE_COMPUTE,", "kernel_source=COMPUTE,")
    source = replace(source, "*job, length])", "*job, length, self._half_column_poison])")
    namespace = dict(vars(baseline), READER=READER, COMPUTE=COMPUTE)
    exec(compile(source, __file__, "exec"), namespace)
    return namespace["half_recurrence"]


RECURRENCE = make_recurrence()


def gdn_method(local_recurrence):
    source = textwrap.dedent(inspect.getsource(baseline.PrefillQKRowMajorGDN._recurrence))
    namespace = dict(vars(baseline), _rm_recurrence=local_recurrence)
    # All selected probe shapes take the explicitly guarded RM branch, not super().
    exec(compile(source, __file__, "exec"), namespace)
    return namespace["_recurrence"]
