# Dynamic throttle harness: the emulation of matmul_block_math_dynamic_throttle on top of the tree's own perf_matmul configuration. sources/matmul_dyn_test.cpp is generated from the tree's sources/matmul_test.cpp: every math matmul call becomes
# "read the FW word at L1 16; odd = throttled form (level 5, tile MOP as ttnn's throttled branch), re-init only when the status changes".
# DYN mode: 0 the word as it stands, 1 forced odd, 2 toggled every call (re-init every call), 3 forced even.
# A tree with _llk_math_matmul_init_switch_ re-inits through it, as matmul_block_math_dynamic_throttle does; main's tree takes main's template
# arguments and its full init. Regenerate by importing the module without DYN_NOGEN; DYN_NOGEN keeps the committed source.
import os, re, tempfile
from dataclasses import dataclass

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity
from helpers.test_variant_parameters import TemplateParameter

_TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _generate():
    src = open(os.path.join(_TESTS, "sources", "matmul_test.cpp")).read()
    llk = open(os.path.join(_TESTS, "..", "tt_llk_blackhole", "llk_lib", "llk_math_matmul.h")).read()
    head = "_llk_math_matmul_init_switch_" in llk
    if head:
        call, init = "MATMUL_MATH_TEMPLATE_ARGS", "MATMUL_MATH_INIT_TEMPLATE_ARGS"
        tcall, tinit = "MATMUL_FIDELITY, 5, false, true, matmul_reuse_dir_k", "MATMUL_FIDELITY, 5, false, true, 0, false, false, matmul_reuse_dir_k, false"
        reinit = "_llk_math_matmul_init_switch_"
    else:
        call = init = "MATH_FIDELITY, THROTTLE_LEVEL"
        tcall = tinit = "MATH_FIDELITY, 5"
        reinit = "_llk_math_matmul_init_"
    defs = f"""
#ifndef tt_l1_ptr
#define tt_l1_ptr
#endif
#ifndef DYN_MODE
#define DYN_MODE 0
#endif
volatile tt_l1_ptr std::uint32_t* dyn_tp = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(16);
static std::uint32_t dyn_status = 0;
static std::uint32_t dyn_calls  = 0;
#define DYN_INIT_ARGS {init}
#define DYN_CALL_ARGS {call}
#define DYN_TINIT_ARGS {tinit}
#define DYN_TCALL_ARGS {tcall}
#define DYN_MATMUL()                                                                                               \\
    do {{                                                                                                           \\
        std::uint32_t fw = *dyn_tp;                                                                                 \\
        if (DYN_MODE == 1) fw |= 1;                                                                                 \\
        if (DYN_MODE == 2) fw ^= (dyn_calls++ & 1);                                                                 \\
        if (DYN_MODE == 3) fw &= ~1u;                                                                               \\
        if (fw % 2) {{                                                                                              \\
            if (dyn_status != 1) {{ {reinit}<DYN_TINIT_ARGS>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, UNPACK_TRANSPOSE_FACES, CT_DIM, RT_DIM); dyn_status = 1; }} \\
            _llk_math_matmul_<DYN_TCALL_ARGS>(0, CT_DIM, RT_DIM);                                                   \\
        }} else {{                                                                                                  \\
            if (dyn_status != 0) {{ {reinit}<DYN_INIT_ARGS>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, UNPACK_TRANSPOSE_FACES, CT_DIM, RT_DIM); dyn_status = 0; }} \\
            _llk_math_matmul_<DYN_CALL_ARGS>(0, CT_DIM, RT_DIM);                                                    \\
        }}                                                                                                          \\
    }} while (0)
"""
    i = src.index("#ifdef LLK_TRISC_MATH")
    j = src.index("void run_kernel", i)
    out = src[:j] + defs + "\n" + src[j:]
    pat = re.compile(r"_llk_math_matmul_<[^>]*>\(\s*/\*\s*dest_index\s*\*/\s*0,\s*CT_DIM,\s*RT_DIM\);")
    k = i + (len(defs) + 1)
    body = out[k:]
    body, n = pat.subn("DYN_MATMUL();", body)
    assert n == 2, n
    out = out[:k] + body
    path = os.path.join(_TESTS, "sources", "matmul_dyn_test.cpp")
    if not (os.path.exists(path) and open(path).read() == out):
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path)); os.write(fd, out.encode()); os.close(fd); os.replace(tmp, path)


if not os.environ.get("DYN_NOGEN"):
    _generate()
import perf_matmul as pm  # noqa: E402

F = DataFormat
CASES = [
    (fmt, sync, (rt, ct, kt), mf)
    for fmt in (InputOutputFormat(F.Float16_b, F.Float16_b), InputOutputFormat(F.Bfp8_b, F.Bfp8_b))
    for sync in (DestSync.Half, DestSync.Full)
    for rt, ct in ((1, 1), (1, 8), (4, 2), (8, 1))
    for kt in (1, 4, 32)
    for mf in (MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4)
]


def DYN(mode):
    @dataclass(repr=False)
    class _D(TemplateParameter):
        def __repr__(self) -> str:
            return f"DYN({mode})"

        def convert_to_cpp(self) -> str:
            return f"#define DYN_MODE {mode}"

    return _D()


def run_dyn(perf_report, fmt, sync, dims, mf, mode):
    rt, ct, kt = dims
    orig = pm.PerfConfig

    def make(source, formats, run_types, templates=None, **kw):
        return orig("sources/matmul_dyn_test.cpp", formats, run_types, templates=list(templates) + [DYN(mode)], **kw)

    pm.PerfConfig = make
    try:
        pm.test_perf_matmul(perf_report, (fmt, DestAccumulation.No, sync, ([rt * 32, kt * 32], [kt * 32, ct * 32])), mf)
    finally:
        pm.PerfConfig = orig
