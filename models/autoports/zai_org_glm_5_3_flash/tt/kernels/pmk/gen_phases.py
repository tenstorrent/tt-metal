import os
import re
import sys

# the kernels tree: GEN_K, else the tree this script sits in. Targets: argv (dst names), else every target -- the
# ffn_* / comb_* / plan_* forms carry later hand edits (v2 levers): regenerate only what you name
K = os.environ.get("GEN_K") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONLY = set(sys.argv[1:])
FULL = "models/autoports/zai_org_glm_5_3_flash/tt/kernels"
HDR = (
    "// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.\n//\n// SPDX-License-Identifier: Apache-2.0\n\n"
    "// PMK phase form of {src} (tt/pmk_moe.py; kernels/pmk/moe_common.hpp): the ORIGINAL kernel's arithmetic, CB roles\n"
    "// and argument order, with its compile-time args at CT0 + i, its runtime args at RT0 + i (the including composite\n"
    "// defines both before this header, inside the phase's namespace) and its CBs at pmk_moe's indices. Generated from\n"
    "// the original by doc/ttnn/16-prefill-production/pmk/moe/gen_phases.py -- edit the original, regenerate.\n"
    "// ---- original header follows ----\n"
)


def xform(src, dst, cbmap, extra=()):
    if ONLY and dst not in ONLY:
        return
    s = open(os.path.join(K, src)).read()
    s = s.replace("#pragma once\n", "")
    # strip the SPDX header of the original (kept in HDR)
    s = re.sub(r"\A// SPDX[^\n]*\n//\n// SPDX[^\n]*\n\n", "", s)
    s = re.sub(r"get_compile_time_arg_val\((\d+)\)", r"get_compile_time_arg_val(CT0 + \1)", s)
    s = re.sub(r"TensorAccessorArgs<(\d+)>", r"TensorAccessorArgs<CT0 + \1>", s)
    s = re.sub(r"get_arg_val<uint32_t>\((\d+)\)", r"get_arg_val<uint32_t>(RT0 + \1)", s)
    s = s.replace("void kernel_main()", "FORCE_INLINE void phase_main()")
    s = s.replace('#include "moe_xffn_common.hpp"', '#include "%s/pmk/moe_common.hpp"' % FULL)
    s = s.replace('#include "moe_xffn_probe.hpp"', '#include "%s/prefill/moe_xffn_probe.hpp"' % FULL)
    s = s.replace("using namespace xffn;", "using namespace pmk_moe;")
    for a, b in cbmap:
        s = re.sub(r"\b%s\b" % a, b, s)
    for a, b in extra:
        s = s.replace(a, b)
    open(os.path.join(K, "pmk", dst), "w").write(HDR.format(src="kernels/" + src) + s)
    print(dst)


xform("prefill/moe_xffn_reader.cpp", "ffn_reader.hpp", [])
xform("prefill/moe_xffn_writer.cpp", "ffn_writer.hpp", [])
xform("prefill/moe_xffn_compute.cpp", "ffn_compute.hpp", [])
plan_cb = [("PLAN_CB", "CB_PLAN")]
xform("prefill/moe_plan_mc_count.cpp", "plan_count.hpp", plan_cb)
xform("prefill/moe_plan_mc_dest.cpp", "plan_dest.hpp", plan_cb)
xform("prefill/moe_plan_mc_runs.cpp", "plan_runs.hpp", plan_cb)
comb = [("CB_IN", "CB_CIN"), ("CB_W", "CB_CW"), ("CB_IDX", "CB_CIDX"), ("CB_OUT", "CB_COUT"), ("CB_RM", "CB_CRM")]
cdef = [
    ("    constexpr uint32_t CB_CIN = 0;\n", ""),
    ("    constexpr uint32_t CB_CW = 1;\n", ""),
    ("    constexpr uint32_t CB_CIDX = 2;\n", ""),
    ("    constexpr uint32_t CB_COUT = 16;\n", ""),
    ("    constexpr uint32_t CB_CRM = 17;\n", ""),
]
# W7 step 5 (G): 14's gang EP expert FFN (a642725faca; names CB_RIN / SEM_RING_* remapped by moe_common.hpp)
xform("prefill/moe_xffn_gang_reader.cpp", "gang_reader.hpp", [])
xform("prefill/moe_xffn_gang_writer.cpp", "gang_writer.hpp", [])
# PMK_PART_FID (as ffn_compute.hpp): the gang's matmuls at G's own fidelity inside a fused composite of a higher one
gang_fid = [
    ("matmul_block_init(", "GANG_MM_INIT("),
    ("matmul_block(", "GANG_MM("),
    (
        "using namespace pmk_moe;\n",
        "using namespace pmk_moe;\n\n"
        "// PMK_PART_FID (W7 step 5: G as a window of the fused layer program, kernel fidelity = the kda items' HiFi4): the\n"
        "// matmuls at G's own fidelity (kernels/pmk/pmk_fid.hpp); standalone = the API calls unchanged\n"
        "// PMK_GANG_FID (program G: the gang's LoFi under the combine's HiFi4 -- the rm combine's bf16 x fp32 weight\n"
        "// multiplies at LoFi read 0.975 x the served HiFi4 combine, d0 gangw 2026-10-25) wins over PMK_PART_FID\n"
        "#ifndef GANG_MM\n#if defined(PMK_GANG_FID)\n"
        "#define GANG_MM_INIT(...) pmk::mm_block_init_fid<PMK_GANG_FID>(__VA_ARGS__)\n"
        "#define GANG_MM(...) pmk::mm_block_fid<PMK_GANG_FID>(__VA_ARGS__)\n#elif defined(PMK_PART_FID)\n"
        "#define GANG_MM_INIT(...) pmk::mm_block_init_fid<PMK_PART_FID>(__VA_ARGS__)\n"
        "#define GANG_MM(...) pmk::mm_block_fid<PMK_PART_FID>(__VA_ARGS__)\n#else\n"
        "#define GANG_MM_INIT(...) matmul_block_init(__VA_ARGS__)\n#define GANG_MM(...) matmul_block(__VA_ARGS__)\n"
        "#endif\n#endif\n",
    ),
]
xform("prefill/moe_xffn_gang_compute.cpp", "gang_compute.hpp", [], gang_fid)
xform("prefill/moe_combine_rm_reader.cpp", "comb_reader.hpp", comb, cdef)
xform("prefill/moe_combine_rm_compute.cpp", "comb_compute.hpp", comb, cdef)
xform("prefill/moe_combine_rm_writer.cpp", "comb_writer.hpp", comb, cdef)
