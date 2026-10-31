# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Materializes the fused head's kernels (the reference builds them as source strings, stitched by concatenation and
.replace) as the files under ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels/.

Run in the reference tree (mstaletovic/moe-trim, no device needed):
  P=/data/mstaletovic/tt-metal-trimb; cd $P && TT_METAL_HOME=$P LD_LIBRARY_PATH=$P/build_Release/lib \
    PYTHONPATH=<pyroot with ttnn>:$P python <this file> --out <clone>/ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels
  --check: rebuild every stitched kernel text from the files (includes inlined) and compare with the reference's
  (both clang-formatted - CLANG_FORMAT names the binary - comments and whitespace aside). Run the repo's
  clang-format on the files after a rewrite.

What changes from the reference text (everything else is the reference's text byte for byte):
  * shared pieces (x_gather3 _CT / _WALK / _LA_* / _RS_*, router_stream _ROWS / LSTREAM, routing_stream RA_SEND,
    prep_stream _ROUTE_FN, routing_a2a _PLAN through routing_stream._plan_fn) are headers, #included where the
    reference concatenated / spliced them;
  * placeholders: routing_a2a's planner __X__ -> RAP_X defines, prep_stream's __X__ -> RW_X defines (the host sets
    them; array tables as RW_XRING_INIT / RW_YRING_INIT / RW_CHIPS_INIT comma lists);
  * se_engine3's GROW(m, lr) macro text -> se_defs.hpp from SE_GROW_OWN_CR / SE_GROW_OWN_RT;
  * addresses the reference baked into defines (SE_EW, RT_ENG, RT_MSYNC, RT_PWORD, RT_PART) -> common runtime args
    (se_defs.hpp, router_addr.hpp), so the program hash holds no address.
"""

import argparse
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 7))
KDIR = "ttnn/cpp/ttnn/operations/experimental/moe_block/fused_head/device/kernels"
SPDX = "// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.\n//\n// SPDX-License-Identifier: Apache-2.0\n\n"
GEN = "// Generated from the reference by tests/.../moe_block/fused_head/gen_head_kernels.py; edit both together.\n"

# routing_stream.RoutingStream.kernel_subs keys (routing_a2a's planner placeholders)
RAP_KEYS = [
    "S", "K", "EPC", "R", "C", "ME", "A", "L", "P", "TPP", "M", "HBP", "BB", "BBP", "PCAP", "NPORT",
    "ROWS_CAP", "ENT_CAP", "WX", "WY", "YT", "DCACHE", "PROBE_NOSTORE", "CT0",
]  # fmt: skip
RW_SUBS = {
    "__WRAPX__": "RW_WRAPX",
    "__WRAPY__": "RW_WRAPY",
    "__XRING__": "RW_XRING_INIT",
    "__YRING__": "RW_YRING_INIT",
    "__CHIPS__": "RW_CHIPS_INIT",
    "__R__": "RW_R",
    "__C__": "RW_C",
    "__ME__": "RW_ME",
    "__K__": "RW_K",
    "__RW__": "RW_RW",
    "__E__": "RW_E",
}


def inc(name):
    return f'#include "{KDIR}/{name}"\n'


def sources():
    from models.demos.moe_block.tt import prep_stream, router_stream, routing_stream, se_engine3, x_gather3

    xg, rs, rt, se = x_gather3, routing_stream, router_stream, se_engine3

    # routing_a2a's planner as routing_stream._plan_fn makes it, its placeholders as RAP_ defines (sentinels: numbers
    # that occur nowhere else, mapped back to names)
    sent = {k: 987650000 + i for i, k in enumerate(RAP_KEYS)}
    plan = rs._plan_fn(sent)
    for k, v in sent.items():
        plan = re.sub(rf"\b{v}\b", f"RAP_{k}", plan)
    assert not re.search(r"\b98765\d{4}\b", plan)

    route = prep_stream._ROUTE_FN
    for a, b in RW_SUBS.items():
        route = route.replace(a, b)
    assert "__" not in route.replace("__attribute__", "")

    headers = {
        "xg_ct.hpp": xg._CT,
        "xg_walk.hpp": xg._WALK,
        "xg_la_reader.hpp": xg._LA_READER,
        "xg_rs_reader.hpp": xg._RS_READER,
        "xg_la_sender.hpp": xg._LA_SENDER,
        "xg_rs_sender.hpp": xg._RS_SENDER,
        "router_lstream.hpp": rt.LSTREAM,
        "router_rows.hpp": rt._ROWS,
        "ra_send.hpp": rs.RA_SEND,
        "ra_plan.hpp": plan,
        "route_words.hpp": route,
    }

    # the kernels: (file, reference text with the pieces as includes, the reference's stitched text for --check)
    def tail_after(full, piece_list):
        """full = concat(pieces) + rest: the rest."""
        pre = "".join(piece_list)
        assert full.startswith(pre)
        return full[len(pre) :]

    rd_pieces = [
        '#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\n',
        xg._CT,
        "constexpr auto x_args = TensorAccessorArgs<15>();\n",
        xg._WALK,
        xg._LA_READER,
        xg._RS_READER,
    ]
    rd_main = tail_after(xg._READER, rd_pieces)
    reader = (
        rd_pieces[0]
        + inc("xg_ct.hpp")
        + rd_pieces[2]
        + inc("xg_walk.hpp")
        + inc("xg_la_reader.hpp")
        + inc("xg_rs_reader.hpp")
        + rd_main
    )

    sd_head = xg._SENDER.split(xg._CT, 1)[0]
    sd_pieces = [sd_head, xg._CT, xg._WALK, xg._LA_SENDER, xg._RS_SENDER, rt.LSTREAM, rs.RA_SEND]
    sd_main = tail_after(xg._SENDER, sd_pieces)
    sender = (
        sd_head
        + inc("xg_ct.hpp")
        + inc("xg_walk.hpp")
        + inc("xg_la_sender.hpp")
        + inc("xg_rs_sender.hpp")
        + inc("router_lstream.hpp")
        + inc("ra_send.hpp")
        + sd_main
    )

    w_head, w_main = xg._WATCHER.split(rs.RA_SEND, 1)
    watcher = w_head + inc("ra_send.hpp") + w_main

    y_head, y_main = rs._Y_SEND.split(rs.RA_SEND, 1)
    ysend = y_head + inc("ra_send.hpp") + y_main

    # router_stream: the planner prefix (router_stream._ra_src: its includes, then ra::plan) under RA_PLAN, then the
    # kernel with _ROWS (and the compute's route words) spliced after its first "#include <cstdint>"
    ra_pre = (
        '#ifdef RA_PLAN\n#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\n' + inc("ra_plan.hpp") + "#endif\n"
    )
    cstd = "#include <cstdint>\n"
    r_ncrisc = inc("router_addr.hpp") + ra_pre + rt._NCRISC.replace(cstd, cstd + inc("router_rows.hpp"), 1)
    r_compute = inc("router_addr.hpp") + rt._COMPUTE.replace(
        cstd, cstd + inc("router_rows.hpp") + "#ifdef RT_ROUTE\n" + inc("route_words.hpp") + "#endif\n", 1
    )
    b_head = rt._BRISC.split(rt._ROWS, 1)[0]
    b_main = tail_after(rt._BRISC, [b_head, rt._ROWS])
    r_brisc = inc("router_addr.hpp") + ra_pre + b_head + inc("router_rows.hpp") + b_main

    kernels = {
        "dataflow/xg_reader.cpp": ("x_gather3.py:_READER (= _CT + _WALK + _LA_READER + _RS_READER + main)", reader),
        "dataflow/xg_sender.cpp": (
            "x_gather3.py:_SENDER (= _CT + _WALK + _LA_SENDER + _RS_SENDER + router_stream.LSTREAM + "
            "routing_stream.RA_SEND + main)",
            sender,
        ),
        "dataflow/xg_watcher.cpp": ("x_gather3.py:_WATCHER (+ routing_stream.RA_SEND under W_RA0)", watcher),
        "dataflow/ra_y_send.cpp": ("routing_stream.py:_Y_SEND (+ RA_SEND)", ysend),
        "dataflow/se_brisc.cpp": ("se_engine3.py:_BRISC", inc("se_defs.hpp") + se._BRISC),
        "dataflow/se_ncrisc.cpp": ("se_engine3.py:_NCRISC", inc("se_defs.hpp") + se._NCRISC),
        "compute/se_compute.cpp": ("se_engine3.py:_COMPUTE", inc("se_defs.hpp") + se._COMPUTE),
        "dataflow/router_ncrisc.cpp": (
            "router_stream.py:_NCRISC (+ _ROWS; + routing_stream._plan_fn(routing_a2a._PLAN) under RA_PLAN)",
            r_ncrisc,
        ),
        "compute/router_compute.cpp": (
            "router_stream.py:_COMPUTE (+ _ROWS; + prep_stream._ROUTE_FN under RT_ROUTE)",
            r_compute,
        ),
        "dataflow/router_brisc.cpp": (
            "router_stream.py:_BRISC (= head + _ROWS + main; + routing_stream._plan_fn under RA_PLAN)",
            r_brisc,
        ),
    }
    refs = {
        "dataflow/xg_reader.cpp": xg._READER,
        "dataflow/xg_sender.cpp": xg._SENDER,
        "dataflow/xg_watcher.cpp": xg._WATCHER,
        "dataflow/ra_y_send.cpp": rs._Y_SEND,
        "dataflow/se_brisc.cpp": se._BRISC,
        "dataflow/se_ncrisc.cpp": se._NCRISC,
        "compute/se_compute.cpp": se._COMPUTE,
        "dataflow/router_ncrisc.cpp": '#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\n'
        + plan
        + rt._NCRISC.replace(cstd, cstd + rt._ROWS, 1),
        "compute/router_compute.cpp": rt._COMPUTE.replace(cstd, cstd + rt._ROWS + route, 1),
        "dataflow/router_brisc.cpp": '#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\n' + plan + rt._BRISC,
    }
    origin = {
        "xg_ct.hpp": "x_gather3.py:_CT",
        "xg_walk.hpp": "x_gather3.py:_WALK",
        "xg_la_reader.hpp": "x_gather3.py:_LA_READER",
        "xg_rs_reader.hpp": "x_gather3.py:_RS_READER",
        "xg_la_sender.hpp": "x_gather3.py:_LA_SENDER",
        "xg_rs_sender.hpp": "x_gather3.py:_RS_SENDER",
        "router_lstream.hpp": "router_stream.py:LSTREAM",
        "router_rows.hpp": "router_stream.py:_ROWS",
        "ra_send.hpp": "routing_stream.py:RA_SEND",
        "ra_plan.hpp": "routing_stream.py:_plan_fn(routing_a2a.py:_PLAN) - placeholders __X__ as RAP_X defines",
        "route_words.hpp": "prep_stream.py:_ROUTE_FN - placeholders __X__ as RW_X defines",
    }
    return headers, origin, kernels, refs


def fragment_note(name):
    return (
        f"// A fragment of the stitched kernel text: it relies on what the including kernel declared before it (no\n"
        f"// include guard beyond #pragma once - included once a kernel).\n"
    )


def write(out, headers, origin, kernels):
    os.makedirs(os.path.join(out, "dataflow"), exist_ok=True)
    os.makedirs(os.path.join(out, "compute"), exist_ok=True)
    for name, text in headers.items():
        with open(os.path.join(out, name), "w") as f:
            f.write(SPDX + f"// Reference: models/demos/moe_block/tt/{origin[name]} (mstaletovic/moe-trim).\n" + GEN)
            f.write(fragment_note(name) + "#pragma once\n" + text.lstrip("\n"))
    for name, (src, text) in kernels.items():
        with open(os.path.join(out, name), "w") as f:
            f.write(SPDX + f"// Reference: models/demos/moe_block/tt/{src} (mstaletovic/moe-trim).\n" + GEN)
            f.write(text.lstrip("\n"))


def expand(out, path, seen=None):
    """A kernel file with our includes inlined and our headers' prologues dropped: the reference's stitched text."""
    with open(path) as f:
        text = f.read()
    body = text.split(GEN, 1)[1] if GEN in text else text
    body = body.replace("#pragma once\n", "", 1) if path.endswith(".hpp") else body
    for line in body.splitlines(keepends=True):
        if line.startswith("// A fragment of the stitched") or line.startswith("// include guard beyond"):
            body = body.replace(line, "", 1)
    pat = re.compile(r'#include "' + re.escape(KDIR) + r'/([^"]+)"\n')

    def sub(m):
        name = m.group(1)
        if name in ("router_addr.hpp", "se_defs.hpp"):
            return ""
        if name == "ra_plan.hpp" or name == "route_words.hpp":
            return "@@" + name + "@@"
        return expand(out, os.path.join(out, name))

    return pat.sub(sub, body)


def clang_format(text, style_root):
    """text formatted as the repo's clang-format formats the kernel files (CLANG_FORMAT: the binary, default the one on
    PATH - pre-commit's v19.1.4), so the check compares like with like (it inserts braces, wraps lines)."""
    import subprocess

    exe = os.environ.get("CLANG_FORMAT", "clang-format")
    return subprocess.run(
        [exe, f"--style=file:{os.path.join(style_root, '.clang-format')}", "--assume-filename=kernel.cpp"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def norm(text):
    """The text with comments and whitespace dropped."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def check(out, headers, kernels, refs):
    bad = 0
    for name, ref in refs.items():
        got = norm(expand(out, os.path.join(out, name)))
        # (the planner prefix's own #ifdef RA_PLAN wrapper and the route words' #ifdef RT_ROUTE: the reference added the
        # text only where they are defined)
        got = got.replace(
            norm('#ifdef RA_PLAN\n#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\n@@ra_plan.hpp@@#endif\n'),
            norm(
                '#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\n'
                + expand(out, os.path.join(out, "ra_plan.hpp"))
            ),
        )
        got = got.replace(
            norm("#ifdef RT_ROUTE\n@@route_words.hpp@@#endif\n"),
            norm(expand(out, os.path.join(out, "route_words.hpp"))),
        )
        ref = clang_format(ref, ROOT)
        ok = got == norm(ref)
        bad += not ok
        print(f"{'ok  ' if ok else 'DIFF'} {name}")
        if not ok:
            ref_n = norm(ref)
            i = next((i for i in range(min(len(got), len(ref_n))) if got[i] != ref_n[i]), min(len(got), len(ref_n)))
            print(f"    first difference at {i}: reference {ref_n[i:i + 120]!r} / file {got[i:i + 120]!r}")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    h, o, k, r = sources()
    if not a.check:
        write(a.out, h, o, k)
    raise SystemExit(check(a.out, h, k, r))
