#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Apply confirmed winners to kTable. Updates existing rows in place, appends new keys.

Written as a script because doing this by hand produced malformed `{{{` entries last time (caught in the
diff, not by the compiler). Verifies the brace shape of every line it writes and refuses to save otherwise.

usage: apply_table.py "SHAPE=Pk,Ns,Sm,kb,nsb[:note]" ...
"""
import re, sys

F = "ttnn/cpp/ttnn/operations/experimental/small_m_matmul/device/small_m_matmul_config.cpp"
cd = lambda v: -(-v // 32)
items = []
for a in sys.argv[1:]:
    shape, _, rest = a.partition("=")
    cfg, _, note = rest.partition(":")
    M, K, N = (int(x) for x in shape.split("x"))
    items.append(((cd(M), cd(K), cd(N)), shape, ",".join(str(int(x)) for x in cfg.split(",")), note))
s = open(F).read()
body_start = s.index("kTable = {")
body_end = s.index("};", body_start)
body = s[body_start:body_end]
n_upd = n_add = 0
for key, shape, cfg, note in items:
    a, b, c = key
    pat = re.compile(r"\{\{%d,\s*%d,\s*%d\},\s*\{([\d,\s]+)\}\},[^\n]*" % (a, b, c))
    m = pat.search(body)
    cfg_sp = ", ".join(cfg.split(","))
    if m:
        was = re.sub(r"\s+", "", m.group(1))
        if was == cfg.replace(" ", ""):
            print("  %-17s already %s, skipped" % (shape, cfg))
            continue
        body = (
            body[: m.start()]
            + "{{%d, %d, %d}, {%s}},  // %s  %s(was %s)" % (a, b, c, cfg_sp, shape, (note + " " if note else ""), was)
            + body[m.end() :]
        )
        n_upd += 1
        print("  %-17s UPDATE %s -> %s" % (shape, was, cfg))
    else:
        body = body.rstrip() + "\n        {{%d, %d, %d}, {%s}},  // %s  %s\n    " % (a, b, c, cfg_sp, shape, note)
        n_add += 1
        print("  %-17s ADD    %s" % (shape, cfg))
bad = [l for l in body.splitlines() if "{{{" in l or re.search(r"\}\},\s*\{", l)]
if bad:
    print("REFUSING TO SAVE - malformed lines:")
    [print("   ", l[:90]) for l in bad]
    sys.exit(1)
open(F, "w").write(s[:body_start] + body + s[body_end:])
print("%d updated, %d added" % (n_upd, n_add))
