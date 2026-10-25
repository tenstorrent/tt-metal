#!/usr/bin/env python3
"""Benign source edits for stability tests: perturb.py BASE_VARIANT_DIR OUT_DIR EDIT...

Each edit is the kind of change a contributor makes without expecting a performance effect; none changes what the
router does per packet. Each writes OUT_DIR/<edit>/ = BASE's files with that one edit applied (exits if an anchor
is missing, so an edit never silently becomes a no-op).
  reorder  swap two members of the sender channel state (first_slot_addr, last_slot_addr)
  field    add a member to the sender channel state, set at init and never read
  hb       write the router heartbeat every 128 outer iterations instead of every 64
Marker-based edits, for any design that carries the markers (comment lines, each on its own line):
  hook       a debug hook a contributor might add: a noinline call inside the main loop, on a value read once before
             the loop that is never 0x5AFE. Needs `// STABILITY_PROBE_DECL` at file scope (once, in the router .cpp),
             then `// STABILITY_PROBE_PRELOOP` before and `// STABILITY_PROBE_LOOP` inside each main loop (in pairs,
             same scope, PRELOOP first). Costs one compare per pass by itself.
  statefield one member added to each struct marked `// STABILITY_PROBE_STATE` (inside the struct body), never read
"""
import os, shutil, sys

CH = "fabric_erisc_datamover_channels.hpp"
RT = "fabric_erisc_router_v2.cpp"
EDITS = {
    "reorder": [
        (CH, "    std::size_t first_slot_addr;\n    std::size_t last_slot_addr;\n",
         "    std::size_t last_slot_addr;\n    std::size_t first_slot_addr;\n"),
    ],
    "field": [
        (CH, "    std::size_t slot_stride;\n    BufferIndex next_packet_buffer_index;\n};",
         "    std::size_t slot_stride;\n    BufferIndex next_packet_buffer_index;\n    std::size_t num_buffers_at_init;\n};"),
        (CH, "        this->next_packet_buffer_index = BufferIndex{0};\n",
         "        this->next_packet_buffer_index = BufferIndex{0};\n        this->num_buffers_at_init = NUM_BUFFERS;\n"),
    ],
    "hb": [
        (RT, "if ((++fabric_heartbeat_counter & 0x3F) == 0) {", "if ((++fabric_heartbeat_counter & 0x7F) == 0) {"),
    ],
}

MARK = "// STABILITY_PROBE_"


def marker_edit(d, e):
    """Apply a marker-based edit to every file in d; exit if the markers it needs are missing."""
    n = {"DECL": 0, "PRELOOP": 0, "LOOP": 0, "STATE": 0}
    for root, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(root, f)
            lines = open(p).read().split("\n")
            out = []
            for line in lines:
                out.append(line)
                if MARK not in line:
                    continue
                kind = line.strip()[len(MARK):].split()[0]
                ind = line[: len(line) - len(line.lstrip())]
                k = n[kind]
                n[kind] += 1
                if e == "hook" and kind == "DECL":
                    out += ["volatile uint32_t stability_probe_word = 0;",
                            "__attribute__((noinline)) void stability_probe_hook(uint32_t v) {",
                            '    asm volatile("" ::"r"(v) : "memory");', "}"]
                elif e == "hook" and kind == "PRELOOP":
                    out.append(f"{ind}const uint32_t stability_probe_{k} = stability_probe_word;")
                elif e == "hook" and kind == "LOOP":
                    out += [f"{ind}if (stability_probe_{k} == 0x5AFEu) {{",
                            f"{ind}    stability_probe_hook(stability_probe_{k});",
                            f"{ind}}}"]
                elif e == "statefield" and kind == "STATE":
                    out.append(f"{ind}uint32_t stability_probe_field_{k};")
            open(p, "w").write("\n".join(out))
    need = {"hook": ("DECL", "PRELOOP", "LOOP"), "statefield": ("STATE",)}[e]
    if any(n[k] == 0 for k in need) or (e == "hook" and (n["DECL"] != 1 or n["PRELOOP"] != n["LOOP"])):
        sys.exit(f"{e}: markers found {n}")


base, out, edits = sys.argv[1], sys.argv[2], sys.argv[3:]
for e in edits:
    d = os.path.join(out, e)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for f in os.listdir(base):
        if os.path.isfile(os.path.join(base, f)):
            shutil.copy(os.path.join(base, f), d)
        elif os.path.isdir(os.path.join(base, f)) and f != "elf":
            shutil.copytree(os.path.join(base, f), os.path.join(d, f))
    if e in ("hook", "statefield"):
        marker_edit(d, e)
        print(f"{e}: {d}")
        continue
    for f, old, new in EDITS[e]:
        p = os.path.join(d, f)
        s = open(p).read()
        if s.count(old) != 1:
            sys.exit(f"{e}: anchor found {s.count(old)} times in {f}")
        open(p, "w").write(s.replace(old, new))
    print(f"{e}: {d}")
