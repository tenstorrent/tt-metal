#!/usr/bin/env python3
"""v3r_renames.py: v3r's identifier renames, and the join/split that moves its two ERISC programs into headers.

  rename SRC OUT [--check]   copy router variant SRC to OUT with RENAMES applied (whole identifiers, every
                             .cpp/.hpp/.h). --check also reports map entries missing from SRC and new names already in
                             it. Identifiers still carrying the study prefixes (e1_, E1_, E1X) are listed at the end.
  join V3R OUT               copy variant V3R to OUT with the program headers (SPLIT) inlined into
                             fabric_erisc_router_v3.cpp between `// v3r_split: begin/end <header>` lines
  split JOINED OUT           the inverse: cut those blocks back out into their headers

v3r = split(rename(v3l2be1b) + cleanup diff). To port a change made on v3l2be1b (e.g. variant v3l3) onto v3r:

    v3r_renames.py rename fabric_exp/variants/v3l2be1b /tmp/a
    v3r_renames.py rename fabric_exp/variants/v3l3     /tmp/b
    (cd /tmp && diff -ruN a b > v3l3.diff)
    v3r_renames.py join  fabric_exp/variants/v3r /tmp/v3r_joined
    patch -d /tmp/v3r_joined -p1 < /tmp/v3l3.diff
    v3r_renames.py split /tmp/v3r_joined fabric_exp/variants/v3r_v3l3

patch may report offsets or fuzz where v3r edited nearby comments; a rejected hunk is a comment conflict to merge by
hand (checked with a synthetic change to the forwarding visit and to local delivery: both applied, one with fuzz 1).

The renamed copy is an intermediate for diffing and need not compile by itself: the receiver ERISC's names are only
valid inside the `receiver_erisc` namespace that the cleanup diff adds. If the ported variant adds identifiers with a
study prefix, `rename` lists them; add them to RENAMES and rename both sides again.
"""
import os, re, shutil, sys

# (old, new), applied in this order. ERISC 1's program moves into `namespace receiver_erisc` and ERISC 0's
# `namespace e0` becomes `sender_erisc`, so ERISC 1's names drop their `e1_`/`E1` prefix.
RENAMES = [
    # ERISC 0: the sender ERISC's program.
    ("e0", "sender_erisc"),
    # ERISC 1: the receiver ERISC's program (namespace receiver_erisc).
    ("e1_receive_program", "runs"),
    ("e1_kernel_main", "run_kernel"),
    ("run_e1_receive_program", "run_main_loop"),
    ("E1Downstreams", "DownstreamConnections"),
    ("E1DownstreamCursor", "DownstreamCursor"),
    ("E1DownstreamSetup", "DownstreamSetup"),
    ("E1RxState", "RxChannelState"),
    ("E1Rx", "RxChannel"),
    ("E1SlotRecords", "SlotRecords"),
    ("E1_SLOT_SRC_MASK", "SLOT_SRC_MASK"),
    ("E1_SLOT_READ_LOCAL", "SLOT_READ_LOCAL"),
    ("E1_SLOT_READ_FWD", "SLOT_READ_FWD"),
    ("E1_SLOT_READS", "SLOT_READS"),
    ("E1_LOCAL_TYPES_READING_SLOT", "LOCAL_TYPES_READING_SLOT"),
    ("e1_stream_count_mask", "stream_count_mask"),
    ("e1_vc0_crosses_to_vc1", "vc0_crosses_to_vc1"),
    ("e1_read_stream_count", "read_stream_count"),
    ("e1_local_read_bits", "local_read_bits"),
    ("e1_slot_add", "slot_add"),
    ("e1_rx_state_init", "init_rx_channel_state"),
    ("e1_downstream_has_space_rt", "downstream_has_space_rt"),
    ("e1_downstream_has_space", "downstream_has_space"),
    ("e1_forward_payload", "forward_payload"),
    ("e1_admit_slots_shared", "admit_slots_shared"),
    ("e1_forward_slots_shared", "forward_slots_shared"),
    ("e1_local_delivery_shared", "local_delivery_shared"),
    ("e1_forward_to_rt", "forward_to_rt"),
    ("e1_forward_to", "forward_to"),
    ("e1_local_delivery", "local_delivery"),
    ("e1_action_uses", "action_uses"),
    ("e1_action_slots", "action_slots"),
    ("e1_needs_reload", "needs_credit_reload"),
    ("e1_read_live_credit", "read_live_credit"),
    ("e1_set_live_credit", "set_live_credit"),
    ("e1_has_credit_if_used", "has_credit_if_used"),
    ("e1_admit_2d", "admit_2d"),
    ("e1_forward_if_used", "forward_if_used"),
    ("e1_forward_2d", "forward_2d"),
    ("e1_rx_completions", "send_completions"),
    ("e1_retire", "retire_slot"),
    ("e1_rx_step", "rx_channel_step"),
    ("e1_downstream_setup", "downstream_setup"),
    ("e1_init_downstreams", "init_downstreams"),
    ("e1_open_downstreams", "open_downstreams"),
    ("e1_setup_downstream_cmd_bufs", "setup_downstream_cmd_bufs"),
    ("e1_downstream_vcs", "downstream_vcs"),
    ("e1_num_downstream_vcs", "num_downstream_vcs"),
    ("e1_downstreams_vc0", "downstreams_vc0"),
    ("e1_downstreams_vc1", "downstreams_vc1"),
    # Added by v3e1c (direct scatter/atomic/fused local delivery) and v3l3 (per-VC combined idle test).
    ("e1_local_write_uncounted", "local_write_uncounted"),
    ("COMBINED_TEST", "SKIP_IF_ALL_IDLE"),
    # Shared: the count field of a stream credit register (the generic loop and both ERISC programs use one mask).
    ("grouped_free_slots_imm_mask", "stream_count_mask"),
    # A pass-through wrapper with no logic of its own, replaced by the function it calls.
    ("generic_rx_local_delivery_impl", "execute_chip_unicast_to_local_chip_impl"),
]

STUDY_PREFIX = re.compile(r"\b(?:e1_\w+|E1_\w+|E1[A-Z]\w*)\b")
EXTS = (".cpp", ".hpp", ".h")

ROUTER = "fabric_erisc_router_v3.cpp"
HDR_DIR = "tt_metal/fabric/hw/inc/edm_fabric"
SPLIT = ["fabric_erisc_router_v3_receiver_erisc.hpp", "fabric_erisc_router_v3_sender_erisc.hpp"]
PREAMBLE = """// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Part of fabric_erisc_router_v3.cpp, which includes it once, at file scope: it uses that file's definitions and is
// not a standalone header.

"""


def files(d):
    for dp, _, fs in os.walk(d):
        for f in fs:
            if f.endswith(EXTS):
                yield os.path.join(dp, f)


def apply(text):
    for old, new in RENAMES:
        text = re.sub(r"\b%s\b" % re.escape(old), new, text)
    return text


def copy(src, out):
    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("elf", "*.pyc", "__pycache__"))


def rename(src, out, check):
    if check:
        text = "".join(open(p).read() for p in files(src))
        for old, new in RENAMES:
            if not re.search(r"\b%s\b" % re.escape(old), text):
                print(f"not found: {old}")
            if re.search(r"\b%s\b" % re.escape(new), text):
                print(f"new name already present: {new} (for {old})")
    copy(src, out)
    left = set()
    for p in files(out):
        t = open(p).read()
        n = apply(t)
        if n != t:
            open(p, "w").write(n)
        left |= set(STUDY_PREFIX.findall(n))
    if left:
        print("identifiers with a study prefix left (extend RENAMES):", " ".join(sorted(left)))


def include_line(h):
    return f'#include "{HDR_DIR}/{h}"\n'


def join(src, out):
    copy(src, out)
    rp = os.path.join(out, ROUTER)
    r = open(rp).read()
    for h in SPLIT:
        hp = os.path.join(out, HDR_DIR, h)
        t = open(hp).read()
        if not t.startswith(PREAMBLE) or r.count(include_line(h)) != 1:
            sys.exit(f"join: {h} lacks the preamble or is not included exactly once")
        r = r.replace(include_line(h), f"// v3r_split: begin {h}\n{t[len(PREAMBLE):]}// v3r_split: end {h}\n")
        os.remove(hp)
    open(rp, "w").write(r)


def split(src, out):
    copy(src, out)
    rp = os.path.join(out, ROUTER)
    r = open(rp).read()
    for h in SPLIT:
        b, e = f"// v3r_split: begin {h}\n", f"// v3r_split: end {h}\n"
        if r.count(b) != 1 or r.count(e) != 1:
            sys.exit(f"split: markers for {h} missing")
        i, j = r.index(b), r.index(e)
        os.makedirs(os.path.join(out, HDR_DIR), exist_ok=True)
        open(os.path.join(out, HDR_DIR, h), "w").write(PREAMBLE + r[i + len(b) : j])
        r = r[:i] + include_line(h) + r[j + len(e) :]
    open(rp, "w").write(r)


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(a) != 3 or a[0] not in ("rename", "join", "split"):
        sys.exit(__doc__)
    if a[0] == "rename":
        rename(a[1], a[2], "--check" in sys.argv)
    elif a[0] == "join":
        join(a[1], a[2])
    else:
        split(a[1], a[2])


if __name__ == "__main__":
    main()
