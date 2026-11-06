#!/usr/bin/env python3
"""mk_roots.py [--dry] [ARM ...]: the slack study's runtime roots (NOTES.md; local, never committed).

Per router (main: /home/aagarwal/tt-metal-main, port: /home/aagarwal/tt-metal-port-slack) and arm, a root
/home/aagarwal/tt-metal-slack-roots/<router>_<arm>: a symlink farm of the tree (mkroot.py's layout: every directory on
an edited file's path is real, everything else a symlink into the tree) with its own empty generated/. Arms:
  plain       no file edited (the base runs)
  z           the call sites + slack_study_spin.hpp at tx 0, rx 0 (offline check only: must equal plain)
  tx<D>/rx<D> the call sites + slack_study_spin.hpp at that side's D, the other side 0 (wall-clock spin)
  mz, mtx<D>/mrx<D>  the same with the mcycle spin (patches/slack_study_spin_mcycle.hpp.in: the study's
              fabric_experiment_spin; the settling A/B of NOTES.md section 10)
The call sites are the same in every patched arm of a router; only the header's two constants differ. Each edit is an
anchored insertion that must match exactly once in the tree's file (else nothing is written). Also writes
patches/<router>_sites.patch (diff -u of the edited files against the tree) for the notes."""
import os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOTS = "/home/aagarwal/tt-metal-slack-roots"
TREES = {"main": "/home/aagarwal/tt-metal-main", "port": "/home/aagarwal/tt-metal-port-slack"}
HDR = "tt_metal/fabric/hw/inc/edm_fabric/slack_study_spin.hpp"
INC = '#include "tt_metal/fabric/hw/inc/edm_fabric/slack_study_spin.hpp"  // LOCAL slack study, never committed'
TX = "slack_study::spin<slack_study::tx_spin_cycles>();  // LOCAL slack study: E0, per packet sent"
RX = "slack_study::spin<slack_study::rx_spin_cycles>();  // LOCAL slack study: E1, per packet received and handled"

# (file, anchor, text inserted after the anchor). Anchors are whole lines (or line blocks) of the tree's file.
EDITS = {
    "port": [
        ("tt_metal/fabric/hw/inc/edm_fabric/router/router_sender.hpp",
         '#include "router_telemetry.hpp"\n', INC + "\n"),
        ("tt_metal/fabric/hw/inc/edm_fabric/router/router_sender.hpp",
         "    remote_update_ptr_val<RemoteReceiverChannel<VC>::packets_received_stream, sender_txq_id>(1);\n",
         "    " + TX + "\n"),
        ("tt_metal/fabric/hw/inc/edm_fabric/router/router_receiver.hpp",
         '#include "router_telemetry.hpp"\n', INC + "\n"),
        ("tt_metal/fabric/hw/inc/edm_fabric/router/router_receiver.hpp",
         "    retire<VC>(progress, sources, source_channel(packet.src_ch_id));\n", "    " + RX + "\n"),
    ],
    "main": [
        ("tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp",
         '#include "hostdevcommon/fabric_telemetry_msgs.h"\n', INC + "\n"),
        # send_next_data: after the packets-received register update, the send's last step
        ("tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp",
         "    SenderLinkTransport::remote_update_ptr_val<to_receiver_pkts_sent_id>(1U);\n}\n",
         None),  # special-cased below: insert before the closing brace
        # run_receiver_channel_step_impl: after the packet is forwarded / delivered and retired
        ("tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp",
         "            wr_sent_counter.increment();\n"
         "            // decrement the to_receiver_pkts_sent_id stream register by 1 since current packet has been processed.\n"
         "            if constexpr (!enable_first_level_ack) {\n"
         "                increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);\n"
         "            }\n",
         "            " + RX + "\n"),
        # speedy path (super_speedy_mode): the inlined send, and the receiver's per-packet block
        ("tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp",
         "            SenderLinkTransport::remote_update_ptr_val<to_receiver_pkts_sent_id>(1U);\n",
         "            " + TX + "\n"),
        ("tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp",
         "        receiver_state.unacked_sends++;\n", "        " + RX + "\n"),
    ],
}


def edited_files(router):
    tree = TREES[router]
    out = {}
    for rel, anchor, ins in EDITS[router]:
        text = out.get(rel) or open(os.path.join(tree, rel)).read()
        n = text.count(anchor)
        if n != 1:
            sys.exit(f"!!! {router} {rel}: anchor found {n} times: {anchor[:80]!r}")
        if ins is None:  # main's send_next_data: the spin goes before the function's closing brace
            text = text.replace(anchor, anchor[:-2] + "    " + TX + "\n}\n")
        else:
            text = text.replace(anchor, anchor + ins)
        out[rel] = text
    return out


def header(tx, rx, mcycle=False):
    t = open(os.path.join(HERE, "patches", "slack_study_spin_mcycle.hpp.in" if mcycle else "slack_study_spin.hpp.in")).read()
    return t.replace("@TX@", str(tx)).replace("@RX@", str(rx))


def farm(tree, root, rels):
    real_dirs = {""}
    for r in rels:
        parts = r.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            real_dirs.add("/".join(parts[:i]))

    def build(rel):
        os.makedirs(os.path.join(root, rel), exist_ok=True)
        for name in sorted(os.listdir(os.path.join(tree, rel))):
            r = f"{rel}/{name}" if rel else name
            if r == "generated":
                continue  # each root its own (the test writes its CSVs there)
            if r in real_dirs:
                build(r)
            elif r not in rels:
                os.symlink(os.path.join(tree, r), os.path.join(root, r))
    build("")
    os.makedirs(os.path.join(root, "generated"), exist_ok=True)


def arm_values(arm):
    if arm.startswith("m"):  # mcycle arms: mz, mtx<D>, mrx<D>
        arm = arm[1:]
    if arm in ("plain", "z"):
        return 0, 0
    side, d = arm[:2], int(arm[2:])
    return (d, 0) if side == "tx" else (0, d)


def main():
    dry = "--dry" in sys.argv
    arms = [a for a in sys.argv[1:] if not a.startswith("--")] or (
        ["plain", "z"] + [f"{s}{d}" for s in ("tx", "rx") for d in (4, 8, 16, 32, 64, 128)])
    for router, tree in TREES.items():
        files = edited_files(router)
        # the record: diff -u of the edited files against the tree (+ the new header at tx 0, rx 0)
        tmp = os.path.join(ROOTS, f".diff_{router}")
        shutil.rmtree(tmp, ignore_errors=True)
        for rel, text in files.items():
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
            open(os.path.join(tmp, rel), "w").write(text)
        d = "".join(subprocess.run(["diff", "-u", "--label", f"a/{rel}", "--label", f"b/{rel}",
                                    os.path.join(tree, rel), os.path.join(tmp, rel)],
                                   capture_output=True, text=True).stdout for rel in sorted(files))
        open(os.path.join(HERE, "patches", f"{router}_sites.patch"), "w").write(d)
        shutil.rmtree(tmp)
        for arm in arms:
            root = os.path.join(ROOTS, f"{router}_{arm}")
            tx, rx = arm_values(arm)
            rels = [] if arm == "plain" else sorted(files) + [HDR]
            print(f"{root}: farm of {tree}; edited: {', '.join(r.split('/')[-1] for r in rels) or '-'}; tx {tx} rx {rx}")
            if dry:
                continue
            if os.path.lexists(root):
                shutil.rmtree(root)
            farm(tree, root, rels)
            for rel in rels:
                open(os.path.join(root, rel), "w").write(header(tx, rx, arm.startswith("m")) if rel == HDR else files[rel])


if __name__ == "__main__":
    main()
