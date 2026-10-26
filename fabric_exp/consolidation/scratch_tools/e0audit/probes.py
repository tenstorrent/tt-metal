#!/usr/bin/env python3
"""probes.py BASE_VARIANT OUT_DIR: two edits a contributor might make to ERISC 0's once-per-round or rare code, written
the same way for any v3 variant (anchors in the router .cpp and the sender header must match):
  ctxsw  a debug word written each time the idle context switch fires (the rare path of the per-round work)
  term   the round test also stops on a second termination value (continue_running_main_run_loop, per round)
Neither changes per-pass work. Used to check whether such edits reach sender_erisc::run_main_loop's code."""
import os, re, shutil, sys
base, out = sys.argv[1], sys.argv[2]
TX = "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_sender_erisc.hpp"
RT = "fabric_erisc_router_v3.cpp"
def edit(name, f, fn):
    d = os.path.join(out, name)
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(base, d)
    p = os.path.join(d, f)
    s = open(p).read()
    s2 = fn(s)
    assert s2 != s
    open(p, "w").write(s2)
def ctxsw(s):
    a = "run_coordinated_context_switch_to_base_firmware(termination_signal_ptr);"
    i = s.index(a)  # the first (IDLE_CONTEXT_SWITCHING) site
    ls = s.rindex("\n", 0, i) + 1
    ind = s[ls:i]
    return s[:ls] + ind + "*reinterpret_cast<volatile uint32_t*>(FABRIC_KERNEL_HEARTBEAT_ADDR + 4) = 0xC5C50000u | did_nothing_count;\n" + s[ls:]
def term(s):
    a = """            return !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr) &&
                   !state_manager_l1->is_non_run_command_pending();
        }
    } else {"""
    b = """            return !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr) &&
                   *termination_signal_ptr != tt::tt_fabric::TerminationSignal::GRACEFULLY_TERMINATE &&
                   !state_manager_l1->is_non_run_command_pending();
        }
    } else {"""
    assert s.count(a) == 1
    return s.replace(a, b)
os.makedirs(out, exist_ok=True)
edit("ctxsw", TX, ctxsw)
edit("term", RT, term)
print("ctxsw term")
