#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# run.sh: the fleet hardware test of one Blackhole Galaxy (32 chips, 4 trays). Plain run.sh (no option) is
# --mode=quick: every fast fault check, resets on, under 30 min; see "run.sh [--mode=..." below.
# --mode=repro is the old tray-drop repro: 4 tries at 1 min of quiet (run-mid.sh: 4 at 1 min, then 4 at 3 min), the
# all-gather tries, the end reset; --tries or --ag-tries without --mode also mean --mode=repro. Each try runs ub0_repro.py:
#   - first, any chip above 800 MHz gets AICLK_GO_LONG_IDLE (ARC 0x54), so the clock drops to idle
#   - the try's quiet Q is split in half: wait until every chip has been at <= 800 MHz for Q/2 in a
#     row (cold quiet), bring up the 4x8 ring fabric and upload (~20 s), idle Q/2 with the device open and no compute,
#     then a 30 s bf16 matmul burst on all 32 chips at stock clock
#   - every quiet (the bus-bar stage's quiet-bursts, chunks, step and long chunk) starts with the clock check
#     (quiet_clock.py): every chip above 800 MHz is asked for its idle clock and the quiet counts
#     only once all read <= 800 MHz, at most 5 min; a chip still above then is a QUIET FAIL finding by BDF and tray 1-4
#     and the run goes on. After the open idle every chip is asked for its busy clock again, logged at the load start
#   - at the end, AICLK_GO_LONG_IDLE again to every chip
# Phase 2, only if every phase-1 try was clean: 4 more tries at 1 min of quiet, the same go-idle, quiet
# and checks, with the all-gather load (ub0_repro.py --load allgather): the 30 s burst is ttnn.all_gather on the 4x8
# ring fabric instead of matmuls, both mesh axes and the full 32-way gather, 2 GiB per chip per gather. A gather that
# hangs with every chip still on PCIe ends the try (its own watchdog; nothing is left running) and the run, and prints
#   CCL hang: chip unknown (32/32 still on PCIe), op <variant, mesh axis> ... - not a tray drop
# A try whose log has not grown for TRAY_REPRO_HANG_S (the hang timeout, default 600 s; an announced load chunk's
# chunk_s is added while that chunk runs) is hung, whatever ub0_repro.py's own watchdog says (one host: 98 min
# at "traffic: fabric init done", the process deaf to SIGINT/SIGTERM): the run names it, dumps the stacks and the hang
# capture while it still hangs, stops the try and ends the tries, then resets as after any fault (C9):
#   HUNG: try hang: no progress for N s (hang timeout TRAY_REPRO_HANG_S=600 s), last step ...
# No run-time budget ever cuts or skips a stage: a stage that keeps progressing runs on.
# A try at 1 min of quiet takes about 110 s; 4 tries and the reset about 8 min. After each try it prints every chip's
# clock after the quiet, after the mesh open and right before the burst, checks which chips still answer on PCIe and
# prints one line:
#   matmul try 1/4  waits: 30 s cold quiet, then 30 s open idle
#   matmul try 1/4  AICLK after the quiet: 800 MHz chips 0-31 | after mesh open: 1350 MHz chips 0-31 | before burst: ...
#   matmul try 3/4  quiet 1 min  chips lost: 0  (110 s)
#   all-gather try 1/4  quiet 1 min  chips lost: 0  (130 s)
# Numeric check (numeric_check.py): during each matmul burst every chip's output is compared bit for bit
# with the majority output (matmul 1, every 500th, the last; TRAY_NUMERIC=0 off). A mismatch never stops the load; before
# the reset the run prints, over every matmul try (numeric_report.py):
#   NUMERIC FAIL <BDF> (tray N) chip N: x/y mismatched, first at iter K in <try>, max abs err E, NaN count M
#   NUMERIC PASS tray N (8 chips, C checks)
# The first loss ends the tries; the RESULT line names the phase and try. Then, loss or not, it resets the chips and re-checks all 32 (the
# rk_* functions below): a glx_reset (tt-smi -glx_reset; never tt-smi -r). It ends with three lines:
#   Reset: glx_reset; ...                     which reset ran and what came back
#   Chips dropped: 24-31 | None                lost during the test, whether or not a reset brought them back
#   Trays dropped: tray 4 (chips 24-31) | None   (glx_reset gate-skipped, glx_reset retry ran: tray 4: chip 25 (0000:82:00.0))
#                                              still missing after every reset that could run (tray 1-4):
#                                              only a power cycle brings them back. Chips back after either reset
#                                              are a chip drop only. A reset that failed or timed out still ran:
#                                              chips missing after it are a tray drop. unknown only if the re-check
#                                              could not run.
# If chips are missing before the first try, the same reset and re-check run first; the tries start
# only if every chip is back. Else it names what is still missing and stops with nothing loaded, without the three
# lines, ending with: Couldn't start - Chips are already dropped
# Root isn't needed: under sudo the load and the glx_reset run as the calling user.
#   run.sh --repro link-a [<bdf>] [--dry-run] [--heavy-logs] [--hold]   (opt-in, not in plain run.sh or --mode=full)
#     The link-A fast repro (H3 bug a): the usual gates (P0, C0, C1, C3) and C2 start reset, then of the two
#     in-box links between 02:00.0 and 06:00.0 (tray 1) it hides e7 <-> e1 (chip L1 only; a glx_reset (C9) restores it),
#     checks that discovery sees only e6 <-> e0, and runs a 2-chip all-gather of 2560x4096 bf16 per chip over it for
#     10 s under a 30 s beat watchdog (launch 0: 60 s); then the C9 end reset. <bdf> replaces chip A (02:00.0); env
#     FLEET_LINKA_B, _KEEP ("6,0"), _DROP ("7,1"), _HW, _S, _HANG_S, _FIRST_S change the rest. One result line:
#       REPRO LINK-A HANG 02:00.0 (tray 1) e6 <-> 06:00.0 (tray 1) e0: time to hang <s> s (...); recovery: a glx_reset (C9, tt-smi -glx_reset) ...
#       REPRO LINK-A PASS ... no hang in <s> s     or     REPRO LINK-A NOT RUN (INCOMPLETE) ...: <why> (rc 9)
#     A HANG is a FAIL of that tray (rc 8). On the host where H3 bug a was seen, it hung on launch 0 in 26 of 26 tries, about 1 min each; the
#     clean control host passed 8 of 8. Only this host's own chips and in-box link: nothing reaches another host
#   run.sh --repro abort-fabric [--dry-run] [--heavy-logs] [--hold]   (opt-in, not in plain run.sh or --mode=full)
#     A job killed while it holds the chips at busy clock, then a fabric job at once (a test box: 7 of 78 such passes dropped
#     chips, 0 of 101 other passes). After the gates (P0, C0, C1, C3) and C2, each of 8 cycles runs 60 s of the bus-bar
#     stage's CCL mix + matmul load, SIGKILLs only that load's own worker process at busy clock, starts 100 s of 4x8 ring
#     all_gather on both mesh axes within 3 s (no reset, no go-idle between; this host's own chips and in-box links), then
#     checks all chips on PCIe by BDF and tray 1-4 and prints each root port's corrected AER count; then the C9 end reset.
#     Env FLEET_ABORTFAB_CYCLES (8), _LOAD_S (60), _AG_S (90-120), _GAP_S (1-3), _HANG_S (270). Result lines:
#       ABORT-FABRIC FAIL 02:00.0 (tray 1): off PCIe in abort-fabric cycle <i>/8, <s> s after the load worker was SIGKILLed ...
#       REPRO ABORT-FABRIC PASS: 8/8 cycles, all 32 chips on PCIe after each, ...   or   ... NOT RUN (INCOMPLETE): <why> (rc 9)
#   run.sh --mode=repro [--tries "1 1 1 1"] [--ag-tries "1 1 1 1"] [--hwmon-guard] [--heavy-logs] [--hold [--hold-timeout=<min>]]
#     --tries     phase 1 (matmul): quiet minutes per try (half before the mesh opens, half after), one whole number
#                 >= 1 per try; the count is the number of tries (--tries "1 1" = 2 tries at 1 min); none = no phase 1
#     --ag-tries  phase 2 (all-gather), the same form (default "1 1 1 1"); none = no phase 2
#     --hwmon-guard chmod 000 a hung chip's hwmon *_input/*_max (modes saved first; undo
#                 heavy/hwmon_guard_undo.sh), so pollers stop sending it ARC messages. ONLY on a host that serves
#                 nothing (no wan pod on the node, no H3 server holding /dev/tenstorrent; check first, read-only): it
#                 refuses while another user's process holds /dev/tenstorrent. It prints the `ttp machines change add`
#                 line to record. The reset and re-check still run
#     --heavy-logs  also the slow BMC reads in every snapshot, the SDR poll and the per-try CPLD/SEL baseline (below);
#                 with --mode too. Off by default: they add 2-4 min to a healthy run and a fault gets them anyway
#     --hold      (any mode; off by default; lib/hold.sh) on a hang with every chip still on PCIe (a link/CCL
#                 hang, an all-gather stall, a chip hang): after the normal capture, keep the hung job alive, do not
#                 reset, print one HOLD banner (host, chips by BDF and tray 1-4, PIDs, tt-triage/tt-exalens commands, STOP
#                 file, deadline) and wait until `touch <run dir>/STOP` or --hold-timeout=<min> (default 240) passes;
#                 then reset and finish as without --hold. Once per run. Drops, numeric faults and chips lost at a
#                 reset are never held. Refused under Slurm (printed; the run goes on). Under a tt-device broker the
#                 hold ends as soon as another broker job queues (or the broker cannot be read)
# Every run collects the fast evidence (lib/heavy_logs.sh): versions and firmware, light snapshots
# before the tries, before the reset and after it (holders, hung chips, hwmon modes, AER + link, lspci -vvv, kernel
# log delta, MCE, tt-smi, ETH counters; about 6 s each), a raw ipmitool power + CPLD vr_pg poll every 2 s during each
# try (tray 1-4 W, HSC W, rails good), besides each try's result JSON, telemetry and 1 s link poll; all under <run
# dir>/heavy, listed in heavy/index.md. --heavy-logs adds the slow, redundant BMC reads to every snapshot (SEL, SDR,
# CPLD dump + 0x5f, sysfs telemetry x2, Redfish power/thermal; about 40 s a snapshot), the SDR poll instead of the
# raw one and the CPLD/SEL baseline before each try (14-31 s a try); README "Evidence logging" says why each is off by
# default. The BMC heat poll (lib/bmc_heat.sh: SDR inlet/outlet, PSU temps and W, tray W, TMAX ASIC and
# retimer, fans, P12V every 2 s to <run dir>/bmc_heat_poll.csv, by tray 1-4) runs the whole run with --heavy-logs
# (any mode; not --mode=full by itself); every drop capture, in any mode, takes one BMC snapshot and each tray CPLD's
# alive (0x1E) and vr_pg (0x19); every mode logs each chip's temp_c (fleet/chip_temp.csv). Sensor reads only. After any fault every snapshot is the full one, with or without --heavy-logs. Read-only; the test flow and
# exit codes do not change. Each snapshot names
# hung chips (lspci rev ff, or 'Timeout waiting for ARC response' in the kernel log since the last snapshot) in
# hung_chips.txt, never reads their hwmon files, and logs every chip's hwmon *_input/*_max mode (hwmon_modes.txt).
# Any fault (chips missing at the start, a chip or tray drop, a CCL hang, a failed try) is also pinpointed
# (lib/fault_capture.sh): fault.txt names each faulting chip, PCI address, root port and tray; a CCL
# hang or a loss gets the NOC snapshot (tt-triage, exalens, fast-dispatch dump) before the reset, and a
# CPLD/SEL/SDR/AER/lspci/dmesg snapshot before and after the reset.
#   run.sh --repro numeric <bdf> [--way heavy|dram|dram-mix|busbar|all] [--tries N] [--known <bdf>] [--dry-run]   fast repro of a chip's numeric fault;
#   run.sh --repro tray-drop <bdf> [--shape h3-replay|h3-cadence|imbalance|weights|cold-full|all] [--tries N] [--dry-run]
#                 the tray-drop repro: H3-shaped bus-bar loads (fleet/drop_shapes.py) on all 4 trays aimed
#                 at the bdf's tray; a POWER DROP or CHIP DROP of that tray is the hit (minutes to it, tray, its BDFs);
#   run.sh --repro warm-reset <bdf> [--tries N] [--dry-run]   the warm-reset loss repro (repro.sh): each try
#                 is one plain run.sh cut to that fault's stages; it stops at the first hit (FAIL line naming the chip
#                 by BDF and tray, the exact finding under it, time to failure, exit 8) or prints PASS after N tries.
#                 --repro comes last: everything after it goes to repro.sh (repro.sh --help). No --way: the fastest
#                 recorded way for that bdf. --known <bdf>[,<bdf>]: a try with only those chips' known faults is a
#                 miss, the loop goes on. Options before --repro (--known-dropped-tray, --since, --hwmon-guard,
#                 --hold, --hold-timeout, --heavy-logs, --dry-run) go to every try.
#   run.sh [--mode=quick|full] [--dry-run]   the fleet hardware test (lib/fleet.sh);
#                 no --mode is quick. Gates, C0 preflight, C1 AER/dmesg/SEL deltas + journal/SEL/SDR history, C3 ETH/QSFP counts + ETH ports,
#                 C2 start glx_reset (tt-smi -glx_reset) and health recount, then the bus-bar stage first in every mode (set by
#                 FLEET_BUSBAR_ORDER): two 60 s clock-checked quiets each
#                 followed by a 30 s lofi_all matmul burst on all 32 chips (qb), then fleet/chunk_load.py chunks of the H3 CCL
#                 mix (FSDP and TP all-gather, reduce-scatter on the ETH links) with mm, SFPU, DRAM and host PCIe on all 32
#                 chips, no heavy part, one command queue, in 1, 2, 3, 4 min chunks, each with an on-device numeric check
#                 read back at its end and 10 s of clock-checked quiet after it (FLEET_BUSBAR_CHUNK_MIN, FLEET_GAP_S);
#                 after chunk 1, before its quiet, G3 (C4d's slot): 15 s of the x/w/y-in-DRAM C4n-size matmul read back and
#                 compared across chips (FLEET_BUSBAR_RW_AFTER, 0 = off, FLEET_BUSBAR_RW_S), and right after it
#                 G3b, 10 s (60 s in --mode=full) of DRAM byte/DBI patterns written, copied on the device and read back bit for bit on every chip
#                 (dram_dbi.py, R03; FLEET_BUSBAR_DBI_WALL_S, 0 = off), and right after it N1, 10 s (60 s in
#                 --mode=full, with the inverse patterns) of L1 march: one shard per Tensix core written, copied L1 to L1 to the
#                 next core and read back, 'L1 FAIL <BDF> tray N core x,y pattern P words k' (l1_march.py;
#                 FLEET_BUSBAR_L1_WALL_S, 0 = off); then
#                 the cold-quiet-to-full-load step (a 10 s quiet + 90 s lofi_all burst); then one 90 s
#                 heavy chunk, stage 5's DIDT numeric chunk (the DIDT bf16 HiFi2 16k matmul then the bf8 LoFi 16k
#                 matmul, back to back, each output compared bit-exactly per launch and across chips; FLEET_BUSBAR_HEAVY_S,
#                 FLEET_DIDT_OPS). A 2 s BMC tray power and
#                 vr_pg poll and C8 telemetry WARNs run alongside. --mode=full runs these same steps in the same order,
#                 only its last CCL-mix chunk is 25 min (1, 2, 3, 25). Then C5 the pairwise link all-gather, C9 the end reset and re-check (a per-link sweep after
#                 it only on a hang or CCL failure); then
#                 findings and one "PASS tray N" / "FAIL tray N (k findings)" line per tray before the three verdict lines,
#                 each finding under its tray as
#                 "    <chip|link|whole tray> | <stage> | <evidence> | cause: ... -> next: ..." (fleet/tray_why.py).
#                 Quick about 30 min (29.7 min budget), full about 62 (the same steps, last chunk 25 min, 62 min budget); --mode=sustained is retired;
#                 --dry-run prints the stage table (stage N name, fleet/stages.tsv) and exits.
#   run.sh --mode=long [--dry-run] [options]: 3 plain run.sh, then 2 --mode=full,
#                 back to back (lib/long.sh), each its own whole run into <out>/leg<i>-<mode> with its own log; stops early only on
#                 exit 6 (box not healthy for the next leg), exit 10 (host busy) or a signal. Ends with one summary: per leg mode,
#                 verdict, duration and dir; faults per tray and BDF with k/5; each leg's HSC lines; last, one combined
#                 'Repair recommendation:' line. Not with --repro, --tries, --ag-tries or --hold. TRAY_RUN_OUT sets the results dir.
#   Every run's last log line (any mode, early exits too) is 'Repair recommendation: ...' (fleet/repair_rec.py).
#   run.sh --stub [--mode=...]   a fake healthy 32-chip host (lib/stubhost.sh): no device, sudo or network; proves the code
#                 paths only (the offline bundle CI check), never hardware. WARNING first and in the summary, VERDICT says STUB.
#   run.sh --mode=smoke [--dry-run]: plain run.sh's steps in the same order through the
#                 same code (FLEET stays quick), only every load chunk, burst, heavy chunk, G3 readback, G3b cap, pair round and quiet
#                 is FLEET_SMOKE_S s (default 2; fl_smoke in lib/fleet.sh); resets, clock checks (5 min cap), the 50 ms
#                 burst gap, the run budget and every timeout are plain's. For device checks during development.
#   run.sh --mode=... --since <time>   (or --history-since, or env FLEET_HISTORY_SINCE; any `date -d` time, e.g.
#                 2026-10-22T20:32-07:00): the host's last repair (bus bar or tray swap, reseat). C1's history (SEL,
#                 kernel journal AER/MCE, all from before the run and WARN only) then counts only events at or after it;
#                 earlier ones print as one "C1 INFO tray N: M events before repair <time>, ignored (...)" line per tray
#                 (chip BDF named), never a finding. A bad or future time exits 1. Unset: the full 7 days.
#                 Resets (C2, C9) run by default whatever the QSFP peer hosts (far hosts) are doing, multi-box clusters included
#                 (resets do not hurt neighbours); it takes no peer snapshot around a reset.
#                 No variable or flag makes a peer a gate. Only this host's own serving gate
#                 (TRAY_REPRO_RESET_GATE: this host serves live H3, or another person's process or run holds its chips)
#                 and the user's hold file (TRAYTEST_HOLD_FILE, lib/holdfile.sh; no automatic check of
#                 another host, of other hosts' serving, of peers, of the site or of etcd, and no hardcoded host list)
#                 can skip a reset: SKIPPED-GATE, the run ends INCOMPLETE (exit 9). A hold on this host at the start: NOT RUN, exit 10.
#                 Links that leave the box (QSFP ports 0,1 of asic 4 on boxes cabled to other boxes) are out of scope.
# FLEET_GATE_DEVICE_ONLY=1  opt-in: the host gate counts only processes holding /dev/tenstorrent/*, not test/H3 names
#                 (for hosts where a device broker serializes the device). Never set on quad hosts.
#                 fleet/gate.sh (this host idle) runs before every phase; busy = no more load, no reset, exit 10.
# Exit: 0 no loss, 5 chip drop only, 3 tray drop, 7 a try CCL hang or try hang (whether or not the reset cleared it; no chip lost), or a fleet pair/stage left hung because no reset could run, 1 unknown (a try failed or the re-check
#       could not run, or chips lost with no reset allowed to run: RESULT "loss: ..." by BDF and tray), 6 couldn't start (chips still missing at the start after every reset, or the re-check could
#       not run; nothing was loaded), 8 (fleet) no loss but a fleet check failed (including a C5 link hang that a reset cleared), 10 (fleet) the host became busy,
#       14 a glx_reset (start, end; fleet C2, C9, C9b) left a chip missing or failed on one, though every chip
#       answers at the end (the glx_reset brought it back): one "RESET FAIL <bdf> (tray N) chip C" line per chip
#       with the count after the glx_reset and after the glx_reset rescan; 3, 5, 7, 1 and 6 outrank it,
#       it outranks 8 and 9,
#       9 INCOMPLETE (a gate-skipped tray drop is rc 3, "tray 4: chip 25 (0000:82:00.0)"): a reset was skipped by a gate (TRAY_REPRO_RESET_GATE), or the glx_reset by TRAY_REPRO_NO_GLX=1, so the run is
#       (external ETH ports to other boxes, asic 4 ports 0, 1, are out of scope and no longer return 9)
#       neither a pass nor a tray drop: no loss seen but recovery was not exercised, or chips still missing that no
#       reset could prove lost (the Reset: line names the skipped reset and the gate's reason).
#       fleet (no --mode, quick, full): a hang no reset could clear is 7 and names the pair left hung
#       (marker FLEET_HUNG_FILE); the next
#       run on the same boot exits 6 unless its C2 reset runs first.
#       12 KNOWN-DROP (--known-dropped-tray=N): 1-8 of tray N's chips were still missing after the start reset, as
#       declared, and the other 24 chips showed no fault; never a PASS (3, 5, 7, 1, 14, 8 and 9 outrank it).
# --tdp=<W> or --tdp <W> (any mode; off by default; lib/tdp.sh): every tt-metal process of the run
#       gets TT_METAL_TDP_LIMIT_WATTS=<W>, which tt-metal (PR 52394 or newer) sets as each chip's firmware TDP limit at
#       open (firmware 19.11.0 or newer). No range check: the operator picks a sane value. The setup lines and the line
#       before RESULT print the requested value and each chip's limit read back by BDF and tray 1-4. The default comes
#       back on every exit of a run that applied it (the end reset, else one short open with TT_METAL_TDP_LIMIT_WATTS=0);
#       a marker lets the next run.sh start restore a limit a killed run left. Nothing is set, restored or opened before
#       this run's own start check passed: in fleet modes the own-host busy gate (fl_start's P0), in --mode=repro the
#       start chip check (rk_start; its start reset runs only past the hold file and TRAY_REPRO_RESET_GATE), nor after
#       either said busy or held: "TDP: not applied by this run, restore skipped". An older
#       tt-metal or firmware: "TDP WARN: --tdp had no effect"
# --known-dropped-tray=N (or TRAY_TEST_KNOWN_DROPPED=N; N 1-4, fleet modes only): a host with one tray known to be down
#       or flapping (powered off, awaiting repair). Tray N fully present: "tray N present", the normal run on all 32.
#       Tray N short (1-8 chips): the start reset and recount run first (where the gates allow; else none). Tray N back:
#       the normal run on all 32. Only tray N still short and the other 24 answering: "KNOWN DROP tray N: k of 8 chips
#       missing ...: <each missing BDF>", tray N off the map, P0, C0, C1, C3, C2, C5, C8 and C9 (resets included) on the
#       other 24 chips. The bus-bar stage needs the 4x8 mesh: "Not run". The per-tray lines say "KNOWN-DROP tray N"
#       for it and PASS/FAIL for the others. Other chips still missing after the start reset: exit 6, as without the flag.
# --kmd-power-policy=<0|1> (opt-in, every mode the same way; lib/kmd_policy.sh): every run's
#       header prints "power_policy: <v>" (the loaded KMD's, read only). With the option, before C2: sudo -n must allow
#       modprobe and no other process may hold /dev/tenstorrent (else refused, nothing changed, exit 11; a holder is never
#       killed or signalled); every chip's power1_max/power1_cap is read, the module is reloaded with power_policy=<v>, the
#       chips are recounted by BDF and tray (a chip missing then gets the start reset, glx_reset with its rescan) and the
#       limits re-read: one that moved is "FINDING KMD POWER LIMIT <bdf> (tray N)", the default driver comes back and the
#       run ends (exit 11). The run's EXIT trap (success, failure, timeout, Ctrl-C) reloads the module with no parameter
#       and logs the power_policy it then reads. No power limit is ever written.
set -u
# The end-of-run reset and re-check (
# "the reset tells us whether the whole tray will drop or recover ... If reset recovers, it's a chip drop. If it
# doesn't, it's a tray drop."). Every run ends with a reset, loss or not, then a re-check of all 32 chips (PCIe config
# space, the tt-kmd node, tt-umd enumeration), then the verdict, always the last two lines:
#   ...the caller's RESULT line...
#   Reset: <which reset ran and what came back>
#   Chips dropped: <chips lost during the test, whether or not the reset brought them back, e.g. 24-31> | None
#   Trays dropped: <trays with chips still missing after every reset that could run, e.g. tray 4 (chips
#                   24-31): only a power cycle brings those back> | None
#   (gate-skip tray drop, tray numbers only: tray 4: chip 25 (0000:82:00.0); several joined with "; ")
# Trays from bus & 0xF0: 00 tray 1, 40 tray 2, c0 tray 3, 80 tray 4; several are joined
# with "; ". Only a re-check that cannot run (or be trusted), or a loss with no reset allowed to run, gives
# "Trays dropped: unknown (...)", never a guess; the no-reset form names each chip still missing by BDF and tray.
# The reset is chosen by itself ("make the reset choice automatic an remove the setting"):
#   0. every reset step (start, C2, C9, C9s, C9b; 'must always try
#      glx_reset' and 'never use -r'): tt-smi -glx_reset (it runs ipmitool raw 0x30 0x8b, the
#      tray power sequence), then the root-port rescan and the recount by BDF and tray. It runs as root, with sudo ALL,
#      as `sudo -n tt-smi -glx_reset` where sudo -n -l lists tt-smi, or as plain tt-smi where sudo -n -l lists the
#      ipmitool raw 0x30 0x8B forms (rk_can_glx; never sudo -n true). tt-smi -r (tt-umd WarmReset) is never run: where
#      glx_reset cannot run (none of those, no tt-smi, IPMI not answering, --hold), no reset runs and a "NO RESET" line
#      says why. The in-band re-power of only some trays (rk_traypwr) never runs before a glx_reset was tried in the step.
#      TRAY_REPRO_NO_GLX=1 skips glx_reset in every caller: one line "tray re-power (tt-smi -glx_reset) skipped by
#      TRAY_REPRO_NO_GLX=1" names each chip still missing by BDF and tray 1-4; those are INCOMPLETE (exit 9).
#   After each (with root): a chip still off the bus gets its root port rescanned once the link is up (rk_rescan,
#   rescan); a chip back only after that is a finding ('reset needed root-port rescan', BDF and tray), never a pass.
#   "a tray drop is only one that's not recoverable by any available reset, ie one that only a power
#   cycle can resolve." Chips back after either reset = chip drop (Trays dropped: None); still missing after every
#   reset that could run = tray drop. A reset that failed or timed out ran too: chips still missing after it are a
#   tray drop, chips back a chip drop (seen on a cluster node); the "Reset:" line says it failed and its rc.
#   A line "Reset: ..." just before the two verdict lines says which reset ran and what came back.
# The caller sets PY (python with tt_umd), AS (the command prefix the device work runs under, may be empty), H and say().
# `. ./run.sh` (the tests) defines these functions and returns before run.sh's own test.
SYS=${TRAY_REPRO_SYS:-/sys}
RK_SETTLE_S=${TRAY_REPRO_RESET_SETTLE_S:-60}   # after the reset, wait up to this long for every chip to answer
RK_TIMEOUT_S=${TRAY_REPRO_RESET_TIMEOUT_S:-300}
RK_LOST_CAP_S=${TRAY_REPRO_LOST_CAP_S:-90}   # the best-effort glx_reset of chips already off the bus gets at most this long
# One whole reset step (C2, C9, C9s: glx_reset, settle, glx_reset retry rounds, rescans) gets at most RK_STEP_S.
# Worst case on record: 414 s (one host's C2: warm 54 s, retry 58 s, rechecks and rescans,
# two glx_reset retry rounds); 600 s leaves margin. Past it rk_bound_hit stops the step, prints k/N and an INCOMPLETE RESULT.
RK_STEP_S=${TRAY_REPRO_RESET_STEP_S:-600}
# The run's own time budget: 4 legs on 10-26 died before or in C9 with no RESULT when the leg driver killed
# them (work/task504/leg.sh, rc 124). Plain run.sh is capped at 1800 s (under 30 min per host).
# Stages start only while RS_C9_S + RS_TAIL_S (+ RS_SWEEP_S below, rs_keep) stay free for C9, the sweep and the RESULT; the C9 step's bound shrinks to what is
# left less RS_TAIL_S (at least 60 s). 0 = no budget (default outside plain/quick, and with --hold).
# Timing (35 healthy plain passes on record): C9's reset and recheck take 44-47 s, fl_c3 c9 31-50 s
# and the RESULT 10-18 s; the last stage gate (host link check post) comes at 351-385 s left of 1800. RS_C9_S 180 +
# RS_TAIL_S 120 = 300 s kept lets every median and p90 pass start every stage; C2 keeps the whole RK_STEP_S (600 s).
# RS_SWEEP_S: the after-fault per-link PINPOINT sweep's share, kept free with C9's once a fault sets
# FL_PIN (rs_sweep_keep; a healthy run keeps none, so its stages start as before): it took 148-205 s on one host
# (10-23..10-25), so 240 s. One run (10-26) lost its sweep with 191 s left after C9. rs_keep is what stages leave free.
RS_SWEEP_S=${TRAY_REPRO_SWEEP_S:-240}
# RS_C5_S: stage 6 pair link test's share, kept free by stage 4's parts while stage 6 is still to run (fl_keep):
# all pairs took 90-93 s on main, plus ~10 s for the PSU read and gate between stages and ~10 s for a heavy chunk running past
# its plan (50 s planned around the chunk, 60 s measured). With C2 running (146 s) stage 4 ran into it and stage 6 got
# 54 s (one host: "LINKS WARN C5 budget spent", 32 pairs not run).
RS_C5_S=${TRAY_REPRO_C5_S:-120}
# One run (one host's plain pass): C9 itself is never cut for the budget (rk_bound): a run that
# comes to C9 late runs past RS_RUN_S, and rk_end logs the overrun in s. RS_C9_S stays 180 (300 s kept): 210 cut the 90 s
# heavy chunk of p90 and slowest healthy passes once stage 6's RS_C5_S is kept too (tests/test_c5_keep.sh); C9 behind the
# leg's reset gate took about 290 s there, so the overrun stays within the leg's 60 s margin.
RS_C9_S=${TRAY_REPRO_C9_S:-180}; RS_TAIL_S=${TRAY_REPRO_TAIL_S:-120}; RS_RUN_S=${TRAY_REPRO_RUN_S:-}; RS_BUDGET_INC=""; RS_BUDGET_WHY=""; HL_DEADLINE=""
RK_RESETS=""; RK_RESET_LBL="glx_reset (tt-smi -glx_reset)"; RK_GLX_TRIED=""   # RK_GLX_TRIED: a glx_reset was tried in this reset step
HB_S=${TRAY_REPRO_HEARTBEAT_S:-20}; HB_PID=""
hb_start() {  # $1 what runs, $2 its log (optional; its last STRESS line is shown): one line every HB_S s until hb_stop.
              # A job runner reaps a job that prints nothing for about a minute as hung (the device broker: 'no output
              # for 72s; reaping as hung', two test jobs), and the tries and the reset write to their own logs.
              # The loop also ends by itself within 1 s once this shell is gone.
  hb_stop; local main=$$ what=$1 log=${2:-}; RS_STAGE=$1
  [ -n "$log" ] && { RS_PROG_T=$(date +%s); RS_PROG_STAGE=$1; RS_PROG_LOG=$log; }   # a load stage's start is a progress mark (rs_progressing)
  ( trap - EXIT; sp=""; p0=$!; trap 'kill $sp 2>/dev/null; [ "$!" = "$p0" ] || kill "$!" 2>/dev/null; exit 0' TERM; s0=$(date +%s); k=0   # hb_stop's TERM ends its sleep too ($!: one forked just before sp= is set; p0: not the caller's)
    while kill -0 "$main" 2>/dev/null; do
      sleep 1 > /dev/null 2>&1 & sp=$!; wait "$sp" || exit 0; k=$((k + 1)); [ "$k" -ge "$HB_S" ] || continue; k=0
      last=""; [ -n "$log" ] && last=$(grep -a '^STRESS|' "$log" 2>/dev/null | tail -1 | cut -d'|' -f3- | cut -c1-120)
      say "$what: running, $(( $(date +%s) - s0 )) s${last:+: $last}"
    done ) &
  HB_PID=$!
}
# rs_stop_pid: a TERM that lands in a fresh subshell before it sets its own TERM trap is lost (bash 5.1 clears
# the parent's pending trap in the child), so one kill left the heartbeat looping and hb_stop's wait hung until the caller's
# timeout (test_stub_mode 'unwritable run parent' rc 124, 1 in 10 loaded runs; every run of a 400 x start/stop loop).
# Resend TERM until the child has exited (a zombie counts; no /proc: one TERM, as before), KILL it after 5 s, then reap it.
rs_stop_pid() { local p=$1 i=0 l; while kill "$p" 2> /dev/null; do { read -r l < "/proc/$p/stat"; } 2> /dev/null || break
    l=${l##*) }; [ "${l%% *}" = Z ] && break; [ $i -ge 100 ] && { kill -KILL "$p" 2> /dev/null; break; }; i=$((i + 1)); sleep 0.05; done
  wait "$p" 2> /dev/null; }
hb_stop() { [ -n "$HB_PID" ] && rs_stop_pid "$HB_PID"; HB_PID=""; }
VD_INC_WHY=""   # why the run is INCOMPLETE (rc 9), set by fl_report; vd_exit prints it
# rs_stopwatch: TRAY_STOP_FILE set (reliability.sh's dev_run, a run.sh step in our broker job), a loop TERMs
# this shell once that file exists, so run.sh takes its own TERM path (rs_dying: the device stages close their devices,
# RESULT: INCOMPLETE, the bounded reap) and the broker job ends itself, before the broker's own TERM and its KILL 10 s
# later. Read-only, one stat a second; it ends by itself within 1 s once this shell is gone, and at the EXIT trap.
# bash runs this shell's TERM trap only once a foreground step returns (a pipeline, a $(...)), so the loop
# also TERMs the current stage's recorded groups (TRAY_PGID_FILE), softly as rs_dying's first reap (pg_reap PG_SOFT: no
# KILL): the stage closes its device and ends, and the stop lands within seconds, not at reliability.sh's STOP_JOB_GRACE.
# It is started again once TRAY_PGID_FILE exists (after rs_traps below), so the loop knows the files.
RS_SW_PID=""
rs_stopwatch() {
  [ -n "${TRAY_STOP_FILE:-}" ] || return 0
  ( trap - EXIT; main=$$; while kill -0 "$main" 2>/dev/null; do
      if [ -e "$TRAY_STOP_FILE" ]; then echo "stop file $TRAY_STOP_FILE: stopping the run (TERM to run.sh $main and its stage's process groups)"; kill -TERM "$main" 2>/dev/null
        [ -n "${TRAY_PGID_FILE:-}" ] && declare -F pg_reap > /dev/null && PG_SOFT=1 PG_DEADLINE_US=$(( ${EPOCHREALTIME/[.,]/} + RS_DIE_S * 1000000 )) pg_reap
        exit 0; fi
      sleep 1; done ) < /dev/null {PG_OUT}>&- &
  RS_SW_PID=$!; }
rs_sw_stop() { local r=$?; [ -n "$RS_SW_PID" ] && kill "$RS_SW_PID" 2>/dev/null; RS_SW_PID=""; return $r; }   # the EXIT trap calls it right after vd_exit (test_verdict_exit.sh extracts up to '^trap .RS_XRC=.*vd_exit;'; tests also find '^rs_traps$')
# vd_say/vd_exit: exactly one 'VERDICT: ...' line on every exit path, so a caller tells 'not run' from a crash.
# vd_say prints it (the first call wins); the EXIT trap calls vd_exit first, which derives it from the exit code if none was
# printed. A rc 0 with no RESULT line (--help) prints none.
VD_DONE=""; VD_TEXT=""; RR_RC=0
vd_say() { [ -n "$VD_DONE" ] && return 0; VD_DONE=1; VD_TEXT=$*; echo "VERDICT: ${TRAY_STUB:+STUB (fake devices, not a real test) }$*"; }   # --stub: never a bare PASS
vd_exit() {
  local rc=$?; RR_RC=$rc; [ -n "$VD_DONE" ] && return 0
  case $rc in
    0) [ -n "$RS_DONE" ] && vd_say PASS ;;
    12) vd_say "KNOWN-DROP" ;;
    9) vd_say "INCOMPLETE (exit $rc${VD_INC_WHY:+: $VD_INC_WHY})" ;;   # the reason, when fl_report set one
    129|130|143) vd_say "INCOMPLETE (exit $rc)" ;;
    10) vd_say "NOT RUN (host busy)" ;;
    11) vd_say "NOT RUN (--kmd-power-policy refused or aborted)" ;;
    6) vd_say "NOT RUN (couldn't start: chips already dropped)" ;;
    1) vd_say "FAIL (exit 1: a try failed, the re-check could not run, or a bad option)" ;;
    *) vd_say "FAIL (exit $rc)" ;;
  esac
  return 0
}
# rr_final: the run's last line is always 'Repair recommendation: ...', after the
# VERDICT line and every exit step: fleet/repair_rec.py maps each failing tray's root-cause line (tray_why.py) to
# the part to repair, by tray 1-4 and chip BDF. A run with no VERDICT line (--help, --dry-run) prints none. It also appends
# one row per numeric fault to $OUT/rootcause.tsv (kind, part, cause, confidence, evidence, ruled_out, fix, next_step;
# the change above), every mode.
RR_DIR=$(dirname "$(readlink -f "$0")")
rr_final() {
  [ -n "$VD_DONE" ] || return 0
  # a run that gave the device to a queued broker job is no fault: nothing to repair
  [ -n "${FL_YIELD:-}" ] && { echo "Repair recommendation: none (run incomplete: yielded)" >&"${PG_OUT:-1}"; return 0; }
  local m=${FLEET:-repro} c r
  [ "${FL_SMOKE:-0}" = 1 ] && m=smoke
  c=$(git -C "$RR_DIR" rev-parse --short HEAD 2> /dev/null || echo "?")
  # dying (a TERM or INT; the broker KILLs 10 s after its TERM): 3 s at most, so the line still lands before that KILL
  r=$(timeout "$([ -n "${PG_DYING:-}" ] && echo 3 || echo 20)" python3 "${TT_DIR:-$RR_DIR}/fleet/repair_rec.py" --rc "$RR_RC" --verdict "$VD_TEXT" --mode "$m" --commit "$c" \
    ${FL_OUT:+--trays "$FL_OUT/trays.txt" --findings "$FL_OUT/findings.txt"} \
    ${OUT:+--fix "$OUT/link/fix.txt" --rootcause "$OUT/rootcause.tsv"} 2> /dev/null | tail -1)
  [[ $r == "Repair recommendation: "* ]] || r="Repair recommendation: unknown: see the FAIL lines above"
  [ -n "${FL_VARIANT_WARN:-}" ] && echo "$FL_VARIANT_WARN" >&"${PG_OUT:-1}"   # dev load variant, in the summary
  [ -n "${FL_SWAP_WARN:-}" ] && echo "$FL_SWAP_WARN" >&"${PG_OUT:-1}"   # dev trial knob, in the summary
  # a plain-words 'Final verdict:' block (what, where by tray / QSFP-DD port / ASIC / BDF, what
  # to do) just before the Repair recommendation line, which stays last; same dying cap as above
  local fv; fv=$(timeout "$([ -n "${PG_DYING:-}" ] && echo 3 || echo 20)" python3 "${TT_DIR:-$RR_DIR}/fleet/final_verdict.py" --rc "$RR_RC" --verdict "$VD_TEXT" --mode "$m" \
    --commit "$c" ${FL_OUT:+--findings "$FL_OUT/findings.txt"} ${OUT:+--result "$OUT/result.line"} \
    --result-text "${RS_RESULT_TEXT:-}" 2> /dev/null)
  [[ $fv == "Final verdict: "* ]] || fv="Final verdict: see the FAIL lines above"
  echo "$fv" >&"${PG_OUT:-1}"
  echo "$r" >&"${PG_OUT:-1}"
}
# layout: run.sh at the repo root, its code under traytest/; a flat copy (the tests) has lib/ beside it
TT_TOP=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")"); TT_DIR=$TT_TOP; [ -d "$TT_TOP/traytest/lib" ] && TT_DIR=$TT_TOP/traytest
. "$TT_DIR/lib/hc.sh"   # the hc core hooks (HC_SETUP, HC_HOLDFILE, HC_HOLD, HC_BROKER_YIELD, HC_SITE_FILE), off unless set
[ -n "${FLEET_MAXUTIL_SKIP:-}" ] || FLEET_MAXUTIL_SKIP=$(hc_site maxutil_skip)
# a subshell killed before its own 'trap - EXIT' (hb_start's heartbeat under a loaded CPU) runs this inherited trap
# on its way out (bash runs the EXIT trap on a fatal signal): only run.sh's own shell does the exit work; rc kept for vd_exit
trap 'RS_XRC=$?; [ "$BASHPID" = "$$" ] || exit "$RS_XRC"; (exit "$RS_XRC"); vd_exit; rs_sw_stop; declare -F by_stop > /dev/null && by_stop; declare -F pg_reap > /dev/null && pg_reap exit; declare -F fl_psu_reap > /dev/null && fl_psu_reap; declare -F pf_reap > /dev/null && pf_reap; { declare -F mg_recover > /dev/null && [ ${#MG_REMOVED[@]} -gt 0 ] && mg_recover; }; declare -F tdp_exit > /dev/null && tdp_exit; hb_stop; fault_exit; declare -F kp_restore > /dev/null && kp_restore; rr_final' EXIT   # kp_restore: lib/kmd_policy.sh (--kmd-power-policy); pg_reap: lib/pgroups.sh; fl_psu_reap: lib/psu_check.sh; pf_reap: lib/psu_fast.sh; fault_exit: lib/fault_capture.sh (a no-op before a fault, and when sourced)
fault_exit() { :; }
# TERM/INT/HUP from the start (pg_traps, lib/pgroups.sh): reap every recorded process group (the running try,
# any load) and exit 143/130/129; the EXIT trap above then runs. Set before any step, not only once $OUT exists.
declare -F pg_traps > /dev/null || . "$TT_DIR/lib/pgroups.sh"; pg_traps
# PIPE caught from the start: with HUP ignored (nohup, setsid) the reader of 'run.sh | tee' can go
# with no signal at all; this shell's next write then fails with EPIPE and the run goes on (its EXIT trap reaps as usual)
# instead of a SIGPIPE death (exit 141). Caught by a no-op, not ignored: children get the default back on exec, so a
# producer in 'cmd | head -1' still stops by SIGPIPE (an ignored PIPE would be inherited). rs_dying and pg_traps ignore it
trap ':' PIPE
declare -F hd_dump > /dev/null || . "$TT_DIR/lib/hsc_dump.sh"   # tray power dump before a glx_reset
declare -F hf_check > /dev/null || if [ -n "${HC_HOLDFILE:-}" ]; then . "$HC_HOLDFILE"; else hc_holdfile_off; fi   # the user's hold file (rk_gate, start)
# A TERM or INT (a leg's timeout: GNU timeout sends TERM, rc 124 to its caller) printed no RESULT line (one
# host's plain run: TERM in C9 and the log ends at 'heavy logs: ... skipped'). rs_traps:
# pg_traps, then TERM/INT/HUP go to rs_dying: reap the recorded groups (pg_reap), stop this shell's own other descendants (by
# pid, from /proc, never by name), print "RESULT: INCOMPLETE: stopped by SIG<sig> during <stage>; chips counted so far:
# k/N ..." (missing chips by BDF and tray 1-4), exit 143/130/129. Nothing is printed twice once the RESULT line is out (RS_DONE).
# RS_OUT: only while PG_OUT is on /dev/null (a shell that sourced run.sh > /dev/null), this shell's own output
# when the traps are set, so rs_dying still has one to go back to; a real run opens no extra fd
RS_DONE=""; RS_STAGE=""; RS_RESULT_TEXT=""
rs_traps() { pg_traps; trap 'rs_dying TERM 143' TERM; trap 'rs_dying INT 130' INT; trap 'rs_dying HUP 129' HUP
  local o1; { o1=$(readlink "/proc/$BASHPID/fd/9" 2> /dev/null); } 9>&1   # this shell's stdout: inside $(...) fd 1 is the pipe
  [ "$(readlink "/proc/$BASHPID/fd/${PG_OUT:-1}" 2> /dev/null)" = /dev/null ] && [ "$o1" != /dev/null ] || return 0
  if [ -n "${RS_OUT:-}" ]; then eval "exec $RS_OUT>&1"; else exec {RS_OUT}>&1; fi
}
rs_desc() {  # [pids] this shell's descendants (pids), from /proc's parent pids, leaving out this subshell's own subtree and
  # the pids given (listed nowhere, their children still listed)
  local f l s me=$BASHPID; for f in /proc/[0-9]*/stat; do read -r l 2> /dev/null < "$f" || continue; s=${l##*) }
    s=${s#* }; echo "${f//[!0-9]/} ${s%% *}"; done | awk -v top=$$ -v me="$me" -v skip=" ${1:-} " '{ k[$2] = k[$2] " " $1 }
    END { q[1] = top; n = 1; for (i = 1; i <= n; i++) { m = split(k[q[i]], c, " "); for (j = 1; j <= m; j++) if (c[j] != me) { q[++n] = c[j]; if (!index(skip, " " c[j] " ")) print c[j] } } }'
}
rs_kids() {  # $1 the deadline (us since the epoch): TERM this shell's own descendants, KILL those still there at it, and
  # any started meanwhile
  local p ps l q=$BASHPID me="" all=" " n s d dl=${1:-$(( ${EPOCHREALTIME/[.,]/} + 2000000 ))}   # no deadline given (rk_bound_hit, the step timer): 2 s
  # called in a subshell: this subshell and its parents up to $$ are never signalled; one that ignores TERM would STOP
  # itself below and hang
  while [ "$q" != $$ ] && [ "$q" -gt 1 ] 2> /dev/null; do me="$me $q"; read -r l 2> /dev/null < "/proc/$q/stat" || break; l=${l##*) }; l=${l#* }; q=${l%% *}; done
  ps=$(rs_desc "$me"); [ -n "$ps" ] || return 0
  # STOP, TERM, CONT: a TERM-default looper dies on CONT, before it can fork a child the snapshot missed
  for p in $ps; do kill -STOP "$p" 2> /dev/null; done; for p in $ps; do kill -TERM "$p" 2> /dev/null; done; for p in $ps; do kill -CONT "$p" 2> /dev/null; done
  while [ "${EPOCHREALTIME/[.,]/}" -lt "$dl" ]; do ps=$(for p in $ps; do kill -0 "$p" 2> /dev/null && echo "$p"; done); [ -n "$ps" ] || break; sleep 0.2; done
  # freeze, then KILL: a looping descendant forks a fresh 'sleep 1' after one snapshot, and its parent KILLed
  # before it leaves that child out of the tree, still alive once run.sh is gone. STOP all, recount until none is new.
  # The recount filters in this shell: a $(...) around it was itself a new pid each round, so all 10 ran
  for p in $ps; do kill -STOP "$p" 2> /dev/null; all="$all$p "; done
  for n in 1 2 3 4 5 6 7 8 9 10; do RS_KIDS_ROUNDS=$n; ps=""; mapfile -t d < <(rs_desc "$me"); for p in "${d[@]}"; do [[ $all == *" $p "* ]] || ps="$ps $p"; done
    [ -n "$ps" ] || break; for p in $ps; do kill -STOP "$p" 2> /dev/null; all="$all$p "; done; done
  for p in $all; do kill -KILL "$p" 2> /dev/null; done
  for n in 1 2 3 4 5 6 7 8 9 10; do ps=""; for p in $all; do s=$(cat "/proc/$p/stat" 2> /dev/null) && [[ ${s##*) } != Z* ]] && ps=1; done
    [ -n "$ps" ] || break; sleep 0.05; done; return 0   # KILL lands asynchronously: gone (or a zombie) before we return
}
# rs_dying: the broker KILLs 10 s after its TERM, and the reaping first then the count took
# 6.3-7.7 s (test_fleet.sh unit 125). Now the RESULT line comes early, its chip count bounded (RS_COUNT_MS, then 0.5 s
# per chip read at most once); then the whole clean-up (the recorded groups, this shell's other descendants, the EXIT
# trap's own pg_reap) ends within RS_DIE_S: groups get TERM and RS_DIE_S - 1 s, then KILL; descendants KILL at RS_DIE_S.
# the device stages (chunk_load.py, agloop.py, fabric_repro.py, h3shape_repro.py, numeric_dram.py,
# numeric_pinpoint.py: fleet/stop_signal.py; maxutil.py) close their devices on TERM. Their groups get the TERM first and
# up to RS_CLOSE_MS to end before the RESULT line (pg_reap PG_SOFT: no KILL), so a stage's close comes before it; the
# RS_DIE_S bound counts from the signal, so the close wait and the count come out of it: all within the broker's 10 s.
RS_DIE_S=${TRAY_REPRO_DIE_S:-5}; RS_COUNT_MS=${TRAY_REPRO_DIE_COUNT_MS:-1500}; RS_CLOSE_MS=${TRAY_REPRO_DIE_CLOSE_MS:-1000}
rs_dying() {  # $1 TERM|INT|HUP, $2 the exit code
  trap '' TERM INT HUP PIPE; PG_DYING=$1; HB_PID=""   # the heartbeat loop is a descendant: rs_kids ends it
  # PIPE: the run's output pipe can be gone first (an ssh drop on 'run.sh | tee'); a write then fails with EPIPE
  # instead of killing this shell (exit 141) before the recorded groups and its descendants are reaped
  # the signal can land while this shell waits inside a step's own redirect (pg_run ... > file, heavy_logs.sh):
  # back to the run's own output (PG_OUT, lib/pgroups.sh) first, stdout and stderr, so the RESULT line, the
  # EXIT trap's VERDICT and its messages and the final 'Repair recommendation:' line land in run.log, not in that step's
  # file. A shell that sourced run.sh > /dev/null (the offline tests) has PG_OUT on /dev/null: its own output (RS_OUT,
  # rs_traps) instead, else the output stays
  local o=${PG_OUT:-1}
  [ "$(readlink "/proc/$BASHPID/fd/$o" 2> /dev/null)" = /dev/null ] && o=${RS_OUT:-}
  [ -z "$o" ] || exec 1>&"$o" 2>&1
  local dl=$(( ${EPOCHREALTIME/[.,]/} + RS_DIE_S * 1000000 ))
  declare -F pg_reap > /dev/null && PG_SOFT=1 PG_DEADLINE_US=$(( ${EPOCHREALTIME/[.,]/} + RS_CLOSE_MS * 1000 )) pg_reap   # the stages close first
  if [ -z "$RS_DONE" ]; then
    local st="" ph=""
    [ -n "${FL_OUT:-}" ] && ph=$(tail -1 "$FL_OUT/phases.txt" 2> /dev/null) && [ -n "$ph" ] && ph="${ph% *} ($(( $(date +%s) - ${ph##* } )) s in)"
    st=${ph:-${RS_STAGE:-the run}}; [ -n "$ph" ] && [ -n "$RS_STAGE" ] && st="$st, last step: $RS_STAGE"
    [ -n "${FL_OUT:-}" ] && [ "$(type -t fl_vd_sum)" = function ] && fl_vd_sum   # finished stages' verdicts and the stages not reached (the stage in progress is not finished); prints only, no chip access
    RS_RESULT_TEXT="RESULT: INCOMPLETE: stopped by SIG$1 during $st; chips counted so far: $(RK_SNAP_DL=$(( ${EPOCHREALTIME/[.,]/} + RS_COUNT_MS * 1000 )) RK_OD_S=0.5 rs_count)${SR_TAG:-}"
    echo "$RS_RESULT_TEXT"; RS_DONE=1
  fi
  PG_DEADLINE_US=$(( dl - 1000000 )); declare -F pg_reap > /dev/null && pg_reap; rs_kids "$dl"   # PG_DEADLINE_US stays set: the EXIT trap's pg_reap waits no more
  exit "$2"
}
rs_count() {  # "k/N answer PCIe config space with a tt-kmd node; missing: <BDF (tray T)>..." (rk_snap, live; RK_SNAP_DL: chips
  # not read by then are "unread", named as missing and counted in a note)
  local snap n tot miss u
  if [ -n "${RK_MAP:-}" ]; then snap=$(rk_snap); n=$(grep -c ' ok$' <<< "$snap"); tot=$(rk_total)
    miss=$(awk '$3 != "ok"' <<< "$snap" | rk_bdfs); [ -n "$(rk_short)" ] && miss=${miss:+$miss; }$(rk_short)
    u=$(grep -c ' unread$' <<< "$snap"); [ -n "${RK_SNAP_DL:-}" ] && [ "$u" -gt 0 ] && u="; $u chip(s) not read within the ${RS_COUNT_MS:-?} ms count bound" || u=""
    echo "$n/$tot answer PCIe config space with a tt-kmd node${miss:+; missing: $miss}$u"
  else echo "none (stopped before the first chip count)"; fi
}
rs_left() { return 0; }   # no run budget: empty, so every gate takes its no-budget path
# (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
# The run budget's clock: it starts only after the fixed setup, the venv check, preflight,
# the start reset (C2, glx_reset) and its recount, discovery (rs_clock_start after fl_plan) and the first mesh open and compile
# (rs_mesh_credit after the first load try). Until then RS_T0 is empty: rs_left is empty and nothing is cut for time.
rs_hl_set() { return 0; }   # (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
rs_clock_start() {  # the test stages start now: RS_T0, RS_SETUP_S (since RS_START), HL_DEADLINE; the first load try's mesh open is credited later
  RS_T0=$(date +%s); RS_SETUP_S=$(( RS_T0 - ${RS_START:-$RS_T0} )); RS_MESH_PEND=1; rs_hl_set
    return 0
}
rs_mesh_credit() {  # <try log> <try start epoch> <try>: the first load try's mesh open, upload and compile (to its first quiet or
  # its upload end) is setup too, once a run: RS_T0 moves on by it
  [ "${RS_MESH_PEND:-0}" = 1 ] && [ -n "${RS_T0:-}" ] || return 0
  local ts e s n; ts=$(grep -a -m1 -oE 'STRESS\|[^|]+\|idle [0-9]+s begin' "$1" 2> /dev/null | cut -d'|' -f2)
  [ -n "$ts" ] || ts=$(grep -a -m1 -oE 'STRESS\|[^|]+\|(upload end|warm-up end|mesh open \()' "$1" 2> /dev/null | cut -d'|' -f2)
  [ -n "$ts" ] || return 0   # no mesh mark (not a load try): the next try with one is credited
  RS_MESH_PEND=0; e=$(date -d "$ts" +%s 2> /dev/null) || return 0; n=$(date +%s)
  s=$(( e - $2 )); [ "$s" -gt $(( n - $2 )) ] && s=$(( n - $2 )); [ "$s" -gt 0 ] || return 0
  [ "$s" -gt $(( n - RS_T0 )) ] && s=$(( n - RS_T0 ))
  RS_MESH_S=$s; RS_SETUP_S=$(( ${RS_SETUP_S:-0} + s )); RS_T0=$(( RS_T0 + s )); [ -n "${HL_DEADLINE:-}" ] && rs_hl_set
  [ "${RS_RUN_S:-0}" -gt 0 ] 2> /dev/null && say "run budget: the first mesh open and compile took $s s ($3): setup, not counted in TRAY_REPRO_RUN_S=$RS_RUN_S"
  return 0
}
rs_time_line() {  # the final log's setup and test-stage times, apart
  [ -n "${RS_T0:-}" ] || { echo "run time: setup $(( $(date +%s) - ${RS_START:-$(date +%s)} )) s, test stages 0 s (none started)"; return 0; }
  echo "run time: setup ${RS_SETUP_S:-0} s (venv, preflight, C2 reset and recount, discovery, first mesh open and compile${RS_MESH_S:+: $RS_MESH_S s}), test stages $(( $(date +%s) - RS_T0 )) s$([ "${RS_RUN_S:-0}" -gt 0 ] 2> /dev/null && echo " of TRAY_REPRO_RUN_S=$RS_RUN_S")"
}
# rs_pin_due / rs_pin_keep (one host's plain run: a NUMERIC FAIL at the bus-bar step with 374 s left, the 90 s heavy chunk
# still ran, the 253 s numeric pinpoint then left -12 s and the leg's timeout cut C9 89 s in, before its reset): once a NUMERIC
# FAIL is on record and the pinpoint will run (no chip lost, no held hang), its TRAY_PINPOINT_S is kept free too. Load parts,
# try caps, C5 and the heavy-log line (HL_DEADLINE) then end or are skipped first; a started pinpoint is never cut.
rs_pin_due() {  # 0 while the numeric pinpoint is still to run after a NUMERIC FAIL (the files pinpoint_numeric reads)
  { [ "${TRAY_PINPOINT:-1}" = 0 ] || [ -n "${RS_PIN_DONE:-}" ] || [ "${seen:-0}" = 1 ] || [ -n "${ccl:-}${TRY_HUNG:-}${FL_HUNG:-}" ]; } && return 1
  grep -aqs '^NUMERIC FAIL' /dev/null ${OUT:+"$OUT/numeric.txt"} ${FL_OUT:+"$FL_OUT/busbar/step/numeric.txt" "$FL_OUT/busbar/report.txt" "$FL_OUT/busbar/heavy/report.txt" "$FL_OUT/c7/report.txt"}
}
rs_pin_keep() { rs_pin_due && echo "${TRAY_PINPOINT_S:-300}" || echo 0; }   # in rs_keep: the numeric pinpoint's share while it is due
rs_sweep_keep() { [ -n "${FL_PIN:-}" ] && [ "${HOLD:-0}" != 1 ] && echo "${RS_SWEEP_S:-0}" || echo 0; }   # only once a fault set FL_PIN: a healthy run keeps no sweep time
rs_keep() { echo $(( RS_C9_S + RS_TAIL_S + $(rs_sweep_keep) + $(declare -F rs_pin_keep > /dev/null && rs_pin_keep || echo 0) )); }   # kept free before C9: its share, the sweep's (after a fault), the numeric pinpoint's, the RESULT's
rs_keep_why() { echo "$(rs_keep) s kept for the C9 reset step$([ "$(rs_sweep_keep)" -gt 0 ] && echo ", the PINPOINT sweep ($RS_SWEEP_S s, $FL_PIN)")$(declare -F rs_pin_keep > /dev/null && [ "$(rs_pin_keep)" -gt 0 ] && echo ", the numeric pinpoint ($(rs_pin_keep) s)") and the RESULT"; }
RS_RESULT_S=${TRAY_REPRO_RESULT_S:-60}   # kept free for the RESULT after the last post-C9 step
rs_fits() { return 0; }   # (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
# No stage skipped for time (one host's quick pass: a bus-bar CCL error set FL_PIN, the PINPOINT sweep's 240 s
# joined rs_keep, and fault capture, stage 6, the heavy-log snapshot and the post link check were skipped as INCOMPLETE): once
# run.sh has found a fault (rs_fault), a stage short of time grows the run's own budget (rs_extend) by what it is short plus
# its need (RS_EXT_S, 120 s, unless the caller names one) instead of being skipped. Every stage keeps its own bounded waits,
# and the budget grows only at a stage's start, so the run stays bounded. A run with no fault keeps its budget as before.
RS_EXT_S=${TRAY_REPRO_EXT_S:-120}; RS_EXT_TOTAL=0; RS_FAULT=""
rs_fault() {  # $1 stage: 0 once run.sh has found a fault (a fault capture is one by its name)
  case ${1:-} in "fault capture"*) return 0 ;; esac
  [ -n "${RS_FAULT:-}${FL_PIN:-}${FL_HUNG:-}${FL_PGDROP:-}${FL_FABSTALL:-}${FL_C3STOP:-}${FL_LINKS_DOWN:-}" ] && return 0
  declare -F rs_pin_due > /dev/null && rs_pin_due
}
rs_extend() { return 0; }   # (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
# Progress widening (a quick pass on one cluster node ended INCOMPLETE, 'run budget 1800 s spent before
# bus-bar step', with no fault): while the run is still progressing, a stage short of time widens the run's budget as a fault
# does. Progress is a new mark within RS_PROG_WIN_S: the start of a load stage (hb_start with a log) or the newest
# STRESS line in that stage's log. A stage with no new mark for longer is stalled: no widening, and its own bounded wait still
# ends it. Widening for progress is capped at RS_PROG_CAP_S in all (a fault's extension is not); the same in every mode.
RS_PROG_WIN_S=${TRAY_REPRO_PROG_WIN_S:-300}; RS_PROG_CAP_S=${TRAY_REPRO_PROG_CAP_S:-900}; RS_PROG_EXT=0; RS_PROG_T=""; RS_PROG_STAGE=""; RS_PROG_LOG=""
rs_prog_mark() {  # the newest progress mark: "<epoch> <what>" (empty: none yet)
  local t=${RS_PROG_T:-} w="${RS_PROG_STAGE:-} started" ts e
  ts=$(grep -a -oE 'STRESS\|[^|]+\|[^|]{0,80}' "${RS_PROG_LOG:-/dev/null}" 2> /dev/null | tail -1)
  [ -n "$ts" ] && e=$(date -d "$(cut -d'|' -f2 <<< "$ts")" +%s 2> /dev/null) && [ "$e" -gt "${t:-0}" ] && { t=$e; w="STRESS: $(cut -d'|' -f3 <<< "$ts")"; }
  [ -n "$t" ] && echo "$t $w"; return 0
}
rs_progressing() {  # 0 while the newest progress mark is at most RS_PROG_WIN_S old
  local m; m=$(rs_prog_mark); [ -n "$m" ] && [ $(( $(date +%s) - ${m%% *} )) -le "$RS_PROG_WIN_S" ]
}
rs_widen() { return 0; }   # (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
rs_need() { return 0; }   # (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
rs_fault_why() { local w; for w in "${RS_FAULT:-}" "${FL_PIN:-}" "${FL_HUNG:-}" "${FL_PGDROP:-}" "${FL_FABSTALL:-}" "${FL_C3STOP:-}" "${FL_LINKS_DOWN:-}"; do [ -n "$w" ] && { echo "$w"; return; }; done; echo "numeric FAIL or fault capture"; }
rs_budget() { return 0; }   # (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
# rs_c9_share <C2 start epoch> <why>: on a host whose C2 needed a glx_reset, whose tt-umd
# is stock or whose warm reset failed (RK_RESET_FAILS), C9 may need as long as C2's reset did (one host: 485 s,
# its C9 got 170 s and was killed mid glx_reset). RS_C9_S becomes C2's measured reset time, at most RK_STEP_S: later load
# stages start only while that share plus RS_TAIL_S remain (rs_budget, fl_try_cap, HL_DEADLINE). A clean host keeps 180 s.
rs_c9_share() {
  local m; [ "${RS_RUN_S:-0}" -gt 0 ] 2> /dev/null || return 0
  m=$(( $(date +%s) - $1 )); [ "$m" -gt "$RK_STEP_S" ] && m=$RK_STEP_S
  [ "$m" -gt "$RS_C9_S" ] || return 0
  RS_C9_S=$m; rs_hl_set
  say "run budget: C9 reset share raised to $m s ($2; C2's reset took $m s): a load stage starts only while $(rs_keep) s of TRAY_REPRO_RUN_S=$RS_RUN_S remain"
}
# rs_c9_loss <where>: a chip lost in a load stage before C9 (fl_lost): C9 needs its warm reset,
# the settle wait and a whole recovery round, so RS_C9_S becomes at least RK_WARM_S + RK_SETTLE_S + RK_ROUND_S (60 + 60 +
# 240 = 360 s); fault capture and the pre-reset steps end by the moved HL_DEADLINE. A clean run keeps 180 s (section 5).
RK_WARM_S=${TRAY_REPRO_WARM_S:-60}   # C9's glx_reset took 44-52 s on record
rs_c9_loss() {
  local m; RS_FAULT=${RS_FAULT:-chips lost in $1}; [ "${RS_RUN_S:-0}" -gt 0 ] 2> /dev/null || return 0
  m=$(( RK_WARM_S + RK_SETTLE_S + RK_ROUND_S )); [ "$m" -gt "$RK_STEP_S" ] && m=$RK_STEP_S
  [ "$m" -gt "$RS_C9_S" ] || return 0
  RS_C9_S=$m; rs_hl_set
  say "run budget: C9 reset share raised to $m s (chips lost in $1: glx_reset $RK_WARM_S + settle $RK_SETTLE_S + a recovery round $RK_ROUND_S s); later steps before C9 end $(rs_keep) s before the end"
}
# the fixed waits that ate 4 legs on 10-26 were not stages: fault capture (tt-smi snapshot 120 s,
# eth counters 180 s, tt-triage, exalens), the pre-reset heavy snapshot and the post host-link check. Each now starts only
# while rs_budget allows (rs_step), and while the budget is on, every heavy_logs step before C9 ends by the same line
# (HL_DEADLINE, lib/heavy_logs.sh hl_step: cut to it, or skipped past it).
rs_step() { return 0; }   # (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
# A stalled chunk (one host: all 4 trays stalled in a bus-bar chunk, the hang was seen with 331 s left and C9 keeping
# 391 s, so its chip-state capture never started): a load stage's hang always gets RS_CAP_S for its capture. The load
# watch stops a load at the line that keeps RS_CAP_S + RS_C9_S + RS_RESULT_S (lib/fleet.sh fl_loads), and the capture
# runs in that reserve (its heavy_logs steps cut to it), even past rs_budget's line; C9's share and the RESULT keep
# their time. Under RS_CAP_MIN_S (20 s) left in it, the capture is skipped and says why.
RS_CAP_S=${TRAY_REPRO_CAP_S:-120}; RS_CAP_HL=""; RS_CAP_TR=""
rs_cap_line() { return 0; }   # (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
rs_cap_step() { return 0; }   # (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
rs_cap_end() { RS_CAP_HL=""; RS_CAP_TR=""; }   # (no run budget cuts the capture, so no line to restore)
rs_hl_end() { HL_DEADLINE=""; }   # (no run budget cuts a heavy-logs step, so none is named here)
rs_traps
rs_stopwatch
RK_N=0   # chips on the map (rk_map, rk_remap); 0 until the first map, so rk_total (tdp_allback, rk_eacces) reads RK_WANT before it
rk_map() {  # at the start, while every chip answers: "<bdf> <chip>" for each Blackhole function (chip from tenstorrent!N)
  local d c v n
  RK_MAP=$(for d in "$SYS"/bus/pci/devices/*; do
    v=; read -r v < "$d/vendor" 2>/dev/null; [ "$v" = 0x1e52 ] || continue
    c=; for n in "$d"/tenstorrent/tenstorrent!*; do [[ ${n##*/tenstorrent!} =~ ^[0-9]+$ ]] && { c=${n##*/tenstorrent!}; break; }; done
    echo "${d##*/} ${c:-?}"; done)
  if [ "${KD_ON:-0}" = 1 ]; then   # --known-dropped-tray: tray N stays off the map (KD_MAP: its chips still on PCIe)
    KD_MAP=$(echo "$RK_MAP" | while read -r d c; do [ -n "$d" ] && [ "$(rk_tray "$d")" = "$KD_TRAY" ] && echo "$d $c"; done)
    RK_MAP=$(echo "$RK_MAP" | while read -r d c; do [ -n "$d" ] && [ "$(rk_tray "$d")" != "$KD_TRAY" ] && echo "$d $c"; done)
  fi
  RK_N=$(echo "$RK_MAP" | grep -c .)
}
rk_state() {  # one chip: ok (answers config space and has its tt-kmd node), lost (gone from the bus or reads ffff),
              # unread (the read timed out or failed: unknown)
  local d=$SYS/bus/pci/devices/$1 w rc
  [ -e "$d" ] || { echo lost; return; }
  w=$(timeout "${RK_OD_S:-2}" od -An -tx2 -N2 "$d/config" 2>/dev/null); rc=$?; w=${w//[[:space:]]/}
  if [ $rc != 0 ] || [ -z "$w" ]; then echo unread
  elif [ "$w" != 1e52 ]; then echo lost
  else for w in "$d"/tenstorrent/tenstorrent!*; do [ -e "$w" ] && { echo ok; return; }; done; echo lost; fi   # inline: tests extract rk_state alone
}
rk_snap() {  # RK_SNAP_DL (us since the epoch; rs_dying): chips not read by then are "unread"
  local b c; while read -r b c; do [ -n "$b" ] || continue
    if [ -n "${RK_SNAP_DL:-}" ] && [ "${EPOCHREALTIME/[.,]/}" -ge "$RK_SNAP_DL" ]; then echo "$b $c unread"; else printf '%s %s ' "$b" "$c"; rk_state "$b"; fi   # rk_state ends the line: no subshell per chip
  done <<< "$RK_MAP"; }
rk_remap() {  # rk_map again (chips back, new tt-kmd nodes), keeping each chip mapped before that is now off the bus: a chip
              # the reset took off the bus stays on the map as lost, never silently dropped from the count
  local old=$RK_MAP; rk_map
  RK_MAP=$({ echo "$RK_MAP"; awk 'NR == FNR { if (NF) m[$1] = 1; next } NF && !($1 in m)' <(echo "$RK_MAP") <(echo "$old"); } | grep . | LC_ALL=C sort -k1,1)
  RK_N=$(echo "$RK_MAP" | grep -c .)
}
rk_total() { echo $(( RK_N > ${RK_WANT:-0} ? RK_N : ${RK_WANT:-0} )); }   # chips there should be
rk_nonode() {  # chips on the map without a tt-kmd node (chip '?') that still have none, as
               # "tray 4 (1 chip without a tt-kmd node)"
  echo "$RK_MAP" | while read -r b c; do [ -n "$b" ] && [ "$c" = "?" ] && [ "$(rk_state "$b")" != ok ] && rk_tray "$b"; done |
    awk '{ k = $1; if (!(k in n)) o[++m] = k; n[k]++ }
         END { for (i = 1; i <= m; i++)
           printf "%stray %s (%d chip%s without a tt-kmd node)", (i > 1 ? "; " : ""), o[i], n[o[i]], (n[o[i]] > 1 ? "s" : "") }'
}
rk_short() {  # with RK_WANT set (run.sh: 32, 4 trays of 8): trays with chips missing from PCIe altogether (not in
              # RK_MAP, e.g. a tray that was down at boot), as "tray 4 (8 of 8 chips not on PCIe)"; else nothing
  local nn pcie="" nt=4; nn=$(rk_nonode); [ "${KD_ON:-0}" = 1 ] && nt=3   # a known dropped tray is not counted
  if [ -n "${RK_WANT:-}" ] && [ "$RK_N" -lt "$RK_WANT" ]; then
    pcie=$(echo "$RK_MAP" | while read -r b c; do [ -n "$b" ] && rk_tray "$b"; done |
      awk -v per=$(( RK_WANT / nt )) -v kd="${KD_TRAY:-0}" -v on="${KD_ON:-0}" '{ n[$1]++ } END {
        for (i = 1; i <= 4; i++) if (n[i] + 0 < per && !(on && i == kd))
          printf "%stray %s (%d of %d chips not on PCIe)", (o++ ? "; " : ""), i, per - n[i], per }')
  fi
  echo "$pcie${pcie:+${nn:+; }}$nn"
}
rk_tray() {  # PCI address -> tray "N" (1-4, "?" if unknown), from bus & 0xF0 (as repro_job.sh). Takes 0000:88:00.0,
             # 88:00.0 (vc_report's short form crashed here on 'line 585: 16#00.0') or an lspci -D line; an
             # address with no bus byte gives "?" and one ERROR line on stderr naming it, never a bash arithmetic error
  local a=${1#"${1%%[![:space:]]*}"}; a=${a%%[[:space:]]*}; a=${a##*/}   # first word; a sysfs path keeps its last part; no fork
  if [[ $a =~ ^([0-9a-fA-F]{4,}:)?([0-9a-fA-F]{2}):[0-9a-fA-F]{2}\.[0-7]$ ]]; then
    case $(( 16#${BASH_REMATCH[2]} & 0xF0 )) in
      0) echo 1 ;; 64) echo 2 ;; 192) echo 3 ;; 128) echo 4 ;; *) echo "?" ;; esac
  else echo "ERROR: PCI address '$1' has no bus byte (want 0000:88:00.0 or 88:00.0): tray 1-4 unknown, named tray ?" >&2; echo "?"; fi
}
rk_list() {  # chip numbers -> "0-3,7"
  tr ' ,' '\n\n' | grep -E '^[0-9]+$' | sort -n | uniq |
    awk 'function f(){ o = o (o == "" ? "" : ",") (s == p ? s : s "-" p) } NR == 1 { s = p = $1; next }
         $1 == p + 1 { p = $1; next } { f(); s = p = $1 } END { if (NR) f(); print o }'
}
rk_trays() {  # "<bdf> <chip> <state>" lines -> "tray 1 (chips 0-7); tray 4 (chips 24-31)", or nothing
  while read -r b c st; do [ -n "$b" ] && [ "$c" != "?" ] && echo "$(rk_tray "$b") $c"; done | sort -k1,1 -k2,2n |
    awk '{ k = $1; if (!(k in l)) { o[++n] = k; l[k] = $2 } else l[k] = l[k] " " $2 }
         END { for (i = 1; i <= n; i++) print o[i] "|" l[o[i]] }' |
    while IFS='|' read -r k cs; do echo "tray $k (chips $(echo "$cs" | rk_list))"; done | paste -sd';' | sed 's/;/; /g'
}
rk_loss_chips() {  # chips named by the FIRST LOSS text in a log ("chip(s) [24, 25]"), space separated
  grep -a -o 'chip(s) \[[0-9, ]*\]' "$1" 2>/dev/null | head -1 | tr -cs '0-9' ' '
}
# rk_tmark <file> begin|end <label>: an epoch mark for the 12 V log judge (fleet/tray_12v.py), which
# masks +-40 s around each reset and flags vr_pg only inside load windows; a timestamp only, no reset reporting change
rk_tmark() { [ -n "${1:-}" ] && printf '%s\t%s\t%s\n' "$(date +%s)" "$2" "${3:-}" >> "$1" 2> /dev/null; return 0; }
rk_reset() {  # $1 a log path: the warm reset of every chip. Sets RK_RESET_OK (1/0), RK_RESET_WHY and RK_RESET_SKIP (the
              # serving gate's refusal when the gate skipped it, else empty)
  local t0 rc ok lostids
  t0=$(date +%s); RK_RESET_OK=0; RK_RESET_WHY=""; RK_RESET_SKIP=""; RK_RESET_NONE=0; RK_GLX_RAN=0; RK_RESET_LBL="glx_reset (tt-smi -glx_reset)"
  # RK_RESET_LOG (settled sa4): the one log every reset-result parse reads (rk_umd_named, rk_slowmark, rk_end_loss, fleet C2).
  # RK_RESET_STAT: OK, FAILED (glx_reset ran and failed), SKIPPED-GATE or NONE, decided here for C2 and C9 alike (settled sa3).
  RK_RESET_LOG="${1%.log}_glx.log"; RK_RESET_STAT=NONE
  if ! rk_gate; then   # the re-check still runs; rk_end_ makes the run INCOMPLETE (exit 9), never a pass or a tray drop
    RK_RESET_SKIP=$RK_GATE_WHY; RK_RESET_WHY="SKIPPED-GATE: $RK_GATE_WHY"; RK_RESET_STAT=SKIPPED-GATE; say "reset: glx_reset $RK_RESET_WHY"; return 0
  fi
  # ('never use -r'): every reset step
  # (start, C2, C9, C9s, C9b) runs tt-smi -glx_reset, in every mode. tt-smi -r (tt-umd WarmReset) is never run. Where
  # glx_reset cannot run (no root or sudo -n -l entry for it, no tt-smi, IPMI not answering, TRAY_REPRO_NO_GLX=1, --hold),
  # no reset runs and a "NO RESET" line says why. The caller (rk_end_) rescans the root ports and recounts.
  if [ "${HOLD_ACTIVE:-0}" = 1 ]; then RK_RESET_WHY="--hold: no reset (a hung chip is left as it is)"
  elif [ "${TRAY_GLX_FIRST:-1}" = 0 ]; then RK_RESET_WHY="glx_reset off (TRAY_GLX_FIRST=0, test hook)"
  elif rk_noglx; then RK_RESET_WHY="glx_reset skipped by TRAY_REPRO_NO_GLX=1"
  elif ! rk_can_glx; then RK_RESET_WHY="tt-smi -glx_reset cannot run here ($RK_GLX_CAN_WHY)"
  else
    rk_glx "$RK_RESET_LOG"
    if [ "$RK_GLX_OK" = 1 ] && [ -z "$RK_GLX_SKIP" ]; then RK_RESET_OK=1; RK_RESET_STAT=OK; RK_RESET_LBL="glx_reset (tt-smi -glx_reset)"; return 0; fi
    [ -n "$RK_GLX_SKIP" ] && RK_RESET_SKIP=$RK_GLX_SKIP
    RK_RESET_WHY="tt-smi -glx_reset did not run or failed (${RK_GLX_WHY:-no reason})"
    if [ "$RK_GLX_RAN" = 1 ]; then   # it ran and failed: a failed reset, the re-check decides; nothing else runs after it
      RK_RESET_STAT=FAILED; say "reset: ${RK_STEP:-reset}: $RK_RESET_WHY; no further reset follows this step (rk_recover may still glx_reset), never tt-smi -r"; return 0; fi
    [ -n "$RK_GLX_SKIP" ] && RK_RESET_STAT=SKIPPED-GATE
  fi
  RK_RESET_NONE=1
  say "reset: ${RK_STEP:-reset}: NO RESET: $RK_RESET_WHY; never tt-smi -r ($(( $(date +%s) - t0 )) s)"
}
rk_gate() {  # the user's hold file (lib/holdfile.sh), then TRAY_REPRO_RESET_GATE (this host's own read-only serving check), right before every reset. Sets RK_GATE_WHY
  local g r; RK_GATE_WHY=""
  if declare -F hf_check > /dev/null; then hf_check || { RK_GATE_WHY="hold: $HF_WHY"; return 1; }; fi   # loaded at the top of run.sh
  [ -z "${TRAY_REPRO_RESET_GATE:-}" ] && return 0
  g=$(bash -c "$TRAY_REPRO_RESET_GATE" 2>&1); r=$?
  echo "$g" | sed '$d' | grep -a . | while IFS= read -r l; do say "reset gate: $l"; done   # WAIT/WARNING lines
  [ $r = 0 ] && return 0
  RK_GATE_WHY="serving gate refused: $(echo "$g" | tail -1 | cut -c1-200)"; return 1
}
# Reset hook (opt-in): lets something outside run.sh take read-only snapshots right before and right after
# every chip reset run.sh runs (warm reset, warm reset of the chips that did not answer, tray re-power). With neither
# variable below set, nothing changes: no line, no file, no call. <step>: start, end (no --mode), C2, C3g, C9, C9s, C9b
# (fleet), with -lost for the warm reset of the chips that did not answer and -glx for the tray re-power (C2-glx, ...).
# <seq> counts the run's resets from 1 (<results dir>/reset_hook.seq). Each reset logs
#   RESET_HOOK pre <seq> <step> <epoch> <bdf>=tray<N>,... wait=<S>s     (every chip the host has; wait: the hook and ack time)
#   RESET_HOOK post <seq> <step> <epoch> rc=<the reset's rc> wait=<S>s
# TRAYTEST_RESET_SYNC_DIR=<dir on the run host>: writes <dir>/<seq>-<step>.pre (the pre line, then "dir <results dir>")
#   before the reset and waits up to TRAYTEST_RESET_SYNC_WAIT_S (default 30, at most 60) for <dir>/<seq>-<step>.pre.ack
#   (else "reset hook: no ack after <N>s, continuing"); writes <dir>/<seq>-<step>.post after it and does not wait.
# TRAYTEST_RESET_HOOK=<executable on the run host>: timeout -k 5 30 <hook> pre|post <step> <results dir>, output to
#   <results dir>/reset_hook.log; a non-zero rc (124: timed out, 137: killed) logs "RESET_HOOK WARN <step> pre|post rc=N".
# Neither ever skips, fails or adds a reset, nor changes a verdict. A reset the gate skipped (also on the re-check after pre) runs nothing: post logs rc=skipped.
rk_hook() {  # $1 pre|post, $2 the step, $3 (post) the reset's rc
  [ -n "${TRAYTEST_RESET_HOOK:-}${TRAYTEST_RESET_SYNC_DIR:-}" ] || return 0
  local d=${OUT:-$PWD} s=${TRAYTEST_RESET_SYNC_DIR:-} t0 e f="" l q rc w b c
  t0=${EPOCHREALTIME/,/.}; e=${t0%.*}; q=$(cat "$d/reset_hook.seq" 2>/dev/null); q=${q:-0}
  if [ "$1" = pre ]; then q=$((q + 1)); echo $q > "$d/reset_hook.seq"
    l="RESET_HOOK pre $q $2 $e $(echo "$RK_MAP" | while read -r b c; do [ -n "$b" ] && printf '%s=tray%s,' "$b" "$(rk_tray "$b")"; done)"; l=${l%,}
  else l="RESET_HOOK post $q $2 $e rc=${3:-?}"; fi
  if [ -n "$s" ]; then f=$s/$q-$2.$1; mkdir -p "$s" 2> /dev/null
    { printf '%s\ndir %s\n' "$l" "$d" > "$f.tmp" && mv -f "$f.tmp" "$f"; } 2> /dev/null || say "reset hook: cannot write $f"; fi
  if [ -n "${TRAYTEST_RESET_HOOK:-}" ]; then
    pg_run timeout -k 5 30 "$TRAYTEST_RESET_HOOK" "$1" "$2" "$d" < /dev/null >> "$d/reset_hook.log" 2>&1; rc=$?
    [ $rc = 0 ] || say "RESET_HOOK WARN $2 $1 rc=$rc"; fi
  if [ -n "$f" ] && [ "$1" = pre ]; then
    w=${TRAYTEST_RESET_SYNC_WAIT_S:-30}; [[ $w =~ ^[0-9]+$ ]] || w=30; [ "$w" -gt 60 ] && w=60
    e=$(( ${t0/./} + w * 1000000 ))   # the deadline in microseconds, from the start of this call
    until [ -e "$f.ack" ]; do [ "${EPOCHREALTIME/[.,]/}" -ge $e ] && { say "reset hook: no ack after ${w}s, continuing"; break; }; sleep 0.2; done; fi
  say "$l wait=$(awk -v a="$t0" -v b="${EPOCHREALTIME/,/.}" 'BEGIN { printf "%.2f", b - a }')s"
  return 0
}
rk_pre() {  # $1 the step: rk_hook pre, then (hook on only) the gate again, so it is the last check before the reset.
            # 1 when it now refuses (RK_GATE_WHY set); the post call is still made (rc=skipped), pre/post stay paired
  rk_hook pre "$1"; [ -n "${TRAYTEST_RESET_HOOK:-}${TRAYTEST_RESET_SYNC_DIR:-}" ] || return 0
  rk_gate && { ! declare -F fl_regate > /dev/null || fl_regate "$1"; } && return 0; [ -n "$RK_GATE_WHY" ] || RK_GATE_WHY=refused; rk_hook post "$1" skipped; return 1
}
# rk_sudo_l (never sudo -n true): the allowed sudo commands, read once with sudo -n -l
rk_sudo_l() { [ -n "${RK_SUDO_L+x}" ] || RK_SUDO_L=$(timeout 15 sudo -n -l 2> /dev/null); printf '%s\n' "$RK_SUDO_L"; }
# rk_can_root: root, or sudo -n -l lists ALL (sysfs rescan, tt-kmd re-probe, any ipmitool command)
rk_can_root() { [ "$(id -u)" = 0 ] || rk_sudo_l | grep -qE 'NOPASSWD: *(.*, *)?ALL *(,|$)'; }
# rk_can_glx: how tt-smi -glx_reset can run here. RK_GLX_SUDO: "" (tt-smi as is; root, sudo ALL, or sudo -n -l lists
# the ipmitool raw 0x30 0x8B forms tt-smi issues, on some clusters) or "sudo -n" (sudo -n -l lists tt-smi, on other sites).
# 1 with RK_GLX_CAN_WHY when none applies: then no reset runs at all.
rk_can_glx() {
  local smi=${TRAY_REPRO_TT_SMI:-tt-smi} l p; RK_GLX_SUDO=""; RK_GLX_RAWONLY=0; RK_GLX_CAN_WHY=""
  command -v "$smi" > /dev/null || { RK_GLX_CAN_WHY="$smi not found"; return 1; }
  rk_can_root && return 0
  rk_glx_host_ok || { RK_GLX_CAN_WHY="no root, and this host is not on the glx_reset allow list (sudo rules alone do not decide)"; return 1; }
  l=$(rk_sudo_l); p=$(command -v "$smi")
  if grep -qE "NOPASSWD:.*(${p//./\\.}|/tt-smi)( |,|$)" <<< "$l"; then RK_GLX_SUDO="sudo -n"; return 0; fi
  if grep -qiE 'NOPASSWD:.*ipmitool raw 0x30 0x8b' <<< "$l"; then RK_GLX_RAWONLY=1; return 0; fi
  RK_GLX_CAN_WHY="no root, and sudo -n -l lists neither tt-smi nor ipmitool raw 0x30 0x8B"; return 1
}
# rk_glx_host_ok: the explicit allow list for a glx_reset that runs on sudoers rules short of ALL: the host names in the
# regex below (site hosts whose sudo rules allow it). Anything else keeps NO RESET.
rk_glx_host_ok() { local h=${TRAY_REPRO_HOST:-$(hostname -s)} re; re=$(hc_site glx_hosts)
  [ -n "$re" ] && [[ $h =~ ^($re)$ ]]; }
rk_noglx() { [ "${TRAY_REPRO_NO_GLX:-}" = 1 ]; }   # TRAY_REPRO_NO_GLX=1: no tray re-power (glx_reset), warm resets still run
rk_noglx_say() {  # [$1 "<bdf> <chip> <state>" lines, default rk_snap] [$2 context]: the one line for a skipped tray re-power
  local a=${1-$(rk_snap)} m   # every chip not answering, and every chip off PCIe altogether (rk_offbus), by BDF and tray 1-4
  m=$({ echo "$a" | awk '$3 != "ok"'; rk_offbus | awk '{ print $1, $2, "off" }'; } | awk 'NF && !s[$1]++' | rk_bdfs)
  say "reset: tray re-power (tt-smi -glx_reset) skipped by TRAY_REPRO_NO_GLX=1${2:+ ($2)}: $([ -n "$m" ] && echo "still missing: $m" || echo "no chip missing")"
}
# rk_ipmi_ok <log>: one read-only in-band Get Device ID (`ipmitool mc info`, at most
# TRAY_REPRO_IPMI_CHK_S, default 10 s) before any glx_reset or in-band tray re-power. tt-smi -glx_reset resets every
# chip, then re-powers the trays in-band: on one host that re-power failed ('Get Device ID
# command failed') and left the 24 healthy chips hung too (24/32 -> 0/32); another host's in-band raw 0x30 0x8B failed
# the same way. No answer: 1, RK_IPMI_DEAD names why (the callers then skip the re-power and touch no chip).
# A check that could not be asked at all (ipmitool not found, or sudo -n refused it: 'a password is required') is a
# setup fault of this host, not of the box: 1 with RK_IPMI_SETUP set instead, never a hardware verdict.
rk_ipmi_ok() {
  local rc ipmi=${TRAY_REPRO_IPMITOOL:-ipmitool} su=() w=${TRAY_REPRO_IPMI_CHK_S:-10} n0 last
  RK_IPMI_DEAD=""; RK_IPMI_SETUP=""; [ "$(id -u)" = 0 ] || su=(sudo -n)
  if ! command -v "$ipmi" > /dev/null; then RK_IPMI_SETUP="$ipmi not found"
  else
    echo "in-band IPMI check: $ipmi mc info (timeout $w s)" >> "$1"; n0=$(grep -c '' "$1")
    pg_run timeout "$w" "${su[@]}" "$ipmi" mc info < /dev/null >> "$1" 2>&1; rc=$?
    [ $rc = 0 ] && return 0
    last=$(tail -n +$((n0 + 1)) "$1" | grep -a . | tail -1 | cut -c1-120)
    if [ ${#su[@]} -gt 0 ] && [ $rc != 124 ] && tail -n +$((n0 + 1)) "$1" | grep -aqiE '^sudo: |a password is required|terminal is required'; then
      RK_IPMI_SETUP="sudo -n refused $ipmi mc info (rc $rc: $last)"
    else RK_IPMI_DEAD="ipmitool mc info $([ $rc = 124 ] && echo "timed out after $w s" || echo "rc $rc: $last")"; fi
  fi
  if [ -n "$RK_IPMI_SETUP" ]; then say "reset: in-band IPMI check not run: setup fault ($RK_IPMI_SETUP; log $1): tray re-power skipped, no chip touched; not a hardware verdict"
  else say "reset: in-band IPMI not answering ($RK_IPMI_DEAD; log $1): tray re-power skipped, no chip touched"; fi
  return 1
}
rk_ipmi_finding() {  # after rk_ipmi_ok failed: one FINDING per chip still missing (and per tray off PCIe), by BDF and tray 1-4
  local b c st
  if [ -n "${RK_IPMI_SETUP:-}" ]; then   # a setup fault names the chips still missing, but is no hardware FINDING
    { awk '$3 != "ok" { print $1, $2 }' <<< "${RK_AFTER:-}"; rk_offbus; } | awk 'NF && !s[$1]++' | while read -r b c; do
      say "reset: SETUP FAULT chip ${c:-?} ($b, tray $(rk_tray "$b" | cut -d' ' -f1)) still missing: tray re-power not run; setup fault ($RK_IPMI_SETUP)"; done
    [ -n "$(rk_short)" ] && say "reset: SETUP FAULT $(rk_short) still missing: tray re-power not run; setup fault ($RK_IPMI_SETUP)"
    return 0
  fi
  { awk '$3 != "ok" { print $1, $2 }' <<< "${RK_AFTER:-}"; rk_offbus; } | awk 'NF && !s[$1]++' | while read -r b c; do
    say "reset: FINDING chip ${c:-?} ($b, tray $(rk_tray "$b" | cut -d' ' -f1)): in-band IPMI not answering; tray re-power skipped; hardware fault ($RK_IPMI_DEAD)"; done
  [ -n "$(rk_short)" ] && say "reset: FINDING $(rk_short): in-band IPMI not answering; tray re-power skipped; hardware fault ($RK_IPMI_DEAD)"
  return 0
}
# rk_missing: "chip <n> (<bdf>, tray <1-4>)" for each chip not ok now (a fresh recount, bounded to 30 s by RK_SNAP_DL) or off
# PCIe altogether (rk_offbus), one per line
rk_missing() {
  local b c RK_SNAP_DL=$(( ${EPOCHREALTIME/[.,]/} + 30000000 ))
  { rk_snap | awk '$3 != "ok" { print $1, $2 }'; rk_offbus; } | awk 'NF && !s[$1]++' | while read -r b c; do
    echo "chip ${c:-?} ($b, tray $(rk_tray "$b" | cut -d' ' -f1))"; done
}
# rk_timeout_say <what> <limit s> <missing>: a bounded wait ran out; one TIMEOUT line per chip still missing,
# by BDF and tray 1-4 (<missing> as rk_missing; empty: the recount found none)
rk_timeout_say() {
  local l
  if [ -z "$3" ]; then say "reset: TIMEOUT $1 after $2 s; the recount finds no chip missing"; return 0; fi
  while read -r l; do [ -n "$l" ] && say "reset: TIMEOUT $1 after $2 s: $l still missing"; done <<< "$3"
}
# rk_glx_evidence <glx log> <tt-smi> (a glx_reset with a whole tray unpowered timed out and the host
# stopped answering until a BMC power cycle): with any chip or tray missing, write the run's evidence to <glx log>_pre.txt
# and sync it BEFORE the reset runs, so it survives a host hang: the missing chips by BDF and tray 1-4, the SEL tail, the
# RT0 (retimer) readings, the reset command and its start time. Each BMC read is bounded (TRAY_REPRO_EVID_S, 20 s).
# Reads only; the glx_reset still runs after it ('must always try glx_reset').
rk_glx_evidence() {
  local f=${1%.log}_pre.txt miss sh trays t ipmi=${TRAY_REPRO_IPMITOOL:-ipmitool} su=() w=${TRAY_REPRO_EVID_S:-20}
  miss=$(rk_missing); sh=$(rk_short)
  [ -n "$miss$sh" ] || return 0
  [ "$(id -u)" = 0 ] || su=(sudo -n)
  # each tray with chips missing, by tray 1-4 and BDF; a tray at 0 of its chips (unpowered or empty) is named so, and the
  # glx_reset still runs: no path skips, refuses or delays it for an unpowered tray
  trays=$(echo "$miss" | sed -n 's/^chip [^ ]* (\([^,]*\), tray \(.\))$/\2 \1/p' | awk -v per=$(( ${RK_WANT:-32} / 4 )) '
    { n[$1]++; b[$1] = b[$1] (b[$1] == "" ? "" : " ") $2 }
    END { for (t = 1; t <= 4; t++) if (n[t] + 0 > 0)
      printf "tray %d %s: missing %s\n", t, (n[t] >= per ? "unpowered or empty (0 of " per " chips present)" : n[t] " of " per " chips missing"), b[t]
      if (n["?"] + 0 > 0) printf "tray ? %d chip(s) missing: %s\n", n["?"], b["?"] }')
  {
    echo "glx_reset evidence, written and synced before the reset"
    echo "step: ${RK_STEP:-reset}"
    echo "command: $2 -glx_reset (timeout $RK_TIMEOUT_S s)"
    echo "start: $(date '+%F %T %Z') (epoch $(date +%s))"
    echo "missing chips: $(echo "$miss" | grep . | paste -sd';' | sed 's/;/; /g')"
    [ -n "$sh" ] && echo "missing trays: $sh"
    [ -n "$trays" ] && echo "$trays"
    echo "--- SEL tail (ipmitool sel elist last 20, timeout $w s)"
    pg_run timeout "$w" "${su[@]}" "$ipmi" sel elist last 20 < /dev/null 2>&1 || echo "(SEL read rc $?)"
    echo "--- RT0 readings (ipmitool sdr elist, timeout $w s)"
    pg_run timeout "$w" "${su[@]}" "$ipmi" sdr elist < /dev/null 2>&1 | grep -a 'RT0' || echo "(no RT0 reading)"
  } > "$f" 2>&1
  sync "$f" 2> /dev/null || sync
  while read -r t; do [ -n "$t" ] && say "reset: before the glx_reset (${RK_STEP:-reset}): $t; the glx_reset runs anyway"; done <<< "$trays"
  say "reset: before the glx_reset (${RK_STEP:-reset}): missing $(echo "$miss" | grep -c .) chip(s)${sh:+, $sh}; evidence written and synced (log $f)"
  # each affected tray's CPLD latch and HSC status, read-only and bounded, before the glx_reset
  # re-powers it; a tray only short as a count (rk_short, no BDF) is still dumped. Never fails: the glx_reset always follows
  hd_dump "${1%.log}_power.txt" "${RK_STEP:-reset}" "$( { echo "$trays" | sed -n 's/^tray \([1-4]\) .*: missing \(.*\)$/\1 \2/p'
    echo "$sh" | tr ';' '\n' | sed -n 's/^ *tray \([1-4]\) .*/\1/p'; } | awk '!s[$1]++')" || :
}
rk_glx() {  # $1 a log path: the tray re-power, tt-smi -glx_reset. Sets RK_GLX_OK (1/0), RK_GLX_WHY and RK_GLX_SKIP (as rk_reset)
  local t0 rc n0 smi=${TRAY_REPRO_TT_SMI:-tt-smi}
  t0=$(date +%s); RK_GLX_OK=0; RK_GLX_RAN=0; RK_GLX_WHY=""; RK_GLX_SKIP=""; RK_IPMI_DEAD=""; RK_IPMI_SETUP=""
  if rk_noglx; then   # nothing runs, not even the gate; RK_GLX_SKIP set so no caller rescans or calls it a tray drop
    RK_GLX_SKIP="TRAY_REPRO_NO_GLX=1"; RK_GLX_WHY="skipped by TRAY_REPRO_NO_GLX=1"; rk_noglx_say; return 0
  fi
  RK_GLX_TRIED=1   # a glx_reset was tried in this step; rk_recover_ may now re-power single trays (rk_traypwr)
  if ! rk_gate; then RK_GLX_SKIP=$RK_GATE_WHY; RK_GLX_WHY="SKIPPED-GATE: $RK_GATE_WHY"
  elif ! command -v "$smi" > /dev/null; then RK_GLX_WHY="$smi not found"
  elif ! { rk_can_glx || :; : > "$1"; [ "$RK_GLX_RAWONLY" = 1 ] && { echo "in-band IPMI check skipped: sudo -n -l lists only the ipmitool raw 0x30 0x8B forms" >> "$1"; say "reset: in-band IPMI check skipped (sudo -n -l lists only the ipmitool raw 0x30 0x8B forms tt-smi -glx_reset issues)"; true; } || rk_ipmi_ok "$1"; }; then   # the check's lines open the glx log, which glx_reset appends to
    if [ -n "$RK_IPMI_SETUP" ]; then RK_GLX_WHY="setup fault ($RK_IPMI_SETUP); tray re-power not run"
    else RK_GLX_WHY="in-band IPMI not answering ($RK_IPMI_DEAD); tray re-power skipped"; fi
  elif ! rk_pre "${RK_STEP:-reset}-glx"; then RK_GLX_SKIP=$RK_GATE_WHY; RK_GLX_WHY="SKIPPED-GATE: $RK_GATE_WHY"
  else
    RK_GLX_RAN=1; rk_glx_evidence "$1" "$smi"   # evidence on disk before a reset that may hang the host
    hd_latch "${1%.log}_latch.txt" "the glx_reset (${RK_STEP:-reset})" || :   # all 8 HSCs' latches, every reset
    n0=$(grep -c '' "$1")
    echo "glx_reset command: ${RK_GLX_SUDO:+$RK_GLX_SUDO }$smi -glx_reset" >> "$1"
    pg_run timeout "$RK_TIMEOUT_S" $RK_GLX_SUDO "$smi" -glx_reset < /dev/null >> "$1" 2>&1; rc=$?; rk_hook post "${RK_STEP:-reset}-glx" $rc
    if [ $rc = 0 ]; then RK_GLX_OK=1; RK_CHIPRESET_T=$(date +%s)
    elif grep -q 'Driver not loaded with' "$1"; then RK_GLX_OK=1; RK_CHIPRESET_T=$(date +%s)   # the reset ran; tt-smi gave up waiting for 32
    elif [ $rc = 124 ]; then RK_GLX_WHY="tt-smi -glx_reset timed out after $RK_TIMEOUT_S s"
      rk_timeout_say "tt-smi -glx_reset (${RK_STEP:-reset})" "$RK_TIMEOUT_S" "$(rk_missing)"
    else RK_GLX_WHY="tt-smi -glx_reset rc $rc: $(tail -n +$((n0 + 1)) "$1" | grep -a . | tail -1 | cut -c1-160)"; fi
  fi
  say "reset: tray re-power (tt-smi -glx_reset, IPMI tray power sequence): $(if [ $RK_GLX_OK = 1 ]; then echo ok
    elif [ -n "$RK_GLX_SKIP" ]; then echo "SKIPPED-GATE ($RK_GLX_SKIP)"; elif [ -n "$RK_IPMI_DEAD$RK_IPMI_SETUP" ]; then echo "SKIPPED ($RK_GLX_WHY)"
    else echo "FAILED ($RK_GLX_WHY)"; fi) ($(( $(date +%s) - t0 )) s; log $1)"
}
rk_enum() {  # tt-umd's own enumeration count, or nothing if it cannot run
  pg_run timeout 120 "${AS[@]}" "$PY" - <<'PYEOF' 2>/dev/null | tail -1 | grep -E '^[0-9]+$'
import tt_umd
print(len(tt_umd.PCIDevice.enumerate_devices()))
PYEOF
}
rk_traysay() { local b; for b in "$@"; do printf '%s (tray %s)\n' "$b" "$(rk_tray "$b" | cut -d' ' -f1)"; done | paste -sd';' | sed 's/;/; /g'; }
# rk_finish <log> <chip ids> <BDFs tt-umd gave up on>: stock tt-umd's warm reset returns at
# the first chip whose reset marker never clears (warm_reset.cpp:277, then "Reset failed" :338), before POST_RESET of
# any chip, so every chip is left on PCIe with its node but unopenable ("Query mappings failed ... No such device";
# one host's c1:00.0 tray 3, 32/32 by config space). The other chips get their warm reset again without the
# named ones; a chip named on that try is left out of one more (2 tries). Appends to <log>, says each try by BDF and
# tray 1-4. RK_FINISH_OK 1 when the other chips' reset worked. The chips it leaves out go to the re-check and recovery.
# rk_reset skips it under stock tt-umd (rk_umd_kind): stock waits on every enumerated chip's marker, the
# named one included, so the retry only times out again.
rk_finish() {
  local named=$3 try others ex c rc n nn
  RK_FINISH_OK=0
  if [ -z "${named// /}" ]; then say "reset: tt-umd names no chip it gave up on, so no reset of the other chips (the re-check and the tray re-power decide)"; return 0; fi
  n=$(echo $2 | wc -w)
  say "reset: ${RK_STEP:-reset}: NO RESET: tt-umd gave up on $(rk_traysay $named); the other $n chips get no glx_reset, never tt-smi -r"
  echo "(NO RESET of the other $n chips; no tt-smi -r)" >> "$1"
  return 0
}
# rk_others <log> <chip ids> <BDFs tt-umd gave up on>: rk_finish, except under stock tt-umd, whose retry
# without the named chip times out on it the same way (rk_umd_kind); used by rk_reset and fl_c2.
rk_others() {
  rk_umd_kind > /dev/null   # not in $( ): it caches RK_UMD_KIND and RK_UMD_WHY for the run
  if [ "$RK_UMD_KIND" = stock ]; then
    say "reset: stock tt-umd ($RK_UMD_WHY): its glx_reset waits on every enumerated chip's reset marker, so no retry of the other chips without $(rk_traysay $3); the re-check and the tray re-power decide"
  else rk_finish "$@"; fi
}
# rk_umd_kind: "fixed" when tt-umd's WarmReset carries our fix's per-chip result (warm_reset_with_result,
# get_last_failed_pci_bdfs or rescan_pci_bdfs; it POST_RESETs every chip that came back), "stock" when it has none of them
# (device/warm_reset.cpp 0.9.12 lines 329-342 wait on the reset marker of every enumerated BDF, so rk_finish's retry
# without the named chip times out on it again: about 40 s per reset on one host's 47:00.0 tray 2),
# "unknown" when the probe gives no answer (rk_finish then runs as before). Asked once per run (RK_UMD_KIND, RK_UMD_WHY).
# TRAY_REPRO_UMD_KIND: stock or fixed without asking (tests).
rk_umd_kind() {
  local out
  if [ -z "${RK_UMD_KIND:-}" ]; then
    if [ -n "${TRAY_REPRO_UMD_KIND:-}" ]; then RK_UMD_KIND=$TRAY_REPRO_UMD_KIND; RK_UMD_WHY="TRAY_REPRO_UMD_KIND=$TRAY_REPRO_UMD_KIND"
    else   # the probe spells no API name in full, so the tests' py stubs never take it for a reset or an enumeration
      out=$(pg_run timeout 60 "${AS[@]}" "$PY" - 2> /dev/null <<'PYEOF' | grep -a '^umd ' | tail -1
import tt_umd
n = " ".join(dir(tt_umd.WarmReset))
try:
    from importlib.metadata import version as _v; ver = _v("tt-umd")
except Exception:
    ver = getattr(tt_umd, "__version__", "?")
print("umd", "fixed" if any(k in n for k in ("with_result", "last_failed", "rescan_pci")) else "stock", ver, flush=True)
PYEOF
)
      RK_UMD_KIND=$(awk '{ print $2 }' <<< "$out"); RK_UMD_KIND=${RK_UMD_KIND:-unknown}
      RK_UMD_WHY=$([ -n "$out" ] && echo "tt-umd $(awk '{ print $3 }' <<< "$out")" || echo "tt-umd probe gave no answer")
    fi
  fi
  echo "$RK_UMD_KIND"
}
# rk_reprobe <what ran> <file> (C2/C9 on one host): after a stock warm reset fails,
# tt-kmd keeps every chip's pre-reset state, so a chip on PCIe with its node still fails to open ("Query mappings failed
# ... No such device"), and the in-band tray re-power does not clear that (0/32 open); glx_reset did. Before glx_reset,
# each "unopen" chip in RK_AFTER is removed from PCI and its root port rescanned, so tt-kmd probes it afresh (a new node,
# maybe a new number: rk_remap). A chip whose tt-kmd node a process holds is left alone (a /proc scan as root,
# TRAY_REPRO_PROC; a scan that times out removes nothing). Needs root and the serving gate. Waits up to
# TRAY_REPRO_REPROBE_WAIT_S (30) for every removed chip and its node, then rk_recheck. Says each step and chip by BDF and
# tray 1-4. Sets RK_REPROBE_SUM (for RK_RECOVERY; empty when nothing was removed).
rk_reprobe() {
  local what=$1 b c t st rp held rc n=0 back=0 t0 gone list="" rps="" wait_s=${TRAY_REPRO_REPROBE_WAIT_S:-30} pr=${TRAY_REPRO_PROC:-/proc}
  local dn=/dev/tenstorrent
  local -A rpof=()
  RK_REPROBE_SUM=""
  [ -n "$(awk '$3 == "unopen"' <<< "$RK_AFTER")" ] || return 0
  if ! rk_can_root; then say "reprobe: after the $what, no tt-kmd re-probe (no root or passwordless sudo)"; return 0; fi
  held=$(rk_root timeout 20 find "$pr"/[0-9]*/fd -maxdepth 1 -lname "$dn/*" -printf '%l\n' 2> /dev/null); rc=$?
  if [ $rc = 124 ]; then say "reprobe: after the $what, the $dn holder scan timed out: no tt-kmd re-probe"; return 0; fi
  if ! rk_gate; then say "reprobe SKIPPED-GATE after the $what ($RK_GATE_WHY): nothing removed"; return 0; fi
  while read -r b c st; do [ "$st" = unopen ] || continue   # each chip's state before its remove: open error, lspci
    echo "== chip $c ($b, tray $(rk_tray "$b" | cut -d' ' -f1)) before the re-probe, $(date '+%F %T %Z')"; awk -v b="$b" '$1 == b' <<< "$RK_OPEN_ERR"
    command -v "${TRAY_REPRO_LSPCI:-lspci}" > /dev/null && rk_root timeout 10 "${TRAY_REPRO_LSPCI:-lspci}" -vvv -s "$b" 2>&1
  done <<< "$RK_AFTER" > "${2%.txt}_capture.txt"
  while read -r b c st; do
    [ "$st" = unopen ] || continue; t=$(rk_tray "$b" | cut -d' ' -f1)
    if grep -qx "$dn/$c" <<< "$held"; then say "reprobe: chip $c ($b, tray $t): $dn/$c is held by a process, not removed"; continue; fi
    rp=$(rk_rp "$b"); rpof[$b]=${rp:-none}; [ -n "$rp" ] && [[ " $rps " != *" $rp "* ]] && rps="$rps $rp"
    echo 1 | rk_root tee "$SYS/bus/pci/devices/$b/remove" > /dev/null 2>&1
    say "reprobe: echo 1 > /sys/bus/pci/devices/$b/remove (chip $c, tray $t): rc $?"; n=$((n + 1)); list="$list $b"
  done <<< "$RK_AFTER"
  [ "$n" -gt 0 ] || return 0
  for rp in $rps; do echo 1 | rk_root tee "$SYS/bus/pci/devices/$rp/rescan" > /dev/null 2>&1; say "reprobe: echo 1 > /sys/bus/pci/devices/$rp/rescan: rc $?"; done
  for b in $list; do [ "${rpof[$b]}" = none ] || continue
    echo 1 | rk_root tee "$SYS/bus/pci/rescan" > /dev/null 2>&1; say "reprobe: echo 1 > /sys/bus/pci/rescan (root port of $b unknown): rc $?"; break; done
  t0=$(date +%s)
  while :; do
    gone=""; for b in $list; do ls "$SYS/bus/pci/devices/$b/tenstorrent" 2> /dev/null | grep -q '^tenstorrent!' || gone="$gone $b"; done
    { [ -z "$gone" ] || [ $(( $(date +%s) - t0 )) -ge "$wait_s" ]; } && break; sleep 1
  done
  say "reprobe: $((n - $(echo $gone | wc -w)))/$n removed chips back on PCIe with a tt-kmd node after $(( $(date +%s) - t0 )) s${gone:+; not back: $(rk_traysay $gone)}"
  rk_remap; rk_recheck "tt-kmd re-probe after the $what" "$2"
  for b in $list; do
    c=$(awk -v b="$b" '$1 == b { print $2 }' <<< "$RK_AFTER"); st=$(awk -v b="$b" '$1 == b { print $3 }' <<< "$RK_AFTER"); t=$(rk_tray "$b" | cut -d' ' -f1)
    if [ "$st" = ok ]; then back=$((back + 1)); say "reprobe: FINDING chip ${c:-?} ($b, tray $t): opens in tt-umd only after a tt-kmd re-probe (PCI remove + rescan, after the $what)"
    else say "reprobe: FAIL chip ${c:-?} ($b, tray $t): ${st:-missing} after the $what and a tt-kmd re-probe"; fi
  done
  RK_REPROBE_SUM="tt-kmd re-probe (PCI remove + rescan) of $n chips: $back/$n back"
}
# rk_open "<bdf> <chip> <state>" lines: the health count opens each chip, not only PCIe presence.
# Sets RK_OPENED (the same lines, each "ok" chip tt-umd cannot open as "unopen"), RK_OPEN_N (chips opened), RK_OPEN_ERR
# ("<bdf> <error>" per unopen chip) and RK_OPEN_RAN (1 if the probe answered). The probe: tt_umd.PCIDevice(N) per chip
# (opens /dev/tenstorrent/N and queries its BAR mappings, the call that failed ENODEV in one run). A probe that prints
# nothing (no tt_umd; a test stub) leaves the states as they are; one its timeout cut off marks the chips it never
# reached unopen. TRAY_REPRO_OPEN_PROBE: a command run instead with the chip ids, printing the same lines (tests).
rk_open() {
  local ids out
  RK_OPENED=$1; RK_OPEN_N=""; RK_OPEN_ERR=""; RK_OPEN_RAN=""
  ids=$(awk '$3 == "ok" && $2 != "?" { print $2 }' <<< "$1" | tr '\n' ' ')
  [ -n "${ids// /}" ] || return 0
  if [ -n "${TRAY_REPRO_OPEN_PROBE:-}" ]; then out=$(pg_run timeout "${RK_OPEN_T:-120}" $TRAY_REPRO_OPEN_PROBE $ids < /dev/null 2>/dev/null)
  else out=$(pg_run timeout "${RK_OPEN_T:-120}" "${AS[@]}" "$PY" - $ids 2>/dev/null <<'PYEOF'
import sys, tt_umd
for c in sys.argv[1:]:
    try:
        d = tt_umd.PCIDevice(int(c)); del d; print("open", c, "ok", flush=True)
    except Exception as e:
        print("open", c, "err", " ".join(str(e).split())[:160], flush=True)
print("open done", flush=True)
PYEOF
); fi
  grep -q '^open [0-9]' <<< "$out" || return 0
  RK_OPEN_RAN=1
  RK_OPENED=$(O=$out awk 'BEGIN { n = split(ENVIRON["O"], l, "\n"); for (i = 1; i <= n; i++) if (split(l[i], f, " ") >= 3 && f[1] == "open") s[f[2]] = f[3] }
    $3 == "ok" && $2 != "?" && s[$2] != "ok" { $3 = "unopen" } { print }' <<< "$1")
  RK_OPEN_N=$(grep -c ' ok$' <<< "$RK_OPENED")
  RK_OPEN_ERR=$(O=$out awk 'BEGIN { n = split(ENVIRON["O"], l, "\n"); for (i = 1; i <= n; i++) if (l[i] ~ /^open [0-9]+ err /) { split(l[i], f, " "); e[f[2]] = substr(l[i], length("open " f[2] " err ") + 1) } }
    $3 == "unopen" { print $1, (($2 in e) ? e[$2] : "no answer before the open check timed out") }' <<< "$RK_OPENED")
}
# rk_traypwr <log> <trays 1-4>: the in-band re-power of only those trays, as tt-smi -glx_reset does for all
# four (ipmitool raw 0x30 0x8B <mask> 0xFF 0x3 0xF, mask 1 << (tray - 1); 9/9 on one host's tray 2). Unavailable
# (RK_TRAYPWR_WHY, nothing sent) without ipmitool or root write access to the in-band IPMI device (TRAY_REPRO_IPMI_DEV,
# default /dev/ipmi0); one tray's failure stops it (one host: 'No data available'). Sets RK_TRAYPWR_OK (1/0)
# and RK_TRAYPWR_SKIP (the gate's refusal).
rk_traypwr() {
  local t m rc ipmi=${TRAY_REPRO_IPMITOOL:-ipmitool} dev=${TRAY_REPRO_IPMI_DEV:-/dev/ipmi0} su=()
  RK_TRAYPWR_OK=0; RK_TRAYPWR_WHY=""; RK_TRAYPWR_SKIP=""; RK_IPMI_DEAD=""; RK_IPMI_SETUP=""; [ "$(id -u)" = 0 ] || su=(sudo -n)
  if ! command -v "$ipmi" > /dev/null; then RK_TRAYPWR_WHY="$ipmi not found"
  elif ! "${su[@]}" test -w "$dev" 2> /dev/null; then RK_TRAYPWR_WHY="in-band IPMI $dev not writable as root"
  elif ! rk_gate; then RK_TRAYPWR_SKIP=$RK_GATE_WHY; RK_TRAYPWR_WHY="SKIPPED-GATE: $RK_GATE_WHY"
  elif ! rk_ipmi_ok "$1"; then RK_TRAYPWR_WHY=${RK_IPMI_DEAD:+"in-band IPMI not answering ($RK_IPMI_DEAD)"}${RK_IPMI_SETUP:+"setup fault ($RK_IPMI_SETUP)"}   # nothing sent
  elif ! rk_pre "${RK_STEP:-reset}-tray"; then RK_TRAYPWR_SKIP=$RK_GATE_WHY; RK_TRAYPWR_WHY="SKIPPED-GATE: $RK_GATE_WHY"
  else
    hd_latch "${1%.log}_latch.txt" "the tray re-power of tray(s) $2 (${RK_STEP:-reset})" || :   # before every power cycle of trays
    rc=0
    for t in $2; do m=$(printf '0x%x' $((1 << (t - 1))))
      echo "tray $t: $ipmi raw 0x30 0x8B $m 0xFF 0x3 0xF" >> "$1"
      pg_run timeout 120 "${su[@]}" "$ipmi" raw 0x30 0x8B "$m" 0xFF 0x3 0xF < /dev/null >> "$1" 2>&1; rc=$?
      [ $rc = 0 ] || { RK_TRAYPWR_WHY="tray $t: rc $rc: $(grep -a . "$1" | tail -1 | cut -c1-120)"; break; }
    done
    rk_hook post "${RK_STEP:-reset}-tray" $rc; [ $rc = 0 ] && { RK_TRAYPWR_OK=1; RK_CHIPRESET_T=$(date +%s); }
  fi
}
# rk_glxchk <what ran> <file prefix> (one host's plain run: round 2's
# glx_reset FAILED ('POST_RESET failed for device /dev/tenstorrent/23'), the re-check 1 s later read 32/32 open and the run
# went on, then C3's discovery waited 180 s on ARC firmware of c1:00.0 (tray 3) and every bus-bar burst hung). After a FAILED
# glx_reset: wait until enumeration stops changing (rk_snap, the tt-kmd nodes and tt-umd's count the same for
# RK_GLX_STABLE_S, at most RK_GLX_SETTLE_S), re-check (rk_recheck), then run tt-umd topology discovery, C3's call (rk_disc),
# before any chip counts as back. A chip discovery names (ARC still waiting, a failed call) or leaves out becomes "undisc"
# (not ok: the callers stop with no load); a discovery that fails naming no chip makes every chip undisc and sets
# RK_CHECK_WHY (unknown, never a tray drop). Appends the settle and the discovery outcome to RK_RECOVERY. Files: <prefix>.txt,
# <prefix>_disc.log.
RK_GLX_SETTLE_S=${TRAY_REPRO_GLX_SETTLE_S:-90}; RK_GLX_STABLE_S=${TRAY_REPRO_GLX_STABLE_S:-15}; RK_GLX_POLL_S=${TRAY_REPRO_GLX_POLL_S:-5}
RK_DISC_T=${TRAY_REPRO_DISC_T:-180}
rk_glxchk() {
  local what=$1 f=$2 t0 ts sig last="" stable=0 s dsum n
  t0=$(date +%s); ts=$t0
  while :; do
    sig=$(rk_snap; ls -d "$SYS"/bus/pci/devices/*/tenstorrent/tenstorrent!* 2> /dev/null | wc -l; rk_enum)
    if [ "$sig" != "$last" ]; then last=$sig; ts=$(date +%s)
    elif [ $(( $(date +%s) - ts )) -ge "$RK_GLX_STABLE_S" ]; then stable=1; break; fi
    [ $(( $(date +%s) - t0 )) -ge "$RK_GLX_SETTLE_S" ] && break; sleep "$RK_GLX_POLL_S"
  done
  s=$(( $(date +%s) - t0 ))
  say "reset: settle after the failed glx_reset: enumeration $([ $stable = 1 ] && echo "unchanged for $RK_GLX_STABLE_S s" || echo "still changing at the $RK_GLX_SETTLE_S s cap") ($s s)"
  rk_recheck "$what settle" "$f.txt"
  rk_disc "${f}_disc.log"; rk_undisc "failed glx_reset"
  echo "$RK_AFTER" > "$f.txt"; n=$(grep -c ' ok$' <<< "$RK_AFTER")
  if [ "$RK_DISC_OK" = 1 ] && [ -z "$RK_DISC_BAD" ]; then dsum="discovery ok"
  elif [ "$RK_DISC_OK" = 1 ]; then dsum="discovery left out $(grep -c . <<< "$RK_DISC_BAD") chip(s)"
  else dsum="discovery FAILED ($RK_DISC_WHY)"; fi
  RK_RECOVERY="$RK_RECOVERY, settle $s s, $dsum"
  say "reset: tt-umd discovery after the failed glx_reset ($what): $dsum; $n/$(rk_total) usable$([ "$n" -lt "$(rk_total)" ] && echo "; not usable: $(awk '$3 != "ok"' <<< "$RK_AFTER" | rk_bdfs)") (log ${f}_disc.log)"
}
rk_undisc() {  # $1 what ran: after rk_disc, each chip it names becomes "undisc" in RK_AFTER; one that failed naming no
               # chip makes every ok chip undisc and sets RK_CHECK_WHY (unknown, never a tray drop)
  if [ -n "$RK_DISC_BAD" ]; then
    RK_AFTER=$(B=$RK_DISC_BAD awk 'BEGIN { n = split(ENVIRON["B"], l, "\n"); for (i = 1; i <= n; i++) { split(l[i], f, " "); s[f[1]] = 1 } }
      $3 == "ok" && ($1 in s) { $3 = "undisc" } { print }' <<< "$RK_AFTER")
  elif [ "$RK_DISC_OK" != 1 ]; then
    RK_AFTER=$(awk '$3 == "ok" { $3 = "undisc" } { print }' <<< "$RK_AFTER")
    RK_CHECK_WHY=${RK_CHECK_WHY:-tt-umd discovery after the $1 FAILED and names no chip ($RK_DISC_WHY)}
  fi
}
# rk_discpoll <log> <what ran> (one run's try 2: 32/32 open, then discovery failed on 41:00.0 (tray 2)
# 'Setting power state failed on device 8'): tt-umd discovery (rk_disc) every RK_DISC_POLL_S (10) until it finishes and
# names no chip, at most RK_DISC_SETTLE_S (90). Each failed try is logged. RK_DPOLL_K tries, RK_DPOLL_S seconds;
# RK_DPOLL_NONE=1 if the discovery printed nothing at all (no tt_umd, a test stub): states stay as they are, as rk_open.
RK_DISC_SETTLE_S=${TRAY_REPRO_DISC_SETTLE_S:-90}; RK_DISC_POLL_S=${TRAY_REPRO_DISC_POLL_S:-10}
rk_discpoll() {
  local t0; t0=$(date +%s); RK_DPOLL_K=0; RK_DPOLL_NONE=""
  while :; do
    RK_DPOLL_K=$((RK_DPOLL_K + 1)); rk_disc "$1"
    [ -s "$1" ] || { RK_DPOLL_NONE=1; break; }
    { [ "$RK_DISC_OK" = 1 ] && [ -z "$RK_DISC_BAD" ]; } && break
    [ $(( $(date +%s) - t0 + RK_DISC_POLL_S )) -gt "$RK_DISC_SETTLE_S" ] && break
    say "reset: tt-umd discovery after the $2 (try $RK_DPOLL_K): $(rk_dsum); again in $RK_DISC_POLL_S s (up to $RK_DISC_SETTLE_S s)"
    sleep "$RK_DISC_POLL_S"
  done
  RK_DPOLL_S=$(( $(date +%s) - t0 ))
}
rk_dsum() {  # the last rk_disc's outcome, each chip it named by BDF and tray 1-4
  if [ "$RK_DISC_OK" = 1 ] && [ -z "$RK_DISC_BAD" ]; then echo "discovery ok"
  elif [ "$RK_DISC_OK" = 1 ]; then echo "discovery left out $(awk '{ print $1 }' <<< "$RK_DISC_BAD" | while read -r b; do rk_traysay "$b"; done | paste -sd';' | sed 's/;/; /g')"
  else echo "discovery FAILED ($RK_DISC_WHY)$([ -n "$RK_DISC_BAD" ] && echo "; names $(awk '{ print $1 }' <<< "$RK_DISC_BAD" | while read -r b; do rk_traysay "$b"; done | paste -sd';' | sed 's/;/; /g')")"; fi
}
# rk_discsettle <what ran> <file prefix>: after a re-power left every chip on PCIe and open in tt-umd, the
# health count also waits for tt-umd discovery (rk_discpoll) before any chip counts as back. A chip it still names after
# the settle becomes "undisc" (the next recovery round, then the missing-chip path, by BDF and tray 1-4). Sets RK_DSET_SUM.
rk_discsettle() {
  local n
  rk_discpoll "${2}_disc.log" "$1"
  if [ -n "$RK_DPOLL_NONE" ]; then RK_DSET_SUM="discovery not checked (no output)"; say "reset: tt-umd discovery after the $1: no output, not checked (log ${2}_disc.log)"; return 0; fi
  rk_undisc "$1"; echo "$RK_AFTER" > "$2.txt"; n=$(grep -c ' ok$' <<< "$RK_AFTER")
  RK_DSET_SUM="$(rk_dsum) after $RK_DPOLL_S s ($RK_DPOLL_K tries)"
  say "reset: tt-umd discovery after the $1: $RK_DSET_SUM; $n/$(rk_total) usable$([ "$n" -lt "$(rk_total)" ] && echo "; not usable: $(awk '$3 != "ok"' <<< "$RK_AFTER" | rk_bdfs)") (log ${2}_disc.log)"
}
rk_disc() {  # $1 a log: tt-umd topology discovery (TopologyDiscovery.create_cluster_descriptor, as C3's fleet_py.py disc).
             # Sets RK_DISC_OK (1 if it finished), RK_DISC_WHY and RK_DISC_BAD ("<bdf> <why>" per chip RK_AFTER has ok that
             # discovery names in an error or leaves out). TRAY_REPRO_DISC_PROBE: a command run instead (tests), same output.
  local rc vis=() why
  RK_DISC_OK=0; RK_DISC_WHY=""; RK_DISC_BAD=""
  # shellcheck disable=SC2046  # one BDF per word; by UMD index (the /dev numbers masked tray 4, not tray N)
  if [ -n "${FW_NR:-}" ] && declare -F fl_vis > /dev/null; then vis=(env TT_VISIBLE_DEVICES="$(fl_vis)")   # firmware-not-ready chips masked as fl_c3 does
  elif [ "${KD_ON:-0}" = 1 ]; then vis=(env TT_VISIBLE_DEVICES="$(rk_umd $(echo "$RK_MAP" | awk '$2 != "?" { print $1 }'))"); fi
  if [ -n "${TRAY_REPRO_DISC_PROBE:-}" ]; then pg_run timeout "$RK_DISC_T" $TRAY_REPRO_DISC_PROBE < /dev/null > "$1" 2>&1
  else pg_run timeout "$RK_DISC_T" "${AS[@]}" "${vis[@]}" "$PY" - > "$1" 2>&1 <<'PYEOF'
import tt_umd
cd = tt_umd.TopologyDiscovery.create_cluster_descriptor()
for c, b in sorted(cd.get_chip_pci_bdfs().items()):
    print("disc chip", int(c), b, flush=True)
print("disc done", flush=True)
PYEOF
  fi; rc=$?
  if [ $rc = 0 ] && grep -q '^disc done' "$1"; then
    RK_DISC_OK=1
    RK_DISC_BAD=$(awk 'NR == FNR { if ($1 == "disc" && $2 == "chip") { b = tolower($4); sub(/^0000:/, "", b); s[b] = 1 }; next }
      $3 == "ok" { b = tolower($1); sub(/^0000:/, "", b); if (!(b in s)) print $1, "left out of tt-umd discovery" }' "$1" - <<< "$RK_AFTER")
    return 0
  fi
  why=$(grep -av '^disc ' "$1" | grep -aiE 'wait|fail|error|timeout|timed out' | grep -aiv 'setting power state' | tail -1 | cut -c1-160)
  RK_DISC_WHY="$([ $rc = 124 ] && echo "timed out after $RK_DISC_T s" || echo "rc $rc")${why:+: $why}"
  # chips its errors name ("... on device 16": the tt-kmd node, as RK_MAP's chip), the power-state warning left out
  RK_DISC_BAD=$(grep -av '^disc ' "$1" | grep -aiv 'setting power state' | grep -aiE 'wait|fail|error|timeout|timed out' \
    | sed -n 's/.*[Dd]evice \([0-9][0-9]*\).*/\1 &/p' | awk '!s[$1]++' | while read -r c l; do
      awk -v c="$c" -v l="$(echo "$l" | cut -c1-120)" '$2 == c && $3 == "ok" { print $1, l }' <<< "$RK_AFTER"; done)
}
# rk_recover <log prefix> <re-check file prefix> <what ran>: for each chip RK_AFTER has not ok (lost, unopen) or off PCIe, at
# most 2 rounds of: re-power only its tray in-band (rk_traypwr); if that is unavailable or fails, tt-smi -glx_reset
# (round 2 is always glx_reset; it is skipped when round 1's glx_reset ran clean); then rk_recheck and rk_rescan. The
# single-tray re-power runs only once a glx_reset was tried in the same step (RK_GLX_TRIED); else round 1 is
# the glx_reset. Logs
# each step by BDF and tray 1-4. Sets RK_GLX_OK / RK_GLX_WHY / RK_GLX_SKIP as rk_glx (the last re-power that ran) and
# RK_RECOVERY (what ran, for RK_RESETS). Files: <log prefix>.log and _tray.log, <re-check prefix>.txt and _rescan.txt;
# round 2 adds a "2" to each. <what ran> names round 1's re-check ("tray re-power"); round 2's adds " round 2".
# stock tt-umd fails the warm reset on every chip and leaves them unopenable on PCIe, so
# the chips to recover come from the recount after the warm reset with those chips re-probed first (rk_reprobe: PCI
# remove + root-port rescan, no power action); only trays that still have a missing or unopenable chip are re-powered.
# --hold (HOLD_ACTIVE=1): no re-probe before the re-power (the chips stay as they are until the re-power, as before).
rk_recover() {
  local pre=""
  if [ -n "$(awk '$3 == "unopen"' <<< "$RK_AFTER")" ]; then
    if [ "${HOLD_ACTIVE:-0}" = 1 ]; then say "reset: --hold: no tt-kmd re-probe before choosing trays; the unopenable chips stay as they are until the re-power"
    else rk_reprobe "${RK_RECOVER_PRE:-glx_reset}" "${2}_prereprobe.txt"; pre=$RK_REPROBE_SUM; fi
  fi
  rk_recover_ "$@"
  [ -n "$pre" ] && RK_RECOVERY="$pre${RK_RECOVERY:+, then $RK_RECOVERY}"
  return 0
}
rk_recover_() {
  local p=$1 q=$2 what=$3 r sfx bad trays how used="" g pw nb e pb=0
  RK_PWR_RAN=""; RK_RECOVERY=""; RK_GLX_OK=0; RK_GLX_WHY=""; RK_GLX_SKIP=""; RK_TRAYPWR_WHY=""; RK_IPMI_DEAD=""; RK_IPMI_SETUP=""   # no refusal from an earlier reset
  for r in 1 2; do
    bad=$({ awk '$3 != "ok" { print $1, $2 }' <<< "$RK_AFTER"; rk_offbus; } | awk 'NF && !s[$1]++')
    [ -n "$bad$(rk_short)" ] || break
    [ $r = 2 ] && [ "$used" = glx ] && [ "$RK_GLX_OK" = 1 ] && break   # the same full re-power that ran clean changes nothing
    [ -n "$RK_GLX_SKIP$RK_IPMI_DEAD$RK_IPMI_SETUP" ] && break
    sfx=$([ $r = 2 ] && echo 2); trays=$(awk '{ print $1 }' <<< "$bad" | while read -r b; do rk_tray "$b" | cut -d' ' -f1; done | grep -x '[1-4]' | sort -u | tr '\n' ' ')
    [ -n "$(rk_short)" ] && trays=$(echo $trays $(rk_short | grep -o 'tray [1-4]' | awk '{ print $2 }') | tr ' ' '\n' | sort -u | tr '\n' ' ')
    if [ $r = 1 ] && { [ -n "$RK_CHECK_WHY" ] || grep -q ' unread$' <<< "$RK_AFTER"; }; then   # no per-chip state
      say "reset: recovery round 1: per-chip state unreadable after the recount (${RK_CHECK_WHY:-chips unread}): re-powering all trays 1-4"
      trays="1 2 3 4 "; fi
    # progress = chips still missing, each counted once: the bad chips on the map, plus per tray the larger of its short
    # count ("N of M chips not on PCIe"; rk_short prints every short tray on one line) and its bad chips off the map.
    # With RK_WANT set, rk_offbus (in $bad) already lists the short tray's chips by BDF (they counted twice)
    nb=$(awk '{ print $1 }' <<< "$bad" | while read -r e; do [ -n "$e" ] && echo "$e $(rk_tray "$e" | cut -d' ' -f1)"; done |
      awk -v sh="$(rk_short)" 'FILENAME == ARGV[1] { if (NF) m[$1] = 1; next } NF { if ($1 in m) n++; else o[$2]++ }
        END { while (match(sh, /tray [1-4] \([0-9]+ of/)) { t = substr(sh, RSTART + 5, 1); k[t] += substr(sh, RSTART + 8, RLENGTH - 11)
            sh = substr(sh, RSTART + RLENGTH) }
          for (t in k) if (k[t] > o[t] + 0) o[t] = k[t]
          for (t in o) n += o[t]; print n + 0 }' <(echo "$RK_MAP") -)
    # under rk_end's bound: a recovery round always runs, never refused for run time. The
    # first round, and a round after one that brought a chip back, widen the step's bound by one round (RK_ROUND_S) when
    # less is left; a round after one with no progress gets no more time and ends under its own bounded wait (the timer
    # waits for a started round at most RK_ROUND_S + 60 s from its start)
    if [ -n "$RK_BOUND_ON" ]; then
      g=$(( ${RK_BOUND_ON%% *} + ${RK_BOUND_ON#* } - $(date +%s) ))
      if [ "$g" -lt "$RK_ROUND_S" ] && { [ $r = 1 ] || [ "$nb" -lt "$pb" ]; }; then
        e=$(( $(date +%s) + (g > 0 ? g : 0) + RK_ROUND_S )); echo "$e" > "$RK_ROUND_F.end"   # the timer waits for it
        RK_BOUND_ON="${RK_BOUND_ON%% *} $(( e - ${RK_BOUND_ON%% *} ))"
        say "reset: recovery round $r: $g s of the reset step's bound left; bound widened by one round ($RK_ROUND_S s, TRAY_REPRO_ROUND_S) to ${RK_BOUND_ON#* } s ($([ $r = 1 ] && echo "first round" || echo "round $((r - 1)) brought $((pb - nb)) chip(s) back"))"
      elif [ "$g" -lt "$RK_ROUND_S" ]; then
        say "reset: recovery round $r: $g s of the reset step's bound left and round $((r - 1)) brought no chip back: bound not widened; the round runs under its own bounded wait (TRAY_REPRO_ROUND_S + 60 s)"; fi
      date +%s > "$RK_ROUND_F"; fi
    pb=$nb
    say "reset: recovery round $r ($what$([ $r = 2 ] && echo " round 2")): $(rk_traysay $(awk '{ print $1 }' <<< "$bad"))${trays:+ (trays $(echo $trays | sed 's/ /, /g'))}"
    how=""; g=""; pw=""; RK_PWR_RAN=1
    # ('must always try glx_reset'): no single-tray re-power before a glx_reset was tried
    # in this step; then round 1 is the glx_reset itself (and round 2 runs only if it failed, as before)
    # --hold keeps the single-tray re-power first (a glx_reset would also reset the hung chip the hold keeps)
    # (TRAY_GLX_FIRST=0 is the offline tests' hook for the old order)
    if [ $r = 1 ] && [ -n "${trays// /}" ] && [ -z "${RK_GLX_TRIED:-}" ] && [ "${HOLD_ACTIVE:-0}" != 1 ] && [ "${TRAY_GLX_FIRST:-1}" != 0 ]; then
      say "reset: recovery round 1: glx_reset first (no glx_reset tried yet in this step), before any single-tray re-power"
    elif [ $r = 1 ] && [ -n "${trays// /}" ]; then
      rk_traypwr "${p}_tray.log" "$trays"   # ${p}_tray.log: round 1 only
      if [ "$RK_TRAYPWR_OK" = 1 ]; then how="tray re-power of trays $(echo $trays | sed 's/ /, /g') (in-band IPMI)"; RK_GLX_OK=1; pw=1
        say "reset: in-band re-power of only trays $(echo $trays | sed 's/ /, /g'): ok (log ${p}_tray.log)"
      elif [ -n "$RK_IPMI_DEAD" ]; then how="tray re-power skipped (in-band IPMI not answering: $RK_IPMI_DEAD)"   # no glx_reset either
      elif [ -n "$RK_IPMI_SETUP" ]; then how="tray re-power not run (setup fault: $RK_IPMI_SETUP)"
      elif [ -n "$RK_TRAYPWR_SKIP" ]; then RK_GLX_SKIP=$RK_TRAYPWR_SKIP; RK_GLX_WHY=$RK_TRAYPWR_WHY; how="tray re-power SKIPPED-GATE ($RK_TRAYPWR_SKIP)"
        say "reset: in-band re-power of trays $(echo $trays | sed 's/ /, /g'): SKIPPED-GATE ($RK_TRAYPWR_SKIP)"
      else say "reset: in-band re-power of only trays $(echo $trays | sed 's/ /, /g'): $([[ $RK_TRAYPWR_WHY == tray* ]] && echo FAILED || echo unavailable) ($RK_TRAYPWR_WHY); glx_reset instead"; fi
    fi
    if [ -z "$how" ]; then used=glx; g=1; rk_glx "${p}${sfx}.log"; [ "$RK_GLX_OK" = 1 ] && [ -z "$RK_GLX_SKIP" ] && pw=1; how="tray re-power (tt-smi -glx_reset)$(if [ -n "$RK_GLX_SKIP" ]; then echo " SKIPPED-GATE ($RK_GLX_SKIP)"
        elif [ -n "$RK_IPMI_DEAD" ]; then echo " skipped (in-band IPMI not answering: $RK_IPMI_DEAD)"
        elif [ -n "$RK_IPMI_SETUP" ]; then echo " not run (setup fault: $RK_IPMI_SETUP)"
        elif [ "$RK_GLX_OK" != 1 ]; then echo " FAILED ($RK_GLX_WHY)"; fi)"; fi
    RK_RECOVERY=${RK_RECOVERY:+$RK_RECOVERY, round 2 }$how
    # in-band IPMI not answering: nothing ran, so no re-check or rescan; each missing chip is a hardware fault
    if [ -n "$RK_IPMI_DEAD$RK_IPMI_SETUP" ]; then rk_ipmi_finding; break; fi
    rk_recheck "$what$([ $r = 2 ] && echo " round 2")" "$q$sfx.txt"
    [ -n "$RK_GLX_SKIP" ] || rk_rescan "$what$([ $r = 2 ] && echo " round 2")" "$q${sfx}_rescan.txt"   # never after a gate-skipped re-power
    # the in-band tray re-power left chips on PCIe but unopenable (tt-kmd's old state): re-probe them before glx_reset
    if [ $r = 1 ] && [[ $how == "tray re-power of trays"* ]]; then rk_reprobe "$what" "${q}_reprobe.txt"; RK_RECOVERY="$RK_RECOVERY${RK_REPROBE_SUM:+, $RK_REPROBE_SUM}"; fi
    # a glx_reset that FAILED: settle, then tt-umd discovery before any chip counts as back
    # --known-dropped-tray=N: a failed glx_reset that names its chips while tray N is still missing needs no
    # unmasked discovery here (tray N's dead chips fail it); tray N's named chips are the known drop, any other named chip is a
    # RESET FAIL (rk_resfail), and C3's discovery, tray N masked, still runs before any load. Tray N back: rk_glxchk
    if [ "$g" = 1 ] && [ "$RK_GLX_OK" != 1 ] && [ -z "$RK_GLX_SKIP" ] && rk_kdnamed "${p}${sfx}.log"; then
      say "reset: the failed glx_reset ($what$([ $r = 2 ] && echo " round 2")) named $(rk_umd_named "${p}${sfx}.log" | while read -r b; do echo "$b (tray $(rk_tray "$b" | cut -d' ' -f1))"; done | paste -sd' '): with --known-dropped-tray=$KD_TRAY and tray $KD_TRAY still missing, its chips are the known drop and any other is a reset failure; C3's discovery (tray $KD_TRAY masked) runs before any load"
      RK_RECOVERY="$RK_RECOVERY, named by tt-umd (--known-dropped-tray=$KD_TRAY)"
    else [ "$g" = 1 ] && [ "$RK_GLX_OK" != 1 ] && [ -z "$RK_GLX_SKIP" ] && rk_glxchk "$what$([ $r = 2 ] && echo " round 2")" "$q${sfx}_settle"; fi
    # a re-power that ran clean and left every chip open: tt-umd discovery too, settled, before any chip counts as back
    if [ "$pw" = 1 ] && [ -z "$(awk '$3 != "ok"' <<< "$RK_AFTER")$(rk_short)" ]; then
      rk_discsettle "$what$([ $r = 2 ] && echo " round 2")" "$q${sfx}_disc"; RK_RECOVERY="$RK_RECOVERY, $RK_DSET_SUM"; fi
  done   # the round's mark stays between rounds, so the bound timer never fires in the gap before the next one
  [ -n "$RK_ROUND_F" ] && rm -f "$RK_ROUND_F"   # a widened bound (.end) holds until rk_end ends
  return 0
}
rk_recheck() {  # $1 what ran, $2 a file for the result: wait up to RK_SETTLE_S for every chip, then re-check all of
                # them. Sets RK_AFTER ("<bdf> <chip> <state>" lines) and RK_CHECK_WHY (empty if the re-check is sound).
  local t0 s nok nnode en
  t0=$(date +%s)
  while :; do
    # chips not on PCIe before, or on PCIe without their tt-kmd node (chip '?'; a function can show before its node):
    # map them again
    { [ -n "${RK_WANT:-}" ] && [ "$RK_N" -lt "$RK_WANT" ]; } || echo "$RK_MAP" | grep -q ' ?$' && rk_remap
    RK_AFTER=$(rk_snap); [ "$(echo "$RK_AFTER" | grep -c ' ok$')" = "$RK_N" ] && [ "$RK_N" -ge "${RK_WANT:-0}" ] && break
    [ $(( $(date +%s) - t0 )) -ge "$RK_SETTLE_S" ] && break; sleep 2
  done
  # then open each chip; one on PCIe that tt-umd cannot open is "unopen" (not ok), until RK_SETTLE_S
  rk_open "$RK_AFTER"
  while [ -n "$RK_OPEN_RAN" ] && [ -n "$RK_OPEN_ERR" ] && [ $(( $(date +%s) - t0 )) -lt "$RK_SETTLE_S" ]; do
    sleep 5; RK_AFTER=$(rk_snap); rk_open "$RK_AFTER"; done
  s=$(( $(date +%s) - t0 )); en=$(rk_enum)
  nok=$(echo "$RK_AFTER" | grep -c ' ok$'); nnode=$(ls -d "$SYS"/bus/pci/devices/*/tenstorrent/tenstorrent!* 2>/dev/null | wc -l)
  RK_AFTER=$RK_OPENED
  say "re-check after the $1 (waited ${s} s): $nok/$(rk_total) answer PCIe config space with a tt-kmd node, $nnode tt-kmd nodes, tt-umd enumerates ${en:-?}, $([ -n "$RK_OPEN_RAN" ] && echo "$RK_OPEN_N/$(rk_total) open in tt-umd" || echo "open check not run (no answer)")"
  [ -n "$RK_OPEN_ERR" ] && say "re-check: on PCIe but tt-umd cannot open: $(while read -r b e; do echo "$b (tray $(rk_tray "$b" | cut -d' ' -f1)) chip $(awk -v b="$b" '$1 == b { print $2 }' <<< "$RK_MAP"): $e"; done <<< "$RK_OPEN_ERR" | paste -sd';' | sed 's/;/; /g')"
  echo "$RK_AFTER" > "$2"
  # the recount's wait ran out with chips still missing: name each by BDF and tray 1-4
  [ "$s" -ge "$RK_SETTLE_S" ] && [ "$nok" != "$(rk_total)" ] && rk_timeout_say "recount after the $1" "$RK_SETTLE_S" \
    "$({ awk '$3 != "ok" { print $1, $2 }' <<< "$RK_AFTER"; rk_offbus; } | awk 'NF && !s[$1]++' | while read -r b c; do echo "chip ${c:-?} ($b, tray $(rk_tray "$b" | cut -d' ' -f1))"; done)"
  RK_CHECK_WHY=""
  if echo "$RK_AFTER" | grep -q ' unread$'; then RK_CHECK_WHY="re-check could not read chips $(echo "$RK_AFTER" | awk '$3 == "unread" { print $2 }' | rk_list)"
  elif [ -z "$en" ]; then RK_CHECK_WHY="re-check could not run tt-umd enumeration"
  # tt-umd lists /dev/tenstorrent, and a chip lost from the bus keeps its node (seen on a cluster node: dev=32, 8 reading
  # ffff), so its count is checked against the tt-kmd nodes; ok vs lost comes from config space plus the node
  elif [ "$en" != "$nnode" ]; then RK_CHECK_WHY="re-check disagrees: $nnode tt-kmd nodes, tt-umd enumerates $en"
  fi
}
# rk_rescan <what ran> <file>, right after rk_recheck (one host's 47:00.0 tray 2 stayed off the bus
# after glx_reset, rc 1 at 31/32, until a rescan of its root port 40:01.7 gave 32/32). The chips it rescans for (review
# rescan): each chip RK_AFTER has lost, and each chip off PCIe altogether (rk_offbus), also one already missing before any
# reset. Wait up to TRAY_REPRO_RESCAN_WAIT_S (60) for each root port's Data Link Layer Link Active (LnkSta bit 13), check
# the serving gate again, write 1 to the root port's rescan, then rk_recheck again (a live per-chip config read). Needs
# root. A gate refusal after the wait writes nothing ('rescan SKIPPED-GATE', RK_RESCAN_GSKIP). A chip back only after the
# rescan goes to RK_RESCANNED ("<bdf>|<root port>|<what>"; rk_resfail reports it, never a silent pass).
rk_root() { if [ "$(id -u)" = 0 ]; then "$@"; else sudo -n "$@"; fi; }
rk_bus() {  # PCI address (0000:88:00.0 or 88:00.0) -> its bus byte; rc 1, no output, when it has none
  [[ $1 =~ ^([0-9a-fA-F]{4,}:)?([0-9a-fA-F]{2}):[0-9a-fA-F]{2}\.[0-7]$ ]] && echo "${BASH_REMATCH[2]}"
}
rk_rp() {  # chip BDF -> its root port, from the map taken at the start (fault_map: a chip off the bus loses its path),
           # else its sysfs path, else the bridge whose secondary bus is the chip's bus (a chip off the bus since boot)
  local rp bus; rp=$(awk -v b="$1" '$1 == b { print $2; exit }' <<< "${FC_MAP:-}")
  [ -n "$rp" ] || { [ -n "$1" ] && [ -e "$SYS/bus/pci/devices/$1" ] && rp=$(basename "$(dirname "$(readlink -f "$SYS/bus/pci/devices/$1")")"); }
  # no bus byte: rk_tray's ERROR line names it, and no grep runs (grep read '-1' as an option and then stdin)
  [ -n "$rp" ] || { if bus=$(rk_bus "$1"); then rp=$(grep -lx -e "$(( 16#$bus ))" "$SYS"/bus/pci/devices/*/secondary_bus_number 2>/dev/null | head -1)
    [ -n "$rp" ] && rp=$(basename "$(dirname "$rp")"); else rk_tray "$1" > /dev/null; fi; }
  case $rp in ????:??:??.?) echo "$rp" ;; esac
}
rk_offbus() {  # "<bdf> <chip or ?>" for each chip off PCIe altogether: FC_MAP chips now gone, and with RK_WANT set each
               # Galaxy chip address (bus & 0xF0 = 00, 40, c0, 80; chips on buses +1..+8, as kd_bdfs) not on PCIe.
               # The known dropped tray (--known-dropped-tray) is left out.
  local b c u k
  { awk 'NF >= 3 { print $1, $3 }' <<< "${FC_MAP:-}"
    [ -n "${RK_WANT:-}" ] && for u in 0 64 192 128; do for k in 1 2 3 4 5 6 7 8; do printf '0000:%02x:00.0 ?\n' $(( u + k )); done; done
  } | awk '!s[$1]++' | while read -r b c; do
    [ "${KD_ON:-0}" = 1 ] && [ "$(rk_tray "$b")" = "$KD_TRAY" ] && continue
    # off the bus, or still listed but dead (config ffff, no tt-kmd node) and not in RK_MAP: a lost chip too
    [ "$(rk_state "$b")" = ok ] || echo "$b $c"
  done
}
rk_rescan() {
  local what=$1 b c rp t v rps="" left nl t0 st cand setpci=${TRAY_REPRO_SETPCI:-setpci} wait_s=${TRAY_REPRO_RESCAN_WAIT_S:-60}
  local -A chips=() lv=() rpof=() cof=()
  cand=$({ awk '$3 == "lost" || $3 == "unopen" { print $1, $2 }' <<< "$RK_AFTER"; rk_offbus; } | awk 'NF && !s[$1]++')
  while read -r b c; do
    [ -n "$b" ] || continue
    rp=$(rk_rp "$b"); t=$(rk_tray "$b" | cut -d' ' -f1)
    if [ -z "$rp" ]; then say "rescan: chip $c ($b, tray $t): root port unknown, not rescanned"; continue; fi
    [ -n "${chips[$rp]:-}" ] || rps="$rps $rp"; chips[$rp]="${chips[$rp]:+${chips[$rp]}; }chip $c ($b, tray $t)"; rpof[$b]=$rp; cof[$b]=$c
  done <<< "$cand"
  [ -n "$rps" ] || return 0
  if [ "${HOLD_ACTIVE:-0}" = 1 ]; then for rp in $rps; do say "rescan: --hold: root port $rp (${chips[$rp]}): lost chip left as it is, nothing written"; done; return 0; fi   # hang-debug hold: no action on a lost chip
  if ! rk_can_root; then say "rescan: after the $what, root port rescan skipped (no root, and sudo -n -l does not allow the sysfs rescan):$rps"; return 0; fi
  if ! command -v "$setpci" > /dev/null; then say "rescan: $setpci not found: LnkSta not polled"; fi
  t0=$(date +%s); left=$rps
  while command -v "$setpci" > /dev/null; do   # every root port at once, so a whole tray waits 60 s, not 8 x 60 s
    nl=""
    for rp in $left; do
      v=$(rk_root "$setpci" -s "$rp" CAP_EXP+0x12.w 2>/dev/null | tr -d ' \n'); lv[$rp]=$v
      if [[ $v =~ ^[0-9a-fA-F]{1,4}$ ]] && (( 16#$v & 0x2000 )); then
        say "rescan: root port $rp (${chips[$rp]}): LnkSta 0x$v, DLL Link Active after $(( $(date +%s) - t0 )) s"
      else nl="$nl $rp"; fi
    done
    left=$nl; { [ -z "$left" ] || [ $(( $(date +%s) - t0 )) -ge "$wait_s" ]; } && break; sleep 1
  done
  for rp in $left; do say "rescan: root port $rp (${chips[$rp]}): DLL Link Active not seen in $wait_s s (LnkSta ${lv[$rp]:+0x}${lv[$rp]:-unreadable}), rescanned anyway"; done
  if ! rk_gate; then   # the wait can take wait_s: the gate again, right before the writes
    for rp in $rps; do say "rescan SKIPPED-GATE after the $what ($RK_GATE_WHY): root port $rp (${chips[$rp]}): nothing written"; done
    RK_RESCAN_GSKIP="${RK_RESCAN_GSKIP:+$RK_RESCAN_GSKIP and }the $what"; return 0
  fi
  RK_RESCAN_RAN="${RK_RESCAN_RAN:+$RK_RESCAN_RAN and }the $what"   # where it ran, for the Reset: line
  for rp in $rps; do
    echo 1 | rk_root tee "$SYS/bus/pci/devices/$rp/rescan" > /dev/null 2>&1
    say "rescan: echo 1 > /sys/bus/pci/devices/$rp/rescan (${chips[$rp]}): rc $?"
  done
  rk_recheck "root port rescan after the $what" "$2"
  for b in "${!rpof[@]}"; do
    c=$(awk -v b="$b" '$1 == b { print $2 }' <<< "$RK_AFTER"); c=${c:-${cof[$b]}}; t=$(rk_tray "$b" | cut -d' ' -f1)
    st=$(awk -v b="$b" '$1 == b { print $3 }' <<< "$RK_AFTER")   # the final recount decides
    if [ "$st" = ok ]; then
      say "rescan: FINDING chip $c ($b, tray $t): reset needed root-port rescan (root port ${rpof[$b]}, after the $what)"
      RK_RESCANNED=${RK_RESCANNED:+$RK_RESCANNED$'\n'}$b\|${rpof[$b]}\|$what
    elif [ "$st" = unopen ] || [ "$st" = undisc ]; then
      say "rescan: chip $c ($b, tray $t): on PCIe with its tt-kmd node after the $what and a rescan of root port ${rpof[$b]}, but tt-umd cannot $([ "$st" = unopen ] && echo open || echo discover) it (not missing; the recovery decides)"
    elif [ "$st" = unread ]; then say "rescan: chip $c ($b, tray $t): state unread after the $what and a rescan of root port ${rpof[$b]}"
    else say "rescan: FAIL chip $c ($b, tray $t): still missing after the $what and a rescan of root port ${rpof[$b]}"; fi
  done
}
rk_rescanned_what() { awk -F'|' 'NF { print $1 }' <<< "${RK_RESCANNED:-}" | while read -r b; do
  echo "chip $(awk -v b="$b" '$1 == b { print $2 }' <<< "$RK_MAP") ($b, tray $(rk_tray "$b" | cut -d' ' -f1))"; done | paste -sd';' | sed 's/;/; /g'; }
rk_back() {  # $1 chip list ("0-3,7"), $2 "<bdf> <chip> <state>" lines: the chips of $1 that are ok in $2, as a list
  local c
  for c in $(echo "$1" | tr ',' '\n' | awk -F- 'NF { e = (NF > 1 ? $2 : $1) + 0; for (i = $1 + 0; i <= e; i++) print i }'); do
    echo "$2" | awk -v c="$c" '$2 == c && $3 == "ok" { print c }'; done | rk_list
}
rk_bdfs() {  # "<bdf> <chip> <state>" lines -> "chip 25 (0000:42:00.0, tray 2); ..." (BDF and tray 1-4)
  while read -r b c st; do [ -n "$b" ] && echo "chip $c ($b, tray $(rk_tray "$b" | cut -d' ' -f1))"; done | paste -sd';' | sed 's/;/; /g'
}
rk_kdnamed() {  # $1 a failed glx_reset's log: true when --known-dropped-tray=N is set, tray N is still missing in RK_AFTER (its dead
  # chips fail unmasked discovery) and tt-umd names a chip there; a tray N back gets rk_glxchk's settle and discovery
  { [ "${KD_ON:-0}" = 1 ] || [ "${KD_PEND:-0}" = 1 ]; } && [ -n "${KD_TRAY:-}" ] && [ -n "$(rk_kdmiss "$RK_AFTER")" ] && [ -n "$(rk_umd_named "$1")" ]
}
rk_kdmiss() {  # $1 "<bdf> <chip> <state>" lines: tray KD_TRAY's chips that do not answer
  awk '$3 != "ok" { print $1 }' <<< "$1" | while read -r b; do [ "$(rk_tray "$b" | cut -d' ' -f1)" = "$KD_TRAY" ] && echo "$b"; done
}
rk_umd_named() {  # $1 a warm reset log: the BDFs tt-umd gave up on (its reset marker never cleared, warm_reset.cpp:277)
  { grep -ao 'Timeout waiting for reset marker to clear for device [0-9a-fA-F:.]*' "$1" 2>/dev/null | awk '{ print $NF }' | sed 's/[.:,;]*$//'   # drop sentence punctuation after the BDF
    grep -aoE 'Timeout waiting for [0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7] to come back' "$1" 2>/dev/null | awk '{ print $4 }'
    # tt-smi -glx_reset names a chip by device path ("POST_RESET failed for device /dev/tenstorrent/N"): N is the chip index of RK_MAP
    grep -aoE "(failed for device |Permission denied: ')/dev/tenstorrent/[0-9]+" "$1" 2>/dev/null | sed 's|.*/||' | while read -r n; do awk -v n="$n" '$2 == n { print $1 }' <<< "${RK_MAP:-}"; done
  } | sort -u
}
# rk_slowmark <warm reset log> <step> (one host's c4:00.0): tt-umd logs 'Still waiting on reset marker to
# clear for device <bdf> (<ms> ms elapsed, <ms> ms budget left)' once a chip's marker is still set after 5 s of its 10 s
# budget. One "RESET WARN <bdf> (tray N)" line per chip with how long it waited; in a --mode run a finding, so fl_report
# counts it under that tray as a warn (never a fail). A marker timeout stays the warm reset's FAILED (rk_resfail, rk_umd_named).
rk_slowmark() {
  local b ms bud l
  for b in $(grep -aoE 'Still waiting (on|for) reset marker to clear for device [0-9a-fA-F:.]*' "$1" 2>/dev/null | awk '{ print $NF }' | sed 's/[.:,;]*$//' | sort -u); do
    ms=$(grep -aF "reset marker to clear for device $b " "$1" | grep -aoE '\([0-9]+ ms elapsed' | tr -dc '0-9\n' | sort -n | tail -1)
    bud=$(grep -aF "reset marker to clear for device $b " "$1" | grep -aoE '\([0-9]+ ms elapsed, [0-9]+ ms budget' | head -1 | awk '{ print substr($1, 2) + $4 }')
    l="RESET WARN $b (tray $(rk_tray "$b")): reset marker slow to clear after the glx_reset ($2): still set after $(awk -v m="${ms:-5000}" 'BEGIN { printf "%.1f", m / 1000 }') s"
    if rk_umd_named "$1" | grep -qxF "$b"; then l="$l, then timed out$([ -n "$bud" ] && awk -v m="$bud" 'BEGIN { printf " at %.1f s", m / 1000 }') (glx_reset FAILED on it)"
    else l="$l, cleared before tt-umd's$([ -n "$bud" ] && awk -v m="$bud" 'BEGIN { printf " %.0f s", m / 1000 }') timeout"; fi
    if [ -n "${FLEET:-}" ] && declare -F fl_find > /dev/null; then fl_find "$l"; else say "$l"; fi
  done
}
# rk_resfail <which reset> <"<bdf> <chip> <state>" after the warm reset> <BDFs tt-umd named> <1 if it said ok> <why>
#            <"<bdf> <chip> <state>" after the tray re-power, empty if none ran> <why the re-power was skipped, or empty>
# A warm reset that ran and left a chip missing (or failed on one) is a FAIL even when the tray re-power brings it back
# (one host's 47:00.0): one "RESET FAIL <bdf> (tray N) chip C" line per chip with the count after the warm
# reset and what the re-power did. Appended to RK_RESET_FAILS (exit 14 when nothing worse); this call's in RK_RESET_NEW.
# rk_pwrlost (one host's tray 3): the trays 1-4 without power at this reset: cpld_drop's CPLD read
# says 12 V lost (TRAY_12V_LOST, BUSBAR_SIG, HOTSWAP_OC in old captures) or reg1e (vr_pg/alive) 00 (TRAY_OFF), or rk_recover's in-band re-power of the
# tray was refused. One "<tray> <evidence>" line per tray.
rk_pwrlost() {
  local t e
  for t in 1 2 3 4; do
    e=$(sed -n "s/^tray $t bus [^:]*: reg1e=\([0-9a-f]*\) dbg5f=\([0-9a-f?]*\) [^>]*-> \([A-Z0-9_]*\).*/\1 \2 \3/p" "${OUT:-/nonexistent}/cpld_reg1e_dbg5f.txt" 2> /dev/null |
      awk '$1 == "00" || $3 ~ /^(TRAY_12V_LOST|BUSBAR_SIG|HOTSWAP_OC)$/ { printf "CPLD %s%s %s", $1, ($2 ~ /^[0-9a-f]+$/ ? "/" $2 : ""), $3; exit }')
    [ -n "$e" ] && [[ ${RK_TRAYPWR_WHY:-} == "tray $t: rc "* ]] && e="$e; its in-band re-power refused (${RK_TRAYPWR_WHY#tray $t: })"
    [ -n "$e" ] && echo "$t $e"
  done
}
# rk_backstep : the recovery step after which the chips were back, from RK_RECOVERY (what ran) and
# its re-probe count: tt-kmd re-probe when it brought chips back, tray re-power when one ran, else the warm reset alone.
rk_backstep() {
  local rp=${RK_RECOVERY:-}
  if [[ $rp == *"tt-kmd re-probe"* ]] && grep -qE 'tt-kmd re-probe \(PCI remove \+ rescan\) of [0-9]+ chips: [1-9][0-9]*/[0-9]+ back' <<< "$rp"; then
    [[ $rp == *"tray re-power"* ]] && echo "tt-kmd re-probe (PCI remove + rescan, after the tray re-power)" || echo "tt-kmd re-probe (PCI remove + rescan)"
  elif [ -n "${RK_PWR_RAN:-}" ] && [[ $rp == *"tray re-power"* ]] && ! [[ $rp == *"SKIPPED"* || $rp == *"skipped"* || $rp == *"not run"* ]]; then echo "tray re-power"
  else echo "the glx_reset (no tray re-power ran)"; fi
}
rk_resfail() {
  local ph=$1 aw=$2 named=$3 wok=$4 why=$5 af=$6 skip=$7 n nf tot bad b c st what l r kdn="" fs pw pt err t e nch=0
  RK_RESET_NEW=""; n=$(echo "$aw" | grep -c ' ok$'); tot=$(rk_total)
  pw=$(rk_pwrlost); pt=$(awk '{ print $1 }' <<< "$pw" | paste -sd' ')
  # --known-dropped-tray: tt-umd timing out on tray N's chips is the known drop; at the start reset (KD_PEND) only
  # while tray N is still missing after it (af, else rk_end_'s settle after a failed glx_reset, else aw): a tray N back 8/8 is the normal run, its named chips RESET FAIL
  if [ "${KD_ON:-0}" = 1 ] || { [ "${KD_PEND:-0}" = 1 ] && [ -n "$(rk_kdmiss "${af:-${RK_SETTLED_AW:-$aw}}")" ]; }; then
    kdn=$(for b in $named; do [ "$(rk_tray "$b")" = "$KD_TRAY" ] && echo "$b"; done)
    named=$(for b in $named; do [ "$(rk_tray "$b")" != "$KD_TRAY" ] && echo "$b"; done)
    # the start reset (KD_PEND): with every other chip answering, tray N's missing chips are the known drop kd_settle names
    if [ "${KD_PEND:-0}" = 1 ] && [ -z "$(awk '$3 != "ok" { print $1 }' <<< "$aw" | while read -r b; do [ "$(rk_tray "$b")" = "$KD_TRAY" ] || echo "$b"; done)" ]; then
      aw=$(while read -r b l; do [ -n "$b" ] && [ "$(rk_tray "$b")" != "$KD_TRAY" ] && echo "$b $l"; done <<< "$aw"); n=$(echo "$aw" | grep -c ' ok$'); tot=$(echo "$aw" | grep -c .)
    fi
  fi
  bad=$({ echo "$aw" | awk '$3 != "ok" { print $1 }'; echo "$named" | tr ' ' '\n'; } | grep . | sort -u)
  if [ -z "$bad" ]; then
    { [ "$wok" = 1 ] || [ -n "$kdn" ]; } && [ "$n" -ge "$tot" ] && {
      [ "$wok" = 1 ] || { RK_RESET_KDONLY=1; say "reset: the $([[ ${RK_RESET_LBL:-} == glx_reset* ]] && echo glx_reset || echo "glx_reset") ($ph) failed only on the known dropped tray $KD_TRAY ($(echo $kdn)); all $n chips answer"; }
      return 0; }
    RK_RESET_NEW="RESET UNATTRIBUTED FAIL: the glx_reset ($ph) FAILED ($why) and names no chip; $n/$tot after the glx_reset"
  else
    [ -n "$af" ] && nf=$(echo "$af" | grep -c ' ok$')
    for b in $bad; do
      c=$(echo "$aw" | awk -v b="$b" '$1 == b { print $2 }'); st=$(echo "$aw" | awk -v b="$b" '$1 == b { print $3 }')
      [ "$st" = ok ] && what="the glx_reset ($ph) FAILED on it (${why:-tt-umd names it}), it answers after it" \
        || what="missing after the glx_reset ($ph)"
      [ "$st" = unopen ] && what="lost after glx_reset ($ph): on PCIe with its tt-kmd node, but tt-umd cannot open it"
      fs=$(echo "$af" | awk -v b="$b" '$1 == b { print $3 }')   # its state after the tray re-power
      l="RESET FAIL $b (tray $(rk_tray "$b" | cut -d' ' -f1)) chip ${c:-?}: $what; $n/$tot after the glx_reset"
      r=$(awk -F'|' -v b="$b" '$1 == b { print "root port " $2 ", after the " $3; exit }' <<< "${RK_RESCANNED:-}")
      err=$(awk -v b="$b" '$1 == b { $1 = ""; print substr($0, 2) }' <<< "${RK_OPEN_ERR:-}" | cut -c1-120)
      if [ "$st" = ok ]; then :
      elif [ -n "$pt" ] && [[ " $pt " != *" $(rk_tray "$b" | cut -d' ' -f1) "* ]] && [ "$([ -n "$af" ] && echo "$fs" || echo "$st")" = unopen ] \
          && grep -qiE 'no such device|ENODEV' <<< "$err"; then   # on PCIe, ENODEV: the reset chain broke on a tray without power
        nch=$((nch + 1))
        l="RESET WARN $b (tray $(rk_tray "$b" | cut -d' ' -f1)) chip ${c:-?}: not opened: reset chain broken by tray $(echo $pt | sed 's/ / and tray /g') power loss ($ph; on PCIe with its tt-kmd node, tt-umd: $err); not this tray's board"
      elif [ -n "$r" ]; then l="$l, reset needed root-port rescan ($r): back, $(echo "$RK_AFTER" | grep -c ' ok$')/$tot"
      elif [ -n "$af" ] && [ "$fs" != ok ] && [ -n "${RK_IPMI_DEAD:-}" ]; then   # no re-power ran, nothing else was touched
        l="$l, in-band IPMI not answering; tray re-power skipped; hardware fault ($RK_IPMI_DEAD)"
      elif [ -n "$af" ] && [ "$fs" != ok ] && [ -n "${RK_IPMI_SETUP:-}" ]; then   # the check could not be asked
        l="$l, tray re-power not run; setup fault ($RK_IPMI_SETUP); no hardware verdict"
      elif [ -n "$af" ] && [ "$fs" = ok ] && [[ $RK_RECOVERY == *FAILED* ]]; then l="$l, back after ${RK_RECOVERY}: $nf/$tot"   # name the steps that ran
      elif [ -n "$af" ] && [ "$fs" = ok ]; then l="$l, recovered by $(rk_backstep) $nf/$tot"   # the step that ran and left every chip back
      elif [ -n "$af" ] && [ "$fs" = undisc ] && [ "${RK_DISC_OK:-0}" != 1 ] && ! awk -v b="$b" '$1 == b { f = 1 } END { exit !f }' <<< "${RK_DISC_BAD:-}"; then
        # discovery failed naming no chip (rk_undisc marked every chip): back on PCIe and opens, recovery unknown as RESULT says
        l="$l, back on PCIe and opens after ${RK_RECOVERY:-the tray re-power}, but tt-umd discovery FAILED naming no chip (${RK_DISC_WHY:-discovery failed}): recovery unknown"
      elif [ -n "$af" ] && [ "$fs" = undisc ]; then   # on PCIe and opens after a FAILED glx_reset, but tt-umd discovery fails on it
        l="RESET FAIL $b (tray $(rk_tray "$b" | cut -d' ' -f1)) chip ${c:-?}: lost after glx_reset, not recovered by tray re-power/glx_reset ($ph; on PCIe and opens in tt-umd, but tt-umd discovery fails on it: $(awk -v b="$b" '$1 == b { $1 = ""; print substr($0, 2); f = 1; exit } END { if (!f) print w }' w="${RK_DISC_WHY:-discovery failed}" <<< "${RK_DISC_BAD:-}" | cut -c1-120)); $n/$tot after the glx_reset, $nf/$tot after ${RK_RECOVERY:-the tray re-power}"
      elif [ -n "$af" ] && [ "$fs" = unopen ]; then   # on PCIe, never usable again: the chip, never the load
        l="RESET FAIL $b (tray $(rk_tray "$b" | cut -d' ' -f1)) chip ${c:-?}: lost after glx_reset, not recovered by tray re-power/glx_reset ($ph; on PCIe with its tt-kmd node, but tt-umd cannot open it: $(awk -v b="$b" '$1 == b { $1 = ""; print substr($0, 2) }' <<< "${RK_OPEN_ERR:-}" | cut -c1-120)); $n/$tot after the glx_reset, $nf/$tot after ${RK_RECOVERY:-the tray re-power}"
      elif [ -n "$af" ]; then l="$l, still missing after the tray re-power ($nf/$tot): tray drop; lost after glx_reset, not recovered by tray re-power/glx_reset"
      elif [[ $skip == SKIPPED-GATE* || $skip == TRAY_REPRO_NO_GLX* ]]; then l="$l, tray re-power skipped ($skip): INCOMPLETE, not proven a tray drop"
      else l="$l, tray re-power skipped (${skip:-not run}): still missing"; fi
      RK_RESET_NEW=${RK_RESET_NEW:+$RK_RESET_NEW$'\n'}$l
    done
    [ "$nch" -gt 0 ] && while read -r t e; do   # the root cause, once, under the tray without power
      RK_RESET_NEW="POWER FAIL tray $t ($(echo "$bad" | while read -r b; do [ "$(rk_tray "$b" | cut -d' ' -f1)" = "$t" ] && printf '%s ' "${b#0000:}"; done | sed 's/ $//')): reset chain broken by its power loss ($ph): $e; $nch chip(s) of the other trays stay on PCIe but do not open (ENODEV)"$'\n'$RK_RESET_NEW
    done <<< "$pw"
  fi
  # the step's reset was tt-smi -glx_reset, not the warm reset: say so in every line
  [[ ${RK_RESET_LBL:-} == glx_reset* ]] && RK_RESET_NEW=${RK_RESET_NEW//warm reset/glx_reset}
  RK_RESET_FAILS=${RK_RESET_FAILS:+$RK_RESET_FAILS$'\n'}$RK_RESET_NEW
}
rk_resfail_what() {  # RK_RESET_FAILS -> "chip 14 (0000:47:00.0, tray 2); ..." for the RESULT line
  echo "$RK_RESET_FAILS" | sed -n 's/^RESET FAIL \([^ ]*\) (\(tray [^)]*\)) chip \([^:]*\):.*/chip \3 (\1, \2)/p' | sort -u | paste -sd';' | sed 's/;/; /g'
}
# rk_eacces <"<bdf> <chip> <state>" lines> (one box: 32 false 'RESET FAIL ... replace the board' lines
# when the user could not open /dev/tenstorrent): 0 when the only fault is this user's access to the device nodes. Every
# chip answers PCIe with its tt-kmd node (state ok or unopen), each unopen chip's tt-umd error is a permission error, and
# a permission error was seen (in those opens, or in the log of a glx_reset that FAILED, $RK_RESET_LOG, settled sa4).
# Classification only: no reset runs here (settled sa3/sa5).
RK_DEVPE="Permission denied: '/dev/tenstorrent/|Errno 13.*/dev/tenstorrent|/dev/tenstorrent[^ ]*: Permission denied"   # a permission error on the device nodes only
rk_eacces() {
  local pe='Permission denied|Errno 13|EACCES'
  [ -n "$1" ] || return 1
  awk '$3 != "ok" && $3 != "unopen" { f = 1 } END { exit f }' <<< "$1" || return 1
  [ "$(grep -cE ' (ok|unopen)$' <<< "$1")" -ge "$(rk_total)" ] || return 1
  if [ -n "${RK_OPEN_ERR:-}" ]; then ! grep -avqE "$pe" <<< "$RK_OPEN_ERR"; return; fi
  [ "${RK_RESET_STAT:-}" = FAILED ] && grep -aqE "$RK_DEVPE" "${RK_RESET_LOG:-/dev/null}" 2> /dev/null   # all open: only a glx_reset refused on the device nodes
}
rk_eacces_why() {  # the one setup error line: who cannot open /dev/tenstorrent
  if [ -z "${RK_OPEN_ERR:-}" ] && [ "${RK_RESET_STAT:-}" = FAILED ]; then
    echo "tt-smi -glx_reset refused: $(grep -aE "$RK_DEVPE" "${RK_RESET_LOG:-/dev/null}" 2> /dev/null | head -1 | cut -c1-160); uid $(id -u) ($(id -un)), groups $(id -Gn | tr ' ' ','); all $(rk_total) chips are on PCIe and open"; return
  fi
  echo "uid $(id -u) ($(id -un)), groups $(id -Gn | tr ' ' ','), cannot open /dev/tenstorrent (Permission denied); all $(rk_total) chips are on PCIe with a tt-kmd node"
}
# rk_eacces_stop <step>: the setup error, no hardware cause line and no repair line, then NOT RUN (exit 6, as a start that
# could not load). The fix is the device nodes' group or mode, or the run user's groups, not the board.
rk_eacces_stop() {
  local w; w=$(rk_eacces_why)
  say "$1: setup error, not a lost chip: $w; nothing was loaded"
  echo "SETUP ERROR: $w; give this user access to /dev/tenstorrent (its group or mode), then rerun. No chip is lost."
  declare -F lk_stop > /dev/null && lk_stop
  local what="$(id -un) cannot open /dev/tenstorrent"
  [ -n "${RK_OPEN_ERR:-}" ] || [ "${RK_RESET_STAT:-}" != FAILED ] || what="tt-smi -glx_reset refused on /dev/tenstorrent"
  vd_say "NOT RUN (setup error: $what: Permission denied)"
  echo "Couldn't start - setup error: $what (Permission denied)"; exit 6
}
# rk_end <log dir> <chips the test reported lost> <1 if the test reported a loss, else 0> [when, default "during the test"]
# Prints the reset and re-check lines. Sets RK_RC (0 none, 5 chip drop only, 3 tray drop, 1 unknown, 9 INCOMPLETE: a
# reset the serving gate skipped left the result unproven), RK_WHAT (for
# the caller's RESULT line), RK_RESETS (which reset ran and what came back), RK_CHIPS and RK_TRAYS (the verdict; print
# all three last with rk_last).
# rk_end: the whole step under its own bound: a watchdog sends USR1 after RK_STEP_S (or less: what the run's
# budget leaves less RS_TAIL_S) and stops the step's processes (bash runs the trap only once the foreground child is gone);
# rk_bound_hit then prints the RESET FAIL line, k/N by BDF and tray 1-4 and the RESULT line, and exits 9.
RK_BOUND_ON=""; RK_BOUND_PID=""; RK_ROUND_F=""; RK_ROUND_S=${TRAY_REPRO_ROUND_S:-240}   # a recovery round: 143 s on record
# rk_bound_names: the reset step was cut at its bound: every chip still missing is a hardware fault,
# named by BDF and tray 1-4 on the same RESET FAIL line run 1 prints, then one "FAIL tray N" line per tray with the exact
# failure under it. Read-only (PCIe state reads): acts on no chip, so --hold changes nothing. rk_bound_hit exits 9 after it.
rk_bound_names() {
  local miss b c st t n u
  miss=$({ RK_SNAP_DL=$(( ${EPOCHREALTIME/[.,]/} + RS_COUNT_MS * 1000 )) RK_OD_S=0.5 rk_snap | awk '$3 != "ok" { print $1, $2, $3 }'; rk_offbus; } | awk 'NF && !s[$1]++')
  [ -n "$miss" ] || return 0
  while read -r b c st; do
    echo "RESET FAIL $b (tray $(rk_tray "$b" | cut -d' ' -f1)) chip ${c:-?}: still missing (${st:-off the bus}) when the ${RK_STEP:-} reset step was cut at its ${RK_BOUND_ON#* } s bound during ${RS_STAGE:-${RK_PHASE:-the reset}}; hardware fault, not recovered by tray re-power/rescan"
  done <<< "$miss"
  for u in $(while read -r b c; do rk_tray "$b" | cut -d' ' -f1; done <<< "$miss" | grep -x '[1-4]' | sort -u); do
    n=$(while read -r b c; do [ "$(rk_tray "$b" | cut -d' ' -f1)" = "$u" ] && printf '%s ' "${b#0000:}"; done <<< "$miss" | sed 's/ $//')
    echo "FAIL tray $u ($(wc -w <<< "$n") finding(s))"
    echo "  RESET FAIL tray $u: chip(s) $n missing after the ${RK_STEP:-} reset step hit its bound; hardware fault (tray lost: the in-band re-power and rescan did not bring them back in time)"
  done
}
rk_bound_hit() {
  [ -n "$RK_BOUND_ON" ] || return 0
  trap '' USR1 TERM INT; declare -F pg_reap > /dev/null && pg_reap; HB_PID=""; rs_kids
  exec 1>&"$RK_BOUND_FD" 2>&1   # the trap can fire inside a step's own redirect (a reset step > reset.log): back to the run's output
  echo "RESET FAIL: ${RK_STEP:-the} reset step still running after $(( $(date +%s) - ${RK_BOUND_ON%% *} )) s (bound ${RK_BOUND_ON#* } s, ${RK_BOUND_VAR:-TRAY_REPRO_RESET_STEP_S}); stopped during ${RS_STAGE:-${RK_PHASE:-the reset}}"
  echo "after the ${RK_STEP:-} reset step (stopped at its bound): $(rs_count)"
  rk_bound_names
  [ -z "$RS_DONE" ] && { echo "RESULT: INCOMPLETE: ${RK_STEP:-the} reset step hit its ${RK_BOUND_ON#* } s bound; chips back: $(rs_count)${SR_TAG:-}"
    RS_DONE=1; }
  exit 9
}
trap rk_bound_hit USR1
rk_end() {
  local b l t rc
  rk_bound "${4:-}"
  t=$(date +%s); RK_BOUND_ON="$t $b"; RK_ROUND_F=$1/.rk_round; rm -f "$RK_ROUND_F" "$RK_ROUND_F.end"; exec {RK_BOUND_FD}>&1
  # the timer holds no run lock (fd 9) or output, and its sleep dies with it: an orphaned sleep kept run.lock for $b s.
  # A recovery round that started (RK_ROUND_F holds its start) is never cut: the timer waits for it, at most
  # RK_ROUND_S + 60 s from its start (inside RS_TAIL_S)
  ( trap - USR1 INT EXIT; exec 9<&-; s=""; p0=$!; trap 'kill $s 2> /dev/null; [ "$!" = "$p0" ] || kill "$!" 2> /dev/null; exit 0' TERM
    sleep "$b" < /dev/null > /dev/null 2>&1 & s=$!; wait "$s" || exit 0
    while { [ -s "$RK_ROUND_F" ] && [ "$(date +%s)" -lt $(( $(cat "$RK_ROUND_F" 2> /dev/null || echo 0) + RK_ROUND_S + 60 )) ]; } \
      || [ "$(date +%s)" -lt "$(cat "$RK_ROUND_F.end" 2> /dev/null || echo 0)" ]; do
      sleep 1 < /dev/null > /dev/null 2>&1 & s=$!; wait "$s" || exit 0; done
    kill -USR1 $$ 2> /dev/null; rs_kids ) & RK_BOUND_PID=$!
  hb_start "reset and re-check"; rk_tmark "${RK_RESET_TIMES:-}" begin "${RK_STEP:-reset}"; rk_end_ "$@"; rc=$?; rk_tmark "${RK_RESET_TIMES:-}" end "${RK_STEP:-reset}"; hb_stop
  RK_BOUND_ON=""; rm -f "$RK_ROUND_F" "$RK_ROUND_F.end"; rs_stop_pid "$RK_BOUND_PID"; RK_BOUND_PID=""; exec {RK_BOUND_FD}>&-
  echo "after the ${RK_STEP:-} reset step ($(( $(date +%s) - t )) s of its $b s bound): $(rs_count)"
  l=$(rs_left); [ -n "$l" ] && [ "$l" -lt 0 ] && echo "run budget: overrun $(( -l )) s past TRAY_REPRO_RUN_S=$RS_RUN_S at the end of the ${RK_STEP:-} reset step (the numeric pinpoint and the C9 reset are never cut for the budget)"
  # a setup error (EACCES on the device nodes) stops the run after the reset's own report (RESET FAIL by
  # BDF and tray for a failed glx_reset, settled sa3), before any verdict line: start, C9 and every rk_end reset
  ! rk_eacces "${RK_AFTER:-}" || rk_eacces_stop "${RK_STEP:-$([ "${4:-}" = "at the start" ] && echo start || echo end)}"
  return $rc
}
rk_bound() {  # in rk_end ($1 its "when"): b, the reset step's bound (s), and RK_BOUND_VAR, the setting that set it
  local l st=${RK_STEP:-$([ "${1:-}" = "at the start" ] && echo start || echo end)}; b=$RK_STEP_S
  # no run budget (full): a multi-round recovery must finish (worst on record 414 s); a larger bound only
  RK_BOUND_VAR=TRAY_REPRO_RESET_STEP_S
  [ "${RS_RUN_S:-0}" -gt 0 ] 2> /dev/null || { b=${TRAY_REPRO_RESET_STEP_FULL_S:-3600}; RK_BOUND_VAR=TRAY_REPRO_RESET_STEP_FULL_S; }
  # the end reset (C9, or rk_end's "end" without --mode) is never shortened for the budget: its reset starts and
  # finishes; rk_end logs any overrun. Other steps (C2, C9b) still end by the budget less RS_TAIL_S (at least 60 s).
  case $st in C9|end) return 0 ;; esac
  rs_need "$st reset" "$b" "$RS_TAIL_S"   # after a fault C2/C9b get their whole bound
  l=$(rs_left); [ -n "$l" ] && [ $(( l - RS_TAIL_S )) -lt "$b" ] && b=$(( l - RS_TAIL_S )) && RK_BOUND_VAR="TRAY_REPRO_RUN_S less TRAY_REPRO_TAIL_S" && [ "$b" -lt 60 ] && b=60
  return 0
}
rk_end_() {
  local out=$1 seen=${3:-0} when=${4:-during the test} lost back notok why skip="" smi=${TRAY_REPRO_TT_SMI:-tt-smi} missing
  local short0 short gone gskip="" glab="" glxran="" aw RK_SETTLED_AW="" st named af="" ph=${RK_PHASE:-$([ "${4:-}" = "at the start" ] && echo "start reset" || echo "end reset")}; RK_GLXRAN=""
  local RK_STEP=${RK_STEP:-$([ "${4:-}" = "at the start" ] && echo start || echo end)}   # rk_hook's step
  declare -F mg_recover > /dev/null && mg_recover   # chips the MCE guard removed from PCI: root port rescan first
  RK_RESET_NEW=""; RK_RESCANNED=""; RK_RESCAN_RAN=""; RK_RESCAN_GSKIP=""; RK_GLX_TRIED=""; RK_BEFORE=$(rk_snap); short0=$(rk_short)
  [ "$when" = "at the start" ] && { rk_open "$RK_BEFORE"; RK_BEFORE=$RK_OPENED; }   # on PCIe but unopenable is lost
  lost=$(echo "$2 $(echo "$RK_BEFORE" | awk '$3 != "ok" { print $2 }')" | rk_list)
  [ -n "$lost$short0" ] && seen=1
  say "before the reset: $(echo "$RK_BEFORE" | grep -c ' ok$')/$(rk_total) chips answer${lost:+; lost $when: chips $lost}${short0:+; not on PCIe: $short0}"
  RK_RECOVERY_FIRST=0; [ "$when" = "at the start" ] && RK_RECOVERY_FIRST=1   # pre-load recovery: glx_reset first
  rk_reset "$out/reset.log"; RK_RECOVERY_FIRST=0; MG_T0=$(date +%s)   # the guard looks only at ARC timeouts after this reset
  rk_recheck "$RK_RESET_LBL" "$out/recheck.txt"; aw=$RK_AFTER; named=$(rk_umd_named "$RK_RESET_LOG")
  [ -n "$RK_RESET_SKIP" ] || rk_rescan "$RK_RESET_LBL" "$out/recheck_rescan.txt"   # chips still off the bus: wait for the link, rescan the root port (never after a gate-skipped reset)
  # a glx_reset that ran and FAILED: settle, then tt-umd discovery before any chip counts as back (as rk_recover_); the recount after a failed glx_reset is the health check before the test
  # --known-dropped-tray=N with tray N still missing and named by tt-umd: no unmasked discovery (tray N's dead chips fail it), as rk_recover_
  # the outcome is rk_reset's RK_RESET_STAT (sa3). The settle is recovery time: the bound timer waits for it as for a
  # started round (RK_ROUND_F), so the settle never trips the bound before recovery runs
  if [ "$RK_RESET_STAT" = FAILED ] && ! rk_kdnamed "$RK_RESET_LOG"; then
    st=$(date +%s); [ -n "$RK_BOUND_ON" ] && [ -n "$RK_ROUND_F" ] && echo "$st" > "$RK_ROUND_F"
    RK_RECOVERY=""; rk_glxchk "$RK_RESET_LBL" "$out/recheck_settle"; RK_SETTLED_AW=$RK_AFTER
    # aw stays the re-check right after the failed glx_reset, so each chip it left missing is RESET FAIL by BDF and tray even when
    # the settle brings it back (sa3); only chips tt-umd discovery names join it; a discovery that names no chip leaves aw as the re-check had it (RK_CHECK_WHY keeps it unknown)
    aw=$(B="$RK_DISC_BAD" awk 'BEGIN { n = split(ENVIRON["B"], l, "\n"); for (i = 1; i <= n; i++) { split(l[i], f, " "); if (f[1] != "") u[f[1]] = 1 } }
      ($1 in u) && $3 == "ok" { $3 = "undisc" } { print }' <<< "$aw")
  fi
  why=$RK_CHECK_WHY   # a failed reset still ran: only a re-check that cannot run (or be trusted) leaves it unknown
  back=$(rk_back "$lost" "$RK_AFTER"); notok=$(echo "$RK_AFTER" | awk '$3 != "ok"'); short=$(rk_short)
  RK_RESETS="$RK_RESET_LBL$([ "$RK_RESET_OK" = 1 ] || echo " FAILED ($RK_RESET_WHY)")"
  if [ -n "$RK_RESET_SKIP" ]; then RK_RESETS="reset SKIPPED-GATE ($RK_RESET_SKIP)"; gskip="the reset"; glab=SKIPPED-GATE; fi
  [ "${RK_RESET_NONE:-0}" = 1 ] && RK_RESETS="NO RESET ($RK_RESET_WHY)"
  if [ -n "$notok$short" ]; then   # chips still missing: the tray re-power, if it can be started here
    # chips missing and no reset could run (never tt-smi -r): INCOMPLETE, never a tray drop
    [ "${RK_RESET_NONE:-0}" = 1 ] && ! rk_noglx && { gskip="${gskip:+$gskip and }the reset (glx_reset could not run)"; glab=${glab:+$glab, }"NO RESET"; }
    if rk_noglx; then skip="TRAY_REPRO_NO_GLX=1"; gskip="${gskip:+$gskip and }the tray re-power"; glab=${glab:+$glab, }TRAY_REPRO_NO_GLX=1
    elif ! rk_gate; then skip="SKIPPED-GATE: $RK_GATE_WHY"; gskip="${gskip:+$gskip and }the tray re-power"
    elif ! rk_can_glx; then skip="$RK_GLX_CAN_WHY"; fi
    if [ "$skip" = TRAY_REPRO_NO_GLX=1 ]; then rk_noglx_say "$RK_AFTER"
      RK_RESETS="$RK_RESETS${back:+ (chips $back came back)}; tray re-power skipped (TRAY_REPRO_NO_GLX=1)"; back=""
    elif [ -n "$skip" ]; then
      say "reset: tray re-power (tt-smi -glx_reset) skipped: $skip"
      RK_RESETS="$RK_RESETS${back:+ (chips $back came back)}; tray re-power skipped ($skip)"; back=""
    else
      RK_RESETS="$RK_RESETS${back:+ (chips $back came back)}"
      rk_recover "$out/reset_glx" "$out/recheck_glx" "tray re-power"; af=$RK_AFTER   # up to 2 rounds
      why=$RK_CHECK_WHY
      back=$(rk_back "$(echo "$notok" | awk '{ print $2 }' | rk_list)" "$RK_AFTER")
      RK_RESETS="$RK_RESETS, then $RK_RECOVERY"
      if [ -n "$RK_GLX_SKIP" ]; then gskip="${gskip:+$gskip and }the tray re-power"; elif [ -n "$RK_PWR_RAN" ]; then glxran=1; RK_GLXRAN=1; fi
    fi
  fi
  if [ -z "$RK_RESET_SKIP" ] && [ "${RK_RESET_NONE:-0}" != 1 ]; then   # the reset ran: its own outcome
    rk_resfail "$ph" "$aw" "$named" "$RK_RESET_OK" "$RK_RESET_WHY" "$af" "$skip"
    [ -n "$RK_RESET_NEW" ] && say "reset: $(echo "$RK_RESET_NEW" | paste -sd'|' | sed 's/|/ | /g')"
  fi
  [ -n "$RK_RESCAN_RAN" ] && RK_RESETS="$RK_RESETS; root port rescan after $RK_RESCAN_RAN${RK_RESCANNED:+ (reset needed root-port rescan: $(rk_rescanned_what))}"
  [ -n "$RK_RESCAN_GSKIP" ] && RK_RESETS="$RK_RESETS; root port rescan SKIPPED-GATE after $RK_RESCAN_GSKIP"
  missing=$(echo "$RK_AFTER" | awk '$3 != "ok" { print $2 }' | rk_list); short=$(rk_short)
  gone=${missing:+chips $missing}; [ -n "$short" ] && gone=${gone:+$gone; }$short
  if [ -n "$gone" ]; then RK_RESETS="$RK_RESETS; ${back:+chips $back came back, }still missing: $gone"
  else RK_RESETS="$RK_RESETS; ${back:+chips $back came back, }all $RK_N chips answer"; fi
  RK_CHIPS=${lost:-None}; [ -n "$short0" ] && RK_CHIPS=${lost:+$lost; }$short0
  [ "$seen" = 1 ] && [ -z "$lost$short0" ] && RK_CHIPS="not named (the test reported a chip loss without naming the chips)"
  # the verdict counts only what is still missing after the last reset that ran (the re-check above)
  if [ -n "$why" ]; then   # a re-check that could not run or cannot be trusted: never a tray drop
    missing=$(echo "$RK_AFTER" | awk '$3 == "lost"' | rk_trays); [ -n "$short" ] && missing=${missing:+$missing; }$short
    RK_RC=1; RK_WHAT="unknown ($why)"; RK_TRAYS="unknown ($why${missing:+; still missing: $missing})"; return
  fi
  if [ -n "$gskip" ] && [ -n "$gone" ] && [ "$glxran" = 1 ]; then   # the warm reset was gate-skipped, but the stronger
    # tray re-power ran (failed or not) and the chips stay missing: a tray drop, each chip by BDF and tray 1-4
    RK_TRAYS=$(echo "$RK_AFTER" | awk '$3 != "ok"' | while read -r b c st; do echo "tray $(rk_tray "$b" | cut -d' ' -f1): chip $c ($b)"; done | paste -sd';' | sed 's/;/; /g')
    [ -n "$short" ] && RK_TRAYS=${RK_TRAYS:+$RK_TRAYS; }$(echo "$short" | sed 's/ (UBB[0-3]: / (/g')
    RK_RC=3; RK_WHAT="tray drop: $RK_TRAYS; still missing after the tray re-power (the glx_reset was SKIPPED-GATE)"; return
  fi
  if [ -n "$gskip" ] && { [ -n "$gone" ] || [ "$seen" != 1 ]; }; then   # a reset the gate skipped: neither a pass nor a
    # tray drop (no reset ran that could prove the loss unrecoverable); a chip drop that came back stays a chip drop
    RK_RC=9
    if [ -n "$gone" ]; then
      missing=$(echo "$RK_AFTER" | awk '$3 != "ok"' | rk_bdfs); [ -n "$short" ] && missing=${missing:+$missing; }$short
      RK_TRAYS="unknown (still missing: $missing; not proven a tray drop: $gskip did not run)"
      RK_WHAT="INCOMPLETE: still missing: $missing; not proven a tray drop: $gskip did not run (${glab:-SKIPPED-GATE})"
    else RK_TRAYS=None; RK_WHAT="INCOMPLETE: no loss, but $gskip did not run (${glab:-SKIPPED-GATE}), so recovery was not exercised"; fi
    return
  fi
  if [ -n "$gone" ] && [ -n "${RK_IPMI_SETUP:-}" ]; then   # the re-power never ran for a setup fault: unknown, never a tray drop
    missing=$(echo "$RK_AFTER" | awk '$3 != "ok"' | rk_bdfs); [ -n "$short" ] && missing=${missing:+$missing; }$short
    RK_RC=1; RK_TRAYS="unknown (still missing: $missing; setup fault: $RK_IPMI_SETUP)"
    RK_WHAT="unknown (setup fault: $RK_IPMI_SETUP; tray re-power not run; still missing: $missing)"; return
  fi
  if [ -n "$gone" ]; then   # still missing after every reset that could run, failed or not: only a power cycle
    RK_TRAYS=$(echo "$RK_AFTER" | awk '$3 != "ok"' | rk_trays); [ -n "$short" ] && RK_TRAYS=${RK_TRAYS:+$RK_TRAYS; }$short
    RK_RC=3; RK_WHAT="tray drop: $RK_TRAYS${RK_IPMI_DEAD:+; in-band IPMI not answering; tray re-power skipped; hardware fault ($RK_IPMI_DEAD)}"; return
  fi
  RK_TRAYS=None   # every lost chip came back after a reset (warm, or the tray re-power): a chip drop
  if [ "$seen" = 1 ]; then RK_RC=5; RK_WHAT="chip drop only (chips ${lost:-${short0:-not named}}; every chip answers after the reset)"
  elif [ -n "$RK_RESET_NEW" ]; then RK_RC=14; _w=$(rk_resfail_what); [ -n "$_w" ] || _w="RESET UNATTRIBUTED FAIL (the reset failed and names no chip)"; RK_WHAT="${RK_RESET_LBL%% (*} FAILED: $_w (every chip answers after the reset)"
  else RK_RC=0; RK_WHAT="no loss"; fi
}
rk_end_loss() {  # $1 reset.log $2 "<bdf> <chip> <state>" after the last reset. Prints "chip lost at end reset: tray 1 0000:01:00.0
                 # (glx_reset marker timeout)", one per missing chip: a loss only the end reset caused is no try's
  local named b c st why; named=$(rk_umd_named "${RK_RESET_LOG:-$1}")
  echo "$2" | awk '$3 != "ok"' | while read -r b c st; do
    if echo "$named" | grep -qx "$b"; then why="glx_reset marker timeout"; else why="${RK_RESET_WHY:-still missing after the reset}"; fi
    echo "tray $(rk_tray "$b" | cut -d' ' -f1) $b ($why)"
  done | paste -sd';' | sed 's/;/; /g; s/^/chip lost at end reset: /'
}
rk_last() { [ -n "${RK_RESETS:-}" ] && echo "Reset: $RK_RESETS"; echo "Chips dropped: $RK_CHIPS"; echo "Trays dropped: $RK_TRAYS"; }
# rk_abort_lines: an early abort (rc 6) still prints the drop lines and a FAIL tray line per tray with the exact failure
# under it, each chip by BDF and tray
rk_abort_lines() {
  local t f
  rk_last
  for t in 1 2 3 4; do
    f=$(grep -E "^(RESET (FAIL|WARN)|POWER FAIL) .*tray $t[ )]" <<< "${RK_RESET_FAILS:-}")
    [ -n "$f" ] && { echo "FAIL tray $t ($(wc -l <<< "$f") finding(s))"; sed 's/^/  /' <<< "$f"; }
  done
  return 0
}
# rk_drop_now <stage/chunk label>: the in-run monitor (fl_watch, fl_lost) saw chips off PCIe: at once, for each chip not yet written,
# one "DROP FAIL <bdf> (tray N): ..." line to the run log (say) and to $OUT/drops_early.txt, flushed, so a SIGKILL of run.sh
# (the tt-device broker's) still leaves it. Read-only: touches no chip, so hold mode is unchanged. The end-of-run
# summary stays as it is and does not count it: its lines are "Chips dropped"/"FAIL tray", these are not.
rk_drop_now() {
  local b c st l; local f=${OUT:-.}/drops_early.txt
  rk_snap | awk '$3 != "ok"' | while read -r b c st; do
    grep -qs "^DROP FAIL $b " "$f" && continue
    l="DROP FAIL $b (tray $(rk_tray "$b" | cut -d' ' -f1)): off PCIe, first seen by the in-run monitor at $(date '+%F %T %Z') in ${1:-unknown stage}"
    echo "$l" >> "$f"; sync "$f" 2> /dev/null; say "$l"
  done
  return 0
}
lost_trays() {  # trays with a chip that no longer answers (a powered-off tray reads ffff)
  rk_snap | awk '$3 != "ok"' | while read -r b c st; do echo "$(rk_tray "$b") $c"; done |
    awk '{ k = "tray " $1; if (!(k in n)) o[++m] = k; n[k]++ } END { for (i = 1; i <= m; i++) printf "%s%s: %d chips", (i > 1 ? ", " : ""), o[i], n[o[i]] }'
}
# cpld_drop <log dir> <1 if the test reported a loss>: after a loss (or with chips not answering), before rk_end's
# reset, read every tray CPLD's reg 0x1E (cpld_5f.py: reads only, no CPLD index write, no power or reset action;
# CPLD debug) into <log dir>/cpld_reg1e_dbg5f.txt, so a whole-tray drop names its CPLD state live (ff ALIVE,
# 00 TRAY_OFF). Logs its SUMMARY line; never changes the run's result or its reset.
cpld_drop() {
  [ "${2:-0}" = 1 ] || [ -n "$(rk_snap | awk '$3 != "ok"')" ] || return 0
  local s; s=$(timeout 120 python3 ./cpld_5f.py --out "$1" 2>&1 | grep -a -m1 '^SUMMARY ')
  say "${s:-cpld: no reg1e/dbg5f summary (see $1/cpld_reg1e_dbg5f.txt)}"
  # the drop's BMC heat snapshot and CPLD alive/vr_pg, unless a --mode drop capture already took one;
  # sourced (the tests: set -u, no FLEET, no lib/bmc_heat.sh) it skips the snapshot and does not abort
  declare -F bh_drop > /dev/null || return 0
  if [ -z "${FLEET:-}" ] || ! ls "${FL_OUT:-$1/fleet}"/capture_*/cpld_vrpg.txt > /dev/null 2>&1; then bh_drop "$1" drop; fi
}
# rk_start <log dir>, after rk_map with RK_WANT=32: chips missing before any load get the same reset and re-check as at
# the end ("a tray drop is only one that's not recoverable by any available reset, ie one that only a
# power cycle can resolve"); the test starts only once every chip is back. Else it names what is still missing and
# exits 6 with nothing loaded, the last line "Couldn't start - Chips are already dropped" and none of the verdict lines
# ("it shouldn't print the same three lines, so we can differentiate").
rk_start() {
  local lost why uo=""; lost=$(lost_trays)
  if [ "$RK_N" = "$RK_WANT" ] && [ -z "$lost" ]; then rk_open "$(rk_snap)"   # and every chip opens
    ! rk_eacces "$RK_OPENED" || rk_eacces_stop start   # a permission-denied open is a setup error, before any reset
    [ -z "$RK_OPEN_ERR" ] && { declare -F lk_settle > /dev/null && lk_settle; return 0; }
    uo=$(rk_traysay $(awk '{ print $1 }' <<< "$RK_OPEN_ERR"))
  fi
  say "start check: $RK_N Blackhole chips on PCIe${lost:+, not answering: $lost}${uo:+, on PCIe but tt-umd cannot open: $uo}; reset and re-check before any load"
  mkdir -p "$1/start"; rk_end "$1/start" "" 0 "at the start"; RK_START_RAN=1   # fleet's C3 pre ran before it: stale
  declare -F lk_settle > /dev/null && lk_settle   # the host links after the start reset decide C0's held lines
  [ "$RK_RC" = 14 ] && RK_RC=5   # every chip back after the tray re-power; its RESET FAIL lines stay (RK_RESET_FAILS)
  [ "$RK_RC" = 5 ] && rk_map   # every chip back: map again, a tray re-power's driver reload can renumber them
  declare -F fl_c0settle > /dev/null && fl_c0settle "$1/start"   # --mode: C0's missing chips, recovered or still missing
  if [ "$RK_RC" != 5 ] || [ "$RK_N" != "$RK_WANT" ] || [ -n "$(lost_trays)" ]; then
    ! rk_eacces "${RK_AFTER:-}" || rk_eacces_stop start
    [ "${KD_PEND:-0}" = 1 ] && kd_settle "$([ "$RK_RC" = 9 ] && echo "${RK_WHAT:-gate skipped the start reset}")" && { say "start check: the reset at the start: $RK_RESETS; starting on the other 24 chips"; return 0; }
    case $RK_RC in
      3) why="still missing after every reset that could run: $RK_TRAYS" ;;
      9) why="$RK_WHAT" ;;
      1) why="re-check $RK_TRAYS" ;;
      *) lost=$(lost_trays); why="$RK_N/$RK_WANT chips on PCIe after the reset${lost:+, not answering: $lost}" ;;
    esac
    say "start check: the reset at the start: $RK_RESETS"; say "start check: $why; nothing was loaded"
    declare -F fl_c3pend > /dev/null && fl_c3pend "start reset left chips missing"   # --mode: pre's held in-box port lines
    rk_abort_lines
    declare -F lk_stop > /dev/null && lk_stop; vd_say "NOT RUN (couldn't start: chips already dropped)"; echo "Couldn't start - Chips are already dropped"; exit 6
  fi
  [ "${KD_PEND:-0}" = 1 ] && { KD_PEND=0; say "tray $KD_TRAY present after the start reset (8 of 8 chips answer): --known-dropped-tray=$KD_TRAY is not needed; the normal run on all 32 chips"; }
  say "start check: all $RK_WANT chips answer and open in tt-umd after the reset ($RK_RESETS); starting the tries"
}
kd_bdfs() {  # tray 1-4 -> its 8 chip addresses (as fl_absent: bus & 0xF0 = 00, 40, c0, 80; chips on buses +1..+8)
  local u k; u=$(case $1 in 1) echo 0 ;; 2) echo 64 ;; 3) echo 192 ;; 4) echo 128 ;; esac)
  for k in 1 2 3 4 5 6 7 8; do printf '%02x:00.0\n' $(( u + k )); done
}
# kd_check, after rk_map with RK_WANT=32 and --known-dropped-tray=N, before any reset: tray N fully present =
# the normal run on all 32 ("tray N present", never refused). Any tray N chip missing (1-8 of 8, or a tray that flaps):
# KD_PEND=1, so rk_start's start reset and recount run first and kd_settle decides after it.
kd_count() {  # "<bdf> <chip> <state>" lines -> KD_OK (tray N chips that answer), KD_OTH (the other trays' chips that answer)
  local b st; KD_OK=0; KD_OTH=0
  while read -r b _ st; do [ -n "$b" ] && [ "$st" = ok ] || continue
    if [ "$(rk_tray "$b")" = "$KD_TRAY" ]; then KD_OK=$((KD_OK + 1)); else KD_OTH=$((KD_OTH + 1)); fi
  done <<< "$1"
}
kd_miss() {  # "<bdf> <chip> <state>" lines -> " c1:00.0 (lost) c8:00.0 (absent)": tray N's chips that do not answer
  local b st; for b in $(kd_bdfs "$KD_TRAY"); do
    st=$(echo "$1" | awk -v b="0000:$b" '$1 == b { print $3 }'); [ "$st" = ok ] || printf ' %s (%s)' "$b" "${st:-absent}"; done
}
kd_check() {
  local s; s=$(rk_snap); kd_count "$s"
  if [ "$KD_OK" = 8 ]; then
    say "tray $KD_TRAY present (8 of 8 chips answer): --known-dropped-tray=$KD_TRAY is not needed; the normal run on all 32 chips"; return 0; fi
  KD_PEND=1
  say "known drop: tray $KD_TRAY has $((8 - KD_OK)) of 8 chips missing at the start:$(kd_miss "$s"); $KD_OTH of the other 24 answer; the start reset and recount first"
}
# kd_settle [why no reset ran], in fl_start/rk_start when chips are still missing with KD_PEND=1: only tray N's chips
# missing = the known fault (KD_ON=1: one KNOWN DROP line naming each BDF, tray N off the map, RK_WANT=24, tray N's own
# RESET FAIL lines dropped), return 0; anything else (another chip missing, a re-check that cannot be trusted): 1.
kd_settle() {
  local s k when="after the start reset"; [ -n "${1:-}" ] && when="at the start (no reset ran: $1)"
  [ "${RK_RC:-0}" = 1 ] && [ -z "${1:-}" ] && return 1
  rk_map; s=$(rk_snap); kd_count "$s"
  [ "$KD_OTH" = 24 ] && [ "$KD_OK" != 8 ] || return 1
  k=$((8 - KD_OK))
  KD_LINE="KNOWN DROP tray $KD_TRAY: $k of 8 chips missing $when:$(kd_miss "$s" | sed 's/ (lost)//g'); --known-dropped-tray=$KD_TRAY: tray $KD_TRAY is not tested, the other 24 chips are"
  say "$KD_LINE"; echo "$(date '+%F %T %Z') $KD_LINE" >> "$OUT/fault.txt"
  # the start reset's fails on tray N (and its unattributed fail: the other 24 all answer) are the known drop
  RK_RESET_FAILS=$(echo "${RK_RESET_FAILS:-}" | grep -av "^RESET FAIL [^ ]* (tray $KD_TRAY)\|^RESET UNATTRIBUTED FAIL: the \(warm reset\|glx_reset\) (start reset)")
  local f=${FL_OUT:-/nonexistent}/findings.txt   # C0's lines on tray N alone are the known drop's evidence, not a FAIL
  [ -s "$f" ] && awk -v n="$KD_TRAY" '{ t = $0; o = 0; k = 0
      while (match(t, /tray [1-4]([^0-9]|$)/)) { k++; if (substr(t, RSTART + 5, 1) != n) o = 1; t = substr(t, RSTART + RLENGTH) }
      if (k && !o) sub(/^PREFLIGHT FAIL /, "PREFLIGHT KNOWN-DROP "); print }' "$f" > "$f.kd" && mv "$f.kd" "$f"
  if declare -F fl_find > /dev/null && [ -n "${FL_OUT:-}" ]; then   # one line per missing chip of tray N, BDF and tray
    local b st; while read -r b st; do   # a chip C0 already named missing (its line above) keeps that one line (
      # F3); a line of another kind (a converted firmware line) does not name it missing
      [ -n "$b" ] && ! grep -q "^PREFLIGHT KNOWN-DROP ${b//./\\.} (tray $KD_TRAY) .*\(not answering\|absent from PCIe\)" "$f" 2> /dev/null || continue
      [ "$st" = absent ] && st="absent from PCIe" || st="not answering ($st)"   # kd_miss's state: absent from the map, or lost/unread
      fl_find "PREFLIGHT KNOWN-DROP $b (tray $KD_TRAY) $st: --known-dropped-tray=$KD_TRAY, not tested"
    done < <(kd_miss "$s" | sed 's/ \([^ ]*\) (\([a-z]*\))/\1 \2\n/g')
  fi
  KD_PEND=0; KD_ON=1; RK_WANT=24; RK_RC=0; rk_map
}
kd_end() {  # tray N's chips that answer at the end (a tray re-power may bring it back), as "k of 8"
  local b n=0; for b in $(kd_bdfs "$KD_TRAY"); do [ "$(rk_state "0000:$b")" = ok ] && n=$((n + 1)); done; echo "$n of 8"
}
kd_tag() {  # an INCOMPLETE RESULT with --known-dropped-tray keeps its reason and names the known drop by BDF and tray
  local b m=""; for b in $(kd_bdfs "$KD_TRAY"); do [ "$(rk_state "0000:$b")" = ok ] || m="$m $b"; done
  echo "known drop: tray $KD_TRAY (--known-dropped-tray=$KD_TRAY, not tested):${m:- none missing at the end}; $(kd_end) chips answer at the end"
}
[[ ${BASH_SOURCE[0]} != "$0" ]] && { declare -F pg_run > /dev/null || . "$TT_DIR/lib/pgroups.sh"; return 0; }   # sourced (the tests): the functions only, no test; pg_run (rk_reset) from lib/pgroups.sh
SELF=$(readlink -f "$0"); cd "$TT_DIR"
unset TRAY_STUB   # only --stub sets it (lib/stubhost.sh); an inherited value would skip setup and label a real run STUB
case " $* " in *" --stub "*) . ./lib/stubhost.sh; SYS=$TRAY_REPRO_SYS ;; esac   # --stub: fake healthy host, WARNING first (lib/stubhost.sh); SYS was set above
# RK_SETTLE_S was set above from the caller's env, before the stub set its 1 s settle; read it
# again so a stub reset waits 1 s, not 60. A real run's value is unchanged (60 s unless TRAY_REPRO_RESET_SETTLE_S is set).
[ -n "${TRAY_STUB:-}" ] && { RK_SETTLE_S=$TRAY_REPRO_RESET_SETTLE_S; echo "stub mode: reset settle ${RK_SETTLE_S} s"; }
if [ -n "${HC_HOLDFILE:-}" ]; then . "$HC_HOLDFILE"; else hc_holdfile_off; fi   # the user's holds and allowlist: a held host runs nothing, exit 10
hf_stop() { hf_check || { echo "NOT RUN: hold: $HF_WHY (TRAYTEST_HOLD_FILE, lib/holdfile.sh)" >&2; vd_say "NOT RUN (hold: $HF_WHY)"; exit 10; }; }   # runs after --help (help always prints) and before any device step
. ./lib/sudo_route.sh   # no or partial passwordless sudo is output line 1, then the summary and RESULT
case " $* " in *" -h "*|*" --help "*) ;; *) sudo_route_check hsc clear cpld sensors drop rescan; if [ -n "$SR_LINE" ]; then echo "$SR_LINE"; fi ;; esac
. ./lib/sha_line.sh; sha_line "$TT_TOP" || true   # own commit, first lines of run.log
. ./lib/kmd_policy.sh; echo "power_policy: $(kp_read)"; KP_OPT=""   # the loaded KMD's power_policy, every mode
chmod u+x setup.sh ub0_repro.py 2>/dev/null
# light evidence logging always; --heavy-logs (or TRAY_REPRO_HEAVY_LOGS=1) adds the slow BMC reads
KD_TRAY=${TRAY_TEST_KNOWN_DROPPED:-}; KD_ON=0; KD_PEND=0
HOLD=0; HOLD_TIMEOUT_MIN=240   # --hold, --hold-timeout (lib/hold.sh): off unless asked for on the command line
QUIETS="1 1 1 1"; AG_QUIETS="1 1 1 1"; HEAVY=${TRAY_REPRO_HEAVY_LOGS:-0}; HWMON_GUARD=0; FLEET=""; FL_SMOKE=0; FL_DRY=0; TRIES=0; FL_UB0_ARGS=(); REPRO=""; REPRO_BDF=""; MODE_SET=0; TDP=""
ARGV0=("$@")   # --repro numeric|warm-reset: the options before it go on to repro.sh, which passes them to every try
while [ $# -gt 0 ]; do
  case $1 in
    --tries) QUIETS=${2:?--tries needs quiet minutes, e.g. \"1 1 3\"}; TRIES=1; shift ;;
    --ag-tries) AG_QUIETS=${2:?--ag-tries needs quiet minutes, e.g. \"1 1 1 1\", or none}; TRIES=1; shift ;;
    --heavy-logs) HEAVY=1 ;;
    --hwmon-guard) HWMON_GUARD=1 ;;
    --no-mce-guard) TRAY_REPRO_MCE_GUARD=0 ;;
    --mode=*) FLEET=${1#--mode=}; MODE_SET=1 ;;
    --mode) FLEET=${2:?--mode needs quick, full, smoke, long or repro}; MODE_SET=1; shift ;;
    --repro=*) REPRO=${1#*=}
      case $REPRO in numeric|warm-reset|tray-drop) echo "--repro $REPRO is spelled --repro $REPRO <bdf> [...], last on the line (run.sh --help)" >&2; exit 1 ;; esac ;;
    --repro) case ${2:-} in numeric|warm-reset|tray-drop) shift   # fast repro of one known fault (repro.sh): repro.sh runs plain run.sh per try until the first hit
        [ -z "$FLEET" ] && [ "$TRIES" = 0 ] && [ -z "$REPRO" ] || { echo "--repro takes no --mode, --tries or --ag-tries (repro.sh picks the stages)" >&2; exit 1; }
        hf_stop; TRAY_REPRO_HEAVY_LOGS=$HEAVY exec bash ./repro.sh "$@" "${ARGV0[@]:0:$(( ${#ARGV0[@]} - $# - 1 ))}" ;; esac
      REPRO=${2:?--repro needs a repro name: link-a, abort-fabric, numeric, warm-reset or tray-drop}; shift   # --repro <name> [<bdf>]
      if [[ ${2:-} =~ ^([0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$ ]]; then REPRO_BDF=$2; shift; fi ;;
    --fleet=*|--fleet)   # the old name, kept for launchers during the switch-over; c7 (sustained) is retired
      if [ "$1" = --fleet ]; then FLEET=${2:?--mode needs quick, full, smoke, long or repro}; shift; else FLEET=${1#--fleet=}; fi
      echo "run.sh: --fleet is now --mode" >&2 ;;
    --dry-run) FL_DRY=1 ;;
    --stub) ;;   # read above (lib/stubhost.sh)
    --host-local-reset) echo "NOTE run.sh: --host-local-reset is gone and ignored: C2 and C9 reset this host's own chips behind its own gate only" >&2 ;;   # callers still pass it (a cluster job.sh)
    --since=*|--history-since=*) FLEET_HISTORY_SINCE=${1#*=} ;;
    --since|--history-since) [ $# -ge 2 ] || { echo "$1 needs a time, e.g. 2026-10-22T20:32-07:00 (or '' for none)" >&2; exit 1; }
      FLEET_HISTORY_SINCE=$2; shift ;;
    --known-dropped-tray=*) KD_TRAY=${1#*=} ;;
    --known-dropped-tray) KD_TRAY=${2:?--known-dropped-tray needs a tray number 1-4}; shift ;;
    --kmd-power-policy=*) KP_OPT=${1#*=} ;;
    --kmd-power-policy) KP_OPT=${2:?--kmd-power-policy needs 0 or 1}; shift ;;
    --tdp=*|--tdp) if [ "$1" = --tdp ]; then TDP=${2:-}; [ $# -ge 2 ] && shift; else TDP=${1#*=}; fi   # no range check
      [[ $TDP =~ ^[0-9]+$ ]] || { echo "--tdp needs whole watts, e.g. --tdp=160 (got '$TDP')" >&2; exit 1; } ;;
    --hold) HOLD=1 ;;
    --hold-timeout=*) HOLD_TIMEOUT_MIN=${1#*=} ;;
    --hold-timeout) HOLD_TIMEOUT_MIN=${2:?--hold-timeout needs minutes}; shift ;;
    -h|--help) sed -n '2,/^set -u/p' "$SELF" | sed '$d' | sed 's/^# \{0,1\}//'
      echo "Stages of the fleet modes (fleet/stages.tsv; the README Stages table has the same numbers and names):"
      awk -F'\t' '!/^#/ { printf "  stage %s %s: %s\n", $1, $3, $4 }' "$TT_DIR/fleet/stages.tsv"; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 1 ;;
  esac; shift
done
hf_stop   # the user's hold, after --help and before any device step
# --mode=long: lib/long.sh chains whole runs, 3 plain then 2 --mode=full, each re-invoking run.sh with
# the other options; it adds no step of its own to either mode
if [ "$FLEET" = long ]; then
  [ -z "$REPRO" ] && [ "$TRIES" = 0 ] && [ "$HOLD" = 0 ] || { echo "--mode=long takes no --repro, --tries, --ag-tries or --hold (it chains plain and full runs)" >&2; exit 1; }
  LG_FWD=(); sk=0
  for a in "${ARGV0[@]}"; do
    [ "$sk" = 1 ] && { sk=0; continue; }
    case $a in --mode=*|--fleet=*) ;; --mode|--fleet) sk=1 ;; *) LG_FWD+=("$a") ;; esac
  done
  exec bash ./lib/long.sh "${LG_FWD[@]}"
fi
# --repro <name> [<bdf>]: one known fault's opt-in fast repro, never part of plain run.sh or --mode=full
if [ -n "$REPRO" ]; then
  [ "$MODE_SET" = 0 ] && [ "$TRIES" = 0 ] || { echo "--repro runs on its own: drop --mode, --tries and --ag-tries" >&2; exit 1; }
  case $REPRO in
    link-a) FLEET=repro-linka; [ -n "$REPRO_BDF" ] && export FLEET_LINKA_A=$REPRO_BDF ;;
    abort-fabric) [ -z "$REPRO_BDF" ] || { echo "--repro abort-fabric takes no chip BDF (it runs on all of this host's chips)" >&2; exit 1; }
      FLEET=repro-abortfab ;;
    numeric|warm-reset|tray-drop) echo "--repro $REPRO is spelled --repro $REPRO <bdf> [...], last on the line (run.sh --help)" >&2; exit 1 ;;
    *) echo "--repro takes link-a, abort-fabric, numeric, warm-reset or tray-drop (got '$REPRO')" >&2; exit 1 ;;
  esac
elif [ "$FLEET" = repro-linka ]; then echo "--mode=repro-linka is spelled --repro link-a" >&2; exit 1
elif [ "$FLEET" = repro-abortfab ]; then echo "--mode=repro-abortfab is spelled --repro abort-fabric" >&2; exit 1; fi
# no --mode is quick; --tries/--ag-tries alone keep the old tray-drop repro (run-mid.sh, old launchers)
[ -z "$FLEET" ] && { [ "$TRIES" = 1 ] && FLEET=repro || FLEET=quick; }
# --mode=smoke is plain run.sh with every load, burst and quiet cut to seconds: FLEET stays quick, so it
# runs quick's code paths; FL_SMOKE only sets the duration knobs (fl_smoke, lib/fleet.sh; tests/test_mode_equal.sh)
[ "$FLEET" = smoke ] && { FLEET=quick; FL_SMOKE=1; }
if [ "$FLEET" = repro ]; then FLEET=""
elif [ "$TRIES" = 1 ]; then echo "--tries and --ag-tries go with --mode=repro (--mode=$FLEET runs no tries)" >&2; exit 1; fi
[ "$QUIETS" = none ] && QUIETS=""; [ "$AG_QUIETS" = none ] && AG_QUIETS=""
[ -z "$KD_TRAY" ] || [[ $KD_TRAY =~ ^[1-4]$ ]] || { echo "--known-dropped-tray (TRAY_TEST_KNOWN_DROPPED) takes a tray number 1-4 (got '$KD_TRAY')" >&2; exit 1; }
[ -z "$KP_OPT" ] || [[ $KP_OPT =~ ^[01]$ ]] || { echo "--kmd-power-policy takes 0 or 1 (got '$KP_OPT')" >&2; exit 1; }
[ -n "$KD_TRAY" ] && [ -z "$FLEET" ] && { echo "--known-dropped-tray goes with the fleet modes (no --mode, quick, full), not --mode=repro" >&2; exit 1; }
FLEET_HISTORY_SINCE=${FLEET_HISTORY_SINCE:-}   # repair-since prints nothing (rc 1) for a host with no repair: no repair baseline;
[[ $FLEET_HISTORY_SINCE =~ ^[[:space:]]*$ ]] && FLEET_HISTORY_SINCE=   # only an all-blank value: spaces inside a time ('7 days ago') stay
if [ -n "${FLEET_HISTORY_SINCE:-}" ]; then   # C1 history from the last repair on;
  # before fl_init, whose --dry-run exits 0
  hs_t=$(date -d "$FLEET_HISTORY_SINCE" +%s 2>/dev/null) || { echo "--since: not a time: '$FLEET_HISTORY_SINCE'" >&2; exit 1; }
  [ "$hs_t" -le "$(date +%s)" ] || { echo "--since: '$FLEET_HISTORY_SINCE' is in the future" >&2; exit 1; }
  export FLEET_HISTORY_SINCE
fi
if [ -n "$FLEET" ]; then   # the fleet profile runs no tries (the opt-in C4 tries are gone): its load stages are fl_loads (lib/fleet.sh)
  QUIETS=""; AG_QUIETS=""; . ./lib/fleet.sh; fl_init
elif [ "$FL_DRY" = 1 ]; then echo "--dry-run goes with the fleet modes (no --mode, quick, full), not --mode=repro" >&2; exit 1; fi
# the run budget: plain run.sh (quick) ends by 1800 s; --mode=full by 1800 s + its longer
# last chunk (FL_EXTRA_S, the same budget paths as plain); --hold and --mode=repro wait on purpose
RS_START=$(date +%s); RS_T0=""; [ -z "$RS_RUN_S" ] && { case $FLEET in quick|full) [ "${HOLD:-0}" != 1 ] && RS_RUN_S=$(( 1800 + ${FL_EXTRA_S:-0} )) || RS_RUN_S=0 ;; *) RS_RUN_S=0 ;; esac; }
RS_PLAN_S=$RS_RUN_S; RS_RUN_S=0   # the planned length only; no stage is skipped or cut for time
HL_DEADLINE=""   # set by rs_clock_start once setup is over; rs_hl_end clears it before C9
[ -n "$FLEET" ] || [ -n "${QUIETS// /}${AG_QUIETS// /}" ] || { echo "--tries and --ag-tries need at least one try between them" >&2; exit 1; }
for q in $QUIETS $AG_QUIETS; do  # >= 1: with 0, ub0_repro.py skips its quiet check and the load could start busy
  [[ $q =~ ^[1-9][0-9]*$ ]] || { echo "--tries/--ag-tries take whole minutes >= 1, one per try (got '$q')" >&2; exit 1; }
done
# the host-wide run lock: fleet/gate.sh on this host counts a run.sh as busy only while it holds
# this lock (TRAY_RUN_LOCK; tests point it at their temp dir). Read-only open: no O_CREAT on another user's lock file
# (fs.protected_regular). Held elsewhere: not taken, and this run's gate then refuses (busy).
RUN_LOCK=${TRAY_RUN_LOCK:-/run/lock/tray-test-run.lock}; unset TRAY_RUN_LOCK_HELD TRAY_RUN_LOCK_BUSY
if command -v flock > /dev/null && { [ -e "$RUN_LOCK" ] || : > "$RUN_LOCK"; } 2> /dev/null && exec 9< "$RUN_LOCK"; then
  if flock -w 3 9; then export TRAY_RUN_LOCK_HELD=$RUN_LOCK; else TRAY_RUN_LOCK_BUSY=1; echo "run lock $RUN_LOCK: held by another run" >&2; fi   # BUSY: no TDP open (lib/tdp.sh)
else echo "WARNING: no run lock ($RUN_LOCK): other runs' gates cannot see this run" >&2; fi
[ -n "${UB0_PYTHON:-}" ] || [ -z "${HC_SETUP:-}" ] || [ -x "${HC_VENV:-venv}/bin/python" ] || echo "first run: setting up ./venv (about 375 MB of pip packages and the SFPI compiler; a few minutes)..."
H=$(hostname -s); say() { echo "$(date '+%F %T %Z') $H: $*"; }
FL_BOOT_UP0=$(cut -d' ' -f1 "${FLEET_PROC:-/proc}/uptime" 2> /dev/null); FL_BOOT_UP0=${FL_BOOT_UP0%%.*}   # uptime at run start (lib/fleet.sh fl_boot_recount)
# the KMD power_policy value run's header prints (bank-3-only numeric fails were tied to it)
TRAY_POWER_POLICY=$(cat /sys/module/tenstorrent/parameters/power_policy 2> /dev/null | tr -d '[:space:]'); export TRAY_POWER_POLICY; [ -n "$TRAY_POWER_POLICY" ] || TRAY_POWER_POLICY=unknown
. ./lib/power_policy.sh; pp_warn   # WARN when power_policy is not Y (reduced DRAM error detection) or unread; never changed
TRAY_IDLE_GRACE_MS=$(cat /sys/module/tenstorrent/parameters/idle_power_down_grace_ms 2> /dev/null | tr -d '[:space:]'); export TRAY_IDLE_GRACE_MS; [ -n "$TRAY_IDLE_GRACE_MS" ] || TRAY_IDLE_GRACE_MS=unknown
# su_run: setup.sh opens no device, so a failed setup (no package index and no venv cache hit, a pin
# mismatch) is a test-infra NOT RUN naming setup's last error line (exit 6, as rk_eacces_stop), never a chip FAIL. Its
# stderr goes on as before; its 'setup: offline:' line (the venv cache it fell back to) also goes in the run log.
su_run() {
  local e rc l why=""; e=$(mktemp "${TMPDIR:-/tmp}/tray-setup.XXXXXX") || e=""
  if [ -n "${HC_SETUP:-}" ]; then bash -c "$HC_SETUP" > /dev/null 2> "${e:-/dev/stderr}"; rc=$?
  else python3 -c 'import ttnn' > /dev/null 2> "${e:-/dev/stderr}"; rc=$?; fi   # no HC_SETUP: the caller's env must have ttnn
  if [ -n "$e" ]; then
    while IFS= read -r l || [ -n "$l" ]; do
      case $l in "setup: offline: "*) echo "$l" ;; *) echo "$l" >&2 ;; esac
      [ -z "${l//[[:space:]]/}" ] || why=$l
    done < "$e"; rm -f -- "$e"
  fi
  [ "$rc" = 0 ] && return 0
  why=${why#setup: }; why=${why#TEST-INFRA: }; why=${why:0:300}
  hb_stop; echo "SETUP ERROR (test-infra, not a chip fault): setup.sh exit $rc${why:+: $why}; run ./setup.sh to see why" >&2
  vd_say "NOT RUN (test-infra: venv setup failed: ${why:-setup.sh exit $rc})"; exit 6
}
RS_SU0=$(date +%s); hb_start setup; [ -n "${TRAY_STUB:-}" ] || su_run; hb_stop
# the one-time ./venv install (330-443 s on a fresh cluster clone) is not a test stage: the run budget starts after it
# setup's seconds always print, apart from the run budget (a TRAY_VENV_CACHE hit or a ready ./venv: about 1 s)
if [ "$RS_RUN_S" -gt 0 ] 2> /dev/null && [ $(( $(date +%s) - RS_SU0 )) -ge 30 ]; then
  echo "setup: $(( $(date +%s) - RS_SU0 )) s (./venv install), not counted in TRAY_REPRO_RUN_S=$RS_RUN_S"
else echo "setup: $(( $(date +%s) - RS_SU0 )) s (./venv ready$([ -L "${HC_VENV:-venv}" ] && echo ", TRAY_VENV_CACHE $(readlink "${HC_VENV:-venv}")"))"; fi
PY=${UB0_PYTHON:-${HC_VENV:-$PWD/venv}/bin/python}; [ -n "${HC_SETUP:-}" ] || PY=${UB0_PYTHON:-python3}
OUT=${TRAY_RUN_OUT:-${TRAY_RUN_PARENT:-$TT_TOP}/run_${H}_$(date -u +%Y%m%dT%H%M%SZ)}; mkdir -p "$OUT" 2> /dev/null \
  || { echo "run.sh: ERROR: cannot create the run dir $OUT: set TRAY_RUN_PARENT to a writable dir" >&2; exit 1; }   # stop here, not at an unbound KC_DIR; TRAY_RUN_PARENT: reliability.sh; TRAY_RUN_OUT: --mode=long gives each leg its own dir
# every setsid child's process group goes in TRAY_PGID_FILE; a broker preemption's TERM reaps them (lib/pgroups.sh)
# pg_reap may run as root: run.sh's own file is 0600; the sudo caller's processes get TRAY_PGID_USER_FILE
. ./lib/pgroups.sh; export TRAY_PGID_FILE=$OUT/.pgids; : > "$TRAY_PGID_FILE"; chmod 600 "$TRAY_PGID_FILE"; TRAY_PGID_USER_FILE=""; rs_traps
AS=()   # the load must not run as root (root runs hung at mesh open): run it as the sudo caller
if [ "$(id -u)" = 0 ]; then
  # sudo's env_reset drops TRAY_HEAVY: AS carries it, so the loads (ub0_repro.py, chunk_load.py) and the numeric pinpoint's
  # repeat all run the run's op; env, not sudo VAR=value, which sudoers may refuse; FL_SMOKE too, so chunk_load's
  # N1 row marks a smoke run under sudo (its short-coverage line INFO, not WARN)
  # $OUT is the sudo caller's (the loads write there), never world-writable; their processes record their groups in a
  # file of their own, which root's pg_reap reads with the same checks as its own (lib/pgroups.sh)
  if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then
    TRAY_PGID_USER_FILE=$OUT/.pgids.user; : > "$TRAY_PGID_USER_FILE"; chmod 644 "$TRAY_PGID_USER_FILE"; chown "$SUDO_USER" "$TRAY_PGID_USER_FILE" "$OUT"
    AS=(sudo -u "$SUDO_USER" -H env TRAY_PGID_FILE="$TRAY_PGID_USER_FILE" ${TRAY_HEAVY:+TRAY_HEAVY="$TRAY_HEAVY"} ${FL_SMOKE:+FL_SMOKE="$FL_SMOKE"})
  else echo "run as a normal user (or with sudo from one): the load must not run as root" >&2; exit 1; fi
fi
rs_sw_stop; rs_stopwatch   # again, now that TRAY_PGID_FILE (and the sudo caller's file) exist, so a stop TERMs the stage's groups too
# the kernel cache: one TT_METAL_CACHE per tray-test sha and ttnn version, in every mode, before any mesh open,
# so each later open (qb2, the chunks process, numeric) reuses what the first one compiled; sudo's env_reset drops it: AS carries it
. ./lib/kcache.sh; KC_AS=(${AS[@]+"${AS[@]}"}); kc_init "$PY" "$PWD" "$([ "${#AS[@]}" -gt 0 ] && echo "${SUDO_USER:-}")" > "$OUT/kcache.log" 2>&1; say "$(head -1 "$OUT/kcache.log")"
[ -n "$KC_DIR" ] && [ "${#AS[@]}" -gt 0 ] && AS+=(TT_METAL_CACHE="$KC_DIR")
. ./lib/umd_index.sh; . ./lib/heavy_logs.sh; . ./lib/fault_capture.sh; . ./lib/link_check.sh   # functions only: heavy_logs_init below starts them
. ./lib/mce_guard.sh   # the MCE guard: armed after the start reset, acts on a chip hang
if [ -n "${HC_HOLD:-}" ]; then . "$HC_HOLD"; else hc_hold_off; fi; hold_init   # --hold: refused under Slurm; armed only with --hold
. ./lib/bmc_heat.sh   # BMC heat poll and the drop's BMC/CPLD snapshot: functions only
. ./lib/pinpoint.sh   # the numeric pinpoint after a NUMERIC FAIL: functions only
. ./lib/tools_probe.sh; _tl=$(mktemp); tools_locate "$PY" > "$_tl"; while IFS= read -r l; do say "$l"; done < "$_tl"; rm -f "$_tl"; UB0_TRIAGE=${CS_TRIAGE:-${UB0_TRIAGE:-}}   # tt-triage/ttexalens
. ./lib/fw_check.sh; fw_preflight   # C0 firmware uniformity from sysfs: before anything opens a device
. ./lib/gddr.sh   # per-chip GDDR telemetry: read after the start reset and before the end reset; functions only
. ./lib/tdp.sh   # --tdp: tdp_start runs only once the start check passed (fl_start's P0 gate; --mode=repro: after rk_start)
HEAVY_PY=$PY HEAVY_AS="${AS[*]}" HEAVY_TRIAGE=${UB0_TRIAGE:-} HEAVY_FULL=$HEAVY   # light snapshots and poll unless --heavy-logs or a fault
# dev_proc of the load, as run_repro.sh logs it: who opens the chips and its CapEff (chip0 REPORT.md: a
# capability in CapEff hangs mesh open). The root refusal above (AS) and run_repro.sh's setpriv drop already guard it: log only
say "dev_proc=$(${AS[@]+"${AS[@]}"} sh -c 'echo "uid:$(id -u)($(id -un)) gid:$(id -g) capeff=$(awk "/^CapEff/ {print \$2}" /proc/self/status)"' 2>/dev/null | sed -E 's/capeff=0+$/capeff=0/') (the load's user at device open)"
HEAVY_HWMON_GUARD=$HWMON_GUARD HEAVY_RUN_USER=${SUDO_USER:-$(id -un)}   # --hwmon-guard: the load's own user may hold the chips
# exact sudo -n -l probes, never sudo -n true: ipmitool by the SEL read below (ub0_repro.py sends fixed forms too), lspci on its own
"${AS[@]}" sudo -n -l ipmitool sel elist last 40 < /dev/null > /dev/null 2>&1 && HX_SUDO=1 || HX_SUDO=0
"${AS[@]}" sudo -n -l lspci -vvv -d 1e52: < /dev/null > /dev/null 2>&1 && HX_LSUDO=1 || HX_LSUDO=0
rk_can_root && HX_RSUDO=1 || HX_RSUDO=0   # try_stacks: cat of a stack, py-spy take varying arguments, so full root only
[ -n "$KP_OPT" ] && { kp_apply "$KP_OPT"; TRAY_POWER_POLICY=$(kp_read); }   # --kmd-power-policy: before the polls, any device open and C2, only after this host's own busy gate (kp_busy); the RESULT line's pp_tag shows the reloaded value
heavy_logs_init "$OUT"   # every run (about 5 s); lib/fleet.sh's fl_capture needs it for heavy_logs_on_hang
bh_poll_want && bh_poll_start "$OUT"   # --heavy-logs (any mode): the SDR heat poll for the whole run
HX=()
try_args() {  # <try>: ub0_repro.py's evidence. Always (no wall time; the record before a fault cannot be read after
              # it): the result JSON (a CCL hang's suspects), full telemetry and the 1 s heartbeat/LnkSta poll, the
              # in-process NOC snapshot of a held all-gather hang, lspci of a lost chip's root port, and the tray CPLD
              # and SEL reads after a loss that classify it (CHIP_HANG / DROP / UNKNOWN; no cost on a healthy try).
              # --heavy-logs adds the CPLD/SEL baseline before each try (14-31 s a try); else the loss's SEL is read
              # against the one SEL read of the run (SEL_BASE)
  HX=(--result "$OUT/try$1.json" --telem "$OUT/try${1}_telem.csv" --hold-on-hang --snapshot-dir "$OUT/heavy/hold_try$1")
  [ "$HX_LSUDO" = 1 ] && HX+=(--lspci 'sudo -n lspci')
  [ "$HX_SUDO" = 1 ] && HX+=(--ipmitool 'timeout 15 sudo -n ipmitool')
  [ -n "$SEL_BASE" ] && HX+=(--sel-baseline "$SEL_BASE")
  return 0
}
# The per-try wall-clock limit. One host: matmul try 1 sat 98 min at "traffic: fabric init done",
# the main thread blocked in a futex: its phase deadlines never fired and SIGINT/SIGTERM were not handled. A try takes
# about 2 min (at most 128 s over 572 tries at 1 min of quiet); a held all-gather hang (ub0_repro.py's own capture,
# "HOLD ON HANG" in the log) gets TRY_HOLD_S more once. Set: the limit in s for every try; 0 = no limit.
# the limit is a no-progress hang timeout, not wall time. The clock restarts each time
# the try's log grows; while the log's last event is a load chunk's start with its chunk_s, that chunk_s is added. Default
# 600 s: 30,799 healthy stage logs since 10-27 were silent at most 296 s outside an announced chunk (a JIT build), while the
# hung qb1 tries on one host sat silent 917-928 s.
TRY_LIMIT_S=${TRAY_REPRO_HANG_S:-${TRAY_REPRO_TRY_LIMIT_S:-600}}; TRY_HOLD_S=${TRAY_REPRO_TRY_HOLD_S:-1500}; TRY_PID=""; TRY_HUNG=""
HANG_S=$TRY_LIMIT_S   # the run's hang timeout as set (a stage may pass its own TRY_LIMIT_S); printed in the summary
try_chunk_s() {  # <log>: the announced chunk_s while the log's last event line is a load chunk's start, else 0
  local l; l=$(tail -c 65536 "$1" 2>/dev/null | grep -a '"event": "' | tail -1)
  [[ $l == *'"event": "start"'*'"chunk_s": '[0-9]* ]] || { echo 0; return 0; }
  l=${l#*'"chunk_s": '}; echo "${l%%[!0-9]*}"
}
# TRY_LOSS_S (set by the bus-bar stage step and by fl_qstep): s allowed after the try's first "FIRST LOSS" log line before the try's group
# is stopped (it captures the other trays meanwhile). rc 3 (TRIP), TRY_LOSSEND set. 0/unset: off. A hung burst never exited on its own (15 min on record)
try_loss_end() {  # <log> <first-loss-epoch or empty> -> prints the epoch of the first sight, else nothing
  [ -n "$2" ] && { echo "$2"; return; }; grep -aq 'FIRST LOSS at' "$1" 2>/dev/null && date +%s; return 0
}
try_infra() {  # <rc> <log>: "test-infra: <why>" when the test binary could not start (rc 126/127, a loader error
  # such as a missing libevent_pthreads-2.1.so.7), else nothing: such a try is NOT_RUN (INCOMPLETE), never a FAIL
  local l; l=$(grep -a -o -m1 -E 'error while loading shared libraries.*|[^ ]*cannot open shared object file.*' "$2" 2>/dev/null | cut -c1-300)
  if [ -n "$l" ]; then echo "test-infra: rc $1, $l"
  elif [ "$1" = 126 ] || [ "$1" = 127 ]; then echo "test-infra: rc $1, the test binary did not start"; fi
}
try_run() {  # <try> <log> <cmd...>: the try in its own process group (setsid: pgid = pid), waited for up to
             # its limit (default 900 s + q min). Sets rc; past the limit try_hung runs and TRY_HUNG is set (rc 124)
             # TRY_FAB_LIMIT_S (the stage-2 design's hang limit; set by the
             # bus-bar quiet-burst tries): once the log shows set_fabric_config(...) returned, the no-progress limit is this
             # (180 s) instead of TRY_LIMIT_S (900 s, kept while the fabric is set up: clean setups on one site took 702-903 s);
             # until "AICLK before burst" it also allows TRY_FAB_IDLE_S, the open quiet (idle plus clock-check cap), which
             # prints nothing while it waits. Clean QB ends never sat over 61 s after setup (3,102 ends). Unset: off
  local i=$1 log=$2 lim t0 held=0 hx=0 fab=0 cur fl="" tp sz psz=-1 ext=0 now; shift 2; TRY_HUNG=""; TRY_LOSSEND=""; TRY_INFRA=""; TRY_YIELDEND=""
  setsid "$@" > "$log" 2>&1 < /dev/null {PG_OUT}>&- &
  TRY_PID=$!; pg_track "$TRY_PID"; t0=$(date +%s); tp=$t0; lim=${TRY_LIMIT_S:-$(( 900 + q * 90 + 3 * ${TRAY_QUIET_CLK_CAP_S:-300} ))}
  # its own session gets no Ctrl-C from the terminal: pass INT on to the leader (sudo relays it), give it 1 s, then end
  # the run as pg_traps' INT does (the try ignored INT and run.sh went on); TERM (a broker preemption) stays
  # rs_dying's (rs_traps): it reaps the try's group with every other recorded one, prints RESULT: INCOMPLETE, then exits
  trap 'kill -INT "$TRY_PID" 2>/dev/null; for _i in 1 2; do kill -0 "$TRY_PID" 2>/dev/null || break; sleep 0.5; done; rs_dying INT 130' INT
  while kill -0 "$TRY_PID" 2>/dev/null; do
    if [ "${TRY_LOSS_S:-0}" != 0 ]; then
      fl=$(try_loss_end "$log" "$fl")
      if [ -n "$fl" ] && [ $(( $(date +%s) - fl )) -ge "$TRY_LOSS_S" ]; then
        TRY_LOSSEND="first loss seen $TRY_LOSS_S s ago: try stopped (rest of the burst not waited for)"; say "$i: $TRY_LOSSEND"
        try_stop "$TRY_PID" > /dev/null 2>&1; break
      fi
    fi
    if [ -n "${FLEET:-}" ] && [ -z "${FL_YIELD:-}" ] && [ -z "${FL_C9_ON:-}" ] && declare -F by_on > /dev/null && by_on; then   # a bus-bar part
      by_check; [ -n "${FL_YIELD:-}" ] && { TRY_YIELDEND=1; fl_skip "$i try ($log, stopped after $(( $(date +%s) - t0 )) s)" "$FL_YIELD"; try_stop "$TRY_PID" > /dev/null 2>&1; break; }   # gives the device up within 30 s: not run, never a loss or hang
    fi
    now=$(date +%s); sz=$(stat -c %s "$log" 2>/dev/null || echo 0)
    [ "$sz" != "$psz" ] && { psz=$sz; tp=$now; ext=$(try_chunk_s "$log"); }   # progress: the log grew
    if [ -n "${TRY_FAB_LIMIT_S:-}" ] && [ "$tp" = "$now" ]; then   # fabric phase, read on growth only
      # the last of the three: ub0_repro.py --cycles prints "cycle N/M begin" before each later open quiet,
      # which gets TRY_FAB_IDLE_S again until its own "AICLK before burst"
      case $(grep -aoE 'AICLK before burst|cycle [0-9]+/[0-9]+ begin|set_fabric_config\(.*\) returned' "$log" | tail -n1) in
        "AICLK before burst") fab=2 ;; cycle*|set_fabric_config*) fab=1 ;; esac
    fi
    cur=$lim; [ "$fab" = 1 ] && cur=$(( TRY_FAB_LIMIT_S + ${TRY_FAB_IDLE_S:-0} )); [ "$fab" = 2 ] && cur=$TRY_FAB_LIMIT_S
    if [ "$lim" != 0 ] && [ $(( now - tp )) -ge $(( cur + hx + ext )) ]; then
      if [ "$held" = 0 ] && grep -aq 'HOLD ON HANG' "$log"; then held=1; hx=$TRY_HOLD_S; continue; fi
      try_hung "$i" "$log" "$TRY_PID" $(( now - tp )); break
    fi
    sleep 1
  done
  rs_traps
  if [ -n "$TRY_LOSSEND" ]; then wait "$TRY_PID" 2>/dev/null; rc=3; elif [ -n "$TRY_YIELDEND" ]; then wait "$TRY_PID" 2>/dev/null; rc=0; elif [ -n "$TRY_HUNG" ]; then rc=124; else wait "$TRY_PID"; rc=$?; fi
  # a plain try that ended on its own CHIP_HANG (exit 3, inside its limit): the MCE guard, timed from its FIRST LOSS
  # line; --hold: it stands down
  [ "$rc" = 3 ] && [ -z "$TRY_HUNG" ] && declare -F mg_try_chip_hang > /dev/null && mg_try_chip_hang "$i" "$log"
  [ -z "$TRY_LOSSEND$TRY_HUNG" ] && [ "$rc" != 0 ] && TRY_INFRA=$(try_infra "$rc" "$log")
  rs_mesh_credit "$log" "$t0" "$i"   # the first load try's mesh open and compile is setup
  return 0
}
try_hangpin() {  # a try hang no chip was named for: the chip whose ETH links stopped sending during the
  # try (frames sent between the heavy-logs reads before the run and before the reset); else the chips the try's
  # telemetry saw at load and those it never did. Reads files only, after the hang: no time in a healthy run
  local pre post m o l
  pre=$(ls -d "$HL_OUT"/[0-9]*_pre 2>/dev/null | head -1); post=$(ls -d "$HL_OUT"/[0-9]*_pre-reset 2>/dev/null | tail -1)
  m=$OUT/.hangpin_map; echo "$RK_MAP" > "$m"
  o=$(python3 -I "$FC_PY" hangpin "$m" "${pre:-/nonexistent}" "${post:-/nonexistent}" "$OUT/try${1}_telem.csv" 2>/dev/null); rm -f "$m"
  [ -n "$o" ] || return 0
  echo "$o" > "$OUT/try${1}_hangpin.txt"
  while read -r l; do fault_note "$l"; [ -n "$FLEET" ] && fl_find "$l; $2"; done <<< "$o"
}
eth_fec_judge() {  # off-box ch8/9 FEC and retrains between the heavy-logs reads before
  # the run and before the reset. Every mode the same. FEC on (nearly) every off-box end at once is the far end resetting:
  # one "ETH INFO" line, never a finding, verdict, RESULT, repair line or exit code. One or two ends that stand out from their
  # siblings: an "ETH FEC WARN" (a warn, like the HSC under-voltage warn), never a FAIL. Reads files only
  # A run it cannot judge prints one "ETH FEC: not judged (why)" line through say, never a finding
  local pre post m o l a b st="the load"
  [ -n "${HL_OUT:-}" ] || { say "ETH FEC: not judged (no heavy-logs reads this run)"; return 0; }
  pre=$(ls -d "$HL_OUT"/[0-9]*_pre 2>/dev/null | head -1); post=$(ls -d "$HL_OUT"/[0-9]*_pre-reset 2>/dev/null | tail -1)
  [ -n "$pre" ] || { say "ETH FEC: not judged (no heavy-logs read before the run)"; return 0; }
  [ -n "$post" ] || { say "ETH FEC: not judged (no heavy-logs read before the reset)"; return 0; }
  if [ -n "$FLEET" ] && [ -s "$FL_OUT/phases.txt" ]; then   # the stages it spans, by number and name (fleet/stages.tsv)
    a=$(awk 'NR == 1 { print $1 }' "$FL_OUT/phases.txt"); b=$(awk 'END { print $1 }' "$FL_OUT/phases.txt")
    st=$(fl_pn "$a"); [ "$b" = "$a" ] || st="$st to $(fl_pn "$b")"
  fi
  m=$OUT/.ethfec_map; echo "$RK_MAP" > "$m"
  o=$(python3 -I "$FC_PY" ethfec "$m" "$pre" "$post" "$st" 2>/dev/null); rm -f "$m"
  [ -n "$o" ] || return 0
  echo "$o" > "$OUT/eth_fec.txt"
  while read -r l; do
    case $l in
      "ETH FEC WARN "*) if [ -n "$FLEET" ]; then fl_find "$l"; else say "$l"; fi ;;
      *) say "$l" ;;   # the info line: never in findings.txt
    esac
  done <<< "$o"
}
try_stacks() {  # <pid> <file> <log> [py-spy cap s]: host-only evidence of a hung try (ps, /proc stacks, faulthandler, py-spy);
                # none of it opens a chip. Called by try_hung, and by the MCE guard just before it stops the try
  local pid=$1 f=$2 log=$3 pycap=${4:-60} py=$1 k t S=()
  [ "$(id -u)" != 0 ] && [ "${HX_RSUDO:-0}" = 1 ] && S=(sudo -n)   # HX_RSUDO: full root (rk_can_root), set once
  for k in 1 2 3; do   # under sudo the python is its child (or grandchild, with use_pty)
    [ "$(ps -o comm= -p "$py" 2>/dev/null)" = sudo ] || break; py=$(ps -o pid= --ppid "$py" | head -1 | tr -d ' ')
  done
  {
    echo "${TRY_HUNG-}"; echo "$(date '+%F %T %Z') process group $pid, python pid ${py:-?}"
    echo "== processes"; ps -o pid,ppid,pgid,stat,etimes,wchan:32,cmd --sid "$pid"
    for t in /proc/"${py:-0}"/task/*; do
      echo "== thread ${t##*/} $(cat "$t/comm" 2>/dev/null) $(grep -a '^State' "$t/status" 2>/dev/null) wchan $(cat "$t/wchan" 2>/dev/null)"
      timeout 5 "${S[@]}" cat "$t/stack" 2>&1
    done
  } > "$f" 2>&1
  [ -n "$py" ] && kill -USR1 "$py" 2>/dev/null && sleep 2   # the Python dump goes to the try log (its stderr)
  if [ -n "$py" ] && command -v py-spy > /dev/null; then
    { echo "== py-spy dump --native"; timeout "$pycap" "${S[@]}" py-spy dump --native --pid "$py"; } >> "$f" 2>&1
  fi
  { echo "== faulthandler (SIGUSR1) in $log"; sed -n '/^Thread 0x\|^Current thread 0x/,$p' "$log"; } >> "$f" 2>&1
}
try_hung() {  # <try> <log> <pid> <s>: a try past its limit. While it still hangs: name it, its stacks (Python, all
              # threads: ub0_repro.py's faulthandler on SIGUSR1, needs no GIL; native: py-spy; kernel: /proc stack),
              # the hang capture (lib/fault_capture.sh: tt-triage NOC snapshot, fast-dispatch dump); then stop it
  local i=$1 log=$2 pid=$3 s=$4 py=$3 k t hb last f=$OUT/try${1}_hung.txt S=()
  # lab, ph and q are the caller's (the try loop sets them); the bus-bar and after-sustained steps (fleet.sh) set none
  local lab=${lab-$1 step} ph=${ph-$1 step} q=${q-0}
  last=$(grep -a '^STRESS|' "$log" | tail -1 | cut -d'|' -f2-)
  TRY_HUNG="HUNG: try hang: no progress for $s s (hang timeout TRAY_REPRO_HANG_S=${HANG_S:-${TRY_LIMIT_S:-600}} s), last step '${last#*|}' at ${last%%|*}"
  # the MCE guard: a chip hung for MMIO (config ffffffff or an ARC timeout): stop the try at once, capped
  # capture, PCI remove; the slow stacks and py-spy (up to 60 s) are skipped so the remove comes well inside 40 s
  if declare -F mg_hang > /dev/null && mg_hang "try$i" "$(date +%s)" "try_stacks $pid $f $log 8; try_stop $pid" $(echo "${RK_MAP:-}" | awk 'NF { print $1 }'); then
    say "$lab  quiet $q min  $TRY_HUNG; MCE guard acted ($OUT/mce_guard/timeline.tsv)"
    fault_name_hang "try hang (no progress for $s s) in $ph ${lab#* }" "$OUT/try$i.json"; return 0
  fi
  try_stacks "$pid" "$f" "$log"
  say "$lab  quiet $q min  $TRY_HUNG; stacks in $f"
  fault_name_hang "try hang (no progress for $s s) in $ph ${lab#* }" "$OUT/try$i.json"; fault_hang $(fault_bdfs $FC_SUS)
  hold_point "try $i hang in $ph ${lab#* }" "$pid" $(fault_bdfs $FC_SUS)   # --hold: the try stays alive until STOP or timeout
  try_stop "$pid" >> "$f" 2>&1; hold_done
}
try_stop() {  # <pid>: TERM, then KILL, to the try's own process group (pgid = pid: started with setsid by try_run).
              # A process still there after KILL (stuck in the kernel) is named, not waited for
  local k n
  kill -TERM -- "-$1" 2>/dev/null; echo "$(date '+%F %T %Z') TERM to process group $1"
  for k in $(seq 10); do kill -0 "$1" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$1" 2>/dev/null && echo "$(date '+%F %T %Z') KILL to process group $1"
  for k in $(seq 30); do kill -0 "$1" 2>/dev/null || break; sleep 1; done
  kill -0 "$1" 2>/dev/null || wait "$1" 2>/dev/null
  n=$(ps -o pgid=,pid=,stat=,wchan:32=,cmd= --sid "$1")   # setsid: session = process group = $1
  if [ -n "$n" ]; then echo "still there after KILL:"; echo "$n"; say "try hang: process group $1 still there after KILL (see $OUT)"
  else echo "$(date '+%F %T %Z') process group $1 gone"; fi
  return 0
}
RK_WANT=32   # a Blackhole Galaxy: 4 trays of 8 chips
rk_map; [ -n "$KD_TRAY" ] && kd_check; fault_map   # every chip's PCI address, number and root port, before any load
idle_clocks() {  # chips left at the busy clock (after a reboot, or a program that skipped its teardown) never reach the
                 # cold quiet (<= 800 MHz): the clock check (quiet_clock.py) asks every chip above 800 MHz
                 # for its idle clock (AICLK_GO_LONG_IDLE, ARC 0x54) and waits for all, at most 5 min
  local hi; hi=$(cat "$SYS"/class/tenstorrent/*/tt_aiclk 2>/dev/null | awk '$1 > 800' | wc -l)
  [ "$hi" = 0 ] && return 0
  "${AS[@]}" "$PY" quiet_clock.py settle "before $lab" > "$OUT/idle_clocks.log" 2>&1
  say "$hi chip(s) were above 800 MHz: sent go-idle; $(grep -a '^QUIET CLOCK .*after' "$OUT/idle_clocks.log" | tail -1)"
  quiet_findings "$OUT/idle_clocks.log" "before $lab"
}
QUIET_FAILS=""; QUIET_WARNS=""
quiet_findings() {  # <log> <where>: a clock check's QUIET FAIL / QUIET WARN lines (a chip above 800 MHz after the 5 min
                    # cap, by BDF and tray 1-4) are findings; the run goes on
  local l; while IFS= read -r l; do
    if [ -n "$FLEET" ]; then fl_find "$l ($2)"; else say "$l ($2)"   # repro mode: FLEET is empty, RESULT reads these
      case $l in "QUIET FAIL"*) QUIET_FAILS+="${QUIET_FAILS:+; }$l ($2)";; *) QUIET_WARNS+="${QUIET_WARNS:+; }$l ($2)";; esac; fi
  done < <(grep -a -o -E 'QUIET (FAIL|WARN) .*' "$1" 2>/dev/null | cut -c1-600)
}
if [ "$RK_N" != "$RK_WANT" ] || [ -n "$(lost_trays)" ]; then   # a fault before any load: name it, capture before the reset
  fault_name "missing at the start" $(rk_snap | awk '$3 != "ok" { print $2 }'); fault_snapshot start-missing
fi
if [ -n "$FLEET" ]; then fl_start "$OUT"; else lk_check pre; rk_start "$OUT"; tdp_start; fi; n=$RK_N; mg_arm   # chips missing before any load: the same reset and re-check first, else exit 6 (fault_exit)
gd_read start   # per-chip GDDR telemetry after the start reset (a reset clears its counters)
heavy_logs_snapshot pre
SEL_BASE=""   # without --heavy-logs: one SEL read a run (about 9 s) that a loss in any try is read against
if [ "$HX_SUDO" = 1 ] && [ "$HEAVY" != 1 ]; then SEL_BASE=$OUT/sel_baseline.txt
  timeout 30 sudo -n ipmitool sel elist last 40 > "$SEL_BASE" 2> /dev/null || : > "$SEL_BASE"; fi
n1=$(echo $QUIETS | wc -w); n2=$(echo $AG_QUIETS | wc -w); total=$(( n1 + n2 ))
i=0; T0=$(date +%s); seen=0; chips=""; failed=""; infra=""; at=""; ccl=""; lab=""   # set per try; a hang in an earlier stage with no tries planned reaches the hangpin below with it empty
PLAN=(); for q in $QUIETS; do PLAN+=("ub0 $q"); done; for q in $AG_QUIETS; do PLAN+=("allgather $q"); done
[ -n "$FLEET" ] && fl_plan   # C8 on; no tries
rs_clock_start   # the run budget counts the test stages only, from here (setup, C2 and discovery are not counted)
fm=(--fmax 0)   # stock clock (no cap), set on the first try; nothing between the tries changes it
half_min() { [ $(( $1 % 2 )) = 0 ] && echo $(( $1 / 2 )) || echo "$(( $1 / 2 )).5"; }   # Q/2 minutes: 1 -> 0.5, 3 -> 1.5
clk() { local v; v=$(grep -a -o -m1 "AICLK $1: .*" "$OUT/try$i.log" | sed "s/^AICLK $1: //"); echo "${v:-${2:--}}"; }
est=180; for q in $QUIETS $AG_QUIETS; do est=$(( est + q * 60 + 120 )); done   # estimate: 3 min start/reset + per try its quiet + about 2 min (measured: 12 tries, 20 min quiet, ~45 min)
[ ${#PLAN[@]} -gt 0 ] && say "estimated run time: $(( est / 60 )) min for $total tries (quiet + about 2 min per try, 3 min start and reset)"
if [ -n "${TRAY_REPRO_BROKER:-$(env | grep -m1 -o '^TT_DEVICE_BROKER[A-Z_]*')}" ] && [ "$est" -gt 1500 ]; then
  say "WARNING: a device broker clamps a job to 1500 s and preempts --preemptable jobs; this run would take about $(( est / 60 )) min. Reduced form: --tries \"3\" --ag-tries \"1 1 1 1\""
fi
say "${TRAY_REPRO_RUNNER:-run.sh}: $total tries; phase 1 matmul quiet minutes: ${QUIETS:-none}; phase 2 all-gather" \
    "quiet minutes: ${AG_QUIETS:-none} (only if phase 1 is clean); logs in $OUT"
for p in "${PLAN[@]}"; do
  load=${p% *}; q=${p#* }; i=$((i + 1)); t0=$(date +%s)
  if [ "$load" = ub0 ]; then lab="matmul try $i/$n1"; ph="phase 1 (matmul)"
  else lab="all-gather try $(( i - n1 ))/$n2"; ph="phase 2 (all-gather)"; fi
  idle_clocks
  hb_start "$lab" "$OUT/try$i.log"
  say "$lab  waits: $(( q * 30 )) s cold quiet, then $(( q * 30 )) s open idle"   # Q split in half
  try_args "$i"; heavy_logs_bmc_poll_start "try$i"
  try_run "$i" "$OUT/try$i.log" "${AS[@]}" "$PY" ub0_repro.py --load "$load" "${fm[@]}" "${FL_UB0_ARGS[@]}" \
    --wait-quiet-min "$(half_min "$q")" --idle-s $(( q * 30 )) "${HX[@]}"; hb_stop
  heavy_logs_bmc_poll_stop
  s=$(( $(date +%s) - t0 )); at="$ph ${lab#* }"; at="$at (quiet $q min)"; fm=()
  say "$lab  AICLK after the quiet: $(clk 'after the quiet') | after mesh open: $(clk 'after mesh open') | before burst: $(clk 'before burst' 'not reached')"
  say "$lab  clock check: $(grep -a -o -E 'QUIET CLOCK (cold|open) quiet: [0-9]+/[0-9]+ chips at <= 800 MHz after [0-9.]+ s' "$OUT/try$i.log" | sed 's/^QUIET CLOCK //' | paste -sd'|' | sed 's/|/ | /g')"
  quiet_findings "$OUT/try$i.log" "$lab"
  gone=$(rk_snap | awk '$3 != "ok" { print $2 }' | tr '\n' ' '); named=$(rk_loss_chips "$OUT/try$i.log")
  k=$(echo $gone $named | rk_list)
  if [ -n "${gone// /}" ] || [ -n "${named// /}" ] || [ "$rc" = 3 ]; then
    seen=1; chips="$gone $named"; lost=$(lost_trays)
    say "$lab  quiet $q min  chips lost: $(echo $chips | tr ' ' '\n' | sort -un | grep -c .)  (chips ${k:-not named}${lost:+; not answering: $lost}; ${s} s; see $OUT/try$i.log)"
    break
  fi
  if [ -n "$TRY_HUNG" ]; then ccl=$TRY_HUNG; break; fi   # named, captured and stopped by try_hung
  case $rc in
    0) say "$lab  quiet $q min  chips lost: 0  (${s} s)" ;;
    4) ccl=$(grep -a -o -m1 'CCL hang: .*' "$OUT/try$i.log" | cut -c1-300); ccl=${ccl:-CCL hang: chip unknown}
       say "$lab  quiet $q min  $ccl - not a tray drop  (${s} s; see $OUT/try$i.log)"; break ;;
    *) if [ -n "$TRY_INFRA" ]; then say "WARNING $lab not run, $TRY_INFRA  (${s} s; see $OUT/try$i.log)"; infra="$lab not run ($TRY_INFRA)"; break; fi
       say "$lab  quiet $q min  the try failed, rc $rc, no chip loss seen  (${s} s; see $OUT/try$i.log)"
       failed="$lab failed (rc $rc)"; break ;;
  esac
done
# give up the brokered device within 30 s of a queued job of another owner (lib/broker_yield.sh)
[ -n "$FLEET" ] && { rk_tmark "$FL_OUT/load_times.tsv" begin loads; fl_loads; rk_tmark "$FL_OUT/load_times.tsv" end loads; by_stop; }   # stage 4, the bus-bar stage (with stage 5 numeric verify inside it), then stage 6, C5
# a fault (lib/fault_capture.sh): name it, then capture before the reset. A CCL hang:
# the NOC snapshot of ub0_repro.py's suspects (none: every chip); a loss: a snapshot and the NOC one
if [ "$seen" = 1 ]; then
  fault_name "lost in $at" $chips
  declare -F mg_loss > /dev/null && { mg_loss "lost in $at" $(fault_bdfs $chips) || true; }   # the guard's cut before any read
  dr=${HEAVY_DEVICE_READS:-auto}; declare -F mg_loss_reads > /dev/null && ! mg_loss_reads $(fault_bdfs $chips) && dr=0   # a lost chip left on PCI: no device read
  ! rs_step "fault capture (lost in $at)" || { HEAVY_DEVICE_READS=$dr fault_snapshot loss; HEAVY_DEVICE_READS=$dr heavy_logs_on_hang $(fault_bdfs $chips); }
elif [ -n "$ccl" ] && [ -z "$TRY_HUNG" ]; then   # a try hang was captured while it still hung (try_hung)
  fault_name_hang "CCL hang in $at" "$OUT/try$i.json"; ! rs_step "fault capture (CCL hang in $at)" || fault_hang $(fault_bdfs $FC_SUS)
  hold_point "CCL hang in $at" "" $(fault_bdfs $FC_SUS); hold_done   # --hold: before the reset (ub0_repro.py already exited)
elif [ -n "$failed" ]; then
  # the chips the try's own error names; else each chip probed alone; else UNATTRIBUTED with every candidate.
  fault_attr "$failed" 1 "$OUT/try$i.log" "$OUT/try$i.json" || :
  fault_eth "$OUT/try$i.log"
  ! rs_step "fault capture ($failed)" || fault_snapshot failed
fi
! rs_step "pre-reset heavy-log snapshot" || heavy_logs_snapshot pre-reset
[ -n "$TRY_HUNG" ] && [ -z "${FC_SUS// /}" ] && try_hangpin "$i" "${lab:-before the tries} hang"   # the chip that stopped
eth_fec_judge   # off-box ETH FEC over the run: a far-end reset is one info line, one standing-out cable a WARN
# the numeric lines: per chip over every matmul try, before the reset; never change the result
[ -n "${QUIETS// /}" ] && python3 ./numeric_report.py "$OUT"/try*.json | tee "$OUT/numeric.txt"   # --mode: fl_report
pinpoint_numeric   # only after a NUMERIC FAIL (a --mode=repro try or stage 5 numeric verify): which part of the chip, before the reset
! rs_step "host link check (post)" || lk_check post   # host link after the load, before the end reset: links, AER rise over the run (lib/link_check.sh)
rs_hl_end
gd_read end   # per-chip GDDR telemetry and its rise over the run, before the end reset clears it
cpld_drop "$OUT" "$seen"   # the CPLD 12 V state, live, before the reset
tdp_applied   # --tdp: each chip's limit read back before the end reset reverts it
if [ -n "$FLEET" ]; then fl_end; else rk_end "$OUT" "$chips" "$seen"; fi; declare -F mg_report > /dev/null && mg_report
# a warm reset (C2, C9, C9b) that left a chip missing is a FAIL even when the end verdict is clean
[ -n "${RK_RESET_FAILS:-}" ] && [ "$RK_RC" = 0 ] && { RK_RC=14; _w=$(rk_resfail_what); [ -n "$_w" ] || _w="RESET UNATTRIBUTED FAIL (the reset failed and names no chip)"; RK_WHAT="${RK_RESET_LBL%% (*} FAILED: $_w"; RK_WHAT=${RK_WHAT%: }; }
heavy_logs_snapshot post-reset
bh_poll_stop   # the heat poll's summary line (by tray 1-4)
[ -n "$HL_OUT" ] && { FC_DONE=1; heavy_logs_bmc_poll_stop; heavy_logs_finish; }   # index.md, before the verdict lines
tdp_end   # --tdp: requested and read-back limits, the default restored (the end reset, else a short open at 0)
[ -n "$FLEET" ] && fl_verdict   # RESET NOT EXERCISED, a pair left hung (FL_RESULT, rc 7) or INCOMPLETE
[ -n "$FLEET" ] && pvc_verdict   # a host crash under an earlier run's power virus: FAIL (rc 8), never a pass
# the RESULT line goes through a file so --mode can name a failed check in it (fl_result_say); RK_RC stays set here
say "$(rs_time_line)"   # setup and test-stage times apart
say "hang timeout: TRAY_REPRO_HANG_S=$HANG_S s with no progress cuts a stage as HUNG (no run-time budget cuts or skips a stage; longest healthy silence on record 296 s)"
{ if [ -n "${FL_RESULT:-}" ]; then echo "RESULT: $FL_RESULT"
else case $RK_RC in
  0) if [ -n "$ccl" ]; then echo "RESULT: $ccl, at $at; after the reset: no loss"; RK_RC=7
     elif [ -n "$failed" ]; then echo "RESULT: $failed; after the reset: no loss"; RK_RC=1
     elif [ -n "$infra" ]; then echo "RESULT: INCOMPLETE: no loss, but $infra"; RK_RC=9
     elif [ -z "$FLEET" ] && [ -n "${FW_FAIL:-}" ]; then   # C0 firmware FAIL: rc 8 as in --mode
       echo "RESULT: FIRMWARE FAIL: $(echo "$FW_FAIL" | grep -c .) chip(s) off the others' firmware (C0), see fault.txt; no loss ($total/$total tries)"; RK_RC=8
     elif [ "$KD_ON" = 1 ]; then echo "RESULT: KNOWN-DROP: tray $KD_TRAY missing (--known-dropped-tray=$KD_TRAY, not tested; $(kd_end) answer at the end); no loss on the other 24 chips ($(( ($(date +%s) - T0) / 60 )) min)"
     elif [ -n "$QUIET_FAILS" ]; then echo "RESULT: $QUIET_FAILS; after the reset: no loss"; RK_RC=1   # a QUIET FAIL is a finding in repro mode
     else echo "RESULT: no loss ($total/$total tries, $(( ($(date +%s) - T0) / 60 )) min)${QUIET_WARNS:+; $QUIET_WARNS}"; fi ;;
  1) echo "RESULT: $RK_WHAT${failed:+; $failed}" ;;
  9) if [[ $RK_WHAT != "INCOMPLETE: no loss"* ]]; then echo "RESULT: $RK_WHAT${failed:+; $failed}, at $at"
     elif [ -n "$ccl" ]; then echo "RESULT: $ccl, at $at; after the reset: $RK_WHAT"; RK_RC=7
     elif [ -n "$failed" ]; then echo "RESULT: $failed; after the reset: $RK_WHAT"; RK_RC=1
     else echo "RESULT: $RK_WHAT ($total/$total tries, $(( ($(date +%s) - T0) / 60 )) min)"; fi ;;
  14) if [ -n "$ccl" ]; then echo "RESULT: $ccl, at $at; after the reset: $RK_WHAT"; RK_RC=7
      elif [ -n "$failed" ]; then echo "RESULT: $failed; after the reset: $RK_WHAT"; RK_RC=1
      else echo "RESULT: $RK_WHAT ($total/$total tries, $(( ($(date +%s) - T0) / 60 )) min)"; fi ;;
  3) if [ "$seen" != 1 ] && [ -z "$ccl$failed" ] && [ -z "$FLEET" ] && [ -n "$(echo "${RK_AFTER:-}" | awk '$3 != "ok"')" ]; then
       echo "RESULT: $(rk_end_loss "$RK_RESET_LOG" "$RK_AFTER")"   # every try was clean: not a tray drop in a try
     else echo "RESULT: $RK_WHAT, at $at"; fi ;;
  *) echo "RESULT: $RK_WHAT, at $at" ;;
esac; fi; } > "$OUT/result.line"
if [ "$KD_ON" = 1 ] && { [ "$RK_RC" = 9 ] || grep -aq '^RESULT: INCOMPLETE' "$OUT/result.line"; }; then   # INCOMPLETE (rc 9) wins over
  printf '%s; %s\n' "$(cat "$OUT/result.line")" "$(kd_tag)" > "$OUT/result.line.kd" && mv "$OUT/result.line.kd" "$OUT/result.line"   # the known drop, named too
fi
[ -n "$FLEET" ] && { [ "$RK_RC" = 9 ] || [ -n "$RS_BUDGET_INC" ]; } && fl_vd_sum fin   # finished stages' VERDICT lines, then what was not reached, before RESULT: INCOMPLETE
_r=$(if [ -n "$FLEET" ]; then fl_result_say "$OUT/result.line"; else cat "$OUT/result.line"; fi); echo "$_r; $(pp_tag)$(tdp_tag)${SR_TAG:-}${FL_PV_SKIP:+; power virus SKIPPED (another process held a Tenstorrent device, or the holder scan was incomplete; see the WARNING above)}${TRAY_STUB:+; STUB MODE: fake devices, not a real test}"; RS_DONE=1   # SR_TAG: the sudo status (lib/sudo_route.sh); power_policy on the RESULT line; tdp_tag: --tdp refused by a gate
[ -z "$FLEET" ] && [ -n "${RK_RESET_FAILS:-}" ] && echo "$RK_RESET_FAILS"   # --mode prints them with its findings
pinpoint_summary   # the NUMERIC-PINPOINT VERDICT lines, if any
gd_summary   # GDDR WARN lines (new uncorrected errors, a corrected-error rise) and one GDDR: line
[ -n "$FLEET" ] && fl_report   # findings and the per-tray lines; the three verdict lines stay last
lk_report   # without --mode the host-link findings (rc 8 for a FAIL); the fix per tray with a PCIe link FAIL
if [ -n "${SR_LINE:-}" ]; then echo "$SR_LINE"; fi   # the startup sudo warning again in the summary
if [ -n "${TRAY_STUB:-}" ]; then echo "$STUB_WARN"; fi   # --stub: the first line again in the summary
if [ "$KD_ON" = 1 ]; then   # never a PASS: the known drop is the verdict when nothing else is wrong
  [ "$RK_RC" = 0 ] && RK_RC=12
  [ -n "${RK_RESET_KDONLY:-}" ] && RK_RESETS="$RK_RESETS; tt-umd failed only on tray $KD_TRAY (the known drop)"
  RK_TRAYS="tray $KD_TRAY (known drop, --known-dropped-tray=$KD_TRAY)$([ "${RK_TRAYS:-None}" = None ] || echo "; $RK_TRAYS")"
fi
( exit "$RK_RC" ); vd_exit   # VERDICT before the three verdict lines, so "Trays dropped:" stays the last line
rk_last; exit "$RK_RC"
