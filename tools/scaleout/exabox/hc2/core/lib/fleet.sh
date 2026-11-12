#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
declare -F pg_run > /dev/null || . "$(dirname "${BASH_SOURCE[0]}")/pgroups.sh"   # setsid children are recorded; pg_run
declare -F rk_umd > /dev/null || . "$(dirname "${BASH_SOURCE[0]}")/umd_index.sh"   # open a chip by its UMD index
declare -F sudo_t > /dev/null || . "$(dirname "${BASH_SOURCE[0]}")/sudo_l.sh"   # exact sudo -n -l probes
# lib/fleet.sh: the fleet profile of run.sh (plain run.sh = --mode=quick, or --mode=full; --mode=sustained is retired;
# plan notes/fleet-test-plan.md). Sourced by run.sh only; defines functions, runs nothing by itself. Stages are numbered 1-7 in run
# order in fleet/stages.tsv: 1 gate (P0), 2 pre-run checks (P1: C0, C1, C3), 3 start reset (C2),
# 4 bus-bar stage (busbar), 5 numeric verify (inside 4: chunk-end readbacks, G3, the heavy chunk), 6 end reset (C9),
# 7 per-link sweep (on a fault only). fl_pn, the dry run, Phase times, tray_why.py and the README read that table. run.sh calls:
#   fl_init                 after the options: the stage table (--dry-run prints it and exits 0)
#   fl_start <out>          instead of rk_start: P0 gate, C0 preflight, C1 pre snapshot, C3 count, C2 start reset
#   fl_plan                 after fl_start: C8 on; no tries run (every mode goes straight to the bus-bar stage)
#   fl_loads                then, only with no loss: the bus-bar stage (one load definition,
#                           FL_LOAD_*, every load stage builds from), then C5 (stage 6); its pairs
#                           rerun in the per-link sweep (stage 8) on a fault, and in --repro link-a.
# --mode=full runs plain run.sh's steps in the same order, only longer: its last
# CCL-mix chunk is FLEET_LONG_MIN (25) min of the one load (1, 2, 3, 25 min chunks; plain 1, 2, 3, 4). Each chunk is followed by 10 s of forced quiet (synchronized,
# nothing enqueued, no data moves), so every restart is a whole-galaxy di/dt step, with an on-device numeric check read
# back at each chunk end (stage 5).
#   fl_end                  instead of rk_end: C8 stop, C9 end reset + recheck (or a read-only recheck when no reset
#                           may run), C1/C3 after, SUSPECT pair reruns
#   fl_report               after the RESULT line, before rk_last: findings, then one PASS/FAIL line per tray
# On a fault: a load hang or a CCL/all-gather failure (the chunks' h3
# part) sets FL_PIN. Its chip state is captured before any reset (a hang: on_hang.sh's tt-triage while the job is alive;
# a hang or a failure: fl_capture's lspci/AER/SEL/dmesg and heavy_logs_on_hang's exalens); the C9 reset runs through the same gates; then
# fl_sweep (stage 8) runs the pairwise per-link all-gather (fl_c5, fl_pair) on the reset chips (FLEET_SWEEP_S 300 s budget, op timeout
# FLEET_PAIR_HANG_S, on_hang.sh per hung link) and prints "PINPOINT LINK HANG|FAIL <bdf> (tray N) e<ch> <-> <bdf>
# (tray N) e<ch>", else "no single link" (every link swept) or "sweep incomplete". A refused reset sweeps with no reset if every chip answers, else "PINPOINT skipped: reset refused (<gate>)" names the chips;
# --hold never sweeps. A run with no fault runs no sweep.
# Gates: fleet/gate.sh (this host idle but for this run's process group) before every phase; a busy gate stops the
# run with no further load and no reset (exit 10). Every reset (C2, C9, rk_start's start reset) runs whatever the QSFP
# far hosts do (resets do not hurt neighbours); run.sh takes no peer snapshot around a reset.
# No variable or flag skips a reset for a peer or cabled-neighbour reason: only this host's own
# serving gate (TRAY_REPRO_RESET_GATE: this host serves live H3, or another person's process or run holds its chips)
# skips one ("SKIPPED-GATE: <why>"; the run is INCOMPLETE, exit 9, "RESET NOT EXERCISED" before the RESULT line);
# a hang left with no reset names the hung chips in the RESULT (exit 7) and in FLEET_HUNG_FILE, which stops the next run
# on this boot until a reset may run (exit 6).
# Report lines name chip BDF and tray (plan section 4). WARN lines are report-only; any other finding fails its tray.
# env (defaults): FLEET_HUNG_FILE (/var/tmp/tray-test-hung.<uid>), FLEET_PAIR_S 10, FLEET_PAIR_HANG_S 30, FLEET_COMPILE_S 300, FLEET_PAIR_COMPILE_S 60 (pairs, launch 0),
#   FLEET_LOAD_TRAYS unset (N[,N..], trays 1-4: the chunk_load.py stages, bus-bar mix and heavy chunk and the
#   sustained chunks, load only those trays, each on its own submesh, the bus-bar step is skipped; other stages still load every chip),
#   FLEET_C7_S unset (quick only: set it to run one chunk of that many load s, FLEET_GAP_S quiet after it, instead of chunks of 1, 2, 3, 4 min), FLEET_CHUNK_MIN 1,2,3,4,15 / FLEET_GAP_S 10 (full), FLEET_C7_HANG_S 120, FLEET_POWER_POLL_S 2,
#   FLEET_POLL_TAIL_S 2 (the wait for one more power poll row after a load; only lib/stubhost.sh sets it),
#   FLEET_LSPCI_POLL_S 10, FLEET_TELEM_S 5, FLEET_NUMERIC_EVERY 500, FLEET_ON_HANG (fleet/on_hang.sh), FLEET_PROC/FLEET_DEVD
#   (gate.sh test hooks); thresholds from the T5 fleet baseline (results/fleet-test/baseline/SUMMARY.md; AER per
#   results/aer-outliers/SUMMARY.md): C1 FLEET_AER_RATE_WARN 6 (correctable/h over the run, at least
#   FLEET_AER_MIN 2 events), FLEET_AER_BOOT_WARN 25 / FLEET_AER_7D_WARN 100 (kernel journal events per chip or root port),
#   FLEET_SEL_BUSCOR_WARN 20, FLEET_SEL_FRB2_WARN 3; C8 FLEET_TEMP_WARN_C 75, FLEET_POWER_WARN_W 320, FLEET_CURR_WARN_A 400,
#   FLEET_VCORE_WARN_V 0.66. Correctable AER is WARN only (rates swing 0-744/h per boot with no effect on work); FAIL is an
#   uncorrectable AER event during the run, a PCIe link speed or width drop, a missing chip or a load fault. A new SEL FRB3
#   during the run is one SEL WARN per tray (an SEL row alone is no 12 V evidence). C1's history (7-day journal AER/MCE, SEL, from before the run) is WARN only, with the last
#   event's time. FLEET_HISTORY_SINCE (unset; run.sh --since): the host's last repair; history
#   before it is one "C1 INFO tray N" line per tray (history_info.txt, report only, never a finding).
FL_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../fleet
. "$FL_DIR/../lib/vcore.sh"   # vc_start/vc_stop/vc_report: the heavy chunk's read-only VCORE sampler
. "$FL_DIR/../lib/thermal.sh"   # th_start/th_stop/th_chunk/th_summary: per-chip and per-GDDR-channel temps and the AICLK throttle cause through every load chunk
. "$FL_DIR/../lib/psu_check.sh"   # fl_psu: the PSU read before and after each load stage
. "$FL_DIR/../lib/psu_fast.sh"   # pf_*: per-PSU input power every 0.5 s over the bus-bar stage
. "$FL_DIR/../lib/pv_crash.sh"   # pvc_open/pvc_check: a host crash under the power virus is that box's FAIL
. "$FL_DIR/../lib/hsc_check.sh"   # fl_hsc_before/after: each tray's HSC UV latches and min VIN around each load leg
. "$FL_DIR/../lib/broker_yield.sh"   # by_start/by_check/by_stop: give the brokered device up to a queued job
. "$FL_DIR/../lib/power_policy.sh"   # pp_tag: power_policy on the run summary line
declare -F rk_can_glx > /dev/null || . "$FL_DIR/../lib/glxcan.sh"   # run.sh defines it; tests source fleet.sh alone
FL_STOP=""; FL_BB_INC=""; FL_BB_TRY=0 FL_BB_NJ=0 FL_BB_RAN=0 FL_C7_NJ=""; FL_HANG_PIN=""; FL_FABSTALL=""; FL_FAB_HIT=""; FL_FAB_RETRY=""; FL_C3STOP=""; FL_C3STOP_CHIPS=""; FL_C3_RC=""; FL_PGDROP=""; FL_NORESET=""; FL_HUNG=""; FL_LINKS_DOWN=""; FL_SUSPECT=(); FL_SKIPPED=(); FL_C8_PID=""; FL_PW_PID=""; FL_AER_PID=""; FL_RESET_SKIPPED=(); FL_LEFT_HUNG=""; FL_LEFT_WHY=""; FL_RESULT=""; FL_PIN=""; FL_SWEEP_HP=""; FL_SWEPT=(); FL_C5_INC=(); FL_C2_NONE=""
FL_SETUP_INC=""; FL_AF_INC=""; FL_LINKA_INC=""; FL_AF_LINE=""; FL_LINKA_LINE=""   # INCOMPLETE causes: one left in the environment or from an earlier run would grade a clean run rc 9
# phase | what | quick s | full s   (plan section 3; stage numbers and names: fleet/stages.tsv)
# Stage list (results/t822/stage-table.md):
# every mode opens with the bus-bar stage right after C2 (FLEET_BUSBAR_ORDER, default qb,mix,step,heavy in every mode): first two
# quiet-to-burst steps (qb, placed by design: 60 s clock-checked quiet + 30 s lofi_all matmul burst on all chips, x2),
# then the CCL mix (fleet/chunk_load.py
# h3shape: H3 all-gather/reduce-scatter + mm/SFPU/DRAM/PCIe, no heavy part, one command queue: e9831c5's chunk, which dropped
# one quad host's tray 3 at 572 s) in 1, 2, 3, 4 min chunks with a clock-checked 10 s quiet
# and numeric readback after each (after chunk 1 also G3: FLEET_BUSBAR_RW_S s of x/w/y-in-DRAM C4n-size
# matmul with readbacks, before that quiet: a quad host's 03:00.0 tray 1 DRAM fault, which C4n caught at 21-22 min and the
# bus-bar numeric at 926 s; G3's catch time is measured),
# then the cold-quiet-to-full-load step (10 s quiet + 90 s lofi_all burst), then in quick and full (
# full runs everything quick runs) one heavy chunk (90 s; 750 s in full), the stage 5 DIDT numeric chunk: the DIDT
# bf16 HiFi2 16k matmul then the bf8 LoFi 16k matmul (FLEET_DIDT_OPS 16k_hifi2,16k_bf8_lofi), back to back in each launch, each output
# compared bit-exactly with its launch-0 reference per launch and across chips; it replaced this chunk's lofi_all, which every mix chunk still runs checked.
# full runs the same parts; only its last mix chunk is longer (25 min). long (one separate 25 min chunk) is opt-in.
# the mix, heavy chunk and long chunk all run the one load (fl_load_args: lofi_all x16 + SFPU in parallel + the
# H3 CCL/mm/SFPU/DRAM/PCIe parts, 2 cqs), and qb and step the same heavy op (fl_burst_args).
# Then C5, the pair link test (stage 6; kept); the per-link sweep (stage 8) reruns its pairs on a fault. The old opt-in stages (C4v, the C4 tries, C6, C6b, qstep) found nothing the chunks do not and are gone.
# P1 45 s: it ran in 9 s (t166); the T5 reads add ~22 s (on a quad host: journal boot 0.8, 7-day 4.9, SDR 8.8, SEL 7.0 s), ~31 s; 45 s again
# with G3 in C4d's old slot, so quick stays under 30 min
# busbar G3 +35 s: 15 s of load + readbacks, wall cap FLEET_BUSBAR_RW_WALL_S 35 s (+ 15 s for its inputs until
# G3 took 20.0-29.5 s end to end, inputs included, in 10-30's runs); G3b +13 s right after it: the DRAM byte/DBI
# pattern check (dram_dbi.py, R03), wall cap FLEET_BUSBAR_DBI_WALL_S 10 s (60 s in --mode=full) + 3 s; N1 +13 s right
# after G3b: the L1 march (l1_march.py), wall cap FLEET_BUSBAR_L1_WALL_S 10 s (60 s in full) + 3 s; G3 takes
# C4d's old slot (same x/w/y-in-DRAM shape, on the bus-bar mesh after chunk 1; C4d itself is gone)
# P0 30 s, C9 150 s, the mix/step/heavy fixed parts: re-based on 161 plain runs since 10-24 (C9 median 83 s, p90 111 s;
# mix 706-716 s, step 146-160 s, heavy 122-157 s on 58 runs) to give the two quiet-to-burst steps their 240 s inside 30 min
FL_TABLE='P0|gate (read-only), setup|30|30|30
P1|C0 preflight, C1 evidence snapshot + 7-day journal/SEL/SDR, C3 ETH/QSFP port count + tt-smi ETH|45|45
C2|start glx_reset (tt-smi -glx_reset) + rescan + recount + C3 (never tt-smi -r)|120|120
busbar|bus-bar stage, first load in every mode (FLEET_BUSBAR_ORDER qb,mix,step,heavy in every mode): 2 x (60 s clock-checked quiet + 30 s burst of the one load definition); then the max load (FL_LOAD_*: lofi_all x16 + SFPU in parallel + H3 CCL/mm/SFPU/DRAM/PCIe) 1, 2, 3, 4 min chunks, 10 s clock-checked quiet + numeric after each, after chunk 1 also G3; then 10 s quiet + 90 s step + numeric; heavy chunk (90 s; 750 s in full); full: the same, last mix chunk 25 min; tray power poll, C8|960|960
numeric|numeric verify inside stage 4: numeric readback after each chunk and the step, G3, the G3b DRAM byte/DBI pattern check and the N1 L1 march after chunk 1, 90 s (750 s in full, ~2 s in smoke) DIDT numeric heavy chunk (bf16 HiFi2 then bf8 LoFi 16k matmuls, bit-exact per launch and across chips, each checked output also through one sharded SFPU op per launch, exp/reciprocal/sqrt/gelu in turn, voted across chips with the core x,y named)|0|0
C5|pairwise local ETH-link all-gather, edge-coloured parallel rounds (70-82 s on 7 legs; the C5 link hang)|120|120
C9|end reset + recheck, C1/C3 after, SUSPECT pair reruns|150|150
sweep|on a fault only (a hang or CCL failure), after the C9 reset: per-link all-gather, up to FLEET_SWEEP_S|0|0'
fl_g3() { [ "${FL_BB_RW:-0}" -gt 0 ] && [ "$FL_BB_RW" -le "${FL_BB_CHUNKS:-0}" ] && case ,${FL_BB_ORDER:-}, in *,mix,*) ;; *) false ;; esac; }   # G3 runs
# the one load definition ('just bring max load to the bus bar stage, make sure it's ACTUALLY the max
# the hardware will do on all units, and that's the sustained stage as well'): every load stage builds from it (fl_init sets FL_LOAD_*;
# tests/test_one_load.sh checks every chunk_load.py and ub0_repro.py load call in this file uses these two)
fl_load_args() {  # [<parts>]: chunk_load.py's args for the load; <parts> narrows its parts (the heavy chunk: heavy)
  echo --parts "${1:-$FL_LOAD_PARTS}" --cqs "$FL_LOAD_CQS" --heavy "$FL_LOAD_HEAVY" --heavy-n "$FL_LOAD_HEAVY_N" --burst-gap-ms "$FL_LOAD_GAP_MS" \
    --burst-steps "$FL_LOAD_STEPS" --heavy-pcie-mb "$FL_LOAD_PCIE_MB" --pcie-every "$FL_LOAD_PCIE_EVERY" --sfpu-par "$FL_LOAD_SFPU_PAR" ${FL_LOAD_SFPU_ACT:+--sfpu-act "$FL_LOAD_SFPU_ACT"} \
    ${FL_LOAD_VARIANT:+--variant "$FL_LOAD_VARIANT"}
}
fl_burst_args() {  # ub0_repro.py's args for the same heavy op in one burst (qb, step, qstep: its lofi_all carries its own PCIe,
  # all-gather and DRAM extras), with the SFPU in parallel like the chunks
  echo --heavy "$FL_LOAD_HEAVY" --burst-gap-ms "$FL_LOAD_GAP_MS" --burst-steps "$FL_LOAD_STEPS" --sfpu-par "$FL_LOAD_SFPU_PAR" ${FL_LOAD_SFPU_ACT:+--sfpu-act "$FL_LOAD_SFPU_ACT"} \
    ${FL_LOAD_VARIANT:+--variant "$FL_LOAD_VARIANT"}
}
fl_load_plan() { echo "$FL_LOAD_HEAVY x$FL_LOAD_HEAVY_N per launch, $FL_LOAD_GAP_MS ms gap after $FL_LOAD_STEPS, SFPU in parallel $FL_LOAD_SFPU_PAR${FL_LOAD_SFPU_ACT:+ ($FL_LOAD_SFPU_ACT)}, $FL_LOAD_PCIE_MB MB PCIe, parts $FL_LOAD_PARTS, $FL_LOAD_CQS cqs${FL_VARIANT_WARN:+ ($FL_VARIANT_WARN)}"; }
fl_swap_plan() {  # the TRAYTEST_MIX_SWAP part, for the plan line
  if [ "$FL_MIX_SWAP" = heavy ]; then echo "$FL_SWAP_S s of the heavy matmul load ($FL_DIDT_OPS) in place of the last mix chunk (TRAYTEST_MIX_SWAP=heavy, trial)"
  else echo "$FL_SWAP_S s of max-util and the power virus ($FL_SWAP_SEGS) in place of the last mix chunk (TRAYTEST_MIX_SWAP=virus, trial)"; fi
}
fl_bb_plan() {  # the bus-bar stage's parts in FLEET_BUSBAR_ORDER order, for the dry run and the header line
  local x o="" order=${FL_BB_ORDER:-}; for x in ${order//,/ }; do case $x in   # unset before fl_init (a busy P0 gate exit): empty plan
    qb) o="$o${o:+, then }$FL_BB_QB_N x ($FL_BB_QB_Q s clock-checked quiet + $FL_BB_QB_B s $FL_LOAD_HEAVY burst)" ;;
    mix) o="$o${o:+, then }the load ($(fl_load_plan)), $([ -n "$FL_C7_LOAD" ] && echo "one $FL_C7_LOAD s chunk" || echo "$FL_BB_MIN min chunks"), $FL_BB_GAP_S s clock-checked quiet after each$(fl_g3 && echo ", G3 ${FL_BB_RW_S} s x/w/y-in-DRAM readback after chunk $FL_BB_RW$([ "$FL_BB_DBI_WALL" -gt 0 ] && echo ", then G3b ${FL_BB_DBI_WALL} s DRAM byte/DBI pattern readback")$([ "$FL_BB_L1_WALL" -gt 0 ] && echo ", then N1 ${FL_BB_L1_WALL} s L1 march")")"; [ "${FL_SWAP_S:-0}" -gt 0 ] && o="$o, then $(fl_swap_plan)" ;;
    step) o="$o${o:+, then }${FLEET_BUSBAR_QUIET_S:-10} s quiet + ${FLEET_BUSBAR_BURST_S:-90} s $FL_LOAD_HEAVY step" ;;
    heavy) o="$o${o:+, then }one $FL_BB_HV s stage 5 DIDT numeric chunk ($FL_DIDT_OPS back to back, each output checked bit-exact and through a sharded SFPU op voted across chips; in its last $FL_DIDT_X_S s also $FL_DIDT_X_OPS), $FL_BB_GAP_S s clock-checked quiet" ;;
    long) o="$o${o:+, then }one $FL_LONG_MIN min long chunk of the load$([ "$FL_LONG_PRE" -gt 0 ] && echo " after a $FL_LONG_PRE s pre-ramp"), $FL_BB_GAP_S s clock-checked quiet (12-14 kW)" ;;
    rw) o="$o${o:+, then }$FL_BB_RW_S s x/w/y-in-DRAM matmul on all chips (numeric_dram.py, C4d shape), a readback every $FL_BB_RW_CK s of load, wall cap $FL_BB_RW_WALL s" ;;
    code) o="$o${o:+, then }code readback (exalens, no load: each core's firmware and kernel code in L1 by strong-majority vote across chips, then a re-read; ${FL_BB_CODE_B} s budget, $FL_BB_CODE_SCOPE scope${FLEET_CODE_ELF_DIR:+, ELF compare}, $([ "$FL_BB_CODE_SCOPE" = smoke ] && echo "2 Tensix + 1 ETH cores per chip" || echo "the next run carries on from the last core read"))" ;;
    maxutil) o="$o${o:+, then }max-util (tt-metal unit_tests_didt_max_util, FPU 92 % DRAM 92 % duty 100 % on every chip, prebuilt archive fleet/maxutil.pin): $FL_BB_MU_LOAD s ($FL_BB_MU_MODE) after ${FL_BB_MU_SETUP} s setup" ;;
    pv) if [ "${FL_PV_SWAP:-}" = maxutil ]; then o="$o${o:+, then }max-util in place of the power virus (FLEET_PV_SWAP=maxutil: this box failed the power virus repeatedly): $FL_BB_MU_LOAD s ($FL_BB_MU_MODE)"
        else o="$o${o:+, then }the power virus (run-ttx single-core-matrix-pv-fp32-noc.ttx broadcast on every chip, fleet/pv.pin; a load only, no numeric check; --tdp does not cap it): $FL_BB_PV_LOAD s, stopped early at a brick >= $FL_PV_BRICK_C C or a feed > $FL_PV_FEED_A A (chips' power never stops it), then a glx_reset and recount (tt-smi -glx_reset with its root-port rescan) before C5 and every later tt-metal step"; fi ;; esac; done; echo "$o"
}
fl_numeric() { grep -q -- '--numeric-every' ./ub0_repro.py 2>/dev/null; }
# --mode=smoke: run.sh maps it to FLEET=quick with FL_SMOKE=1, so it runs plain run.sh's
# steps in the same order through the same code; fl_smoke only sets the duration knobs from one scale knob, FLEET_SMOKE_S
# (1-60, default 2 s): every mix chunk (still 4 of them), quiet-to-burst quiet and burst, step quiet and burst,
# heavy chunk, clock-checked quiet, G3 readback, G3b wall cap, C5 pair round and the post-chunk sysfs peak poll window is S s,
# never more than plain's own (quiet 10, G3 15, G3b 10, pair 10, peak poll 4 s); G3b still runs its first pass. A knob already set wins.
# Unchanged: the 5 min quiet clock cap, the 50 ms burst gap, the resets, the hang timeout and every timeout.
fl_smoke() {
  [ "${FL_SMOKE:-0}" = 1 ] || return 0
  local s=${FLEET_SMOKE_S:-2} m
  [[ $s =~ ^[1-9][0-9]*$ ]] && [ "$s" -le 60 ] || { echo "FLEET_SMOKE_S takes whole seconds 1-60 (got '$s')" >&2; exit 1; }
  m=$(awk -v s="$s" 'BEGIN { printf "%g", s / 60 }')
  _s() { echo $(( s < $1 ? s : $1 )); }   # S, never above plain's own default
  [ -z "${FLEET_CHUNK_MIN:-}" ] && FL_SMOKE_CM=1   # smoke's own chunks: fl_init drops the last one where plain does (max-util)
  : "${FLEET_CHUNK_MIN:=$m,$m,$m,$m}" "${FLEET_GAP_S:=$(_s 10)}"
  : "${FLEET_BUSBAR_QB_QUIET_S:=$(_s 60)}" "${FLEET_BUSBAR_QB_BURST_S:=$(_s 30)}" "${FLEET_BUSBAR_QUIET_S:=$(_s 10)}" "${FLEET_BUSBAR_BURST_S:=$(_s 90)}"
  : "${FLEET_BUSBAR_HEAVY_S:=$(_s 90)}" "${FLEET_BUSBAR_HEAVY_MIN_S:=$(_s 30)}" "${FLEET_BUSBAR_RW_S:=$(_s 15)}" "${FLEET_BUSBAR_DBI_WALL_S:=$(_s 10)}" "${FLEET_BUSBAR_L1_WALL_S:=$(_s 2)}" "${FLEET_PAIR_S:=$(_s 10)}"
  : "${FLEET_FASTPOLL_S:=$(_s 4)}"
  unset -f _s
  export FLEET_CHUNK_MIN FLEET_GAP_S FLEET_BUSBAR_QB_QUIET_S FLEET_BUSBAR_QB_BURST_S FLEET_BUSBAR_QUIET_S FLEET_BUSBAR_BURST_S \
    FLEET_BUSBAR_HEAVY_S FLEET_BUSBAR_HEAVY_MIN_S FLEET_BUSBAR_RW_S FLEET_PAIR_S FLEET_FASTPOLL_S \
    FL_SMOKE   # the N1 row's smoke flag: its short-coverage line is INFO in smoke
}
fl_mode() { [ "${FL_SMOKE:-0}" = 1 ] && echo smoke || echo "$FLEET"; }   # the mode's name in the dry run and the log
# TRAYTEST_MIX_SWAP (a trial): the swap part's segments, time and WARNING. virus alternates the weaker
# max-util load (FLEET_SWAP_MU_S, 180 s) and the heavier power virus (FLEET_SWAP_PV_S, 240 s) over the swapped chunk's
# seconds, the last segment cut to fit; each power virus segment ends with the shared glx_reset and recount (fl_bb_pv, rk_reset)
fl_swap_init() {
  [ -n "$FL_MIX_SWAP" ] || return 0
  # a site max-util skip (FL_MU_SKIP: a site's max-util ban, which skips the power virus too) refuses
  # virus here, before any device open, with the skip's own WARNING; heavy runs no max-util or power virus and stays allowed
  [ "$FL_MIX_SWAP" = virus ] && [ -n "${FL_MU_SKIP:-}" ] \
    && { echo "WARNING: TRAYTEST_MIX_SWAP=virus refused: its max-util and power virus segments fall under the site skip ($FL_MU_SKIP)" >&2; exit 1; }
  local mu=${FLEET_SWAP_MU_S:-180} pv=${FLEET_SWAP_PV_S:-240} left k n x   # FL_BB_PV_SETUP, FL_BB_PV_RST: fl_init's power virus lines
  [[ $mu =~ ^[1-9][0-9]*$ && $pv =~ ^[1-9][0-9]*$ ]] || { echo "FLEET_SWAP_MU_S and FLEET_SWAP_PV_S take whole seconds (got '$mu' '$pv')" >&2; exit 1; }
  if [ "$FL_SWAP_S" -gt 0 ]; then
    if [ "$FL_MIX_SWAP" = heavy ]; then
      FL_SWAP_PART_S=$(( FL_SWAP_S + FL_BB_GAP_S + 10 + 40 ))
    else
      [ "${FL_SMOKE:-0}" = 1 ] && { mu=$FL_SWAP_S; pv=$FL_SWAP_S; left=$(( 2 * FL_SWAP_S )); } || left=$FL_SWAP_S
      for k in mu pv mu pv mu pv mu pv mu pv mu pv mu pv mu pv; do
        [ "$left" -gt 0 ] || break
        n=${!k}; [ "$n" -gt "$left" ] && n=$left; left=$(( left - n )); FL_SWAP_SEGS="$FL_SWAP_SEGS${FL_SWAP_SEGS:+ }$k:$n"
        [ $k = pv ] && x=$(( n + FL_BB_PV_SETUP + 20 + FL_BB_PV_RST )) || x=$(( n + FL_BB_MU_SETUP + FL_BB_MU_MARGIN ))
        FL_SWAP_PART_S=$(( FL_SWAP_PART_S + x ))
      done
    fi
    FL_EXTRA_S=$(( FL_EXTRA_S + FL_SWAP_PART_S - FL_SWAP_S )); FL_BB_S=$(( FL_BB_S + FL_SWAP_PART_S ))
    FL_SWAP_WARN="WARNING: TRAYTEST_MIX_SWAP=$FL_MIX_SWAP (dev trial knob): the last CCL-mix chunk ($FL_SWAP_S s) runs as $([ "$FL_MIX_SWAP" = heavy ] && echo "the heavy matmul load ($FL_DIDT_OPS)" || echo "the power virus alternating with max-util ($FL_SWAP_SEGS)") instead; not a standard run"
  else
    FL_SWAP_WARN="WARNING: TRAYTEST_MIX_SWAP=$FL_MIX_SWAP (dev trial knob) is set, but this mode has no long CCL-mix chunk to swap: runs as main"
  fi
  echo "$FL_SWAP_WARN" >&2
}

fl_init() {  # FLEET (quick|full|repro-linka|repro-abortfab) and FL_DRY set by run.sh's options
  case $FLEET in quick|full|repro-linka|repro-abortfab) ;; *) echo "--mode takes quick, full, smoke, long or repro (got '$FLEET'; --mode=sustained is retired: --mode=full runs its load)" >&2; exit 1 ;; esac
  fl_smoke
  case ${FLEET_REPRO:-} in '') ;; numeric|warm-reset|tray-drop) [ "$FLEET" = quick ] || { echo "FLEET_REPRO (run.sh --repro) runs plain run.sh only (got --mode=$FLEET)" >&2; exit 1; } ;;
    *) echo "FLEET_REPRO takes numeric, warm-reset or tray-drop (got '$FLEET_REPRO')" >&2; exit 1 ;; esac
  # FLEET_BUSBAR_SHAPE (repro.sh tray-drop): the bus-bar mix as one H3-shaped load (fleet/drop_shapes.py), aimed
  # at FLEET_BUSBAR_SHAPE_TRAY; the shapes that load some trays only need FLEET_LOAD_TRAYS (one submesh per tray)
  if [ -n "${FLEET_BUSBAR_SHAPE:-}" ]; then
    python3 "$FL_DIR/drop_shapes.py" list | cut -f1 | grep -qxF -- "$FLEET_BUSBAR_SHAPE" \
      || { echo "FLEET_BUSBAR_SHAPE takes $(python3 "$FL_DIR/drop_shapes.py" list | cut -f1 | paste -sd' ') (got '$FLEET_BUSBAR_SHAPE')" >&2; exit 1; }
    [[ ${FLEET_BUSBAR_SHAPE_TRAY:-2} =~ ^[1-4]$ ]] || { echo "FLEET_BUSBAR_SHAPE_TRAY takes a tray 1-4 (got '$FLEET_BUSBAR_SHAPE_TRAY')" >&2; exit 1; }
  fi
  # the separate sustained part (C7) is gone: full's 25 min last mix chunk is its place
  # the bus-bar stage, first load in every mode: FLEET_BUSBAR_ORDER (qb,mix,step) runs
  # two quiet-to-burst steps (qb), then the CCL-mix chunks (FLEET_BUSBAR_CHUNK_MIN, default 1, 2, 3, 4 min, full 1, 2, 3, 25: the e9831c5 schedule, which dropped one quad host's tray 3
  # 155 s into its 4 min chunk, 572 s into the stage; FLEET_CHUNK_MIN also sets it (full raises its last chunk to FLEET_LONG_MIN), FLEET_C7_S <s> one chunk of <s>),
  # before them the quiet-to-burst steps (qb: FLEET_BUSBAR_QB_TRIES 2 x FLEET_BUSBAR_QB_QUIET_S
  # 60 s quiet + FLEET_BUSBAR_QB_BURST_S 30 s burst, fl_bb_step, no numeric readback; they run first), the cold-quiet-to-full-load step (FLEET_BUSBAR_QUIET_S 10 s + FLEET_BUSBAR_BURST_S 90 s, ~40 s mesh open, ~40 s numeric
  # readback) and, in quick and full, the heavy chunk: one FLEET_BUSBAR_HEAVY_S (90 s; 750 s in full) chunk_load.py
  # chunk of the heavy part alone, its output compared across chips at the chunk end, which caught a quad host's 88:00.0 tray 4 numeric
  # Inf ([heavy] 1/193 launches, r010/r011; the mix chunks run that lofi_all heavy part checked). Its load is FLEET_DIDT_OPS, the DIDT bf16 HiFi2 matmul then the bf8 LoFi matmul that caught a quad host's 81:00.0 tray 4,
  # 16 of each per launch, SFPU off as both tests ran them; the 50 ms burst gap and the chunk length are unchanged. Every
  # one of those 16 outputs per op is compared with its reference on the device (--heavy-check-each 1), not only the last, and the
  # report prints one NUMERIC RATE line per chip (BDF, tray, launches, seconds, launches/s, checked outputs/s per op)
  # the bus-bar stage is the sustained stage. Full runs
  # the same parts in the same order as plain run.sh; its last CCL-mix chunk is the FLEET_LONG_MIN (25) min chunk.
  # long (the old separate 25 min chunk) is opt-in through FLEET_BUSBAR_ORDER, the same in both modes
  # the stage ends with two max loads, the weaker for FLEET_BUSBAR_MAXLOAD_WEAK_S (180 s, the 3 min
  # chunk's place), the heavier for FLEET_BUSBAR_MAXLOAD_HEAVY_S (240 s, the 4 min chunk's place), in plain, full and (seconds) smoke.
  # FL_BB_MAXLOAD is the one order setting, weaker first: the power virus counts as heavier until the 'URGENT power
  # virus vs max-util A/B' result; if max-util proves heavier, flip it to pv,maxutil here
  FL_BB_MAXLOAD=${FLEET_BUSBAR_MAXLOAD:-maxutil,pv}
  case $FL_BB_MAXLOAD in maxutil,pv|pv,maxutil) ;; *) echo "FLEET_BUSBAR_MAXLOAD takes maxutil,pv or pv,maxutil, weaker first (got '$FL_BB_MAXLOAD')" >&2; exit 1 ;; esac
  FL_BB_ORDER=${FLEET_BUSBAR_ORDER:-qb,mix,step,heavy,code,$FL_BB_MAXLOAD}
  [[ ,$FL_BB_ORDER, =~ ^(,(qb|mix|step|heavy|long|rw|code|maxutil|pv))+,$ ]] && ! tr , '\n' <<< "$FL_BB_ORDER" | sort | uniq -d | grep -q . \
    || { echo "FLEET_BUSBAR_ORDER takes qb, mix, step, heavy, long, rw, code, maxutil and pv, each at most once, comma-separated (got '$FL_BB_ORDER')" >&2; exit 1; }
  # TRAYTEST_MIX_SWAP (goal 54 stage 3 trial): dev trial knob. heavy: the last CCL-mix chunk (full's
  # 25 min chunk, smoke's last chunk) runs the same seconds of the heavy matmul load instead; virus: the same seconds of the
  # power virus alternating with max-util. Unset or empty: identical to main. A loud WARNING goes on the first line, the chunk line
  # and the summary, and reliability.sh --install refuses it. An unknown value exits here, before any device open
  FL_MIX_SWAP=${TRAYTEST_MIX_SWAP:-}; FL_SWAP_WARN=""; FL_SWAP_S=0; FL_SWAP_SEGS=""; FL_SWAP_PART_S=0
  case $FL_MIX_SWAP in ''|heavy|virus) ;; *) echo "TRAYTEST_MIX_SWAP takes heavy or virus (got '$FL_MIX_SWAP')" >&2; exit 1 ;; esac
  # FLEET_MAXUTIL_SKIP: site data, never a host list in tray-test. Set (e.g. by
  # a site's launchers), it drops maxutil from the order, and FL_MU_SKIP's WARNING goes on the
  # plan line, at the part's place in the stage and in the run summary (never a quiet skip); the other parts run
  # Unset, the box's own site file (TRAY_SITE_FILE, default ~/.config/tray-test/site, a "maxutil_skip=<text>" line) gives it, so
  # launchers that reach run.sh over ssh or through reliability.sh carry it too
  FL_MU_SKIP=""; local mus=${FLEET_MAXUTIL_SKIP:-} sf=${TRAY_SITE_FILE:-$HOME/.config/tray-test/site}
  [ -z "$mus" ] && [ -r "$sf" ] && mus=$(sed -n 's/^maxutil_skip=//p' "$sf" | head -1)
  if [ -n "$mus" ]; then
    case $mus in "WARNING: max-util skipped"*) FL_MU_SKIP=$mus ;; *) FL_MU_SKIP="WARNING: max-util skipped: $mus (site data)" ;; esac
    # the power virus is skipped with it (a site that bans max-util bans the PV too)
    FL_MU_SKIP="$FL_MU_SKIP; WARNING: the power virus is skipped too, the 4 min chunk stays"
    FL_BB_ORDER=$(tr , '\n' <<< "$FL_BB_ORDER" | grep -vx 'maxutil\|pv' | paste -sd,)
  fi
  FL_BB_QB_N=${FLEET_BUSBAR_QB_TRIES:-2}; FL_BB_QB_Q=${FLEET_BUSBAR_QB_QUIET_S:-60}; FL_BB_QB_B=${FLEET_BUSBAR_QB_BURST_S:-30}
  [[ $FL_BB_QB_N =~ ^[1-9][0-9]*$ && $FL_BB_QB_Q =~ ^[0-9]+$ && $FL_BB_QB_B =~ ^[1-9][0-9]*$ ]] \
    || { echo "FLEET_BUSBAR_QB_TRIES, FLEET_BUSBAR_QB_QUIET_S and FLEET_BUSBAR_QB_BURST_S take whole numbers (got '$FL_BB_QB_N' '$FL_BB_QB_Q' '$FL_BB_QB_B')" >&2; exit 1; }
  FL_BB_QB_S=$(( FL_BB_QB_N * (FL_BB_QB_Q + FL_BB_QB_B + 30) ))   # each: quiet + burst + ~30 s mesh open and close (no numeric readback)
  # FLEET_LOAD_TRAYS: chunk_load.py --trays, the chunk_load.py stages (bus-bar mix and heavy chunk,
  # sustained chunks) load only these trays; unset: every chip, as before
  [ -z "${FLEET_LOAD_TRAYS:-}" ] || [[ ,$FLEET_LOAD_TRAYS, =~ ^(,[1-4])+,$ ]] \
    || { echo "FLEET_LOAD_TRAYS takes trays 1-4, comma-separated (got '$FLEET_LOAD_TRAYS')" >&2; exit 1; }
  # --mode=full (formerly): the same 1, 2, 3, 4 min quick jumps, then two 20 min chunks,
  # 50 min of CCL-mix load: past every late bus-bar drop on record by max(20%, 3 min) (latest a quad host's tray 4 at 24:31
  # of the CCL-only C7) and about 95% per full run on an outside galaxy's tray 4 (notes/full-mode-coverage.md, tests/test_full_soak.py);
  # full's 2 x 20 min went into one 25 min chunk, past both late drops on record.
  # Full = plain, just longer: the same chunks and quiets, its last chunk raised to FLEET_LONG_MIN (1, 2, 3, 25 min)
  FL_LONG_MIN=${FLEET_LONG_MIN:-25}; [[ $FL_LONG_MIN =~ ^[1-9][0-9]*$ ]] || { echo "FLEET_LONG_MIN takes whole minutes (got '$FL_LONG_MIN')" >&2; exit 1; }
  # with max-util in the order, the default bus-bar chunks are 1, 2, 3 min
  # and the 240 s max-util part at the stage end takes the 4 min chunk's place; full keeps 1, 2, 3, 25 min plus the same 240 s.
  # A site max-util skip (FL_MU_SKIP) keeps the 4 min chunk. One code path for run.sh and reliability.sh
  # each max load (maxutil, pv) takes one chunk's place from the end: plain and smoke run 1, 2 min.
  # Full keeps main's 1, 2, 3, 25 min: a quad host's tray 4 drop at 24.5 min of mix load needs 29.4 min of it
  # (tests/test_full_soak.py); the early drops come in the 1 and 2 min chunks, and both max loads run hotter
  local bm=${FLEET_CHUNK_MIN:-1,2,3,4} mu4=0 n=0 x bp; FL_EXTRA_S=0
  { [ -z "${FLEET_CHUNK_MIN:-}" ] || [ "${FL_SMOKE_CM:-0}" = 1 ]; } && [ -z "${FLEET_BUSBAR_CHUNK_MIN:-}" ] \
    && for x in maxutil pv; do case ,$FL_BB_ORDER, in *,$x,*) mu4=1; n=$(( n + 1 )) ;; esac; done
  while [ "$n" -gt 0 ] && [[ $bm == *,* ]]; do bm=${bm%,*}; n=$(( n - 1 )); done   # plain and smoke: the max loads take the last chunks' places
  bp=$bm; [ "$FLEET" = full ] && bm=${FLEET_CHUNK_MIN:-1,2,3,4}   # full keeps every chunk; its last becomes the 25 min chunk below
  if [ "$FLEET" = full ] && [ -z "${FLEET_BUSBAR_CHUNK_MIN:-}" ] && [ -z "${FLEET_C7_S:-}" ] && [[ $bm =~ ^(.*,)?([0-9]+)$ ]] \
    && [ "${BASH_REMATCH[2]}" -lt "$FL_LONG_MIN" ]; then
    FL_EXTRA_S=$(( (FL_LONG_MIN - BASH_REMATCH[2]) * 60 )); bm=${BASH_REMATCH[1]}$FL_LONG_MIN   # duration only: run.sh adds it to the run budget
    [ "$mu4" = 1 ] && FL_EXTRA_S=$(awk -v f="$bm" -v p="$bp" 'BEGIN { nf = split(f, a, ","); np = split(p, b, ",")   # plain lacks the chunks the max loads took: those and their quiets are extra
      for (i = 1; i <= nf; i++) t += a[i]; for (i = 1; i <= np; i++) t -= b[i]; printf "%d", t * 60 + (nf - np) * 120 }')
  fi
  # the swapped chunk leaves the CCL-mix schedule and runs as its own part right after the mix (fl_bb_swap); plain
  # (no long chunk) runs as main, with the WARNING saying so
  if [ -n "$FL_MIX_SWAP" ] && [ -z "${FLEET_BUSBAR_CHUNK_MIN:-}" ] && [ -z "${FLEET_C7_S:-}" ] && [[ ,$FL_BB_ORDER, == *,mix,* ]] \
    && { [ "$FL_EXTRA_S" -gt 0 ] || [ "${FL_SMOKE_CM:-0}" = 1 ]; } && [[ $bm =~ ^(.+),([^,]+)$ ]]; then
    bm=${BASH_REMATCH[1]}; FL_SWAP_S=$(awk -v m="${BASH_REMATCH[2]}" 'BEGIN { s = int(m * 60 + 0.5); print (s < 1 ? 1 : s) }')
  fi
  read -r FL_BB_CHUNKS FL_BB_LOAD_S FL_BB_GAP_S FL_BB_MIN < <(FLEET_CHUNK_MIN=${FLEET_BUSBAR_CHUNK_MIN:-$bm} python3 "$FL_DIR/chunk_sched.py" sched) \
    && [ -n "$FL_BB_MIN" ] || { echo "fleet: bad bus-bar chunk schedule (FLEET_BUSBAR_CHUNK_MIN '${FLEET_BUSBAR_CHUNK_MIN:-$bm}', FLEET_GAP_S '${FLEET_GAP_S:-}')" >&2; exit 1; }
  FL_C7_LOAD=""; if [ -n "${FLEET_C7_S:-}" ]; then FL_C7_LOAD=$FLEET_C7_S
    FL_BB_MIX_S=$(awk -v l="$FL_C7_LOAD" -v g="$FL_BB_GAP_S" 'BEGIN { printf "%d", l + g + 10 }')
  else FL_BB_MIX_S=$(awk -v l="$FL_BB_LOAD_S" -v n="$FL_BB_CHUNKS" -v g="$FL_BB_GAP_S" 'BEGIN { printf "%d", l + n * (g + 10) + 40 }'); fi
  # G3: the x/w/y-in-DRAM readback after mix chunk FLEET_BUSBAR_RW_AFTER (1; 0 = off), its wall cap + 15 s of inputs
  # the mix's back-to-back matmul: 16k bf8 LoFi x2 per launch, no burst gap, beside every CCL part;
  # its 4 MiB x chips PCIe write on every 2nd launch (in line on cq 0, every launch paced b02u08's 16 GT/s links to half rate)
  # it was replaced by the one load definition: the sustained part's load, which drew more box kW than the mix in 14 of
  # 17 same-run pairs (full runs 10-25/26: +0.1 to +0.5 kW, sustained chunks mean 11.40 kW, max 12.74 kW), with the SFPU in
  # parallel and the CCL mix kept. FLEET_LOAD_* override it; the old FLEET_SUSTAIN_* names still work.
  # Over-limit bursts: the load runs as bursts of FL_LOAD_STEPS ops with a FL_LOAD_GAP_MS pause after
  # each, which drives each chip well over its power limit at every burst start (an outside galaxy's power notes: chip peak
  # max 295 W, median 282 W at 50 ms vs 210 / 180 W with no gap, 130 W limit). Never 0 without a peak comparison
  # (tests/test_burst_gap.sh). The SFPU op is gelu:1 (notes/max-load.md: 32/32 chips limiter-bound on a quad host
  # at their 130 W limit, 27/32 with the old gelu:0).
  FL_LOAD_HEAVY=${FLEET_LOAD_HEAVY:-lofi_all}; FL_LOAD_HEAVY_N=${FLEET_LOAD_HEAVY_N:-16}; FL_LOAD_GAP_MS=${FLEET_LOAD_GAP_MS:-50}
  FL_LOAD_STEPS=${FLEET_LOAD_STEPS:-8}; FL_LOAD_PCIE_MB=${FLEET_LOAD_PCIE_MB:-${FLEET_SUSTAIN_PCIE_MB:-16}}; FL_LOAD_PCIE_EVERY=${FLEET_LOAD_PCIE_EVERY:-1}
  FL_LOAD_SFPU_PAR=${FLEET_LOAD_SFPU_PAR:-${FLEET_SUSTAIN_SFPU_PAR:-1}}; FL_LOAD_SFPU_ACT=${FLEET_LOAD_SFPU_ACT:-${FLEET_SUSTAIN_SFPU_ACT:-gelu:1}}
  FL_LOAD_PARTS=${FLEET_LOAD_PARTS:-${FLEET_BUSBAR_PARTS:-heavy,h3,mm,sfpu,dram,pcie}}; FL_LOAD_CQS=${FLEET_LOAD_CQS:-2}
  # dev load variants (fleet/load_variant.py: dram-mm, host-stress, no-gap, combined; dram-eth refused with its
  # reason): fl_load_args and fl_burst_args pass --variant (sudo -u drops the environment) to chunk_load.py and the qb/step
  # bursts; no-gap is the 0 ms gap for every stage, pinpoint included. A WARNING naming the parts each variant
  # ran in: here (the run's first line), on every chunk's LOAD line, in each burst's log and before the summary line
  FL_LOAD_VARIANT=${FLEET_LOAD_VARIANT:-}; FL_VARIANT_WARN=""
  if [ -n "$FL_LOAD_VARIANT" ]; then
    local lv; lv=$(python3 -I "$FL_DIR/load_variant.py" "$FL_LOAD_VARIANT") || exit 1   # it says why on stderr
    FL_LOAD_VARIANT=${lv%%$'\n'*}; FL_VARIANT_WARN=${lv#*$'\n'}
    case ,$FL_LOAD_VARIANT, in *,no-gap,*) FL_LOAD_GAP_MS=0 ;; esac
    export FLEET_LOAD_VARIANT=$FL_LOAD_VARIANT; echo "$FL_VARIANT_WARN" >&2
  fi
  [[ $FL_LOAD_HEAVY_N$FL_LOAD_STEPS$FL_LOAD_PCIE_MB$FL_LOAD_PCIE_EVERY =~ ^[0-9]+$ && $FL_LOAD_GAP_MS =~ ^[0-9]+(\.[0-9]+)?$ && $FL_LOAD_SFPU_PAR =~ ^[01]$ && $FL_LOAD_CQS =~ ^[12]$ ]] \
    || { echo "FLEET_LOAD_HEAVY_N, _STEPS, _PCIE_MB, _PCIE_EVERY take whole numbers, _GAP_MS ms, _SFPU_PAR 0 or 1, _CQS 1 or 2" >&2; exit 1; }
  FL_BB_RW=${FLEET_BUSBAR_RW_AFTER:-1}; FL_BB_RW_S=${FLEET_BUSBAR_RW_S:-15}; FL_BB_RW_WALL=${FLEET_BUSBAR_RW_WALL_S:-35}
  [[ $FL_BB_RW =~ ^[0-9]+$ && $FL_BB_RW_S =~ ^[1-9][0-9]*$ && $FL_BB_RW_WALL =~ ^[1-9][0-9]*$ ]] \
    || { echo "FLEET_BUSBAR_RW_AFTER (chunk, 0 = off), FLEET_BUSBAR_RW_S and FLEET_BUSBAR_RW_WALL_S take whole numbers (got '$FL_BB_RW' '$FL_BB_RW_S' '$FL_BB_RW_WALL')" >&2; exit 1; }
  FL_BB_DBI_WALL=${FLEET_BUSBAR_DBI_WALL_S:-$([ "$FLEET" = full ] && echo 60 || echo 10)}   # G3b: DRAM byte/DBI patterns after G3, 0 = off
  [[ $FL_BB_DBI_WALL =~ ^[0-9]+$ ]] || { echo "FLEET_BUSBAR_DBI_WALL_S takes whole seconds, 0 = off (got '$FL_BB_DBI_WALL')" >&2; exit 1; }
  FL_BB_L1_WALL=${FLEET_BUSBAR_L1_WALL_S:-$([ "$FLEET" = full ] && echo 60 || echo 10)}   # N1: L1 march after G3b, 0 = off
  [[ $FL_BB_L1_WALL =~ ^[0-9]+$ ]] || { echo "FLEET_BUSBAR_L1_WALL_S takes whole seconds, 0 = off (got '$FL_BB_L1_WALL')" >&2; exit 1; }
  fl_g3 && FL_BB_MIX_S=$(( FL_BB_MIX_S + FL_BB_RW_WALL + (FL_BB_DBI_WALL > 0 ? FL_BB_DBI_WALL + 3 : 0) + (FL_BB_L1_WALL > 0 ? FL_BB_L1_WALL + 3 : 0) ))
  fl_g3 && [ "$FL_BB_DBI_WALL" -gt 10 ] && FL_EXTRA_S=$(( FL_EXTRA_S + FL_BB_DBI_WALL - 10 ))   # a longer G3b (full: 60 s) is duration only
  fl_g3 && [ "$FL_BB_L1_WALL" -gt 10 ] && FL_EXTRA_S=$(( FL_EXTRA_S + FL_BB_L1_WALL - 10 ))   # a longer N1 (full: 60 s) too
  # rw (opt-in, repro.sh's dram way): numeric_dram.py on its own mesh, FLEET_BUSBAR_RW_S of load, a readback every
  # FLEET_BUSBAR_RW_CHECK_S (1.5 s, as C4d), FLEET_BUSBAR_RW_WALL_S cap; ~60 s mesh open and inputs
  FL_BB_RW_CK=${FLEET_BUSBAR_RW_CHECK_S:-1.5}; [[ $FL_BB_RW_CK =~ ^[0-9]+(\.[0-9]+)?$ ]] && [ "$FL_BB_RW_CK" != 0 ] \
    || { echo "FLEET_BUSBAR_RW_CHECK_S takes seconds > 0 (got '$FL_BB_RW_CK')" >&2; exit 1; }
  FL_BB_RWP_S=$(( FL_BB_RW_WALL + 60 ))
  FL_BB_STEP_S=$(( ${FLEET_BUSBAR_QUIET_S:-10} + ${FLEET_BUSBAR_BURST_S:-90} + 60 ))   # setup 60 s (measured 39-49 s)
  # full lengthens only this chunk, 750 s (same code path as plain's 90 s): at the measured 11.19 checked
  # 16k_bf8_lofi outputs/s per chip that is 8,392 >= 8,100 (750 s keeps the full plan 15 s inside its budget, test_plan_margin) (the catch target for a quad host's 81:00.0 fault). Duration only: added to the run budget
  FL_DIDT_X_DEF=10; FL_BB_HV_DEF=90; [ "$FLEET" = full ] && [ -z "${FLEET_BUSBAR_HEAVY_S:-}" ] && { FL_BB_HV_DEF=750; FL_DIDT_X_DEF=90; FL_EXTRA_S=$(( ${FL_EXTRA_S:-0} + 660 )); }
  FL_BB_HV=${FLEET_BUSBAR_HEAVY_S:-$FL_BB_HV_DEF}; [[ $FL_BB_HV =~ ^[1-9][0-9]*$ ]] || { echo "FLEET_BUSBAR_HEAVY_S takes whole seconds (got '$FL_BB_HV')" >&2; exit 1; }
  # the heavy chunk's ops: tt-metal tests/didt/test_minimal_matmul.py bf16_HiFi2 (ub0_repro 16k_hifi2), then the
  # bf8 LoFi matmul of diag (16k_bf8_lofi), each 16384^3, minimal_matmul M8/K16/N8, subblock 2x4
  FL_DIDT_OPS=${FLEET_DIDT_OPS:-16k_hifi2,16k_bf8_lofi}; [[ $FL_DIDT_OPS =~ ^[0-9a-z_]+(,[0-9a-z_]+)*$ ]] \
    || { echo "FLEET_DIDT_OPS takes ub0_repro.py HEAVY names, comma-separated (got '$FL_DIDT_OPS')" >&2; exit 1; }
  # the heavy chunk's SFPU vote (exp, reciprocal, sqrt, gelu in turn on each checked output, voted across
  # chips) costs ~1 s per 50 s of heavy chunk, rounded up: 2 s plain (90 s), 15 s full (750 s), 1 s smoke; the chunk's own load
  # time is unchanged. Scaled from the chunk length, not the mode; time over plain's 2 s is duration only, on the budget
  FL_N4_S=$(( (FL_BB_HV + 49) / 50 )); [ "$FL_N4_S" -gt 2 ] && FL_EXTRA_S=$(( ${FL_EXTRA_S:-0} + FL_N4_S - 2 ))
  # goal 54 stage 2 N5: the extra number formats (bf16 HiFi4, bfp4 LoFi, fp32 HiFi4, int32 eltwise) run only in the
  # heavy chunk's last FL_DIDT_X_S s (10 plain, 90 full, the whole chunk in smoke), inside its FL_BB_HV: no stage time added
  FL_DIDT_X_OPS=${FLEET_DIDT_X_OPS:-16k_hifi4,16k_bfp4_lofi,8k_fp32_hifi4,8k_int32_eltwise}; [[ $FL_DIDT_X_OPS =~ ^[0-9a-z_]+(,[0-9a-z_]+)*$ ]] \
    || { echo "FLEET_DIDT_X_OPS takes ub0_repro.py HEAVY names, comma-separated (got '$FL_DIDT_X_OPS')" >&2; exit 1; }
  FL_DIDT_X_S=${FLEET_DIDT_X_S:-$FL_DIDT_X_DEF}; [[ $FL_DIDT_X_S =~ ^[1-9][0-9]*$ ]] || { echo "FLEET_DIDT_X_S takes whole seconds (got '$FL_DIDT_X_S')" >&2; exit 1; }
  [ "$FL_DIDT_X_S" -le "$FL_BB_HV" ] || FL_DIDT_X_S=$FL_BB_HV
  FL_BB_HEAVY_S=$(( FL_BB_HV + FL_BB_GAP_S + 10 + 40 + FL_N4_S ))   # chunk + clock-checked quiet + chunk-end readback + mesh open, compile, references + SFPU vote
  # the long chunk: FLEET_LONG_MIN min, FLEET_PRERAMP_S of pre-ramp first (0: the mix, step and heavy chunk ramped
  # the chips already) + FLEET_PRERAMP_WARM_S after a top-up; ~90 s mesh open + compile + references, as the sustained part took
  FL_LONG_PRE=$(awk -v p="${FLEET_PRERAMP_S:-0}" -v w="${FLEET_PRERAMP_WARM_S:-30}" 'BEGIN { printf "%d", (p > 0 ? p + w : 0) }')
  FL_BB_LONG_S=$(( FL_LONG_MIN * 60 + FL_BB_GAP_S + 10 + 90 + FL_LONG_PRE ))
  # max-util: its load s from fleet/maxutil.py's table for the mode (plain and
  # full one 240 s burst at max power, smoke two FLEET_SMOKE_S bursts with as much idle between), + its SETUP_S
  # (60 s: archive check or unpack, device open, init, kernel compile) + FLEET_BUSBAR_MU_MARGIN_S (30 s: exit and judge)
  FL_BB_MU_MODE=$([ "${FL_SMOKE:-0}" = 1 ] && echo smoke || { [ "$FLEET" = full ] && echo full || echo quick; })
  # the weaker max load (first of FL_BB_MAXLOAD) runs FLEET_BUSBAR_MAXLOAD_WEAK_S, the heavier FLEET_BUSBAR_MAXLOAD_HEAVY_S
  FL_BB_WEAK_S=${FLEET_BUSBAR_MAXLOAD_WEAK_S:-180}; FL_BB_HEAVYL_S=${FLEET_BUSBAR_MAXLOAD_HEAVY_S:-240}
  [[ $FL_BB_WEAK_S =~ ^[1-9][0-9]*$ && $FL_BB_HEAVYL_S =~ ^[1-9][0-9]*$ ]] \
    || { echo "FLEET_BUSBAR_MAXLOAD_WEAK_S and FLEET_BUSBAR_MAXLOAD_HEAVY_S take whole seconds (got '$FL_BB_WEAK_S' '$FL_BB_HEAVYL_S')" >&2; exit 1; }
  FL_BB_MU_ARG=$([ "${FL_BB_MAXLOAD%%,*}" = maxutil ] && echo "$FL_BB_WEAK_S" || echo "$FL_BB_HEAVYL_S")
  FL_BB_MU_LOAD=$(python3 "$FL_DIR/maxutil.py" table "$FL_BB_MU_MODE" "${FLEET_SMOKE_S:-2}" "$FL_BB_MU_ARG" | awk 'NF == 2 { s += $1 + $2 } END { if (s > 0) print s }')
  FL_BB_MU_SETUP=60; FL_BB_MU_MARGIN=${FLEET_BUSBAR_MU_MARGIN_S:-30}
  [[ $FL_BB_MU_LOAD =~ ^[1-9][0-9]*$ && $FL_BB_MU_MARGIN =~ ^[0-9]+$ ]] \
    || { echo "fleet: bad max-util table for mode $FL_BB_MU_MODE (fleet/maxutil.py table: '$FL_BB_MU_LOAD') or FLEET_BUSBAR_MU_MARGIN_S '$FL_BB_MU_MARGIN'" >&2; exit 1; }
  FL_BB_MU_S=$(( FL_BB_MU_LOAD + FL_BB_MU_SETUP + FL_BB_MU_MARGIN ))
  # code (goal 54 stage 2 N3): lib/exalens_probe.py --code reads each core's firmware and kernel code
  # in L1 (Tensix BRISC/NCRISC/TRISC0-2, ETH base FW) on every chip and flags a word that a strong majority of chips disagree with
  # and a re-read confirms, by BDF, core and RISC. No load. Budgeted per run: plain 5 s, full 60 s, smoke 1 s and 2 Tensix + 1 ETH
  # cores; a cursor file carries the next core into the next run, so plain runs rotate over the box. + ~25 s exalens init and exit
  FL_BB_CODE_SCOPE=$(case $FL_BB_MU_MODE in smoke|full) echo "$FL_BB_MU_MODE" ;; *) echo plain ;; esac)   # the reviewed duration-only mode name above
  FL_BB_CODE_B=${FLEET_CODE_BUDGET_S:-$(case $FL_BB_CODE_SCOPE in smoke) echo 1 ;; full) echo 60 ;; *) echo 5 ;; esac)}
  [[ $FL_BB_CODE_B =~ ^[0-9]+$ ]] || { echo "FLEET_CODE_BUDGET_S takes whole seconds (got '$FL_BB_CODE_B')" >&2; exit 1; }
  FL_BB_CODE_S=$(( FL_BB_CODE_B + 25 ))
  case ,$FL_BB_ORDER, in *,code,*) [ "$FL_BB_CODE_SCOPE" = full ] && [ -z "${FLEET_CODE_BUDGET_S:-}" ] && FL_EXTRA_S=$(( FL_EXTRA_S + 55 )) ;; esac   # full's longer budget is duration only
  FL_BB_S=0; case ,$FL_BB_ORDER, in *,mix,*) FL_BB_S=$FL_BB_MIX_S ;; esac; case ,$FL_BB_ORDER, in *,step,*) FL_BB_S=$(( FL_BB_S + FL_BB_STEP_S )) ;; esac
  case ,$FL_BB_ORDER, in *,heavy,*) FL_BB_S=$(( FL_BB_S + FL_BB_HEAVY_S )) ;; esac; case ,$FL_BB_ORDER, in *,qb,*) FL_BB_S=$(( FL_BB_S + FL_BB_QB_S )) ;; esac
  case ,$FL_BB_ORDER, in *,rw,*) FL_BB_S=$(( FL_BB_S + FL_BB_RWP_S )) ;; esac; case ,$FL_BB_ORDER, in *,long,*) FL_BB_S=$(( FL_BB_S + FL_BB_LONG_S )) ;; esac
  case ,$FL_BB_ORDER, in *,maxutil,*) FL_BB_S=$(( FL_BB_S + FL_BB_MU_S )) ;; esac; case ,$FL_BB_ORDER, in *,code,*) FL_BB_S=$(( FL_BB_S + FL_BB_CODE_S )) ;; esac
  # the power virus: its load s (smoke: 2 x FLEET_SMOKE_S), + FLEET_BUSBAR_PV_SETUP_S (15 s: pin check, 32 run-ttx
  # unpacks) + 20 s stop and judge + FLEET_BUSBAR_PV_RESET_S (65 s: the glx_reset, rescan and recount after it, so later steps open clean; C2 glx_reset took 33 s median, 36 s max in 113 runs; 65 s keeps plain's plan at 1781 s, under test_plan_margin's 1785 s)
  FL_BB_PV_LOAD=$([ "${FL_BB_MAXLOAD%%,*}" = pv ] && echo "$FL_BB_WEAK_S" || echo "$FL_BB_HEAVYL_S")
  [ "${FL_SMOKE:-0}" = 1 ] && FL_BB_PV_LOAD=$(( 2 * ${FLEET_SMOKE_S:-2} ))
  FL_BB_PV_SETUP=${FLEET_BUSBAR_PV_SETUP_S:-15}; FL_BB_PV_RST=${FLEET_BUSBAR_PV_RESET_S:-65}
  FL_PV_BRICK_C=${FLEET_PV_BRICK_C:-88}; FL_PV_FEED_A=${FLEET_PV_FEED_A:-219}   # no chip-power stop
  # FLEET_PV_SWAP: maxutil runs max-util in the power virus's place, its own part
  # (busbar/maxutil-pv). reliability.sh sets it for a box that failed the power virus repeatedly; unset: the power virus as main
  FL_PV_SWAP=${FLEET_PV_SWAP:-}; case $FL_PV_SWAP in ''|maxutil) ;; *) echo "FLEET_PV_SWAP takes maxutil (got '$FL_PV_SWAP')" >&2; exit 1 ;; esac
  [[ $FL_BB_PV_SETUP =~ ^[0-9]+$ && $FL_BB_PV_RST =~ ^[0-9]+$ && $FL_PV_BRICK_C =~ ^[1-9][0-9]*$ && $FL_PV_FEED_A =~ ^[1-9][0-9]*$ ]] \
    || { echo "FLEET_BUSBAR_PV_SETUP_S, FLEET_BUSBAR_PV_RESET_S, FLEET_PV_BRICK_C and FLEET_PV_FEED_A take whole numbers" >&2; exit 1; }
  FL_BB_PV_S=$(( FL_BB_PV_LOAD + FL_BB_PV_SETUP + 20 + FL_BB_PV_RST ))
  case ,$FL_BB_ORDER, in *,pv,*) FL_BB_S=$(( FL_BB_S + FL_BB_PV_S )) ;; esac
  fl_swap_init
  [ "$FLEET" = repro-linka ] && { fl_linka_init; return 0; }   # run.sh --repro link-a: its own checks and dry run
  [ "$FLEET" = repro-abortfab ] && { fl_abortfab_init; return 0; }   # run.sh --repro abort-fabric
  local p w q f tot=0 su=0
  [ "${FL_DRY:-0}" = 1 ] || return 0
  echo "run.sh --mode=$(fl_mode) dry run: phases and time budget (nothing is opened, reset or loaded)"
  printf '%-9s %-24s %-90s %6s\n' phase stage what min
  while IFS='|' read -r p w q f; do
    case $p in numeric) printf '%-9s %-24s %-90s %6s\n' "$p" "$(fl_sn $p)" "$w" "in 4"; continue ;; sweep) printf '%-9s %-24s %-90s %6s\n' "$p" "$(fl_sn $p)" "$w" fault; continue ;; esac
    case $p in busbar|C5) [ -n "${FLEET_REPRO:-}" ] && [[ " $(fl_load_phases) " != *" $p "* ]] \
      && { printf '%-9s %-24s %-90s %6s\n' "$p" "$(fl_sn $p)" "$w: not in the $FLEET_REPRO repro (repro.sh)" skip; continue; } ;; esac
    [ "$p" = busbar ] && { q=$FL_BB_S; f=$FL_BB_S; w="bus-bar stage, first load ($FL_BB_ORDER): $(fl_bb_plan)${FL_MU_SKIP:+; $FL_MU_SKIP}"; }
    s=$q; case $p in P0|P1|C2) su=$((su + s)) ;; *) tot=$((tot + s)) ;; esac
    printf '%-9s %-24s %-90s %6s\n' "$p" "$(fl_sn $p)" "$w" "$(awk -v s="$s" 'BEGIN { printf "%.1f", s / 60 }')"
  done <<< "$FL_TABLE"
  echo "total $(fl_mode): $(awk -v s=$tot 'BEGIN { printf "%.1f", s / 60 }') min ($tot s, budget $(awk -v s=$(( 1800 + FL_EXTRA_S )) 'BEGIN { printf "%.1f", s / 60 }') min; a glx_reset fallback in C2 or C9 adds up to 2 min)$(case ",$FL_BB_ORDER," in *,long,*) echo "; long chunk mesh open + compile on a cold cache is not on record (plan section 3 open item)" ;; esac)"
  echo "setup (not in the total; the budget counts test stages only): $(awk -v s=$su 'BEGIN { printf "%.1f", s / 60 }') min ($su s: P0, P1, C2); total with setup: $(awk -v s=$((tot + su)) 'BEGIN { printf "%.1f", s / 60 }') min ($((tot + su)) s)"
  echo "on a fault only (a hang or CCL failure): the C9 reset, then a per-link all-gather sweep of up to $(( ${FLEET_SWEEP_S:-300} / 60 )) min (FLEET_SWEEP_S); not in the total"
  echo "resets (C2, C9): run whatever the QSFP far hosts do; no peer snapshot is taken"
  exit 0
}
# fl_find <line>: a finding. A fault (FAIL, HANG, DROP, LOST, MISSING; not a WARN, NEIGHBOUR CHANGE or GATE BUSY)
# turns on full capture: every later snapshot of the run reads lspci -vvv, SEL, exalens.
# Most callers run it in a pipeline subshell, so FC=1 alone would be lost: $FL_OUT/fc_on carries it (lib/heavy_logs.sh hl_full).
fl_find() {
  # what the yield's own kill leaves in the stopped load is no finding; from C9 on, findings count again
  if [ -n "${FL_YIELD:-}" ] && [ -z "${FL_C9_ON:-}" ]; then say "not judged (the load was stopped for the broker yield): $*"; return 0; fi
  case "$*" in *WARN*|"NEIGHBOUR CHANGE"*|"GATE BUSY"*) ;; *FAIL*|*HANG*|*DROP*|*LOST*|*MISSING*) fl_bb_dup find "$*" && return 0 ;; esac
  echo "$*" >> "$FL_OUT/findings.txt"; say "$*"
  local p; p=$(tail -n1 "$FL_OUT/phases.txt" 2>/dev/null)   # the stage it came from, for the per-tray why lines
  case "$*" in NUMERIC\ *|L1\ FAIL\ *|L1\ WARN\ *) p=numeric ;; esac   # a numeric mismatch is stage 5, never the bus-bar stage
  printf '%s\t%s\n' "${p%% *}" "$*" >> "$FL_OUT/finding_stage.tsv"
  case "$*" in *WARN*|"NEIGHBOUR CHANGE"*|"GATE BUSY"*) ;;
    *FAIL*|*HANG*|*DROP*|*LOST*|*MISSING*|"ETH INCOMPLETE "*"start reset skipped; ETH not-active-at-open not judged"*) : > "$FL_OUT/fc_on"; declare -F fc_init > /dev/null && fc_init   # an ETH INCOMPLETE (not judged) core line still gets the capture
      printf '%s\t%s\n' "$(date +%s)" "$*" >> "$FL_OUT/fault_times.tsv" ;; esac; }   # the thermal report's cut-off
# the bus-bar stage's one record of its faults (coordinator 22:24 PST 11-06): after a hang, CCL failure
# or vr_pg fall the stage's later parts run on (fl_bb_cont), and a later part never re-reports a fault the stage already recorded.
# Each FAIL/HANG/DROP finding (fl_find, source "find") and fault.txt entry (fault_note, source "note") in a part records keys in
# busbar/named.tsv: "<kind> <BDF>" for each chip it names (root ports, bus 00, are not chips), or, when it names no chip,
# "<kind> = <its text>" with paths and measured times blanked, so a different chip-less line is always new. A line whose every
# key an earlier part recorded is not written again; its whole text stays in the run log and busbar/not_reported.txt,
# so the first part's finding, fault.txt entry and pinpoint stand
fl_bb_dup() {  # <source: find|note> <line>: 0 = an earlier part of this bus-bar stage recorded it (not written again); 1 = new (now recorded)
  [ -n "${FL_BB_PART:-}" ] || return 1
  local src=$1 f=$FL_OUT/busbar/named.tsv k ks b new=0 l=${2,,} by=""
  case $l in *hang*) k=hang ;; *drop*|*lost*|*missing*) k=drop ;; *) k=fail ;; esac
  # a BDF key: device 00-1f, function 0-7, then no digit, letter or dot; an mm:ss.f duration ("12:05.3 s", "12:34.56") is not one.
  # Each BDF is marked @bb#dd.f@ in a loop that keeps the character after it, so the next chip of a list still has its boundary
  ks=$(sed -E 's/(^|[^0-9a-f:])[0-9]{1,2}:[0-5][0-9]\.[0-9]+ ?(ms|s|sec|min)([^a-z]|$)/\1<t>\3/g' <<< "$l" \
    | sed -E ':a; s/(^|[^0-9a-f:])(([0-9a-f]{4}:)?[0-9a-f]{2}):([01][0-9a-f]\.[0-7])([^0-9a-z.]|$)/\1@\2#\4@\5/; ta' \
    | grep -oE '[0-9a-f]{2}#[01][0-9a-f]\.[0-7]@' | tr '#' ':' | tr -d '@' | grep -v '^00:' | sort -u | sed "s/^/$k /")
  [ -n "$ks" ] || ks="$k = $(sed -E 's#/[^ ;,)]*#<path>#g; s/[0-9]+(\.[0-9]+)? ?(ms|s|min|w|kw|mv|v|a|%)([^a-z]|$)/<n>\2\3/g; s/[[:space:]]+/ /g' <<< "$l")"
  while IFS= read -r b; do
    b=$(awk -F'\t' -v s="$src" -v p="$FL_BB_PART" -v k="$b" '$1 == s && $2 != p && $3 == k { print $2; exit }' "$f" 2>/dev/null)
    [ -n "$b" ] && by=${by:-$b} || new=1
  done <<< "$ks"
  if [ "$new" = 0 ]; then
    say "bus-bar stage: $FL_BB_PART: not reported again ($by already recorded the same $k in this stage): $2"
    printf '%s\t%s\t%s\n' "$FL_BB_PART" "$src" "$2" >> "$FL_OUT/busbar/not_reported.txt"; return 0; fi
  while IFS= read -r b; do printf '%s\t%s\t%s\n' "$src" "$FL_BB_PART" "$b" >> "$f"; done <<< "$ks"; return 1
}
FL_STAGES_TSV=$FL_DIR/stages.tsv   # the one stage table: number, code, name; the README Stages table matches it
fl_sn() { awk -F'\t' -v c="$1" '!/^#/ && $2 == c { print "stage " $1 " " $3; f = 1 } END { exit !f }' "$FL_STAGES_TSV"; }   # <code>: "stage N name"
fl_pn() { case $1 in linka) echo "link-A repro" ;; PV) echo "the glx_reset after the power virus" ;; abortfab) echo "abort-fabric repro" ;; *) fl_sn "$1" || echo "$1" ;; esac; }   # a phase's shown name: its stage number and name (fleet/stages.tsv)
fl_t() { local pv; pv=$(tail -n1 "$FL_OUT/phases.txt" 2>/dev/null); fl_vd_fin "${pv%% *}"; [ "$1" = C9 ] || by_check
  echo "$1 $(date +%s)" >> "$FL_OUT/phases.txt"; }   # a broker yield acts at the next stage boundary (named by the stage it came in); a phase starts (fl_report prints each one's time); the one before it has ended
# one "STAGE <n> <name> VERDICT PASS|WARN|FAIL" line the moment a stage ends, in every mode (n and name: fleet/stages.tsv).
# A stage that was skipped (FL_SKIPPED) or never started is not finished: fl_vd_sum names it under STAGES NOT REACHED. Stage 5
# (code numeric) runs inside stage 4 and ends with it. Prints only; touches no chip (also in --hold).
FL_VD_DONE=" "
fl_vd_fin() {  # <phase code>: print its verdict once
  local c=$1 k n w v; [ -n "$c" ] && [ "$c" != end ] || return 0
  case "$FL_VD_DONE" in *" $c "*) return 0 ;; esac
  n=$(fl_pn "$c"); for k in "${FL_SKIPPED[@]}"; do case "$k" in "$n ("*) return 0 ;; esac; done
  FL_VD_DONE="$FL_VD_DONE$c "; [ "$c" = busbar ] && fl_vd_fin numeric
  case $c in numeric) n=$(fl_sn numeric) ;; esac
  # history from before the run (pre-existing, not from this run) never sets a verdict
  v=PASS; w=$(awk -F'\t' -v c="$c" '$1 == c && !/pre-existing, not from this run/' "$FL_OUT/finding_stage.tsv" 2>/dev/null)
  if [ -n "$w" ]; then v=WARN; echo "$w" | grep -av 'HSC WARN ' | grep -aqE 'FAIL|HANG|DROP|MISSING|LOST|lost' && v=FAIL; fi   # an HSC WARN never makes a stage FAIL
  echo "STAGE $(fl_sn "$c" | sed 's/^stage //') VERDICT $v"
}
fl_vd_sum() {  # an INCOMPLETE end (budget or signal): the finished stages' verdicts, then the stages not reached
  local c nr="" cur; cur=$(tail -n1 "$FL_OUT/phases.txt" 2>/dev/null); cur=${cur%% *}
  [ "${1:-}" = fin ] && fl_vd_fin "$cur"
  for c in P0 P1 C2 busbar numeric C5 C9 sweep; do
    case "$FL_VD_DONE" in *" $c "*) continue ;; esac
    [ "$c" = sweep ] && ! printf '%s\n' "${FL_SKIPPED[@]}" | grep -aq '^per-link sweep' && continue
    nr="$nr${nr:+, }$(fl_sn "$c" | sed 's/^stage //')"
  done
  echo "STAGES NOT REACHED: ${nr:-none}"
}
# fl_capture <label> <hang|fail|drop> <bdf...>: fault capture, always on, before any reset. A hang already
# has its chip state from on_hang.sh (tt-triage NOC0/NOC1 while the job is alive); here: lspci -vvv of each chip and
# its root port, AER, SEL, dmesg tail (the tray CPLDs after a drop: run.sh's cpld_drop, also before the reset). Then lib/heavy_logs.sh's
# heavy_logs_on_hang when it is loaded (run.sh --mode runs heavy_logs_init), which adds exalens and the fast-dispatch
# dump, and lib/fault_capture.sh's naming (fl_fault: $OUT/fault.txt). FL_CAP_FAULT=0 (a C4 fault or any drop: run.sh
# names and snapshots those itself after fl_loads) skips both. Read-only; every step bounded.
fl_fault() {  # <what> <hang|drop> <bdf...>: name the fault in $OUT/fault.txt; FC=1, so run.sh snapshots after the reset
  declare -F fault_name > /dev/null || return 0
  local what=$1 k=$2 b c=(); shift 2
  for b in "$@"; do c+=($(awk -v b="$b" '$1 == b { print $2; exit }' <<< "$RK_MAP")); done
  # a load hang's PINPOINT (fl_loads: chunk_sched.py stalled) names the whole box when every chip stalled together
  if [ "$k" = hang ] && [ -n "${FL_HANG_PIN:-}" ] && { [ ${#c[@]} = 0 ] || [ ${#c[@]} -ge "${RK_WANT:-32}" ]; }; then
    FC_SUS=""; fault_note "$what ($(tools_captured_txt)): ${FL_HANG_PIN#PINPOINT: }"
  elif [ "$k" = hang ] && { [ ${#c[@]} = 0 ] || [ ${#c[@]} -ge "${RK_WANT:-32}" ]; }; then fault_name_hang "$what" /dev/null
  elif [ "$k" = fail ] && { [ ${#c[@]} = 0 ] || [ ${#c[@]} -ge "${RK_WANT:-32}" ]; }; then fault_name "$what (chip not named; $(tools_captured_txt))"
  else fault_name "$what" "${c[@]}"; fi
  fc_init
}
fl_capture() {
  local d=$FL_OUT/capture_$1 k=$2 b rp; shift 2; mkdir -p "$d"   # each read probes its own exact command (sudo_t)
  for b in "$@"; do
    rp=$(basename "$(dirname "$(readlink -f "$SYS/bus/pci/devices/$b")")")
    { echo "== $b (root port $rp)"; sudo_t 20 lspci -vvv -s "$b"; sudo_t 20 lspci -vvv -s "$rp"; } >> "$d/lspci.txt" 2>&1
  done
  python3 "$FL_DIR/fleet_py.py" aer "$SYS" "$@" > "$d/aer.tsv" 2>&1
  sudo_t 60 ipmitool sel elist > "$d/sel.txt" 2>&1; sudo_t 60 dmesg 2>&1 | tail -300 > "$d/dmesg_tail.txt"
  if [ "${FL_CAP_FAULT:-1}" = 1 ]; then
    local n=${d##*/capture_}
    # FL_CAP_NAMED=1: fault_attr already wrote this fault's fault.txt entry, so only FC=1 is set (one entry)
    if [ "${FL_CAP_NAMED:-0}" = 1 ]; then declare -F fc_init > /dev/null && fc_init; else fl_fault "fleet $n $k" "$k" "$@"; fi
    local go=1   # a fault's capture always runs (no run budget skips it)
    if declare -F heavy_logs_on_hang > /dev/null && [ $go = 1 ]; then heavy_logs_on_hang "$@" > "$d/heavy_on_hang.log" 2>&1
      [ "$k" = hang ] && say "fault capture (fleet $n hang) done for $# chip(s) ($(tools_captured_txt)): $(grep -c . "${HL_OUT:-/nonexistent}/steps.tsv" 2>/dev/null || echo 0) heavy-log step(s) in ${HL_OUT:-?}/steps.tsv, log $d/heavy_on_hang.log"; fi
    declare -F rs_cap_end > /dev/null && rs_cap_end
    # --hold: after the capture, before any kill or reset (lib/hold.sh); a drop or a second hang is not held
    [ "$k" = hang ] && declare -F hold_point > /dev/null && hold_point "fleet $n hang" "${FL_HELD[*]}" "$@"
  fi
  [ ${#FL_HELD[@]} = 0 ] || { fl_unhold; declare -F hold_done > /dev/null && hold_done >> "$d/heavy_on_hang.log" 2>&1; }
  [ "$k" = drop ] && declare -F bh_drop > /dev/null && bh_drop "$d" drop   # BMC heat snapshot + CPLD alive/vr_pg
  say "fault capture ($k, $# chip(s)): $d"; return 0
}
fl_ccl_err() {  # <run.log>: 0 when its last 40 lines hold a CCL error line: the active
  # Ethernet-core timeout (its TT_THROW and 'Device N: Timed out' lines), a '| critical |' line naming an ERISC (its port status),
  # or a Traceback frame, RuntimeError or TT_THROW line with a CCL word. Info, warning and debug lines never count: Fabric's
  # set-up lines and tt-metal's 'The following ttnn.all_gather args are deprecated' warning are in every load's log
  tail -40 "$1" 2>/dev/null | grep -avE '\| *(info|warning|debug|trace) *\|' \
    | grep -aiqE 'Timed out while waiting for active ethernet core|\| *critical *\|.*erisc|(Traceback|^[[:space:]]*File "|RuntimeError|TT_THROW).*(all_gather|reduce_scatter|fabric|erisc|ccl)|(all_gather|reduce_scatter|fabric|erisc|ccl).*(Traceback|RuntimeError|TT_THROW)'
}
fl_nj_open() {  # <what> <loaded 0|1> <log>...: 0 when a log ends in a CCL error (fl_ccl_err) whose ETH lines (fault_eth, notes
  # off) all name off-box links with the peer host down or resetting, read from this host's own chips only, and
  # the part wrote no load rows before it (a peer-down timeout mid-run stays judged): each line then gets one
  # fault.txt note and one finding. Else it writes nothing, so a CCL error keeps fault_attr's one entry
  local w=$1 ld=$2 l fe; shift 2; [ "$ld" = 0 ] || return 1
  for l; do fl_ccl_err "$l" && break; l=""; done; [ -n "$l" ] || return 1
  FE_NONOTE=1 fault_eth "$@"; [ -n "$FE_LINES" ] && ! grep -aqv 'off-box link not judged$' <<< "$FE_LINES" || return 1
  while read -r fe; do [ -n "$fe" ] && { fault_note "$fe"; fl_find "$w $fe"; }; done <<< "$FE_LINES"
}
fl_dev_err() {  # <run.log>: 0 when its last 40 lines hold a device error line that is not a CCL error: a chip gone
  # from PCIe ('No such device', 'Read 0xffffffff over PCIe ID n'), a firmware startup error, or tt-metal's device timeout. Info,
  # warning and debug lines never count, as in fl_ccl_err
  tail -40 "$1" 2>/dev/null | grep -avE '\| *(info|warning|debug|trace) *\|' \
    | grep -aiqE 'No such device|Firmware startup error|Read 0xffffffff|device timeout|potential hang detected'
}
fl_setup_err() {  # <run.log>: 0 when its last 40 lines hold a fabric-setup error, not this box's hardware: a
  # TT_FATAL/TT_THROW/Error naming physical_grouping_descriptor or "SAT placement: mesh ... has no grouping" (seen on a quad host),
  # or "RuntimeError: bad file: ...arch.yaml" from a broken tt-metal tree (seen on a dev box), which counts only with
  # set_fabric_config/SetFabricConfig in the same tail: elsewhere it stays a plain WARN. Info,
  # warning and debug lines never count, as in fl_ccl_err: the PGD warning's "physical_grouping_descriptor" paths are in every load's log
  local t; t=$(tail -40 "$1" 2>/dev/null | grep -aivE '\| *(info|warning|debug|trace) *\|')
  grep -aqE '(TT_FATAL|TT_THROW|Error).*physical_grouping_descriptor|SAT placement: mesh' <<< "$t" \
    || { grep -aqE 'RuntimeError: bad file: .*\.yaml' <<< "$t" && grep -aqE 'set_fabric_config|SetFabricConfig' <<< "$t"; }
}
fl_c7_named() {  # <label> <tag> <stage and part> <rc> <c7 dir>: fl_c7's CCL or device error: one finding naming the chip from
  # its run.log by BDF and tray 1-4 (fl_attr), its capture (one fault.txt entry, fault_attr's) and FL_PIN for the sweep
  local w=$1 t=$2 sn=$3 rc=$4 d=$5
  fl_attr "$t" FAIL "$w rc $rc" 0 "$d/run.log"
  if [ -n "$FA_CHIPS" ]; then
    FL_PIN="$sn $w rc $rc, chip $(fc_tag $FA_CHIPS) ($FA_WHY)"
    FL_CAP_NAMED=1 fl_capture "${d##*/}" fail $(echo "$FA_LINES" | awk '{ print $2 }')   # FA_LINES column 2: plain BDFs, not fc_bdfk's chip@bdf
  else
    FL_PIN="$sn $w rc $rc (chip not named; every chip captured)"
    FL_CAP_NAMED=1 fl_capture "${d##*/}" fail $(fl_all)
  fi
}
fl_all() { echo "$RK_MAP" | awk '{ print $1 }'; }   # every chip's BDF
fl_ubb() { echo "tray $(rk_tray "$1")"; }   # bdf -> "tray N"
fl_lk() { echo "${1#0000:} ($(fl_ubb "$1"))"; }   # bdf -> "02:00.0 (tray 1)" (tray 1-4, no UBB number: by design)
fl_gate() {  # $1 the phase about to start: 0 ok; else FL_STOP set, nothing more loads or resets
  local o rc l; o=$(FLEET_PROC=${FLEET_PROC:-/proc} bash "$FL_DIR/gate.sh" "$FL_PGID" 2>&1); rc=$?
  echo "$1 $o" >> "$FL_OUT/gate.txt"; [ $rc = 0 ] && return 0
  l=$(grep -a '^gate busy: ' <<< "$o" | tail -1); l=${l:-$o}   # the verdict line, not the stderr info ("host load: ...", WARNING) before it
  FL_STOP="gate busy before $(fl_pn "$1"): ${l#gate busy: }"; fl_find "GATE BUSY before $(fl_pn "$1"): ${l#gate busy: }"; return 1
}
fl_discs() {  # this run's UMD discoveries that hold a JSON line (a failed one proves nothing)
  local f; for f in "$FL_OUT"/disc_*.json; do grep -qa '^{' "$f" 2>/dev/null && printf '%s ' "$f"; done
}
fl_peer() {  # $1 the reset about to run: 0 when it may run, else FL_PEER_WHY. Only this host's serving gate (rk_gate) can skip it;
  # no peer snapshot or peer gate
  if ! rk_gate; then   # this host or its quad serves (or cannot be verified): no C2/C9 reset
    FL_PEER_WHY=$RK_GATE_WHY; say "serving gate before $1: $FL_PEER_WHY; the reset is skipped"; return 1
  fi
  return 0
}
fl_attr() {  # <phase> <FAIL|HANG> <what> <probe 0|1> <file>...: one finding naming the chips (fault_attr), else every candidate
  local p=$1 k=$2 w=$3 pr=$4; shift 4
  if fault_attr "$p $k: $w" "$pr" "$@"; then
    local c e   # one finding per chip, so fl_report lists each under its own tray
    if [ "$(echo "$FA_LINES" | grep -c .)" -gt 1 ]; then
      while read -r c _ e; do [ -n "$c" ] && fl_find "$p $k $(fc_tag "$c"): $w (its error: ${e:0:140})"; done <<< "$FA_LINES"
    else fl_find "$p $k $FA_TAG: $w ($FA_WHY)"; fi
  else fl_find "$p UNATTRIBUTED $k $w ($FA_WHY); candidates: $FA_TAG"; fi
}
fl_kdwhy() { echo "known drop: tray $KD_TRAY is down and the load needs the 4x8 mesh"; }
# firmware not ready (lib/fw_check.sh FW_NR): those chips stay out of every device open, so the 4x8 mesh
# cannot open; C5's pairs come from the masked discovery and run
fl_nrwhy() { echo "firmware not ready on tray $(fl_nrtrays) ($(echo "$FW_NR" | awk 'NF { sub(/^0000:/, "", $1); print $1 }' | paste -sd, | sed 's/,/, /g')): the load needs the 4x8 mesh"; }
fl_nrtrays() { echo "$FW_NR" | while read -r b c v; do [ -n "$b" ] && rk_tray "$b"; done | sort -u | paste -sd, | sed 's/,/, /g'; }
fl_vis() {  # the TT_VISIBLE_DEVICES env for a whole-box device open: the map's chips less FW_NR's (none: the open sees all),
  # as UMD indices (a /dev number masked the other tray's chips on trays 3/4)
  [ "${KD_ON:-0}" = 1 ] || [ -n "${FW_NR:-}" ] || return 0
  # shellcheck disable=SC2046  # one BDF per word
  rk_umd $(echo "$RK_MAP" | awk -v nr="$(echo "${FW_NR:-}" | awk 'NF { printf "%s ", $1 }')" \
    'BEGIN { n = split(nr, a, " "); for (i = 1; i <= n; i++) x[a[i]] = 1 } $2 != "?" && !($1 in x) && !(("0000:" $1) in x) { print $1 }')
  return 0
}
fl_skip() { FL_SKIPPED+=("$1 ($2)"); say "$1 not run: $2"; }
fl_rskip() { FL_RESET_SKIPPED+=("$1 (SKIPPED-GATE: $2)"); fl_skip "$1" "SKIPPED-GATE: $2"   # a C2/C9 reset that did not run
  return 0
}
fl_hungf() { echo "${FLEET_HUNG_FILE:-/var/tmp/tray-test-hung.$(id -u)}"; }
fl_boot() { cat "${FLEET_BOOT_ID:-/proc/sys/kernel/random/boot_id}" 2>/dev/null; }
fl_hung_mark() {  # $1 what is left hung: remember it for the next run on this boot (fl_start)
  local f; f=$(fl_hungf)
  { echo "boot_id $(fl_boot)"; echo "pair $1"; echo "run $OUT"; echo "time $(date '+%F %T %Z')"; } > "$f.tmp.$$" 2>/dev/null \
    && mv -f "$f.tmp.$$" "$f" && say "hung chips recorded in $f" || say "could not write $f"
}
fl_hung_clear() { local f; f=$(fl_hungf); [ -e "$f" ] && rm -f "$f" && say "hung marker $f cleared (clean reset and recheck)"; return 0; }
fl_links() {  # PCIe link per chip: "<bdf> <speed>x<width> <slot position>"
  local b c s w u; echo "$RK_MAP" | while read -r b c; do [ -n "$b" ] || continue   # builtins only: no fork per chip
    s=; { read -r s < "$SYS/bus/pci/devices/$b/current_link_speed"; } 2>/dev/null; w=; { read -r w < "$SYS/bus/pci/devices/$b/current_link_width"; } 2>/dev/null
    u=${b#*:}; echo "$b ${s// /}x$w $(( 0x${u%%:*} & 63 ))"; done
}
fl_c1() {  # $1 pre|post: AER per endpoint and root port, dmesg since the start, SEL (read-only); pre also judges the
  # history (kernel journal of this boot and 7 days, SEL 7 days, SDR, F18 chain), post the run window (AER, link, dmesg, new SEL)
  local d=$FL_OUT/c1_$1 l; mkdir -p "$d"   # each read probes its own exact command (sudo_t)
  python3 "$FL_DIR/fleet_py.py" aer "$SYS" $(echo "$RK_MAP" | awk '{ print $1 }') > "$d/aer.tsv" 2>&1
  sudo_t 120 dmesg > "$d/dmesg.txt" 2>&1; cut -d' ' -f1 /proc/uptime > "$d/uptime"
  sudo_t 60 ipmitool sel elist > "$d/sel.txt" 2>&1 || echo "(ipmitool sel elist rc $?)" >> "$d/sel.txt"
  if [ "$1" = pre ]; then
    # _TRANSPORT=kernel, not -k: -k is the current boot only
    sudo_t 120 journalctl _TRANSPORT=kernel -b 0 -o short-unix --no-pager > "$d/journal_boot.txt" 2>&1
    sudo_t 120 journalctl _TRANSPORT=kernel --since "7 days ago" -o short-unix --no-pager > "$d/journal_7d.txt" 2>&1
    sudo_t 90 ipmitool sdr elist > "$d/sdr.txt" 2>&1
    local hs=()   # since the last repair: HIST lines -> hist.txt -> one C1 INFO line per tray, never fl_find
    [ -n "${FLEET_HISTORY_SINCE:-}" ] && hs=("$(date -d "$FLEET_HISTORY_SINCE" +%s)" "$FLEET_HISTORY_SINCE")
    { python3 "$FL_DIR/fleet_py.py" journal "$d/journal_boot.txt" "$d/journal_7d.txt" "$d/aer.tsv" "${FLEET_AER_BOOT_WARN:-25}" "${FLEET_AER_7D_WARN:-100}" "${hs[@]}"
      python3 "$FL_DIR/fleet_py.py" sel "$d/sel.txt" "$(date +%s)" "${FLEET_SEL_BUSCOR_WARN:-20}" "${FLEET_SEL_FRB2_WARN:-3}" "${hs[@]}"
      python3 "$FL_DIR/fleet_py.py" sdr "$d/sdr.txt"
      python3 "$FL_DIR/fleet_py.py" chain "$d/aer.tsv" "$(date +%s)" ${hs:+--since "${hs[0]}"} "$d/journal_7d.txt" "$d/journal_boot.txt"; } 2>> "$d/judge.err" |
      python3 "$FL_DIR/fleet_py.py" hist1 "$d/history.txt" 2>> "$d/judge.err" | while read -r l; do   # one WARN per check
      case $l in HIST\ *) echo "$l" >> "$d/hist.txt" ;; *) fl_find "$l" ;; esac; done
    if [ ${#hs[@]} -gt 0 ]; then
      python3 "$FL_DIR/fleet_py.py" histinfo "$FLEET_HISTORY_SINCE" "$d/hist.txt" 2>> "$d/judge.err" > "$FL_OUT/history_info.txt"
      [ -s "$FL_OUT/history_info.txt" ] || echo "C1 INFO: no history events before repair $FLEET_HISTORY_SINCE" > "$FL_OUT/history_info.txt"
      while read -r l; do say "$l"; done < "$FL_OUT/history_info.txt"
    fi
    return 0
  fi
  local p=$FL_OUT/c1_pre n   # AER rise and link drop over the run: lk_check post (run.sh, before fl_end)
  fl_links > "$d/links.txt"
  n=$(awk -v t0="$(cat "$p/uptime")" -F'[][]' 'NF > 2 && $2 + 0 > t0 + 0' "$d/dmesg.txt" |
    grep -aiE 'AER|machine check|mce:|i2c_target|tenstorrent.*(error|fail|timeout)' | tee "$d/dmesg_flagged.txt" | grep -c .)
  # SMBus i2c_target lines: report only (T5 baseline: 0 to 47563 per host, no clean/bad split)
  l=$(grep -ac 'i2c_target' "$d/dmesg_flagged.txt"); n=$(( n - l ))
  [ "$l" -gt 0 ] && say "dmesg: $l SMBus i2c_target line(s) during the run (report only)"
  [ "$n" -gt 0 ] && fl_find "DMESG WARN $n line(s) during the run (AER/MCE/tt-kmd): $(grep -av 'i2c_target' "$d/dmesg_flagged.txt" | head -1 | cut -c1-140)"
  # new SEL events; the finding names tray N for the BMC's UBBn, sel_flagged.txt keeps the raw rows
  python3 "$FL_DIR/fleet_py.py" selnew "$p/sel.txt" "$d/sel.txt" "$d/sel_flagged.txt" 2>> "$d/judge.err" | while read -r l; do fl_find "$l"; done
  # F18 chain (ARC timeout -> root port AER -> MCE) during the run: FAIL; pre already reported one from before the run
  sudo_t 120 journalctl _TRANSPORT=kernel -b 0 -o short-unix --no-pager > "$d/journal_boot.txt" 2>&1
  local t1; t1=$(date +%s)   # the run's end, after the post read: kernel AER events stamped from FL_T0 to here are graded, history is not
  python3 "$FL_DIR/fleet_py.py" aerrun "$d/journal_boot.txt" "$d/aer.tsv" "$FL_T0" "$t1" "${FLEET_AER_RATE_WARN:-6}" "${FLEET_AER_MIN:-2}" "$FL_OUT/findings.txt" 2>> "$d/judge.err" | while read -r l; do fl_find "$l"; done
  python3 "$FL_DIR/fleet_py.py" chain "$d/aer.tsv" "$FL_T0" "$d/journal_boot.txt" 2>> "$d/judge.err" | grep -a '^CHAIN FAIL' | while read -r l; do
    grep -qxF "$l" "$FL_OUT/findings.txt" 2>/dev/null || fl_find "$l"; done
  return 0
}
fl_c3() {  # $1 label: UMD discovery (read-only device open) -> FL_OUT/disc_<label>.json, the count line in count_<label>.txt
  # (fl_c3cmp prints it). Its FAIL/WARN lines print as C3 finds them, not only in the end report (gap G6).
  fl_c3eth "$1"
  local vis=()   # --known-dropped-tray: tray N's dead chips break the UMD open, so it sees the 24 others only;
  # firmware not ready (FW_NR): UMD waits 300 s on such a chip at every open (results/t3008), so it is left out
  { [ "${KD_ON:-0}" = 1 ] || [ -n "${FW_NR:-}" ]; } && vis=(env TT_VISIBLE_DEVICES="$(fl_vis)")
  [ "$1" = pre ] || fl_c3both "$1"
  pg_run timeout "${FL_C3_DISC_T:-180}" "${AS[@]}" "${vis[@]}" "$PY" "$FL_DIR/fleet_py.py" disc > "$FL_OUT/disc_$1.json" 2> "$FL_OUT/disc_$1.err" || { FL_C3_RC=$?; say "C3 $1: count failed ($FL_OUT/disc_$1.err)"; return 1; }
  python3 "$FL_DIR/fleet_py.py" count "$FL_OUT/disc_$1.json" > "$FL_OUT/count_$1.txt"
  fl_c3links "$1"
}
# C3's tt-umd discovery failed after the start reset: every load stage after it hung 5 min at
# set_fabric_config (a quad host, C3 c2 timed out on c1:00.0 (tray 3) "Still waiting on ARC firmware": two bus-bar
# quiet-bursts, then no verdict past 45 min). FL_STOP skips every later load stage (fl_plan, fl_loads; C9 as for any stop),
# one C3 FAIL line per chip tt-umd named in its log or missing from PCIe, and fl_verdict prints an INCOMPLETE RESULT naming them.
# No reset is added: this is the stop after goal 9's health step.
fl_c3stop() {  # $1 the C3 label, $2 what it came after
  local f=$FL_OUT/disc_$1 b n m l
  FL_C3STOP="C3 $1 tt-umd discovery failed after the $2 ($([ "${FL_C3_RC:-}" = 124 ] && echo "timed out after 180 s" || echo "exit ${FL_C3_RC:-?}"))"
  FL_C3STOP_CHIPS=""; FL_STOP=${FL_STOP:-$FL_C3STOP}
  while IFS=$'\t' read -r n m; do   # "device N" / "PCIe ID N" in tt-umd's log lines: its last warning/error/wait line per chip
    b=$(echo "$RK_MAP" | awk -v n="$n" '$2 == n { print $1; exit }'); [ -n "$b" ] || continue
    l="$(fl_lk "$b") chip $n"; FL_C3STOP_CHIPS="${FL_C3STOP_CHIPS:+$FL_C3STOP_CHIPS; }$l"
    fl_find "C3 FAIL $l: not discovered by tt-umd after the $2: $m ($f.json)"
  done < <(fl_c3named "$f")
  while read -r b n; do   # a chip gone from PCIe (fl_c2 and C3g exit 6 on these; --known-dropped-tray's tray N stays named)
    l="$(fl_lk "$b") chip $n"; FL_C3STOP_CHIPS="${FL_C3STOP_CHIPS:+$FL_C3STOP_CHIPS; }$l (missing from PCIe)"
    fl_find "C3 FAIL $l: missing from PCIe after the $2"
  done < <(rk_snap 2>/dev/null | awk '$3 != "ok" { print $1, $2 }')
  say "$FL_C3STOP: no load stage runs ($FL_STOP); chips: ${FL_C3STOP_CHIPS:-none named by tt-umd ($f.json)}"
}
fl_c3named() {  # <FL_OUT/disc_label>: "N<TAB>tt-umd's last warning/error/wait line" per chip its log names as "device N" or
  # "PCIe ID N" (seen on a quad host: "Read 0xffffffff over PCIe ID 0: the board should be reset")
  cat "$1.json" "$1.err" 2>/dev/null | grep -aiE '(device|pcie id) [0-9]+' | grep -aiE 'warn|error|critical|exception|still waiting|timed? ?out|fail|reset' |
    awk '{ if (!match($0, /([Dd]evice|PCIe ID) [0-9]+/)) next; n = substr($0, RSTART, RLENGTH); sub(/.* /, "", n); m = $0
      sub(/.*\| */, "", m); sub(/ *\([A-Za-z0-9_.]+:[0-9]+\)[. ]*$/, "", m); sub(/^[A-Za-z_][A-Za-z0-9_.]*(Exception|Error): */, "", m); last[n] = substr(m, 1, 160); if (!(n in seen)) { seen[n] = 1; o[++k] = n } }
      END { for (i = 1; i <= k; i++) printf "%s\t%s\n", o[i], last[o[i]] }'
}
# fl_c3retry (a quad host's soak step: C3 c2 failed on 01:00.0 (tray 1) chip 0,
# "Read 0xffffffff over PCIe ID 0: the board should be reset", its config space answering and 32/32 on PCIe, so the stage
# never ran). C3 c2 failed after the C2 start reset: the cause line names each chip tt-umd named (BDF, tray 1-4) and its error;
# one more tt-smi -glx_reset (rk_reset, settled sa3; never tt-smi -r) with its root-port rescan and recount by BDF and tray;
# C3 c2 again. 0: C3 c2 counted after the retry (a C3 WARN per chip named the first time); 1: fl_c3stop follows, as before.
# --hold: 1 with no reset. Leaving the chips out and running the no-CCL fallback load on the others waits for
# fallback load on main.
fl_c3retry() {
  local f=$FL_OUT/disc_c2 n m b x named="" cause="" miss
  [ "${HOLD_ACTIVE:-0}" = 1 ] && return 1   # --hold: the chip is left as it is for tt-triage
  named=$(while IFS=$'\t' read -r n m; do b=$(echo "$RK_MAP" | awk -v n="$n" '$2 == n { print $1; exit }'); [ -n "$b" ] && printf '%s\t%s\t%s\n' "${b#0000:}" "$n" "$m"; done < <(fl_c3named "$f"))
  cause=$(echo "$named" | awk -F'\t' 'NF { printf "%s%s chip %s: %s", (k++ ? "; " : ""), $1, $2, $3 }')
  miss=$(rk_snap | awk '$3 != "ok"' | rk_trays)
  say "C3 c2: tt-umd discovery failed after the C2 start reset, $(rk_snap | grep -c ' ok$')/$(rk_total) chips on PCIe${miss:+ (missing: $miss)}: ${cause:-no chip named by tt-umd} ($f.err)"
  for x in json err; do cp "$f.$x" "${f}_first.$x" 2> /dev/null || :; done
  RK_STEP="C3 retry" rk_reset "$FL_OUT/c3_retry.log"
  say "C3 retry: one more $RK_RESET_LBL: ${RK_RESET_STAT:-NONE}${RK_RESET_WHY:+ ($RK_RESET_WHY)}; root-port rescan and recount follow ($RK_RESET_LOG)"
  RK_BEFORE=$(rk_snap); rk_recheck "C3 retry $RK_RESET_LBL" "$FL_OUT/c3_retry_recheck.txt"
  RK_RESCANNED=""; RK_RESCAN_RAN=""; RK_RESCAN_GSKIP=""; [ -n "${RK_RESET_SKIP:-}" ] || rk_rescan "C3 retry $RK_RESET_LBL" "$FL_OUT/c3_retry_rescan.txt"
  rk_map; echo "$RK_MAP" > "$FL_OUT/map.txt"; miss=$(rk_snap | awk '$3 != "ok"' | rk_trays)
  say "C3 retry: recount by BDF and tray: $(rk_snap | grep -c ' ok$')/$(rk_total) on PCIe${miss:+; missing: $miss}"
  fl_c3 c2 || return 1
  echo "$named" | while IFS=$'\t' read -r b n m; do [ -n "$b" ] && fl_find "C3 WARN $(fl_lk "$b") chip $n: not discovered by tt-umd after the C2 start reset ($m), discovered after one more glx_reset (${f}_first.err)"; done
  return 0
}
fl_c3both() {  # $1 a post-reset label (c2, c3g, c9): enabled ETH ports down before the run and still down after this reset
  # (seen on a quad host's 47:00.0) FAIL at the first C3 after a reset, minute 2-3 after C2, not only after C9 (
  # gap G6). A later reset rechecks them: a chip and ports already named are said again, not counted twice. Off the quads,
  # the run's link map and the newest peer gate (the one before this reset) let a cluster node judge ports 10-13.
  local l k h=()
  # a later recheck passes C2's snapshot, so a far-end group port that came back at C2 and dropped again FAILs
  [ "$1" != c2 ] && [ -s "$FL_OUT/ttsmi_c2.json" ] && h=(--held "$FL_OUT/ttsmi_c2.json")
  # shellcheck disable=SC2046  # fl_discs is a list of paths
  python3 "$FL_DIR/fleet_py.py" ethboth "$FL_OUT/ttsmi_pre.json" "$FL_OUT/ttsmi_$1.json" "$1" --host "$(hostname -s)" "${h[@]}" --map $(fl_discs) 2>/dev/null | while read -r l; do
    # a sub-torus judged quad host's cross-quad ports (ETH EXT): out of scope, like fl_c3links' EXT lines
    case $l in "ETH EXT "*) say "C3 $1: ${l#ETH EXT }: out of scope"; continue ;; esac
    k=${l%% down before the run*}
    if grep -aqF "$k down before the run" "$FL_OUT/findings.txt" 2>/dev/null; then say "C3 $1 recheck: $k still down after the $1 reset (named above)"
    else fl_find "$l"; fi; done
}
fl_c3links() {  # $1 label: fleet_py.py ethlinks on the discovery against fleet/eth_map.txt. A local ETH port not up
  # FAILs naming BDF, port, tray and the far end. A QSFP port not up (a link to another host) is a WARN, not judged: no
  # state of the far host is read. Each line once.
  # C3 ETH rule: after a reset (any label but pre) an in-box port that UMD
  # or tt-smi (ttsmi_<label>.json; tt-smi only on hosts in fleet/eth_peers.txt, elsewhere ethboth judges it) still shows down FAILs ("; still down after the <label> reset"), and ethback first
  # appends to each earlier label's line whether this reset brought its ports back (C9: back up | still down).
  # pre's in-box port lines wait for C2's reset (eth_pre_local.txt,
  # said as "C3 pre: ..."): a port up after it is a WARN "recovered by C2 reset", one still down C2's FAIL; no C2 verdict
  # (reset skipped, run stopped, C3 c2 count failed): fl_c3pend makes them FAILs.
  # on a cluster node fleet_py.py reads its own map file (UMD's in-box link map) instead, so a FAIL names the right peer.
  # ports that leave the box on this platform (X lines; on a cluster node: ports 0,1 of asic 4) come as ETH EXT
  # lines, never a finding: pre holds them (eth_pre_ext.txt) and fl_c3pend makes them INCOMPLETE if no reset judged them.
  local out l k p mp=$FL_DIR/eth_map.txt hs=(--host "$(hostname -s)") pf=$FL_OUT/eth_pre_local.txt
  local xf=$FL_OUT/eth_pre_ext.txt nf=$FL_OUT/eth_notcabled.txt
  local re='^ETH FAIL .* pre: local ETH ports '
  [ "$1" != pre ] && [ -n "${FL_RESET_T:-}" ] && hs+=(--since "$FL_RESET_T")
  if [ "$1" != pre ]; then
    hs+=(--smi "$FL_OUT/ttsmi_$1.json" --named "$FL_OUT/findings.txt" --nc "$nf")
    [ "$1" = c2 ] && [ -s "$pf" ] && hs+=(--pre "$pf")
    python3 "$FL_DIR/fleet_py.py" ethback "$FL_OUT/findings.txt" "$FL_OUT/finding_stage.tsv" "$FL_OUT/disc_$1.json" \
      "$FL_OUT/ttsmi_$1.json" "$1" "$mp" "${hs[1]}" 2>/dev/null | while read -r l; do say "$l"; done
  fi
  out=$(python3 "$FL_DIR/fleet_py.py" ethlinks "$FL_OUT/disc_$1.json" "$1" "$mp" /dev/null "${hs[@]}" 2>/dev/null)
  if grep -aq '^ETH EXT ' <<< "$out"; then
    [ "$1" = pre ] && grep -a '^ETH EXT ' <<< "$out" > "$xf"
    grep -a '^ETH EXT ' <<< "$out" | while read -r l; do l=${l#ETH EXT }
      if [ "$1" = pre ]; then say "C3 pre: $l; out of scope: a link that leaves the box (QSFP to another node), not judged by a single-box run"
      else say "C3 $1: $l after the $1 reset: out of scope"; fi; done
    out=$(grep -av '^ETH EXT ' <<< "$out")
  fi
  [ "$1" = c2 ] && [ -e "$xf" ] && mv -f "$xf" "$xf.judged"   # out of scope either way (fl_c3pend)
  # an axis whose wrap ports were all down at pre and still are after the reset is a
  # box cabled without them (partial torus): ETH NOT-CABLED, INFO, never a finding; fl_report lists them on one line
  if grep -aq '^ETH NOT-CABLED ' <<< "$out"; then
    grep -a '^ETH NOT-CABLED ' <<< "$out" >> "$nf"
    grep -a '^ETH NOT-CABLED ' <<< "$out" | while read -r l; do say "C3 $1: INFO ${l#ETH NOT-CABLED }"; done
    out=$(grep -av '^ETH NOT-CABLED ' <<< "$out")
  fi
  # the in-box cabling layout C3 judged this discovery against, the closest of the valid ones
  # (fleet_py.py closest): INFO, never a finding; only links missing from it are judged
  if grep -aq '^ETH LAYOUT ' <<< "$out"; then
    grep -a '^ETH LAYOUT ' <<< "$out" | while read -r l; do say "C3 $1: INFO layout ${l#ETH LAYOUT "$1": }"; done
    out=$(grep -av '^ETH LAYOUT ' <<< "$out")
  fi
  if [ "$1" = pre ]; then
    grep -a "$re" <<< "$out" > "$pf"; out=$(grep -av "$re" <<< "$out")
    grep -a . "$pf" | while read -r l; do say "C3 pre: ${l#ETH FAIL }; C2's reset judges it (WARN if it brings them back, else FAIL)"; done
  elif [ "$1" = c2 ] && [ -e "$pf" ]; then mv -f "$pf" "$pf.judged"; fi
  grep -a . <<< "$out" | while read -r l; do
    k=${l%% "$1": *}; p=${l#* "$1": }; p=${p%%;*}
    grep -aF "$k " "$FL_OUT/findings.txt" 2>/dev/null | grep -qF "$p" || fl_find "$l"; done
}
fl_c3pend() {  # $1 why: pre's held-back in-box port lines (fl_c3links pre) that no C2 verdict reached become FAILs
  # its external port lines (cluster node, eth_pre_ext.txt) are links that leave the box: out of scope,
  # never a finding, whether or not a reset ran (fl_report lists them on one "Out of scope:" line)
  local pf=$FL_OUT/eth_pre_local.txt xf=$FL_OUT/eth_pre_ext.txt l
  [ -e "$xf" ] && mv -f "$xf" "$xf.judged"
  [ -e "$pf" ] || return 0
  grep -a . "$pf" | while read -r l; do fl_find "$l; no C2 verdict ($1)"; done
  mv -f "$pf" "$pf.judged"
}
fl_c3eth() {  # $1 label: tt-smi snapshot -> ETH WARN per chip with ENABLED_ETH ports not up (ETH_LIVE_STATUS low 16
  # bits; T5: 0 on 61/63 chips). WARN only until a fault is proven. A chip already named with the same ports is not repeated.
  local smi=${TRAY_REPRO_TT_SMI:-tt-smi} l k p
  command -v "$smi" > /dev/null || return 0
  local vis=()   # as fl_c3: no open of a chip whose firmware is not ready
  [ -n "${FW_NR:-}" ] && vis=(env TT_VISIBLE_DEVICES="$(fl_vis)")
  pg_run timeout "${FL_C3_SMI_T:-120}" "${AS[@]}" "${vis[@]}" "$smi" -s --snapshot_no_tty < /dev/null > "$FL_OUT/ttsmi_$1.json" 2> "$FL_OUT/ttsmi_$1.err"
  local out
  out=$(python3 "$FL_DIR/fleet_py.py" ethdown "$FL_OUT/ttsmi_$1.json" "$1" 2>/dev/null)
  if [ -n "$out" ]; then  # the first snapshot can catch link training in flight; settle and recheck before WARNing
    sleep "${FL_ETH_RECHECK_S:-3}"
    if pg_run timeout "${FL_C3_SMI_T:-120}" "${AS[@]}" "${vis[@]}" "$smi" -s --snapshot_no_tty < /dev/null > "$FL_OUT/ttsmi_$1_re.json" 2> "$FL_OUT/ttsmi_$1_re.err"; then
      out=$(python3 "$FL_DIR/fleet_py.py" ethdown2 "$FL_OUT/ttsmi_$1.json" "$FL_OUT/ttsmi_$1_re.json" "$1" "${FL_ETH_RECHECK_S:-3}" 2>/dev/null)
    fi
  fi
  printf '%s\n' "$out" | while read -r l; do
    [ -n "$l" ] || continue
    k=${l%% "$1": *}; p=${l#* "$1": }; p=${p%% (*}   # "ETH WARN <bdf> (tray N)" and "ETH ports [..] enabled but not up"
    grep -aF "$k " "$FL_OUT/findings.txt" 2>/dev/null | grep -qF "$p" || fl_find "$l"; done
}
fl_c3cmp() {  # $1 before label, $2 after label: LINKS WARN when local ETH or QSFP counts went down
  local a b; a=$(cat "$FL_OUT/count_$1.txt" 2>/dev/null); b=$(cat "$FL_OUT/count_$2.txt" 2>/dev/null)
  say "C3 ports $1: ${a:-none} | $2: ${b:-none}"; [ -n "$a" ] && [ -n "$b" ] || return 0
  local k va vb
  for k in eth_local qsfp_up; do
    va=$(echo "$a" | grep -o "$k=[0-9]*" | cut -d= -f2); vb=$(echo "$b" | grep -o "$k=[0-9]*" | cut -d= -f2)
    [ "${vb:-0}" -lt "${va:-0}" ] && fl_find "LINKS WARN after $2: $k $vb (before $va)"
  done; return 0
}
fl_c0() {  # C0 on top of rk_map: a chip off the others' firmware, the host-link check (lib/link_check.sh: off PCIe, LnkSta
  # unreadable, link below the design, on the chip or its root port) as fleet findings; missing chips held for fl_c0settle
  local b c sh maj
  fl_links > "$FL_OUT/c0_links.txt"; lk_check pre
  [ -n "${FW_FAIL:-}" ] && echo "$FW_FAIL" >> "$FL_OUT/findings.txt"   # run.sh's fw_preflight (lib/fw_check.sh), already printed
  maj=$(awk '{ print $2 }' "$FL_OUT/c0_links.txt" | sort | uniq -c | sort -rn | awk 'NR == 1 { print $2 }')
  : > "$FL_OUT/c0_missing.txt"; sh=$(rk_short); [ -n "$sh" ] && fl_c0miss - 0 "$sh ($RK_N/$RK_WANT on PCIe)"
  echo "$RK_MAP" | while read -r b c; do [ -n "$b" ] && [ "$(rk_state "$b")" != ok ] &&
    ! grep -q "^PCIE FAIL ${b#0000:} " "$OUT/link/findings.txt" 2>/dev/null && fl_c0miss "$b" 0 "chip $c: not answering"; done
  say "C0 preflight: $RK_N/$RK_WANT Blackhole chips on PCIe${KD_LINE:+ (tray $KD_TRAY: known drop)}, link $maj on $(grep -c " $maj " "$FL_OUT/c0_links.txt")"
}
# fl_c0miss <bdf, or - for the tray count> <1: also to fault.txt> <what>: a chip missing at C0 is not yet a finding (
# 'The test is supposed to get the device healthy before starting the test!'); fl_c0settle decides once the
# start reset (rk_start: warm reset, recount, root port rescan, tray re-power) ran or could not run
fl_c0miss() {
  printf '%s|%s|%s\n' "$1" "$2" "$3" >> "$FL_OUT/c0_missing.txt"
  if [ "$1" = - ]; then say "C0: $3 at the start; the start reset and recount before any verdict"
  else say "C0: ${1#0000:} ($(fl_ubb "$1")) $3 at the start; the start reset and recount before any verdict"; fi
}
# fl_c0settle [start reset dir]: C0's missing chips after the start reset (rk_start, its re-check files in the dir) or, with
# no dir, after a recount with no reset. A chip that answers now is a PREFLIGHT WARN recovery line naming its BDF, its tray
# 1-4 and the first step whose re-check saw it answer, never a FAIL; a chip still missing is its PREFLIGHT FAIL
# line (KNOWN-DROP after kd_settle), the tray count line only while a tray is still short. Runs once per run.
fl_c0settle() {
  local p=${FL_OUT:-/nonexistent}/c0_missing.txt d=${1:-} b ft w l s f c sh
  declare -F lk_settle > /dev/null && lk_settle   # the host-link lines C0 held for chips not answering
  [ -s "$p" ] || return 0
  mv "$p" "$p.settled"
  while IFS='|' read -r b ft w; do
    if [ "$b" = - ]; then
      sh=$(rk_short); [ -n "$sh" ] && fl_find "PREFLIGHT FAIL $sh ($RK_N/$RK_WANT on PCIe${d:+ after the start reset})"; continue; fi
    l="PREFLIGHT FAIL ${b#0000:} ($(fl_ubb "$b")) $w"
    if [ "$(rk_state "$b")" != ok ]; then   # still missing: its PCIE FAIL line from lk_settle names it, else this line
      grep -q "^PCIE FAIL ${b#0000:} " "$OUT/link/findings.txt" 2> /dev/null && continue
      fl_find "$l"; [ "$ft" = 1 ] && echo "$(date '+%F %T %Z') $l" >> "$OUT/fault.txt"; continue; fi
    s=""; [ -s "${RK_RESET_LOG:-/nonexistent}" ] || s="the recount (no reset ran)"   # follows RK_RESET_LOG only
    # --known-dropped-tray=N: the start glx_reset is the tray re-power that brings tray N back, main's wording
    [ -n "$d" ] && for f in "recheck|the $([ -n "${KD_TRAY:-}" ] && echo "tray re-power" || echo glx_reset) (tt-smi -glx_reset, start reset)" "recheck_rescan|the root port rescan after the glx_reset (tt-smi -glx_reset, start reset)" \
        "recheck_glx|the tray re-power (tt-smi -glx_reset, start reset)" "recheck_glx_rescan|the root port rescan after the tray re-power (start reset)"; do
      awk -v b="$b" '$1 == b && $3 == "ok" { f = 1 } END { exit !f }' "$d/${f%%|*}.txt" 2> /dev/null && { s=${f#*|}; break; }; done
    c=$(echo "$RK_MAP" | awk -v b="$b" '$1 == b { print $2 }')
    w=${w#chip * }; [ "$ft" = 1 ] && w="${w%%;*} (${w#*; })"
    fl_find "PREFLIGHT WARN ${b#0000:} ($(fl_ubb "$b"))${c:+ chip $c}: recovered: $w at the start, back after ${s:-the glx_reset (tt-smi -glx_reset, start reset)}"
  done < "$p.settled"
}
fl_start() {  # $1 out dir (run.sh's OUT)
  FL_OUT=$1/fleet; mkdir -p "$FL_OUT"; RK_RESET_TIMES=$FL_OUT/reset_times.tsv; FL_PGID=$(ps -o pgid= -p $$ | tr -d ' '); FL_T0=$(date +%s)
  by_start   # the broker yield watcher runs from the run's start, so setup (C2, C3) is watched too (a no-op off a broker)
  if [ "$FLEET" = repro-linka ]; then say "fleet repro-linka: phases P0 C0 C1 C3 C2 link-A repro ($(fl_linka_plan)) C9; logs in $FL_OUT"; fl_t P0
  elif [ "$FLEET" = repro-abortfab ]; then say "fleet repro-abortfab: phases P0 C0 C1 C3 C2 abort-fabric repro ($(fl_abortfab_plan)) C9; logs in $FL_OUT"; fl_t P0
  else
  [ "${FL_SMOKE:-0}" = 1 ] && say "smoke (run.sh --mode=smoke): plain run.sh's steps, durations cut to seconds (FLEET_SMOKE_S ${FLEET_SMOKE_S:-2}): mix chunks ${FLEET_CHUNK_MIN} min, quiet ${FLEET_GAP_S} s, qb ${FLEET_BUSBAR_QB_QUIET_S}+${FLEET_BUSBAR_QB_BURST_S} s, step ${FLEET_BUSBAR_QUIET_S}+${FLEET_BUSBAR_BURST_S} s, heavy ${FLEET_BUSBAR_HEAVY_S} s, G3 ${FLEET_BUSBAR_RW_S} s, G3b ${FLEET_BUSBAR_DBI_WALL_S} s, N1 ${FLEET_BUSBAR_L1_WALL_S} s, pairs ${FLEET_PAIR_S} s, peak poll ${FLEET_FASTPOLL_S} s; a smoke PASS is no soak verdict"
  [ -z "${FLEET_REPRO:-}" ] || say "repro $FLEET_REPRO (repro.sh): load stages $(fl_load_phases | tr -s ' ' | sed 's/^ *$/none/')"
  say "fleet $FLEET: stages $(awk -F'\t' -v bb="$(fl_bb_plan)" '!/^#/ { printf "%s%s %s%s", (n++ ? ", " : ""), $1, $3, ($2 == "busbar" ? " (" bb ")" : $2 == "numeric" ? " (inside 4)" : $2 == "sweep" ? " (on a fault only)" : "") }' "$FL_STAGES_TSV"); logs in $FL_OUT; power_policy=${TRAY_POWER_POLICY:-unknown} idle_power_down_grace_ms=${TRAY_IDLE_GRACE_MS:-unknown}"; fl_t P0
  fi
  fl_gate P0 || { vd_say "NOT RUN (host busy: ${FL_STOP#*: })"; echo "Couldn't start - host busy: ${FL_STOP#*: }"; exit 10; }
  ! declare -F tdp_start > /dev/null || tdp_start   # --tdp only now: the gate passed and this run holds the chips
  pvc_check > "$FL_OUT/pv_crash_check.txt"   # an earlier run's power virus leg the host died under: this box FAILs (pvc_verdict)
  if [ -n "$PVC_CRASH" ]; then fl_find "$PVC_CRASH"; elif [ -s "$FL_OUT/pv_crash_check.txt" ]; then say "$(cat "$FL_OUT/pv_crash_check.txt")"; fi
  local hf hp; hf=$(fl_hungf)
  if [ -s "$hf" ]; then   # an earlier run left chips hung with no reset: only a reset may clear them
    hp=$(sed -n 's/^pair //p' "$hf" | head -1)
    local bt mt; bt=$(awk -v t="${FLEET_BTIME:-}" '/^btime / { b = $2 } END { print (t != "" ? t : b) }' /proc/stat 2>/dev/null); mt=$(stat -c %Y "$hf" 2>/dev/null)
    if [ -n "$bt" ] && [ -n "$mt" ] && [ "$mt" -lt "$bt" ]; then   # a reboot clears any hang: the marker no longer means anything
      say "stale hung-chip marker from before boot $(date -d "@$bt" '+%F %T %Z'): $hp; set aside"; mv -f "$hf" "$hf.stale-$(date +%Y%m%d-%H%M%S)"
    elif [ "$(sed -n 's/^boot_id //p' "$hf" | head -1)" != "$(fl_boot)" ]; then say "hung marker $hf is from another boot; ignored"; rm -f "$hf"
    else
      fl_peer start-hung || { say "start check: chips left hung by an earlier run ($hf) and no reset may run: $FL_PEER_WHY"; echo "SKIP tray $(echo "$hp" | sed -n 's/.*(tray \([1-4]\)).*/\1/p' | head -1): chips left hung by an earlier run: $(echo "$hp" | sed 's/^[A-Z-]* HANG //')"; vd_say "NOT RUN (couldn't start: chips left hung by an earlier run)"; echo "Couldn't start - chips left hung by an earlier run: $hp"; exit 6; }
      say "chips left hung by an earlier run ($hp): the start reset (C2) runs first"
    fi
  fi
  fl_t P1; fl_psu_bg "before load"   # the before-load PSU read runs in the background over C0..C2 (fl_psu_join 1 below, before any load)
  echo "$RK_MAP" > "$FL_OUT/map.txt"; fl_c0; fl_c1 pre; fl_c3 pre
  if [ "$RK_N" != "$RK_WANT" ] || [ -n "$(lost_trays)" ]; then   # rk_start's start reset reaches the QSFP peers too
    if ! fl_peer start; then   # --known-dropped-tray with only tray N short: named the known fault, no reset
      fl_c0settle
      [ "${KD_PEND:-0}" = 1 ] && kd_settle "$FL_PEER_WHY" || { say "start check: chips missing and no reset may run; nothing was loaded"; fl_c3pend "no reset may run"; lk_stop; vd_say "NOT RUN (couldn't start: chips already dropped)"; echo "Couldn't start - Chips are already dropped"; exit 6; }
    else rk_start "$1"; fi
    echo "$RK_MAP" > "$FL_OUT/map.txt"
  fi
  fl_c0settle   # a chip C0 saw missing that answered again before any reset
  fl_hsc_latch_start   # tray HSC STATUS_WORD/STATUS_INPUT before C2, then one CLEAR_FAULTS per HSC unless one reads OFF
  pf_prep   # the PSU sampler's route and SDR dump (in-band IPMI) only after the HSC latch reads and CLEAR_FAULTS; C2 covers the dump
  rk_tmark "$RK_RESET_TIMES" begin C2; fl_c2; rk_tmark "$RK_RESET_TIMES" end C2
  local w="after the start reset (C2)"
  [[ " ${FL_RESET_SKIPPED[*]:-} " == *" C2 (SKIPPED-GATE"* ]] && w="after the start check (the C2 reset was skipped)"
  fw_nr_settle "$w"   # no-op when fl_c2 already did it
  fl_psu_join 1   # a lost PSU or lost redundancy starts no load (FL_PSU_STOP); the read began at P1
}
fl_regate() {  # $1 rk_hook's step: after the hook's pre call (rk_pre), the C9 and power virus (PV) resets' fleet gates again (gate.sh, this host's serving gate); 1 when one refuses
  local st=${1%-glx}; st=${st%-lost}; st=${st%-others}; st=${st%-tray}; case $st in C9|C9s|C9b|PV) ;; *) return 0 ;; esac
  if [ -n "${FL_STOP:-}" ]; then   # gate.sh already stopped the run: nothing more resets, no second finding
    RK_GATE_WHY=$FL_STOP; return 1; fi
  if ! fl_gate "$st"; then RK_GATE_WHY=$FL_STOP; return 1; fi
  if ! fl_peer "$st"; then RK_GATE_WHY=$FL_PEER_WHY; return 1; fi
  return 0
}
fl_c2trays() {  # the trays of the chips C2 left missing or unopenable: "FAIL tray N (k findings)", then tray_why.py's lines
  local u nf f=$FL_OUT/findings.txt
  for u in $(awk '$3 != "ok" { print $1 }' <<< "$RK_AFTER" | while read -r b; do rk_tray "$b" | cut -d' ' -f1; done | grep -x '[1-4]' | sort -u); do
    nf=$(grep -av ' WARN ' "$f" 2>/dev/null | grep -cE "tray $u([^0-9]|$)")
    echo "FAIL tray $u ($nf finding$([ "$nf" -gt 1 ] && echo s))"
    python3 "$FL_DIR/tray_why.py" "$f" "$u" --stages "$FL_OUT/finding_stage.tsv" 2>/dev/null
  done | tee -a "$FL_OUT/trays.txt"   # the run's last line reads them (rr_final)
}
fl_c2() {  # C2: tt-smi -glx_reset (never tt-smi -r), rescan, recount, recovery; FL_STOP when chips stay missing
  local ids j ok failed b t0 aw named af="" gs="" RK_STEP=C2   # RK_STEP: rk_hook's step (C2-glx)
  fl_t C2; fl_gate C2 || { fl_rskip C2 "$FL_STOP"; fl_c3pend "C2 reset skipped"; return 0; }
  fl_peer C2 || { FL_NORESET=$FL_PEER_WHY; fl_rskip C2 "$FL_PEER_WHY"; fl_c3pend "C2 reset skipped: $FL_PEER_WHY"; fl_c3nopre; return 0; }
  fl_c2redo   # the C3 pre baseline must date from after the start reset, before C2's own reset
  ids=$(echo "$RK_MAP" | awk '$2 != "?" { print $2 }' | tr '\n' ' '); t0=$(date +%s); FL_RESET_T=$t0
  if [ -n "${TRAYTEST_RESET_HOOK:-}${TRAYTEST_RESET_SYNC_DIR:-}" ]; then   # the hook's pre call can take 60 s: both gates again
    rk_hook pre C2
    if ! fl_gate C2; then rk_hook post C2 skipped; fl_rskip C2 "$FL_STOP"; fl_c3pend "C2 reset skipped"; return 0; fi
    if ! fl_peer C2; then rk_hook post C2 skipped; FL_NORESET=$FL_PEER_WHY; fl_rskip C2 "$FL_PEER_WHY"; fl_c3pend "C2 reset skipped: $FL_PEER_WHY"; fl_c3nopre; return 0; fi
  fi
  # Steer: C2 runs tt-smi -glx_reset through rk_reset, the same
  # function C9 uses (settled sa3), so OK, FAILED and NO RESET are decided in one place. tt-smi -r is never run. NO RESET only
  # when no glx_reset ran (--hold, TRAY_GLX_FIRST=0, TRAY_REPRO_NO_GLX=1, it cannot run here); then no reset result is judged
  # (no RESET FAIL, no RESET UNATTRIBUTED) and a missing chip is named as missing. A glx_reset that ran and failed goes
  # through rk_resfail below (RESET FAIL by BDF and tray 1-4). Every reset-result parse reads $RK_RESET_LOG (settled sa4).
  RK_C2_GLX=0; RK_GLX_TRIED=""; local c2none=0
  rk_reset "$FL_OUT/c2_reset.log"; failed=""
  case $RK_RESET_STAT in
    OK) RK_C2_GLX=1; ok=1
      say "C2 start reset (tt-smi -glx_reset): ok ($(( $(date +%s) - t0 )) s); root-port rescan and recount follow" ;;
    FAILED) ok=0
      say "C2: start reset FAILED: $RK_RESET_WHY; no further reset follows this step (rk_recover may still glx_reset), never tt-smi -r"
      say "C2 start reset (tt-smi -glx_reset): FAILED ($RK_RESET_WHY; $(( $(date +%s) - t0 )) s); root-port rescan and recount follow" ;;
    *) ok=0; c2none=1; FL_C2_NONE=${RK_RESET_WHY:-no glx_reset ran}   # no start reset ran (NO RESET)
      if [ "$RK_RESET_STAT" = "SKIPPED-GATE" ]; then   # a gate-skipped C2 reports INCOMPLETE like the other gate skips
        fl_rskip C2 "$RK_RESET_WHY"
        say "C2: reset not run: SKIPPED-GATE: $RK_RESET_WHY; never tt-smi -r"
      fi   # no reset ran, so rk_reset printed the one NO RESET line (settled sa3)
      rk_hook post C2 noreset
      say "C2 start reset: none ran ($RK_RESET_WHY; $(( $(date +%s) - t0 )) s); the root-port rescan and recount still follow" ;;
  esac
  named=$(rk_umd_named "$RK_RESET_LOG" | grep . | sort -u); local c2lbl="start $RK_RESET_LBL (C2)"
  [ "$c2none" = 1 ] && c2lbl="C2 recount (no reset ran)"
  RK_BEFORE=$(rk_snap); rk_recheck "$c2lbl" "$FL_OUT/c2_recheck.txt"; aw=$RK_AFTER
  RK_RESCANNED=""; RK_RESCAN_RAN=""; RK_RESCAN_GSKIP=""; rk_rescan "$c2lbl" "$FL_OUT/c2_recheck_rescan.txt"
  named=$(rk_umd_named "$RK_RESET_LOG" | grep . | sort -u); rk_slowmark "$RK_RESET_LOG" C2
  # chips on PCIe that this user may not open (EACCES on the open or the glx_reset) are a setup error, not lost:
  # no recovery runs for it; a glx_reset that failed is still reported (RESET FAIL by BDF and tray, settled sa3), then the stop
  local ea=""; ! rk_eacces "$RK_AFTER" || ea=1
  [ -z "$ea" ] || [ "${RK_RESET_STAT:-}" = FAILED ] || rk_eacces_stop C2   # the glx_reset ran OK: no reset result to report
  if [ -z "$ea" ] && [ -n "$(echo "$RK_AFTER" | awk '$3 != "ok"')" ]; then
    if rk_noglx; then gs="TRAY_REPRO_NO_GLX=1"; rk_noglx_say "$RK_AFTER" C2   # warm reset only: no re-power, no re-check after it
    elif rk_can_glx; then RK_RECOVER_PRE="$c2lbl" rk_recover "$FL_OUT/c2_glx" "$FL_OUT/c2_recheck_glx" "tray re-power (C2)"; af=$RK_AFTER   # tray, then glx_reset; 2 rounds
    else gs="$RK_GLX_CAN_WHY"; say "C2: tray re-power skipped ($gs)"; fi
  fi
  [ -n "$ea" ] || ! rk_eacces "$RK_AFTER" || ea=1
  # one RESET FAIL line per chip the glx_reset left missing or failed on, recovered by the re-power or not.
  # No glx_reset ran (NO RESET above): no reset result is judged, only the recount names what is missing (settled sa3).
  if [ "$c2none" = 1 ]; then RK_RESET_NEW=""
    echo "$RK_AFTER" | awk '$3 != "ok"' | while read -r b c st; do   # named as missing, never as "after the glx_reset"
      fl_find "CHIP MISSING $b (tray $(rk_tray "$b" | cut -d' ' -f1)) chip $c: missing at C2, no reset ran ($RK_RESET_WHY)${gs:+; tray re-power skipped ($gs)}: INCOMPLETE, not proven a tray drop"; done
  else rk_resfail "C2 start reset" "$aw" "$named" "$ok" "${RK_RESET_WHY:-tt-smi -glx_reset}" "$af" "$gs"; fi
  if [[ $RK_RESET_NEW == "RESET UNATTRIBUTED"* ]]; then
    fl_attr RESET FAIL "glx_reset at the start (C2) failed and names no chip: $(grep -a . "$RK_RESET_LOG" 2>/dev/null | tail -1 | cut -c1-120); $(echo "$aw" | grep -c ' ok$')/$(rk_total) after it" 0 "$RK_RESET_LOG"
  else echo "$RK_RESET_NEW" | grep -a . | while read -r l; do fl_find "$l"; done; fi
  [ -z "$ea" ] || rk_eacces_stop C2   # after the reset's own report, before any tray line or verdict
  if [ -n "$(echo "$RK_AFTER" | awk '$3 != "ok"')" ]; then   # as rk_start: nothing loads, none of the verdict lines
    if [ "$c2none" = 1 ]; then say "C2: chips missing, no reset ran ($RK_RESET_WHY): $(echo "$RK_AFTER" | awk '$3 != "ok" { print $1 }' | while read -r b; do echo "$b (tray $(rk_tray "$b" | cut -d' ' -f1))"; done | paste -sd';' | sed 's/;/; /g') ($FL_OUT/findings.txt); nothing was loaded"
    else say "C2: chips still missing after the start reset ($FL_OUT/findings.txt); nothing was loaded"; fi
    fl_c2trays   # each failing tray's line with its exact failure under it, before the exit
    fl_c3pend "chips missing after the C2 reset"; lk_stop; vd_say "NOT RUN (couldn't start: chips already dropped)"; echo "Couldn't start - Chips are already dropped"; exit 6
  fi
  fl_c2fw "$FL_OUT/c2_fwnr"   # chips on PCI with firmware not ready count as missing: a second tray re-power round
  fw_nr_settle "after the start reset (C2)"   # still not ready -> FAIL tray N now, and left out of C3 c2 on
  # C9 keeps a reset-sized share of the run budget on a host whose C2 needed it
  if [ -n "$af${RK_RECOVERY:-}" ] || [ -n "${RK_RESET_FAILS:-}" ]; then
    ! declare -F rs_c9_share > /dev/null || rs_c9_share "$t0" "$([ -n "$af${RK_RECOVERY:-}" ] && echo "C2 needed a tray re-power" || echo "a glx_reset failed")"; fi
  rk_map; echo "$RK_MAP" > "$FL_OUT/map.txt"; fl_c3 c2 || fl_c3retry || { fl_c3stop c2 "C2 start reset$([ "${HOLD_ACTIVE:-0}" = 1 ] || echo " and one more glx_reset")"; return 0; }   # nothing loads; one retry
  fl_c3cmp pre c2; [ "$ok" = 1 ] && fl_hung_clear
  fl_c3glx
}
# fl_c2redo: C2 is about to run after a start reset (RK_START_RAN=1): retake the C3 pre baseline now, so
# fl_c3cmp/fl_c3glx/fl_ethdown compare C2 with ports as they were after the start reset. Reset skipped: baseline unchanged.
# Read-only discovery; it never acts on a chip, so --hold (HOLD_ACTIVE=1) leaves a hung chip as it is.
# fl_c2fw <dir> (17; 5/5 legs left tray 3 on PCI with firmware not ready after C2's one re-power, C3 then
# timed out on c1:00.0): a chip on PCI whose firmware is not ready counts as missing. Its tray gets one more in-band re-power round
# and a recount before C3; still not ready: it joins FW_NR, so fw_nr_settle prints FAIL tray N with the exact failure (BDF, tray 1-4)
# and the run exits non-zero. --hold (HOLD_ACTIVE=1), TRAY_REPRO_NO_GLX=1 or no root: nothing acts, the chip stays as it is.
fl_c2fw() {
  local t=$1.txt nr b c v sv=$RK_AFTER
  [ "${HOLD_ACTIVE:-0}" = 1 ] && return 0
  fw_table > "$t"; nr=$(fw_notready "$t"); [ -n "$nr" ] || return 0
  say "C2: firmware not ready on PCI after the reset: $(awk '{ sub(/^0000:/, "", $1); print $1 }' <<< "$nr" | paste -sd, | sed 's/,/, /g'); counted as missing, second tray re-power round"
  if rk_noglx || ! rk_can_glx; then say "C2: second re-power round skipped (no tray re-power allowed here)"
  else
    RK_AFTER=$({ awk 'NR == FNR { n[$1] = 1; next } $1 in n { $3 = "fwnr" } { print }' <(echo "$nr") <(echo "$RK_AFTER"); } | sed 's/ $//')
    RK_RECOVER_PRE="firmware not ready after C2" rk_recover "$1" "$1_recheck" "tray re-power (C2 firmware not ready)"
    fw_table > "$t"; nr=$(fw_notready "$t")
  fi
  RK_AFTER=$sv
  FW_NR=$(printf '%s\n%s\n' "${FW_NR:-}" "$nr" | awk 'NF && !s[$1]++'); FW_NR_DONE=""
  while read -r b c v; do [ -n "$b" ] || continue
    fl_find "C2 FAIL ${b#0000:} (tray $(rk_tray "$b")) chip $c: hardware fault: firmware not ready (tt_fw_bundle_ver $v) after the glx_reset and two tray re-power rounds"
  done <<< "$nr"
}
fl_c2redo() {
  [ "${RK_START_RAN:-0}" = 1 ] || return 0
  fl_c3redo; say "C3 pre baseline retaken after the start reset, before C2"
}
fl_c3nopre() {   # no count at all
  [ "${RK_START_RAN:-0}" = 1 ] && fl_c3redo   # C3 pre ran before the start reset
  [ -s "$FL_OUT/count_pre.txt" ] || fl_c3stop pre "start$([ "${RK_START_RAN:-0}" = 1 ] && echo " reset") (the C2 reset was skipped)"
}
# fl_c3redo (t3340 try 2 defect a): C3 pre runs before rk_start, so after a start reset its discovery is stale
# (a quad host: pre failed on 41:00.0 (tray 2) while tray 2 was down; the start reset brought 32/32 open
# 8 min later, and with C2 gate-skipped that stale failure stopped the run). Keep the old files as *_before_start_reset.*,
# wait for tt-umd discovery (rk_discpoll: every 10 s, up to 90 s), then run C3 pre again; a chip it still names is
# stopped on by fl_c3stop, by BDF and tray 1-4.
fl_c3redo() {
  local f sv=$RK_AFTER
  for f in disc_pre.json disc_pre.err count_pre.txt; do [ -e "$FL_OUT/$f" ] && mv -f "$FL_OUT/$f" "$FL_OUT/${f%.*}_before_start_reset.${f##*.}"; done
  RK_AFTER=$(rk_snap); rk_discpoll "$FL_OUT/disc_settle_pre.log" "start reset"; RK_AFTER=$sv
  say "C3 pre: its discovery ran before the start reset (kept as disc_pre_before_start_reset.*); again after it: $([ -n "$RK_DPOLL_NONE" ] && echo "settle not checked (no output)" || echo "$(rk_dsum) after $RK_DPOLL_S s ($RK_DPOLL_K tries)")"
  fl_c3 pre || :
}
fl_ethl() { grep -o 'eth_local=[0-9]*' "$FL_OUT/count_$1.txt" 2>/dev/null | cut -d= -f2; }   # $1 label: its C3 eth_local count
fl_ethdown() {  # $1 label: the local ETH links up at pre and down at $1, one "<bdf> (tray N) - <bdf> (tray M): ..." per chip pair
  python3 "$FL_DIR/fleet_py.py" ethdiff "$FL_OUT/disc_pre.json" "$FL_OUT/disc_$1.json" 2>/dev/null | paste -sd';' | sed 's/;/; /g'
}
# C2's glx_reset can leave local ETH links down (an outside galaxy, run 1: eth_local 248 of 256, then tt-metal's mesh
# open failed with 'SAT placement: mesh M0 has no grouping'); a glx_reset cleared it once. When C3 after C2 counts
# fewer local links than before it: tt-smi -glx_reset under the same gates as C2's reset (fl_peer), recount, both counts
# logged. Links still down: FL_LINKS_DOWN and a "C3 FAIL mesh placement failed: ETH links down" naming each chip pair.
fl_c3glx() {
  # shellcheck disable=SC2034  # RK_STEP is read by rk_glx (dynamic scope)
  local a b c l=c2 why="" af RK_STEP=C3g   # RK_STEP: rk_hook's step (C3g-glx)
  FL_LINKS_DOWN=""; a=$(fl_ethl pre); b=$(fl_ethl c2)
  [ -n "$a" ] && [ -n "$b" ] && [ "$b" -lt "$a" ] || return 0
  say "C3: eth_local $b after the C2 glx_reset, $a before it ($(fl_ethdown c2)): tray re-power (tt-smi -glx_reset) and recount before C4"
  if [ -n "$FL_STOP" ]; then why="not run: $FL_STOP"
  elif rk_noglx; then why="skipped by TRAY_REPRO_NO_GLX=1"
  elif ! rk_can_glx; then why="skipped: $RK_GLX_CAN_WHY"
  elif ! fl_gate C3g; then why="not run: $FL_STOP"
  elif ! fl_peer C3g; then why="SKIPPED-GATE: $FL_PEER_WHY"
  else
    FL_RESET_T=$(date +%s); RK_BEFORE=$(rk_snap); rk_glx "$FL_OUT/c3g_glx.log"
    rk_recheck "tray re-power for ETH links (C3)" "$FL_OUT/c3g_recheck.txt"; af=$RK_AFTER
    if [ -n "$(echo "$af" | awk '$3 != "ok"')" ]; then
      echo "$af" | awk '$3 != "ok" { print $1, $2 }' | while read -r bdf ch; do
        fl_find "RESET FAIL $(fl_lk "$bdf") chip $ch: missing after the tray re-power for ETH links down (C3); $(echo "$af" | grep -c ' ok$')/$(rk_total) after it"; done
      say "C3: chips missing after the tray re-power ($FL_OUT/findings.txt); nothing was loaded"
      vd_say "NOT RUN (couldn't start: chips already dropped)"; echo "Couldn't start - Chips are already dropped"; exit 6
    fi
    [ "$RK_GLX_OK" = 1 ] || why="$(if [ -n "${RK_IPMI_DEAD:-}" ]; then echo "SKIPPED, hardware fault"; elif [ -n "${RK_IPMI_SETUP:-}" ]; then echo "SKIPPED, setup fault"; else echo FAILED; fi): $RK_GLX_WHY"
    rk_map; echo "$RK_MAP" > "$FL_OUT/map.txt"; if fl_c3 c3g; then l=c3g; else fl_c3stop c3g "tray re-power for ETH links (C3)"; return 0; fi; c=$(fl_ethl c3g)
    say "C3 ETH links: eth_local before C2 $a, after C2 $b, after glx_reset ${c:-not counted}"
  fi
  [ -n "$why" ] && say "C3: tray re-power for ETH links $why"
  c=$(fl_ethl "$l")
  if [ -n "$c" ] && [ "$c" -ge "$a" ]; then say "C3: the tray re-power brought the ETH links back (eth_local $c of $a)"; return 0; fi
  FL_LINKS_DOWN="eth_local ${c:-?} of $a after $([ $l = c3g ] && echo "C2 and glx_reset" || echo "C2 (glx_reset $why)"): $(fl_ethdown "$l")"
  fl_find "C3 FAIL mesh placement failed: ETH links down: $FL_LINKS_DOWN"
}
# C4d is gone: on a quad host's 03:00.0 (tray 1) the bus-bar stage's numeric caught the
# same fault first (926 s vs C4d at 1060 s). numeric_dram.rw_burst stays as the numeric pinpoint's rw split and G3's
# readback after bus-bar chunk 1.
fl_plan() {  # after fl_start: C8 on; no tries in any fleet mode (the opt-in C4v and C4 are gone)
  FL_UB0_ARGS=(); PLAN=(); total=0; QUIETS=""
  fl_c8_start; return 0
}
fl_rd() { local _l=; { read -r _l < "$2"; } 2> /dev/null; printf -v "$1" %s "${_l:--}"; }   # first line of $2 into $1, "-" if none (no fork)
fl_c8_start() {  # C8: per-chip temp, power, AICLK, current, vcore from sysfs (read-only, no device open) every FLEET_TELEM_S
  local main=$$ d h b t c
  # each chip's limits once, "<bdf> <power1_max uW (TDP)> <curr1_max mA (TDC)> <power1_cap uW>": the LOAD lines' power-limit
  # count and LOAD LOW's own limit (power1_cap where the driver has it); only read, never written
  for d in "$SYS"/class/tenstorrent/tenstorrent!*; do
    h=$(ls -d "$d"/device/hwmon/hwmon* 2>/dev/null | head -1)
    echo "$(basename "$(readlink -f "$d/device")") $(cat "$h/power1_max" 2>/dev/null || echo -) $(cat "$h/curr1_max" 2>/dev/null || echo -) $(cat "$h/power1_cap" 2>/dev/null || echo -)"
  done > "$FL_OUT/c8_limits.txt" 2>/dev/null
  # each chip's own thermal limit, "<bdf> <temp1_max milli-degC>" (read-only): the thermal report's threshold
  for d in "$SYS"/class/tenstorrent/tenstorrent!*; do
    h=$(ls -d "$d"/device/hwmon/hwmon* 2>/dev/null | head -1)
    echo "$(basename "$(readlink -f "$d/device")") $(cat "$h/temp1_max" 2>/dev/null || echo -)"
  done > "$FL_OUT/c8_tlimit.txt" 2>/dev/null
  local ct=$FL_OUT/chip_temp.csv; echo "utc,epoch,bdf,tray,temp_c" > "$ct"   # every mode (was --mode=full only)
  local iso=""; declare -F mg_iso_path > /dev/null && iso=$(mg_iso_path)   # chips the MCE guard cut: never read again
  # no fork per chip per round (15 per chip, also on hardware): a chip's BDF, tray and hwmon dir are resolved
  # once per tt-kmd node and again only when its device link no longer points at that BDF; the reads are builtins
  ( trap - EXIT; local -A cb ctr; local -a isl; local e u p a ci cv cm s
    while kill -0 "$main" 2>/dev/null; do
      isl=(); [ -n "$iso" ] && [ -r "$iso" ] && mapfile -t isl < "$iso" 2> /dev/null
      for d in "$SYS"/class/tenstorrent/tenstorrent!*; do
        b=${cb[$d]:-}
        if [ -z "$b" ] || ! [ "$d/device" -ef "$SYS/bus/pci/devices/$b" ]; then
          b=$(basename "$(readlink -f "$d/device")"); cb[$d]=$b; ctr[$b]=$(rk_tray "$b"); fi
        for s in "${isl[@]}"; do [ "$s" = "$b" ] && continue 2; done
        h=; for h in "$d"/device/hwmon/hwmon*; do [ -e "$h" ] && break; h=; done
        fl_rd t "$h/temp1_input"   # milli-degC, read-only
        printf -v e '%(%s)T' -1
        # each chip's temp_c by BDF and tray 1-4; a sentinel or missing read is empty
        [ -n "$ct" ] && { [[ "$t" =~ ^[0-9]{1,6}$ ]] && c=$((10#$t / 1000)).$((10#$t % 1000 / 100)) || c=""
          TZ=UTC printf -v u '%(%FT%TZ)T' "$e"; echo "$u,$e,$b,tray ${ctr[$b]},$c" >> "$ct"; }
        fl_rd p "$h/power1_input"; fl_rd a "$d/tt_aiclk"; fl_rd ci "$h/curr1_input"; fl_rd cv "$h/in0_input"; fl_rd cm "$h/curr1_max"
        echo "$e $b $t $p $a $ci $cv $cm"
      done; sleep "${FLEET_TELEM_S:-5}"
    done ) >> "$FL_OUT/c8_telemetry.txt" 2>/dev/null &
  FL_C8_PID=$!
}
fl_c8_stop() {
  [ -n "$FL_C8_PID" ] || return 0; kill "$FL_C8_PID" 2>/dev/null; wait "$FL_C8_PID" 2>/dev/null; FL_C8_PID=""
  # C8 (WARN only): per-chip max temp/power/current, min vcore of the valid samples; 0xFFFF.. sentinels are ignored
  local l
  python3 "$FL_DIR/fleet_py.py" telem "$FL_OUT/c8_telemetry.txt" "${FLEET_TEMP_WARN_C:-75}" "${FLEET_POWER_WARN_W:-320}" \
    "${FLEET_CURR_WARN_A:-400}" "${FLEET_VCORE_WARN_V:-0.66}" "$FL_OUT/fault_times.tsv" 2> "$FL_OUT/c8_judge.err" | while read -r l; do fl_find "$l"; done
  [ -s "$FL_OUT/c8_judge.err" ] && say "$(tail -1 "$FL_OUT/c8_judge.err") (report only)"
  # thermal: a chip at or over its thermal limit before its first fault -> THERMAL FAIL by
  # BDF, tray and stage; samples at or after a chip's first fault, and implausible ones, are only counted
  python3 "$FL_DIR/fleet_py.py" thermal "$FL_OUT/c8_telemetry.txt" "$FL_OUT/c8_tlimit.txt" "$FL_OUT/phases.txt" "$FL_STAGES_TSV" \
    "$FL_OUT/fault_times.tsv" "${FLEET_TEMP_TRIP_C:-90}" "${FLEET_TEMP_TRIP_SRC:-records: hwmon temp1_max 90 C}" \
    2> "$FL_OUT/c8_thermal.err" | while read -r l; do fl_find "$l"; done
  [ -s "$FL_OUT/c8_thermal.err" ] && say "$(tail -1 "$FL_OUT/c8_thermal.err")"
}
# fl_watch <cap_s> <hang_s> <bdfs> <hangdir> <pid:beat:label>...: wait for jobs started with setsid (pid = pgid). A
# job whose beat says "launch N" and has not changed for hang_s (FLEET_COMPILE_S for launch 0: the first launch
# compiles), or that runs past cap_s, is hung: the op-timeout hook (on_hang.sh: snapshot while alive) runs, then the
# job's own process group is stopped. With FL_WATCH_LOSS set, chips lost from PCIe (polled every FLEET_LSPCI_POLL_S)
# end every job. Sets FL_RC[label] (exit code, or "hang"/"loss") per job.
declare -A FL_RC FL_JBDF   # FL_JBDF[label]: the chips a job runs on (the hook's snapshot list)
FL_HELD=()   # --hold: hung jobs fl_watch left running for fl_capture's hold (their pids = pgids)
fl_unhold() { local p; for p in "${FL_HELD[@]}"; do fl_kill "$p"; done; FL_HELD=(); }   # stop them (this run's own only)
fl_watch() {
  local cap=$1 hang=$2 bdfs=$3 hd=$4 t0 j pid beat lab live age lim k=0 g hc det hits; local -a stop=(); shift 4; t0=$(date +%s)
  # a kmd burst or AER storm from before the run's first watched load: one WARN line
  # per chip, pre-existing, not from this run; it never sets the verdict, stops nothing and cuts no chip (mg_watch then
  # counts only from the run's first load start, FL_MG_T0: a burst between stages is still caught)
  if [ -z "${FL_MG_PRE:-}" ] && [ "${MG_ARMED:-0}" = 1 ] && declare -F mg_preload > /dev/null; then FL_MG_PRE=1; FL_MG_T0=$t0
    hits=$(mg_preload "$t0"); [ -n "$hits" ] && while read -r g; do
      fl_find "MCE GUARD WARN $(mg_tag "${g%% *}"): ${g#* } before the run's load started: pre-existing, not from this run (verdict not set)"; done <<< "$hits"
  fi
  while :; do
    live=0; k=$((k + 1))
    for j in "$@"; do
      pid=${j%%:*}; beat=${j#*:}; lab=${beat#*:}; beat=${beat%%:*}
      [ -n "${FL_RC[$lab]:-}" ] && continue
      if ! kill -0 "$pid" 2>/dev/null; then wait "$pid"; FL_RC[$lab]=$?
        # a job that ended on its own hang (HANG file): the MCE guard removes a chip it left hung for MMIO
        # timed from the HANG file's first line (on_hang.sh writes its detection time there first)
        [ -e "$hd/$lab/HANG" ] && { FL_RC[$lab]=hang; declare -F mg_hang > /dev/null && MG_NOCAP=1 mg_hang "$lab" "$(date -d "$(head -1 "$hd/$lab/HANG")" +%s 2> /dev/null || date +%s)" "" ${FL_JBDF[$lab]:-$bdfs}; }; continue; fi
      live=1; lim=$hang; age=$(( $(date +%s) - $(stat -c %Y "$beat" 2>/dev/null || echo "$t0") ))
      # a beat ("open", "launch N", "done N", "close") counts only while in a launch; a row file counts once it
      # exists; before that, and in launch 0 (it compiles), the limit is FL_LAUNCH0_S: for pairs FLEET_PAIR_COMPILE_S (60 s), and every
      # other caller keeps the 300 s compile limit (FLEET_COMPILE_S). The fleet/*.py beats read
      # "launch N <epoch>" ('^launch 0$' never matched them)
      if [ ! -e "$beat" ] || grep -qE '^launch 0( |$)' "$beat"; then lim=${FL_LAUNCH0_S:-${FLEET_COMPILE_S:-300}}
      elif grep -qE '^(open|done|close)' "$beat"; then lim=$cap; fi
      # a hang (its own limits): the capture (on_hang.sh, FLEET_HANG_CAP_S 180) ends by FL_HANG_END (C5 under the run
      # budget), and under 20 s left it is skipped (HANG says why). FL_WATCH_END (epoch, C5's line): a job
      # still running there under its limits is stopped as not run, FL_RC budget, never hung
      if [ "$age" -ge "$lim" ] || [ $(( $(date +%s) - t0 )) -ge "$cap" ]; then
        det=$(date +%s)   # the first detection: both guard calls below are timed from here
        declare -F th_stop > /dev/null && th_stop now   # the thermal sampler never reads a hung chip
        # the MCE guard: a chip of this job already hung for MMIO (config ffffffff, ARC timeout): stop the
        # job first, capped capture, PCI remove (never held); else the capture while alive as before, and a chip that
        # went dead during it is removed then
        if declare -F mg_hang > /dev/null && mg_hang "$lab" "$det" "fl_kill $pid" ${FL_JBDF[$lab]:-$bdfs}; then FL_RC[$lab]=hang; continue; fi
        hc=${FLEET_HANG_CAP_S:-180}   # a hang's capture always gets its whole time
        FLEET_HANG_CAP_S=$hc FLEET_HANG_DIR=$hd/$lab FLEET_HANG_BDFS="${FL_JBDF[$lab]:-$bdfs}" bash "${FLEET_ON_HANG:-$FL_DIR/on_hang.sh}" > /dev/null 2>&1
        FL_RC[$lab]=hang
        # --hold (lib/hold.sh): a stage that captures its hang (FL_HOLD_OK) keeps the job alive for fl_capture's hold,
        # unless a chip is already off PCIe (a drop is never held)
        if [ "${FL_HOLD_OK:-0}" = 1 ] && [ "${HOLD_ACTIVE:-0}" = 1 ] && [ -z "${HOLD_USED:-}" ] && [ -z "$(rk_snap | awk '$3 != "ok"')" ]; then
          FL_HELD+=("$pid")
        else fl_kill "$pid"; declare -F mg_hang > /dev/null && MG_NOCAP=1 mg_hang "$lab" "$det" "" ${FL_JBDF[$lab]:-$bdfs}; fi
      fi
    done
    [ ${#stop[@]} -gt 0 ] && { fl_kill_all "${stop[@]}"; stop=(); }   # together: one 3 s TERM wait, not one per job
    [ $live = 0 ] && return 0
    # FL_WATCH_PG (a power.txt): a vr_pg fall read by the BMC poll ends the jobs too (it comes 2-7 s before the PCIe loss);
    # FL_WATCH_PG_T0: only falls from that epoch on (the bus-bar stage's parts share one power.txt)
    if [ -n "${FL_WATCH_LOSS:-}" ] && { { [ $((k % ${FLEET_LSPCI_POLL_S:-10})) = 0 ] && [ -n "$(rk_snap | awk '$3 != "ok"')" ]; } ||
         { [ -n "${FL_WATCH_PG:-}" ] && [ $((k % 2)) = 0 ] && [ -n "$(fl_pg_fell "$FL_WATCH_PG" | awk -v t="${FL_WATCH_PG_T0:-0}" '$1 >= t')" ]; }; }; then
      declare -F rk_drop_now > /dev/null && rk_drop_now "${at:-$lab}"   # first, before the kills below
      declare -F th_stop > /dev/null && th_stop now   # no thermal read of a lost chip
      for j in "$@"; do pid=${j%%:*}; lab=${j#*:}; lab=${lab#*:}; [ -z "${FL_RC[$lab]:-}" ] && { FL_RC[$lab]=loss; fl_kill "$pid"; }; done
      # every lost chip still on PCI is cut at the monitor's DROP, ffff or not (left one readable);
      # none lost (a vr_pg fall with PCIe up): no cut, as mg_loss: mg_hang's empty list means every chip, all MG_LOST
      local lostb; lostb=$(rk_snap | awk '$3 == "lost" { print $1 }')
      # shellcheck disable=SC2086  # a word list of BDFs
      if [ -n "$lostb" ] && declare -F mg_hang > /dev/null; then MG_LOST=1 MG_NOCAP=1 mg_hang "loss" "$(date +%s)" "" $lostb; fi
      return 0
    fi
    # the MCE guard's monitor: a kmd 'Failed to set initial power state' burst or an AER storm on a chip's root
    # port ends this run's jobs and cuts only that chip (mg_watch prints nothing under --hold or with the guard unarmed).
    # Under our load it is a fault: one FAIL line by BDF and tray 1-4, as the summary counts it
    if [ $((k % ${FLEET_LSPCI_POLL_S:-10})) = 0 ] && declare -F mg_watch > /dev/null && hits=$(MG_LOAD_T0=${FL_MG_T0:-} mg_watch $bdfs) && [ -n "$hits" ]; then
      declare -F th_stop > /dev/null && th_stop now
      det=$(date +%s); while read -r g; do fl_find "FAIL MCE GUARD $(mg_tag "${g%% *}"): ${g#* }"; done <<< "$hits"
      for j in "$@"; do pid=${j%%:*}; lab=${j#*:}; lab=${lab#*:}; [ -z "${FL_RC[$lab]:-}" ] && { FL_RC[$lab]=hang; stop+=("$pid"); }; done
      [ ${#stop[@]} -gt 0 ] && { fl_kill_all "${stop[@]}"; stop=(); }
      # shellcheck disable=SC2046  # a word list of BDFs
      MG_LOST=1 MG_WHAT="$(cut -d' ' -f2- <<< "$hits" | sort -u | paste -sd';')" mg_hang "monitor" "$det" "" $(cut -d' ' -f1 <<< "$hits")
      return 0
    fi
    # the broker yield: a foreign queued broker job ends this run's own jobs within 1 s of
    # the watcher's read (lib/broker_yield.sh), as not run (FL_RC budget), never hung or lost
    # (also when the yield came before this watch started, until C9)
    [ -z "${FL_YIELD:-}" ] && by_on && by_check
    if [ -n "${FL_YIELD:-}" ] && [ -z "${FL_C9_ON:-}" ]; then
      for j in "$@"; do pid=${j%%:*}; lab=${j#*:}; lab=${lab#*:}; [ -z "${FL_RC[$lab]:-}" ] && { FL_RC[$lab]=budget; stop+=("$pid"); }; done
      [ ${#stop[@]} -gt 0 ] && { fl_kill_all "${stop[@]}"; stop=(); }
      return 0
    fi
    sleep 1
  done
}
fl_kill_all() {  # fl_kill for each pid given (each a job this run started with setsid), one TERM wait for all
  local p g; local -a gs=()
  for p in "$@"; do g=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' '); [ "$g" = "$p" ] && gs+=("-$p") || gs+=("$p"); done
  for p in "${gs[@]}"; do kill -TERM -- "$p" 2>/dev/null; done; sleep 3
  for p in "${gs[@]}"; do kill -KILL -- "$p" 2>/dev/null; done
  for p in "$@"; do wait "$p" 2>/dev/null; done; return 0
}
fl_kill() {  # $1 the pid of a job this run started with setsid (its own process group): TERM, then KILL
  local g; g=$(ps -o pgid= -p "$1" 2>/dev/null | tr -d ' '); [ "$g" = "$1" ] || g=""
  if [ -n "$g" ]; then kill -TERM -- "-$g" 2>/dev/null; sleep 3; kill -KILL -- "-$g" 2>/dev/null
  else kill -TERM "$1" 2>/dev/null; sleep 3; kill -KILL "$1" 2>/dev/null; fi
  wait "$1" 2>/dev/null; return 0
}
fl_env() {  # the env of a device job: op timeout $1 s, the op-timeout hook (on_hang.sh) into $2
  echo "TT_METAL_OPERATION_TIMEOUT_SECONDS=$1 TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE=$FL_DIR/on_hang.sh FLEET_HANG_DIR=$2${HOLD_INSPECTOR:+ TT_METAL_INSPECTOR_LOG_PATH=$HOLD_INSPECTOR}"
}
fl_lost() {  # after a load phase: chips lost from PCIe -> seen/chips (run.sh's), a loss line; 0 if none
  local g; g=$(rk_snap | awk '$3 != "ok" { print $2 }' | tr '\n' ' ')
  [ -z "${g// /}" ] && return 1
  [ -z "${FL_C9_ON:-}" ] && declare -F rs_c9_loss > /dev/null && rs_c9_loss "$1"   # before C9 only: its share fits a recovery round (FL_C9_ON, not FL_RESET_T, which C2 sets)
  declare -F rk_drop_now > /dev/null && rk_drop_now "$1"
  FL_CAP_FAULT=0 fl_capture "$(echo "$1" | tr -c 'A-Za-z0-9\n' _)" drop $(rk_snap | awk '$3 != "ok" { print $1 }')
  # the MCE guard's cut before any read of a lost chip (heavy exalens/tt-triage reads wait for its rescan)
  declare -F mg_loss > /dev/null && { mg_loss "$1" $(rk_snap | awk '$3 != "ok" { print $1 }') || true; }
  seen=1; chips="$chips $g"; at="$1"; say "$1: chips lost: $(echo $g | rk_list) (not answering: $(lost_trays))"; return 0
}
# Per-stage AER delta and the post-boot settle recount (rows R77, R80, R81 of notes/run-sh-coverage.md).
# Reads only: sysfs AER counters and the chip recount; no reset of their own (a loss goes through fl_lost and run.sh's C9).
fl_uptime() { local u; read -r u _ 2> /dev/null < "${FLEET_PROC:-/proc}/uptime" && echo "${u%%.*}"; }   # 2> first: a missing file says nothing
: "${FL_BOOT_UP0=$(fl_uptime)}"   # host uptime (s) at run start; run.sh sets it before its start reset
FL_BOOT_DONE=""; FL_AER_PRE_DONE=""
fl_aer_stage_before() {  # <phase>: each answering chip's and its root port's AER counters before the stage
  local d=$FL_OUT/aer_stage c l; mkdir -p "$d"
  c=$(rk_snap | awk '$3 == "ok" { print $1 }'); [ -n "$c" ] || return 0
  # shellcheck disable=SC2086  # $c: BDF words
  python3 "$FL_DIR/fleet_py.py" aer "${SYS:-${TRAY_REPRO_SYS:-/sys}}" $c > "$d/$1_pre.tsv" 2> /dev/null || { rm -f "$d/$1_pre.tsv"; return 0; }
  date +%s > "$d/$1_t0"
  [ -z "$FL_AER_PRE_DONE" ] || return 0
  FL_AER_PRE_DONE=1   # uncorrectable counts already set before the run: at most one WARN, kept out of the verdict
  # baseline: the run-start read (lk_check pre, before C2), so a count set between run start and the first stage is judged
  # by the stage deltas and the whole-run delta, never excused as pre-existing; else the first stage's read
  local b="${FL_OUT%/fleet}/link/aer_pre.tsv"; [ -s "$b" ] || b="$d/$1_pre.tsv"
  while IFS= read -r l; do if [ -n "$l" ]; then fl_find "$l"; fi; done < <(python3 "$FL_DIR/fleet_py.py" aerpre "$b" "${H:-?}" 2> /dev/null)
}
fl_aer_stage_after() {  # <phase>: read again; an uncorrectable rise is an AER FAIL, a correctable one at FLEET_AER_RATE_WARN/h
  # and FLEET_AER_STAGE_MIN events an AER WARN (the whole-run delta's grading, lk_check, with a higher event floor: a short
  # stage's hours floor makes 2 events read as 120/h), each naming BDF, tray and stage. Reported lines go
  # to aer_stage/reported, so lk_check post shows the same chip's whole-run line as INFO, not a second finding.
  local d=$FL_OUT/aer_stage t0 h l; [ -s "$d/$1_pre.tsv" ] || return 0
  # shellcheck disable=SC2046  # the endpoints read before the stage
  python3 "$FL_DIR/fleet_py.py" aer "${SYS:-${TRAY_REPRO_SYS:-/sys}}" $(awk -F'\t' 'NR > 1 && $2 == "ep" { print $1 }' "$d/$1_pre.tsv") > "$d/$1_post.tsv" 2> /dev/null || return 0
  read -r t0 < "$d/$1_t0" || t0=$(date +%s); h=$(awk -v s=$(( $(date +%s) - t0 )) 'BEGIN { printf "%.4f", s / 3600 }')
  # FLEET_AER_STAGE_FAIL (20, 0 = off): one root port +N correctable with every other +2 or less is an AER FAIL naming the
  # chip; one the AER poll already reported live this stage (same chip and port) is not written twice
  while IFS= read -r l; do if [ -n "$l" ]; then
      case $l in "AER FAIL "*": correctable +"*) grep -qxF "${l%%, host *}" "$d/live" 2> /dev/null && continue ;; esac
      fl_find "$l"; echo "${l%%, host *}" >> "$d/reported"; fi
  done < <(python3 "$FL_DIR/fleet_py.py" aerdelta "$d/$1_pre.tsv" "$d/$1_post.tsv" \
    "${FLEET_AER_RATE_WARN:-6}" "$h" "${FLEET_AER_STAGE_MIN:-10}" "${H:-?}" "during $(fl_pn "$1")" "${FLEET_AER_STAGE_FAIL:-20}" 2> /dev/null)
  rm -f "$d/live"   # this stage's live lines, judged
}
fl_boot_recount() {  # at each stage boundary: a run that started under 15 min after boot recounts its chips by BDF and tray
  # once, at the first boundary at or after 15 min of uptime (R77: 43:00.0 off the bus 9 min after boot). It never waits.
  # A loss is a POST-BOOT DROP per chip, then fl_lost: run.sh's own loss path and C9 glx_reset recover it.
  local u b n lost; [ -z "$FL_BOOT_DONE" ] && [ -n "$FL_BOOT_UP0" ] && [ "$FL_BOOT_UP0" -lt 900 ] || return 0
  u=$(fl_uptime); [ -n "$u" ] && [ "$u" -ge 900 ] || return 0
  FL_BOOT_DONE=1; n=$(echo "${RK_MAP:-}" | grep -c . || true)
  if [ "${seen:-0}" = 1 ]; then say "post-boot settle recount: chips already lost in ${at:-a load stage}, recount left to their loss path"; return 0; fi
  lost=$(rk_snap | awk '$3 != "ok" { print $1 }')
  if [ -z "$lost" ]; then say "post-boot settle recount: $n/$n chips answer at uptime $((u / 60)) min (the run started $((FL_BOOT_UP0 / 60)) min after boot)"; return 0; fi
  for b in $lost; do fl_find "POST-BOOT DROP $b (tray $(rk_tray "$b")), host ${H:-?}: off the bus at the post-boot settle recount, uptime $((u / 60)) min (the run started $((FL_BOOT_UP0 / 60)) min after boot)"; done
  fl_lost "post-boot settle recount"
}
fl_boot_unreached() {  # after the last stage: say so when the recount never came due (no wait is added for it)
  [ -z "$FL_BOOT_DONE" ] && [ -n "$FL_BOOT_UP0" ] && [ "$FL_BOOT_UP0" -lt 900 ] || return 0
  local u; u=$(fl_uptime); say "post-boot settle recount not reached: uptime $(( ${u:-0} / 60 )) min at the end of the load stages (run started $((FL_BOOT_UP0 / 60)) min after boot; no wait added)"
}
fl_pair() {  # $1 label, $2 devA, $3 devB, $4 dir, $5 their BDFs: start one 2-chip all-gather in its own process group; adds pid:beat:label to FL_JOBS
  # opened by the BDFs' UMD indices, never devA/devB (/dev 16-31 as indices opened the other tray's twins,
  # and the 4 tray 1/2 <-> 3/4 links came out UNTESTED); fl_c5_cls checks the open against them (rk_opencheck)
  local u; mkdir -p "$4"; unset 'FL_RC[$1]'   # the sweep reuses C5's labels: a stale rc would end its watch at once
  # shellcheck disable=SC2086  # $5: two BDF words
  umd_snap "$4/$1.log"   # the open-time BDF -> /dev map rk_opencheck reads
  if u=$(rk_umd $5); then
    setsid "${AS[@]}" env TT_VISIBLE_DEVICES="$u" $(fl_env "${FLEET_PAIR_HANG_S:-30}" "$4/$1") \
      "$PY" "$FL_DIR/fabric_repro.py" --mode ag --n 2 --seconds "${FLEET_PAIR_S:-10}" --rows "$4/$1.rows" --beat "$4/$1.beat" \
      --label "$1" > "$4/$1.log" 2>&1 < /dev/null {PG_OUT}>&- &
  else   # a chip with no tt-kmd node has no UMD index: not opened, the pair fails (rc 1)
    setsid bash -c 'echo "run.sh: a chip of $0 has no tt-kmd node: no UMD index, not opened"; exit 1' "$5" > "$4/$1.log" 2>&1 < /dev/null {PG_OUT}>&- &
  fi
  pg_track $!; FL_JOBS+=("$!:$4/$1.beat:$1"); FL_JBDF[$1]=${5:-}
}
fl_nproc_raise() {  # once a run, before C5: a soft process limit (ulimit -u) below its hard limit is raised to it for this
  # run only (Slurm copies the submit host's soft limit into the job: 512 under a 2318132 hard limit on a cluster node)
  local s h; [ -n "${FL_NPROC_RAISED:-}" ] && return 0; FL_NPROC_RAISED=1
  [ "${FLEET_NPROC_RAISE:-1}" = 1 ] && [ -z "${FLEET_NPROC_LIMIT:-}" ] || return 0
  s=$(ulimit -S -u); h=$(ulimit -H -u); [ "$s" != unlimited ] && [ "$s" != "$h" ] || return 0
  [ "$h" = unlimited ] || [ "$s" -lt "$h" ] || return 0
  ulimit -S -u "$h" 2>/dev/null && say "C5: raised this run's soft process limit (ulimit -u) from $s to its hard limit $h"
  return 0
}
fl_pair_cap() {  # FL_PAIR_CAP: how many C5 pairs may run at once under the process/thread limits (0: no bound), FL_CAP_WHY the limits.
  # Room = the tightest of ulimit -u minus this user's tasks and each cgroup level's pids.max minus pids.current, less a
  # reserve; each pair is counted at FLEET_PAIR_THREADS (default nproc + 64: OpenBLAS starts one thread per CPU, plus
  # torch, tt-metal and MPI threads). Test overrides: FLEET_PAIR_MAX, FLEET_NPROC_LIMIT, FLEET_NPROC_USED, FL_PROC_CGROUP
  local lim used u per room="" why="" c p f m cur base
  FL_PAIR_CAP=0; FL_CAP_WHY=""
  [ -n "${FLEET_PAIR_MAX:-}" ] && { FL_PAIR_CAP=$FLEET_PAIR_MAX; FL_CAP_WHY="FLEET_PAIR_MAX $FLEET_PAIR_MAX"; return 0; }
  per=${FLEET_PAIR_THREADS:-$(( $(nproc 2>/dev/null || echo 64) + 64 ))}
  lim=${FLEET_NPROC_LIMIT:-$(ulimit -u)}
  if [ "$lim" != unlimited ]; then
    u=$(id -u); [ -n "${AS[*]:-}" ] && [ -n "${SUDO_USER:-}" ] && u=$(id -u "$SUDO_USER" 2>/dev/null || echo "$u")
    used=${FLEET_NPROC_USED:-$(ps -L -U "$u" -o lwp= 2>/dev/null | wc -l)}
    room=$(( lim - used )); why="ulimit -u $lim with $used in use"
  fi
  while IFS=: read -r _ c p; do   # v2: "0::/path"; v1: "N:pids:/path"
    case $c in "") base=${SYS:-/sys}/fs/cgroup ;; pids) base=${SYS:-/sys}/fs/cgroup/pids ;; *) continue ;; esac
    while :; do
      f=$base${p%/}
      if [ -r "$f/pids.max" ] && m=$(cat "$f/pids.max" 2>/dev/null) && [ "$m" != max ] && [ -n "$m" ]; then
        cur=$(cat "$f/pids.current" 2>/dev/null || echo 0)
        if [ -z "$room" ] || [ $(( m - cur )) -lt "$room" ]; then room=$(( m - cur )); why="cgroup pids.max $m with $cur in use"; fi
      fi
      { [ -z "$p" ] || [ "$p" = / ]; } && break; p=${p%/*}
    done
  done < "${FL_PROC_CGROUP:-/proc/self/cgroup}"
  [ -n "$room" ] || return 0
  FL_PAIR_CAP=$(( (room - ${FLEET_PAIR_RESERVE:-32}) / per )); [ "$FL_PAIR_CAP" -ge 1 ] || FL_PAIR_CAP=1
  FL_CAP_WHY="$why, about $per threads per pair"
}
fl_c5_rsrc() {  # $1 a pair log: 0 if it shows a process/thread shortage (EAGAIN from fork or pthread_create)
  grep -aqE 'Resource temporarily unavailable|pthread_create failed|RLIMIT_NPROC|pmix_progress_thread_start failed|fork: retry|EAGAIN' "$1" 2>/dev/null
}
fl_c5_go() {  # in fl_c5: run pair indices $2.. in batches of at most $1 (0: all at once); their jobs go to rj, the pairs the
  # budget left unrun (only between batches) to left
  local cap=$1 n i; shift
  while [ $# -gt 0 ]; do
    n=$#; [ "$cap" -gt 0 ] && [ "$cap" -lt "$n" ] && n=$cap
    FL_JOBS=(); bdfs=""
    for i in "${@:1:n}"; do
      IFS=$'\t' read -r idx r a b ba bb da db ca cb <<< "$(awk -F'\t' -v i="$i" '$1 == i' "$d/pairs.tsv")"
      fl_pair "p$idx" "$da" "$db" "$d" "$ba $bb"; bdfs="$bdfs $ba $bb"
    done
    shift "$n"
    hb_start "$nm round $ro (${#FL_JOBS[@]} pairs)"
    FL_HOLD_OK=1 FL_LAUNCH0_S=${FLEET_PAIR_COMPILE_S:-60} fl_watch $(( ${FLEET_COMPILE_S:-300} + 60 )) "${FLEET_PAIR_HANG_S:-30}" "$bdfs" "$d" "${FL_JOBS[@]}"; hb_stop
    rj+=("${FL_JOBS[@]}")
  done
}
fl_c5_inc() {  # in fl_c5: pair label $1 did not run, $2 why (test infrastructure): INCOMPLETE, never a chip/link FAIL or a pass
  IFS=$'\t' read -r idx r a b ba bb da db ca cb <<< "$(awk -F'\t' -v i="${1#p}" '$1 == i' "$d/pairs.tsv")"
  echo "$1 INCOMPLETE" >> "$d/result.txt"; FL_C5_INC+=("$1"); FL_C5_GAP="${FL_C5_GAP:+$FL_C5_GAP; }$1 not run (test-infra)"
  FL_SKIPPED+=("$nm ${ba#0000:}<->${bb#0000:} e$ca (test-infra: $2)")
  fl_find "${px}LINK WARN $(fl_lk "$ba") e$ca <-> $(fl_lk "$bb") e$cb: INCOMPLETE (test-infra): pair not run, $2 ($d/$1.log)"
}
fl_c5_cls() {  # in fl_c5: classify pair label $1; $2 1 on the retry; a resource failure goes to retry (first try) or INCOMPLETE
  local lab=$1 rc=${FL_RC[$1]}
  IFS=$'\t' read -r idx r a b ba bb da db ca cb <<< "$(awk -F'\t' -v i="${lab#p}" '$1 == i' "$d/pairs.tsv")"
  rk_opencheck "$d/$lab.log" "$nm pair $lab" "$ba" "$bb" || :   # the open must be these two chips
  case $rc in
    0) echo "$lab PASS" >> "$d/result.txt" ;;
    hang) if [ -n "$px" ]; then FL_SWEPT+=("$lab"); else FL_SUSPECT+=("$lab"); fi; printf '%s\thang\n' "$lab" >> "$d/failed.tsv"
          bad="$bad $a $b"; echo "$lab HANG" >> "$d/result.txt"; fl_capture "${sn}_$lab" hang "$ba" "$bb"
          say "$nm SUSPECT $(fl_lk "$ba") e$ca <-> $(fl_lk "$bb") e$cb: hang (snapshot $d/$lab)" ;;
    *) if grep -aq 'only 1 devices are available in the system mesh' "$d/$lab.log" 2>/dev/null; then
         # tt-metal opens these 2 chips as a 1x1 system mesh (seen on the e4,5 tray 1<->3 / tray 2<->4 links):
         # the link is not tested here, which says nothing about the hardware
         echo "$lab UNTESTED" >> "$d/result.txt"; FL_SKIPPED+=("$nm ${ba#0000:}<->${bb#0000:} e$ca (the 2 chips opened share no Ethernet link: a 1-device mesh)")
         say "$nm not tested $(fl_lk "$ba") e$ca <-> $(fl_lk "$bb") e$cb: the 2 chips opened share no Ethernet link, a 1-device mesh ($d/$lab.log)"
         return 0
       fi ;;&
    4) echo "$lab FAIL" >> "$d/result.txt"; printf '%s\tdata mismatch\n' "$lab" >> "$d/failed.tsv"; local ld ua ub; ua=$(UMD_MAP="$d/$lab.log.map" rk_umd "$ba") || ua="?"; ub=$(UMD_MAP="$d/$lab.log.map" rk_umd "$bb") || ub="?"
       ld=$(python3 "$FL_DIR/link_pattern.py" summary "$d/$lab.rows" "$ua" "$(fl_lk "$ba")" "$ub" "$(fl_lk "$bb")" 2>/dev/null | cut -c1-600)   # rows name chips by UMD index
       fl_find "${px}LINK FAIL $(fl_lk "$ba") e$ca <-> $(fl_lk "$bb") e$cb: link-data mismatch${ld:+: $ld} ($d/$lab.log)" ;;   # sender -> receiver, words, bits, 32 B pos
    loss) FL_C5_GAP="${FL_C5_GAP:+$FL_C5_GAP; }$lab rc $rc"; printf '%s\trc %s\n' "$lab" "$rc" >> "$d/failed.tsv"
          fl_find "${px}LINK WARN $(fl_lk "$ba") e$ca <-> $(fl_lk "$bb") e$cb: pair run failed, rc $rc ($d/$lab.log)" ;;
    *) # out of processes/threads: its log shows EAGAIN, or it aborted (134/139) beside one that did in the
       # same try (rsb: a log of that try's batches shows EAGAIN; a lone 134/139 on the retry is a pair run failure)
       if fl_c5_rsrc "$d/$lab.log" || { [ "$rsb" = 1 ] && { [ "$rc" = 134 ] || [ "$rc" = 139 ]; }; }; then
         if [ "${2:-0}" = 0 ]; then retry+=("${lab#p}")
         else fl_c5_inc "$lab" "out of processes/threads on its retry at $rat at once, rc $rc (${FL_CAP_WHY:-no limit read})"; fi
       else
         FL_C5_GAP="${FL_C5_GAP:+$FL_C5_GAP; }$lab rc $rc"; printf '%s\trc %s\n' "$lab" "$rc" >> "$d/failed.tsv"
         fl_find "${px}LINK WARN $(fl_lk "$ba") e$ca <-> $(fl_lk "$bb") e$cb: pair run failed, rc $rc ($d/$lab.log)"
       fi ;;
  esac
}
fl_c5_one() {  # in fl_c5: every failing pair shares one chip -> FAIL naming that chip (its tray only; the far
  # ends' trays keep their grade), its links and tt-metal's Device N errors on it; fault.txt names it; C5 sets FL_PIN (sweep)
  local one ch
  one=$(python3 "$FL_DIR/fleet_py.py" c5one "$d/pairs.tsv" "$d" "$d/failed.tsv" "$FL_OUT/map.txt" "$nm" 2>/dev/null) || return 0
  ch=${one#LINK FAIL one chip }; ch=${ch%%: in all *}
  # every failing pair only a 'pair run failed, rc N' (a LINK WARN in fl_c5_cls: no hang, no data mismatch)
  # and tt-metal names no error on this chip: the line is a WARN too, labelled so (label only: FL_PIN and fault.txt as before)
  if [[ $one != *"tt-metal errors on this chip"* ]] && ! awk -F'\t' '$2 !~ /^rc [0-9]+$/ { f = 1 } END { exit !f }' "$d/failed.tsv"; then
    one="LINK WARN one chip $ch (pair runs failed only, rc N: no hang, data mismatch or tt-metal error on this chip): in all ${one#*: in all }"; fi
  fl_find "$px$one"; declare -F fault_note > /dev/null && fault_note "$px$one"
  [ -n "$px" ] || [ -n "$FL_PIN" ] || FL_PIN="C5 link failures all on one chip, $ch"
}
fl_c5_hard() { hard=""; deadline=$(( $(date +%s) + 86400 )); }  # in fl_c5: no round is stopped for time; each pair ends on its own op timeout (FLEET_PAIR_HANG_S) (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
fl_c5() {  # pairwise link sweep; a hang names the pair SUSPECT (rerun after the end reset) and ends the load phases.
  # fl_sweep's on-fault run: FL_SW=1 fl_c5 sweep <discovery label> <budget s>: out in fleet/sweep, a hang goes to
  # FL_SWEPT (not SUSPECT), lines name trays and start PINPOINT; FL_C5_GAP says why links went untested (fl_sweep)
  local d=$FL_OUT/${1:-c5} r=${2:-} rounds line idx ro a b ba bb da db ca cb t0 lab bdfs bad="" deadline hard l over="" nm=C5 px="" rat rsb sn=${1:-c5}
  local -a todo=() rj=() left=() retry=()
  [ "${FL_SW:-0}" = 1 ] && { nm="PINPOINT sweep"; px="PINPOINT "; }
  mkdir -p "$d"; : > "$d/failed.tsv"; t0=$(date +%s); FL_C5_GAP=; fl_nproc_raise
  # under run.sh's run budget C5 ends by hard, the C9 share and RS_TAIL_S before the end: no
  # batch starts at or after it, a pair still running there under its own hang limits is stopped and reported not run (run
  # budget; fl_watch FL_WATCH_END), and a real hang's capture ends by hard + FLEET_C5_CAPX_S (60): C9 then still has RS_C9_S - 60 s and the RESULT RS_TAIL_S
  # the per-link sweep (FL_SW=1) runs after C9 or with C9 refused: its line keeps only the RESULT and a hang's capture
  fl_c5_hard "$nm" "${3:-300}"
  [ -n "$r" ] || { [ -s "$FL_OUT/disc_c2.json" ] && r=c2 || r=pre; }
  python3 "$FL_DIR/fleet_py.py" pairs "$FL_OUT/disc_$r.json" "$FL_OUT/map.txt" > "$d/pairs.tsv" 2> "$d/pairs.err" \
    || { FL_C5_GAP="no pair list (UMD discovery $r failed)"; fl_find "${px}LINKS WARN $nm not run: $FL_C5_GAP"; return 0; }
  rounds=$(awk -F'\t' 'NR > 1 { print $2 }' "$d/pairs.tsv" | sort -un)
  say "$nm pair sweep: $(( $(grep -c . "$d/pairs.tsv") - 1 )) local pairs in $(echo $rounds | wc -w) rounds (${FLEET_PAIR_S:-10} s each, op timeout ${FLEET_PAIR_HANG_S:-30} s)"
  for ro in $rounds; do
    fl_c5_hard "$nm" "${3:-300}"   # an extension since (a fault found in this stage, a hang's capture) moves hard
    [ -n "$over" ] || [ "$(date +%s)" -lt $(( t0 + ${3:-300} )) ] \
      || { over=1; say "$nm past its planned ${3:-300} s: rounds from $ro still run, each pair ended only by its own op timeout"; }
    todo=(); rj=(); left=(); retry=(); rsb=0
    while IFS=$'\t' read -r idx r a b ba bb da db ca cb; do
      [ "$r" = "$ro" ] || continue
      case " $bad " in *" $a "*|*" $b "*) continue ;; esac
      todo+=("$idx")
    done < <(tail -n +2 "$d/pairs.tsv")
    [ ${#todo[@]} = 0 ] && continue
    # at most as many pairs at once as the process/thread limits allow; unbounded hosts run them all at once
    fl_pair_cap; rat=${#todo[@]}
    if [ "$FL_PAIR_CAP" -gt 0 ] && [ "$FL_PAIR_CAP" -lt "$rat" ]; then rat=$FL_PAIR_CAP
      say "$nm round $ro: ${#todo[@]} pairs, at most $rat at once ($FL_CAP_WHY)"; fi
    fl_c5_go "$FL_PAIR_CAP" "${todo[@]}"
    for line in "${rj[@]}"; do fl_c5_rsrc "$d/${line##*:}.log" && rsb=1; done
    for line in "${rj[@]}"; do fl_c5_cls "${line##*:}" 0; done
    if [ ${#retry[@]} -gt 0 ]; then   # once more, at half the concurrency (or less, if the limits now say so)
      rat=$(( rat / 2 )); [ "$rat" -ge 1 ] || rat=1; fl_pair_cap
      [ "$FL_PAIR_CAP" -gt 0 ] && [ "$FL_PAIR_CAP" -lt "$rat" ] && rat=$FL_PAIR_CAP
      say "$nm round $ro: ${#retry[@]} pair(s) ran out of processes/threads; retrying them at most $rat at once (${FL_CAP_WHY:-no limit read})"
      # the first try's beat and rows go aside with its log: fl_watch judges the retry by its beat, and a stale one older
      # than FLEET_PAIR_HANG_S would call it hung at once
      for idx in "${retry[@]}"; do
        for x in log beat rows; do mv -f "$d/p$idx.$x" "$d/p$idx.try1.$x" 2>/dev/null; done; rm -f "$d/p$idx.beat.tmp"
        unset 'FL_RC[p'"$idx"']'
      done
      todo=("${retry[@]}"); retry=(); rj=(); rsb=0
      fl_c5_go "$rat" "${todo[@]}"
      for line in "${rj[@]}"; do fl_c5_rsrc "$d/${line##*:}.log" && rsb=1; done
      for line in "${rj[@]}"; do fl_c5_cls "${line##*:}" 1; done
    fi
    if fl_lost "$nm round $ro"; then   # in the sweep (after the reset) a lost chip is a finding of its own, so it sets the exit code
      say "$nm one-chip check not run: the sweep stopped in round $ro (chips lost), $(grep -c . "$d/failed.tsv") failing pair(s) so far ($d/failed.tsv)"
      [ -n "$px" ] && { FL_C5_GAP="${FL_C5_GAP:+$FL_C5_GAP; }chips lost in round $ro"
        fl_find "PINPOINT CHIP LOST $(rk_snap | awk '$3 != "ok" { print $1 }' | while read -r b; do FL_SW=1 fl_lk "$b"; done | paste -sd, - | sed 's/,/, /g'): gone from PCIe in $nm round $ro, after the reset ($d)"; }
      return 0; fi
  done
  say "$nm pair sweep: $(cat "$d/result.txt" 2>/dev/null | grep -c PASS) pass, $(cat "$d/result.txt" 2>/dev/null | grep -c UNTESTED) not testable, $(cat "$d/result.txt" 2>/dev/null | grep -c INCOMPLETE) not run (test-infra), $([ -n "$px" ] && echo ${#FL_SWEPT[@]} || echo ${#FL_SUSPECT[@]}) hang ($(( $(date +%s) - t0 )) s)"
  fl_c5_one
}
# run.sh --repro link-a (bug a; kit results/t2379, t2500): the link-A fast loop. Of the two in-box
# links between 02:00.0 and 06:00.0 (tray 1), hide e7 <-> e1 (ETH FW port_status 3 on both ends, fleet/ethmask.py; chip L1
# only, the C9 glx_reset retrains and restores it), check that UMD discovery sees only e6 <-> e0, then one 2-chip
# all-gather of an odd per-worker page count (2560x4096 bf16 per chip, output in DRAM; 13-20 MiB hang, 12 MiB and below pass)
# over that link. On the affected host it hung on launch 0, 2-3 s after open, in 26 of 26 tries (about 1 min each); the clean
# control host passed 8 of 8. Only this host's own two chips and their in-box link: no reset, link check or fabric
# change reaches another host, and run.sh's gates (P0, the C2/C9 peer gate) run unchanged. The C9 end reset recovers it.
# FLEET_LINKA_A/_B (BDFs, run.sh --repro link-a <bdf> sets A), FLEET_LINKA_KEEP/_DROP (A,B channel pairs), FLEET_LINKA_HW,
# FLEET_LINKA_S (seconds of launches), FLEET_LINKA_HANG_S (beat limit), FLEET_LINKA_FIRST_S (launch 0 and open limit).
fl_bdf() { case $1 in ????:??:??.?) echo "$1" ;; ??:??.?) echo "0000:$1" ;; *) echo "$1" ;; esac; }   # 02:00.0 -> 0000:02:00.0
fl_linka_init() {
  FL_LINKA_A=$(fl_bdf "${FLEET_LINKA_A:-0000:02:00.0}"); FL_LINKA_B=$(fl_bdf "${FLEET_LINKA_B:-0000:06:00.0}")
  FL_LINKA_KEEP=${FLEET_LINKA_KEEP:-6,0}; FL_LINKA_DROP=${FLEET_LINKA_DROP:-7,1}; FL_LINKA_HW=${FLEET_LINKA_HW:-2560,4096}
  FL_LINKA_S=${FLEET_LINKA_S:-10}; FL_LINKA_HANG=${FLEET_LINKA_HANG_S:-30}; FL_LINKA_FIRST=${FLEET_LINKA_FIRST_S:-60}
  local x re='^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$'
  for x in "$FL_LINKA_A" "$FL_LINKA_B"; do [[ $x =~ $re ]] || { echo "--repro link-a takes a chip BDF such as 02:00.0 (got '$x')" >&2; exit 1; }; done
  [ "$FL_LINKA_A" != "$FL_LINKA_B" ] || { echo "--repro link-a: FLEET_LINKA_B is the same chip as A ($FL_LINKA_A)" >&2; exit 1; }
  [[ $FL_LINKA_KEEP =~ ^[0-9]+,[0-9]+$ && $FL_LINKA_DROP =~ ^[0-9]+,[0-9]+$ ]] && [ "${FL_LINKA_KEEP%,*}" != "${FL_LINKA_DROP%,*}" ] \
    && [ "${FL_LINKA_KEEP#*,}" != "${FL_LINKA_DROP#*,}" ] \
    || { echo "FLEET_LINKA_KEEP and FLEET_LINKA_DROP take two different 'chA,chB' ETH channel pairs (got '$FL_LINKA_KEEP' '$FL_LINKA_DROP')" >&2; exit 1; }
  [[ $FL_LINKA_HW =~ ^[1-9][0-9]*,[1-9][0-9]*$ ]] && [ $(( ${FL_LINKA_HW%,*} % 32 + ${FL_LINKA_HW#*,} % 32 )) = 0 ] \
    || { echo "FLEET_LINKA_HW takes H,W in multiples of 32 (got '$FL_LINKA_HW')" >&2; exit 1; }
  [[ $FL_LINKA_S =~ ^[1-9][0-9]*$ && $FL_LINKA_HANG =~ ^[1-9][0-9]*$ && $FL_LINKA_FIRST =~ ^[1-9][0-9]*$ ]] \
    || { echo "FLEET_LINKA_S, FLEET_LINKA_HANG_S and FLEET_LINKA_FIRST_S take whole seconds (got '$FL_LINKA_S' '$FL_LINKA_HANG' '$FL_LINKA_FIRST')" >&2; exit 1; }
  [ "${FL_DRY:-0}" = 1 ] || return 0
  local s=$(( 150 + 60 + FL_LINKA_FIRST + FL_LINKA_S + 30 ))   # links-up wait (at most), mask + discovery, open + launch 0, launches, close
  echo "run.sh --repro link-a dry run: phases and time budget (nothing is opened, reset or loaded)"
  printf '%-9s %-90s %6s\n' phase what min
  printf '%-9s %-90s %6s\n' P0 "gate (read-only), setup" 0.5 P1 "C0 preflight, C1 evidence snapshot, C3 ETH/QSFP port count + tt-smi ETH" 0.8 \
    C2 "start glx_reset (tt-smi -glx_reset) of this host's chips + recount + C3" 2.0 linka "link-A repro: $(fl_linka_plan)" \
    "$(awk -v s=$s 'BEGIN { printf "%.1f", s / 60 }')" C9 "end reset of this host's chips (recovers a hang, unhides the link) + recheck" 2.5
  echo "total repro-linka: $(awk -v s=$((s + 30 + 45 + 120 + 150)) 'BEGIN { printf "%.1f", s / 60 }') min at most (not in plain run.sh or --mode=full)"
  echo "only this host's own chips and in-box link: the resets (C2, C9) take no peer snapshot"
  exit 0
}
fl_linka_plan() {  # the link-A step in one phrase
  echo "hide e${FL_LINKA_DROP%,*} <-> e${FL_LINKA_DROP#*,}, all-gather ${FL_LINKA_HW/,/x} bf16 per chip over $(fl_lk "$FL_LINKA_A") e${FL_LINKA_KEEP%,*} <-> $(fl_lk "$FL_LINKA_B") e${FL_LINKA_KEEP#*,} for ${FL_LINKA_S} s, hang at ${FL_LINKA_HANG} s (launch 0: ${FL_LINKA_FIRST} s)"
}
FL_LINKA_RECOVERY="glx_reset (tt-smi -glx_reset, run.sh's C9 end reset, runs it and unhides the masked link)"
fl_linka_line() {  # <PASS|HANG> <seconds> [detail]: the one result line, chips by BDF and tray 1-4 (never a UBB number)
  local w; if [ "$1" = HANG ]; then w="time to hang $2 s${3:+ ($3)}; recovery: $FL_LINKA_RECOVERY"; else w="no hang in $2 s${3:+ ($3)}"; fi
  echo "REPRO LINK-A $1 $(fl_lk "$FL_LINKA_A") e${FL_LINKA_KEEP%,*} <-> $(fl_lk "$FL_LINKA_B") e${FL_LINKA_KEEP#*,}: $w"
}
fl_linka_hidden() {  # the link-A mask with no later reset that worked: one WARN naming the hidden link and its undo
  [ -n "${FL_LINKA_MASKED:-}" ] && [ "${RK_CHIPRESET_T:-0}" -lt "${FL_LINKA_MASK_T:-0}" ] || return 0
  local w; w=$(printf '%s; ' "${FL_RESET_SKIPPED[@]}" | sed 's/; $//')
  echo "LINK WARN $FL_LINKA_MASKED still hidden: the link-A repro wrote its port_status mask at $(date -d "@$FL_LINKA_MASK_T" +%H:%M:%S) and no reset of this host's chips worked after it (${w:-the C9 end reset did not run or failed}); undo: tt-smi -glx_reset (a reset of this host's chips)"
}
fl_linka_inc() {  # the step could not run: INCOMPLETE (fl_verdict, rc 9), never a pass
  FL_LINKA_INC=$1; FL_LINKA_LINE="REPRO LINK-A NOT RUN (INCOMPLETE) $(fl_lk "$FL_LINKA_A") e${FL_LINKA_KEEP%,*} <-> $(fl_lk "$FL_LINKA_B") e${FL_LINKA_KEEP#*,}: $1"
  FL_SKIPPED+=("link-A repro (INCOMPLETE: $1)"); say "$FL_LINKA_LINE"
}
fl_linka() {
  local d=$FL_OUT/linka a=$FL_LINKA_A b=$FL_LINKA_B ka=${FL_LINKA_KEEP%,*} kb=${FL_LINKA_KEEP#*,} ma=${FL_LINKA_DROP%,*} mb=${FL_LINKA_DROP#*,}
  local da db ua w=0 ws dz="" rc t0 el bt E=$FL_DIR/ethmask.py; mkdir -p "$d"
  local dp=${FLEET_LINKA_DISC_AT_S:-150} gu=${FLEET_LINKA_UP_S:-240}   # the disc point and give-up point (tests shorten them)
  da=$(echo "$RK_MAP" | awk -v x="$a" '$1 == x { print $2 }'); db=$(echo "$RK_MAP" | awk -v x="$b" '$1 == x { print $2 }')
  [[ $da =~ ^[0-9]+$ && $db =~ ^[0-9]+$ ]] || { fl_linka_inc "chip $(fl_lk "$a") or $(fl_lk "$b") is not on this host"; return 0; }
  # da/db: /dev numbers, for ethmask's TTDevice.create; ua: the two UMD indices for TT_VISIBLE_DEVICES
  ua=$(rk_umd "$a" "$b") || { fl_linka_inc "chip $(fl_lk "$a") or $(fl_lk "$b") has no tt-kmd node (no UMD index)"; return 0; }
  umd_snap "$d/linka.log"   # the map ua was taken from: rk_opencheck reads it
  # 1. both links up after the C2 glx_reset (they retrain: up to 150 s; then one discovery open, which trains them, and 90 s more).
  # Wall clock from ws: a UMD call that stalls to its timeout counts, so the wait ends near 240 s, not ~50 min.
  ws=$(date +%s)
  until "${AS[@]}" timeout $(( gu - w < 60 ? (gu - w > 5 ? gu - w : 5) : 60 )) "$PY" "$E" up "$da:$ka,$ma" "$db:$kb,$mb" > "$d/up.json" 2>> "$d/up.err"; do
    w=$(( $(date +%s) - ws ))
    if [ $w -ge "$gu" ]; then fl_linka_inc "the two links were not up after $w s ($d/up.json)"; return 0
    elif [ $w -ge "$dp" ] && [ -z "$dz" ]; then dz=1
      "${AS[@]}" env TT_VISIBLE_DEVICES="$ua" timeout $(( gu - w > 5 ? gu - w : 5 )) "$PY" "$E" disc > "$d/disc0.json" 2> "$d/disc0.err"; fi
    sleep 5; w=$(( $(date +%s) - ws )); done
  # 2. hide the other link on both ends (refused unless both ends are one PORT_UP link), 3. discovery sees only the kept one.
  # FL_LINKA_MASKED: the link a port_status mask may hide from now on (rc 6: refused, nothing written); fl_report names it
  # unless a reset of this host's chips worked after FL_LINKA_MASK_T
  "${AS[@]}" timeout 90 "$PY" "$E" drop "$da" "$ma" "$db" "$mb" > "$d/mask.json" 2> "$d/mask.err"; rc=$?
  [ $rc = 6 ] || { FL_LINKA_MASKED="$(fl_lk "$a") e$ma <-> $(fl_lk "$b") e$mb"; FL_LINKA_MASK_T=$(date +%s); }
  [ $rc = 0 ] || { fl_linka_inc "mask of e$ma <-> e$mb refused or failed (rc $rc, $d/mask.json)"; return 0; }
  "${AS[@]}" env TT_VISIBLE_DEVICES="$ua" timeout 120 "$PY" "$E" disc > "$d/disc.json" 2> "$d/disc.err" \
    && python3 -I "$E" kept "$d/disc.json" "$ka" "$kb" > "$d/kept.txt" 2>&1 \
    || { fl_linka_inc "discovery did not show the single kept link e$ka <-> e$kb ($d/disc.json)"; return 0; }
  # 4. the all-gather over the kept link, under the beat watchdog
  t0=$(date +%s); FL_JOBS=()
  # shellcheck disable=SC2046  # fl_env prints VAR=value words, split on purpose (as in fl_pair)
  setsid "${AS[@]}" env TT_VISIBLE_DEVICES="$ua" $(fl_env "$FL_LINKA_HANG" "$d/linka") "$PY" "$FL_DIR/fabric_repro.py" --mode ag --n 2 \
    --hw "$FL_LINKA_HW" --out-mem dram --seconds "$FL_LINKA_S" --rows "$d/linka.rows" --beat "$d/linka.beat" --label linka \
    > "$d/linka.log" 2>&1 < /dev/null {PG_OUT}>&- &
  pg_track $!; FL_JOBS+=("$!:$d/linka.beat:linka"); FL_JBDF[linka]="$a $b"; hb_start "link-A repro" "$d/linka.log"
  FL_LAUNCH0_S=$FL_LINKA_FIRST fl_watch $(( FL_LINKA_S + ${FLEET_COMPILE_S:-300} )) "$FL_LINKA_HANG" "$a $b" "$d" "${FL_JOBS[@]}"; hb_stop
  rc=${FL_RC[linka]}; unset 'FL_RC[linka]'; el=$(( $(date +%s) - t0 )); bt=$(cat "$d/linka.beat" 2>/dev/null)
  rk_opencheck "$d/linka.log" "link-A repro" "$a" "$b" || :
  case $rc in
    0) FL_LINKA_LINE=$(fl_linka_line PASS "$el" "$(grep -c '"event": "iter"' "$d/linka.rows" 2>/dev/null) launches"); say "$FL_LINKA_LINE" ;;
    hang) fl_capture linka hang "$a" "$b"
          FL_HUNG="LINK-A HANG $(fl_lk "$a") e$ka <-> $(fl_lk "$b") e$kb"
          fl_find "$(fl_linka_line HANG "$el" "beat '${bt% *}' unchanged at ${bt##* } = +$(( ${bt##* } - t0 )) s; snapshot $d")" ;;
    loss) ;;   # a chip left PCIe: fl_lost below names it
    *) fl_find "LINK-A FAIL $(fl_lk "$a") e$ka <-> $(fl_lk "$b") e$kb: rc $rc ($([ "$rc" = 4 ] && echo "data mismatch" || grep -a . "$d/linka.log" | tail -1 | cut -c1-140)); log $d/linka.log" ;;
  esac
  fl_lost "link-A repro"; return 0
}
# run.sh --repro abort-fabric : on a dev box, broker fabric passes run right after a
# failed job dropped chips in 7 of 78 passes, other fabric passes 0 of 101 (10-25 21:24 PDT: chip 24 dropped 2-3 s after
# job 986 died with rc 1, 31 chips at 1350 MHz). Each cycle: (a) the bus-bar stage's CCL mix + matmul chunk load
# (chunk_load.py) for FLEET_ABORTFAB_LOAD_S (60 s) of load; (b) SIGKILL of that load's own worker process only (the
# chunk_load.py run.sh started, found under its launch pid), while it holds the device with the chips at busy clock;
# (c) within FLEET_ABORTFAB_GAP_S (3 s), with no reset and no go-idle step between, FLEET_ABORTFAB_AG_S (100 s, 90-120)
# of 4x8 ring all_gather on both mesh axes (agloop.py --topo ring --ring-axes 0,1 --no-mm: this box's own chips and
# in-box links only); (d) every chip on PCIe by BDF and tray 1-4, and each root port's corrected AER
# count (fleet_py.py aer, sysfs). FLEET_ABORTFAB_CYCLES (8) cycles; a loss or hang ends them. Opt-in only: never in
# plain run.sh or --mode=full until it catches a fault no other stage does.
fl_abortfab_init() {
  FL_AF_N=${FLEET_ABORTFAB_CYCLES:-8}; FL_AF_LOAD=${FLEET_ABORTFAB_LOAD_S:-60}; FL_AF_AG=${FLEET_ABORTFAB_AG_S:-100}
  FL_AF_GAP=${FLEET_ABORTFAB_GAP_S:-3}; FL_AF_HANG=${FLEET_ABORTFAB_HANG_S:-270}; FL_AF_BUSYW=${FLEET_ABORTFAB_BUSY_WAIT_S:-15}
  [[ $FL_AF_N =~ ^[1-9][0-9]*$ && $FL_AF_LOAD =~ ^[1-9][0-9]*$ && $FL_AF_HANG =~ ^[1-9][0-9]*$ && $FL_AF_BUSYW =~ ^[0-9]+$ ]] \
    || { echo "FLEET_ABORTFAB_CYCLES, _LOAD_S, _HANG_S and _BUSY_WAIT_S take whole numbers (got '$FL_AF_N' '$FL_AF_LOAD' '$FL_AF_HANG' '$FL_AF_BUSYW')" >&2; exit 1; }
  [[ $FL_AF_AG =~ ^[0-9]+$ ]] && [ "$FL_AF_AG" -ge 90 ] && [ "$FL_AF_AG" -le 120 ] \
    || { echo "FLEET_ABORTFAB_AG_S takes 90-120 s of all_gather (got '$FL_AF_AG')" >&2; exit 1; }
  [[ $FL_AF_GAP =~ ^[1-3]$ ]] || { echo "FLEET_ABORTFAB_GAP_S takes 1-3 s (got '$FL_AF_GAP')" >&2; exit 1; }
  [ "${FL_DRY:-0}" = 1 ] || return 0
  local c=$(( FL_AF_LOAD + 45 + FL_AF_AG + 30 ))   # load (after its open), mesh open + compile, all_gather, close + checks
  echo "run.sh --repro abort-fabric dry run: phases and time budget (nothing is opened, reset or loaded)"
  printf '%-9s %-90s %6s\n' phase what min
  printf '%-9s %-90s %6s\n' P0 "gate (read-only), setup" 0.5 P1 "C0 preflight, C1 evidence snapshot, C3 ETH/QSFP port count + tt-smi ETH" 0.8 \
    C2 "start glx_reset (tt-smi -glx_reset) of this host's chips + recount + C3" 2.0 abortfab "abort-fabric repro: $(fl_abortfab_plan)" \
    "$(awk -v s=$((c * FL_AF_N)) 'BEGIN { printf "%.1f", s / 60 }')" C9 "end reset of this host's chips + recheck" 2.5
  echo "total repro-abortfab: $(awk -v s=$((c * FL_AF_N + 30 + 45 + 120 + 150)) 'BEGIN { printf "%.1f", s / 60 }') min at most (not in plain run.sh or --mode=full)"
  echo "only this host's own chips and in-box links: the resets (C2, C9) take no peer snapshot"
  exit 0
}
fl_abortfab_plan() {  # the abort-fabric step in one phrase
  echo "$FL_AF_N cycles of ${FL_AF_LOAD} s CCL mix + matmul load, SIGKILL of its worker at busy clock, ${FL_AF_AG} s 4x8 ring all_gather on both axes started within ${FL_AF_GAP} s (no reset, no idle), PCIe recount + root-port AER"
}
fl_af_inc() {  # the step could not run: INCOMPLETE (fl_verdict, rc 9), never a pass
  FL_AF_INC=$1; FL_AF_LINE="REPRO ABORT-FABRIC NOT RUN (INCOMPLETE): $1"; FL_SKIPPED+=("abort-fabric repro (INCOMPLETE: $1)"); say "$FL_AF_LINE"
}
fl_af_ms() { echo $(( $(date +%s%N) / 1000000 )); }
fl_af_alive() {  # $1 pid: running (not gone, not a zombie)
  local s; s=$(sed 's/.*) //' "/proc/$1/stat" 2>/dev/null) || return 1; [ -n "$s" ] && [ "${s%% *}" != Z ]
}
fl_af_worker() {  # $1 the load's launch pid ($!), $2 its beat file: the chunk_load.py process itself, $1 or a descendant
  # (under sudo, sudo's child) whose command line names this cycle's beat file; nothing else is ever a kill target
  local af_q=("$1") n=0 p c
  while [ ${#af_q[@]} -gt 0 ] && [ $n -lt 64 ]; do
    p=${af_q[0]}; af_q=("${af_q[@]:1}"); n=$((n + 1))
    c=$(tr '\0' ' ' 2>/dev/null < "/proc/$p/cmdline")
    case $c in *chunk_load.py*"--beat $2 "*) fl_af_alive "$p" && { echo "$p"; return 0; } ;; esac
    # shellcheck disable=SC2207  # ps prints one pid per line
    af_q+=($(ps -o pid= --ppid "$p" 2>/dev/null))
  done; return 1
}
fl_af_kill() { kill -KILL "$1"; }   # (b): one SIGKILL to the load worker alone (tests trace it)
fl_af_clocks() {  # "<chips above 800 MHz> <chips read> <lowest MHz>" from sysfs tt_aiclk (no device open)
  local f v b=0 n=0 lo=""
  for f in "${SYS:-/sys}"/class/tenstorrent/tenstorrent\!*/tt_aiclk; do
    [ -r "$f" ] || continue; v=$(cat "$f" 2>/dev/null); [[ $v =~ ^[0-9]+$ ]] || continue
    n=$((n + 1)); [ "$v" -gt 800 ] && b=$((b + 1)); { [ -z "$lo" ] || [ "$v" -lt "$lo" ]; } && lo=$v
  done; echo "$b $n ${lo:--}"
}
fl_af_aer() {  # <pre> <post>: "<rp> (<bdf> (tray N)) +<corrected>" per root port, one line each; the sum on the last line
  awk -F'\t' 'NR == FNR { if ($2 == "rp" && $4 != "-") a[$1] = $4; next }
    $2 == "rp" && !($1 in s) { s[$1] = 1; d = ($4 == "-" || !($1 in a)) ? "?" : $4 - a[$1]; print $1, $3, d; if (d != "?") t += d }
    END { print "TOTAL", "-", t + 0 }' "$1" "$2" 2> /dev/null
}
fl_abortfab() {
  local d=$FL_OUT/abortfab cd i lp w el t0 tl tk ts gap g c rc bt ck mx=0 gaps="" late=0 aert=0 ld=""
  local bdfs; bdfs=$(echo "$RK_MAP" | awk '{ print $1 }'); mkdir -p "$d"
  FL_AF_DONE=0
  for i in $(seq 1 "$FL_AF_N"); do
    cd=$d/c$i; mkdir -p "$cd"; t0=$(date +%s)
    python3 "$FL_DIR/fleet_py.py" aer "$SYS" $bdfs > "$cd/aer_pre.tsv" 2>&1
    # (a) the load: one chunk longer than the load time, so the kill lands in load, never in its quiet
    # shellcheck disable=SC2046  # fl_env prints VAR=value words, split on purpose
    setsid "${AS[@]}" env TT_METAL_INSPECTOR=1 $(fl_env 60 "$cd/hang") "$PY" "$FL_DIR/chunk_load.py" \
      --chunk-min "$(( (FL_AF_LOAD + FL_AF_BUSYW + 299) / 60 ))" --gap-s 0 $(fl_load_args) \
      --rows "$cd/load.rows" --beat "$cd/load.beat" --label "abortfab$i" > "$cd/load.log" 2>&1 < /dev/null {PG_OUT}>&- &
    lp=$!; pg_track $lp; hb_start "abort-fabric cycle $i/$FL_AF_N load" "$cd/load.log"
    # wait for the load to run (past launch 0, which compiles), then FL_AF_LOAD s of it; a loss or an early exit ends it
    tl=""; while :; do
      if ! fl_af_alive "$lp"; then wait "$lp"; rc=$?; ld="load exited by itself (rc $rc) before the kill"; break; fi
      [ -z "$tl" ] && grep -qE '^(done 0|launch [1-9])' "$cd/load.beat" 2>/dev/null && tl=$(date +%s)
      [ -n "$tl" ] && [ $(( $(date +%s) - tl )) -ge "$FL_AF_LOAD" ] && break
      [ -z "$tl" ] && [ $(( $(date +%s) - t0 )) -ge "${FLEET_COMPILE_S:-300}" ] && { ld="load did not start within ${FLEET_COMPILE_S:-300} s"; break; }
      [ $(( ($(date +%s) - t0) % ${FLEET_LSPCI_POLL_S:-10} )) = 0 ] && [ -n "$(rk_snap | awk '$3 != "ok"')" ] && { ld=loss; break; }
      sleep 1; done
    if [ -n "$ld" ]; then hb_stop; fl_af_alive "$lp" && fl_kill "$lp"
      if [ -n "$(rk_snap | awk '$3 != "ok"')" ]; then fl_af_drop "$i" "in its load, before the kill"; break; fi
      fl_af_inc "cycle $i/$FL_AF_N: $ld ($cd/load.log)"; break
    fi
    # busy clock: the chips must be above 800 MHz at the kill (wait up to FL_AF_BUSYW s more)
    ck=$(fl_af_clocks); g=0
    while [ "${ck%% *}" -lt "$(echo "$ck" | cut -d' ' -f2)" ] && [ $g -lt "$FL_AF_BUSYW" ]; do sleep 1; g=$((g + 1)); ck=$(fl_af_clocks); done
    w=$(fl_af_worker "$lp" "$cd/load.beat") || { hb_stop; fl_kill "$lp"; fl_af_inc "cycle $i/$FL_AF_N: no chunk_load.py worker under pid $lp to kill"; break; }
    bt=$(cat "$cd/load.beat" 2>/dev/null)
    # (b) SIGKILL the worker alone, (c) the fabric job within FL_AF_GAP s: only the wait for the worker to leave lies
    # between (at most FLEET_ABORTFAB_REAP_MS, 2 s: the kernel closes its device handles as it exits); no reset, no idle
    tk=$(fl_af_ms); fl_af_kill "$w"
    while fl_af_alive "$w" && [ $(( $(fl_af_ms) - tk )) -lt "${FLEET_ABORTFAB_REAP_MS:-2000}" ]; do sleep 0.05; done
    # shellcheck disable=SC2046
    setsid "${AS[@]}" env TT_METAL_INSPECTOR=1 SNAP_TRIAGE_RUN=$FL_AF_TRI $(fl_env 60 "$cd/hang") "$PY" "$FL_DIR/agloop.py" \
      --topo ring --ring-axes 0,1 --no-mm --seconds "$FL_AF_AG" --rows "$cd/ag.rows" --beat "$cd/ag.beat" --label "abortfab$i" \
      > "$cd/ag.log" 2>&1 < /dev/null {PG_OUT}>&- &
    ts=$(fl_af_ms); FL_AF_PID=$!; pg_track "$FL_AF_PID"; hb_stop; hb_start "abort-fabric cycle $i/$FL_AF_N all_gather" "$cd/ag.log"
    gap=$(( ts - tk )); [ $gap -gt $mx ] && mx=$gap; gaps="$gaps $gap"; [ $gap -gt $(( FL_AF_GAP * 1000 )) ] && late=$((late + 1))
    printf 'cycle %s: load pid %s worker %s killed at %s ms (beat %s), clocks %s busy of %s (lowest %s MHz), fabric pid %s started at %s ms: gap %s ms\n' \
      "$i" "$lp" "$w" "$tk" "${bt% *}" ${ck} "$FL_AF_PID" "$ts" "$gap" >> "$d/cycles.txt"
    say "abort-fabric cycle $i/$FL_AF_N: load worker $w SIGKILLed at ${ck%% *}/$(echo "$ck" | cut -d' ' -f2) chips busy (lowest $(echo "$ck" | cut -d' ' -f3) MHz); 4x8 ring all_gather started $(awk -v g=$gap 'BEGIN { printf "%.2f", g / 1000 }') s later"
    fl_af_alive "$lp" && fl_kill "$lp"; wait "$lp" 2>/dev/null   # the load's own launcher (sudo), once its worker is gone
    FL_WATCH_LOSS=1 FL_HOLD_OK=1 SNAP_TRIAGE_RUN=$FL_AF_TRI fl_watch $(( FL_AF_AG + ${FLEET_COMPILE_S:-300} )) "$FL_AF_HANG" "" "$cd" "$FL_AF_PID:$cd/ag.beat:abortfab"; hb_stop
    rc=${FL_RC[abortfab]}; unset 'FL_RC[abortfab]'; el=$(( $(date +%s) - t0 ))
    # (d) every chip on PCIe by BDF and tray 1-4, each root port's corrected AER count
    python3 "$FL_DIR/fleet_py.py" aer "$SYS" $bdfs > "$cd/aer_post.tsv" 2>&1
    fl_af_aer "$cd/aer_pre.tsv" "$cd/aer_post.tsv" > "$cd/aer_delta.txt"; c=$(awk '$1 == "TOTAL" { print $3 }' "$cd/aer_delta.txt")
    aert=$(( aert + ${c:-0} ))
    say "abort-fabric cycle $i/$FL_AF_N AER corrected per root port: $(awk '$1 != "TOTAL" { t = $2; sub(/^0000:/, "", t); printf "%s%s (%s) +%s", (n++ ? ", " : ""), $1, t, $3 }' "$cd/aer_delta.txt")"
    if [ -n "$(rk_snap | awk '$3 != "ok"')" ]; then fl_af_drop "$i" "$(awk -v g=$gap 'BEGIN { printf "%.2f", g / 1000 }') s after the load worker was SIGKILLed at busy clock"; break; fi
    case $rc in
      0) FL_AF_DONE=$i; say "abort-fabric cycle $i/$FL_AF_N: clean, all chips on PCIe ($el s)" ;;
      hang) fl_capture abortfab hang $bdfs
            fl_attr ABORT-FABRIC HANG "4x8 ring all_gather of cycle $i/$FL_AF_N, $(awk -v g=$gap 'BEGIN { printf "%.2f", g / 1000 }') s after the load worker's SIGKILL (snapshot $cd/hang)" 0 "$cd/ag.log"
            FL_HUNG="ABORT-FABRIC HANG cycle $i/$FL_AF_N"; break ;;
      4) fl_attr ABORT-FABRIC FAIL "data mismatch or non-finite output in cycle $i/$FL_AF_N ($cd/ag.rows)" 0 "$cd/ag.log" "$cd/ag.rows"; break ;;
      loss) fl_af_drop "$i" "during the all_gather"; break ;;
      *) fl_attr ABORT-FABRIC FAIL "all_gather rc $rc in cycle $i/$FL_AF_N ($(grep -a . "$cd/ag.log" | tail -1 | cut -c1-140))" 0 "$cd/ag.log"; break ;;
    esac
  done
  [ $late -gt 0 ] && fl_find "ABORT-FABRIC WARN: $late of $i cycles started the all_gather more than $FL_AF_GAP s after the kill (longest $(awk -v g=$mx 'BEGIN { printf "%.2f", g / 1000 }') s; $d/cycles.txt): those cycles did not test the trigger"
  [ -n "${FL_AF_INC:-}" ] && return 0
  if [ "$FL_AF_DONE" = "$FL_AF_N" ]; then
    FL_AF_LINE="REPRO ABORT-FABRIC PASS: $FL_AF_N/$FL_AF_N cycles, all $(echo "$RK_MAP" | grep -c .) chips on PCIe after each, kill to all_gather start$(echo "$gaps" | awk '{ s = ""; for (i = 1; i <= NF; i++) s = s sprintf(" %.2f", $i / 1000); print s }') s; AER corrected on the root ports +$aert in all"
    say "$FL_AF_LINE"
  fi
  return 0
}
fl_af_drop() {  # $1 cycle, $2 when: one FAIL per missing chip by BDF and tray 1-4, then fl_lost's loss bookkeeping
  local b s
  while read -r b _ s; do
    fl_find "ABORT-FABRIC FAIL $(fl_lk "$b"): off PCIe in abort-fabric cycle $1/$FL_AF_N, $2 ($s; snapshot $FL_OUT/abortfab/c$1)"
  done < <(rk_snap | awk '$3 != "ok"')
  FL_AF_LINE="REPRO ABORT-FABRIC DROP in cycle $1/$FL_AF_N ($2): $(rk_snap | awk '$3 != "ok" { print $1 }' | while read -r b; do fl_lk "$b"; done | paste -sd, | sed 's/,/, /g')"
  [ "${seen:-0}" = 1 ] || fl_lost "abort-fabric repro"; say "$FL_AF_LINE"
}
# the abort-fabric repro's tt-triage scripts on a stall (Inspector on: dump_op_mesh names the chips stuck in the oldest op)
FL_AF_TRI=check_arc,check_noc_status,check_eth_status,dump_fabric_stream_regs,dump_callstacks,dump_running_operations,dump_op_mesh,dump_op_window,device_telemetry
fl_power_start() {  # [file, default $FL_OUT/c7/power.txt] every FLEET_POWER_POLL_S s (in-band ipmitool raw reads, read-only, ~0.4 s a round):
  # "<epoch> <tray 1-4 W> <HSC W> <vr_pg tray 1-4>": Get Sensor Reading 0x35-0x38 (x10 W) and 0x60 (x2 W), tray CPLD
  # (tray 1-4 = bus 0x11/0x13/0x0D/0x0F at 0x80, cpld_5f.py TRAY_BUS) reg 0x19, ff = all rails good; a busbar drop reads
  # ~10 W and 00 (seen on a quad host: tray 4 10 W with bus 0x0F 00). Before that the vr_pg columns were in
  # bus order 0x0D 0x0F 0x11 0x13 (= tray 3 4 1 2).
  # NA: the sensor says unavailable; ERR: no answer. (dcmi power reading returns nothing on these hosts.)
  # Then Power_PSU0-3, sensors 0x31-0x34 (x20 W): columns 11-14, their sum is the box's kW (the tray sensors
  # cap at 2550 W each, so they cannot show 12-14 kW). Nothing reads these to stop or cap the load.
  # Then TEMP_INLET, TEMP_OUTLET, TEMP_OUTLET2, sensors 0x1E, 0x1F, 0x3E (x1 deg C): columns 15-17, the box's
  # air in and out under load (some boxes have no TEMP_OUTLET2: ERR, an empty cell in box_power.csv).
  # Every poll row also goes to $FL_OUT/box_power.csv (fl_bp_row), the run's power trace over time.
  # Each read goes through hsc_i2c (lib/hsc_check.sh), one at a time with the HSC reads: overlapping in-band I2C sets the HSCs' CML.
  local S=() main=$$ f=${1:-$FL_OUT/c7/power.txt} bp; bp=${FLEET_BOX_POWER_CSV:-${FL_OUT:-${f%/*}}/box_power.csv}; mkdir -p "${f%/*}" "${bp%/*}"
  [ "$(id -u)" = 0 ] || { timeout -k 2 5 sudo -n -l "${FLEET_IPMITOOL:-ipmitool}" raw 0x04 0x2d 0x31 > /dev/null 2>&1 && S=(sudo -n); }   # the exact read
  [ -s "$bp" ] || echo "$FL_BP_HDR" > "$bp"
  ( trap - EXIT; sp=""; trap '[ -z "$sp" ] || kill "$sp" 2>/dev/null; exit 0' TERM   # fl_power_stop: end the poll sleep too, no orphan
    while kill -0 "$main" 2>/dev/null; do
      t=$(date +%s.%3N); l=${t%.*}
      rd() { for n; do
        read -r v f _ <<< "$(hsc_i2c timeout 5 "${S[@]}" "${FLEET_IPMITOOL:-ipmitool}" raw 0x04 0x2d "${n%:*}" 2>/dev/null)"
        if ! [[ "$v$f" =~ ^[0-9a-fA-F]{4}$ ]]; then l="$l ERR"; elif (( 16#$f & 0x20 )); then l="$l NA"
        else l="$l $(( 16#$v * ${n#*:} ))"; fi
      done; }
      rd 0x35:10 0x36:10 0x37:10 0x38:10 0x60:2
      for b in 0x11 0x13 0x0D 0x0F; do
        v=$(hsc_i2c timeout 5 "${S[@]}" "${FLEET_IPMITOOL:-ipmitool}" raw 0x06 0x52 "$b" 0x80 0x01 0x19 2>/dev/null | tr -d ' \n'); l="$l ${v:-ERR}"
      done
      rd 0x31:20 0x32:20 0x33:20 0x34:20
      rd 0x1E:1 0x1F:1 0x3E:1   # TEMP_INLET, TEMP_OUTLET, TEMP_OUTLET2 (deg C): columns 15-17, ERR where a box has no such sensor
      echo "$l"; fl_bp_row "$t" "$l" >> "$bp"; sleep "${FLEET_POWER_POLL_S:-2}" & sp=$!; wait "$sp"; sp=""; done ) >> "$f" 2>/dev/null &
  FL_PW_PID=$! FL_PW_FILE=$f
}
# box_power.csv: one row per fl_power_start poll, in every mode, for every stage that polls. UTC ISO time
# and epoch s of the poll, the stage number and name from fleet/stages.tsv (the phase running then, phases.txt), Power_PSU0-3
# W, Power_UBB0-3 W (tray 1-4) and box kW (PSU0-3 sum; empty unless all four read), then TEMP_INLET, TEMP_OUTLET and
# TEMP_OUTLET2 in deg C (appended last so older readers keep their columns). A failed read (ERR, NA) is an empty cell.
FL_BP_HDR="timestamp_utc,epoch_s,stage_no,stage,PSU0_W,PSU1_W,PSU2_W,PSU3_W,Power_UBB0_W,Power_UBB1_W,Power_UBB2_W,Power_UBB3_W,box_kW,TEMP_INLET,TEMP_OUTLET,TEMP_OUTLET2"
fl_bp_row() {  # <epoch s.ms> <power.txt row>: its csv row
  local p n=""; p=$(tail -n1 "$FL_OUT/phases.txt" 2>/dev/null); p=${p%% *}; p=$(fl_pn "$p")
  [[ $p =~ ^stage\ ([0-9]+)\ (.*)$ ]] && { n=${BASH_REMATCH[1]}; p=${BASH_REMATCH[2]}; }
  echo "$2" | awk -v t="$1" -v iso="$(date -u -d "@$1" +%FT%T.%3NZ)" -v n="$n" -v p="${p//,/;}" '
    function w(x) { return (x ~ /^[0-9]+$/) ? x : "" }
    { k = 0; s = 0; for (i = 11; i <= 14; i++) if (w($i) != "") { k++; s += $i }
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", iso, t, n, p, w($11), w($12), w($13), w($14), w($2), w($3), w($4), w($5),
        (k == 4 ? sprintf("%.3f", s / 1000) : ""), w($15), w($16), w($17) }'
}
fl_power_stop() { [ -n "$FL_PW_PID" ] && { sleep "${FLEET_POLL_TAIL_S:-2}"; kill "$FL_PW_PID" 2>/dev/null; wait "$FL_PW_PID" 2>/dev/null; }; FL_PW_PID=""; }
fl_aer_start() {  # <rows file> <stage name>: every FLEET_AER_POLL_S (60) s, read-only, the corrected AER count of each chip's
  # upstream root port (sysfs aer_rootport_total_err_cor / aer_dev_correctable, and corrected AER/GHES lines per BDF in dmesg:
  # a firmware-first host leaves sysfs at 0), one row per port per sample (fleet_py.py aerrate, well under 1 s). Runs beside
  # the load, so the stage takes no longer (a dev box's 41:00.0, 5,230 GHES corrected
  # errors in one run that reported nothing, dropped 2 h later). fl_aer_stop takes the last sample and judges.
  [ -n "${FL_AER_PID:-}" ] && return 0
  local S=() main=$$ f=$1 st ch; mkdir -p "${f%/*}"; st=${f%.txt}.state.json; rm -f "$st"; FL_AER_STAGE=$2
  read -ra S <<< "$(sudo_pfx dmesg)"   # fl_aer_sample's one sudo read
  ch=$(echo "$RK_MAP" | awk 'NF { print $1 }' | while read -r b; do printf '%s=%s ' "$b" "$(rk_tray "$b")"; done)
  FL_AER_ARGS=("$st" "${S[*]}" "$ch")
  ( trap 'kill "$sp" 2>/dev/null; exit 0' TERM; trap - EXIT; sp=""
    while kill -0 "$main" 2>/dev/null; do fl_aer_sample "${FL_AER_ARGS[@]}" | fl_aer_live; sleep "${FLEET_AER_POLL_S:-60}" & sp=$!; wait "$sp"; done ) >> "$f" 2>/dev/null &
  FL_AER_PID=$! FL_AER_FILE=$f
}
fl_aer_sample() {  # <state.json> <sudo prefix or ""> <bdf=tray list>: one aerrate row per port on stdout
  local s=() ; [ -n "$2" ] && read -ra s <<< "$2"
  timeout 10 "${s[@]}" dmesg > "$1.dmesg" 2>/dev/null || : > "$1.dmesg"
  # shellcheck disable=SC2086  # $3 is the bdf=tray word list
  FLEET_AER_STAGE_FAIL=${FLEET_AER_STAGE_FAIL:-20} FLEET_AER_HOST=${H:-?} timeout 20 python3 "$FL_DIR/fleet_py.py" aerrate "${SYS:-/sys}" "$1" "$1.dmesg" "$(date +%s)" "${FLEET_AER_MIN_WARN:-500}" $3
}
fl_aer_live() {  # the AER poll's rows on stdin to stdout; its live AER FAIL (one port crossed FLEET_AER_STAGE_FAIL this
  # stage, the others +2 or less) is a finding the minute it crosses, keyed in aer_stage/live so the stage end does not repeat it
  local l; while IFS= read -r l; do case $l in
    "AER FAIL "*) fl_find "$l"; mkdir -p "$FL_OUT/aer_stage"; echo "${l%%, host *}" | tee -a "$FL_OUT/aer_stage/reported" >> "$FL_OUT/aer_stage/live" ;;
    *) echo "$l" ;; esac; done
}
fl_aer_stop() {  # stop the poll, one last sample (any rise since the last one, per_min over at least a minute), then one
  # AER WARN 'weak PCIe link' per root port that rose over FLEET_AER_MIN_WARN (500)/min, naming its chips by BDF and tray
  [ -n "${FL_AER_PID:-}" ] || return 0
  kill "$FL_AER_PID" 2>/dev/null; wait "$FL_AER_PID" 2>/dev/null; FL_AER_PID=""
  fl_aer_sample "${FL_AER_ARGS[@]}" 2>/dev/null | fl_aer_live >> "$FL_AER_FILE"; rm -f "${FL_AER_ARGS[0]}.dmesg"
  python3 "$FL_DIR/fleet_py.py" aerrate-judge "$FL_AER_FILE" "${FLEET_AER_MIN_WARN:-500}" "$FL_AER_STAGE" 2>/dev/null | while read -r l; do fl_find "$l"; done
  local n; n=$(grep -c ' OVER$' "$FL_AER_FILE" 2>/dev/null); say "$FL_AER_STAGE: corrected AER per root port each minute in ${FL_AER_FILE#"$FL_OUT"/} (${n:-0} port-minute(s) over ${FLEET_AER_MIN_WARN:-500}/min)"
  return 0
}
fl_load_line() {  # <power.txt> <ub0 result.json> <label> [<telem.csv>]: the load-strength line, with
  # how many chips sit at their power or current limit from C8; with telem.csv also the per-tray swing
  local l le; le=$(mktemp); l=$(python3 "$FL_DIR/load_line.py" "$1" "$2" "$3" "$FL_OUT/c8_telemetry.txt" "$FL_OUT/c8_limits.txt" ${4:+"$4"} 2>"$le" | grep -a '^LOAD ')
  [ -n "$l" ] || l="LOAD unavailable ($(tail -n1 "$le" 2>/dev/null | cut -c1-200))"; rm -f "$le"; say "$l"; echo "$l" >> "$FL_OUT/load.txt"
}
fl_stage_load() {  # <C5>: the stage's TFLOP/s line from the files it wrote (fleet/stage_load.py)
  local l; l=$(python3 "$FL_DIR/stage_load.py" "$1" "$FL_OUT/$(echo "$1" | tr C c)" 2>&1 | head -1); say "${l:-LOAD $1: not read}"; echo "${l:-LOAD $1: not read}" >> "$FL_OUT/load.txt"
}
fl_pw_at() {  # $1 power.txt, $2 epoch: the poll row at or before $2 (a drop's last good one), else the last; "" if none
  awk -v t="$2" 'NF >= 10 { if ($1 <= t) b = $0; l = $0 } END { print (b != "" ? b : l) }' "$1" 2>/dev/null
}
fl_pg_fell() {  # $1 power.txt: "<epoch> <tray N> <vr_pg>" for each tray whose vr_pg fell: non-ff on 2 polls in a row (an ff resets the count)
  # (one misread, or a CPLD that never reads ff, is no fall); the epoch is that of the first non-ff poll
  awk 'NF >= 10 { for (u = 0; u < 4; u++) { v = tolower($(7 + u))
         if (v !~ /^[0-9a-f][0-9a-f]$/ || (u in s)) continue
         if (v == "ff") { delete b[u]; n[u] = 0; continue }
         if (!(u in b)) { b[u] = $1 " " v }
         if (++n[u] >= 2) { s[u] = 1; split(b[u], x, " "); print x[1], u + 1, x[2] } } }' "$1" 2>/dev/null
}
fl_pg_start() {  # $1 power.txt: "<tray N> <vr_pg>" for each tray already non-ff on its first poll with vr_pg read (no fall: it never read ff)
  awk 'NF >= 10 { for (u = 0; u < 4; u++) { v = tolower($(7 + u))
         if (!(u in s) && v ~ /^[0-9a-f][0-9a-f]$/) { s[u] = 1; if (v != "ff") print u + 1, v } } }' "$1" 2>/dev/null
}
fl_pw_says() {  # $1 a power.txt row: "tray 1 470 W, ..., vr_pg ff ff ff ff"
  echo "$1" | awk 'NF >= 10 { printf "tray 1 %s W, tray 2 %s W, tray 3 %s W, tray 4 %s W, HSC %s W, vr_pg %s %s %s %s", $2, $3, $4, $5, $6, $7, $8, $9, $10
    if (NF >= 14 && $11 $12 $13 $14 ~ /^[0-9]+$/) printf ", box (PSU0-3) %.2f kW", ($11 + $12 + $13 + $14) / 1000 }'
}
fl_drop_tb() {  # <vr_pg fall lines> <stage start epoch> <s to the drop being seen>: the last good poll's epoch, 1 s before the
  # first vr_pg fall, else one PCIe poll before the drop was seen (with no fall read, echo "" is still one line,
  # and its -1 put b02u08's 5:38 drop "before chunk 1 (mesh open, compile, references)")
  local t; t=$(echo "$1" | awk 'NR == 1 && NF { print $1 - 1 }'); echo "${t:-$(( $2 + $3 - ${FLEET_LSPCI_POLL_S:-10} ))}"
}
fl_c7() {  # <dir under FL_OUT> <stage name> <chunk_load.py args...>: chunk_load.py chunks with the power and PCIe polls; a loss ends it.
  # fl_busbar runs it for the bus-bar stage's mix chunks (FL_BB_*, quick's FLEET_C7_S one chunk), its heavy
  # chunk and its long chunk (the old sustained part), inside the stage's power poll; the args are fl_load_args'.
  local d=$FL_OUT/$1 st=$2 t0 rc el pw g ub pid pg lost bd v tb cr own="" nm=chunks tag=H3SHAPE
  local cm=$FL_BB_MIN cs=$FL_BB_MIX_S cn=$FL_BB_CHUNKS gp=$FL_BB_GAP_S; shift 2
  # the bus-bar stage's heavy chunk (quick and full): one FLEET_BUSBAR_HEAVY_S chunk, chunk_load.py's own --heavy-pcie-mb as main's quick chunk ran it
  [ "${d##*/}" = heavy ] && { cm=$(awk -v s="$FL_BB_HV" 'BEGIN { printf "%g", s / 60 }'); cs=$FL_BB_HEAVY_S; cn=1; nm="heavy chunk"; tag=CHUNK; }
  [[ $d == */swap/heavy ]] && nm="heavy chunk (swap, WARNING: TRAYTEST_MIX_SWAP=heavy)"
  [ "${d##*/}" = long ] && { cm=$FL_LONG_MIN; cs=$FL_BB_LONG_S; cn=1; nm="long chunk"; }   # the old sustained part's place
  mkdir -p "$d"; th_start "$d"; [ "${d##*/}" = heavy ] && vc_start "$d"   # stage 5: VCORE min/mean per chip during the heavy part only
  FL_C7_NJ=""   # set when this chunk failed at device open on off-box links only (not judged)
  t0=$(date +%s); [ -n "$FL_PW_PID" ] || { fl_power_start "$d/power.txt"; fl_aer_start "$d/aer_rate.txt" "$st"; own=1; }; local pf=$FL_PW_FILE
  # quick's one chunk (FLEET_C7_S) exposes numeric faults like the chunks
  [ -n "$FL_C7_LOAD" ] && [ "${d##*/}" = busbar ] && { nm=chunk; tag=CHUNK; cn=1; cm=$(awk -v s="$FL_C7_LOAD" 'BEGIN { printf "%g", s / 60 }'); }
  # the long chunk's pre-ramp, only with FLEET_PRERAMP_S > 0 (the mix, step and
  # heavy chunk before it ramp the chips); its C8 telemetry tops up only the chips below their own power or current limit
  # (limits read, never written), and only with FLEET_TOPUP=1
  local pre=()
  [ "${d##*/}" = long ] && [ "$FL_LONG_PRE" -gt 0 ] && pre=(--pre-ramp-s "${FLEET_PRERAMP_S:-0}" \
    --pre-warm-s "${FLEET_PRERAMP_WARM_S:-30}" --topup-telem "$FL_OUT/c8_telemetry.txt" --topup-limits "$FL_OUT/c8_limits.txt")
  local tr=()   # FLEET_LOAD_TRAYS: only these trays loaded, each on its own submesh; the rest enumerated, no kernels
  [ -n "${FLEET_LOAD_TRAYS:-}" ] && { tr=(--trays "$FLEET_LOAD_TRAYS"); say "$st $nm: load on tray $FLEET_LOAD_TRAYS only (FLEET_LOAD_TRAYS; other trays enumerated, no kernels)"; }
  setsid "${AS[@]}" env TT_METAL_INSPECTOR=1 $(fl_env 60 "$d/hang") "$PY" "$FL_DIR/chunk_load.py" --chunk-min "$cm" "${pre[@]}" "${tr[@]}" "$@" \
    --gap-s "$gp" --rows "$d/rows.jsonl" --beat "$d/beat" --label "${d##*/}" > "$d/run.log" 2>&1 < /dev/null {PG_OUT}>&- &
  pid=$!; pg_track $pid; hb_start "$st $nm" "$d/run.log"
  # the load runs until it ends or its beat stalls (FLEET_C7_HANG_S); no run budget stops it
  FL_HOLD_OK=1 FL_WATCH_LOSS=1 FL_WATCH_PG=$pf FL_WATCH_PG_T0=$t0 fl_watch $(( cs + ${FLEET_COMPILE_S:-300} + (cn + 1) * (${TRAY_QUIET_CLK_CAP_S:-300} + 70) )) "${FLEET_C7_HANG_S:-120}" "" "$d" "$pid:$d/beat:hang"; hb_stop
  vc_stop; th_stop   # before any capture or hold: the samplers never read a hung chip
  rc=${FL_RC[hang]}; unset 'FL_RC[hang]'; el=$(( $(date +%s) - t0 ))
  # a vr_pg fall leaves PCIe a few s later: wait up to 2 PCIe polls for it, so the loss below names its endpoints
  [ "$rc" = loss ] && [ -n "$(fl_pg_fell "$pf" | awk -v t="$t0" '$1 >= t')" ] && [ -z "$(rk_snap | awk '$3 != "ok"')" ] && sleep $(( 2 * ${FLEET_LSPCI_POLL_S:-10} ))
  sleep "${FLEET_POLL_TAIL_S:-2}"; [ -n "$own" ] && { kill "$FL_PW_PID" 2>/dev/null; wait "$FL_PW_PID" 2>/dev/null; FL_PW_PID=""; fl_aer_stop; }   # one more poll row after a drop
  pg=$(fl_pg_fell "$pf" | awk -v t="$t0" '$1 >= t')   # this run's falls only: the bus-bar stage's parts share one power.txt
  [ "${d##*/}" = heavy ] || fl_pg_start "$pf" | while read -r tn v; do   # a tray whose vr_pg was not ff when the stage began: its own FAIL, not a fall
    bd=$(rk_snap | while read -r b c st; do [ "$(rk_tray "$b")" = "$tn" ] && printf '%s ' "${b#0000:}"; done)
    fl_find "POWER FAIL tray $tn (${bd% }): vr_pg $v before the $st load started (never read ff)"; done
  fl_chunk_report "$d"   # the chunk and quiet windows and the chunk-end numeric lines, whatever ended it
  # a LOAD line per chunk: how many chips sit at their power or current limit
  python3 "$FL_DIR/load_line.py" chunks "$d/rows.jsonl" "$FL_OUT/c8_telemetry.txt" "$FL_OUT/c8_limits.txt" "$st" \
    "$d/power_limit.json" 2>&1 | while IFS= read -r l; do say "$l"; echo "$l" >> "$FL_OUT/load.txt"; done
  fl_chips_line "$d/rows.jsonl" "$st ${d#"$FL_OUT"/}"
  if [ "$rc" = loss ] || [ -n "$pg" ] || fl_lost "$st $nm"; then
    fl_lost "$st $nm at $((el / 60)):$(printf %02d $((el % 60)))" > /dev/null
    [ -n "$pg" ] && FL_PGDROP="vr_pg fell on tray $(echo "$pg" | awk '{ print $2 }' | sort -u | paste -sd,) in the ${st% stage} stage"
    echo "$t0 $el $(echo "$pg" | awk 'NR == 1 { print $1 }')" > "$d/drop_s.txt"   # stage start epoch, s to the drop being seen, epoch of the first vr_pg fall
    pw=$(fl_pw_says "$(awk 'NF >= 10 { l = $0 } END { print l }' "$pf" 2>/dev/null)")
    tb=$(fl_drop_tb "$pg" "$t0" "$el")   # the last good poll
    say "$st BMC before the drop: $(fl_pw_says "$(fl_pw_at "$pf" "$tb")"); after: ${pw:-unread}"
    say "$st drop $(python3 "$FL_DIR/chunk_sched.py" where "$d/rows.jsonl" "$tb" $(rk_snap | awk '$3 != "ok" { print $1 }') 2>&1 | head -1)"
    # one FAIL finding per tray that lost endpoints or read a vr_pg fall, naming its lost BDFs
    lost=$(rk_snap | awk '$3 != "ok"' | while read -r b c st; do echo "$(rk_tray "$b") ${b#0000:}"; done)
    for ub in $( { echo "$lost" | awk 'NF { print $1 }'; echo "$pg" | awk 'NF { print $2 }'; } | sort -u); do
      g=$(echo "$lost" | awk -v u="$ub" '$1 == u' | grep -c .); bd=$(echo "$lost" | awk -v u="$ub" '$1 == u { printf "%s%s", (n++ ? " " : ""), $2 }')
      v=$(echo "$pg" | awk -v u="$ub" '$2 == u { printf "vr_pg %s at %d s of the %s", $3, $1 - t0, s }' t0="$t0" s="$st")
      fl_find "$([ "$g" -ge 8 ] || [ -n "$v" ] && echo "POWER DROP" || echo "CHIP DROP") tray $ub ($g chips lost${bd:+: $bd}${v:+; $v}) at $((el / 60)):$(printf %02d $((el % 60))) of the $st, BMC after: ${pw:-unread}"
    done
    return 0
  fi
  pw=$(fl_pw_says "$(awk 'NF >= 10 { l = $0 } END { print l }' "$pf" 2>/dev/null)")
  case $rc in
    0) say "$st $(case $nm in chunk) echo "chunk (FPU/SFPU/DRAM/CCL/PCIe + numeric)" ;; "heavy chunk") echo "heavy chunk (stage 5 numeric DIDT: $FL_DIDT_OPS)" ;; *) echo "$nm" ;; esac): clean ($el s; BMC at the end: ${pw:-unread})" ;;
    4) if ! grep -aqE '^(NUMERIC|LINK-DATA|L1) FAIL' "$d/report.txt" 2>/dev/null; then   # rc 4 with no chip-end line: name the chips from the log
         fl_attr CHUNK FAIL "rc 4, no chunk-end NUMERIC FAIL line ($d/run.log)" 0 "$d/run.log"; fi
       # the chunks' h3 part (all-gather/reduce-scatter) mismatched: a CCL failure, swept per link after the reset
       # (h3-ref: its launch-0 reference; CCL_PARTS in chunk_sched.py), wherever it sorts in the part list ([dram-ref,h3])
       cr='^NUMERIC FAIL .*\[([a-z0-9-]+,)*h3(-ref)?[],]'
       l=$(grep -aE "$cr" "$d/report.txt" 2>/dev/null | head -1 | sed "s/\]: .*/]/")
         [ -n "$l" ] && { FL_PIN="$st chunks: CCL (h3 part) mismatch, $l"   # capture the chips it names before the reset
           # every chip a line names: a CCL group finding lists its members after its source
           fl_capture "${d##*/}" fail $(grep -aE "$cr" "$d/report.txt" | grep -oE '[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]' | sed 's/^/0000:/' | sort -u); } ;;
    # the stopped chips (BDF, tray 1-4) from the per-chip launch counts and readback errors, or the whole box
    # when every chip stalled together; only those chips are captured, before the deadline cuts in (rs_cap_step)
    hang) FL_HANG_PIN=$(python3 "$FL_DIR/chunk_sched.py" stalled "$d/rows.jsonl" 2>/dev/null); v=$(sed -n 2p <<< "$FL_HANG_PIN")
          FL_HANG_PIN=$(sed -n 1p <<< "$FL_HANG_PIN"); [ -n "$FL_HANG_PIN" ] || FL_HANG_PIN="PINPOINT: not pinpointed: no per-chip readback rows ($d/rows.jsonl)"
          # every chip stalled together and none was lost (a loss returned above): a box-wide hang names no chip
          # or tray and gives no hardware verdict (fl_report: INCOMPLETE); stopped chips keep their HANG FAIL by BDF and tray
          case $FL_HANG_PIN in "PINPOINT: whole box: "*) v=""; fl_find "HANG INCOMPLETE $(fl_wholebox "${FL_HANG_PIN#PINPOINT: }") at $el s of the $st; no chip named, no hardware verdict" ;;
            *) fl_find "HANG $FL_HANG_PIN" ;; esac   # under each FAIL tray line, after a power loss (tray_why.py)
          fl_hang_aer "$t0" "$el" "$st" "$d"
          fl_capture "${d##*/}" hang ${v:-$(echo "$RK_MAP" | awk '{ print $1 }')}
          if [[ $FL_HANG_PIN == "PINPOINT: whole box: "* ]]; then FL_HANG_PIN=""; FA_CHIPS=""; FL_HUNG="$tag HANG whole box (chip not named)"
          else fl_attr $tag HANG "at $el s of the $st$([ -e "$d/hang" ] && echo " (snapshot $d/hang)"); ${FL_HANG_PIN#PINPOINT: }" 0 "$d/run.log"; FL_HANG_PIN=""; FL_HUNG="$tag HANG ${FA_CHIPS:+$FA_TAG}${FA_CHIPS:-(chip not named)}"; fi
          say "$st hang $(python3 "$FL_DIR/chunk_sched.py" where "$d/rows.jsonl" "$(stat -c %Y "$d/beat" 2>/dev/null || echo $(( t0 + el )))" $(fc_bdfk $FA_CHIPS) 2>&1 | head -1)"
          FL_PIN="$st $nm hang at $el s" ;;   # the chunk it hung in: the 'sustained hang' line above
    # rc 1 from chunk-end readback errors: the CHUNK FAIL lines above name each part, chip and cause once
    # an exception in a CCL op (not a hang, not a mismatch) is a CCL failure too (fl_ccl_err). Its one finding is fl_attr's,
    # naming the chip from the log by BDF and tray 1-4, and its one fault.txt entry is fault_attr's (FL_CAP_NAMED);
    # a device error (fl_dev_err: No such device, 0xffffffff, firmware startup, device timeout) is captured and
    # named the same way, labelled a device error; a fabric-setup error (fl_setup_err) is a SETUP/SOFTWARE WARN,
    # never CCL; any other run failure names no chip (FA_CHIPS may be an earlier fault's)
    *) if [ "$rc" = 1 ] && grep -aqE '"event": "end".*"verify_errors": [1-9]' "$d/rows.jsonl" 2>/dev/null && grep -aqE '^(CHUNK FAIL|HANG INCOMPLETE whole box) ' "$d/report.txt" 2>/dev/null; then
         :   # rc 1 with a chunk-end NUMERIC FAIL: its chips are already named by the CHUNK FAIL lines above
       elif ! fl_dev_err "$d/run.log" && fl_setup_err "$d/run.log"; then   # a fabric-setup error is no CCL fault; test-side,
         # so a named WARN with no chip capture and no FL_PIN, never a hardware FAIL
         fl_find "$tag WARN SETUP/SOFTWARE: fabric setup failed rc $rc (test-side setup or software, not a hardware fault)"
         FL_SETUP_INC="${FL_SETUP_INC:+$FL_SETUP_INC; }$st $nm rc $rc"   # that stage did not run: INCOMPLETE, never a pass
       elif fl_nj_open "$st $nm" "$(test -s "$d/rows.jsonl" && echo 1 || echo 0)" "$d/run.log"; then
         # as fl_bb_step: the open timed out only on off-box links whose own evidence says the peer
         # host is down or resetting (ethcore's WARN, this host's own chips only), before any load row: a named WARN, no
         # chip FAIL or pin; the chunk is not judged (fl_busbar's not-tested check). Mixed on-box lines or rows
         # written first: the CCL error path below, with no ETH note first
         FL_C7_NJ=1
         say "$st $nm: chunk_load.py rc $rc at device open, only off-box links with the peer host down or resetting (not judged; see $d/run.log)"
       elif fl_ccl_err "$d/run.log"; then fl_c7_named "CCL error" "$tag" "$st $nm" "$rc" "$d"
       elif fl_dev_err "$d/run.log"; then fl_c7_named "device error" "$tag" "$st $nm" "$rc" "$d"   # never labelled CCL
       else
         # no CCL or device error line and no chunk-end NUMERIC FAIL: nothing here pins a chip, and FA_CHIPS may still hold an
         # earlier fault's chip (C2, abort-fabric, an earlier chunk), so this line names none
         fl_find "$tag WARN run failed, rc $rc: $(grep -a . "$d/run.log" | tail -1 | cut -c1-140)"
       fi ;;
  esac
}
fl_hang_aer() {  # <t0> <el> <stage> <dir>: FLEET_HANG_AER_NAME (1, 0 = off): a stall whose pinpoint names no
  # single chip (whole box, several, or none) and whose stage's AER poll shows one root port +3 or more, every other +1 or less,
  # is an AER FAIL naming that port's chip (filed under its tray, not the box's first); read-only, the poll's rows only
  local n s; [ "${FLEET_HANG_AER_NAME:-1}" != 0 ] && [ -s "${FL_AER_FILE:-}" ] || return 0
  n=$(grep -oE '[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]' <<< "$FL_HANG_PIN" | sort -u | grep -c .); [ "$n" != 1 ] || return 0
  s=$(stat -c %Y "$4/beat" 2> /dev/null || echo $(( $1 + $2 )))
  python3 "$FL_DIR/fleet_py.py" aerhang "$FL_AER_FILE" "$s" "${H:-?}" 2> /dev/null | while IFS= read -r l; do fl_find "$l at $2 s of the $3"; done
}
fl_wholebox() {  # $1 chunk_sched.py stalled's whole-box line (no "PINPOINT: "): the same words with its tray list dropped
  sed -E 's/^(whole box: all [0-9]+ chips) \((tray [1-4?](, )?)+\)/\1/' <<< "$1"
}
fl_chips_line() {  # <rows.jsonl|result.json> <label>: the stage's per-chip LOAD line (W, A, MHz next to its own power and current
  # limit, TFLOP/s, the limit each chip hits) and a LOAD WARN finding per chip whose median W over the stage's polls is under 50% of
  # the box's median chip (a LOAD WARN only, never a FAIL; fleet/load_line.py chips;
  # limits only read)
  # per tray, the chips' summed power against their own power limits; a tray under
  # 90% for over half the load is LOAD LOW WARN, one with too little read LOAD UNREAD WARN: warns, never a clean pass, never a fail
  { python3 "$FL_DIR/load_line.py" chips "$1" "$FL_OUT/c8_telemetry.txt" "$FL_OUT/c8_limits.txt" "$2" 2>&1
    python3 "$FL_DIR/load_line.py" target "$1" "$FL_OUT/c8_telemetry.txt" "$FL_OUT/c8_limits.txt" "$2" "$FL_OUT/fault_times.tsv" 2>&1; } | grep -a '^LOAD ' \
    | while IFS= read -r l; do case $l in "LOAD WARN "*|"LOAD LOW WARN "*|"LOAD UNREAD WARN "*) fl_find "$l" ;; *) say "$l" ;; esac; echo "$l" >> "$FL_OUT/load.txt"; done
}
fl_chunk_report() {  # $1 the c7 dir: chunk_sched.py's lines -> report.txt; chunk/quiet lines said, LOAD lines also to load.txt, NUMERIC FAIL, CHUNK FAIL (not checked), QUIET FAIL (a chip not at idle clock after the 5 min cap) and WARN lines are findings
  # the bus-bar stage's chunks are judged on drops, not on box kW: no 12-14 kW verdict there, but for
  # its long chunk (the sustained part's place, goal 8's 12-14 kW)
  local l x="" xn=" (stage 5 numeric verify)"; [ "${1##*/}" = heavy ] && { x=" (stage 4 bus-bar stage heavy chunk)"; xn=" (stage 5 numeric verify: heavy chunk, DIDT $FL_DIDT_OPS)"; }   # the heavy chunk: not CCL-mix chunk 1; a numeric line is stage 5, never the bus-bar stage
  CHUNK_KW_TARGET=$(case $1 in */busbar/long) echo 1 ;; */busbar|*/busbar/*) echo 0 ;; *) echo 1 ;; esac) python3 "$FL_DIR/chunk_sched.py" report "$1/rows.jsonl" "$1/power.txt" "$FL_OUT/c8_telemetry.txt" "$FL_OUT/c8_limits.txt" > "$1/report.txt" 2>&1 || return 0
  while IFS= read -r l; do
    case $l in
      "NUMERIC LAUNCHES "*|"NUMERIC RATE "*) [ "${1##*/}" = heavy ] && say "$l" ;;   # once per stage: only the heavy chunk's report says the launch counts and rates
      "NUMERIC FAIL "*|"NUMERIC WARN "*|"L1 FAIL "*|"L1 WARN "*) fl_find "$l$xn" ;;   # N1's L1 lines are stage 5 numeric too
      "LINK-DATA FAIL "*) fl_find "$l$x" ;;
      "NUMERIC "*) say "$l$xn" ;;
      "CHUNK FAIL "*|"QUIET FAIL "*|"HANG INCOMPLETE "*|*" WARN "*) fl_find "$l$x" ;;   # HANG INCOMPLETE: a box-wide hang, no tray
      "LOAD "*|"CHIPS "*|"PEAK "*) say "$l$x"; echo "$l$x" >> "$FL_OUT/load.txt" ;;
      *) say "$l$x" ;;
    esac
  done < "$1/report.txt"
  [ "${1##*/}" = heavy ] && [ -s "$1/vcore.txt" ] && while IFS= read -r l; do say "$l"; done < <(vc_report "$1" "$1/report.txt" "$FL_OUT/c8_telemetry.txt")
  while IFS= read -r l; do say "$l"; done < <(th_chunk "$1")   # AICLK below target and why, per chip
}
# a hang, chip loss, CCL failure or vr_pg fall earlier in the
# bus-bar stage no longer skips its remaining parts. Once per new fault, the stage recovers with glx_reset (rk_reset, never
# tt-smi -r), a root-port rescan and a recount, then runs the next part. The part is not run (a "not run" line) when a hang
# or CCL failure had no OK glx_reset, when chips are still missing after the recount (named by BDF and tray), or in hold mode.
# after a vr_pg fall no more load goes on the fallen tray. The chunk_load.py parts (mix, heavy, long) run on
# the trays whose rails did not fall (FL_BB_TRAYS, passed as FLEET_LOAD_TRAYS: the fallen tray enumerated, no kernels); the
# parts whose load reaches every tray (qb, step, rw, maxutil) are not run, naming the fallen tray. The mix's shape tray moves off
# a fallen tray. Chips still missing after the recovery stop every later part, named by BDF and tray.
fl_bb_notrun() {  # <part> <why> [skipped]: the line for a part fl_bb_cont does not run; stage 6 (C5) comes after the stage
  local v=${3:-not run}
  if [ "$1" = "stage 6" ]; then say "stage 6 (C5) $v after the bus-bar stage: $2"; else say "bus-bar stage: $1 $v ($2)"; fi
}
fl_bb_ncyc() {  # <ub0_repro.py --cycles result.json>: the cycles whose burst ended (0 if none or unread)
  python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("cycles") or []))' "$1" 2>/dev/null || echo 0
}
fl_bb_cont() {  # <part>: 0 to run it, 1 not to; FL_BB_TRAYS: the trays the part may load ("" = as FLEET_LOAD_TRAYS)
  local p=$1 gone sig why f t lt=""
  FL_BB_TRAYS=""
  gone=$(rk_snap | awk '$3 != "ok" { print $1 }' | sed 's/^[0-9a-fA-F]\{4,\}://' | paste -sd' ' -)   # BDFs in run.sh's short form (41:00.0)
  [ -n "$gone$FL_HUNG$FL_PIN$FL_PGDROP" ] || return 0
  why=$(printf '%s' "${gone:+chips lost: $gone; }${FL_HUNG:+hang; }${FL_PIN:+CCL failure; }${FL_PGDROP:+$FL_PGDROP; }" | sed 's/; $//')
  [ "${HOLD_ACTIVE:-0}" = 1 ] && { fl_bb_notrun "$p" "$why; hold mode keeps the chips as they are" skipped; return 1; }
  sig="$gone|$FL_HUNG|$FL_PIN|$FL_PGDROP"
  if [ "$sig" != "${FL_BB_RECSIG:-}" ]; then
    f="$FL_OUT/busbar/recover_${p// /_}"
    say "bus-bar stage: $why before $p: recovering with glx_reset, root-port rescan and recount"
    RK_STEP="bus-bar stage recovery before $p" rk_reset "$f.log"; FL_BB_RECST=${RK_RESET_STAT:-NONE}
    say "bus-bar stage: recovery before $p: $RK_RESET_LBL $FL_BB_RECST${RK_RESET_WHY:+ ($RK_RESET_WHY)}"
    RK_BEFORE=$(rk_snap); rk_recheck "bus-bar stage recovery before $p" "$f.check"
    RK_RESCANNED=""; RK_RESCAN_RAN=""; RK_RESCAN_GSKIP=""; [ -n "${RK_RESET_SKIP:-}" ] || rk_rescan "bus-bar stage recovery before $p" "$f.rescan"
    gone=$(rk_snap | awk '$3 != "ok" { print $1 }' | sed 's/^[0-9a-fA-F]\{4,\}://' | paste -sd' ' -)   # BDFs in run.sh's short form (41:00.0)
    if [ -n "$FL_HUNG$FL_PIN" ]; then   # the first hang or CCL failure is kept for the run's verdict (restored at the stage end)
      FL_BB_FAULTS="${FL_BB_FAULTS:+$FL_BB_FAULTS; }before $p: ${FL_HUNG:-$FL_PIN}"
      [ -z "$FL_BB_HUNG1$FL_BB_PIN1" ] && { FL_BB_HUNG1=$FL_HUNG; FL_BB_PIN1=$FL_PIN; }
      [ "$FL_BB_RECST" = OK ] && { FL_HUNG=""; FL_PIN=""; }
    fi
    if [ -n "$FL_PGDROP" ]; then   # every fallen tray of the stage, and each fall's line for the verdict (FL_PGDROP at the stage end)
      case "; $FL_BB_PGS; " in *"; $FL_PGDROP; "*) ;; *) FL_BB_PGS="${FL_BB_PGS:+$FL_BB_PGS; }$FL_PGDROP" ;; esac
      t=$(printf '%s' "$FL_PGDROP" | sed -n 's/^vr_pg fell on tray \([0-9,]*\) .*/\1/p')
      FL_BB_PGT=$(printf '%s\n' "$FL_BB_PGT" "$t" | tr , '\n' | grep -E '^[1-4]$' | sort -un | paste -sd, -)
    fi
    FL_BB_RECSIG="$gone|$FL_HUNG|$FL_PIN|$FL_PGDROP"
  fi
  if [ -n "$FL_HUNG$FL_PIN" ]; then
    fl_bb_notrun "$p" "$why; recovery glx_reset $FL_BB_RECST"; return 1; fi
  # chips still missing: no part runs. chunk_load.py opens the full mesh and fl_watch checks every chip, so a part on the
  # other trays would be cut at once and name the missing chips as a new drop
  if [ -n "$gone" ]; then
    fl_bb_notrun "$p" "chips still missing after glx_reset $FL_BB_RECST, rescan and recount: $(for f in $gone; do printf '%s (tray %s) ' "$f" "$(rk_tray "$f")"; done | sed 's/ $//')"; return 1; fi
  if [ -n "$FL_BB_PGT" ]; then
    case $p in
      mix|heavy|long)
        for t in 1 2 3 4; do
          case ",$FL_BB_PGT," in *",$t,"*) continue ;; esac
          [ -z "${FLEET_LOAD_TRAYS:-}" ] || case ",$FLEET_LOAD_TRAYS," in *",$t,"*) ;; *) continue ;; esac
          lt=${lt:+$lt,}$t
        done
        [ -n "$lt" ] || { fl_bb_notrun "$p" "vr_pg fell on tray $FL_BB_PGT: no other tray left to load"; return 1; }
        FL_BB_TRAYS=$lt
        say "bus-bar stage: $p runs on tray $lt only ($why; recovery glx_reset $FL_BB_RECST, all chips back; no more load on tray $FL_BB_PGT, whose vr_pg fell)"; return 0 ;;
      *)
        fl_bb_notrun "$p" "its load reaches every tray, and vr_pg fell on tray $FL_BB_PGT: no more load there"; return 1 ;;
    esac
  fi
  say "bus-bar stage: $p runs after an earlier fault ($why; recovery glx_reset $FL_BB_RECST, all chips back)"
}
fl_bb_t() { echo "$1 $(date +%s)" >> "$FL_OUT/busbar/parts.txt"; }   # a bus-bar part starts ("end": the stage's parts are over)
fl_busbar() {  # every mode, first load after C2 (one try): the bus-bar stage,
  # FLEET_BUSBAR_ORDER (qb,mix,step) under one tray power poll ($FL_OUT/busbar/power.txt): qb = FL_BB_QB_N quiet-to-burst steps
  # (fl_bb_step in busbar/qb<i>), mix = the CCL-mix chunks (fl_c7 in
  # $FL_OUT/busbar, chunk_load.py h3shape parts, one cq, as e9831c5 ran them: every bus-bar drop so far came under the CCL mix,
  # not at peak kW; G3 after chunk 1), step = fl_bb_step. After a loss, vr_pg fall, hang or CCL failure the next part runs once
  # fl_bb_cont has recovered. Never writes a power limit.
  local x mk lt0=${FLEET_LOAD_TRAYS:-}; FL_BB_RECSIG=""; FL_BB_RECST=""; FL_BB_HUNG1=""; FL_BB_PIN1=""; FL_BB_FAULTS=""; FL_BB_PGS=""; FL_BB_PGT=""; FL_BB_TRAYS=""; FL_BB_BACK=0; FL_BB_PART=""; mkdir -p "$FL_OUT/busbar"; : > "$FL_OUT/busbar/named.tsv"; : > "$FL_OUT/busbar/parts.txt"; fl_power_start "$FL_OUT/busbar/power.txt"; fl_aer_start "$FL_OUT/busbar/aer_rate.txt" "bus-bar stage"; pf_start "$FL_OUT/busbar"
  FL_BB_TRY=0; FL_BB_NJ=0; FL_BB_RAN=0; FL_C7_NJ=""   # ub0_repro.py and chunk_load.py tries, of them not judged at device open, other parts that loaded
  for x in ${FL_BB_ORDER//,/ }; do
    by_check; [ -n "${FL_YIELD:-}" ] && { fl_skip "bus-bar stage $x" "$FL_YIELD"; continue; }   # no part starts after the broker yield
    fl_bb_t "$x"   # each part's wall time (fl_report's "Bus-bar part times" line)
    FLEET_LOAD_TRAYS=$lt0; FL_BB_PART=$x; pf_mark "$x"; fl_bb_cont "$x" || continue
    FLEET_LOAD_TRAYS=${FL_BB_TRAYS:-$lt0}   # after a vr_pg fall: the trays whose rails did not fall (fl_bb_cont)
    fl_bb_budget "bus-bar stage $x" || continue
    mk="${FL_MU_SKIP:-}|${FL_MU_INC:-}|${#FL_SKIPPED[@]}"   # tells a skipped maxutil or pv part (it loaded nothing)
    # shellcheck disable=SC2046  # fl_load_args prints the load's args, split on purpose
    case $x in mix) local sh=() st=${FLEET_BUSBAR_SHAPE_TRAY:-2}; [ -n "${FLEET_BUSBAR_SHAPE:-}" ] && {
                      # after a vr_pg fall the shape aims at a tray the mix still loads, never the fallen one
                      [ -n "$FL_BB_TRAYS" ] && case ",$FL_BB_TRAYS," in *",$st,"*) false ;; *) true ;; esac \
                        && { say "bus-bar stage: shape $FLEET_BUSBAR_SHAPE moved from tray $st to tray ${FL_BB_TRAYS%%,*} (the mix loads tray $FL_BB_TRAYS only)"; st=${FL_BB_TRAYS%%,*}; }
                      sh=(--shape "$FLEET_BUSBAR_SHAPE" --shape-tray "$st")
                      say "bus-bar stage: shape $FLEET_BUSBAR_SHAPE aimed at tray $st (drop_shapes.py)"; }
                    fl_bb_mixfit   # the step and the heavy chunk keep their time
                    FL_BB_MIN=$FL_BB_MINC FL_BB_MIX_S=$FL_BB_MIXC fl_c7 busbar "bus-bar stage" $(fl_load_args) --rw-after "$FL_BB_RW" --rw-s "$FL_BB_RW_S" --rw-wall-s "$FL_BB_RW_WALL" --dbi-wall-s "$FL_BB_DBI_WALL" --l1-wall-s "$FL_BB_L1_WALL" $([ "$FLEET" = full ] && echo --l1-full) "${sh[@]}"
                    if [ "${FL_SWAP_S:-0}" -gt 0 ]; then fl_bb_swap; fi ;;   # the trial knob's part, right after the mix
      qb) if [ -n "${FLEET_LOAD_TRAYS:-}" ]; then say "bus-bar stage: qb skipped (FLEET_LOAD_TRAYS=$FLEET_LOAD_TRAYS: its ub0_repro.py bursts load every tray)"; else
            local i=1
            if [ "${TRAY_QB_ONEPROC:-1}" = 1 ] && [ "$FL_BB_QB_N" -gt 1 ] && { by_check; [ -z "${FL_YIELD:-}" ]; }; then   # all FL_BB_QB_N in one process (one mesh open); TRAY_QB_ONEPROC=0: one process each
              fl_bb_step "bus-bar stage quiet-burst 1-$FL_BB_QB_N/$FL_BB_QB_N" qb1 "$FL_BB_QB_Q" "$FL_BB_QB_B" 0 "$FL_BB_QB_N"
              # a cycle that ended the process (a loss, a hang, a failure) stands for its own step: the ones after it run as
              # their own steps, as with TRAY_QB_ONEPROC=0 (fl_bb_cont first)
              i=$(( $(fl_bb_ncyc "$FL_OUT/busbar/qb1/result.json") + 2 ))
            fi
            for (( ; i <= FL_BB_QB_N; i++ )); do
              by_check; [ -n "${FL_YIELD:-}" ] && { fl_skip "bus-bar stage quiet-burst $i-$FL_BB_QB_N/$FL_BB_QB_N" "$FL_YIELD"; break; }
              [ "$i" -gt 1 ] && { fl_bb_t "qb$i"; FL_BB_PART=qb$i; ! fl_bb_cont "qb$i"; } && continue
              [ "$i" -gt 1 ] && ! fl_bb_budget "bus-bar stage quiet-burst $i-$FL_BB_QB_N/$FL_BB_QB_N" && break
              fl_bb_step "bus-bar stage quiet-burst $i/$FL_BB_QB_N" "qb$i" "$FL_BB_QB_Q" "$FL_BB_QB_B" 0
            done; fi ;;
      step) if [ -n "${FLEET_LOAD_TRAYS:-}" ]; then say "bus-bar stage: step skipped (FLEET_LOAD_TRAYS=$FLEET_LOAD_TRAYS: its ub0_repro.py burst loads every tray)"; else fl_bb_step; fi ;;
      heavy) mkdir -p "$FL_OUT/busbar/heavy"; ln -sfn ../power.txt "$FL_OUT/busbar/heavy/power.txt"   # the stage's one power poll
             fl_bb_hvfit || continue   # cut to end before stage 6's and C9's reserve
             FL_BB_HV=$FL_BB_HVC FL_BB_HEAVY_S=$(( FL_BB_HEAVY_S - FL_BB_HV + FL_BB_HVC )) fl_c7 busbar/heavy "bus-bar stage" $(fl_load_args heavy) --heavy "$FL_DIDT_OPS" --sfpu-par 0 --heavy-check-each 1 --sfpu-vote 1 --heavy-x "$FL_DIDT_X_OPS" --heavy-x-s "$FL_DIDT_X_S" ;;   # the DIDT ops, numeric; every output checked; SFPU vote; N5: extra formats in the last FL_DIDT_X_S s
      long) mkdir -p "$FL_OUT/busbar/long"; ln -sfn ../power.txt "$FL_OUT/busbar/long/power.txt"
            fl_c7 busbar/long "bus-bar stage" $(fl_load_args) ;;
      rw) fl_bb_rw ;; code) fl_bb_code ;; maxutil) fl_bb_mu ;;
      pv) if [ "${FL_PV_SWAP:-}" = maxutil ]; then FL_MU_NM="bus-bar stage max-util in place of the power virus (FLEET_PV_SWAP=maxutil)" FL_MU_OUT=$FL_OUT/busbar/maxutil-pv fl_bb_mu
          else fl_bb_pv; fi ;; esac
    case $x in maxutil|pv) [ "$mk" != "${FL_MU_SKIP:-}|${FL_MU_INC:-}|${#FL_SKIPPED[@]}" ] && continue ;; esac   # skipped: it loaded nothing
    xk=$x; [ "$x" = pv ] && [ "${FL_PV_SWAP:-}" = maxutil ] && xk=maxutil   # a swapped pv ran max-util and counts as it
    # a mix, heavy or long chunk (fl_c7), the rw part and the max-util part count as tries; one that failed
    # at device open on off-box links only is not judged. The power virus (pv, not swapped) runs run-ttx over PCIe with no tt-metal ETH
    # open, so it cannot fail that way
    case $xk in mix|heavy|long|rw|maxutil) FL_BB_TRY=$(( FL_BB_TRY + 1 )); [ -n "$FL_C7_NJ" ] && { FL_BB_NJ=$(( FL_BB_NJ + 1 )); continue; } ;; esac
    case $xk in mix|heavy|long|rw|maxutil|pv) FL_BB_RAN=$(( FL_BB_RAN + 1 )) ;; esac   # its own failure is a FAIL or WARN line (fl_find)
  done
  fl_bb_t end; FLEET_LOAD_TRAYS=$lt0; FL_BB_PART=""   # stage 6 and later stages report on their own
  [ -n "${FL_MU_SKIP:-}" ] && say "bus-bar stage max-util: $FL_MU_SKIP"   # where the part would have run, last in the stage
  # a fault in the stage's last part gets the same one recovery, so stage 6 runs once every chip is back
  # (FL_BB_BACK, fl_loads); the faults stay for the verdict
  [ "${FL_C5_PEND:-0}" = 1 ] && fl_bb_cont "stage 6" && FL_BB_BACK=1
  [[ "$FL_BB_PGS" == *"; "* ]] && FL_PGDROP=$FL_BB_PGS   # every vr_pg fall of the stage, for the verdict
  [ -n "$FL_BB_HUNG1$FL_BB_PIN1" ] && { FL_HUNG=${FL_BB_HUNG1:-$FL_HUNG}; FL_PIN=${FL_BB_PIN1:-$FL_PIN}; say "bus-bar stage: faults in the stage: $FL_BB_FAULTS"; }
  # no part opened its mesh and every ub0_repro.py try failed at device open on off-box links only
  # (peer host down or resetting, judged from this host's own chips): INCOMPLETE, never a pass or a fault; the off-box WARN stays
  if [ "$FL_BB_NJ" -gt 0 ] && [ "$FL_BB_NJ" = "$FL_BB_TRY" ] && [ "$FL_BB_RAN" = 0 ]; then
    FL_BB_INC="bus-bar stage not tested: every chunk failed at device open ($FL_BB_NJ of $FL_BB_TRY, only off-box links with the peer host down or resetting; see $FL_OUT/busbar)"
    fl_skip "$(fl_pn busbar)" "INCOMPLETE: $FL_BB_INC"; fi
  fl_power_stop; fl_aer_stop; pf_stop   # the plug side over the whole stage: peak, share and each onset
  while IFS= read -r x; do say "$x"; echo "$x" >> "$FL_OUT/load.txt"; done < <(pf_report "$FL_OUT/busbar" "bus-bar stage" "$FL_OUT/busbar/power.txt")
}
# In earlier plain runs, bus-bar quiet-bursts each sat ~304 s in set_fabric_config (tt-umd "Still waiting
# on ARC firmware"; healthy: 3-4 s), and no budget check ran inside the stage. Now each part and each
# quiet-burst after the first starts only while rs_budget allows, and its try limit is cut to what is left before the C9 reserve.
# a try whose set_fabric_config took FLEET_FAB_STALL_S (120 s) or more (or never
# returned; seen on three boxes, cause unknown) gives one FABRIC WARN line per chip tt-umd named (BDF, tray 1-4) and
# extends the run's budget by the stall; a failed try is retried once per run (fl_bb_step), and no later part or stage is
# skipped for the stall. The stall alone makes no INCOMPLETE RESULT.
fl_bb_budget() { return 0; }  # $1 part: always starts (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
fl_widen() { return 0; }  # no run budget to widen (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
fl_keep() {  # what stage 4's parts leave free: rs_keep, plus RS_C5_S for stage 6 pair link test while it is still to run (FL_C5_PEND,
  # fl_loads). With C2 at 146 s (a quad host) the heavy chunk started 502 s before the end, kept
  # only C9's share and the RESULT, and stage 6 had 54 s: "LINKS WARN C5 budget spent", 32 pairs not run
  local k; k=$(rs_keep); [ "${FL_C5_PEND:-0}" = 1 ] && k=$(( k + ${RS_C5_S:-120} )); echo "$k"; }
fl_keep_why() { if [ "${FL_C5_PEND:-0}" = 1 ]; then echo "${RS_C5_S:-120} s kept for stage 6 pair link test, $(rs_keep_why)"; else rs_keep_why; fi; }
fl_bb_hvfit() { FL_BB_HVC=$FL_BB_HV; }  # the heavy chunk keeps its whole length (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
fl_bb_mixfit() { FL_BB_MINC=$FL_BB_MIN; FL_BB_MIXC=$FL_BB_MIX_S; }  # the mix chunks keep their plan (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
fl_try_cap() { echo "$1"; }  # $1 a try's own hang timeout, unchanged (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
fl_try_need() { return 0; }  # (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
fl_try_fits() { return 0; }  # $1 step: always runs (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
fl_fabstall() {  # $1 a try's log, $2 the part, $3 the step's start (epoch s; waits count from it, not from older log stamps): once its last set_fabric_config took FLEET_FAB_STALL_S or more
  # (or never returned): one FABRIC WARN per chip tt-umd named (BDF, tray 1-4), FL_FABSTALL and FL_FAB_HIT set, and the run's budget extended by the stall
  local log=$1 nm=$2 c r s b n m l ch="" nr=""; FL_FAB_HIT=""
  c=$(grep -a 'set_fabric_config(.*) called' "$log" | tail -1 | cut -d'|' -f2); [ -n "$c" ] || return 0
  r=$(grep -a 'set_fabric_config(.*) returned' "$log" | tail -1 | cut -d'|' -f2); [ -n "$r" ] || nr=", never returned"
  c=$(date -d "$c" +%s 2> /dev/null) || return 0; r=$( [ -n "$r" ] && date -d "$r" +%s 2> /dev/null || date +%s)
  [ -n "${3:-}" ] && [ "$c" -lt "$3" ] && c=$3
  s=$(( r - c )); [ "$s" -ge "${FLEET_FAB_STALL_S:-120}" ] || return 0
  while IFS=$'\t' read -r n m; do   # "device N" in tt-umd's lines: its last warning/error/wait line per chip, as fl_c3stop
    b=$(echo "$RK_MAP" | awk -v n="$n" '$2 == n { print $1; exit }'); [ -n "$b" ] || continue
    l="$(fl_lk "$b") chip $n"; ch="${ch:+$ch; }$l"
    fl_find "FABRIC WARN $l: set_fabric_config waited $s s in the $nm$nr (healthy: under 10 s): $m ($log)"
  done < <(grep -aiE 'device [0-9]+' "$log" | grep -aiE 'warn|error|still waiting|timed? ?out|fail' |
    awk '{ if (!match($0, /[Dd]evice [0-9]+/) || substr($0, RSTART + RLENGTH) ~ /^ *(min|sec|s|ms|h|hours?|minutes?|seconds?)( |,|$)/) next; n = substr($0, RSTART + 7, RLENGTH - 7); m = $0
      sub(/.*\| */, "", m); last[n] = substr(m, 1, 160); if (!(n in seen)) { seen[n] = 1; o[++k] = n } }
      END { for (i = 1; i <= k; i++) print o[i] "\t" last[o[i]] }')
  [ -n "$ch" ] || fl_find "FABRIC WARN: set_fabric_config waited $s s in the $nm$nr (healthy: under 10 s); tt-umd named no chip ($log)"
  FL_FABSTALL="set_fabric_config waited $s s in the $nm$nr (chips: ${ch:-none named})"; FL_FAB_HIT=1
  say "$FL_FABSTALL: every later stage still runs"   # no run budget gates a later stage
}
fl_bb_rw() {  # the bus-bar stage's rw part (opt-in, repro.sh's dram way): C4d shape, numeric_dram.py
  # on its own mesh, FL_BB_RW_S s of x/w/y-in-DRAM C4n-size matmul on every chip, read back every FL_BB_RW_CK s of
  # load and compared with the majority (03:00.0 tray 1: 4/4 at 0-30 s of load). Its NUMERIC FAIL lines name the chip by
  # BDF and tray with DRAM banks and lanes; a loss is named by tray and BDF. Never writes a power limit
  local nm="stage 5 numeric verify (G3 readback)" d=$FL_OUT/busbar/rw t0 el lost nb l ub g bd pw pg v pb pf=$FL_PW_FILE
  mkdir -p "$d"; t0=$(date +%s); rm -f "$d/numeric.txt"; FL_C7_NJ=""   # set when it failed at device open on off-box links only
  say "$nm: $FL_BB_RW_S s of x/w/y-in-DRAM matmul on all chips (numeric_dram.py), a readback every $FL_BB_RW_CK s of load, wall cap $FL_BB_RW_WALL s"
  hb_start "$nm" "$d/run.log"
  TRY_LIMIT_S=$(( FL_BB_RWP_S + 240 )) TRY_LOSS_S=${FLEET_BUSBAR_LOSS_S:-60} try_run busbar "$d/run.log" "${AS[@]}" "$PY" numeric_dram.py --out "$d/numeric.txt" \
    --burst-s "$FL_BB_RW_S" --wall-s "$FL_BB_RW_WALL" --per-check 8 --check-s "$FL_BB_RW_CK"; hb_stop
  el=$(( $(date +%s) - t0 ))
  nb=$(grep -aE '^NUMERIC (FAIL|WARN) ' "$d/numeric.txt" 2>/dev/null)
  [ -n "$nb" ] && while IFS= read -r l; do fl_find "$l (${nm/#bus-bar stage /stage 5 numeric verify, after the })"; done <<< "$nb"
  l=$(grep -a '^C4d numeric:' "$d/numeric.txt" 2>/dev/null); [ -z "$l" ] || say "$l ($nm, $el s)"
  pg=""; [ -n "$pf" ] && pg=$(fl_pg_fell "$pf" < /dev/null | awk -v t="$t0" '$1 >= t')   # this part's vr_pg falls only (the stage's one power.txt), as fl_bb_step
  lost=$(rk_snap | awk '$3 != "ok"' | while read -r bb c st; do echo "$(rk_tray "$bb") ${bb#0000:}"; done)
  if [ -n "$lost$pg" ]; then
    fl_lost "$nm at $el s" > /dev/null
    [ -n "$pg" ] && FL_PGDROP="vr_pg fell on tray $(echo "$pg" | awk '{ print $2 }' | sort -u | paste -sd,) in the $nm"
    pw=$(fl_pw_says "$(awk 'NF >= 10 { l = $0 } END { print l }' "$pf" 2>/dev/null < /dev/null)")
    for ub in $( { echo "$lost" | awk 'NF { print $1 }'; echo "$pg" | awk 'NF { print $2 }'; } | sort -u); do
      g=$(echo "$lost" | awk -v u="$ub" '$1 == u' | grep -c .); bd=$(echo "$lost" | awk -v u="$ub" '$1 == u { printf "%s%s", (n++ ? " " : ""), $2 }')
      v=$(echo "$pg" | awk -v u="$ub" -v t="$t0" '$2 == u { printf "vr_pg %s at %d s into the rw part", $3, $1 - t }')
      pb=""; [ -n "$v" ] && pb=$(fl_pw_says "$(fl_pw_at "$pf" "$(echo "$pg" | awk -v u="$ub" '$2 == u { print $1 - 1 }')")")
      fl_find "$([ "$g" -ge 8 ] || [ -n "$v" ] && echo "POWER DROP" || echo "CHIP DROP") tray $ub ($g chips lost${bd:+: $bd}${v:+; $v}) in the $nm ($FL_BB_RW_S s x/w/y-in-DRAM load), BMC after: ${pw:-unread}${pb:+; before the fall: $pb}"
    done
    return 0
  fi
  if [ -n "${TRY_INFRA:-}" ]; then   # could not start, test-infra (INCOMPLETE), not a FAIL
    FL_INFRA_INC="${FL_INFRA_INC:+$FL_INFRA_INC; }$nm: $TRY_INFRA"; fl_skip "$nm" "NOT_RUN, WARN $TRY_INFRA (see $d/run.log)"
  # an open that timed out only on off-box links with the peer host down or resetting, before its mesh opened, is
  # not judged (fl_nj_open, as the fl_c7 chunks)
  elif [ "$rc" != 0 ] && [ -z "$nb" ] && fl_nj_open "$nm" "$(grep -aq ' C4D mesh open ' "$d/run.log" 2>/dev/null && echo 1 || echo 0)" "$d/run.log"; then
    FL_C7_NJ=1; say "$nm: numeric_dram.py rc $rc at device open, only off-box links with the peer host down or resetting (not judged; see $d/run.log)"
  elif [ "$rc" != 0 ]; then
    fl_find "$nm FAIL: numeric_dram.py rc $rc${TRY_HUNG:+, $TRY_HUNG} (see $d/run.log)"
  elif [ -z "$l" ]; then fl_find "NUMERIC WARN $nm: numeric_dram.py wrote no result (see $d/run.log)"
  else [ -n "$nb" ] && say "$nm: numeric mismatch ($el s)" || say "$nm: clean ($el s)"; fi
}
fl_bb_code() {  # the bus-bar stage's code readback part (goal 54 stage 2 N3): lib/exalens_probe.py --code with no
  # load on the box. Its CODE FAIL lines name the chip by BDF, tray, core, RISC and region; a read timeout or a missing exalens is a
  # WARNING (never a FAIL, never a quiet skip); an exalens init error makes the part not run (fl_skip). Single box only
  local nm="bus-bar stage code readback" d=$FL_OUT/busbar/code t0 el l t w x vp rc=0
  mkdir -p "$d"; t0=$(date +%s)
  vp=$(type tools_venv_python > /dev/null 2>&1 && tools_venv_python "$PY") || vp=""
  [ -n "$vp" ] || { fl_skip "$nm" "WARNING: no python with ttexalens on this box"; return 0; }
  say "$nm: $FL_BB_CODE_B s budget ($FL_BB_CODE_SCOPE), firmware and kernel code in L1 of every chip by majority vote${FLEET_CODE_ELF_DIR:+ and ELF compare}"
  hb_start "$nm" "$d/run.log"
  TRY_LIMIT_S=$(( FL_BB_CODE_S + 120 )) try_run busbar "$d/run.log" "${AS[@]}" "$vp" "$(dirname "$FL_DIR")/lib/exalens_probe.py" --code \
    --budget-s "$FL_BB_CODE_B" --scope "$FL_BB_CODE_SCOPE" --cursor-file "${FLEET_CODE_CURSOR:-$(dirname "$FL_OUT")/code-cursor.json}" \
    ${FLEET_CODE_ELF_DIR:+--elf-dir "$FLEET_CODE_ELF_DIR"} --out "$d/result.json" || rc=$?; hb_stop
  el=$(( $(date +%s) - t0 ))
  l=$(grep -a '^CODE ERROR ' "$d/run.log" 2>/dev/null | head -1)
  [ -n "$l" ] && { fl_skip "$nm" "WARNING: ${l#CODE ERROR } (see $d/run.log)"; return 0; }
  # "CODE FAIL <bdf> tray <n> core x,y RISC region addr ...": the probe names the tray ('chip<id> tray ?' with no BDF)
  grep -a '^CODE FAIL ' "$d/run.log" 2>/dev/null | while read -r _ _ x _ t w; do fl_find "CODE FAIL ${x#0000:} tray $t: $w (the $nm, see $d/run.log)"; done
  grep -a '^CODE WARN ' "$d/run.log" 2>/dev/null | head -5 | while read -r _ _ x _ t w; do say "$nm WARNING: ${x#0000:} tray $t: $w"; done
  l=$(grep -a '^CODE SUMMARY ' "$d/run.log" 2>/dev/null | tail -1)
  if [ -z "$l" ]; then say "$nm WARNING: no summary line, rc $rc${TRY_HUNG:+, $TRY_HUNG} (see $d/run.log)"
  else say "${l} ($nm, rc $rc, $el s)"; fi
}
fl_bb_mu() {  # the bus-bar stage's max-util part: tt-metal's unit_tests_didt_max_util from the prebuilt
  # archive that fleet/maxutil.pin names, FPU 92 % DRAM 92 % duty 100 % on every chip, FL_BB_MU_LOAD s of load (plain and full one 240 s burst;
  # smoke two FLEET_SMOKE_S bursts with as much idle between) after its setup. A setup error (no archive, a bad
  # sha) makes the run INCOMPLETE (FL_MU_INC), never a quiet skip. Its MAXUTIL FAIL lines name the chip by BDF and tray; a
  # loss or vr_pg fall is named by tray and BDF, as the rw part. Never writes a power limit (maxutil.py drops TT_METAL_TDP_LIMIT_WATTS)
  local nm="${FL_MU_NM:-bus-bar stage max-util}" d=${FL_MU_OUT:-$FL_OUT/busbar/maxutil} t0 el lost l ub g bd pw pg v pb c w b pf=$FL_PW_FILE
  mkdir -p "$d"; t0=$(date +%s); FL_C7_NJ=""   # set when it failed at device open on off-box links only
  say "$nm: $FL_BB_MU_LOAD s of unit_tests_didt_max_util on all chips ($FL_BB_MU_MODE), after about $FL_BB_MU_SETUP s of setup"
  hb_start "$nm" "$d/run.log"
  TRY_LIMIT_S=$(( FL_BB_MU_S + 240 )) TRY_LOSS_S=${FLEET_BUSBAR_LOSS_S:-60} try_run busbar "$d/run.log" "${AS[@]}" "$PY" "$FL_DIR/maxutil.py" run \
    --mode "$FL_BB_MU_MODE" --smoke-s "${FLEET_SMOKE_S:-2}" --load-s "${FL_MU_LOAD_S:-$FL_BB_MU_ARG}" --limit-s $(( FL_BB_MU_S + 120 )) --log "$d/test.log" --result "$d/result.json"; hb_stop
  el=$(( $(date +%s) - t0 ))
  # setup failure only by its line: rc 3 also comes from try_run's FIRST LOSS end and the test's own exit 3
  local se; se=$(grep -a '^MAXUTIL ERROR setup' "$d/run.log" 2>/dev/null | head -1)
  # a binary that cannot load (maxutil.py's NOT_RUN line, or try_run's TRY_INFRA: rc 126/127, a loader error) is a
  # test-infra WARN and the part NOT_RUN (INCOMPLETE, as a setup error), never a chip FAIL; a loss still wins
  local ni; ni=$(grep -a '^MAXUTIL NOT_RUN ' "$d/run.log" 2>/dev/null | head -1); ni=${ni#MAXUTIL NOT_RUN }
  [ -z "$ni" ] && ni=${TRY_INFRA:-}
  grep -a '^MAXUTIL WARN test-infra' "$d/run.log" 2>/dev/null | while read -r w; do say "WARNING ${w#MAXUTIL WARN }"; done
  pg=""; [ -n "$pf" ] && pg=$(fl_pg_fell "$pf" < /dev/null | awk -v t="$t0" '$1 >= t')   # this part's vr_pg falls only, as fl_bb_rw
  lost=$(rk_snap | awk '$3 != "ok"' | while read -r bb c st; do echo "$(rk_tray "$bb") ${bb#0000:}"; done)
  # no loss, no setup error and an open that timed out only on off-box links with the peer host down or resetting,
  # before its program was ready: not judged (fl_nj_open, as the fl_c7 chunks); maxutil.py's no-chip-passed MAXUTIL FAIL is not filed
  if [ -z "$lost$pg" ] && [ -z "$se" ] && [ "$rc" != 0 ] \
    && fl_nj_open "$nm" "$(grep -aq 'Main program ready\|Iteration ' "$d/test.log" 2>/dev/null && echo 1 || echo 0)" "$d/run.log" "$d/test.log"; then
    FL_C7_NJ=1; say "$nm: maxutil.py rc $rc at device open, only off-box links with the peer host down or resetting (not judged; see $d/run.log)"; return 0
  fi
  l=$(grep -a '^MAXUTIL FAIL ' "$d/run.log" 2>/dev/null)
  [ -n "$l" ] && [ -z "$ni" ] && while read -r _ _ c w; do   # "<bdf> tray <n> (chip <kmd>)"; a chip not in RK_MAP (or "-") keeps its number
    b=$(echo "$RK_MAP" | awk -v c="$c" '$2 == c { print $1; exit }')
    if [ -n "$b" ]; then b="${b#0000:} tray $(rk_tray "$b") (chip $c)"
    elif [ "$c" = - ] && [[ $w == "its own limit "* ]]; then b="limit timeout, no chip named"   # --limit-s kill, not a chip fault
    else b="chip $c"; fi
    fl_find "MAXUTIL FAIL $b: $w (the $nm, see $d/run.log)"
  done <<< "$l"
  if [ -n "$lost$pg" ]; then
    fl_lost "$nm at $el s" > /dev/null
    [ -n "$pg" ] && FL_PGDROP="vr_pg fell on tray $(echo "$pg" | awk '{ print $2 }' | sort -u | paste -sd,) in the $nm"
    pw=$(fl_pw_says "$(awk 'NF >= 10 { l = $0 } END { print l }' "$pf" 2>/dev/null < /dev/null)")
    for ub in $( { echo "$lost" | awk 'NF { print $1 }'; echo "$pg" | awk 'NF { print $2 }'; } | sort -u); do
      g=$(echo "$lost" | awk -v u="$ub" '$1 == u' | grep -c .); bd=$(echo "$lost" | awk -v u="$ub" '$1 == u { printf "%s%s", (n++ ? " " : ""), $2 }')
      v=$(echo "$pg" | awk -v u="$ub" -v t="$t0" '$2 == u { printf "vr_pg %s at %d s into the max-util part", $3, $1 - t }')
      pb=""; [ -n "$v" ] && pb=$(fl_pw_says "$(fl_pw_at "$pf" "$(echo "$pg" | awk -v u="$ub" '$2 == u { print $1 - 1 }')")")
      fl_find "$([ "$g" -ge 8 ] || [ -n "$v" ] && echo "POWER DROP" || echo "CHIP DROP") tray $ub ($g chips lost${bd:+: $bd}${v:+; $v}) in the $nm ($FL_BB_MU_LOAD s max-util load), BMC after: ${pw:-unread}${pb:+; before the fall: $pb}"
    done
    return 0
  fi
  if [ -n "$ni" ]; then
    FL_MU_INC="$ni"; fl_skip "$nm" "NOT_RUN, WARN $ni (see $d/run.log)"; return 0
  fi
  if [ -n "$se" ]; then
    FL_MU_INC="max-util setup failed: ${se#MAXUTIL ERROR setup: }"; FL_MU_INC=${FL_MU_INC%: }
    fl_skip "$nm" "INCOMPLETE: $se (see $d/run.log)"; return 0
  fi
  # a load cut by TERM or INT (rc 143/130 with maxutil.py's stopped line): INCOMPLETE, no verdict
  se=$(grep -a '^MAXUTIL INCOMPLETE - stopped' "$d/run.log" 2>/dev/null | tail -1)
  if [ -n "$se" ] && { [ "$rc" = 143 ] || [ "$rc" = 130 ]; }; then
    FL_MU_INC="max-util ${se#MAXUTIL INCOMPLETE - }"
    fl_skip "$nm" "INCOMPLETE: $se (see $d/run.log)"; return 0
  fi
  l=$(grep -a '^MAXUTIL [0-9?]*/[0-9?]* chips passed' "$d/run.log" 2>/dev/null | tail -1)
  if [ "$rc" != 0 ] && ! grep -aq '^MAXUTIL FAIL ' "$d/run.log" 2>/dev/null; then
    fl_find "$nm FAIL: maxutil.py rc $rc${TRY_HUNG:+, $TRY_HUNG} (see $d/run.log)"
  else say "${l:-MAXUTIL no summary line} ($nm, rc $rc, $el s)"; fi
}
fl_bb_swap() {  # TRAYTEST_MIX_SWAP's part (a trial), FL_SWAP_S s cut from the mix's
  # last chunk. heavy: one heavy chunk of the DIDT ops (fl_c7's heavy path). virus: FL_SWAP_SEGS, max-util alternating with the
  # power virus (fl_bb_mu, fl_bb_pv and its shared glx_reset). A loss, vr_pg fall or hang ends it. A dev trial knob only.
  local x n i=0 d=$FL_OUT/busbar/swap/$FL_MIX_SWAP
  say "bus-bar stage swap: $FL_SWAP_WARN"; mkdir -p "$d"
  if [ "$FL_MIX_SWAP" = heavy ]; then ln -sfn ../../power.txt "$d/power.txt"
    # shellcheck disable=SC2046  # fl_load_args prints the load's args, split on purpose
    FL_BB_HV=$FL_SWAP_S FL_BB_HEAVY_S=$FL_SWAP_PART_S fl_c7 busbar/swap/heavy "bus-bar stage" $(fl_load_args heavy) --heavy "$FL_DIDT_OPS" --sfpu-par 0 --heavy-check-each 1
    return 0; fi
  for x in $FL_SWAP_SEGS; do i=$(( i + 1 )); n=${x#*:}
    [ -n "$(rk_snap | awk '$3 != "ok"')$FL_HUNG$FL_PIN$FL_PGDROP" ] && { say "bus-bar stage swap: segment $i ($x) skipped (a loss, vr_pg fall, hang or CCL failure earlier in the stage)"; continue; }
    if [ "${x%%:*}" = pv ]; then
      FL_BB_PV_LOAD=$n FL_BB_PV_S=$(( n + FL_BB_PV_SETUP + 20 + FL_BB_PV_RST )) FL_PV_OUT=$d/pv$i \
        FL_PV_NM="bus-bar stage swap power virus $i (WARNING: TRAYTEST_MIX_SWAP=virus)" fl_bb_pv
    else
      FL_BB_MU_LOAD=$n FL_BB_MU_S=$(( n + FL_BB_MU_SETUP + FL_BB_MU_MARGIN )) FL_MU_LOAD_S=$n FL_MU_OUT=$d/mu$i \
        FL_MU_NM="bus-bar stage swap max-util $i (WARNING: TRAYTEST_MIX_SWAP=virus)" fl_bb_mu
    fi
  done
}
fl_pv_brick_start() {  # <file>: read-only TEMP_PWRBRK* temperatures ("epoch name C") every FLEET_PV_BRICK_POLL_S, until fl_pv_brick_stop
  #: ipmitool sdr type Temperature through the HSC route (root, or sudo -n where sudo -n -l allows that exact read)
  local ip=() main=$$; FL_PV_BRICK_PID=""; FL_PV_BRICK_WHY=""
  if [ ${#HSC_IP[@]} = 0 ] && ! hsc_route; then FL_PV_BRICK_WHY="no ipmitool route ($HSC_WHY)"; return 1; fi
  ip=("${HSC_IP[@]}")
  if [ "${ip[0]}" = sudo ] && ! timeout -k 2 5 sudo -n -l ipmitool sdr type Temperature > /dev/null 2>&1; then
    FL_PV_BRICK_WHY="sudo -n -l refuses ipmitool sdr type Temperature"; return 1; fi
  ( trap - USR1 INT TERM EXIT
    while kill -0 "$main" 2> /dev/null; do
      hsc_i2c timeout -k 2 20 "${ip[@]}" sdr type Temperature 2> /dev/null < /dev/null \
        | awk -F'|' -v e="$(date +%s)" '$1 ~ /PWRBRK/ && match($5, /[0-9.]+ degrees C/) { n = $1; gsub(/ /, "", n); print e, n, substr($5, RSTART, RLENGTH - 10) + 0 }'
      sleep "${FLEET_PV_BRICK_POLL_S:-10}"; done ) < /dev/null >> "$1" 2> /dev/null &
  FL_PV_BRICK_PID=$!
}
fl_pv_brick_stop() { [ -n "${FL_PV_BRICK_PID:-}" ] || return 0; kill "$FL_PV_BRICK_PID" 2> /dev/null; wait "$FL_PV_BRICK_PID" 2> /dev/null; FL_PV_BRICK_PID=""; }
fl_pv_psu_start() {  # <file>: the BMC's per-PSU W ("epoch name W") as fast as psu_read answers, every FLEET_PV_PSU_POLL_S (1),
  # until fl_pv_psu_stop (the fast PSU reads pv.py streams to disk). No read: one "# no PSU read: <why>" line, then it ends
  local main=$$; FL_PV_PSU_PID=""
  ( trap - USR1 INT TERM EXIT; local r=$1.r w
    while kill -0 "$main" 2> /dev/null; do
      w=$(psu_read "$r") || { echo "# no PSU read: $w"; break; }
      awk -F'|' -v e="$(date +%s)" 'NF >= 2 && $2 ~ /[0-9]/ { n = $1; gsub(/ /, "", n); v = $2; gsub(/ /, "", v); print e, n, v; fflush() }' "$r.watts"
      sleep "${FLEET_PV_PSU_POLL_S:-1}"; done; rm -f "$r".* ) < /dev/null >> "$1" 2> /dev/null &
  FL_PV_PSU_PID=$!
}
fl_pv_psu_stop() { [ -n "${FL_PV_PSU_PID:-}" ] || return 0; kill "$FL_PV_PSU_PID" 2> /dev/null; wait "$FL_PV_PSU_PID" 2> /dev/null; FL_PV_PSU_PID=""; }
fl_bb_pv() {  # the bus-bar stage's power virus part: fleet/pv.py, run-ttx of the power virus
  # on every chip for FL_BB_PV_LOAD s. A load only (no numeric check); --tdp does not cap it, so pv.py stops it early on a brick
  # >= FL_PV_BRICK_C C or a feed > FL_PV_FEED_A A (the busbar leg's HSC sampler); the chips' power is never a stop.
  # No other stop, cap or skip: the full FL_BB_PV_LOAD always runs. The HSC, brick, telemetry
  # and PSU reads stream to $d/stream (fsync'd each second) and pvc_open marks the leg open, so a host that dies under it is
  # this box's FAIL on the next run (pvc_check: stage, seconds into the leg, BERT/MCE)
  # Then tt-smi -glx_reset with its rescan and recount (never tt-smi -r), so the virus kernels end and later steps open clean;
  # where glx_reset cannot run it says so plainly. A chip missing after it is a POWER DROP or CHIP DROP by tray and BDF
  local nm="${FL_PV_NM:-bus-bar stage power virus}" d=${FL_PV_OUT:-$FL_OUT/busbar/pv} t0 el l af lost all ub g ga bd hf="" pg st
  mkdir -p "$d"; t0=$(date +%s); : > "$d/brick.txt"; : > "$d/psu.txt"; pvc_open "$nm load" "$d" "$FL_BB_PV_LOAD"
  fl_pv_psu_start "$d/psu.txt"
  [ -n "${HSC_LEG:-}" ] && [ -n "${HSC_POLL_PID:-}" ] && hf="$(fl_hsc_f "$HSC_LEG").vin"
  fl_pv_brick_start "$d/brick.txt" || say "$nm: no brick temperature reads ($FL_PV_BRICK_WHY); it stops on feeds where the HSC reads work, else it runs with no stop (chips' power is not a stop)"
  [ -n "$hf" ] || say "$nm: no HSC feed reads in this leg; it stops on bricks where read, else it runs with no stop (chips' power is not a stop)"
  say "$nm: $FL_BB_PV_LOAD s of run-ttx single-core-matrix-pv-fp32-noc.ttx on all chips (a load only, no numeric check; --tdp does not cap it)"
  hb_start "$nm" "$d/run.log"
  TRY_LIMIT_S=$(( FL_BB_PV_S + 120 )) TRY_LOSS_S=${FLEET_BUSBAR_LOSS_S:-60} try_run busbar "$d/run.log" "$PY" "$FL_DIR/pv.py" \
    --dir "${FLEET_PV_DIR:-$FL_DIR/pv}" --load-s "$FL_BB_PV_LOAD" --limit-s $(( FL_BB_PV_S + 60 )) --hsc "$hf" --feed-a "$FL_PV_FEED_A" \
    --brick "$d/brick.txt" --brick-c "$FL_PV_BRICK_C" --telem "$FL_OUT/c8_telemetry.txt" \
    --psu "$d/psu.txt" --stream "$d/stream" --scratch "$d/tmp" --pgids "$d/pgids" --log-dir "$d/chips"; hb_stop; fl_pv_brick_stop; fl_pv_psu_stop
  pvc_stage "glx_reset after the $nm"
  grep -a '^# no PSU read' "$d/psu.txt" 2> /dev/null | head -1 | while read -r l; do say "$nm: ${l#\# }; the other reads still streamed to $d/stream"; done
  # pv.py killed before its own cleanup: end any of our own run-ttx process groups still there (by pgid, from its own list),
  # only while the group leader still has the start time pv.py recorded: a reused pgid is someone else's
  if [ -s "$d/pgids" ]; then while read -r pg st; do [[ $pg =~ ^[0-9]+$ && ${st:-0} =~ ^[1-9][0-9]*$ ]] && kill -0 -- "-$pg" 2> /dev/null \
    && [ "$(pg_start "$pg")" = "$st" ] \
    && { say "$nm: run-ttx group $pg still running after pv.py: SIGKILL (our own)"; kill -KILL -- "-$pg" 2> /dev/null; }; done < "$d/pgids"; fi
  [ -n "${d:?}" ] && rm -rf "${d:?}/tmp"
  el=$(( $(date +%s) - t0 )); grep -a '^PV \(STOP\|WARN\|LEFT\|LOAD done\)' "$d/run.log" 2> /dev/null | while read -r l; do
    [[ $l =~ ^PV\ WARN\ chip\ [0-9]+\ \(([0-9a-f]{4}:[0-9a-f:.]+)\) ]] \
      && l=${l/ (${BASH_REMATCH[1]})/ (${BASH_REMATCH[1]#0000:}, tray $(rk_tray "${BASH_REMATCH[1]}" | cut -d" " -f1))}
    say "$nm: $l"; done
  if grep -aq '^PV ERROR setup' "$d/run.log" 2> /dev/null; then
    l=$(grep -a '^PV ERROR setup' "$d/run.log" | head -1); FL_PV_INC="power virus setup failed: ${l#PV ERROR setup: }"
    fl_skip "$nm" "INCOMPLETE: $l (see $d/run.log)"
  fi
  # pv.py's rc (its docstring): 5 = a process group of ours left after SIGKILL, 124 = its limit hit before the load end
  case ${rc:-0} in
    5) pg=$(grep -a '^PV LEFT process groups' "$d/run.log" | head -1 | sed 's/^PV LEFT process groups \(.*\) still running.*/\1/')
       fl_find "WARN $nm: run-ttx process group(s) pgid ${pg:-$(awk '{ printf "%s ", $1 }' "$d/pgids" 2> /dev/null)} left after pv.py's SIGKILL (see $d/run.log)" ;;
    124) FL_PV_INC="power virus hit its limit"
       fl_skip "$nm" "INCOMPLETE: pv.py hit its $(( FL_BB_PV_S + 60 )) s limit before the end of its $FL_BB_PV_LOAD s load (rc 124, at $el s; see $d/run.log)" ;;
  esac
  # the shared reset path, as C2 (settled sa3/sa4): rk_reset (hold-file and serving gate, one log $RK_RESET_LOG), recount,
  # root-port rescan, tray re-power where glx_reset can run, then rk_resfail; NO RESET only when glx_reset did not run
  local lb="tt-smi -glx_reset after the power virus" aw named ok=1 gs=""; af=""
  # the fleet gate first, as before C9 (gate.sh): another process on the chips now means no reset under it (FL_STOP, GATE BUSY)
  if [ -n "${FL_STOP:-}" ] || ! fl_gate PV; then say "$nm: no glx_reset after it ($FL_STOP); later steps open the chips as the virus left them"; pvc_close; return 0; fi
  RK_BEFORE=$(rk_snap); RK_STEP=PV rk_reset "$d/reset.log"   # PV: the hook step (PV-glx), gated again like C9 (fl_regate): no reset under another process
  if [ "$RK_RESET_STAT" != OK ] && [ "$RK_RESET_STAT" != FAILED ]; then
    say "$nm: no reset after it (${RK_RESET_WHY:-glx_reset did not run}); later steps open the chips as the virus left them"; pvc_close; return 0; fi
  [ "$RK_RESET_STAT" = OK ] || { ok=0; say "$nm: glx_reset after it FAILED ($RK_RESET_WHY; see $RK_RESET_LOG)"; }
  rk_recheck "$lb" "$d/recheck.txt"; aw=$RK_AFTER
  RK_RESCANNED=""; RK_RESCAN_RAN=""; RK_RESCAN_GSKIP=""; rk_rescan "$lb" "$d/recheck_rescan.txt"
  named=$(rk_umd_named "$RK_RESET_LOG" | grep . | sort -u)
  if [ -n "$(echo "$RK_AFTER" | awk '$3 != "ok"')" ]; then
    if rk_can_glx; then RK_RECOVER_PRE="$lb" rk_recover "$d/glx" "$d/recheck_glx" "tray re-power ($nm)"; af=$RK_AFTER
    else gs="$RK_GLX_CAN_WHY"; say "$nm: tray re-power skipped ($gs)"; fi
  fi
  rk_resfail "$nm glx_reset" "$aw" "$named" "$ok" "${RK_RESET_WHY:-tt-smi -glx_reset}" "$af" "$gs"
  if [[ $RK_RESET_NEW == "RESET UNATTRIBUTED"* ]]; then
    fl_attr RESET FAIL "glx_reset after the $nm failed and names no chip: $(grep -a . "$RK_RESET_LOG" 2> /dev/null | tail -1 | cut -c1-120); $(echo "$aw" | grep -c ' ok$')/$(rk_total) after it" 0 "$RK_RESET_LOG"
  else echo "$RK_RESET_NEW" | grep -a . | while read -r l; do fl_find "$l"; done; fi
  lost=$(echo "$RK_AFTER" | awk '$3 != "ok" { print $1 }' | while read -r b; do echo "$(rk_tray "$b") ${b#0000:}"; done)
  if [ -n "$lost" ] && [ "$ok" = 1 ]; then   # a FAILED glx_reset's lost chips are its RESET FAIL lines above
    fl_lost "$nm at $el s" > /dev/null
    # one cause line per lost chip: a chip rk_resfail's lines already name gets no DROP line here
    # a tray that lost all 8 keeps one tray-level POWER DROP line (tray_why: whole-tray power loss; the drop-tray lists),
    # naming only the chips no RESET line names
    all=$lost; lost=$(echo "$lost" | while read -r ub b; do [ -n "$b" ] && ! grep -aqF "$b (tray " <<< "$RK_RESET_NEW" && echo "$ub $b"; done)
    for ub in $(echo "$all" | awk 'NF { print $1 }' | sort -u); do
      ga=$(echo "$all" | awk -v u="$ub" '$1 == u' | grep -c .); g=$(echo "$lost" | awk -v u="$ub" '$1 == u' | grep -c .)
      bd=$(echo "$lost" | awk -v u="$ub" '$1 == u { printf "%s%s", (n++ ? " " : ""), $2 }')
      if [ "$ga" -ge 8 ]; then
        fl_find "POWER DROP tray $ub ($ga chips lost${bd:+: $bd}$([ "$g" -lt "$ga" ] && echo "; $([ -n "$bd" ] && echo "the others" || echo each) in its RESET line above")) after the $nm ($FL_BB_PV_LOAD s run-ttx load) and its glx_reset"
      elif [ "$g" -gt 0 ]; then fl_find "CHIP DROP tray $ub ($g chips lost: $bd) after the $nm ($FL_BB_PV_LOAD s run-ttx load) and its glx_reset"; fi
    done
  elif [ -z "$lost" ]; then rk_map > /dev/null 2>&1 || true; say "$nm: glx_reset and recount after it: $(echo "$RK_AFTER" | grep -c ' ok$')/$(rk_total) chips ($el s)"; fi
  pvc_close
}
fl_topo() {  # "<ring|linear>: <why>" for the bus-bar CCL's fabric from the latest
  # discovery (c9, c2, pre) and the platform's wrap ports (W lines of eth_map_<platform>.txt): ring only with every wrap up
  local r
  for r in c9 c2 pre; do [ -s "$FL_OUT/disc_$r.json" ] && break; done
  [ -s "$FL_OUT/disc_$r.json" ] || { echo "ring: no discovery read (as before)"; return 0; }
  python3 "$FL_DIR/fleet_py.py" topo "$FL_OUT/disc_$r.json" "$FL_DIR/eth_map.txt" --host "$(hostname -s)" 2>/dev/null | grep -a . ||
    echo "ring: topology not read (as before)"
}
fl_bb_step() {  # [<label> <busbar/ subdir> <quiet s> <burst s> <numeric 0|1> [<cycles>]] (no args: the step, as before; qb passes its own)
  # the bus-bar stage's cold-quiet-to-full-load step:
  # mesh open (~40 s idle), FLEET_BUSBAR_QUIET_S of forced quiet (counted once every chip is at <= 800 MHz), then one FLEET_BUSBAR_BURST_S full-load burst on all 32
  # chips: the strongest load measured (lofi_all, 50 ms gaps, 8 steps: an outside galaxy's swing legs, 7.2-12.8 kW box); its numeric
  # outputs are read after the burst (the burst has no numeric check of its own, and 88:00.0 tray 4 failed only the heavy part; the heavy chunk is in quick and full, not sustained), so
  # nothing syncs inside it. 90 s covers the 52 s tray 1/2/3 drop of a quad host (C6b)
  # and the first-burst drop of a cluster node's tray 3 (0.09 s) with margin. A loss is named
  # by tray 1-4 and BDF; run.sh's loss path (after fl_loads) captures the evidence. Never writes a power limit.
  # nu=0 (qb) passes --numeric-every -1 (off): ub0_repro.py's default 500 would read back 30-90 s per burst
  # nc > 1 (TRAY_QB_ONEPROC=1): nc x (quiet + burst) in one ub0_repro.py process (--cycles), one mesh open
  local nm=${1:-bus-bar stage step} d=$FL_OUT/busbar/${2:-step} q=${3:-${FLEET_BUSBAR_QUIET_S:-10}} b=${4:-${FLEET_BUSBAR_BURST_S:-90}} nu=${5:-1} nc=${6:-1}
  local t0 el lost ub g bd pg v pw pb tl cy nb l i pf=$FL_PW_FILE tp lg=$d/run.log
  local fs="" fq=""; [[ ${2:-} == qb* ]] && { fs=${FLEET_BUSBAR_QB_STALL_S:-180}; fq=$(( q + ${TRAY_QUIET_CLK_CAP_S:-300} + 60 )); }
  local lim=${FLEET_BUSBAR_LIMIT_S:-$(( 600 + ${TRAY_QUIET_CLK_CAP_S:-300} + (nc - 1) * (q + b + ${TRAY_QUIET_CLK_CAP_S:-300} + 60) ))}
  fl_try_fits "$nm" $(( nc * (q + b) + ${FLEET_BB_SETUP_S:-60} )) || return 0   # mesh open ~40 s + numeric read: 90 s burst needs ~150 s
  mkdir -p "$d"; tp=$(fl_topo)
  say "$nm: $([ "$nc" -gt 1 ] && echo "$nc x ($q s quiet, then one $b s burst) in one process" || echo "$q s quiet, then one $b s") full-load burst on all chips ($FL_LOAD_HEAVY, $FL_LOAD_GAP_MS ms gaps$([ "$nu" = 1 ] && echo ", numeric read after the burst"); fabric $tp)"
  while :; do t0=$(date +%s)
    hb_start "$nm" "$lg"
    fl_try_need "$nm" "$lim"
    # quiet-burst tries: 180 s with no
    # progress after fabric setup is a hang (900 s kept during fabric setup), plus the open quiet before the burst
    TRY_FAB_LIMIT_S=$fs TRY_FAB_IDLE_S=$fq \
    TRY_LOSS_S=${FLEET_BUSBAR_LOSS_S:-60} TRY_LIMIT_S=$(fl_try_cap "$lim") try_run busbar "$lg" "${AS[@]}" "$PY" ub0_repro.py --load ub0 \
      $(fl_burst_args) $(fl_numeric && { [ "$nu" = 1 ] && echo --numeric-every "${FLEET_NUMERIC_EVERY:-500}" || echo --numeric-every -1; }) \
      --wait-quiet-min 0 --idle-s "$q" --burst-s "$b" $([ "$nc" -gt 1 ] && echo --cycles "$nc") $([ "${tp%%:*}" = linear ] && echo --topo linear) --result "$d/result.json" --telem "$d/telem.csv"; hb_stop
    fl_fabstall "$lg" "$nm" "$t0"
    # a try that failed after a set_fabric_config stall is retried once per run while every chip
    # answers, under the same fl_try_cap limit (the stall's time was added to the budget); its first log stays as run.log
    [ -n "$FL_FAB_HIT" ] && [ "$rc" != 0 ] && [ -z "$FL_FAB_RETRY" ] && [ -z "$(rk_snap | awk '$3 != "ok"')" ] || break
    FL_FAB_RETRY=$nm; lg=$d/run_retry.log; mv -f "$d/result.json" "$d/result_stall.json" 2> /dev/null; mv -f "$d/telem.csv" "$d/telem_stall.csv" 2> /dev/null
    say "$nm: retried once after the set_fabric_config stall (ub0_repro.py rc $rc; first try: $d/run.log, retry: $lg)"
  done
  FL_BB_TRY=$(( ${FL_BB_TRY:-0} + 1 ))   # a part that ran ub0_repro.py (fl_busbar's not-tested check)
  sleep "${FLEET_POLL_TAIL_S:-2}"; el=$(( $(date +%s) - t0 ))   # one more poll row
  if [ "$nc" -gt 1 ]; then   # a LOAD line per cycle under the label a step of its own has ("... quiet-burst 2/2"); the chips line over all
    # the cycle the process ended in (a loss, a hang, a failure) gets one too, its burst up to the end (or none if it never began)
    local ny; ny=$(fl_bb_ncyc "$d/result.json")
    [ "$ny" -lt "$nc" ] && { [ "$rc" != 0 ] || [ -n "$(rk_snap | awk '$3 != "ok"')" ]; } && ny=$(( ny + 1 ))
    for (( i = 1; i <= ny; i++ )); do
      python3 -c 'import json,sys
try: r = json.load(open(sys.argv[1]))
except (OSError, ValueError): r = {}
i, c = int(sys.argv[2]), r.get("cycles") or []
b = c[i - 1] if i <= len(c) else dict(t0=r["burst_t0"], seconds=max(0.0, round(float(sys.argv[4]) - r["burst_t0"], 1))) if r.get("cycle") == i and r.get("burst_t0") else {}
json.dump(dict(r, burst=b), open(sys.argv[3], "w"))' "$d/result.json" "$i" "$d/cycle$i.json" "$(( t0 + el ))" \
        && fl_load_line "$pf" "$d/cycle$i.json" "${nm% *} $i/$nc" "$d/telem.csv"; done
  else fl_load_line "$pf" "$d/result.json" "$nm" "$d/telem.csv"; fi
  fl_chips_line "$d/result.json" "$nm"
  nb=""; if [ "$nu" = 1 ]; then   # qb asks for no numeric readback, so it judges none
    python3 ./numeric_report.py "$d/result.json" > "$d/numeric.txt" 2>&1; nb=$(grep -aE '^NUMERIC (FAIL|WARN) ' "$d/numeric.txt"); fi
  [ -n "$nb" ] && while IFS= read -r l; do fl_find "$l (${nm/#bus-bar stage /stage 5 numeric verify, after the })"; done <<< "$nb"   # fault capture + finding_stage.tsv
  quiet_findings "$lg" "$nm"   # its quiet counts from every chip at <= 800 MHz; a chip above after the 5 min cap
  pg=$(fl_pg_fell "$pf" | awk -v t="$t0" '$1 >= t')
  case ${FL_BB_ORDER%%,*}:${2:-} in qb:qb1|step:) fl_pg_start "$pf" | while read -r tn v; do   # the stage's first part: a tray not ff at its start, as fl_c7
    fl_find "POWER FAIL tray $tn ($(rk_snap | while read -r bb c st; do [ "$(rk_tray "$bb")" = "$tn" ] && printf '%s ' "${bb#0000:}"; done | sed 's/ $//')): vr_pg $v before the bus-bar stage load started (never read ff)"; done ;; esac
  lost=$(rk_snap | awk '$3 != "ok"' | while read -r bb c st; do echo "$(rk_tray "$bb") ${bb#0000:}"; done)
  if [ -n "$lost$pg" ]; then
    fl_lost "$nm at $el s" > /dev/null
    [ -n "$pg" ] && FL_PGDROP="vr_pg fell on tray $(echo "$pg" | awk '{ print $2 }' | sort -u | paste -sd,) in the $nm"
    # the t0 of the burst the loss fell in: burst_t0 is set at each burst's start (--cycles: the cycle it names), burst.t0 at its end
    # (--cycles: a loss in cycle 2's quiet has no burst_t0, and burst.t0 is cycle 1's: '-', the step's t0, as with TRAY_QB_ONEPROC=0)
    read -r tl cy < <(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r.get("burst_t0") or r.get("cycle", 1) == 1 and r["burst"]["t0"] or "-", r.get("cycle", 1))' "$d/result.json" 2>/dev/null)
    [ "$tl" = - ] && tl=
    [ -n "$cy" ] || cy=$(grep -ao 'cycle [0-9]*/[0-9]* begin' "$lg" 2>/dev/null | tail -1 | sed 's|^cycle \([0-9]*\)/.*|\1|')   # no result.json: its last phase mark
    [ "$nc" -gt 1 ] && nm="$nm, cycle ${cy:-?}/$nc"
    pw=$(fl_pw_says "$(awk 'NF >= 10 { l = $0 } END { print l }' "$pf" 2>/dev/null)")
    for ub in $( { echo "$lost" | awk 'NF { print $1 }'; echo "$pg" | awk 'NF { print $2 }'; } | sort -u); do
      g=$(echo "$lost" | awk -v u="$ub" '$1 == u' | grep -c .); bd=$(echo "$lost" | awk -v u="$ub" '$1 == u { printf "%s%s", (n++ ? " " : ""), $2 }')
      v=$(echo "$pg" | awk -v u="$ub" -v t="${tl:-$t0}" '$2 == u { printf "vr_pg %s at %d s into the burst", $3, $1 - t }')
      pb=""; [ -n "$v" ] && pb=$(fl_pw_says "$(fl_pw_at "$pf" "$(echo "$pg" | awk -v u="$ub" '$2 == u { print $1 - 1 }')")")
      fl_find "$([ "$g" -ge 8 ] || [ -n "$v" ] && echo "POWER DROP" || echo "CHIP DROP") tray $ub ($g chips lost${bd:+: $bd}${v:+; $v}) in the $nm ($q s quiet then $b s full load), BMC after: ${pw:-unread}${pb:+; before the fall: $pb}"   # BMC after first: fl_report and tray_why.py drop from there, so the W of other trays name no tray
    done
    return 0
  fi
  case $rc in
    0) [ -n "$nb" ] && say "$nm: numeric mismatch ($el s)" || say "$nm: clean ($el s)" ;;
    *) # the chips its own error names (C4's try-log attribution, folded in when C4 went), each under its tray
       # the open timed out only on off-box links whose own evidence says the peer
       # host is down or resetting (ethcore's WARN): that link and chip count in no FAIL; read from this host only.
       # Only before any load row (a telem.csv row in a load phase): a peer-down timeout mid-run stays judged.
       # As fl_c7: fl_nj_open notes only when it takes the step; mixed on-box and off-box ETH lines, or none,
       # go to fault_attr with no ETH note first, so the step writes one fault.txt entry
       local fc fe ferr ld fv=""   # fv: the chips whose ETH core line is not judged
       ld=$(awk -F, '$7 ~ /^(upload|idle|warmup|burst|post_burst)$/ { print 1; exit }' "$d/telem.csv" 2>/dev/null)
       if fl_nj_open "$nm" "${ld:-0}" "$lg"; then FL_BB_NJ=$(( ${FL_BB_NJ:-0} + 1 ))
         say "$nm: ub0_repro.py rc $rc at device open, only off-box links with the peer host down or resetting (not judged; see $lg)"
       else
         FE_NONOTE=1 fault_eth "$lg"
         while read -r fe; do [ -n "$fe" ] || continue; fe=$(fl_eth_noreset "$nm" "$fe"); fl_find "$fe"
           [[ $fe == "ETH INCOMPLETE "* ]] && fv="$fv|${fe#ETH INCOMPLETE }"; done <<< "$FE_LINES"
         if fault_attr "$nm FAIL: ub0_repro.py rc $rc" 0 "$lg"; then
           while read -r fc _ ferr; do [ -n "$fc" ] || continue
             if [ -n "$fv" ] && [[ $fv == *"|$(fc_tag "$fc"): "* ]] && [[ $ferr == *"active ethernet core"*"active again"* ]]; then   # the same error, the same chip
               fl_find "ETH INCOMPLETE $(fc_tag "$fc"): $nm: start reset skipped; ETH not-active-at-open not judged: ub0_repro.py rc $rc (its error: ${ferr:0:140}; see $lg); no glx_reset ran at C2 ($FL_C2_NONE)"
             else fl_find "$nm FAIL $(fc_tag "$fc"): ub0_repro.py rc $rc (its error: ${ferr:0:140}; see $lg)"; fi; done <<< "$FA_LINES"
         else fl_find "$nm FAIL: ub0_repro.py rc $rc${TRY_HUNG:+, $TRY_HUNG} (see $lg)"; fi
       fi ;;
  esac
}
fl_gone_named() {  # "; still missing: <BDF> (tray N) ..." for each chip not answering now, or nothing
  local g; g=$(rk_snap | awk '$3 != "ok" { print $1 }'); [ -n "$g" ] || return 0
  printf '; still missing: %s' "$(for b in $g; do printf '%s (tray %s) ' "$b" "$(rk_tray "$b")"; done | sed 's/ $//')"
}
fl_load_phases() {  # fl_loads' phases in order for this FLEET (tests/test_mode_equal.sh: full runs exactly the ones quick runs)
  [ "$FLEET" = repro-linka ] && { echo linka; return 0; }   # run.sh --repro link-a: the link-A repro alone (never in quick or full)
  [ "$FLEET" = repro-abortfab ] && { echo abortfab; return 0; }   # run.sh --repro abort-fabric alone (never in quick or full)
  # FLEET_REPRO (repro.sh): numeric runs only the bus-bar stage (its FLEET_BUSBAR_* way), warm-reset no load
  case ${FLEET_REPRO:-} in numeric|tray-drop) echo busbar; return ;; warm-reset) return ;; esac   # tray-drop: its shape
  echo busbar C5; }   # C5 stays
fl_loads() {  # the bus-bar stage, first in every mode (full: with its long chunk, the old sustained part C7) -> numeric verify -> C5; any loss, hang or busy gate ends them (a hung chip would only hang the next mesh)
  local p bk; FL_C5_PEND=0; FL_BB_BACK=0; case " $(fl_load_phases) " in *" C5 "*) FL_C5_PEND=1 ;; esac   # stage 4 leaves stage 6 its RS_C5_S (fl_keep)
  [ "$seen" = 1 ] && FL_CAP_FAULT=0 fl_capture c4 drop $(rk_snap | awk '$3 != "ok" { print $1 }')
  for p in $(fl_load_phases); do
    by_check   # a queued foreign broker job seen by the watcher: FL_STOP, the stages left skip
    fl_psu_join 0   # the read after the last stage, if done; fl_psu_join 1 below before any load starts
    fl_boot_recount   # the post-boot settle recount at the first boundary past 15 min of uptime
    if [ -n "$FL_STOP" ]; then fl_skip "$(fl_pn $p)" "$FL_STOP"; continue; fi
    if [ -n "$FL_PSU_STOP" ]; then fl_skip "$(fl_pn $p)" "INCOMPLETE: $FL_PSU_STOP"; continue; fi   # a lost PSU: no more load
    if [ "${KD_ON:-0}" = 1 ] && [ "$p" != C5 ]; then fl_skip "$(fl_pn $p)" "$(fl_kdwhy)"; continue; fi
    if [ -n "${FW_NR:-}" ] && [ "$p" != C5 ]; then fl_skip "$(fl_pn $p)" "$(fl_nrwhy)"; continue; fi
    # once the bus-bar stage's own recovery brought every chip back (FL_BB_BACK), the stages after it
    # run; its loss, vr_pg fall or hang stays for the verdict
    bk=0; [ "$p" != busbar ] && [ "$FL_BB_BACK" = 1 ] && { [ "$seen" = 1 ] || [ -n "$FL_PGDROP$FL_HUNG" ]; } && bk=1 \
      && say "$(fl_pn $p): runs: the bus-bar stage's recovery brought every chip back (faults kept for the verdict: $(printf '%s' "${at:+chips lost in $at; }${FL_PGDROP:+$FL_PGDROP; }${FL_HUNG:+$FL_HUNG; }" | sed 's/; $//'))"
    if [ "$seen" = 1 ] && [ "$bk" = 0 ]; then fl_skip "$(fl_pn $p)" "chips lost in $at$(fl_gone_named)"; continue; fi
    if [ -n "$FL_PGDROP" ] && [ "$bk" = 0 ]; then fl_skip "$(fl_pn $p)" "$FL_PGDROP"; continue; fi   # a tray's rails fell (no chip lost yet): no more load on it
    if [ -n "$ccl$failed" ]; then fl_skip "$(fl_pn $p)" "C4: ${ccl:-$failed}"; continue; fi
    if [ ${#FL_SUSPECT[@]} -gt 0 ] || { [ -n "$FL_HUNG" ] && [ "$bk" = 0 ]; }; then fl_skip "$(fl_pn $p)" "a hang in an earlier phase"; continue; fi
    fl_psu_join 1   # a load stage starts only once the read after the last stage is in (bounded by PSU_JOIN_S)
    if [ -n "$FL_PSU_STOP" ]; then fl_skip "$(fl_pn $p)" "INCOMPLETE: $FL_PSU_STOP"; continue; fi
    fl_t "$p"; fl_gate "$p" || { fl_skip "$(fl_pn $p)" "$FL_STOP"; continue; }
    fl_aer_stage_before "$p"   # per-chip AER counters before the stage (reads only)
    fl_hsc_before "$p"   # HSC STATUS_WORD/STATUS_INPUT/VIN before the leg, then the VIN sampler (reads only)
    # the box power and air temperature poll through every load stage: the bus-bar stage starts its own
    # (fl_busbar); stage 6 and the repro phases get one here, to $FL_OUT/<c5|linka|abortfab>/power.txt and box_power.csv
    local own=""; [ "$p" = busbar ] || [ -n "${FL_PW_PID:-}" ] || { fl_power_start "$FL_OUT/$(echo "$p" | tr C c)/power.txt"; own=1; }
    case $p in busbar) fl_busbar ;; C5) FL_C5_PEND=0; fl_c5; fl_stage_load C5 ;; linka) fl_linka ;; abortfab) fl_abortfab ;; esac
    [ -z "$own" ] || fl_power_stop
    fl_aer_stage_after "$p"   # the stage's AER rise per chip, by BDF, tray and stage
    fl_hsc_after "$p"   # a UV latch, OFF or IOUT_OC set in the leg: an HSC FAIL naming the tray and HSC; min VIN < 11.0 V: evidence only
    fl_psu_bg "after $(fl_pn $p)" $([ -s "$FL_OUT/$(echo $p | tr C c)/power.txt" ] && echo "$FL_OUT/$(echo $p | tr C c)/power.txt")   # per-PSU W beside the stage's LOAD line
  done
  fl_boot_recount; fl_boot_unreached   # the boundary after the last stage
}
fl_sweep() {  # after the C9 reset, only when FL_PIN (a hang or CCL failure) is set: C5's per-link all-gather on the reset
  # chips, then one PINPOINT line per hung or failed link (both ends' eth channels, BDFs, trays 1-4); a hung link gets
  # one more gated reset so no chip is left hung
  # FL_SWEEP_NORESET=<why>: the end reset was refused but every chip still answers: sweep the chips as the
  # fault left them (no reset before or after it; a hung link stays hung and is named in the RESULT). --hold: no sweep.
  local d=$FL_OUT/sweep lab idx r a b ba bb da db ca cb t0 rc1 tr1 ch1 rs1 hp="" np nu nt m nrw=${FL_SWEEP_NORESET:-} aft="after the reset"
  [ -n "$FL_PIN" ] || return 0
  if [ "${HOLD:-0}" = 1 ]; then say "PINPOINT skipped: --hold (hang-debug hold mode): no per-link sweep, the chips are left as the fault and the hold left them ($FL_PIN)"; FL_SKIPPED+=("per-link sweep (--hold)"); return 0; fi
  if [ -z "$nrw" ]; then
    if [ "$RK_RC" = 9 ]; then fl_skip "per-link sweep" "the serving gate refused the end reset (${RK_RESET_SKIP:-SKIPPED-GATE})"; return 0; fi
    if [ "$RK_RC" != 0 ] && [ "$RK_RC" != 5 ] && [ "$RK_RC" != 14 ]; then fl_skip "per-link sweep" "chips missing after the end reset"; return 0; fi
  else aft="with no reset"; fi
  fl_t sweep; fl_gate sweep || { fl_skip "per-link sweep" "$FL_STOP"; return 0; }
  rk_map; echo "$RK_MAP" > "$FL_OUT/map.txt"; FL_SWEPT=(); t0=$(date +%s)
  if [ -n "$nrw" ]; then say "PINPOINT: $FL_PIN; the end reset was refused ($nrw) but every chip answers: per-link all-gather sweep with no reset"; r=""
  else say "PINPOINT: $FL_PIN; state captured before the reset (fault captures in $FL_OUT); per-link all-gather sweep after it"
    [ -s "$FL_OUT/disc_c9.json" ] && r=c9 || r=pre; fi
  ! declare -F rs_need > /dev/null || rs_need "per-link sweep" $(( ${FLEET_SWEEP_S:-300} + ${FLEET_C5_CAPX_S:-60} )) "${RS_RESULT_S:-60}"   # after a fault: its whole limit
  FL_SW=1 fl_c5 sweep "$r" "$(fl_sweep_s)"
  for lab in "${FL_SWEPT[@]}"; do
    IFS=$'\t' read -r idx r a b ba bb da db ca cb <<< "$(awk -F'\t' -v i="${lab#p}" '$1 == i' "$d/pairs.tsv")"
    fl_find "PINPOINT LINK HANG $(FL_SW=1 fl_lk "$ba") e$ca <-> $(FL_SW=1 fl_lk "$bb") e$cb: its 2-chip all-gather hung $aft ($FL_PIN; triage $d/$lab)"
    hp=${hp:-"PINPOINT LINK HANG $(FL_SW=1 fl_lk "$ba") e$ca <-> $(FL_SW=1 fl_lk "$bb") e$cb"}
  done
  np=$(grep -c PASS "$d/result.txt" 2>/dev/null); nu=$(grep -c UNTESTED "$d/result.txt" 2>/dev/null)
  nt=$(grep -c . "$d/result.txt" 2>/dev/null); m=$(tail -n +2 "$d/pairs.tsv" 2>/dev/null | grep -c .)
  if [ -z "$hp" ] && ! grep -q 'FAIL' "$d/result.txt" 2>/dev/null; then
    # "no single link" only when every local link was swept and passed (or cannot be run pairwise); else say what is missing
    if [ "${m:-0}" -gt 0 ] && [ $(( ${np:-0} + ${nu:-0} )) = "$m" ] && [ -z "$FL_C5_GAP" ]; then
      fl_find "PINPOINT WARN no single link: $FL_PIN, but all ${np:-0} local links passed the per-link all-gather $aft (${nu:-0} not testable pairwise; QSFP links not swept); the fault needs the full mesh load ($d)"
    else fl_find "PINPOINT WARN sweep incomplete: ${nt:-0} of ${m:-0} local links tested (${FL_C5_GAP:-no local pairs}); $FL_PIN stays named by its stage only ($d)"; fi
  fi
  say "PINPOINT sweep: $(( $(date +%s) - t0 )) s"
  [ -n "$hp" ] || return 0
  [ -n "$nrw" ] && { FL_SWEEP_HP=$hp; return 0; }   # no reset may run: the hung link stays hung (fl_end names it left hung)
  rc1=$RK_RC; tr1=$RK_TRAYS; ch1=$RK_CHIPS; rs1=$RK_RESETS; mkdir -p "$OUT/end_sweep"
  if fl_peer C9s; then
    RK_PHASE="reset after the per-link sweep" RK_STEP=C9s rk_end "$OUT/end_sweep" "" 0 "in the per-link sweep"
    echo "$RK_RESET_NEW" | grep -a . | while read -r l; do fl_find "$l"; done
    RK_RESETS="$rs1; after the per-link sweep: $RK_RESETS"
    if [ "$RK_RC" = 0 ] || [ "$RK_RC" = 5 ] || [ "$RK_RC" = 14 ]; then RK_RC=$rc1; RK_TRAYS=$tr1; RK_CHIPS=$ch1
    elif [ -n "$RK_RESET_SKIP" ] && [[ $RK_WHAT == "INCOMPLETE: no loss"* ]]; then RK_RC=$rc1; RK_TRAYS=$tr1; RK_CHIPS=$ch1; fi   # refused, nothing lost: C9's verdict stands (as C9b)
    [ -n "$RK_RESET_SKIP" ] && { fl_rskip "reset after the per-link sweep" "$RK_RESET_SKIP"; FL_LEFT_HUNG=$hp; FL_LEFT_WHY=$RK_RESET_SKIP; }
  else RK_RESETS="$rs1; reset after the per-link sweep skipped ($FL_PEER_WHY)"; fl_rskip "reset after the per-link sweep" "$FL_PEER_WHY"
    FL_LEFT_HUNG=$hp; FL_LEFT_WHY=$FL_PEER_WHY; fi
}
fl_noreset() {  # $1 why: no reset may run; a read-only recheck sets run.sh's verdict variables (never a tray drop; lost chips by BDF and tray)
  local lost; RK_BEFORE=$(rk_snap); rk_recheck "run (no reset)" "$OUT/recheck.txt"
  lost=$(echo "$chips $(echo "$RK_AFTER" | awk '$3 != "ok" { print $2 }')" | rk_list)
  RK_RESETS="skipped ($1)${lost:+; still missing: chips $lost}"; RK_CHIPS=${lost:-None}
  # name each lost chip by BDF and tray 1-4; without a reset a chip or tray drop is not proven (rc 1 stays)
  if [ -n "$lost" ]; then local ids miss still; ids=" $(echo "$chips $(echo "$RK_AFTER" | awk '$3 != "ok" { print $2 }')" | tr ' ,' '\n\n' | grep -E '^[0-9]+$' | sort -nu | paste -sd' ') "
    miss=$(echo "$RK_MAP" | awk -v ids="$ids" 'index(ids, " " $2 " ")' | rk_bdfs); still=$(echo "$RK_AFTER" | awk '$3 != "ok"' | rk_bdfs)
    RK_RC=1; RK_WHAT="loss: chips $lost lost: ${miss:-not mapped}; chip or tray drop not proven: no reset could run ($1)"
    if [ -n "$RK_CHECK_WHY" ]; then RK_TRAYS="unknown ($RK_CHECK_WHY)"
    elif [ -n "$still" ]; then RK_TRAYS="unknown (still missing: $still; not proven a tray drop: no reset could run: $1)"
    else RK_TRAYS="none still missing (chips came back: ${miss:-not mapped})"; fi
  else RK_RC=0; RK_TRAYS=None; RK_WHAT="no loss"; [ -n "$RK_CHECK_WHY" ] && { RK_RC=1; RK_WHAT="unknown ($RK_CHECK_WHY)"; RK_TRAYS="unknown ($RK_CHECK_WHY)"; }; fi
}
fl_sweep_s() { echo "${FLEET_SWEEP_S:-300}"; }  # the per-link sweep's planned length; fl_c5 logs once when it is passed and never stops on it
fl_pin_noreset() {  # $1 why no reset could run, with FL_PIN set: every chip on the map answers (state ok: on the
  # bus, config space reads 1e52, tt-kmd node) -> the per-link sweep runs with no reset; else one named skip line with the
  # chips it could not test by BDF and tray 1-4. A load gate stop (FL_STOP) or --hold never sweeps.
  local bad
  if [ "${HOLD:-0}" = 1 ]; then fl_sweep; return 0; fi   # its --hold skip line
  if [ -n "$FL_STOP" ]; then say "PINPOINT skipped: reset refused ($1); the load gate stopped the run, so no sweep either; no link tested ($FL_PIN)"
    FL_SKIPPED+=("per-link sweep (no reset could run, load gate stop)"); return 0; fi
  bad=$(rk_snap | awk '$3 != "ok" { print $1, $3 }' | while read -r b st; do echo "$(fl_lk "$b") $st"; done | paste -sd, - | sed 's/,/, /g')
  if [ -n "$bad" ]; then say "PINPOINT skipped: reset refused ($1); chips not answering, not tested: $bad; no link swept ($FL_PIN)"
    FL_SKIPPED+=("per-link sweep (INCOMPLETE: no reset could run, chips not answering)"); return 0; fi
  FL_SWEEP_NORESET=$1 fl_sweep
}
fl_post_fits() { return 0; }  # $1 post-C9 step: always runs (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
fl_c3_fits() { FL_C3_SMI_T=120; FL_C3_DISC_T=180; }  # post-C9 fl_c3 keeps its whole tt-smi (120 s) and discovery (180 s) timeouts (no run budget skips or cuts a stage; a stage ends only on its own hang timeout)
fl_end() {  # C9: the end reset and recheck when the gates allow it, then C1/C3 after and the SUSPECT reruns
  local rc1 tr1 ch1 rs1 s lab idx r a b ba bb da db ca cb d=$FL_OUT/c5 nr="" hp="" p1="" c3s=""
  fl_psu_join 0   # the last stage's PSU read if done; C9 does not wait for it: fl_verdict joins it before the RESULT
  fl_c8_stop; while IFS= read -r l; do say "$l"; done < <(th_summary); fl_unhold; fl_t C9; FL_C9_ON=1   # fl_unhold: no held job outlives the stages (--hold); FL_C9_ON: later losses (the sweep) leave C9's share
  # a failed C3 discovery alone does not skip C9's recovery: a C3 stop goes to C9 and the end checks like a fault (not in --hold, which leaves the chips as they are)
  c3s=""; [ -n "${FL_C3STOP:-}" ] && [ "${FL_STOP:-}" = "$FL_C3STOP" ] && [ "${HOLD:-0}" != 1 ] && [ "${HOLD_ACTIVE:-0}" != 1 ] && { say "C9: runs although $FL_C3STOP (recovery)"; c3s=1; FL_STOP=""; }
  local ys=""; [ -n "${FL_YIELD:-}" ] && [ "${FL_STOP:-}" = "$FL_YIELD" ] && [ "${HOLD_ACTIVE:-0}" != 1 ] && { say "C9: runs after the broker yield"; ys=1; FL_STOP=""; }
  [ -z "$FL_STOP" ] && fl_gate C9
  if [ -n "$FL_STOP" ]; then fl_noreset "$FL_STOP"; fl_rskip C9 "$FL_STOP"; nr=$FL_STOP
  elif ! fl_peer C9; then fl_noreset "$FL_PEER_WHY"; fl_rskip "C9 reset" "$FL_PEER_WHY"; nr=$FL_PEER_WHY
  else
    FL_RESET_T=$(date +%s); RK_PHASE="C9 end reset" RK_STEP=C9 rk_end "$OUT" "$chips" "$seen"; FL_YIELD_C9=1   # the verdict's 'chips after the C9 reset'
    if fl_c3_fits "fl_c3 c9"; then fl_c3 c9; fl_c3cmp pre c9; fi; FL_C3_SMI_T=120; FL_C3_DISC_T=180
    echo "$RK_RESET_NEW" | grep -a . | while read -r l; do fl_find "$l"; done   # the reset left a chip missing
    [ -n "$RK_RESET_SKIP" ] && { fl_rskip "C9 reset" "$RK_RESET_SKIP"; [ "$RK_GLXRAN" = 1 ] || nr=$RK_RESET_SKIP; }   # the gate refused inside rk_end (rc 9); nr only if no reset ran
    [ -n "$ys" ] && FL_STOP=$FL_YIELD   # after the yield's C9 no sweep and no reruns: the device goes to the queued job
    if [ -z "$nr" ] && [ -n "$FL_PIN" ]; then
      [ -n "${ys:-}" ] || fl_sweep   # --hold: fl_sweep's skip line; chip BDF and tray as FL_PIN names them: no sweep after the yield's C9
    fi   # a hang or CCL failure (FL_PIN; none: no sweep, no budget check): the per-link sweep now that the chips are reset (no reset ran: fl_pin_noreset below)
    if [ ${#FL_SUSPECT[@]} -gt 0 ] && [ -z "$FL_STOP" ]; then
      if [ -n "$FL_LEFT_HUNG" ]; then s="not run: the reset after the per-link sweep was refused ($FL_LEFT_WHY)"   # a link is still hung
      elif [ "$RK_RC" = 9 ]; then s="not run: the end reset was skipped by the serving gate"
      elif [ "$RK_RC" != 0 ] && [ "$RK_RC" != 5 ] && [ "$RK_RC" != 14 ]; then s="not run: chips missing after the end reset"
      else s=""; rk_map; echo "$RK_MAP" > "$FL_OUT/map.txt"; fi
      for lab in "${FL_SUSPECT[@]}"; do
        IFS=$'\t' read -r idx r a b ba bb da db ca cb <<< "$(awk -F'\t' -v i="${lab#p}" '$1 == i' "$d/pairs.tsv")"
        if [ -z "$s" ]; then
          FL_JOBS=(); fl_pair "${lab}r" "$(echo "$RK_MAP" | awk -v x="$ba" '$1 == x { print $2 }')" "$(echo "$RK_MAP" | awk -v x="$bb" '$1 == x { print $2 }')" "$d" "$ba $bb"
          FL_LAUNCH0_S=${FLEET_PAIR_COMPILE_S:-60} fl_watch $(( ${FLEET_COMPILE_S:-300} + 60 )) "${FLEET_PAIR_HANG_S:-30}" "$ba $bb" "$d" "${FL_JOBS[@]}"
          rk_opencheck "$d/${lab}r.log" "SUSPECT rerun ${lab}r" "$ba" "$bb" || :
          [ "${FL_RC[${lab}r]}" = 0 ] && r=PASS || r=HANG
        else r=$s; fi
        if [ "$r" = PASS ]; then fl_find "LINK WARN ${ba#0000:} ($(fl_ubb "$ba")) e$ca <-> ${bb#0000:} ($(fl_ubb "$bb")) e$cb: hung once, passed after the reset"
        else fl_find "LINK HANG ${ba#0000:} ($(fl_ubb "$ba")) e$ca <-> ${bb#0000:} ($(fl_ubb "$bb")) e$cb: triage saved $d/$lab (rerun after reset: $r)"; fi
        [ "$r" = HANG ] && s="not run: an earlier rerun hung"   # hp: a pair still hung (a rerun hung, or no end reset ran)
        { [ "$r" = HANG ] || [ -n "$nr" ]; } && hp=${hp:-"LINK HANG ${ba#0000:} ($(fl_ubb "$ba")) e$ca <-> ${bb#0000:} ($(fl_ubb "$bb")) e$cb"}
      done
      if [ -n "$(grep -l '^LINK HANG.*rerun after reset: HANG' "$FL_OUT/findings.txt" 2>/dev/null)" ]; then   # leave no chip hung
        rc1=$RK_RC; tr1=$RK_TRAYS; ch1=$RK_CHIPS; rs1=$RK_RESETS; mkdir -p "$OUT/end2"
        if fl_peer C9b; then
          RK_PHASE="C9b reset after the SUSPECT reruns" RK_STEP=C9b rk_end "$OUT/end2" "" 0 "in the SUSPECT reruns"
          echo "$RK_RESET_NEW" | grep -a . | while read -r l; do fl_find "$l"; done
          RK_RESETS="$rs1; after the SUSPECT reruns: $RK_RESETS"
          # every chip back: C9's verdict stands (a C9b glx_reset failure: exit 14 via RK_RESET_FAILS unless worse)
          if [ "$RK_RC" = 0 ] || [ "$RK_RC" = 5 ] || [ "$RK_RC" = 14 ]; then RK_RC=$rc1; RK_TRAYS=$tr1; RK_CHIPS=$ch1
          elif [ -n "$RK_RESET_SKIP" ] && [[ $RK_WHAT == "INCOMPLETE: no loss"* ]]; then RK_RC=$rc1; RK_TRAYS=$tr1; RK_CHIPS=$ch1; fi
          # the gate refused inside rk_end: the pair is still hung (rc 7 and the hung marker, fl_verdict)
          [ -n "$RK_RESET_SKIP" ] && { fl_rskip "C9b reset" "$RK_RESET_SKIP"; FL_LEFT_HUNG=$hp; FL_LEFT_WHY=$RK_RESET_SKIP; }
        else RK_RESETS="$rs1; reset after the SUSPECT reruns skipped ($FL_PEER_WHY)"; fl_rskip "C9b reset" "$FL_PEER_WHY"
          FL_LEFT_HUNG=$hp; FL_LEFT_WHY=$FL_PEER_WHY; fi
      fi
    fi
  fi
  [ ${#FL_SUSPECT[@]} -gt 0 ] && [ -z "$(grep '^LINK' "$FL_OUT/findings.txt" 2>/dev/null)" ] && for lab in "${FL_SUSPECT[@]}"; do
    IFS=$'\t' read -r idx r a b ba bb da db ca cb <<< "$(awk -F'\t' -v i="${lab#p}" '$1 == i' "$d/pairs.tsv")"
    fl_find "LINK HANG ${ba#0000:} ($(fl_ubb "$ba")) e$ca <-> ${bb#0000:} ($(fl_ubb "$bb")) e$cb: triage saved $d/$lab (rerun after reset: not run, no reset)"
    p1=${p1:-"LINK HANG ${ba#0000:} ($(fl_ubb "$ba")) e$ca <-> ${bb#0000:} ($(fl_ubb "$bb")) e$cb"}; done
  if [ -n "$nr" ] && [ -n "$FL_PIN" ]; then fl_pin_noreset "$nr"; fi   # no reset ran: sweep the chips as they are, if they all answer
  if [ -n "$nr" ]; then   # no reset ran: whatever hung is still hung (a C5 pair, a swept link or a load stage)
    FL_LEFT_HUNG=${p1:-${hp:-${FL_SWEEP_HP:-${FL_HUNG:-${ccl:+C4 $ccl}}}}}; FL_LEFT_WHY=$nr
  elif [ -z "$FL_LEFT_HUNG" ] && { [ "$RK_RC" = 0 ] || [ "$RK_RC" = 14 ]; }; then fl_hung_clear; fi
  [ -n "$FL_LEFT_HUNG" ] && fl_hung_mark "$FL_LEFT_HUNG"
  fl_c1 post
  fl_hsc_latch_end   # tray HSC latches after C9 against the start read: a bit set during the run is an HSC FAIL
}
# fl_12v <findings>: the same path in plain, full and smoke. The fleet launcher
# polls the host's BMC SDR once a second from the controller host (tools/tray-power-poll, read-only) for the whole run and merges power.csv
# into $FL_OUT (or FLEET_12V_CSV names it); fleet/tray_12v.py flags vr_pg != 0xff under load and a tray at <= 50 W while the
# others hold >= 600 W, never within 40 s of a reset, by tray 1-4 with its BDFs. No power.csv yet: FLEET_12V_POLL set means
# the launcher polls and judges it after collection; unset, the host has no BMC route.
fl_12v() {
  local c=${FLEET_12V_CSV:-$FL_OUT/power.csv} l
  if [ -s "$c" ]; then
    python3 "$FL_DIR/tray_12v.py" --csv "$c" --resets "$FL_OUT/reset_times.tsv" --loads "$FL_OUT/load_times.tsv" --map "$FL_OUT/map.txt" > "$FL_OUT/12v.txt" 2>&1
    while IFS= read -r l; do case $l in "12V FAIL "*) echo "$l" >> "$1"; printf '12v\t%s\n' "$l" >> "$FL_OUT/finding_stage.tsv" ;; *) echo "$l" ;; esac; done < "$FL_OUT/12v.txt"
  elif [ -n "${FLEET_12V_POLL:-}" ]; then echo "12V log: polled from ${FLEET_12V_POLL} by the launcher; its verdict is added after collection"
  else echo "12V log: no BMC route"; fi
}
# with no start reset (NO RESET at C2), an in-box ETH core "not active again at mesh open"
# is not graded as a box fault: its line becomes ETH INCOMPLETE ... start reset skipped; ETH not-active-at-open not judged, so the tray and
# the run are INCOMPLETE (rc 9) and the Repair recommendation names no hardware. With the start reset run, it stays FAIL.
fl_eth_noreset() {  # <stage name> <fault_eth line>: the finding line
  local nm=$1 fe=$2 hd
  if [ -z "$FL_C2_NONE" ] || [[ $fe != "ETH FAIL "*": ETH core "*" not active again at mesh open"* ]]; then echo "$nm $fe"; return 0; fi
  hd=${fe%%: ETH core *}
  echo "ETH INCOMPLETE ${hd#ETH FAIL }: $nm: start reset skipped; ETH not-active-at-open not judged: ${fe#"$hd": }; no glx_reset ran at C2 ($FL_C2_NONE)"
}

fl_report() {  # after the RESULT line: findings, skipped phases, one line per tray; adjusts RK_RC (8 a failed check, 10 gate)
  local f=$FL_OUT/findings.txt u nf nw ni nd nx xs ff fails dt fx xr='^\(ETH\|HANG\) INCOMPLETE ' wb; touch "$f"; fl_c3pend "C3 c2 did not judge them"; fl_t end
  grep -ahE '^NUMERIC (FAIL|WARN) ' "$OUT/numeric.txt" 2>/dev/null >> "$f"   # run.sh's numeric_report.py lines (WARN: unread burst outputs)
  fl_12v "$f"   # the per-second 12 V log: its 12V FAIL lines count under their trays
  echo "fleet $FLEET: $(grep -c . "$f") finding(s) in $(( ($(date +%s) - FL_T0) / 60 )) min; $(pp_tag); logs $FL_OUT"
  echo "Phase times: $(awk 'NR == FNR { if (!/^#/) { split($0, a, "\t"); n[a[2]] = "stage " a[1] " " a[3] }; next } FNR > 1 { printf "%s%s %d s", (FNR > 2 ? ", " : ""), p, $2 - t } { p = ($1 in n ? n[$1] : $1); t = $2 }' "$FL_STAGES_TSV" "$FL_OUT/phases.txt")"
  [ -s "$FL_OUT/busbar/parts.txt" ] && echo "Bus-bar part times: $(awk 'NR > 1 { printf "%s%s %d s", (NR > 2 ? ", " : ""), p, $2 - t } { p = $1; t = $2 }' "$FL_OUT/busbar/parts.txt")"   # qb is qb1; fleet/run_times.py takes medians
  grep -a . "$f" | grep -v '^GATE BUSY'
  cat "$FL_OUT"/heavy/vcore_chips.txt "$FL_OUT"/*/heavy/vcore_chips.txt 2> /dev/null   # per-chip VCORE under load, report only
  [ -n "${FL_LINKA_LINE:-}" ] && echo "$FL_LINKA_LINE"   # --repro link-a: its PASS or NOT RUN line (a HANG is a finding above)
  [ -n "${FL_AF_LINE:-}" ] && echo "$FL_AF_LINE"   # --repro abort-fabric: its PASS, DROP or NOT RUN line (each FAIL is a finding above)
  ff=$(fl_linka_hidden); [ -n "$ff" ] && { echo "$ff"; echo "$ff" >> "$f"; printf 'linka\t%s\n' "$ff" >> "$FL_OUT/finding_stage.tsv"; }   # a warn under its trays
  cat "$FL_OUT/history_info.txt" 2>/dev/null   # history before the last repair, never counted
  [ ${#FL_SKIPPED[@]} -gt 0 ] && echo "Not run: $(printf '%s; ' "${FL_SKIPPED[@]}" | sed 's/; $//')"
  ff=$(cat "$FL_OUT"/eth_pre_ext.txt* 2> /dev/null | sed -n 's/^ETH EXT \([^ ]*\) (tray \([1-4]\)) [^:]*: external ETH ports \[\([0-9, ]*\)\].*/\3 of \1 (tray \2)/p' | sed 's/, /,/' | sort -u | paste -sd';' | sed 's/;/; /g')
  [ -n "$ff" ] && echo "Out of scope (links that leave the box, QSFP to other nodes): external ETH ports $ff not up; not judged"
  ff=$(sed -n 's/^ETH NOT-CABLED \([^ ]*\) (tray \([1-4]\)) [^:]*: local ETH ports \[\([0-9, ]*\)\].*/\3 of \1 (tray \2)/p' "$FL_OUT/eth_notcabled.txt" 2> /dev/null | sed 's/, /,/g' | sort -u | paste -sd';' | sed 's/;/; /g')
  [ -n "$ff" ] && echo "Not cabled (INFO; wrap links this box is built without, partial torus): in-box ETH ports $ff down at pre and after the C2 reset; not judged"
  # a chunk whose load died because a tray lost chips (POWER/CHIP DROP) names every tray's chips in its CHUNK FAIL
  # line; it counts only on the trays that lost chips. The others get INCOMPLETE (numeric not checked), never FAIL.
  dt=$(grep -aoE '^(POWER|CHIP) DROP tray [1-4]' "$f" | awk '{ print $NF }' | sort -u | paste -sd' ')
  fx=$FL_OUT/findings_nodrop.txt   # only a dead chunk of a stage that had a drop (or of no known stage) is discounted
  awk -F'\t' 'NR == FNR { st[$2] = $1; if ($2 ~ /^(POWER|CHIP) DROP tray /) ds[$1] = 1; next }
    /CHUNK FAIL chunk [0-9]+\/[0-9]+ .*no chunk-end readback, the load died or hung/ && (!($0 in st) || st[$0] in ds) { next } { print }' \
    "$FL_OUT/finding_stage.tsv" "$f" > "$fx" 2>/dev/null || cp "$f" "$fx"
  [ -s "$FL_OUT/finding_stage.tsv" ] || grep -av 'CHUNK FAIL chunk [0-9]*/[0-9]* .*no chunk-end readback, the load died or hung' "$f" > "$fx"
  fails=$(grep -av ' WARN ' "$f" | grep -av "$xr" | grep -c .)   # external ports not checked are no fail
  # --known-dropped-tray: a finding that names only tray N (its chip and power rows) is the known drop's evidence, not a failed check
  [ "${KD_ON:-0}" = 1 ] && fails=$(grep -av ' WARN ' "$f" | grep -av "$xr" | sed 's/, BMC after: .*//; s/ (to OM-[^)]*)//' | awk -v n="$KD_TRAY" '{ t = $0; o = 0; k = 0
    while (match(t, /tray [1-4]([^0-9]|$)/)) { k++; if (substr(t, RSTART + 5, 1) != n) o = 1; t = substr(t, RSTART + RLENGTH) }
    if (o || !k) c++ } END { print c + 0 }')
  # tt-metal's "Device N: ..." errors in this run's logs, each mapped to its chip (BDF, tray 1-4) by the log's
  # own UMD id list and map.txt; listed under the chip's tray line (no grade change)
  find "$FL_OUT" -name '*.log' -size -20M 2>/dev/null | sort | xargs -r python3 "$FL_DIR/fleet_py.py" devmap "$FL_OUT/map.txt" "$FL_OUT" > "$FL_OUT/devmap.txt" 2>/dev/null
  wb=$(grep -a '^HANG INCOMPLETE whole box' "$f" | head -1 | cut -c17-156)   # a box-wide hang (no tray named)
  for u in 1 2 3 4; do   # a sustained drop line's "BMC after:" tail names every tray's power: only its subject tray counts;
    # a C3 QSFP line's " (to <HOST> tray N ...)" names the cable peer's tray on another host: not counted here
    [ "${KD_ON:-0}" = 1 ] && [ "$u" = "$KD_TRAY" ] && { echo "KNOWN-DROP tray $u (--known-dropped-tray=$u: not tested; $(kd_end) chips answer at the end)"; continue; }
    nd=""; ff=$f   # ff: the findings counted for this tray
    if [ -n "$dt" ] && [[ " $dt " != *" $u "* ]]; then ff=$fx
      nd=$(grep -avxFf "$fx" "$f" | grep -aE 'CHUNK FAIL chunk [0-9]+/[0-9]+ \(started [0-9:]+\): no chunk-end readback, the load died or hung' |
        grep -E "tray $u([^0-9]|$)" | head -1 | grep -oE 'chunk [0-9]+/[0-9]+' | head -1); fi   # only a line naming this tray
    nf=$(grep -av ' WARN ' "$ff" | grep -av "$xr" | sed 's/, BMC after: .*//; s/ (to OM-[^)]*)//' | grep -cE "tray $u([^0-9]|$)")
    nw=$(grep -a ' WARN ' "$f" | sed 's/ (to OM-[^)]*)//; s/ by tray [1-4]\( and tray [1-4]\)* power loss//' | grep -cE "tray $u([^0-9]|$)")   # the tray it blames is not this warn's
    ni=$(grep -a 'INCOMPLETE (test-infra)' "$f" | grep -cE "tray $u([^0-9]|$)")
    xs=$(grep -a "$xr" "$f" | sed -n "s/^ETH INCOMPLETE \([^ ]*\) (tray $u) [^:]*: external ETH ports \[\([0-9, ]*\)\] not up; not checked: \([^(]*\).*/\2 of \1 not checked: \3/p" |
      sed 's/, /,/g; s/ *$//' | paste -sd';' | sed 's/;/; /g'); nx=$(grep -c . <<< "$xs")
    local nv; nv=$(grep -a "$xr" "$f" | grep -aF 'start reset skipped; ETH not-active-at-open not judged' | grep -cE "tray $u([^0-9]|$)")
    if [ "$nf" -gt 0 ]; then echo "FAIL tray $u ($nf finding$([ "$nf" -gt 1 ] && echo s)$([ "$nw" -gt 0 ] && echo ", $nw warn"))"
    elif [ "$ni" -gt 0 ]; then echo "INCOMPLETE tray $u ($ni C5 pair(s) not run: test-infra$([ "$nw" -gt "$ni" ] && echo ", $((nw - ni)) warn"))"
    elif [ -n "$nd" ]; then echo "INCOMPLETE tray $u (lost nothing; numeric not checked: the load of $nd died when tray $(echo $dt | sed 's/ / and tray /g') lost chips$([ "$nw" -gt 0 ] && echo ", $nw warn"))"
    elif [ "$nx" -gt 0 ]; then echo "INCOMPLETE tray $u (external ETH ports $xs$([ "$nw" -gt 0 ] && echo ", $nw warn"))"
    elif [ "$nv" -gt 0 ]; then echo "INCOMPLETE tray $u ($nv ETH core(s) not active again at mesh open: start reset skipped; ETH not-active-at-open not judged$([ "$nw" -gt 0 ] && echo ", $nw warn"))"
    elif [ -n "$FL_C3STOP" ]; then echo "INCOMPLETE tray $u (no load ran: $FL_C3STOP$([ "$nw" -gt 0 ] && echo ", $nw warn"))"
    elif [ -n "$FL_PSU_STOP" ]; then echo "INCOMPLETE tray $u ($(fl_psu_where): $FL_PSU_STOP$([ "$nw" -gt 0 ] && echo ", $nw warn"))"
    elif [ -n "$wb" ]; then echo "INCOMPLETE tray $u (whole-box hang, no chip named, no hardware verdict: $wb$([ "$nw" -gt 0 ] && echo ", $nw warn"))"
    else echo "PASS tray $u$([ "$nw" -gt 0 ] && echo " ($nw warn)")"; fi
    # under it, one line per finding: part | stage | evidence | cause -> next; a clean tray costs nothing
    [ $((nf + nw + nx + nv)) -gt 0 ] && python3 "$FL_DIR/tray_why.py" "$f" "$u" --stages "$FL_OUT/finding_stage.tsv" ${dt:+--drop-trays "$dt"} 2>/dev/null
    awk -F'|' -v t="(tray $u)" 'index($1, t) { print "    tt-metal " $2 " = " $1 ": " $3 }' "$FL_OUT/devmap.txt" 2>/dev/null
  done | tee "$FL_OUT/trays.txt"   # the tray lines for the run's last line (rr_final); the loop only prints
  fl_hsc_latch_table   # the tray HSC latch before/after table, or its one not-read line
  [ -n "$FL_STOP" ] && case $FL_STOP in gate*) RK_RC=10 ;; esac
  [ "$RK_RC" = 0 ] && [ "$fails" -gt 0 ] && RK_RC=8   # 14 (a glx_reset left a chip missing) already counts as a fail
  [ "$RK_RC" = 0 ] && [ "${KD_ON:-0}" != 1 ] && grep -aq "$xr" "$f" && RK_RC=9   # external ports not checked: INCOMPLETE
  [ ${#FL_RESET_SKIPPED[@]} -gt 0 ] && { [ "$RK_RC" = 0 ] || [ "$RK_RC" = 8 ]; } && RK_RC=9   # INCOMPLETE
  # C5 pairs not run (test-infra): INCOMPLETE, not a pass; --known-dropped-tray keeps its rc 12
  [ ${#FL_C5_INC[@]} -gt 0 ] && [ "$RK_RC" = 0 ] && RK_RC=9   # also with --known-dropped-tray (INCOMPLETE wins)
  [ -n "${FL_LINKA_INC:-}" ] && [ "$RK_RC" = 0 ] && RK_RC=9   # --repro link-a did not run: INCOMPLETE, never a pass
  [ -n "${FL_AF_INC:-}" ] && [ "$RK_RC" = 0 ] && RK_RC=9   # --repro abort-fabric did not run: INCOMPLETE, never a pass
  [ -n "${FL_SETUP_INC:-}" ] && [ "$RK_RC" = 0 ] && RK_RC=9   # a stage stopped in fabric setup (test-side): INCOMPLETE, never a pass
  [ "$RK_RC" = 9 ] && VD_INC_WHY=$(fl_inc_why "$f")   # the verdict's reason, every cause
  return 0
}
fl_inc_why() {  # $1 findings: every cause of an INCOMPLETE verdict, '; ' joined, each cut to 220 chars
  local f=$1 x n w=() xr='^\(ETH\|HANG\) INCOMPLETE ' ux='s/ (tray ?)/ (unmapped BDF, no tray)/'
  x=$(grep -a "$xr" "$f" 2>/dev/null | grep -aF 'start reset skipped; ETH not-active-at-open not judged' | grep -oE '^ETH INCOMPLETE [0-9a-f:.]+ \(tray [1-4?]\)' | cut -d' ' -f3- | sed "$ux" | sort -u | paste -sd';' | sed 's/;/, /g')
  [ -n "$x" ] && w+=("ETH core(s) not active again at mesh open (start reset skipped; ETH not-active-at-open not judged: ${FL_C2_NONE:-no glx_reset ran}) on $x")
  x=$(grep -a "$xr" "$f" 2>/dev/null | sed -n 's/^ETH INCOMPLETE \([^ ]*\) (tray \([1-4?]\)) [^:]*: external ETH ports \[\([0-9, ]*\)\] not up.*/[\3] of \1 (tray \2)/p' | sed "$ux" | sort -u | paste -sd';' | sed 's/;/, /g')
  [ -n "$x" ] && w+=("external ETH ports not up, not checked: $x")
  x=$(grep -a '^HANG INCOMPLETE whole box' "$f" 2>/dev/null | head -1 | cut -c17-156); [ -n "$x" ] && w+=("box-wide hang, no chip named: $x")
  x=$(grep -a "$xr" "$f" 2>/dev/null | grep -avF 'start reset skipped; ETH not-active-at-open not judged' | grep -av 'external ETH ports .* not up' | grep -av '^HANG INCOMPLETE whole box')
  n=$(printf '%s' "$x" | grep -c .); [ "$n" -gt 0 ] && w+=("$n other INCOMPLETE finding(s), first: $(head -1 <<< "$x")")
  [ ${#FL_RESET_SKIPPED[@]} -gt 0 ] && w+=("reset not exercised: $(printf '%s, ' "${FL_RESET_SKIPPED[@]}" | sed 's/, $//')")
  [ ${#FL_C5_INC[@]} -gt 0 ] && w+=("${#FL_C5_INC[@]} C5 pair(s) did not run (test-infra: out of processes/threads)")
  [ -n "${FL_LINKA_INC:-}" ] && w+=("the link-A repro did not run ($FL_LINKA_INC)")
  [ -n "${FL_AF_INC:-}" ] && w+=("the abort-fabric repro did not run ($FL_AF_INC)")
  [ -n "${FL_SETUP_INC:-}" ] && w+=("fabric setup failed, so these parts did not run ($FL_SETUP_INC); test-side setup or software, not a hardware fault")
  [ -n "${RS_BUDGET_INC:-}" ] && w+=("$RS_BUDGET_INC")
  [ -n "${FL_PSU_STOP:-}" ] && w+=("$FL_PSU_STOP")
  [ -n "${FL_C3STOP:-}" ] && w+=("$FL_C3STOP, so no load stage ran")
  case ${RK_WHAT:-} in "INCOMPLETE: "*) w+=("${RK_WHAT#INCOMPLETE: }") ;; esac
  [ ${#w[@]} -gt 0 ] || return 0
  printf '%s\n' "${w[@]}" | cut -c1-220 | paste -sd'\t' | sed 's/\t/; /g'
}
fl_result_say() {  # $1 run.sh's RESULT line: a failed check fl_report prints as a FAIL tray line is named in it, never 'no loss'
  local r h="" dt=""; r=$(cat "$1")
  if [[ $r == *"no loss"* ]]; then
    dt=$(grep -aoE '^(POWER|CHIP) DROP tray [1-4]' "$FL_OUT/findings.txt" 2>/dev/null | awk '{ print $NF }' | sort -u | paste -sd' ')
    h=$(python3 "$FL_DIR/tray_why.py" "$FL_OUT/findings.txt" --result --also "$OUT/numeric.txt" --stages "$FL_OUT/finding_stage.tsv" \
      ${dt:+--drop-trays "$dt"} $([ "${KD_ON:-0}" = 1 ] && echo "--known-tray $KD_TRAY") 2>/dev/null)
  fi
  if [ -n "$h" ] && [[ $r =~ ^RESULT:\ no\ loss\ (\(.*\))$ ]]; then r="RESULT: $h; no chip lost ${BASH_REMATCH[1]}"
  elif [ -n "$h" ]; then r=${r//no loss/"$h; no chip lost"}; fi
  echo "$r"
}

fl_verdict() {  # before the RESULT line: a skipped reset and what it leaves (FL_RESULT replaces run.sh's RESULT text)
  local w inc nrs=${#FL_RESET_SKIPPED[@]}; FL_RESULT=""
  fl_psu_join 1   # the last stage's PSU read, run in the background over C9, is in before the RESULT (FL_PSU_STOP: INCOMPLETE)
  [ -n "${FL_MU_SKIP:-}" ] && echo "$FL_MU_SKIP"   # the run summary says the site skip too
  if [ -n "${FL_YIELD:-}" ]; then   # the device went to a queued broker job: never a fault, never a pass
    local cnt=""; declare -F rs_count > /dev/null && cnt=$(rs_count 2> /dev/null | head -1)
    if [ -n "${FL_YIELD_C9:-}" ]; then FL_RESULT="INCOMPLETE: $FL_YIELD during stage $FL_YIELD_AT; chips after the C9 reset: ${cnt:-not counted}"
    else FL_RESULT="INCOMPLETE: $FL_YIELD during stage $FL_YIELD_AT; no C9 reset ran (${FL_STOP:-see the Reset: line}); chips: ${cnt:-not counted}"; fi
    case $RK_RC in 0|8|9) RK_RC=9 ;; *) FL_RESULT="$FL_RESULT; $RK_WHAT" ;; esac   # a chip the C9 glx_reset left missing (14) stays a FAIL
    return 0
  fi
  if [ -n "${FL_C3STOP:-}" ]; then   # C3 discovery failed, so nothing loaded: INCOMPLETE naming the chips, never a pass
    [ ${#FL_RESET_SKIPPED[@]} -gt 0 ] && echo "RESET NOT EXERCISED: $(printf '%s; ' "${FL_RESET_SKIPPED[@]}" | sed 's/; $//')"
    FL_RESULT="INCOMPLETE: $FL_C3STOP, so no load stage ran; chips: ${FL_C3STOP_CHIPS:-none named by tt-umd (fleet/disc_*.json)}"
    case $RK_RC in 0) RK_RC=$([ -n "${FL_C3STOP_CHIPS:-}" ] && echo 8 || echo 9) ;; 8|9) ;; *) FL_RESULT="$FL_RESULT; $RK_WHAT" ;; esac   # 14 (a glx_reset left a chip missing), 1 stay
    return 0
  fi
  [ -n "${FL_LINKA_INC:-}" ] && [ "$RK_RC" = 0 ] && [ $nrs -eq 0 ] && { FL_RESULT="INCOMPLETE: no loss, but the link-A repro did not run ($FL_LINKA_INC)"; RK_RC=9; }
  [ -n "${RS_BUDGET_INC:-}" ] && [ "$RK_RC" = 0 ] && [ $nrs -eq 0 ] && { FL_RESULT="INCOMPLETE: no loss, but $RS_BUDGET_INC"; RK_RC=9; }
  inc=${FL_INFRA_INC:+a part did not run ($FL_INFRA_INC)}   # a part that could not load (test-infra), named with its reason
  [ -n "${FL_MU_INC:-}" ] && inc="${inc:+$inc; }the max-util part did not run ($FL_MU_INC)"   # max-util is never a quiet skip
  [ -n "${FL_PV_INC:-}" ] && inc="${inc:+$inc; }the power virus part did not complete ($FL_PV_INC)"   # the power virus (its part and the TRAYTEST_MIX_SWAP=virus segments) has its own reason
  [ -n "${FL_BB_INC:-}" ] && inc="${inc:+$inc; }$FL_BB_INC"   # every bus-bar chunk failed at device open (off-box links only)
  [ -n "$inc" ] && [ "$RK_RC" = 0 ] && [ $nrs -eq 0 ] && { FL_RESULT="INCOMPLETE: no loss, but $inc"; RK_RC=9; }   # all print
  [ -n "${FL_PSU_STOP:-}" ] && [ "$RK_RC" = 0 ] && { FL_RESULT="INCOMPLETE: $FL_PSU_STOP: $(fl_psu_where); no loss"; RK_RC=9; }   # never a pass
  # a fabric-setup failure is test-side, so the stage it stopped did not test
  # anything: INCOMPLETE (rc 9), never a pass; a hardware FAIL finding beside it keeps fl_report's rc 8
  [ -n "${FL_SETUP_INC:-}" ] && [ "$RK_RC" = 0 ] && [ $nrs -eq 0 ] && ! grep -av ' WARN ' "$FL_OUT/findings.txt" 2>/dev/null | grep -aqv '^ETH INCOMPLETE ' &&
    { FL_RESULT="INCOMPLETE: no loss, but fabric setup failed, so these parts did not run ($FL_SETUP_INC); test-side setup or software, not a hardware fault"; RK_RC=9; }
  [ -n "${FL_AF_INC:-}" ] && [ "$RK_RC" = 0 ] && [ $nrs -eq 0 ] && { FL_RESULT="INCOMPLETE: no loss, but the abort-fabric repro did not run ($FL_AF_INC)"; RK_RC=9; }
  if [ $nrs -eq 0 ]; then   # C5 pairs that could not start (test-infra): never a pass
    # (with --known-dropped-tray too; INCOMPLETE wins and run.sh's kd_tag names the known drop on the RESULT line)
    [ ${#FL_C5_INC[@]} -gt 0 ] && [ "$RK_RC" = 0 ] && [ -z "${failed:-}" ] && [ -z "${ccl:-}" ] &&
      FL_RESULT="INCOMPLETE: no loss ($total/$total tries, $(( ($(date +%s) - T0) / 60 )) min), but ${#FL_C5_INC[@]} C5 pair(s) did not run (test-infra: out of processes/threads)"
    return 0
  fi
  w=$(printf '%s; ' "${FL_RESET_SKIPPED[@]}" | sed 's/; $//'); echo "RESET NOT EXERCISED: $w"
  if [ -n "$FL_LEFT_HUNG" ]; then
    # rc 9 'no loss': rk_end's end reset was gate-skipped (C9/C9b) and nothing else is wrong; the hang is what is left
    case $RK_RC in 0|7|9) [ "$RK_RC" != 9 ] || [[ $RK_WHAT == "INCOMPLETE: no loss"* ]] &&
      { FL_RESULT="$FL_LEFT_HUNG left hung: no reset could run (${FL_LEFT_WHY:-$w})"; RK_RC=7; } ;; esac
  elif [ "$RK_RC" = 0 ] && [ -n "$failed" ]; then FL_RESULT="$failed; read-only recheck (no reset could run): no loss"; RK_RC=1
  elif [ "$RK_RC" = 0 ]; then FL_RESULT="INCOMPLETE: no loss ($total/$total tries, $(( ($(date +%s) - T0) / 60 )) min), but the reset was not exercised"; fi
  return 0
}
