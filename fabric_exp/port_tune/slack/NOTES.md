# Slack study: per-side headroom at saturation, main against the port (2026-11-05)

User-approved (2026-11-05): how much headroom each ERISC has at saturation, for the report page and to answer which side
bounds the rows at the wire. Local study patches only; nothing here goes into the PR branch, nothing was committed.

**Status (2026-11-05, after section 10): the wall-clock-spin ladder of sections 2-7 is SUPERSEDED for slack values by
the mcycle-spin settling A/B (section 10, jobs 840 / 841).** The bound sides of section 6 hold under both spins; the
wall-clock D is not core cycles, so its slack numbers are kept only as `wallclock_*` records.

Method (the study's one-side delay A/B): a fixed busy-wait of D ticks per packet on ONE side, raised along the ladder
0, 4, 8, 16, 32, 64 (128 where a side had not clearly dropped at 64). Slack = the largest D with no drop beyond noise.
A side with slack 0 bounds the row. Both routers ran in the same jobs, interleaved per arm, in two passes with the arm
order reversed.

## 1. Trees and roots

- **main:** /home/aagarwal/tt-metal-main at d2bd12fcf1f, its Release build. Its only local change is the
  test_tt_fabric.cpp TT_FABRIC_NO_BW_TELEMETRY switch.
- **port:** /home/aagarwal/tt-metal-port-slack, a new worktree made for this study.
  - At origin/aagarwal/fabric-router-mainline e7fe1899049, detached.
  - Carries exactly local_patches/test_tt_fabric_no_bw_telemetry.patch.
  - Built here (Release, same CMake options as main's build, target test_tt_fabric).
  - Submodules were copied from tt-metal-mainline-cleanup (same commits), and the CPM cache hard-linked from
    tt-metal-port-pr.
  - tt-metal-port-pr belongs to the PR perf-refresh agent and was not used.
- **Runtime roots:** /home/aagarwal/tt-metal-slack-roots/<router>_<arm>, built by mk_roots.py.
  - Each root is a symlink farm of its tree (the cisp technique) with its own generated/.
  - Only the spin's call-site files and slack_study_spin.hpp are real files.
  - The router JIT-compiles from the root, so arms differ only in device code; one host binary per router.
  - Arms: plain (base runs), z (D=0 with the patch, offline check only), tx4..tx128 and rx4..rx128.
  - roots_manifest.json holds the md5 of every edited file. slack_runs.sh checks it, both trees' HEAD and both trees'
    local diffs before any run.

## 2. Patch design

One header, identical in both trees: `tt_metal/fabric/hw/inc/edm_fabric/slack_study_spin.hpp`
(patches/slack_study_spin.hpp.in). Each arm's root holds its own copy with that arm's two constants; the call sites
are the same in every patched arm of a router.

```cpp
namespace slack_study {
constexpr uint32_t tx_spin_cycles = @TX@;   // E0, per packet sent
constexpr uint32_t rx_spin_cycles = @RX@;   // E1, per packet received and handled
template <uint32_t cycles>
FORCE_INLINE void spin() {
    if constexpr (cycles > 0) {
        const uint32_t start = get_timestamp_32b();          // RISCV_DEBUG_REG_WALL_CLOCK_L (risc_common.h)
        while (get_timestamp_32b() - start < cycles) {
        }
    }
}
}
```

The counter is the documented RISC wall clock's low word, through the existing helper `get_timestamp_32b()`.
`risc_common.h` is identical in both trees. There is no inline asm, no register pinning, and no new compile args;
at D=0 the spin compiles out. The sites are the same anchor in both routers:

- **E0:** right after the per-packet packets-received register update to the remote router, the last step of a send.
- **E1:** right after the packet is forwarded or delivered and its slot retired.

Main has two copies of each step: the regular path and the speedy path (`super_speedy_mode`). Its 2D routers here run
the regular path (ENABLE_SPEEDY_VC0=0, 4 VC0 sender channels); its NeighborExchange routers run the speedy path
(ENABLE_SPEEDY_VC0=1, 1 sender channel), from the device caches' named args. Both copies are patched.

Port (patches/port_sites.patch):

```diff
--- a/tt_metal/fabric/hw/inc/edm_fabric/router/router_receiver.hpp
 #include "router_telemetry.hpp"
+#include "tt_metal/fabric/hw/inc/edm_fabric/slack_study_spin.hpp"  // LOCAL slack study, never committed
@@ handle()
     retire<VC>(progress, sources, source_channel(packet.src_ch_id));
+    slack_study::spin<slack_study::rx_spin_cycles>();  // LOCAL slack study: E1, per packet received and handled
     return true;
--- a/tt_metal/fabric/hw/inc/edm_fabric/router/router_sender.hpp
 #include "router_telemetry.hpp"
+#include "tt_metal/fabric/hw/inc/edm_fabric/slack_study_spin.hpp"  // LOCAL slack study, never committed
@@ send<CH>()
     remote_update_ptr_val<RemoteReceiverChannel<VC>::packets_received_stream, sender_txq_id>(1);
+    slack_study::spin<slack_study::tx_spin_cycles>();  // LOCAL slack study: E0, per packet sent
     sent++;
```

Main (patches/main_sites.patch):

```diff
--- a/tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
 #include "hostdevcommon/fabric_telemetry_msgs.h"
+#include "tt_metal/fabric/hw/inc/edm_fabric/slack_study_spin.hpp"  // LOCAL slack study, never committed
@@ send_next_data()
     SenderLinkTransport::remote_update_ptr_val<to_receiver_pkts_sent_id>(1U);
+    slack_study::spin<slack_study::tx_spin_cycles>();  // LOCAL slack study: E0, per packet sent
 }
@@ run_receiver_channel_step_impl(), end of the can_send_to_all_local_chip_receivers block
             if constexpr (!enable_first_level_ack) {
                 increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
             }
+            slack_study::spin<slack_study::rx_spin_cycles>();  // LOCAL slack study: E1, per packet received and handled
--- a/tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp
@@ run_sender_channel_step_speedy(), the inlined send
             SenderLinkTransport::remote_update_ptr_val<to_receiver_pkts_sent_id>(1U);
+            slack_study::spin<slack_study::tx_spin_cycles>();  // LOCAL slack study: E0, per packet sent
@@ run_receiver_channel_step_speedy()
         receiver_state.unacked_sends++;
+        slack_study::spin<slack_study::rx_spin_cycles>();  // LOCAL slack study: E1, per packet received and handled
```

## 3. Offline check (before any device run)

offline_check.sh ran CPU-only mock builds on fabric_exp/a09u32_cluster_desc.yaml, one process per slack_rows.yaml
test, for all 28 roots: 56 builds, all reached "Launching programs". img_check.py then paired every router image
(E0 = active_erisc, E1 = subordinate_active_erisc) with the plain root's by its named compile-time args. Output:
img_check_mock.txt. After the runs the same check ran on the device JIT caches (img_check_device.txt, 128 images per
router). Both print IMG_CHECK_OK.

- **z (D=0 patched):** every image byte-identical to plain (loadable segments), both routers, both fabrics.
- **tx<D>:** every E0 image differs, in one function only (port `fabric_router::sender::run`, main `kernel_main`).
  Every E1 image is byte-identical. Port: +6 instructions per sender channel (+24 on 2D, 4 channels; +6 on NX). Main:
  +4 to +36 instructions, one copy per inlined send.
- **rx<D>:** every E1 image differs, in one function only (port `fabric_router::receiver::run`, main `kernel_main`).
  Every E0 image is byte-identical.
- **Register allocation moves around the sites.** Port 2D E1 images change by -4 / -1 / +6 instructions, not a clean
  add. Main's D=16 arms allocate registers differently from its other D values (cause not looked into). So an arm
  adds the spin plus a small code-generation shift in the same function. That shift is the same in every D>0 arm
  except main's D=16.
- **Fit:** largest E0 + largest E1 loadable bytes against the 26,624 B ERISC kernel config buffer:

  | router | plain | patched |
  | --- | ---: | ---: |
  | main | 18,644 | 18,788 |
  | port | 15,580 | 15,676 |

  Every program fits; all JIT links succeeded.

## 4. Runs

All on bh-lb-120-a09u32 through the tt-device broker, TT_FABRIC_NO_BW_TELEMETRY=1. The metric is test_tt_fabric's
own summary CSV: cycles/packet = 1.35e9 / avg_packets_per_s.

| broker job | pass | order | time |
| --- | --- | --- | --- |
| 833 | a | base, tx4..tx64, base2, rx4..rx64; main then port per arm | 535 s |
| 836 | a | main_rx128 only: main's Mesh E1 had not clearly dropped at 64; every other side had | 22 s |
| 837 | b | main_rx128, then rx64..rx4, base2, tx64..tx4, base; port then main per arm | 235 s |

No hang and no failed run; the post-job health gate was healthy after each job. Rows (slack_rows.yaml):

- Mesh1HopEW unicast write: chip [0,0] -> [0,1], 2D mesh, one flow, 50k packets, at 2048 (the contrast row), 4096
  and 4352 B.
- NeighborExchange unicast write: the CI ubench row at 4096 B, num_links 1 and 2, 3 iterations of 100k packets.

Results: fabric_exp/results/slack/s<pass>_<router>_<arm>.{log,csv,_detail.csv}. Table: slack_report.py
(slack_report.txt has every pass value; slack_summary.json).

## 5. Results (wall-clock spin; SUPERSEDED for slack values by section 10)

Each cell is the rate change against the D=0 mean, in cycles/packet, averaged over the two passes; `*` marks a drop
beyond the noise band.

- **Passes:** the two passes agree within 0.1-0.3 at almost every point. Exceptions are main's NeighborExchange rows,
  which flip between two modes about 3 cycles apart (base runs 126.4 / 129.5), and the port Mesh 4352 E0 at D=4/8
  (142.1 / 141.1).
- **Noise band:** the largest base-run spread over the router's rows of that test, at least 0.5. That gives 3.07 for
  main's NeighborExchange rows and 0.5 for everything else.
- **Slack:** the user's reading (no drop beyond the band).
- **Clear drop:** the same reading at a 4-cycle threshold, above any noise seen. The two differ only for main's Mesh
  E1.
- **% of wire:** build_page.py's 400G frame model (2D: 118.5 at 4096 B, 125.4 at 4352 B; NeighborExchange 117.2 at
  4096 B).

| row | router | D=0 rate (runs) | % of wire | side | D=4 | D=8 | D=16 | D=32 | D=64 | D=128 | slack (knee) | clear-drop slack |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| Mesh1HopEW 2048 B | main | 223.84 (223.84 / 223.85 / 223.77 / 223.89) | 27.6% | E0 | +30.2* | +30.2* | +40.6* | +50.6* | +84.8* |  | 0 (0-4) | 0 (0-4) |
| | | | | E1 | +0.4 | +0.1 | +0.1 | +0.8* | +1.1* | +59.1* | 16 (16-32) | 64 (64-128) |
| Mesh1HopEW 2048 B | port | 122.41 (122.41 / 122.45 / 122.35 / 122.43) | 50.4% | E0 | +19.9* | +19.9* | +22.2* | +41.8* | +83.3* |  | 0 (0-4) | 0 (0-4) |
| | | | | E1 | +11.1* | +11.1* | +12.1* | +28.1* | +63.0* |  | 0 (0-4) | 0 (0-4) |
| Mesh1HopEW 4096 B | main | 222.53 (222.50 / 222.62 / 222.51 / 222.49) | 53.3% | E0 | +31.8* | +31.9* | +39.9* | +50.2* | +87.4* |  | 0 (0-4) | 0 (0-4) |
| | | | | E1 | -1.8 | -0.3 | -0.1 | +0.7* | +1.3* | +60.1* | 16 (16-32) | 64 (64-128) |
| Mesh1HopEW 4096 B | port | 126.85 (126.83 / 126.88 / 126.88 / 126.81) | 93.4% | E0 | +14.1* | +14.7* | +16.4* | +36.9* | +78.9* |  | 0 (0-4) | 0 (0-4) |
| | | | | E1 | +8.0* | +8.0* | +8.9* | +24.4* | +58.9* |  | 0 (0-4) | 0 (0-4) |
| Mesh1HopEW 4352 B | main | 222.72 (222.69 / 222.72 / 222.69 / 222.78) | 56.3% | E0 | +31.6* | +31.6* | +39.8* | +49.5* | +87.2* |  | 0 (0-4) | 0 (0-4) |
| | | | | E1 | -2.1 | -0.4 | -0.4 | +0.6* | +0.8* | +59.9* | 16 (16-32) | 64 (64-128) |
| Mesh1HopEW 4352 B | port | 128.46 (128.45 / 128.43 / 128.48 / 128.47) | 97.6% | E0 | +13.1* | +13.3* | +14.7* | +35.1* | +77.2* |  | 0 (0-4) | 0 (0-4) |
| | | | | E1 | +6.7* | +6.7* | +7.6* | +22.9* | +57.3* |  | 0 (0-4) | 0 (0-4) |
| NeighborExchange 4096 B, 1 link | main | 126.44 (126.44 / 126.43 / 126.41 / 126.46) | 92.7% | E0 | +30.1* | +30.9* | +31.4* | +49.9* | +85.6* |  | 0 (0-4) | 0 (0-4) |
| | | | | E1 | +1.2 | +1.1 | +1.1 | +16.3* | +43.8* | +110.4* | 16 (16-32) | 16 (16-32) |
| NeighborExchange 4096 B, 1 link | port | 121.80 (121.79 / 121.80 / 121.80 / 121.81) | 96.2% | E0 | +0.1 | +0.2 | +0.2 | +4.4* | +45.5* |  | 16 (16-32) | 16 (16-32) |
| | | | | E1 | -0.0 | -0.0 | -0.0 | +14.5* | +50.3* |  | 16 (16-32) | 16 (16-32) |
| NeighborExchange 4096 B, 2 links | main | 127.27 (126.43 / 129.50 / 126.45 / 126.72) | 92.1% | E0 | +29.3* | +30.1* | +35.1* | +49.0* | +84.8* |  | 0 (0-4) | 0 (0-4) |
| | | | | E1 | -0.2 | +1.3 | +0.7 | +15.8* | +44.2* | +109.6* | 16 (16-32) | 16 (16-32) |
| NeighborExchange 4096 B, 2 links | port | 121.80 (121.80 / 121.80 / 121.80 / 121.80) | 96.2% | E0 | +0.1 | +0.2 | +0.2 | +4.4* | +45.5* |  | 16 (16-32) | 16 (16-32) |
| | | | | E1 | +0.0 | +0.0 | -0.0 | +14.5* | +50.3* |  | 16 (16-32) | 16 (16-32) |

## 6. Which side bounds each row (from these A/Bs)

- **Port, Mesh1HopEW 1-hop E/W write, 4096 B and 4352 B** (93.4% and 97.6% of the wire): **E0 and E1** by the
  definition. Both drop at the first rung (D=4).
  - E0 pays about twice what E1 pays at every D: 4352 B E0 +13.1 / +35.1 / +77.2 against E1 +6.7 / +22.9 / +57.3 at
    D=4 / 32 / 64.
  - So E0 is the tighter side, and E1 has some headroom, but less than one minimal spin.
  - The contrast row at 2048 B (50% of the wire) reads the same: both sides, E0 +19.9, E1 +11.1 at D=4. The port's
    2D 1-hop rows are bound by its ERISCs at every size tested.
- **Port, NeighborExchange 4096 B, 1 and 2 links** (96.2% of the wire): **neither ERISC**.
  - Both sides take D=16 with no change (at most +0.2), and both knees lie between 16 and 32 (E0 +4.4, E1 +14.5 at
    D=32).
  - The row is bound elsewhere: the wire (the frame model says 117.2, measured 121.8) or the test's worker. This A/B
    does not separate the two.
- **Main, Mesh1HopEW at every size** (about 223 cycles/packet, 28-56% of the wire): **E0**. It pays +30 to +32 at
  D=4. E1 has a lot of headroom: small rises of +0.6 to +1.3 from D=32, and the hard knee between 64 and 128 (+60 at
  128).
- **Main, NeighborExchange 4096 B, 1 and 2 links** (92% of the wire): **E0**. It pays +29 to +30 at D=4. E1's knee
  lies between 16 and 32 (+16 at 32), with about +1 at D=4..16, within the 3-cycle mode noise of these rows.

Against earlier answers (RESULTS §2, slack.json):

- **Main:** E0-bound on its 1-hop writes, as in RESULTS §2 (2026-10-09, older main, 2 KiB: neighbor-exchange write and
  mesh write TX-bound).
- **Port NeighborExchange:** "neither" matches the study's reading of v4trm / v4rxv2i3 / v4txa11 at 4352 B.
- **Port Mesh 4352 B, flagged here; SETTLED in section 10:** the study's v4txa11 set (2026-11-03, mcycle spin, slack from the N+12
  model: 44 - c32) read E1 slack 24.6 and E0 5.8, so "E0-bound". Here E1 also drops at D=4.
  - At D=32 the two instruments agree closely: port +35.1 / +22.9 against v4txa11's c32 +38.2 / +19.4.
  - The difference is at small D, which the study's 0/32/96 ladder never tested. The routers also differ (port drift,
    memory project-study-port-drift).
  - The cheapest settling A/B, not run: the same sites with the study's mcycle spin (`fabric_experiment_spin`, a 1-cycle
    CSR read). If the port's 2D E1 then shows no cost at D=4, the wall-clock spin's minimum duration exceeds E1's
    headroom (hypothesis 1 below).

## 7. Caveats and hypotheses (not settled by A/B)

1. **WRONG in part (section 10):** a D=1 or D=2 wall-clock spin costs the same as the mcycle spin there, so there is no
   long minimum at D<=2. The wall-clock spin does step up at D=4, by ~10 cycles against mcycle, but the port's 2D E1
   also drops at D=1 under the mcycle spin (+1.3). As first written: **The minimum spin is longer than D=4.** On every side that drops at D=4, the cost is about flat from D=4 to D=16
   (port Mesh 4352 B E0: +13.1 / +13.3 / +14.7). So D up to 16 runs about one fixed minimum. Hypothesis: the
   debug-register load is slow, so a spin of 16 ticks or less ends at its first poll.
   - So "slack 0" here means less headroom than that minimum, not zero.
   - The minimum's length in core cycles was not measured. The study's mcycle spin cost N+12.
2. **Resolved in part by section 10:** with the mcycle spin, port 2D E1 pays +1.3-1.6 at D=1..4, so of the +6.7 at
   wall-clock D=4 about 5 cycles are the wall-clock spin's longer length. The ~1.3 left stays unattributed between the
   shortest spin's own work and the register-allocation shift: both are present at every D>0. As first written:
   **Port 2D E1 at D=4 (+6.7 to +11.1)** could be the minimum spin exceeding E1's headroom (hypothesis 1), or the
   register-allocation shift in receiver::run (section 3). Both are hypotheses. The port's NeighborExchange E1 pays 0.0
   at D=4..16 with its own code change, so the change is not costly there.
3. **Still open after section 10 (TODO.md):** with the mcycle spin main's E0 pays +10.4-12.7 at D=1 against the
   port's +1.2-2.4. As first written: **Main's E0 pays more for the same spin than the port's E0:** +30 to +32 at D=4 against +13 to +20. The cause is not
   isolated. Hypotheses: code generation in main's kernel_main around its several inlined send copies, or the spin also
   delaying work main's E0 does after a send. It does not change the bound side: only a side without headroom pays at
   all.
4. **Cost per tick past the knee:** a saturated side pays 0.9-1.3 cycles/packet per tick.
   - Main NeighborExchange E1 from 64 to 128: 1.03.
   - Port E1 from 32 to 64: 1.08-1.12. Port E0 from 32 to 64: 1.28-1.31.
   - Whether the wall clock ticks at the 1.35 GHz measurement clock was not checked (hypothesis: about 1:1, from E1).
   - So D is in counter ticks. The slack values do not compare number for number with the study's N+12 cycles.
5. **Main Mesh E1 runs 1.8-2.1 faster at D=4** (4096 / 4352 B, both passes). It is reproducible and tiny; the
   mechanism is unknown.

## 8. Data

- **report/data/slack.json:** 355 records added, for main and router `port@e7fe1899049`, all dated 2026-11-05 and
  marked `same_job_with_main: true`. The 1,538 earlier records are unchanged (checked against a copy). No missing
  required fields, no duplicate ids.
  - **mcycle set** (jobs 840 / 841), the current set: 200 records, `instrument` "mcycle spin ...".
    - `rate_base` with the four base runs in `passes`.
    - `slack_E0` / `slack_E1` (the knee reading) and `slack_E*_clear` (the 4-cycle reading), unit "D (core cycles,
      mcycle spin)".
    - `rate_E*_D<d>`, and `rate_E*_wallclock_D<d>` (the wall-clock floor arms at D 1, 2, 4 from the same jobs).
  - **wall-clock set** (jobs 833 / 836 / 837): 155 records, SUPERSEDED. Every metric carries a `wallclock_` prefix and
    every note starts with "SUPERSEDED by the mcycle set ...", so readers of `rate_base` / `slack_E*` see only the
    mcycle set.
  - Optional fields: `num_links`, `instrument`. The 2-link NeighborExchange row has config "two ERISCs, 2 links".
- **Builder change:** report/scripts/build_slack.py has `PORT_SETS` (the two sets) and `port_slack_records()`, which
  loads port_tune/slack/slack_report.py's `load(tag)` / `analyse(...)`, plus a docstring line. Before the change the
  builder reproduced slack.json exactly.
- **Page:** not changed here. build_page.py (the coordinator's, 17:38) holds `port@e7fe1899049` via `SLACK_HOLD`
  pending this A/B.

## 9. Files

- mk_roots.py, patches/: the roots, the site patches, the header template; roots_manifest.json.
- offline_check.sh, img_check.py: the offline check (img_check_mock.txt, img_check_device.txt).
- slack_rows.yaml, slack_runs.sh: the rows and the broker runner.
- slack_report.py: the reading (slack_report.txt, slack_summary.json; mcycle set: slack_report_mcycle.txt,
  slack_summary_mcycle.json; floor arms: slack_report_floor.txt). `--tag m --prefix m` reads the mcycle set.
- img_check_mock_mcycle.txt, img_check_device_mcycle.txt: the offline check of the mcycle and floor roots, and the
  same check on their device JIT caches (IMG_CHECK_OK).
- Device caches (~20 GB) and mock caches were deleted after the checks; the checks' outputs above are the record.
- Caches ~/.tt-metal-cache-slack-<router>-<arm>. Roots /home/aagarwal/tt-metal-slack-roots. Worktree
  /home/aagarwal/tt-metal-port-slack. All are local to this study and can be removed when no longer needed.

## 10. Settling A/B: the study's mcycle spin at the same sites (2026-11-05, jobs 840 / 841)

The coordinator asked for this after the first report, per CLAUDE.md: the port's 2D E1 dropping at D=4 conflicted with
the study's v4txa11 reading (E1 slack 24.6 at 4352 B), so the conflict had to be settled by A/B before the numbers go on
the page.

### 10.1 What changed

- **Spin:** only the header. patches/slack_study_spin_mcycle.hpp.in spins on `csrr mcycle`, the study's
  `fabric_core_cycles` / `fabric_experiment_spin`. The call sites are the same files and edits as sections 2-3.
- **Arms:** mz, mtx<D>, mrx<D> (mk_roots.py).
- **Floor arms:** the wall-clock spin at D=1, 2 (tx1, tx2, rx1, rx2), and D=4 as an anchor to the first set, run in
  the same jobs.
- **Offline check:** img_check_mock_mcycle.txt, IMG_CHECK_OK.
  - mz is byte-identical to plain on both routers and both fabrics.
  - Every mtx/mrx and tx1/tx2/rx1/rx2 arm changes only E0 or only E1 images, in the sender or receiver function.
  - The port's 2D E1 shows the same small register shift as the wall-clock arms (-5..+5 instructions).
  - Everything fits (largest E0+E1: main 18,772 B, port 15,676 B).
- **Disk incident:** 16 mock builds first failed because the disk filled. My unpruned mock caches (39 GB in the
  scratchpad) were part of that; they are pruned now and offline_check.sh prunes after each build. The failed roots
  were rebuilt.
  - Jobs 833 / 836 / 837 ran before that window. All 50 CSVs are complete and no log shows a write error.
- **Arms run:**
  - Port: D = 1, 2, 4, 8, 16, 32 on E0 and E1 (mcycle), plus the floor arms.
  - Main: mcycle E0 at D = 1, 2, 4, 32 and E1 at D = 4, 16, 32, 64, 128; wall-clock E0 at D = 1, 2, 4.
- **Jobs:** 840 (pass a, 34 runs, 682 s) and 841 (pass b, arms reversed, 235 s). No hang, the device was healthy after
  each job, and all 68 CSVs are complete.
- **Anchors reproduce the first set:** port E0 wall D=4 +13.0 (was +13.1), E1 +6.7 (was +6.7); main E0 +31.6 (was
  +31.6). The D=0 rates match within 0.05 (e.g. port Mesh 4352 B 128.44 against 128.46).

### 10.2 The spin's floor (cost at D=1/2 against D=4; mean of the two passes, cycles/packet)

| row | router | side | wall D=1 | mcycle D=1 | wall D=2 | mcycle D=2 | wall D=4 | mcycle D=4 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Mesh1HopEW 2048 B | main | E0 | +10.4 | +10.4 | +29.1 | +11.4 | +30.2 | +11.3 |
| Mesh1HopEW 2048 B | port | E0 | +2.4 | +2.4 | +4.2 | +3.9 | +19.8 | +3.9 |
| Mesh1HopEW 2048 B | port | E1 | +1.2 | +1.0 | +1.6 | +1.3 | +11.1 | +1.3 |
| Mesh1HopEW 4096 B | main | E0 | +11.4 | +11.3 | +31.1 | +12.2 | +31.9 | +12.2 |
| Mesh1HopEW 4096 B | port | E0 | +1.3 | +1.2 | +2.6 | +2.4 | +14.4 | +2.4 |
| Mesh1HopEW 4096 B | port | E1 | +1.6 | +1.4 | +1.8 | +1.7 | +8.0 | +1.7 |
| Mesh1HopEW 4352 B | main | E0 | +11.1 | +11.1 | +30.9 | +12.0 | +31.6 | +12.0 |
| Mesh1HopEW 4352 B | port | E0 | +1.4 | +1.3 | +1.8 | +1.6 | +13.0 | +1.6 |
| Mesh1HopEW 4352 B | port | E1 | +1.5 | +1.3 | +2.0 | +1.6 | +6.7 | +1.6 |
| NeighborExchange 4096 B, 1 link | main | E0 | +12.7 | +10.6 | +14.8 | +12.9 | +30.1 | +12.1 |
| NeighborExchange 4096 B, 1 link | port | E0 | -0.1 | -0.1 | -0.1 | -0.1 | +0.1 | -0.1 |
| NeighborExchange 4096 B, 1 link | port | E1 | +0.0 | +0.0 | +0.0 | +0.0 | -0.0 | +0.0 |
| NeighborExchange 4096 B, 2 links | main | E0 | +12.6 | +10.6 | +25.0 | +12.8 | +30.1 | +12.1 |

- **The floor does not depend on the primitive.** At D=1 the two spins cost the same on every side: port 2D E0 +1.2 to
  +2.4, E1 +1.0 to +1.6, main E0 +10.4 to +12.7, port NeighborExchange 0. So a side's minimum cost per spin is a
  property of that core and site (the shortest spin's own work plus the code-generation shift; not split by A/B), not
  of the counter.
- **The wall-clock spin steps up early.**
  - It steps up at D=4 on the port (E0 +13, E1 +6.7) and already at D=2 on main's E0 (+29 to +31).
  - The mcycle spin stays at its D=1 cost through D=4 there.
  - So a wall-clock spin of D>=4 (>=2 on main's E0) runs about 10 or more cycles longer than D core cycles.
  - Hypothesis: the debug wall-clock register is slow to read or advances coarsely, so the loop takes a second slow
    poll. The read was not timed directly.
  - The first set's D therefore is not core cycles. Its knees stand as orderings, not as headroom in cycles.

### 10.3 The mcycle ladder (change against D=0, cycles/packet, mean of the two passes; `*` beyond the noise band)

The noise band is the largest base-run spread per router and test over both sets: 3.07 for main's NeighborExchange
rows, 0.5 elsewhere. Both passes agree within 0.1-0.5 except main's NeighborExchange mode flips. Per-pass values are in
slack_report_mcycle.txt.

| row | router | D=0 rate | side | D=1 | D=2 | D=4 | D=8 | D=16 | D=32 | D=64 | D=128 | slack (knee) | clear-drop slack |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Mesh1HopEW 2048 B | main | 223.88 | E0 | +10.4* | +11.4* | +11.3* |  |  | +46.2* |  |  | 0 (0-1) | 0 (0-1) |
| | | | E1 |  |  | +0.4 |  | +0.2 | +0.7* | +1.0* | +53.1* | 16 (16-32) | 64 (64-128) |
| Mesh1HopEW 2048 B | port | 122.40 | E0 | +2.4* | +3.9* | +3.9* | +16.0* | +17.4* | +39.2* |  |  | 0 (0-1) | 4 (4-8) |
| | | | E1 | +1.0* | +1.3* | +1.3* | +7.7* | +8.3* | +24.3* |  |  | 0 (0-1) | 4 (4-8) |
| Mesh1HopEW 4096 B | main | 222.46 | E0 | +11.3* | +12.2* | +12.2* |  |  | +43.6* |  |  | 0 (0-1) | 0 (0-1) |
| | | | E1 |  |  | -1.9 |  | -0.7 | +1.3* | +1.5* | +54.6* | 16 (16-32) | 64 (64-128) |
| Mesh1HopEW 4096 B | port | 126.85 | E0 | +1.2* | +2.4* | +2.4* | +11.2* | +12.3* | +34.3* |  |  | 0 (0-1) | 4 (4-8) |
| | | | E1 | +1.4* | +1.7* | +1.7* | +5.8* | +6.0* | +20.7* |  |  | 0 (0-1) | 4 (4-8) |
| Mesh1HopEW 4352 B | main | 222.72 | E0 | +11.1* | +12.0* | +12.0* |  |  | +43.9* |  |  | 0 (0-1) | 0 (0-1) |
| | | | E1 |  |  | -2.1 |  | -0.8 | +0.9* | +1.0* | +54.3* | 16 (16-32) | 64 (64-128) |
| Mesh1HopEW 4352 B | port | 128.44 | E0 | +1.3* | +1.6* | +1.6* | +9.5* | +11.2* | +32.4* |  |  | 0 (0-1) | 4 (4-8) |
| | | | E1 | +1.3* | +1.6* | +1.6* | +4.9* | +5.2* | +19.2* |  |  | 0 (0-1) | 4 (4-8) |
| NeighborExchange 4096 B, 1 link | main | 126.43 | E0 | +10.6* | +12.9* | +12.1* |  |  | +44.8* |  |  | 0 (0-1) | 0 (0-1) |
| | | | E1 |  |  | +0.2 |  | +0.2 | +12.4* | +43.3* | +103.3* | 16 (16-32) | 16 (16-32) |
| NeighborExchange 4096 B, 1 link | port | 121.79 | E0 | -0.1 | -0.1 | -0.1 | -0.1 | -0.0 | +1.4* |  |  | 16 (16-32) | >= 32 |
| | | | E1 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +10.3* |  |  | 16 (16-32) | 16 (16-32) |
| NeighborExchange 4096 B, 2 links | main | 126.43 | E0 | +10.6* | +12.8* | +12.1* |  |  | +44.8* |  |  | 0 (0-1) | 0 (0-1) |
| | | | E1 |  |  | +1.6 |  | +0.8 | +13.0* | +43.5* | +103.3* | 16 (16-32) | 16 (16-32) |
| NeighborExchange 4096 B, 2 links | port | 121.80 | E0 | -0.1 | -0.1 | -0.1 | -0.1 | -0.0 | +1.4* |  |  | 16 (16-32) | >= 32 |
| | | | E1 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +10.3* |  |  | 16 (16-32) | 16 (16-32) |

### 10.4 Reading: the conflict settled

- **The study's numbers reproduce on the port.** At D=32 the port pays E0 +32.4 and E1 +19.2, against v4txa11's c32
  of +38.2 and +19.4. Read through the N+12 model (slack = 44 - c32), that gives E0 ~12 and E1 ~25, as the study
  read.
- **The ladder does not support "E1 has ~25 cycles of headroom".**
  - E1 pays +1.3 at D=1 (the same as E0) and +4.9 at D=8; its clear-drop knee is between 4 and 8 cycles, the same as
    E0's.
  - E1's cost then grows about 0.6 per cycle (+5.2 at 16, +19.2 at 32). E0's grows about 1:1 (+11.2 at 16, +32.4 at
    32).
  - The N+12 reading assumes a hard max (a side pays nothing until its slack runs out, then N+12 minus slack). E1's
    response is soft, and one point at D=32 turns that into slack.
- **Settled:**
  - The port's 2D E1 does drop at small D under the study's own spin. The conflict is between the N+12 reading and the
    ladder, not a wall-clock artifact.
  - The wall-clock spin exaggerated the size of the small-D drop: +6.7 at its D=4, against +1.6 at mcycle D=4.
  - The study's v4txa11 "E1 slack ~25" on this row is a model reading, not measured headroom. It is a hypothesis that
    v4txa11 behaves the same way: its c32 values match, but the ladder was not re-run on v4txa11, so its slack.json
    records are not marked WRONG (TODO.md).
- **Which side bounds each row (unchanged from section 6; now with the knees in core cycles):**
  - **Port Mesh1HopEW 2048 / 4096 / 4352 B: E0 and E1.** Each pays 1-2.4 for even a 1-cycle spin, and both
    clear-drop knees are between 4 and 8 cycles. E0 is the steeper side past the knee.
  - **Port NeighborExchange 4096 B, 1 and 2 links: neither ERISC.** E0 takes D=16 at 0 cost (+1.4 at 32). E1 takes
    D=16 at 0 (+10.3 at 32). The row is bound elsewhere: wire or worker, not separated.
  - **Main, every row: E0,** at +10.4 to +12.9 for a 1- to 4-cycle spin. Main's E1 has headroom:
    - Mesh: small rises of about +1 from D=32, hard knee between 64 and 128.
    - NeighborExchange: knee 16-32.
- **Main's E0 pays about 11 cycles/packet for a 1-cycle spin** (the port's E0 pays about 1.3). This persists with the
  fine spin, so it is not the wall clock. The cause is not isolated. Hypotheses: code generation in main's kernel_main
  around the inlined send copies (+12 instructions at mtx1), or the spin delaying main's E0 work after a send (TODO.md).
- **Main Mesh E1 is 1.9-2.1 faster at D=4** with the mcycle spin too. It is reproducible; the mechanism is unknown.
