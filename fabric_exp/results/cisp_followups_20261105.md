# CI streaming-profiler follow-ups (2026-11-05)

These follow-ups come after the port repro of CI run 41963172504 (job 810, results/cisp_ci_20261105-062618): (a) a PC
split of the router's time with the profiler off and on, and (b) the same off/on A/B on main.

- Host: bh-lb-120-a09u32.
- Telemetry: on (`TELEM=ci`, the user's CI-fidelity exception for these runs only).
- Tools: port_tune/cisp/ (cisp_ab.sh, cisp_pcs.sh, cisp_pcs_split.py, cisp_cmp.py, cisp_cores.py).

## (b) Main off/on (broker 818, results/cisp_ci_main_20261105-074351)

The main tree is /home/aagarwal/tt-metal-main at d2bd12fcf1f. Its only local change is test_tt_fabric.cpp (+7/-2), the
same test change as on the port. Arms off / on, two passes (a: off, on; b: on, off). All four runs passed their 624
tests; the profiler capture started 4 times in each on run.

The ratio is on/off bandwidth for the same pass. Bounds: median 0.996, worst 0.989.

| tree | pass | median | worst | best |
| --- | --- | ---: | ---: | ---: |
| main d2bd12fcf1f | a | 1.0051 | 1.0011 | 1.0458 |
| main d2bd12fcf1f | b | 1.0049 | 1.0011 | 1.0454 |
| port 57c55f98118 (job 810) | a | 0.9986 | 0.9873 | 1.0041 |
| port 57c55f98118 (job 810) | b | 0.9982 | 0.9861 | 1.0069 |

- **On main the profiler-on build is faster on every one of the 48 rows, in both passes.** The pass-to-pass spread of
  the off runs is under 0.03% on the NeighborExchange rows.
- **The largest gains are on NeighborExchange write and fused-atomic rows** (+3.7 to +4.6%). The FullRing rows gain
  +0.1 to +0.2%, the scatter rows +0.4 to +0.5%, and the SingleSender rows +0.4 to +1.4%.
- **The port is faster than main with the profiler off:** 1.09x to 1.61x on all 48 rows.

Per test (on/off for passes a and b; "port off / main off" uses pass a):

| test | main off GB/s | main a | main b | port off GB/s | port a | port b | port off / main off |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FullRing/2048/fused_atomic_inc/L1 | 13.140 | 1.0018 | 1.0017 | 16.904 | 0.9988 | 0.9988 | 1.2865 |
| FullRing/2048/fused_atomic_inc/L2 | 13.144 | 1.0017 | 1.0017 | 16.905 | 0.9988 | 0.9987 | 1.2862 |
| FullRing/2048/scatter_write/L1 | 12.648 | 1.0016 | 1.0013 | 17.007 | 0.9988 | 0.9988 | 1.3446 |
| FullRing/2048/scatter_write/L2 | 12.656 | 1.0011 | 1.0011 | 17.008 | 0.9987 | 0.9987 | 1.3438 |
| FullRing/2048/write/L1 | 13.469 | 1.0023 | 1.0023 | 17.541 | 0.9991 | 0.9992 | 1.3024 |
| FullRing/2048/write/L2 | 13.473 | 1.0023 | 1.0023 | 17.538 | 0.9991 | 0.9990 | 1.3017 |
| FullRing/4096/fused_atomic_inc/L1 | 25.945 | 1.0015 | 1.0016 | 33.333 | 0.9980 | 0.9981 | 1.2848 |
| FullRing/4096/fused_atomic_inc/L2 | 25.946 | 1.0014 | 1.0014 | 33.270 | 0.9977 | 0.9975 | 1.2822 |
| FullRing/4096/scatter_write/L1 | 25.113 | 1.0022 | 1.0021 | 33.521 | 0.9986 | 0.9985 | 1.3348 |
| FullRing/4096/scatter_write/L2 | 25.116 | 1.0018 | 1.0016 | 33.444 | 0.9985 | 0.9986 | 1.3316 |
| FullRing/4096/write/L1 | 26.618 | 1.0021 | 1.0021 | 34.537 | 0.9989 | 0.9990 | 1.2975 |
| FullRing/4096/write/L2 | 26.622 | 1.0020 | 1.0020 | 34.439 | 0.9989 | 0.9987 | 1.2936 |
| NeighborExchange/2048/fused_atomic_inc/L1 | 20.116 | 1.0413 | 1.0400 | 22.019 | 0.9993 | 0.9993 | 1.0946 |
| NeighborExchange/2048/fused_atomic_inc/L2 | 20.169 | 1.0374 | 1.0370 | 22.030 | 0.9992 | 0.9993 | 1.0923 |
| NeighborExchange/2048/scatter_write/L1 | 18.248 | 1.0050 | 1.0048 | 22.999 | 0.9992 | 0.9993 | 1.2604 |
| NeighborExchange/2048/scatter_write/L2 | 18.262 | 1.0049 | 1.0051 | 23.010 | 0.9993 | 0.9992 | 1.2600 |
| NeighborExchange/2048/write/L1 | 20.235 | 1.0416 | 1.0413 | 24.489 | 0.9994 | 0.9995 | 1.2102 |
| NeighborExchange/2048/write/L2 | 20.273 | 1.0385 | 1.0380 | 24.500 | 0.9995 | 0.9995 | 1.2085 |
| NeighborExchange/4096/fused_atomic_inc/L1 | 39.533 | 1.0442 | 1.0442 | 43.430 | 0.9992 | 0.9992 | 1.0986 |
| NeighborExchange/4096/fused_atomic_inc/L2 | 39.652 | 1.0399 | 1.0399 | 43.448 | 0.9993 | 0.9992 | 1.0957 |
| NeighborExchange/4096/scatter_write/L1 | 36.039 | 1.0040 | 1.0040 | 45.360 | 0.9991 | 0.9992 | 1.2586 |
| NeighborExchange/4096/scatter_write/L2 | 36.060 | 1.0043 | 1.0041 | 45.382 | 0.9992 | 0.9991 | 1.2585 |
| NeighborExchange/4096/write/L1 | 39.778 | 1.0458 | 1.0454 | 48.465 | 0.9957 | 0.9949 | 1.2184 |
| NeighborExchange/4096/write/L2 | 39.900 | 1.0411 | 1.0410 | 48.490 | 0.9873 | 0.9861 | 1.2153 |
| SSLinear/2048/fused_atomic_inc/L1 | 15.556 | 1.0127 | 1.0128 | 19.659 | 0.9970 | 0.9932 | 1.2638 |
| SSLinear/2048/fused_atomic_inc/L2 | 15.631 | 1.0095 | 1.0096 | 19.699 | 0.9952 | 0.9949 | 1.2602 |
| SSLinear/2048/scatter_write/L1 | 15.099 | 1.0052 | 1.0055 | 19.994 | 0.9965 | 0.9965 | 1.3242 |
| SSLinear/2048/scatter_write/L2 | 15.142 | 1.0047 | 1.0048 | 20.044 | 0.9960 | 0.9963 | 1.3237 |
| SSLinear/2048/write/L1 | 15.850 | 1.0135 | 1.0140 | 20.871 | 0.9967 | 0.9968 | 1.3168 |
| SSLinear/2048/write/L2 | 15.961 | 1.0079 | 1.0084 | 20.932 | 0.9947 | 0.9954 | 1.3115 |
| SSLinear/4096/fused_atomic_inc/L1 | 29.144 | 1.0081 | 1.0083 | 38.508 | 0.9969 | 0.9962 | 1.3213 |
| SSLinear/4096/fused_atomic_inc/L2 | 29.250 | 1.0061 | 1.0062 | 38.586 | 0.9958 | 0.9957 | 1.3192 |
| SSLinear/4096/scatter_write/L1 | 29.089 | 1.0082 | 1.0082 | 39.110 | 0.9955 | 0.9956 | 1.3445 |
| SSLinear/4096/scatter_write/L2 | 29.193 | 1.0064 | 1.0065 | 39.164 | 0.9953 | 0.9948 | 1.3415 |
| SSLinear/4096/write/L1 | 29.220 | 1.0091 | 1.0082 | 40.240 | 0.9958 | 0.9952 | 1.3772 |
| SSLinear/4096/write/L2 | 29.332 | 1.0063 | 1.0063 | 40.322 | 0.9956 | 0.9953 | 1.3747 |
| SSMesh/2048/fused_atomic_inc/L1 | 12.171 | 1.0057 | 1.0056 | 19.504 | 1.0030 | 0.9946 | 1.6026 |
| SSMesh/2048/fused_atomic_inc/L2 | 12.194 | 1.0044 | 1.0043 | 19.539 | 1.0041 | 1.0069 | 1.6023 |
| SSMesh/2048/scatter_write/L1 | 12.218 | 1.0057 | 1.0059 | 19.573 | 1.0032 | 1.0031 | 1.6019 |
| SSMesh/2048/scatter_write/L2 | 12.246 | 1.0044 | 1.0042 | 19.681 | 0.9970 | 0.9931 | 1.6071 |
| SSMesh/2048/write/L1 | 12.176 | 1.0054 | 1.0054 | 19.655 | 0.9989 | 0.9991 | 1.6143 |
| SSMesh/2048/write/L2 | 12.196 | 1.0041 | 1.0043 | 19.677 | 0.9972 | 0.9953 | 1.6134 |
| SSMesh/4096/fused_atomic_inc/L1 | 24.104 | 1.0054 | 1.0046 | 38.427 | 0.9979 | 0.9979 | 1.5942 |
| SSMesh/4096/fused_atomic_inc/L2 | 24.186 | 1.0036 | 1.0038 | 38.262 | 0.9988 | 0.9920 | 1.5820 |
| SSMesh/4096/scatter_write/L1 | 24.262 | 1.0042 | 1.0045 | 38.419 | 0.9982 | 0.9937 | 1.5835 |
| SSMesh/4096/scatter_write/L2 | 24.305 | 1.0035 | 1.0033 | 38.452 | 0.9958 | 0.9958 | 1.5821 |
| SSMesh/4096/write/L1 | 24.169 | 1.0052 | 1.0050 | 38.422 | 0.9949 | 0.9949 | 1.5897 |
| SSMesh/4096/write/L2 | 24.236 | 1.0037 | 1.0037 | 38.473 | 0.9964 | 0.9984 | 1.5874 |

### Per router (telemetry cycles per router, cisp_cores.py)

The test's bandwidth is the mean of the per-router telemetry rows. On the NeighborExchange 4096 B write rows, the two
trees move in opposite directions.

- **Main, profiler off:** the routers sit at several cycle levels.
  - L1 (20 routers): 13.85M (10 routers), 13.94M (2) and 14.24-14.35M (8).
  - L2 (40 routers): 13.8M to 14.4M.
- **Main, profiler on:** every router drops to one level. L1 is 13.44M to 13.45M on all 20; L2 is 13.4M to 13.6M.
  - Per-router off/on is 1.030 to 1.067 on L1, so the slowest off routers gain the most.
  - The scatter and FullRing rows show the same collapse onto a single level, at smaller gaps.
- **Port, profiler off:** every router is at one level, 11.53M to 11.54M cycles.
- **Port, profiler on:** the routers split into levels.
  - L2: 11.54M (about half), 11.72M (+1.6%, 1.85 cycles/packet) and 11.90M (+3.1%, 3.6 cycles/packet).
  - L1: 15 of 20 routers unchanged and 5 at 11.7M.

The router images gain the same function families with the profiler on, on both trees.
- **Main active ERISC:** +3532 to +3852 B. It adds RouterPort / PortBase / TransmitterPort / ReceiverPort (link_sync),
  eth_ptp install/uninstall and kernel_profiler mark_zone_long / ring_ensure_room_slow. LINK_SYNC_ROLE is 1 on 65
  images and 2 on 71.
- **Main subordinate ERISC:** +1040 to +1044 B, the kernel_profiler functions only.
- **Port:** +3588 to +3720 B on the active ERISC and +1032 to +1040 B on the subordinate.

Not established (hypotheses only; no isolating A/B has been run):
- Why main gets faster with the profiler on. Candidates:
  - the link_sync rounds pace the senders and remove the per-link spread;
  - the extra code changes placement or timing in main's loop.
- Which piece costs the port its 1.6% / 3.1% levels. Candidates: the hook's per-round work on some routers (the
  LINK_SYNC_ROLE split is a candidate), or placement. **WRONG (2026-11-05, section (c)):** the levels are per-iteration
  events of about 540-570k cycles in 1 or 2 of a test's 3 iterations, averaged. They also occur with the router hook
  compiled out (job 811's no-hook arm), and they hit transmitters and receivers alike.
- Whether the level a router lands on follows its link_sync role. That is a correlation check, not an attribution.
  **Answered in (c): it doesn't.**

## (a) PC split on the port (results/cisp_pcs_20261105-074220)

The run was NeighborExchange 4096 B write, 80M packets per router, L1 and L2, profiler off and on, sampled over the
debug bus in steady state. The table gives each code group's share of traffic-carrying kernel samples, pooled per
ERISC.

| ERISC0 | passes | end_round | link_sync | eth_ptp | kernel_profiler | telemetry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| on L2 | 86.545% | 0.758% | 0.488% | 0 | 0.026% | 12.182% |
| off L2 | 87.177% | 0.586% | 0 | 0 | 0 | 12.237% |
| on L1 | 86.575% | 0.701% | 0.510% | 0 | 0.030% | 12.184% |
| off L1 | 87.242% | 0.604% | 0 | 0 | 0 | 12.154% |

- **ERISC1:** passes about 97.8%, end_round 0.7-0.8% and telemetry 1.4%, the same with the profiler off and on.
- **Steady-state rate, profiler on vs off:**
  - **L1:** all 20 routers within 0.1% (11.70-11.71 Mpps).
  - **L2:** 39 of 40 routers at 11.70 Mpps on vs 11.71 off. One router, MeshCoordinate [0, 1] core 0-7, ran at
    7.99 Mpps: 13.52G cycles vs 9.23G for the others, 4.3G more. It is the only outlier in the four runs. It doesn't
    show in the sampled window (no router there has an outlying share), so the extra time fell outside it.
- **Correction to the earlier chat summary:** "L2 0.9916 (cycles +1.2%)" was a mean over 40 routers. It is that one
  router. The other 39 are at 0.999. Cause unknown, one occurrence.
- On the port, the hook's link_sync and end_round code takes 0.5% + 0.17% of ERISC0 samples. Those are shares of
  polling time, not the bound (PC samples cannot name the bound).

## (c) Per-router map of the port's on/off telemetry (jobs 810, 811, 818; no device runs)

Tool: port_tune/cisp/cisp_router_map.py.

**Topology.**
- Each router's physical chip comes from the run's physical_chip_mesh_coordinate_mapping. It is the same on main and the
  port: [0,0]=1, [0,1]=0, [0,2]=2, [0,3]=3, [1,0]=5, [1,1]=4, [1,2]=6, [1,3]=7.
- Its peer comes from fabric_exp/a09u32_cluster_desc.yaml. All 15,120 link rows of the runs' raw telemetry CSVs match
  it: 40 routers, 0 mismatches.
- Link-sync role (ERISC0): transmitter when the router's chip id is below its peer's (link_sync.cpp role_of). Every
  link of this mesh is synced, so there are 20 transmitters and 20 receivers.

**Method.**
- The unit is one router in one iteration of one test. Each test runs 3 iterations, and the bandwidth figure is their
  mean.
- The baseline is the median of the profiler-off run's 3 iterations, same pass.

**Port, profiler on (810 a/b, 811 a):**
- **Steady cost.** Most router-iterations are +1k to +10k cycles (about 0.05%).
- **Discrete events.** On top, there are events of +300k to +573k cycles on one router in one iteration. The median is
  361k to 543k, with a cluster at 540-570k, about 0.4 ms at 1.35 GHz (from cycles, pps and packets).
  - Count: 68, 110 and 87 router-iterations of 2160.
- **Off runs** deviate by more than 300k in 3-4 of 2160 router-iterations.
- **Where the levels come from.** cisp_cores.py averaged the 3 iterations, so one event in 3 iterations shows as
  +180k (the 11.72M level) and two as +360k (the 11.90M level).
- **Where the events fall.** They come at the same test-iterations in every on run (pass a, pass b, 811):
  - NX 4096 B write L2 iteration 1: 21 / 23 / 26 of 40 routers;
  - NX 4096 B write L2 iteration 2: 12 / 13 / 10;
  - NX 4096 B write L1 iteration 3: 5 / 6 / 3;
  - FullRing 4096 B fused_atomic_inc L2 iterations 1-3: 5-22 each;
  - a few FullRing write / scatter rows with 1-3 routers;
  - none in the SingleSender rows except one.
- **Which routers are hit changes from run to run.** In NX 4096 B write L1 iteration 3, chip0/eth7 (T) is hit in 3 of
  4 runs and chip4/eth11 (R) in 3 of 4. No other router repeats much, and the larger events hit 10-26 routers at once.
- **No role preference.** Big events hit transmitters and receivers alike: 38/30, 58/52 and 43/44 (T/R) of 1080 each.
- **The router hook is not needed.** Job 811's no-hook arm (router link-sync hook compiled out, profiler on, NX rows
  only) has the same events at the same places:
  - NX 4096 B write L1 iteration 3: 4 routers;
  - L2 iteration 1: 18;
  - L2 iteration 2: 4;
  - L2 iteration 3: 2;
  - sizes 533-564k.
- **Hypothesis, not tested.** The events come from the profiler's activity outside the router hook. One candidate is the
  router's kernel_profiler ring waiting for a host drain (ring_ensure_room_slow is in both the on and no-hook images).
  Another is a host-side capture step at points that are deterministic in the test sequence. The cheapest isolating A/B
  is profiler on with the router's own zones off, or with a larger ring.
- **Smaller repeat hits in the SingleSender rows.** These rows load only chip 1's routers ([0,0], the sender). There,
  chip1/eth8 and eth10 (T, toward chip 5) are +0.6% (Linear) and +2.4% (Mesh), and chip1/eth5 and eth7 (R) +2.5%. They
  appear in every on run. The no-hook arm didn't run these rows.

**Main, profiler on (818 a/b):**
- No event over 300k in 2160 router-iterations per pass.
- Off runs deviate in 0 and 8 router-iterations.
- chip1/eth10 (T) has a +0.55% repeat in 2-3 SingleSender Linear iterations.
- Main's speed-up with the profiler on is the collapse of the off-run spread described above, not events.

**Long-run outlier.** In the PC-sampling run (on, NX 4096 B write L2, 80M packets), MeshCoordinate [0,1] core 0-7 is
chip0/eth7, a transmitter toward chip2/eth4.
- It had 4.29G more cycles than the others, about 3.2 s.
- It is one of the routers hit most often in the short runs (5, 4 and 4 events, and 1 in no-hook).
- One occurrence; cause unknown.

## (d) Where the routers are during the stall events (broker 819, results/cisp_pcs2_20261105-083111)

**Run.**
- Tree: port 57c55f98118, Tracy, TELEM=ci.
- Rows: ubench's NX group as CI runs it (100k packets × 3 iterations, num_links 1 and 2), 8 times in one process
  (cisp/pcs_nx_ci_x8.yaml). Arms: profiler on, then off.
- Sampling: pc_sampler.py --only-loaded on the 40 router cores' ERISC0 and ERISC1 (80 targets, about 960 Hz each),
  across the whole run, 0 read errors.
- Samples: on 826,560 (10.8 s); off 859,520 (11.0 s).
- Iteration windows come from the log's "Launching programs" / "Finished." lines. They hold 37% of the samples, and the
  first ~30% of each window has no traffic yet.
- Tools: cisp_pcs2_report.py, cisp_pcs2_busy.py, cisp_pcs2_span.py.

**The events reproduce.** 288 router-iterations are more than 300k cycles above the off median. They fall at NX 4096 B
write L2 it1 (103) and it2 (47), NX 2048 B write L2 it1 (59), and the write L1 it3 rows (48).

**ERISC0 share by bucket, on vs off:**

| bucket | all samples | inside iteration windows |
| --- | ---: | ---: |
| end_round | +0.63 | +0.41 |
| ctx_switch (follow_context_switch / parked) | +0.63 | +0.34 |
| link_sync | +0.82 | +0.70 |
| ring_ensure_room_slow | +0.007 | +0.015 |
| other kernel_profiler | +0.008 | +0.007 |
| eth_ptp | +0.002 | 0 |
| base firmware | −1.05 | 0 |
| telemetry | −0.14 | −0.20 |
| passes | −0.89 | −1.27 |

- Base firmware samples (outside the kernel image) appear only between iterations.
- ERISC1 has the same mix with the profiler on and off.

**Event router-iterations against the others, same rows (NX 4096 B write L2):**
- **The bucket mix matches.**
  - ERISC0: passes 87.0 / 85.8%, telemetry 10.7 / 11.3%, end_round 1.6 / 1.6%; no firmware; ring_ensure_room_slow 1
    sample of 4167.
  - ERISC1: passes 94.2 / 95.3%.
- **The busy share matches,** with the same profile across the window:
  - ERISC0 sender::send 28.4 / 28.1%;
  - ERISC1 handle / forward 35.0 / 34.1%.
- **The idle pass lines match:** holds_packet 24.5 / 25.6% of not-busy samples, then visit_worker, check_connection and
  so on.
- **Real-time send timeline** (128 event router-iterations against the same iteration's quiet routers, 1 kHz samples):
  - first sender::send sample +0.14 ms (SE 0.11);
  - last sender::send sample −0.09 ms (SE 0.11);
  - the telemetry's excess for the same router-iterations is +0.40 ms.
  - So the event routers start and finish sending at the same times as their quiet peers. The 0.4 ms doesn't show as
    later or longer sending.

**Reading (hypotheses, not tested):**
- The PC samples don't place the ~0.42 ms in any code region: no ring wait, no firmware, no context switch beyond the
  on-arm's steady +0.6 points.
- The routers don't send later or longer. So the events look like counted time in the perf recorder rather than lost
  sending time. **WRONG (2026-11-05, section (f)):** the comparison was against quiet routers in the same iteration. The
  test's own sender-side bandwidth drops by the same ~4.6% in the same iterations, so an event iteration is slower in
  real time. The recorder sums the durations of rounds that sent a packet or ended with one waiting, using the RISC
  wall clock.
- Two candidates:
  1. A step in the wall clock the recorder reads (RISCV_DEBUG_REG_WALL_CLOCK) during a counted round.
  2. Rounds at an iteration's start that hold a packet waiting for the remote side are counted before sending can
     begin.
- Distinguishing them needs a device run with a recorder variant: per iteration, the longest counted round and where in
  the iteration it falls, or the wall-clock delta beside the ERISC cycle counter. That is a design to discuss first,
  not built.
- The CI metric is computed from these cycles, so it sees the events either way.

## (e) Recorder instrumentation (broker 820, results/cisp_recdbg_20261105-085747)

**Build and run.** The build is the same NX ×8 run as (d), from the runtime root cisp_recdbg: 57c55f98118 plus
cisp/recdbg/recdbg.diff, local only. Each sender ERISC writes one 128 B entry per iteration into a ring in L1 at
0x60000; hang_capture dumped L1 after each arm. Tool: cisp/recdbg/recdbg_report.py.

**Alignment.** All 8640 telemetry rows per arm match a ring entry, with entry busy equal to the log's cycles in every
row.

**Side effect of the instrumentation.** The recorder's bandwidth rises about 12% in every row, on and off. The
instrumentation's per-round L1 writes fall outside the counted window (after the round's end read), which shows how much
the recorder's figure depends on what sits inside a round. The events still reproduce: 254 router-iterations, median
excess 461k cycles.

**Event router-iterations against quiet ones (on arm):**
- **The longest counted round is about 2.9k cycles in both** (median 2924 / 2906, max 3073 / 3158). There is no long
  round.
- **The clocks agree.** The wall-clock delta equals the get_timestamp() delta (difference 0-4 cycles). Backwards reads:
  0. Steps above 2^31: 0. The largest step between wall reads is about 2.9k cycles, which is one round.
- **No firmware involvement.** fw_ran is 0 in the longest round of every event.
- **Held-only rounds are the same.** Counted rounds with tx_packets == 0 are about 1.2 per iteration (about 900 cycles)
  in both groups.
- **The difference is the number of rounds and how many packets each carries.** Per row (on arm; off for reference):

  | row | group | cycles | counted rounds | cycles / round | packets / round |
  | --- | --- | ---: | ---: | ---: | ---: |
  | NX 4096 B write L2 | events (126) | 10.79M | 4703 | 2293 | 21.3 |
  | NX 4096 B write L2 | quiet (834) | 10.33M | 4367 | 2366 | 22.9 |
  | NX 4096 B write L2 | off (960) | 10.32M | 4403 | 2345 | 22.7 |

  The other event rows (NX 2048 B write L1 and L2, 4096 B write L1) show the same pattern.
- **Event iterations run about 8% more counted rounds, each with fewer packets** (21.2 against 22.9). The extra ~340
  rounds at roughly 1.3k cycles of fixed per-round cost each add up to the ~0.46M-cycle excess.

**Reading (hypothesis, not tested):**
- The "events" are iterations in which packets reach the sender in smaller batches, so it runs more rounds per packet
  and the recorder counts more per-round overhead.
- They are not stalls. There is no long round and no clock step, and in (d) the real-time send span was unchanged.
  **WRONG in part (section (f)):** there is no long round, but the iteration is really slower. The sender-side bandwidth
  drops the same ~4.6% in the same iterations.
- The profiler would cause this by changing the supply pattern, for example the worker kernels' profiler activity. That
  is untested.
- The CI metric (recorder bandwidth) sees it. The router's throughput may not change. **WRONG (section (f)):** the
  sender-side throughput drops in those iterations too.
- **Torn reads explain the long-run outlier (+2^32 exactly), not these events.**

## (f) The test's own sender-side bandwidth (broker 821 port, 822 main; results/cisp_ci_20261105-091517, cisp_ci_main_20261105-092048)

**Runs.** The same runs as 810 / 818: CI ubench, *Unicast*, TELEM=ci, off/on, two reversed passes. cisp_ab.sh now keeps
generated/fabric/bandwidth_results_blackhole.csv per arm and pass. That file has one sender row per iteration (device
7, W). Tool: cisp/cisp_sender.py.

**Medians and worst over the 48 tests** (on/off, pass a / b; CI bounds: median ≥ 0.996, worst ≥ 0.989):

| tree | metric | median | worst | worst test |
| --- | --- | ---: | ---: | --- |
| port 57c55f98118 | sender-side | 0.9987 / 0.9986 | **0.9537 / 0.9694** | NX 4096 B write L2 |
| port 57c55f98118 | router telemetry | 0.9986 / 0.9986 | 0.9865 / 0.9884 | NX 4096 B write L2 |
| main d2bd12fcf1f | sender-side | 1.0013 / 1.0011 | 0.9977 / 0.9975 | SSMesh 4096 B write L2 / NX 4096 B scatter L2 |
| main d2bd12fcf1f | router telemetry | 1.0048 / 1.0050 | 1.0012 / 1.0014 | FullRing 2048 B scatter L2 |

**Port, rows of interest:**

| row | sender off GB/s | sender on/off (a / b) | telemetry on/off (a / b) |
| --- | ---: | ---: | ---: |
| NX 4096 B write L2 | 45.61 | 0.9537 / 0.9694 | 0.9865 / 0.9884 |
| NX 4096 B write L1 | 45.61 | 0.9842 / 0.9843 | 0.9972 / 0.9972 |
| FullRing 4096 B fused L1 | 32.09 | 0.9982 / 0.9977 | 0.9981 / 0.9979 |
| FullRing 4096 B fused L2 | 32.02 | 0.9974 / 0.9978 | 0.9974 / 0.9975 |
| NX 4096 B fused, scatter | — | 0.9998-1.0001 | 0.9991-0.9992 |
| SSLinear | — | 0.9902-0.9990 | 0.9931-0.9969 |
| SSMesh | — | 0.990-1.011 (noisy) | 0.992-1.016 |

**The sender's per-iteration rows show discrete events.**
- An off iteration is 45.61 GB/s. An on iteration is either 45.61 or ~43.5 (−4.6%).
- The hits fall in the same places as the router events:
  - NX 4096 B write L2: all 3 iterations in pass a, it1 and it2 in pass b;
  - NX 4096 B write L1: it3 in both passes.
- So in those iterations the sender (worker) really is slower, by ~0.4 ms.

**Main.**
- The sender-side NX write rows are +3.5% to +5.2% faster with the profiler on, like the telemetry.
- Off, main's sender is at 38.0 GB/s on NX 4096 B write, against the port's 45.6.

**Reading (hypotheses, not tested):**
- **WRONG in part (830): the same ~4.7% slow state occurs with the profiler off when the rows are reordered (C). The
  profiler changes which iterations land in it in the stock order; it is not itself the slowdown's cost.**
- The profiler slows the sender worker in particular iterations. With the port's faster router, the worker becomes the
  limiting stage on the highest-bandwidth row, so its hiccups show: as fewer packets per router round (section (e)) and
  as a lower sender-side figure.
- Main's slower router hides them.
- Separating the worker-side profiler cost from the router side needs a device A/B: profiler on with worker zones off
  (or the router hook off) on the sender-side metric.

## (g) The stall hunt: step 1 (zone timeline, broker 823) and step 2 (Tensix PC samples, broker 824)

### Step 1 (results/cisp_zones_20261105-093338)

**What the zone CSV covers.**
- BRISC-KERNEL start/end on the test's worker cores (96 cores, 288 launches per chip), plus a few IDLE-ERISC-KERNEL
  zones.
- No dispatch, relay or NCRISC zones. The OPS CSV is empty.
- Timestamps are per-chip raw cycles with constant offsets, so they can't be compared across chips; within a chip they
  are exact.

**Findings.**
- The senders are BRISC kernels on (15, y) cores; benchmark-mode receivers finish in ~3.6k cycles.
- In a slow iteration every sender kernel on an affected chip runs +576k to +646k cycles (median ~590k, 0.43-0.46 ms).
  Starts stay aligned within a chip, so this is a longer kernel, not a late start.
- The excess appears on most or all 8 chips in the same iteration: 41 of 288 iterations, all of them unicast_write rows.
  (Router-tally review: each sender's cycles include the test's end sync with its chip's sync core and the nearest
  neighbors, so one slow flow stretches every sender on its chip and its neighbors. This is not evidence of a
  chip-wide state; the router telemetry shows per-flow slowdowns.)

### Step 2 (results/cisp_tpcs_20261105-093846)

**Run.** pc_sampler.py --tensix sampled BRISC and NCRISC of the 48 sender/sync cores (about 780 Hz per target, 0 read
errors). It also snapshotted each core's mailbox head every 50 ms; the kernel's load address (config base + text offset)
moves per launch. The sender kernel has no NCRISC build, so NCRISC samples are the baseline (firmware).

**Slow iterations against normal ones on the NX write rows** (device 7's sender row below 97% of the off median: 29 slow
iterations; BRISC samples 16.4k slow, 31.2k normal):

| bucket (inline chain) | slow | normal | off arm |
| --- | ---: | ---: | ---: |
| BRISC firmware (between kernels) | 46.9% | 52.2% | 51.3% |
| sender: fabric send path | 21.5% | 19.8% | 22.3% |
| sender: NoC command buffer (NOC_CMD_BUF_READ/WRITE_REG, inline-write state) | **13.0%** | 8.3% | 6.9% |
| sender: fabric connection credits (get_num_free_write_slots, wait_for_empty_write_slot) | **7.0%** | 9.0% | 9.1% |
| sender: NoC write ack / flush wait | **1.3%** | 0.2% | 0.1% |
| sync kernel | 10.2% | 10.7% | 10.3% |
| profiler code | 0.0% | 0.0% | — |

**Within the sender kernel's own samples, a slow iteration shifts from waiting on fabric credits to waiting on the NoC:**
- NoC command-buffer reads and writes go from 17% to 25% of kernel samples.
- Write-ack / flush waits go from 0.4% to 2.4%.
- Credit waits go from 19% to 13%.
- close_finish (connection close at the end) roughly doubles.
- No samples fall in profiler code.

**Reading (hypothesis, not tested):**
- In slow iterations the senders are held back by their NoC write path rather than by the routers' credits, which points
  at NoC contention.
- That fits your prediction of a periodic, data-volume-triggered transfer that reaches every chip in the same iteration,
  for example the streaming profiler's relays or DRAM writes shipping accumulated zones.
- To confirm: sample the relay cores (eth_relay on the IDLE-ERISC cores; drisc_relay on the DRAM cores, if their PCs are
  readable) on the same timeline, then an A/B on the relay or DRAM settings (design first).

### Step 3: relays on the senders' timeline (broker 825, results/cisp_relays_20261105-094258)

**Run.** pc_sampler.py --cores relay_cores.txt sampled 136 targets: the 48 senders' BRISC, the idle-eth relay core (noc0
1,1) ERISC0 and ERISC1 on all 8 chips, and every DRAM tile's DRISC on chips 0, 3 and 7. 0 read errors.

**Eth relay (1,1).** With the profiler on it runs code at 0x108xx (ERISC0) and 0x11bxx (ERISC1); with it off, base
firmware (0x3660 / 0x94f0). Its activity, measured as the share of samples outside its six most common PCs, is flat:
87.5% in slow iterations against 87.7% in normal ones. No burst is visible.

**DRAM tiles.** 8 of 24 per chip show the same DRISC PC loop (0xdc0, 94%) with the profiler on and off; the other 16
read PC 0. So the drisc_relay's activity is not visible through `drisc_pc` on these tiles. Either the relay runs on a
RISC that signal doesn't cover, or on a tile not sampled. Open.

**The slow clusters are periodic in time** (step-1 data). **WRONG as a reading (829 review): the ~0.7 s / ~18-iteration
spacing is the yaml's row layout (unicast_write blocks recur every 18 iterations and are the only rows that ever go
slow), not a clock; see "The ~18-iteration period is the yaml's layout" below.**
- Hits come in clusters of 2-5 consecutive iterations (0.10-0.15 s). Clusters start at 0.70, 1.34, 2.03, 3.36, 4.09,
  4.80, 5.43, 6.79, 7.50, 8.22, 8.93 and 10.26 s into the run.
- That is a period of ~0.68-0.73 s; the ~1.33 s gaps are two periods whose cluster fell on rows that don't show it.
- Each affected sender kernel runs ~4.8% longer. This looks like sustained activity for ~0.1-0.15 s every ~0.7 s, not a
  single 0.43 ms pause.

**Data volume.**
- The profiler moved 141,455 frames (16.2 MB) from 8 devices in 10.9 s: ~2 MB per device, ~7 KB per device per
  iteration.
- About 17.5 iterations per period gives ~120-130 KB per device per period.

**Threshold candidate (drisc_relay.cpp).**
- A worker core's ring is 512 words; a core ships once its fullest lane holds 25% (kShipMinPct, kLaneShipWords = 128
  words).
- Every sender core runs the same kernels at the same rate on every chip, so all of them would cross the 25% mark in the
  same iteration. That fits the global, periodic pattern.
- The relays gather on the NoC opposite their egress ("the other NoC carries gathers").
- Hypothesis, untested: the synchronized gather bursts contend with the senders' NoC writes.

### Ship-threshold A/B (broker 826, results/cisp_shipab_20261105-094658)

**Arms.** NX ×8 with the profiler on; each relay variant is a runtime-root farm with one line of drisc_relay.cpp edited.
The caches confirm each arm compiled its own source (its dephash names its root).

| arm | cluster period (s) | iterations per period | clusters | hit iterations | sender-CSV slow | excess (median, k cycles) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base (stock) | 0.59-0.73 (~0.70) | 17-19 | 13 | 43 | 36 | 603 |
| shipA (kShipMinPct 25 -> 50) | 0.56-0.76 (~0.68) | 17-20 | 12 | 40 | 29 | 593 |
| pumpB (kIdlePumpStride 8 -> 16) | 0.62-0.70 (~0.69) | 17-19 | 12 | 42 | 33 | 599 |

- Gaps of ~1.3-1.4 s are two periods whose cluster fell on rows that don't show it.
- Cluster lengths are 1-5 iterations in all arms.
- **Neither change moves the period, so the 25% ship threshold and the idle-pump stride are ruled out as the trigger.**
**WRONG (829 review): the period is the row layout and could not move under any trigger. The slow counts (36, 29, 33)
barely move: weak evidence against these constants, not a ruling-out.**
  The prediction for shipA was a ~1.4 s period.

**First-principles estimate.** A sender's BRISC lane gets about 7 words per iteration (one long kernel zone: 5 words plus
a 2-word marker). 128 words / 7 ≈ 18 iterations ≈ 18 × ~38 ms ≈ 0.68 s. So the threshold would have fit the observed
period, which is why the A/B was the right test.

**Second clock with the same period (code reading, hypothesis).** kMaxDeferSweeps = 2048: "A deferred core sends anyway
after this many sweeps". If one relay sweep takes ~0.34 ms, 2048 sweeps ≈ 0.7 s. In shipA the 50% threshold would have
needed ~36 iterations, but the deferral cap would still send every ~0.7 s, so the period could not move.

**Next isolating A/B (design to the coordinator, not run):**

| arm | change | prediction if the relay's ship cadence is the trigger |
| --- | --- | --- |
| C | kShipMinPct 50 + kMaxDeferSweeps 4096 | period ~1.4 s |
| D | kShipMinPct 50 + kMaxDeferSweeps 1024 | period ~0.35 s, twice the clusters |

If neither moves, the relay's ship cadence is ruled out, and the ~0.7 s clock is elsewhere: host drain, link-sync
remeasure, or the clock sampler.

### Defer A/B (broker 827, results/cisp_deferab_20261105-095131)

Both relay arms were built from their roots (cache dephash checked).

| arm | cluster period | iterations per period | clusters | hit iterations | sender-CSV slow | excess (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base (stock) | ~0.68 s (0.62-0.75) | 17-19 | 12 | 39 | 28 | 588k |
| C: kShipMinPct 50 + kMaxDeferSweeps 4096 | ~0.69 s (0.58-0.74) | 16-19 | 12 | 40 | 31 | 592k |
| D: kShipMinPct 50 + kMaxDeferSweeps 1024 | ~0.68 s (0.55-0.78) | 17-19 | 12 | 42 | 32 | 593k |

**Neither arm moves the period, against predictions of ~1.4 s (C) and ~0.35 s (D).** **WRONG as evidence (829 review):
the period is the row layout and could not move; the slow counts (28, 31, 32) don't halve or double either, which is
weak evidence only. The two bullets below are not established.** Taken with 826:
- the drisc_relay's ship cadence (threshold, deferral cap, idle-pump stride) is ruled out as the clock of the slow
  clusters;
- the ~0.7 s / ~18-iteration clock is somewhere else.

**Candidates, not tested:**
- the host-side profiler service's drain or poll cadence;
- the link-sync remeasure (UpdatePeriod);
- the eth clock sampler;
- a producer-side event at a fixed word count, which the relay constants don't govern.

Stopped here for the coordinator's review before looking at host cadences.

### Time vs count (broker 828, results/cisp_lenab_20261105-095525; no build)

| num_packets | run | cluster period (s) | iterations per period | clusters | hit iterations | sender-CSV slow | excess per hit chip-iteration (median) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ×1 (826/827 base) | 11.0 s | ~0.68-0.70 | 17-19 | 12-13 | 39-43 | 28-36 | ~590k |
| ×2 | 13.9 s | ~0.84-0.97 | 17-20 | 13 | 42 | 32 | 1169k |
| ×4 | 18.7 s | ~1.10-1.23 | 17-19 | 12 | 41 | 32 | 2351k |

**The clock counts iterations, not time.** **WRONG as worded (829 review): the ~18 iterations are the row layout, so
there is no clock in evidence. What stands: the slow iterations sit at the same positions inside the unicast_write
blocks at ×1, ×2 and ×4, so they follow the row sequence, not elapsed time.**
- The period stays ~18 iterations at every length. Its length in seconds follows the iteration time (iterations take
  ~38 / ~48 / ~65 ms, because launch overhead doesn't scale).
- The excess scales with the iteration (~590k, ~1169k, ~2351k cycles): a hit iteration's senders run ~4.8% slower for
  the whole iteration. That's a rate change, not a fixed stall.

**Kernel placement checked (824 data).** From its PCs, the chip-7 sender's exact load address (kcb + 0xc0) in each
unicast_write iteration doesn't separate slow from normal: the same addresses and alignments appear in both. That
hypothesis is not supported.

**Periodic components (code reading). None at ~0.6-0.8 s, and the clock counts iterations anyway:**
- **Host, receiver.cpp:** a refclk burst every 10 ms (kRefclkBurstPeriod). It reads the root device's PCIe tile SII
  timer and a Tensix wall clock over PCIe through mapped TLB loads: 30 reads per burst, root device only.
- **Host, ingest threads:** they poll the host FIFOs with sleeps of at most 100 µs and return credits to the devices
  (PCIe writes) every kAckBatchPages pages.
- **Host, clock_map.cpp:** a 100 ms TSC-rate measurement, once at start.
- **Host, clock solver:** works on 1 s windows of bursts, host-side only.
- **Host, device_programs.cpp:** 1 ms polls during setup only.
- **Device, link_sync in the routers:** 10 ms rounds (kLinkSyncRoundTicks = refclk / 100) of 32 bursts; remeasure every
  2^20 cycles (~0.78 ms).
- **Device, eth_clock_sampler / eth_clock_model:** a sample every refclk update (80 ns), turned into clock points.
- **Device, drisc relay:** continuous sweeps. The ship threshold, deferral cap and idle-pump stride were ruled out (826,
  827).
- **Producer (kernel_profiler):** publishes its tail every 16 words and at every launch boundary. The ring is 512 words.

**Count-based candidates left (hypotheses, not tested):**
- the producer's word count crossing some boundary (~128 words ≈ 18 iterations at ~7 words per iteration), through
  something other than the relay's threshold;
- a host-side per-program or per-launch counter, for example dispatch or kernel-config ring state that recurs every
  ~18 programs and changes something the senders use, on the profiler-on build only.

Stopped here for review.

### Dispatch observation (broker 829, results/cisp_dispatch_20261105-100012; no build)

**Run.** NX ×8 with TELEM=ci, arm on (streaming profiler) then off, each arm its own cache (cisp_dispatch.sh).
pc_sampler.py --cores sampled 77 targets at ~975 Hz each, with mailbox snapshots every 50 ms:
- the dispatch column (noc0 x 16, rows 2-11, BRISC and NCRISC) on chips 0, 3 and 7;
- those chips' senders (f15-y BRISC).

The test's sender CSV marks the slow iterations: 38 of 288 on, 0 off. Dispatch PCs map to the cq_prefetch / cq_dispatch /
cq_dispatch_subordinate images in each arm's cache (load address = kcb + text offset). Tool:
port_tune/cisp/cisp_dispatch_report.py; its output is report.txt in the run dir.

**Active dispatch cores (the same on all three chips):**
- f16-2 BRISC runs cq_prefetch, f16-3 BRISC runs cq_dispatch, and f16-3 NCRISC runs cq_dispatch_subordinate.
- f16-4..10 stay in idle firmware.
- f16-11 is the realtime profiler's reserved core. It is active only in the off arm (kernel images of 1516 B on BRISC,
  1652 B on NCRISC). The streaming profiler leaves it idle (device_programs.cpp: "idle while the streaming profiler
  runs").
- In the off arm, dispatch_s also writes the realtime profiler's per-program records (open_record,
  record_realtime_timestamp: ~40% of its samples). **So the off arm is "streaming profiler off, realtime profiler on",
  not "no profiler".**

**Chip 7, share of each RISC's samples inside NX unicast_write iterations** (chips 0 and 3 are within ±4 points):

| RISC | where (source line) | on, slow | on, normal | off |
| --- | --- | ---: | ---: | ---: |
| prefetch BRISC | process_stall (cq_prefetch.cpp:1909) | 41.7% | 41.4% | 36.5% |
| | fetch_q_get_cmds, waiting on the host fetch queue (:943) | 31.2% | 28.2% | 30.6% |
| | semaphore load + invalidate_l1_cache + prefetch_q_read_slot | 27.2% | 30.3% | 32.8% |
| dispatch BRISC | process_wait, wait_stream: worker completions in its own stream register (cq_dispatch.cpp:1076, :1135) | 62.2% | 65.5% | 62.8% |
| | acquire_pages, waiting on prefetch (cq_common.hpp:664) | 35.8% | 32.0% | 34.9% |
| | everything else (invalidate_l1_cache, write barrier) | 2.0% | 2.5% | 2.3% |
| dispatch_s NCRISC | cb_acquire_pages_dispatch_s / wrap_gt / stream_wrap_gt / invalidate_l1_cache | 100% | 100% | ~60% |
| | realtime profiler records | — | — | ~40% |

**Answer to the question: no.** In slow iterations, prefetch, dispatch and dispatch_s sit in the same wait loops as in
normal iterations and in the off arm, in the same proportions (within ±6 points, about the chip-to-chip spread).
- Each loop polls the dispatch core's own L1 or stream registers. Worker completion arrives as a NoC atomic into
  dispatch's stream register.
- None of the loops touches the sender tiles.
- Writes to workers (process_write_packed_large, write barriers) are at most 0.2% of samples in every group.
- **Dispatch is not active during the slow iterations.**

**Senders in the same run** (f15-y BRISC, chips 0, 3, 7). Slow iterations show the 824 shift again:
- NoC command-buffer waits 9.3% → 12.9% of all BRISC samples;
- write-ack waits 0.1% → 1.5%;
- fabric credit waits 9.4% → 7.0%.

### Kernel-config ring and launch ring (offline: 829 snapshots and sender PCs)

- **Kernel-config ring:** from MEM_MAP_END (0xa3a0), 69 KB (default_l1_kernel_config_size, bh_hal_tensix.cpp), so
  0xa3a0..0x1b7a0. The same in both arms.
- **Launch-message ring:** 8 slots in both arms.
- **One sender program per iteration** on a sender core.

| arm | sender BRISC text | config stride per program | programs per ring wrap | positions in the ring |
| --- | ---: | ---: | ---: | --- |
| on | 12200 / 12216 B | 12400 / 12432 B | 5 | drift by 32 B with the mix of preceding programs (0xd410/0xd430, 0x10480-0x104c0, 0x134f0-0x13550, 0x16580-0x165c0) |
| off | 10728 / 10712 B | 10912 / 10944 B | 6 | fixed (0xa3a0, 0xce40, 0xf8e0, 0x12380, 0x14e40, 0x17900) |

- The profiler's zone code makes the sender ~1.5 KB larger, so 5 programs fit per wrap instead of 6.
- The joint phase of the two rings repeats every lcm(5, 8) = 40 launches on and lcm(6, 8) = 24 off. **No ~18-program
  cycle falls out.**
- Per iteration (sender PC → the launch slot whose image holds it → that slot's kcb; chips 0, 3, 7), neither the kcb nor
  the launch slot separates slow from normal:
  - each of the 13 on-arm kcb values has both slow and normal iterations (most 3 and 3);
  - each launch slot has 4-6 slow and 5-8 normal.
- This agrees with 824's load-address check.

### The ~18-iteration period is the yaml's layout (offline: 824, 826 ×3 arms, 828 ×2, 829)

**Row layout.** Each repeat of pcs_nx_ci_x8.yaml runs size 2048, then size 4096. Each size runs {unicast_write,
fused_atomic_inc, unicast_scatter_write} × num_links {1, 2} × 3 iterations, so a repeat is 36 iterations. A 6-iteration
unicast_write block therefore starts every 18 iterations.

**Only unicast_write rows ever go slow, in every run:**
- unicast_write: 29-38 of 96 iterations slow;
- fused_atomic_inc and unicast_scatter_write: 0 of 96 each. Their ratio to the off median never drops below 0.9995.
- unicast_write is bimodal: ~0.952 or ~1.000. Only 2 of 672 iterations fall between 0.97 and 0.99.

**Where in a block** (7 runs, 229 slow iterations, by position 0-5 inside the unicast_write block):

| position | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| slow iterations | 1 | 15 | 61 | 74 | 60 | 18 |

Positions 0-2 are num_links 1, iterations 0-2; positions 3-5 are num_links 2. The slowdown almost never hits a block's
first iteration. It starts 1-3 iterations in, crosses the links-1 → links-2 row boundary, and often ends before the
block does. 10-12 of the 16 blocks per run show it.

829, arm on, one line per 36-iteration repeat (S: slow):

```
....................SSS.............
..S..................SSS............
...................SSSS.............
..SSSS..............SS..............
.SSSSS..............................
.SSS................SSS.............
..SS................SSSS............
...................SSSS.............
```

**What this changes:**
- Any trigger would give a period of ~18 iterations here: the period is set by where the unicast_write rows sit. So:
  - 826/827's "the period didn't move, so the relay constants are ruled out" doesn't hold. Marked WRONG above.
  - 828's "the clock counts iterations" doesn't hold. Marked WRONG above.
- What 828 still shows: the slow positions inside the blocks are the same at ×1, ×2 and ×4 packets. A time-based clock
  would drift against the rows as the iteration length changes, so the slowdown follows the row sequence.
- The words-vs-programs discriminator assumed an ~18-count clock. It is not needed in that form.

**Reading (hypothesis, untested).** The slowdown is a profiler-on state that builds up over consecutive unicast_write
iterations and is absent in fused_atomic_inc and scatter rows. **WRONG (830): with write rows only (A) it doesn't build
up; it stays at the same positions of each 6-iteration size group. With one write per row (C) the same ~4.7% state
appears with the profiler off.** It is not a periodic event. Dispatch,
kernel-config placement and launch slot are not involved (829, 824).

**Proposed next check (no build, yaml only; for the coordinator, not run).** Change only the row order, with the profiler
on:
- **A:** unicast_write rows only, ×8. If the slowdown builds up with consecutive write iterations, long slow runs follow
  a short lead-in. If a preceding non-write row sets it off, nothing is slow.
- **B:** the stock rows with the unicast_write block moved behind the scatter rows. The slow positions should move with
  the block.

Stopped here for review.

### Row-order checks (broker 830, results/cisp_orderab_20261105-101713; yaml only, no build)

**Arms.** Each variant ran profiler on, then off (the sender-side baseline), in one job. Variants
(port_tune/cisp/mk_order_yamls.py):
- **S:** the stock yaml (pcs_nx_ci_x8.yaml), a same-job control;
- **A:** unicast_write rows only, ×8: 12 back-to-back write iterations per repeat (2048 L1, L2, then 4096 L1, L2);
- **B:** the stock rows with the ntype order fused, scatter, write. The write blocks move to positions 12-17 and 30-35;
- **C:** one write iteration per row, each followed by a tiny fused_atomic_inc row (2048 B, 1000 packets, same
  num_links). Writes never run back to back.

A row is slow below 97% of the off arm's median for the same size / ntype / num_links, over all repeats. Tool:
port_tune/cisp/cisp_order_report.py; report.txt in the run dir.

| variant | slow write iterations, on | slow, off | slow by position in the write block |
| --- | ---: | ---: | --- |
| S (stock) | 33/96 (12 of 16 blocks) | 0/96 | 0:0 1:1 2:10 3:11 4:9 5:2 |
| B (write block last) | 32/96 (12 of 16) | 0/96 | 0:0 1:1 2:10 3:11 4:9 5:1 |
| A (write only; 12 per repeat) | 30/96 (7 of 8 repeats) | 0/96 | 0:0 1:0 2:3 3:5 4:4 5:2 6:0 7:0 8:5 9:6 10:5 11:0 |
| C (one write per row; 12 per repeat) | 28/96 | **14/96** | on 0:2 1:3 2:1 3:2 4:1 5:0 6:4 7:3 8:6 9:5 10:1 11:0; **off 0:4 1:2 6:7 7:1** |

Fused and scatter rows are never slow, in any variant or arm (C's tiny gap rows scatter ±0.8% around their median, from
their size).

**B: the slowdown moves with the write block.** Its position distribution inside the block is the same as S's. It
follows the row sequence, not the position in the test or the time since the test started.

**A: no build-up.**
- 12 writes run back to back, yet the slowdown stays at positions 2-4 of each 6-iteration size group (L1 it2, L2 it0,
  L2 it1), as in S.
- The first two iterations of each size group (6, 7) are never slow, even right after slow iterations of the previous
  group.
- So neither "builds up over consecutive writes" nor "set off by a preceding non-write row" holds.
- The slow window sits around the num_links 1 → 2 switch within a size.

A, on arm, ratio per iteration (one line per repeat):

```
1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000
1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 0.954 0.954 1.000
1.000 1.000 1.000 0.954 0.954 1.000 1.000 1.000 0.952 0.954 0.954 1.000
1.000 1.000 0.953 0.955 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000
1.000 1.000 1.000 0.954 0.942 0.953 1.000 1.000 0.952 0.954 0.954 1.000
1.000 1.000 0.953 0.954 0.955 1.000 1.000 1.000 0.952 0.954 1.000 1.000
1.000 1.000 0.953 1.000 1.000 1.000 1.000 1.000 0.952 0.954 0.954 1.000
1.000 1.000 1.000 0.955 0.955 0.953 1.000 1.000 0.952 0.954 0.954 1.000
```

**C: the slow state appears with the profiler off.**
- 14 of 96 off-arm write iterations sit at 0.952-0.958, the same levels as the on arm's.
- They fall on the first writes of a num_links-1 group:
  - w06 (the first 4096 B L1 write, right after a fused L2 gap row) in 7 of 8 repeats;
  - w00 or w01 (the first 2048 B L1 writes) in 6 of 8.
- With the profiler on, C's slow writes spread over most positions.
- **So the ~4.7% slow state is not a profiler cost.** It is a state of the write path that the row sequence sets, and
  the profiler changes which iterations land in it. In the stock order the off arm never lands in it (0 of 96 in 8
  stock runs), which is why the CI comparison shows it as a profiler regression.

**Discrete levels.** Slow iterations sit at ~0.953, a few at ~0.958 and ~0.942-0.943. Nothing falls in between.

**Reading (hypotheses, untested):**
- The slow state is a setup-dependent condition of the unicast_write path, for example which link or router a flow
  lands on, or where the sender's or receiver's buffers sit. It is selected per row or iteration by the sequence of
  test configs and is independent of the profiler.
- The ~4.7% step and the discrete levels suggest one of several flows (links or directions) running slower, rather
  than a uniform slowdown.

**Not done; for the coordinator:**
- Which flow or core is slow in a slow iteration: per-device / per-direction rows, or per-sender zone durations.
- What differs in the setup between C's slow and normal off-arm writes: allocations, link choice, buffer addresses
  from the log.

Stopped here for review.

### Is the slow state an AICLK change? (offline: host time against device cycles, 830 and 826-829)

**Question (coordinator).** The sender CSV and the router telemetry count AICLK cycles and convert them at a fixed
1.35 GHz. If AICLK ran ~5% higher in slow iterations, NoC- and link-bound work would take more cycles with real time
unchanged.

**(a) Per-iteration ns from the profiler: not available.**
- The zone CSV gives cycles only ("time[cycles since reset]", header CHIP_FREQ 1350).
- The clock solver's "AICLK chip{N} (GHz)" plots go to Tracy only, and no Tracy capture was taken.
- The sync check (TT_METAL_STREAMING_PROFILER_SYNC_CHECK) prints per-chip AICLK statistics, but it was off in every
  run. Turning it on rebuilds the routers (LINK_SYNC_CHECK), so it isn't a read-only option.

**(b) Host time settles it.**
- Per unicast_write iteration, the window from "Waiting for programs" to "Test ... Finished." is taken from log
  timestamps (1 ms resolution, averaged over many iterations).
- It is compared with what the sender row's extra cycles would add at 1.35 GHz.
- A clock change with real time unchanged predicts a window difference near 0. A real slowdown predicts the cycle
  excess.
- Tool: port_tune/cisp/cisp_hosttime.py.

| runs | row | slow / normal | window diff, slow - normal | cycles predict |
| --- | --- | ---: | ---: | ---: |
| 12 runs at 100k packets (826 ×3, 827 ×3, 829, 830 S/A/B/C on, C off) | 2048 L1 | 57 / 231 | +0.32 ± 0.08 ms | +0.44 ms |
| | 2048 L2 | 88 / 200 | +0.48 ± 0.06 ms | +0.43 ms |
| | 4096 L1 | 81 / 207 | +0.42 ± 0.06 ms | +0.45 ms |
| | 4096 L2 | 138 / 150 | +0.42 ± 0.05 ms | +0.43 ms |
| 828 ×2 (200k packets) | 2048 L2 | 8 / 16 | +0.94 ± 0.06 ms | +0.84 ms |
| | 4096 L2 | 15 / 9 | +0.78 ± 0.14 ms | +0.87 ms |
| 828 ×4 (400k packets) | 2048 L2 | 9 / 15 | +1.71 ± 0.15 ms | +1.73 ms |
| | 4096 L2 | 13 / 11 | +1.51 ± 0.22 ms | +1.70 ms |

- **Slow iterations take longer in host time by about the cycle excess.** The difference scales with the run length
  (×1, ×2, ×4), and it holds for the profiler-off slow writes of C.
- Real time and cycles grow together. AICLK in slow iterations is the same as in normal ones within about 1%:
  - (cycle excess − time excess) / kernel time ≈ 0.4% ± 0.6% at ×1;
  - ≈ 0.6% ± 0.5% at ×4.
- A 5% AICLK change is ruled out, so no AICLK read-back run is needed. **The slowdown is real throughput loss.**

**Pointer for step 1 (not analyzed).** The per-router telemetry lines in the logs show two discrete router rates
within one iteration. For example, 830 S, R03 4096 L2: some routers at ~48.45 GB/s (~11.54M cycles), others at ~46.3
(~12.08M), a ~4.5% step. Whether the share of routers at the low rate separates slow from normal iterations can be
tallied offline from the existing logs.

Stopped here for review.

### Per-router tally (offline: 826-830, every arm; port_tune/cisp/cisp_router_tally.py)

**Method.**
- Per NX unicast_write iteration, each router's telemetry cycles (the router's own sending, busy rounds) are compared
  with its baseline: the median of the same row type (size / num_links / packets) in the job's off arm, or in its own
  iterations for an off arm.
- A router is low-rate above +2%.
- Runs: 826 base / shipA / pumpB / off, 827 base / deferC / deferD / off, 828 ×2 and ×4 on / off, 829 on / off, and
  830 S / A / B / C on / off.
- Naming: physical chip and eth channel, the peer it sends to, the direction, the plane (P0 is the num_links-1 link,
  P1 the second link), and the link-sync role (T when chip id < peer's).
- A telemetry line is the sender side of the flow chip → peer.
- Output: results/cisp_orderab_20261105-101713/router_tally_826_830.txt.

**Two discrete levels per router.**
- Every router-iteration is either at its baseline or about +4.6-4.9% above it. Nothing falls in between: 31 of
  ~63.8k router-iterations sit at "+4%", the rest at +0% or +5%.
- Routers that are low in the same iteration share the same step (within 0.3% in 535 of 545 iterations).

| row | normal | low | step | per packet |
| --- | ---: | ---: | ---: | ---: |
| 2048 L1, 100k | 11.56M | 12.12M | +4.81% | +5.6 cycles |
| 2048 L2, 100k | 11.55M | 12.08M | +4.59% | +5.3 cycles |
| 4096 L1, 100k | 11.55M | 12.11M | +4.87% | +5.6 cycles |
| 4096 L2, 100k | 11.54M | 12.08M | +4.64% | +5.4 cycles |

- At 200k and 400k packets the steps are the same (+4.6-4.9%).
- Both sizes run at ~115.5 cycles per packet, so the rate is packet-bound, and the low level adds a fixed ~5.5 cycles
  per packet.

**How many routers are low, slow against normal iterations (CSV flag).**
- Slow iterations: 1 to 25 low routers, out of 20 (num_links 1) or 40 (num_links 2).
- Normal iterations: mostly none. 7-11 per arm have 1-7 low routers.
- The off arms of the stock order (S, A, B and 826-829 off): no low router in any write iteration.
- C off: 14 slow iterations (1-7 low routers each) and 7 normal ones with 1-3.

**The CSV flag follows the end-of-traffic sync, exactly.**
- The CSV row is chip 7's W sender. Every slow-flagged iteration in all 13 arms that have one has a low router whose
  source is chip 7, 3 or 6 (chip 7 and its two sync neighbors).
- No normal-flagged iteration does: their low routers are all on other chips (0 exceptions in ~1250 write iterations).
- The slow state therefore occurs in more iterations than the CSV shows, for example 826 base: 36 flagged plus 7
  elsewhere.

**Not a stable subset, and no direction or role preference.**
- Pooled over all arms, every one of the 40 routers is low in some slow iterations (9-38% of them).
- No router is low in at least half the slow iterations of more than 2 of 13 arms. Each arm's majority set is empty or
  one router, and it changes from arm to arm (Jaccard 0).
- In num_links 2 iterations:
  - N/S and E/W routers are low equally often (10.0% / 10.2% of router-iterations), and so are T and R (10.0% / 10.2%).
  - P1 is slightly more often low than P0 (11.7% / 8.6%).

**Not chip-wide, not link-wide.**
- In (iteration, source chip) groups with a low router, the chip's other active routers are all low in only 29 of
  2113.
- Destination chips with every inbound router low: 22 of 2190.
- The opposite direction of a low router's link is also low for 31% of low routers, about the overall share in those
  iterations.

**The test's end sync** (tt_fabric_test_sender.cpp, tt_fabric_test_sync.cpp, tt_fabric_test_common.hpp:1652).
- Each sender, after close_connections, takes two local syncs with the chip's sync core:
  - the first tells the sync core the senders are done;
  - the second waits for the sync core's global sync.
- The measured cycles (total_elapsed_cycles_outer_loop) stop only after both, so every sender on a chip ends together.
- For NeighborExchange the global sync is with the nearest neighbors only (global_sync_val = their number). So one slow
  flow stretches every sender on its own chip and on its neighbors.
- **So 823's "every sender kernel on an affected chip runs long" doesn't show a chip-wide slowdown.** The per-router
  data shows per-flow slowdowns: random subsets of routers, each at the same discrete +~5.5 cycles per packet. The
  coordinator's "chip-wide" reading of 823 is not supported. 823's zone excess is the end sync waiting for the slowest
  flow among the chip and its neighbors.

**Reading (hypotheses, untested).**
- The slow state is a per-flow (per sender → router path) state with a fixed extra ~5.5 cycles per packet.
- Which flows enter it changes from iteration to iteration. Many flows can be in it at once.
- It depends on the row sequence, and with the profiler on it is entered more often.
- A router-side cause would show as the sender waiting more on credits, but 824 showed less credit wait and more NoC
  command-buffer / ack wait in slow iterations. So the extra per-packet cost sits on the sender's NoC path into the
  router, or in the router's per-packet handling of that channel, rather than in link or credit capacity. Not settled.

Stopped here for review. The low-rate routers are not a stable subset, so the read-only router-state capture design
waits for the coordinator.

### Slow-state checks: one flow at a time (broker 831) and the local routes (offline)

**Device check X1** (results/cisp_flowab_20261105-162525; yaml only; port_tune/cisp/cisp_flow_ab.sh, mk_flow_yamls.py).
- **X1:** the test infra's sequential_neighbor_exchange on the NeighborExchange topology, so the router images are the
  same as the NX rows'. Each iteration runs one directed neighbor pair (all 20 in turn), unicast_write, 2048 and 4096 B,
  num_links 1 and 2, 100k packets, ×8 repeats: 640 iterations per arm.
- An explicit-senders yaml on the NeighborExchange topology crashes in ControlPlane::get_chip_neighbors on the mock, so
  it wasn't used.
- **Control A:** the NX write rows, same job.
- Profiler on, then off, for both.
- **Measure:** the router tally (+2% over the off arm's per-router median), with the mesh mapping now pinned to the
  hardware's (port_tune/cisp/a09u32_hw_mesh_mapping.yaml). The live generated/fabric file is rewritten by every run,
  mock runs included. The pinned mapping matches all 197,760 raw-telemetry link rows of 823-831.

| arm | num_links 1 (one flow, one link) | num_links 2, link-0 router (P0) | num_links 2, link-1 router (P1) |
| --- | ---: | ---: | ---: |
| X1 on | 0 / 320 | 80 / 320 | 0 / 320 |
| X1 off | 0 / 320 | 10 / 320 | 0 / 320 |
| A on (NX control) | 28 of 96 iterations slow, 1-23 low routers each | | |
| A off | 0 / 96 | | |

**A single flow on a single link never enters the state** (0 of 640 router-iterations, profiler on or off). With two
links of one pair active, only the link-0 router slows, and on the on arm only for a fixed set of pairs:
(**Too narrow, WRONG as a condition (coordinator / user review):** the NX num_links 1 rows slow too, with the same
+~5.5 cycles/packet, and there each chip runs 2-3 links, one per neighbor. Only X1's lone flow, one active link per
chip, never slowed. The condition is two or more concurrent flows on a chip, in any arrangement, chip-dependent, at a
per-packet cost. The 2048 and 4096 B rows take the same cycles/packet.)

| pair (source → destination, direction) | P0 / P1 channels at the source | P0 low, on | P0 low, off |
| --- | --- | ---: | ---: |
| c1 → c5 (S) | e8 / e10 | 9/16 | 0/16 |
| c1 → c0 (E) | e5 / e7 | 11/16 | 1/16 |
| c2 → c6 (S) | e8 / e10 | 6/16 | 2/16 |
| c2 → c3 (E) | e5 / e7 | 14/16 | 0/16 |
| c5 → c1 (N) | e11 / e9 | 8/16 | 0/16 |
| c5 → c4 (E) | e7 / e5 | 14/16 | 2/16 |
| c6 → c2 (N) | e11 / e9 | 7/16 | 2/16 |
| c6 → c7 (E) | e7 / e5 | 11/16 | 0/16 |
| the other 12 pairs: all from c0, c3, c4, c7, and the W pairs of c2 and c6 | | 0 each | 0-2 each |

- The low pairs' sources are chips 1, 2, 5 and 6, at mesh columns 0 and 2. Their S/N and E pairs are low; their W pairs
  are not.
- Full per-pair list: results/cisp_flowab_20261105-162525/x1_pairs.txt.

**Route analysis (offline).**
- **Cores.** From a debug-logging mock of the NX and X1 rows (port tree, separate build_Dlog with
  TT_METAL_ENABLE_LOGGING; "registered with connection_manager" lines):
  - every chip has one sender core per link plane, serving all of the chip's directions on that link (round-robin
    over 2-3 traffic configs in NX): logical (11,8) = physical (15,10) for link 0, (11,7) = (15,9) for link 1;
    **WRONG for the benchmark runs: the mock ran with benchmark_mode false. In bandwidth mode the test infra
    splits every multi-pattern sender by direction (tt_fabric_test_config.cpp, split_senders_by_direction_for_benchmark),
    so NX runs one sender core per (direction, link): the 823 zone CSV shows chip 0 with senders (15,8), (15,9),
    (15,10) plus the sync core (15,11) in num_links 1 rows, and six senders in num_links 2 rows. X1's (15,10) / (15,9) are
    right (one direction, two links).**
  - the sync core is (15,11);
  - receivers are logical (0,0..) = physical (1,2), (1,3), ..., one per incoming flow up to the pool size (X1 num_links
    2: (1,2) and (1,3)).
- **Eth cores** (arch order with eth_harvesting_mask 288): e4 (3,1), e5 (4,1), e6 (13,1), e7 (12,1), e8 (6,1),
  e9 (11,1), e10 (7,1), e11 (10,1).
- **NoCs.**
  - Sender → router writes: NoC0 (824).
  - Router → receiver: NoC1 (receiver_channel_local_write_noc_ids = 1 in every case, erisc_datamover_builder.cpp,
    fabric_core_placement.cpp).
  - Credits back to the sender: NoC0 or NoC1 by the routers' x order.
- **Routing model.** NoC0 routes +x first, then +y; NoC1 routes −y first, then −x (both wrap; NoC1 as the tree's own
  comments state it).
- **X1, two links:**
  - Source chip: the senders' X segments run on rows 10 and 9, and their Y segments on the two routers' columns. No
    link carries both.
  - Destination chip: the two deliveries climb the two routers' columns, then run west on rows 2 and 3. No shared
    link.
  - Only the small NoC0 write acks share links (row 1 east of the routers, column 15).
- **Same geometry, different outcome.** c1 → c5 (low 9/16) and c0 → c4 (never low) use the same channels on both ends
  (e8/e10 out, e11/e9 in), the same sender and receiver cores, and the same harvesting (mask 128 on every chip). Their
  NoC paths are identical, yet one pair is low and the other never is. So link sharing on the local NoC paths doesn't
  explain which flows slow.
- **NX overlap correlation: not done.** Two NX flows land on the same receiver core where a chip has more incoming
  flows than receivers, and the round-robin assignment of flows to receivers wasn't logged. Given the X1 result it
  can't separate the cases. X2 (moving cores to disjoint routes) isn't indicated: X1 already has disjoint routes.

**Reading (hypotheses, untested).**
- **Not (B) as stated.** A lone router never enters the state, with the profiler on or off.
- **Not (A) as local-NoC link sharing.** It needs a second active link of the same chip pair, yet it occurs with
  disjoint NoC paths and depends on which physical chips carry the pair. (WRONG in part: "a second active link of the same chip pair" is too narrow; any second
  concurrent flow on the chip, see above.)
- **Candidates:**
  - something shared by the two links of a pair outside the modeled NoC links: the eth ports' physical layer
    (adjacent ports, error/retry behavior), or a NoC router port or NIU at a crossing point;
  - a chip-specific property of chips 1, 2, 5 and 6.
- **Why the profiler raises the rate.** The profiler-on arm enters the state far more often (80 against 10 of 320),
  and its pairs are the same chips. One hypothesis: the profiler's idle-eth relay or its NoC traffic on those chips is
  the extra load.

Stopped here for review.

### X3: outbound against inbound concurrency on one chip (broker 832, results/cisp_flow3ab_20261105-165255)

**Rows** (yaml only; mk_flow_yamls.py; NeighborExchange topology, num_links 1, unicast_write, 2048 and 4096 B, 100k
packets, 3 iterations, ×8). The subject chips are mesh [0,0] = chip 1 and [0,2] = chip 2, both slow in X1:
- **Lone:** the subject sends one flow (E).
- **SO:** the subject sends two flows (E and S). Nothing is inbound to it.
- **SR:** the subject sends S while its E neighbour sends W into it.
- **RR:** [0,2] only; its W and E neighbours send into it, and it sends nothing.

Senders are given by destination hops. Two things fail on the mock: explicit device destinations on this topology
(ControlPlane::get_chip_neighbors), and N hops from row 1 (MeshGraph::chip_to_coordinate map::at). So inbound flows
come from row-0 neighbours, and [0,0] has no RR arm. **WRONG (cause): a yaml `device: [a, b]` is [mesh_id, chip_id], not [row, col]
(row-major chip ids, [0, 4..7] the bottom row). Both crashes came from writing a row-1 chip as [1, c], a mesh id of 1:
X1's explicit-device yaml and the N hops. The X3 / X4 rows use [0, c] = mesh chip c = mesh [0, c], so they are
correct as run. X5 writes row-1 chips as [0, 4 + c].** The NX write rows ran as a same-job control.

| arm | profiler on: iterations with a low router (of 48) | off |
| --- | ---: | ---: |
| Lone [0,0] / [0,2] | 0 / 0 | 0 / 0 |
| SO [0,0] / [0,2] (two outbound) | 5 / 6, the E router only (c1/e5, c2/e5); S never | 0 / 1 |
| SR [0,0] / [0,2] (one out, one in) | 0 / 0 | 0 / 0 |
| RR [0,2] (two inbound) | 0 | 0 |
| NX control (of 96) | 42 | 0 |

**Concurrency alone isn't enough.** Inbound concurrency (receiving while sending, or two receptions) never enters the
state here. Two outbound flows from one chip do, at a lower rate than NX. **WRONG as a trigger (X4 / X5): X3's arrangement didn't reproduce, and full NX
concurrency at 1 link never entered the state.**

**Sender cores** (debug-logging mock): SO's two flows have their own sender cores. The E flow goes from (15,10)
(logical 11,8) to e5, and the S flow from (15,9) (logical 11,7) to e8, each a single-config stateful sender.
- In SO and in X1 (two links), the flow that slows is always the one from (15,10), and only while (15,9) also sends.
- In NX one core per plane serves all of a chip's directions (round-robin), and flows on both planes slow. **WRONG: on hardware NX runs one sender core per (direction, link) (the benchmark-mode split), so NX's concurrency is also several sender cores.**

**Reading (hypotheses, untested):**
- **The shared per-packet resource is on the outbound side.** Two sender cores write into their routers concurrently.
- **Candidates:**
  - the senders' per-packet NoC0 write acks and the routers' per-packet credit returns, which converge on the
    senders' column (x = 15) and run along row 1;
  - the crossing of the (15,9) flow's column segment with the (15,10) flow's row-10 segment at one NoC router;
  - something in the eth subsystem that both TX paths use.
- **Next observation, no build** (for the coordinator):
  - SO with the cores swapped (a core list, if the yaml allows placing sender cores), to see whether the slowed flow
    follows the core (15,10) or the direction / router;
  - SO on chips 0 and 3, to see whether the chip dependence holds for outbound-only concurrency.

Stopped here for review.

**Correction (the sender arrangement):**
- In bandwidth mode, split_senders_by_direction_for_benchmark gives every (device, direction, link) its own sender core.
  So NX, X1 (two links), X3's SO and X4 all run their concurrent outbound flows on separate sender cores.
- "NX slows with a single core" (from my mock-based claim) doesn't hold.
- The yaml sender policy (allocation_policies.sender) only packs whole sender entries, and the split runs before
  allocation. A one-core arm with two directions can't be expressed in benchmark mode without a build, so X5's SO1 / NX1
  arms (broker 835, queued before this was found; the cancel was refused) run the same as SO2 / NXs.

### Start skew between neighbouring chips (offline, 823 zone CSV; port_tune/cisp/cisp_skew.py)

**Method.**
- Each chip's sync core (15,11) runs one kernel per iteration. Sender kernels (x = 15, other rows) start within about
  0.1 us of it and are assigned to the nearest sync start.
- Per (chip, iteration): the first sender start and the last sender end.
- For each mesh-neighbour pair, the start (end) difference minus its median over the CSV-normal NX write iterations.
  That removes each chip's constant clock offset and any fixed launch order.
- Pairs are split by whether a chip of the pair sources a low-rate router in that iteration (router telemetry, as in
  the tally).

**Starts line up. No neighbour starts late in slow iterations.**
- Neighbour start skew moves by a few us around its median in every group:
  - pairs with neither chip low: median 7.7 us, p90 23 us;
  - pairs with one chip low, CSV-slow iterations: median 8.7 us, p90 24 us.
- Signed, low-source chip minus its neighbour: median +0.07 us, p10 −17.6, p90 +18.0 (n = 150).
- A ~20 us start jitter is 0.2% of a ~9 ms kernel. The slowdown is ~435 us.

**The slow flow runs longer.**
- Sender span per chip (first start to last end):
  - ~9020-9040 us in iterations with no low router anywhere;
  - ~9450-9480 us in iterations where the chip itself sources a low router.
- The same ~9450-9480 us holds where only another chip does, through the nearest-neighbour end sync.
- End-skew outliers of ~445 us (p90, CSV-slow pairs with neither chip low) are that sync: one chip of the pair waits
  for its other, slow neighbour.
- Together with the router telemetry (+4.6-4.9% busy rounds on the low router), the slowdown is a lower per-packet
  rate over the whole iteration, not a late start.

### X4 and X5 (broker 834, 835; results/cisp_flow4ab_20261105-170357, cisp_flow5ab_20261105-170558)

**Setup** (1 link, write, 2048 / 4096 B, 3 iterations ×8, profiler on and off, the NX write rows as same-job control):
- **X4:** SO with placed sender cores (yaml `core`). SOdef: E (11,8), S (11,7), as X3's allocation. SOswap: the two
  swapped. SOfar: S from (5,2) = noc0 (6,4), off column 15 and off (15,10)'s path. Subjects mesh [0,0] and [0,2], plus
  SOdef on [0,1] and [0,3].
- **X5:** SO as one entry (SO1) or two (SO2), and NX's 20 flows written explicitly as one entry per chip (NX1) or per
  flow (NXs). The benchmark split makes SO1 = SO2 and NX1 = NXs, so X5 is a replicate.

| arm | on: iterations with a low router | off |
| --- | ---: | ---: |
| X4 SOdef [0,0] / [0,1] / [0,2] / [0,3] | 0 / 0 / 0 / 0 (of 48 each) | 0 |
| X4 SOswap, SOfar [0,0] / [0,2] | 0 each | 0 |
| X4 control: NX write rows | 39 / 96 | 0 |
| X5 SO1, SO2 [0,0] / [0,2] | 0 each (of 48) | 0 |
| X5 NX1, NXs: NX's 20 flows, explicit, 1 link | 0 / 48 each (0 of 960 router-iterations) | 0 |
| X5 control: NX write rows | 37 / 96 (316 of 2880 router-iterations) | 0 |

- **X3's SO doesn't reproduce.** SOdef has X3's exact arrangement and is 0 / 48 here, against X3's 5 / 48 and 6 / 48.
  So the swap and move arms can't discriminate.
- **The NX flows alone don't enter the state.** NX's full set of 20 flows, written explicitly and run at 1 link, never
  enters it (0 / 960 router-iterations). The stock NX write rows in the same job do (316 / 2880).
- **Across all runs, the state appears where num_links 2 rows run:**
  - the NX rows, where a size group runs L1 ×3 then L2 ×3, and the slow positions are L1 it2, L2 it0 and L2 it1;
  - X1, where the L2 pairs slowed and L1 never did.
  - The 1-link-only tests (X3, X4, X5) were nearly or entirely free of it, X3's 11 of 192 events aside.
- **WRONG, as in this section's earlier reading:** "outbound concurrency is the trigger" (X3). Outbound-only concurrency
  doesn't reproduce it here, and full NX concurrency at 1 link doesn't either.

**Reading (hypothesis, untested).** The state is tied to the num_links 2 configuration or the transition into it (for
example fabric or router state set up when a 2-link test opens), not to concurrent flows as such.

**Next check (yaml only, for the coordinator):** in one job, the stock NX write rows as L1-only blocks against L2-only
blocks against the stock L1/L2 alternation, with profiler on and off.

### X6: the num_links discriminator (broker 838, results/cisp_flow6ab_20261105-171231)

**Arms** (same job, profiler on and off; yaml only, mk_order_yamls.py): the NX write rows (2048 and 4096 B, 3 iterations
per row, ×8) as:
- **A:** the stock L1/L2 alternation (per size, L1 ×3 then L2 ×3);
- **L1:** L1-only blocks;
- **L2:** L2-only blocks.

The L2-with-link-0-only arm can't be written: expand_link_duplicates copies every sender onto every link when
num_links > 1 and overwrites the yaml's link_id. It would need a build.

| arm | on: iterations with a low router | off |
| --- | ---: | ---: |
| A (alternation) | 38 / 96 | 0 / 96 |
| L1-only | 0 / 48 | 0 / 48 |
| L2-only | 0 / 48 | 0 / 48 |

**A, on.**
- By plane, low router-iterations: P0 150 / 1920, P1 126 / 960.
- By direction: N 63 / 576, E 82 / 864, S 51 / 576, W 80 / 864.
- Every chip sources low routers (23-50 each).
- Positions in each test's 12 iterations (2048: L1 0-2, L2 3-5; 4096: L1 6-8, L2 9-11): 0:0 1:2 2:5 3:5 4:4 5:1 6:0
  7:3 8:6 9:6 10:5 11:1.
- So the state builds up over the L1 iterations that follow an L2 block (never the first L1 iteration), runs across the
  L1 → L2 switch for two iterations, and is gone by L2 it2.

**Neither num_links 1 nor num_links 2 alone enters the state.** Only the alternation does. It ties the slow state to
the switching between link counts: the link-1 routers going idle after carrying traffic, then active again. Neither the
second link's traffic nor concurrency as such is enough.

**Hypotheses, untested:**
- Activity on the routers that just went idle (link 1 during L1 rows after an L2 block) slows the active ones until the
  idle routers carry traffic again. Examples: the profiler's per-router link-sync rounds or the idle path's periodic
  work, through a shared eth / NoC resource. The profiler-on arm is where it shows here.
- C's off-arm events (one write per row with gap rows) fit the same reading: rows that change which routers carry
  traffic.

**Next (for the coordinator):** read the idle routers during an L1 row that follows an L2 block, read-only.
- Sampling, no build: pc_sampler on the link-1 routers' ERISCs.
- Or yaml only: L2 → L1 alternation with an idle gap between them.

Stopped here for review.

### X7: router PC samples, L1 after L2 against L1-only, plus an idle gap (broker 839, results/cisp_pcs3_20261105-173030)

**Run** (profiler on, read-only): consolidation/pc_sampler.py --only-loaded on all 40 routers' ERISC0 and ERISC1
(about 78k samples/s, 0 read errors). Three arms:
- (i) the stock L1/L2 alternation (pcs_nx_wonly_x8.yaml);
- (ii) L1-only blocks;
- gap: the alternation with a near-idle pause after each L2 block, 10 iterations of a 1-packet NX row at num_links 1
  (pcs_nx_gap_x8.yaml).

The same yamls ran with the profiler off, unsampled, as the router tally's baselines. Tool: port_tune/cisp/
cisp_x7_report.py (x7_report.txt, x7_report_full.txt).

**Slow-state incidence in this run:** (i) 39 / 96, (ii) 0 / 48, gap 41 / 96 write iterations (gap rows excluded). This
repeats X6.

**The idle link-1 routers look the same whether the state is building or not.** ERISC0 bucket shares:

| group | passes | end_round | ctx_switch | telemetry | link_sync | holds_packet (line) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (i) idle link-1, slow iterations (4146) | 83.7% | 2.9% | 1.4% | 11.0% | 0.8% | 1.18% |
| (i) idle link-1, normal iterations (5936) | 83.5% | 2.8% | 1.9% | 11.0% | 0.9% | 0.76% |
| (ii) idle link-1, never slow (8149) | 83.5% | 3.3% | 2.0% | 10.2% | 1.0% | 0.91% |
| gap idle link-1, slow / normal (4030 / 5908) | 84.5 / 84.4% | 2.8 / 2.8% | 1.5 / 1.9% | 10.0 / 9.9% | 1.1 / 1.0% | 0.82 / 0.66% |

- ERISC1 matches the same way: passes 89-91%, with the same top lines (follow_context_switch, end_round,
  get_timestamp).
- The holds_packet line (SenderChannel::holds_packet, router_sender.hpp:75) is within sampling noise across groups
  (about 50-75 samples each).
- Nothing on the idle routers' PCs separates (i) from (ii): no retire, credit, link-sync or poll loop appears only
  where the state builds.
- The active routers' samples are also alike when low and when not, in every arm: ERISC0 passes 85.5-87.1%, telemetry
  10.4-11.5%.

**The idle gap moves the state earlier.** Slow write iterations by position, of 8 each:

| block | L1 it0 | L1 it1 | L1 it2 | L2 it0 | L2 it1 | L2 it2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 B, after the 4096 block's gap | 5 | 2 | 1 | 5 | 4 | 0 |
| 4096 B, after the 2048 block's gap | 4 | 4 | 4 | 7 | 5 | 0 |

- In the stock order, the first L1 iteration after an L2 block is never slow (X6: 0 of 16). After a 10-iteration gap
  it is slow in 9 of 16.
- The state builds with time spent at num_links 1 after an L2 block, near-idle gap rows included, not with write
  traffic. It still clears by L2 it2.

**Reading (hypotheses, untested):**
- Something that builds over time once the link-1 routers go idle after carrying traffic, and clears after two
  iterations with them active again, slows the active routers' per-packet rate. Profiler on; rare with it off.
- The idle routers' own PCs show no extra work. The state may then sit outside the routers' code paths:
  - the eth subsystem of the idle links: link-level state, retrain or FEC activity, the MAC;
  - the profiler's own components on those chips: the eth relay, link-sync, the DRAM relay, which aren't router PCs.
- A read of the idle links' eth status counters (link up / retrain / error counters) over time, in (i) against (ii),
  would test the first. It's a read-only debug-bus or register poll, for the coordinator.

Stopped here for review.

### X8: Ethernet link state and error counters over time, (i) / (ii) / gap (broker 842, results/cisp_ethpoll_20261105-174703)

**Run** (profiler on, read-only): consolidation/eth_poller.py polls all 40 router cores, one sweep every 9.5 ms (about
3 polls per iteration cycle), with 0 read errors. Each poll reads:
- PCS_STATUS and ERR_STAT;
- the three TX queues' CMD / STATUS / TRANSFER / PKT_START / PKT_END / WORD / RESEND counters;
- the base FW debug buffer and boot results: port and training state, heartbeats, the live link snapshot (retrain,
  flap, frames and bytes with bad FCS and dropped, corrected and uncorrected FEC codewords, TXQ resend and RXQ drop
  totals), link-down / recovery / runtime reinit counts;
- the metal mailbox's next-link-check time.

Each polled yaml ran unpolled right after it, then the three yamls ran profiler off. Tool: port_tune/cisp/
cisp_x8_report.py (x8_report.txt).

**Perturbation: none seen.**
- Median of the normal iterations, polled against unpolled: within 0.01% on every row type.
- Slow iterations, polled / unpolled: (i) 38 / 39, (ii) 0 / 0, gap 40 / 38.

**Link health: clean on every core, in every order.**
- PCS_STATUS stays 1, ERR_STAT 0, port UP, training PASS, rx_link_up 1.
- These stay at 0: retrains, CRC errors (bad-FCS frames, TX and RX), dropped frames, uncorrected FEC codewords, TXQ
  resends (the live RESEND_CNT registers and the FW snapshot), RXQ drops, link-down / recovery / runtime reinit counts.
- link_flap_count rises by 1 per core per run. It happens after the last iteration (teardown), in every order.

**Nothing moves differently where the slow state builds.** On the idle link-1 cores during the L1 write rows:
- txq0 and txq1 send 1.347 M one-word packets/s each with no transfers issued (hardware sequence-update / keepalive
  traffic).
- txq2 issues 12.8 k transfers/s (174.7 k packets/s) on every core. Hypothesis: this is the streaming profiler's
  link-sync port, which sends on a TX queue of its own (router_round.hpp end_round); the profiler-off arm wasn't
  polled.
- These medians agree to 4 digits across (i) slow iterations, (i) normal iterations, (ii), and gap slow / normal.
- The link-0 cores' TX rates follow their traffic, with no consistent slow / normal split.

**FEC corrected codewords** (the 1 Hz snapshot): about 6.5 k/s per link on every link.
- Per-core rate, (i) / (ii): median 1.06 on link-1 (range 0.84–1.33), 1.04 on link-0.
- Per-core rate, gap / (ii): median 1.05 on link-1, 0.95 on link-0.
- Snapshot intervals with more slow-state time don't have more corrections.

**Base FW link status checks** (about 1/s per core) fall in each iteration class in proportion to its time.

**Reading:** none of the polled link-level state or counters on link-0 or link-1 cores tracks the slow state. This
rules out, as far as these counters reach:
- link errors and retrains;
- TX resends;
- a change in the idle links' background TX traffic, including the link-sync port's.

Not covered:
- MAC / PCS registers (unsafe to read);
- RX-side queue registers (not documented);
- anything finer than a 9.5 ms poll.

Stopped here for review.

### Worker cores: L1 rows after an L2 block against L1 with no L2 before (offline, 823 zone CSV; cisp_cores_check.py)

**No: the NX worker cores don't change.**
- **Data.** In the 823 zone CSV (stock CI order), each row type has one core set per chip in all of its 24 iterations:
  senders, receivers and the sync core. That includes the run's first L1 write block (iterations 0-2, no L2 before
  it in the process) and the 21 L1 write iterations that follow an L2 block. Output:
  results/cisp_zones_20261105-093338/cores_check.txt.
- **num_links 1, write:**
  - chips 0, 2, 4, 6: senders (15,8), (15,9), (15,10), sync (15,11), receivers (1,2), (1,3), (1,4);
  - chips 1, 3, 5, 7: senders (15,9), (15,10), sync (15,11), receivers (1,2), (1,3).
- **num_links 2, write:**
  - chips 0, 2, 4, 6: senders (15,5)-(15,10), receivers (1,2)-(1,7);
  - chips 1, 3, 5, 7: senders (15,7)-(15,10), receivers (1,2)-(1,5).
- **Code.** The test driver calls reset_devices() after every test iteration. That clears the allocator
  (GlobalAllocator::reset drops all device resources), so each iteration's placement is a function of its own test
  config only.
- **Limits.**
  - L1-only blocks (838 L1, X7 / X8 (ii)) run the same row definitions through the same reset path. Their cores
    follow from the code; no zone CSV was taken there.
  - The zone CSV doesn't name directions. Per-chip sets that are identical in every iteration, plus a placement that
    depends only on the test config, give the same core per (chip, direction).

## (g) Parked: where the slow state stands (2026-11-05)

**Established:**
- **A discrete per-flow state.**
  - An affected router's telemetry cycles sit at +4.6-4.9%; the sender row is at ~0.953 (two discrete levels per
    router).
  - The slowdown is real: host time grows with the cycle excess, and AICLK is unchanged within ~1% (830).
  - It isn't a stable router subset, has no direction or role preference, and isn't chip- or link-wide. The test's
    end sync stretches the senders of the affected chip and its neighbours (per-router tally, 826-830).
- **The trigger is the alternation of num_links 1 and 2 rows (X6, profiler on):** alternation 38/96, L1-only 0/48,
  L2-only 0/48.
- **The timing:**
  - In the stock order it builds over the L1 iterations after an L2 block and is never there at L1 it0.
  - It carries into L2 it0 / it1 and is gone by L2 it2.
  - After a near-idle gap it is already there at L1 it0 (9/16, X7). So it builds with time spent at num_links 1 after
    L2 (the link-1 routers idle after carrying traffic), not with write traffic.
- **The profiler:** it shows with the profiler on in every alternating order. With the profiler off it appears only
  in 830's arm C (one write per row with fused rows between: 14/96).

**Ruled out, as far as each test reaches:**
- **Profiler relay knobs:** the ship threshold, idle-pump stride and deferral (826 / 827). The apparent period was the
  row layout (829).
- **Clock and placement:** an AICLK change (830); kernel placement or load address (824); neighbour start skew (823).
- **Concurrency alone:** a single flow on one link never enters the state, and neither does inbound concurrency (X1,
  X3-X5).
- **Router code:** the router code paths. The PC shares of the idle link-1 and the active routers match in slow,
  normal and L1-only iterations (X7).
- **The Ethernet links:** errors, retrains, resends, drops and FEC. So is the idle links' background TX, including
  txq2 (hypothesis: the profiler's link-sync) (X8).
- **Worker placement:** it is identical with and without a preceding L2 block (this section).

**Remains (hypotheses, untested):**
- **The cause is outside everything polled or sampled so far.** It builds while the link-1 routers sit idle after
  carrying traffic and clears after about two iterations with them active again. Candidates:
  - the profiler's non-router parts on those links (eth relay, DRAM relay, the clock model);
  - NoC / NIU state on the router or receiver tiles;
  - MAC / PCS internals (unsafe to read);
  - RX-queue state (registers undocumented).
- **Why the profiler-off arm shows it only in C.**
- **Next isolating A/Bs if resumed** (each needs a build or new tooling):
  - profiler components off one at a time (link-sync port, eth relay) in the alternation;
  - NIU transaction counters polled on the router and receiver tiles, (i) against (ii);
  - receiver-side Tensix PC samples, (i) against (ii).

Parked here on the coordinator's call.

