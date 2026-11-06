# Receiver tuning sweep on the port: idle passes before a partial credit, completion-proof checks

Started 2026-11-04 (coordinator, approved by the user). Neither value was ever tuned on v4. Causes are hypotheses
unless an arm isolates them; this sweep changes one constant per arm and says nothing about mechanisms.

## 1. The two constants and what they touch

Port 22cdb1e55c3, `router_receiver.hpp`:

- `IDLE_PASSES_BEFORE_PARTIAL_CREDIT = 4` (:63). After this many receiver passes without a packet, a channel credits a
  partial completion group (`slots_due_for_credit`). It was chosen in v3's E1-28 bundle. v3v4 had 8 slightly better
  on 2D (−0.8 to −1.8); 16 lost badly (LEDGER §28). Never isolated, never measured on v4.
- `COMPLETION_PROOF_MAX_CHECKS = 32` (:65). The most status reads a completion group's proof waits for
  (`trid_writes_sent_within`). If none succeeds, the credit is left to a later pass. With against without was
  measured (E1-25; v4 job 079: keep it in the group path, drop it for single slots). The value 32 never was.
- Both are used only where `completion_group_size > 1`. Rings and tori (first-level acks, G 1) don't use them.

## 2. Tree, patch and offline gate

**Tree.** Worktree /home/aagarwal/tt-metal-port-tune (detached at 22cdb1e55c3) plus a local patch, never committed
or pushed: `fabric_exp/port_tune/port_tune.patch` (`stage_port_tree.sh` regenerates it from the worktree). The
patch has three parts.

1. **Host.** `compute_mesh_router_builder.cpp::create_kernel` turns five environment variables into router kernel
   defines, when set: TT_FABRIC_EXP_IDLE_PASSES, TT_FABRIC_EXP_PROOF_CHECKS, TT_FABRIC_EXP_TX_DELAY,
   TT_FABRIC_EXP_RX_DELAY and TT_FABRIC_EXP_DRISC_DELAY become `FABRIC_EXP_*`. It logs `port_tune: router define
   NAME=VALUE` once per process, and every run checks that line.
2. **Kernel.** `router_receiver.hpp` takes the two constants from `FABRIC_EXP_IDLE_PASSES` /
   `FABRIC_EXP_PROOF_CHECKS` (`#ifdef` override; the defaults 4 and 32 otherwise).
   - The delay hooks have the study's v4 semantics (v4txa11d, V4_BOUNDS.md):
     - a spin after each packet sent (`FABRIC_EXP_TX_DELAY`);
     - a spin after each packet received (`_RX_DELAY`);
     - a spin per pass (`_DRISC_DELAY`: low 16 bits each sender pass, high 16 bits each receiver pass).
   - The spin helper (`fabric_experiment_spin`, mcycle) is in `router_utils.hpp`. Every spin is compiled out at 0.
3. **Test infra.** The study's test_infra changes merged three-way onto the port (base 6d8a1032acc; 19 files, no
   conflicts): stream mixes, idle segments (bursty senders), per-segment flush, the receivers-ready start gate and
   the VC2 replay fix.
   - Left out: the study's phase profiler (its types header is gone from the port), the diagnostic launch delay
     and the study's own telemetry switch (the port has 7d442203c01's).
   - Needed because exp_burst / exp_burst612 / p3_mixes use streams and idle segments, which main's test_tt_fabric
     lacks.

**Mock check of the host part** (CPU-only, bh_2d with TT_FABRIC_EXP_IDLE_PASSES=8, _PROOF_CHECKS=128,
_DRISC_DELAY=1048576): all 80 router compiles carry the three `-D` flags, and the log line appears.

**Mock vet of every suite** (`mock_vet.sh`; first test, benchmark off, i1 arm env): val, val1d, 2d, 1d, mix, burst,
burst612, pert, vc1 and vc1i (two ranks), plus the census suites. Each reaches "Launching programs" with the
defines logged and no kernel build error. The first vet missed a test-kernel build error because it checked only the
launch line; see §4, job 176206.

**Offline gate** (`gate.sh`, `gate.txt`). Router-only rebuilds of 18 mocked cases:
- P150: 1D, 2D, ring and torus at one and two ERISCs; neighbour exchange; 2D with VC2 and torus with VC2; mux 1D,
  2D and ring; the VC1 split;
- Galaxy: 2D, ring and torus.

That is 1,328 router images per build, compared by loadable bytes (`cov/cov.py`, a copy of the port agent's tool,
with a `--defines` option that adds an arm's flags to every compile).

| Comparison | Identical | Differ |
| --- | ---: | ---: |
| unpatched port (clean worktree at 22cdb1e55c3) vs patched, env unset | 1,328 | 0 |
| unpatched port vs patched, default arm (`-DFABRIC_EXP_IDLE_PASSES=4 -DFABRIC_EXP_PROOF_CHECKS=32`) | 1,328 | 0 |
| default arm vs each of i1, i2, i8, c8, c128 | 868 | 460 |

**Gate 1 passes:** the default arm is image-identical to the unpatched port.

**Gate 2 (scope) passes.** Every ring and torus image is identical across all five arms:
- bh_ring, bh_ring_1erisc, bh_ring_mux, bh_torus, bh_torus_1erisc, glx_ring, glx_torus.

The 460 changed images are the same per-case set for every arm. They are only receivers:
- the E1 images of the 2-ERISC mesh, line, neighbour-exchange and VC1-split routers (every E0 image is identical);
- every single-ERISC and mux image of mesh and line routers;
- on the torus with VC2, 24 of 40 E1 images. That case is a Y torus, so its X axis is a line. The 24 changed images
  have ENABLE_DEADLOCK_AVOIDANCE 0 and no first-level acks, i.e. the X-axis (line) routers. The 16 unchanged ones
  are the Y-axis torus routers (deadlock avoidance 1, first-level acks on VC0), and their VC2 channel is unchanged
  too.

Every receiver with deadlock avoidance is identical across the arms.

The delay builds (tx16, rp16) change 800 images, as expected, rings included.

## 3. Arms and rows

Runner: `fabric_exp/port_tune/tune_runs.sh SET`. The tree is staged lean at /data/aagarwal/lb2/trees/port_tune_22cd
and copied on the node to /home/aagarwal/tt-metal-port-tune.

- Passes: pass a in table order, pass b reversed, both in one lb2 job (one host).
- Cache: a fresh kernel cache per arm and pass.
- Settings: TT_FABRIC_NO_BW_TELEMETRY=1, fast dispatch, profiler off.
- Validation: a data-checked run per arm in pass a (val: 2D 1-hop E/W and N/S and 3-hop E/W writes, 4352 B × 60 and
  16 B × 2000; val1d: Linear 1-hop and 3-hop).
- Rules: no ring or full-ring rows; a hang or failure stops the job.

**Sets:**

| Set | Arms (IDLE_PASSES, PROOF_CHECKS) | Base |
| --- | --- | --- |
| idle | i1 (1, 32), i2 (2, 32), i4 (4, 32), i8 (8, 32) | i4 |
| proof | c8 (4, 8), c32 (4, 32), c128 (4, 128) | c32 |
| pert | per idle 1, 2, 4, 8: d0, tx8 / tx16 / tx32 (spin after each send), rp8 / rp16 / rp32 (spin per receiver pass) | iN_d0 |

**Suites** (idle and proof; the configs with grouped credit only):

| Suite | Yaml | Rows |
| --- | --- | --- |
| 2d | exp_2d_4352.yaml | 2D 1-hop E/W, N/S, bidir; 3-hop E/W; multicast (write and scatter, 4352 B) |
| 1d | exp_1d_4352.yaml | Linear 1-hop, bidir, 3-hop, multicast, half-ring multicast, neighbour exchange |
| mix | streams/p3_mixes.yaml | E/W, N/S and forwarded: command / size / chunk mixes, pure write 16-4352 B, scatter, fused, atomic |
| burst | exp_burst.yaml | bursts of 1-32 packets with 0 / 500 / 2000-cycle gaps, E/W and N/S, 16 and 4352 B |
| burst612 | exp_burst612.yaml | bursts of 6 and 12 (not multiples of the completion group) |
| vc1 | vc1/vc1_alloc.yaml | the P150 VC1 split (two ranks): all-to-all and inter-mesh neighbour exchange (data checked), VC1 1-hop / 2-hop / merge / multicast / bursts |
| vc1i | vc1/vc1_inter.yaml | inter-mesh VC1 rows |
| pert | port_tune/yamls/pert.yaml | bursts N1 / N4 / N8 / N32 at G500 and N8 at G2000 (E/W), N8 at G500 (N/S), N6 at G500, 1-hop and 3-hop E/W writes |

**Scoring** (`tune_report.py`):
- Rows are cycles/packet at 1.35 GHz.
- A stream mix is one summary row, which is its link total.
- VC1 tests: the slowest sender (census rule), or, where two or more senders share a link, that link's total (the
  2-slot VC1 senders upstream are the hold-sensitive case).
- MixCmd is out of the means (bimodal).
- A row "moved" when both passes moved the same way by more than max(0.5, the base's pass spread).

**Acceptance:**
- A value changes only if the full census shows no suite regressing (generic over narrow).
- Otherwise the current value stays, and its comment gains the measured reason.

## 4. Jobs

| Job | Set | Status |
| --- | --- | --- |
| lb2 176008 | idle | **stopped before the command** on a04u24: 3 baseline links (chip 1 eth (11,1), chip 4 and chip 7 eth (6,1)) stayed down through the reset and both extra resets (SUMMARY rc=1, stage=reset). No device run. |
| lb2 176009 / 176010 | proof / pert | cancelled while queued (their afterok dependency could not be met) |
| lb2 176121-176123 | idle / proof / pert | queued on a07u28 only, then cancelled before they started (a07u28 taken until ~02:15) |
| lb2 176130 | idle | `-n a07u28,a04u32,a04u28` (the harness's approved set, per the coordinator; a04u24 left out while its links are down; a08u28 is below the space floor). **Stopped before the command** on a07u28: chip 6 eth (6,1) stayed down through three resets. No device run. |
| lb2 176131 / 176132 | proof / pert | cancelled while queued (their dependency could not be met) |
| lb2 176205 | dry run | the new link gate's harness (LB2_HARNESS.md "Link gate"), `-n a07u28,a04u32,a04u28,a04u24`, stage src-20261104-234519 |
| lb2 176206 | idle | a04u24. Link gate PASS (first live run). **Stopped at the first data-checked run:** the merged test kernels used the study's `NocFlushType` (FLUSH_RELAXED), which the port's packet header does not have, so the receiver / sender / sync kernels did not compile. The mock vet had missed it: it checked "Launching programs" only, and the build error comes after that line. |
| lb2 176207 / 176208 | proof / pert | cancelled while queued |
| fix | | `flush_or` returns main's `bool flush` (the port's atomic headers); port_tune.patch regenerated, tree restaged. mock_vet.sh now fails on any kernel build error. All 16 suites vetted again, and the test kernels compile with no errors (compile commands logged for val and 1d). |
| lb2 176267 | idle | **done**, a04u24, 16 min, rc 0 (stage src-20261105-001529; collected, manifest results/lb2/176267_a04u24.tsv) |
| lb2 176268 | proof | **done**, a07u28, 14 min, rc 0 (results/lb2/176268_a07u28.tsv) |
| lb2 176269 | pert | **done**, a07u28, 18 min, rc 0 (results/lb2/176269_a07u28.tsv) |

The ports that failed the link check (a04u24: chip 1 (11,1), chips 4 and 7 (6,1); a07u28: chip 6 (6,1)) are not in
either node's ethernet_connections, so no fabric link uses them. This assumes the core-to-channel mapping of
LB2_HARNESS's a07u28 calibration list: (6,1) = channel 8 and (11,1) = channel 9. The harness still refuses any
baseline link that is down.

## 5. Results

Cycles/packet are suite means of pass a / pass b; negative is faster. MixCmd is out of the mix means. Each set ran
in one job on one host, so its arms compare within that job only. The (4, 32) base reads 142.6 on 2D in the idle
job (a04u24) and 143.7 in the proof job (a07u28): the same config differs across hosts, as LB2_HARNESS's calibration
warns for multi-hop rows. Pass spreads are at most 0.1 on every suite mean, and at most 0.3 on rows except VC1
inter-mesh (≤1.3). Every run passed: data-checked val / val1d per arm, VC1 all-to-all and neighbour exchange on both
ranks.

The port runs main's slot table. VC1 senders on the split get main's tier rows, which include 2-slot forwarding
channels (the hold-sensitive case). The VC1 numbers below are for those slots; the allocator will change them.

### 5.1 Idle passes before a partial credit (lb2 176267, a04u24)

| Suite (rows) | i1 | i2 | i4 (base) | i8 |
| --- | ---: | ---: | ---: | ---: |
| 2d (10) | 143.74 / 143.77 (+1.1) | 143.15 / 143.15 (+0.5) | 142.63 / 142.63 | 142.52 / 142.53 (−0.1) |
| 1d (16) | 154.67 / 154.67 (+0.1) | 154.64 / 154.65 (0.0) | 154.63 / 154.60 | 154.56 / 154.55 (−0.1) |
| mix (30) | 128.12 / 128.13 (+0.4) | 127.81 / 127.80 (+0.1) | 127.74 / 127.75 | 127.69 / 127.70 (−0.1) |
| burst (44) | 523.28 / 523.27 (+0.2) | 523.12 / 523.12 (0.0) | 523.11 / 523.12 | 523.15 / 523.16 (0.0) |
| burst612 (16) | 226.46 / 226.49 (−0.2) | 226.62 / 226.61 (−0.1) | 226.71 / 226.71 | 226.63 / 226.63 (−0.1) |
| VC1 split (13) | 222.29 / 222.29 (−1.8) | 223.17 / 223.17 (−0.9) | 224.11 / 224.09 | 224.44 / 224.43 (+0.3) |
| VC1 inter-mesh (8) | 693.40 / 693.41 (−7.0) | 695.75 / 695.78 (−4.7) | 700.42 / 700.45 | 694.92 / 694.82 (−5.6) |

Rows that moved in both passes beyond max(0.5, the base's spread):
- **Fewer idle passes (1, 2).**
  - 2D 3-hop E/W write costs +7.2 / +3.3 and 2D multicast write +2.5 / +1.4.
  - Small E/W writes cost +2.3 at i1.
  - The VC1 1-hop and burst rows cost +3.4 to +4.5 at i1.
  - VC1 2-hop gains −26.3 / −12.1 (4352 B) and −22.7 / −6.6 (16 B).
  - VC1 inter-mesh 0_0 → 1_1 gains −27.7 / −16.5 (4352 B).
  - Bursts of 8 at 16 B gain −3.1 to −3.8 at i1; bursts of 8 and 32 at 4352 B cost +2.7.
- **More idle passes (8).** 2D 3-hop gains −0.8. VC1 2-hop costs +1.5 to +2.8. VC1 inter-mesh 1_2 at 16 B gains
  −16.1, and the inter-mesh mean −5.6.
  - That row is not monotonic in the idle count: i4 is the slowest of the four arms there. This hints at a sampled
    effect (a timing cliff). It is a hypothesis; the perturbation pass (§5.3) is the test for it.

**Reading.**
- No value is at least as fast as 4 on every suite:
  - i1 and i2 regress 2D (Mesh3HopEW and multicast);
  - i8 regresses the VC1 split.
- So 4 stays (acceptance rule).
- What each arm isolates is the idle count alone. Why VC1 2-hop gains so much from crediting partial groups sooner is
  a hypothesis: the 2-slot upstream VC1 forwarding window waits on a partial group's hold. It would need a test at
  other slot rows (the allocator's) before any idle-count rule per window is designed.

### 5.2 Completion-proof checks (lb2 176268, a07u28)

| Suite (rows) | c8 | c32 (base) | c128 |
| --- | ---: | ---: | ---: |
| 2d (10) | 143.37 / 143.40 (−0.3) | 143.74 / 143.71 | 143.67 / 143.64 (−0.1) |
| 1d (16) | 155.09 / 155.08 (−0.2) | 155.27 / 155.29 | 155.37 / 155.40 (+0.1) |
| mix (30) | 127.45 / 127.46 (−0.3) | 127.73 / 127.74 | 127.37 / 127.38 (−0.4) |
| burst (44) | 523.03 / 523.05 (−0.1) | 523.10 / 523.11 | 523.10 / 523.10 (0.0) |
| burst612 (16) | 226.66 / 226.66 (0.0) | 226.68 / 226.68 | 226.69 / 226.69 (0.0) |
| VC1 split (13) | 222.78 / 222.66 (−0.7) | 223.41 / 223.37 | 223.44 / 223.37 (0.0) |
| VC1 inter-mesh (8) | 698.84 / 698.89 (0.0) | 698.88 / 698.81 | 698.80 / 698.93 (0.0) |

Rows that moved:
- **c8:** 2D multicast scatter −1.6; 1D unidirectional multicast scatter −1.4; VC1 2-hop −1.0 to −2.1. VC1
  inter-mesh rows move both ways (−5.7 to +4.1 at 16 B).
- **c128:** forwarded fused / atomic / scatter −2.1 to −2.3, E/W fused −1.0; Linear 1-hop E/W and N/S scatter +0.8.

**Reading, against the coordinator's key.**
- 8 is not slower, so a group's proof rarely needs more than 8 reads.
- 128 differs on forwarded fused, atomic and scatter rows (−2.2), so the cap of 32 does bind there sometimes.
  **WRONG** (2026-11-05, a09u32 job 800, section 7): counted directly on 6dad5fec638, every proof on those rows
  passed on its first status read, on both receivers. There were 0 cap hits in about 48,000 proofs per row and pass.
  The cap never binds there, so the −2.2 is not proofs reaching the cap.
- Also not reported above: FWD_MixCmd under c128 ran 171.0 in pass a and 210.4 in pass b (c32 173.5 / 173.5), a
  one-pass outlier of +37 cycles/packet. It was left out of the "rows that moved" list.
- Both effects are under 1 cycle on every suite mean, which is the band the user does not chase (1-2 cycles).
- c8 is at or below the base on every suite mean. A change to 8 would need the full census
  (`tune_runs.sh census`, TUNE_CAND=4,8) and its own perturbation pass: the bounded wait was the fix for a timing
  cliff (LEDGER E1-24 / E1-25), and a smaller cap gives that back in part. Recommendation: keep 32 unless the user
  wants that census.

### 5.3 Delay perturbation on the idle-pass arms (lb2 176269, a07u28)

**Setup.**
- Each idle arm (1, 2, 4, 8; proof 32) with no delay, a spin after each packet sent (tx 8 / 16 / 32), or a spin per
  receiver pass (rp 8 / 16 / 32), on the pert rows; both passes.
- Every run passed. The largest pass spread over all 56 arm-passes is 0.6.
- **tx8 and tx16 measure the same** on every row (to 0.3): the spin has a floor of about 14 cycles, as in the
  study's v3 sweep (RESULTS §98, perturbation B). So the sender-side sweep has two effective points, ~16 and 32.

**Response to the delay** (row at the delay minus the same arm's no-delay row; mean of passes; i1 / i2 / i4 / i8):

| Row | tx16 | tx32 | rp16 | rp32 |
| --- | --- | --- | --- | --- |
| Mesh 3-hop E/W write 4352 B | +31.6 / +30.5 / +26.0 / +20.9 | +53.3 / +52.8 / +48.4 / +40.5 | +10.9 / +11.4 / +12.1 / +13.5 | +19.5 / +20.7 / +21.8 / +23.3 |
| Mesh 1-hop E/W write 4352 B | +17.8 / +15.0 / +11.8 / +11.5 | +39.4 / +37.6 / +33.7 / +30.5 | +7.3 / +7.3 / +7.6 / +7.5 | +18.5 / +18.5 / +18.8 / +18.8 |
| Burst E/W N8 G500, 16 B | +19.4 / +14.5 / +8.9 / +6.5 | +42.4 / +36.2 / +28.6 / +23.4 | +12.0 / +12.0 / +10.9 / +9.3 | +19.6 / +18.2 / +15.7 / +13.9 |
| Burst E/W N32 G500, 16 B | +23.9 / +19.4 / +15.6 / +14.0 | +46.9 / +41.3 / +36.4 / +32.9 | +13.7 / +13.9 / +14.0 / +13.6 | +20.8 / +20.5 / +20.7 / +20.7 |
| Burst N/S N8 G500, 16 B | +13.6 / +8.3 / +6.5 / +5.2 | +36.8 / +29.2 / +23.8 / +20.7 | +16.4 / +13.2 / +11.7 / +10.9 | +24.9 / +22.7 / +21.3 / +20.5 |
| Burst E/W N6 G500, 4352 B | +1.0 / +0.8 / +0.8 / +0.2 | +16.5 / +13.7 / +8.0 / +5.0 | +4.3 / +3.2 / −0.4 / −0.7 | +6.9 / +2.4 / −0.3 / 0.0 |
| Bursts N1 / N4 at G500, N8 at G2000 | ≤ 0.6 everywhere | | | |

**Readings.**
- **No steps.** Every arm's response grows with the delay at every resolved point (tx: 0, ~16, 32; rp: 0, 8, 16, 32;
  per-row tables in tune_report.py pert). Across the idle arms the responses are ordered (i1 ≥ i2 ≥ i4 ≥ i8 under tx),
  not erratic.
- **Fewer idle passes amplify sender-side timing.** With the sender slowed by a fixed spin, i1 and i2 lose much more
  than i4 and i8: 3-hop E/W +31.6 against +20.9 at tx16, bursts of 8 at 16 B +19.4 against +6.5. i8 is the least
  sensitive; i4 is close to it (within 2-8 cycles).
  - That fewer idle passes then credit partial groups mid-burst, more often and so more expensively, is a hypothesis
    (not isolated).
- **Under a receiver-pass spin the idle arms respond alike.** They are within ~4 cycles on most rows; the exception
  is bursts of 6 at 4352 B, where i1 loses +4 to +7 while i4 / i8 do not move.
- **For the decision.** The perturbation argues against 1 and 2, on top of their 2D regression (§5.1), and is
  neutral to slightly favourable for 8 against 4. The choice between 4 and 8 waits for the allocator check (§6).

### 5.4 Proposed comment text (for the port agent; this sweep edits no router source)

- `COMPLETION_PROOF_MAX_CHECKS = 32`: "Measured against 8 and 128 on Blackhole (P150): every suite mean within 0.7
  cycles/packet. 128 gains ~2 on forwarded fused / atomic / scatter rows; 8 is never slower but gives back most of
  the bounded wait, which removed a timing cliff (the completion check landing just before the group's last write
  left)."
- `IDLE_PASSES_BEFORE_PARTIAL_CREDIT = 4`: pending the allocator check (§6). The measured reason so far: 1 or 2 cost
  2D 3-hop up to +7 and amplify sender-side timing; 8 is level with 4 except on VC1 at the tier table's 2-slot
  forwarding windows.

## 6. Follow-up on the allocator's slot rows (submitted 2026-11-05: lb2 176425)

Coordinator (2026-11-05):
- The proof cap stays at 32; no c8 census unless the user asks.
- The decision on idle passes waits for a check at the staged allocator's slot rows. Those give VC1 forwarding 7
  slots at 4352 B, not tier's 2, so the VC1 2-hop and inter-mesh gains from fewer idle passes may not survive there.
- The allocator, local 4b524653490 in /home/aagarwal/tt-metal-port-alloc, waits on the user's go for its own device
  A/B. This rerun goes after it.

**Steps:**
1. **Tree.** A new worktree /home/aagarwal/tt-metal-port-tune-alloc at the allocator commit the A/B ran (4b524653490
   unless it moves), plus port_tune.patch.
   - Apply with `git apply -3`. Since 22cdb1e55c3 the cleanup commits changed compute_mesh_router_builder.cpp and
     router_receiver.hpp, which the patch touches, so expect to resolve hunks by hand.
   - Submodules, sfpi and the CPM cache are copied as for the first tree. Build test_tt_fabric.
2. **Offline gate** on that tree:
   - mock captures (cov.py with TT_COV_TREE pointing at it): bh_2d, bh_1d, bh_vc1, bh_ring, bh_torus;
   - the default arm (4, 32) image-identical to the unpatched allocator commit;
   - ring and torus images identical across i2, i4 and i8;
   - the slot rows checked in the host's debug log: the VC1 forwarding channels at 7 on the split at 4352 B.
3. **Mock vet** (`mock_vet.sh`, PT pointing at the new tree), including the kernel build check.
4. **Stage** at /data/aagarwal/lb2/trees/port_tune_alloc_4b52 (`stage_port_tree.sh` with W / T for that tree).
5. **One lb2 job, both passes on one host:**

   ```
   PORT_TREE=/home/aagarwal/tt-metal-port-tune-alloc \
   PORT_TREE_SRC=/data/aagarwal/lb2/trees/port_tune_alloc_4b52 \
   bash fabric_exp/port_tune/tune_runs.sh idlealloc
   ```

   - Arms i2 / i4 / i8 at proof 32, on suites 2d (Mesh 3-hop E/W, multicast and the 1-hop controls), vc1 (the split)
     and vc1i.
   - About 6 arm-passes of about 1 minute each, so under 15 minutes on the device.
   - Node list a07u28, a04u32, a04u28, a04u24.
6. **Decision.**
   - i4 stays unless another value is at least as fast on all three suites at the allocator's rows.
   - If one is, the full census (`TUNE_CAND=<i>,32 tune_runs.sh census`) plus a perturbation pass decide (acceptance
     rule).
   - Either way, the comment on IDLE_PASSES_BEFORE_PARTIAL_CREDIT gets the measured reason; the port agent applies
     it.

**Done so far (2026-11-05; the user approved the allocator's device A/B, and the coordinator said go):**
- **Tree.** /home/aagarwal/tt-metal-port-tune-alloc = 4b524653490 + port_tune.patch. It applied cleanly with -3
  (no hand edits); built.
- **Gate** (fabric_exp/port_tune/gate_alloc.txt; mock captures of bh_2d, bh_1d, bh_vc1, bh_ring, bh_torus):
  - unpatched 4b524653490 vs the default arm: 416 / 416 identical;
  - i2 and i8 change only the 1D, 2D and VC1-split receivers (124 images);
  - ring and torus identical.
- **Slot rows** (probe: each router's slot arrays forced into a compile error from its logged command):
  - all 45 VC1-split E0 images have sender slots 14, 12, 11, 11, 7, 7, 7 (VC0 worker, VC0 forwarding ×3, VC1
    forwarding ×3) and receivers 8, 4;
  - the port's tier table gives the same routers 51, 4, 4, 4, 2, 2, 2 / 8, 4 (the 22cdb1e55c3 captures).
  - The host's slot log is log_debug, which a Release build compiles out, hence the probe.
- **Mock vet** (`PORT_TREE=... mock_vet.sh`): 2d, vc1, vc1i, val and val1d OK, including the kernel build check.
- **Staged** at /data/aagarwal/lb2/trees/port_tune_alloc_4b52; stage_port_tree.sh now takes PORT_TREE /
  PORT_TREE_SRC / PORT_BASE and checks that the tree carries the same patch.
- **Submitted:** lb2 176425 (`-n any`, stage src-20261105-013146; labels `ptaa_` / `ptab_`; report:
  `tune_report.py idlealloc --prefix pta`).
- **lb2 176425 hung at its first data-checked run** (a04u24, 01:35): ptaa_idlealloc_i2_val (ValMesh1HopEW 4352 B,
  60 packets, idle 2). The capture was decoded offline against router images rebuilt by a mock of the same yaml and
  arm (results/ptaa_idlealloc_i2_val_hangcap/decode_offline.txt). The image-to-config match is approximate: several
  cores matched configs of another channel.
  - Every router is drained, with nothing unsent and nothing pending.
  - The test's data flow completed: 60 sent = validated.
  - The sync kernels on chips 2, 3, 6 and 7 wait at the end global sync, 1 to 3 sync packets missing (semaphore 76
    or 74 of 77).
  - The decoder also flags forwarding channel 1 on eth5 / eth7 of chips 0, 2, 4 and 6 at 16 against an expected 15
    ("over-credit"). Given the approximate config match, that is not established.
- **The allocator agent's own A/B hung the same way** before any perf row, on its first data-checked run (lb2 176405,
  a07u28, `paa_staged_pval2d_4352`). That arm is the staged allocator without this patch, at default idle passes.
- So the hang is not the idle-pass arm. On the tier table (§5) every arm, i1 and i2 included, passed the same val
  yaml. Hang rule: the follow-up stops here, with no reruns. It waits for the allocator's own diagnosis and fix.
  Then it reruns unchanged (the tree is staged; `tune_runs.sh idlealloc` with the same PORT_TREE /
  PORT_TREE_SRC after a restage).

**Prep for the rerun on the fixed allocator (2026-11-05, offline only; not restaged, not submitted):**
- **Trees.** /home/aagarwal/tt-metal-port-tune-alloc (+ port_tune.patch, same diff) and -alloc-base are both moved to
  c53b5a5cbbd, the fixed staged allocator (local, unpushed). The host is rebuilt.
- **Gate.** fabric_exp/port_tune/cov with COV /home/aagarwal/.tt-metal-port-tune-alloc-cov; the 4b524653490 captures
  are kept in old_4b52/. The five cases (bh_2d, bh_1d, bh_vc1, bh_ring, bh_torus) were recaptured on the c53b host:
  - unpatched c53b vs the patched tree, no defines: 416 / 416 identical;
  - unpatched vs the default arm (idle 4, proof 32): 416 / 416 identical;
  - i2 and i8 change only the 1D, 2D and VC1-split receivers (124 images); ring and torus are identical.
- **Slot rows** (fabric_exp/port_tune/slot_probe.py, saved as slot_probe_alloc_c53b.txt), at 4352 B:

| config | sender slots | receiver slots |
| --- | --- | --- |
| VC1 split | 15, 11, 11, 11 + 7, 7, 7 (VC1 forwarding 7) | 8, 4 |
| 2D | 20, 15, 15, 15 | 16 |
| torus | 20, 15, 15, 15 | 16 |
| 1D | 24, 27 | 32 |
| ring | 24, 27 | 32 |

- **Mock vet.** `PORT_TREE=... mock_vet.sh 2d vc1 vc1i val val1d`, i2 arm: all OK, including the kernel build check
  and the logged define.
- **Waiting.** The user is deciding two allocator tweaks for the P150 misses, which would change the 2D rows:
  (a) the 2D worker full-rate depth 20 -> 18, (b) receivers first in the leftover order. Once the final commit
  exists: move both trees onto it, rerun the gate and the slot probe, restage (stage_port_tree.sh with
  PORT_TREE / PORT_TREE_SRC / PORT_BASE), then submit `tune_runs.sh idlealloc` with PT_PREFIX=pta.
- **Rebased onto the final allocator, 9ce0acbeaba** (on e5b75d1fbcd; the coordinator's go, 2026-11-05):
  - both trees moved, host rebuilt;
  - recaptured and gated: none = base and d4_32 = base, 416 / 416 each; i2 / i8 change only the 1D, 2D and VC1-split
    receivers (124); ring and torus identical;
  - slot probe (port_tune/slot_probe_alloc_9ce0.txt), at 4352 B, all as the coordinator expected:
    - 2D and torus: 17, 16, 16, 16, receiver 16 (the tier row);
    - 1D and ring: 26, 25, receiver 32;
    - VC1 split: 15, 11, 11, 11 + 7, 7, 7, receivers 8, 4 (VC1 forwarding 7);
    - the 3072 B rows were not probed: the recheck's suites all run at 4352 B;
  - mock vet with the i2 arm: OK;
  - staged at /data/aagarwal/lb2/trees/port_tune_alloc_9ce0 (762 MB);
  - lb2 dry 177103 OK on a07u28;
  - **submitted lb2 177108** (-n a07u28,a08u28,a04u32,a04u28; labels ptaa_ / ptab_; report:
    `tune_report.py idlealloc --prefix pta`).
- **Result, lb2 177108** (a07u28, 06:04-06:11; all 23 runs passed, data-checked val and val1d included; report
  fabric_exp/port_tune/report_idlealloc_177108.md). Suite means in cycles/packet, pass a / pass b, against i4:

| suite (rows) | i2 | i4 (base) | i8 |
| --- | ---: | ---: | ---: |
| 2d (10) | 143.96 / 143.98 (+0.27 / +0.26) | 143.70 / 143.71 | 143.48 / 143.53 (−0.21 / −0.19) |
| vc1 split (13) | 183.70 / 183.72 (+0.78 / +0.83) | 182.92 / 182.89 | 182.82 / 182.84 (−0.09 / −0.06) |
| vc1i (8) | 323.02 / 323.04 (+0.03 / −0.03) | 322.99 / 323.07 | 323.01 / 323.01 (+0.02 / −0.06) |

  - **Rows that moved in both passes, same sign:**
    - i2: Mesh3HopEW write / scatter +0.8 to +1.1; VC1 2-hop merge +1.9 to +2.4; VC1 16 B bursts and 1-hop
      +0.5 to +1.4; VC1 multicast +0.8.
    - i8: Mesh3HopEW write −1.4 / −1.3.
  - **Against the tier-row sweep (5.1):** the VC1 inter-mesh sensitivity is gone on the allocator's rows. It was
    −4.7 to −7.0 at i1 / i2 / i8; now every arm is within ±0.06. i2 is now slower than i4 on 2D and VC1.
  - **Decision rule (step 6):** i8 is at least as fast as i4 on all three suites. The margins are 0.2 cycles or less,
    and vc1i is within ±0.06, under the 1-2 cycle band the user does not chase. By the rule, i8 qualifies for the full
    census (`TUNE_CAND=8,32 tune_runs.sh census`) plus a perturbation pass. 5.3's perturbation favoured more idle passes
    (bursts of 8 at 16 B: i1 +19.4, i8 +6.5 at tx16). The decision is the user's.
  - **Decision (user, 2026-11-05): IDLE_PASSES_BEFORE_PARTIAL_CREDIT stays 4; no census for 8.** Per lb2 177108:
    i8 is within 0.2 cycles of i4, i2 is slower on 2D and VC1, and VC1 inter-mesh is flat on the allocator's rows.
    The pushed comment (57c55f98118) already carries the measured reason, so there is no code change. The proof cap
    stays 32 (5.2, 7). **The receiver tuning work is closed.**
- **Expected probe rows after the tweaks** (coordinator, 2026-11-05):
  - 2D at 4352 B with (a): senders 17, 16, 16, 16, receiver 16, the tier row. The depth of 18 still does not fit,
    since 18 + 48 + 16 = 82 > 81.
  - 2D at 3072 B with (b): senders 20, 21, 21, 21, receiver 32.
  - VC1 forwarding stays 7.
  - Check these against the allocator agent's gate table before restaging.

## 7. Completion-proof counts (a09u32 broker job 800, 2026-11-05; diagnostic, cycles not scored)

**Build.**
- 6dad5fec638 with LOCAL counters in router_receiver.hpp, in runtime root
  /home/aagarwal/tt-metal-port-gap-roots/proof_count, uncommitted. Host and test infra are
  /home/aagarwal/tt-metal-port-testinfra: 6dad5fec638 + test_infra_6dad.patch, the same harness as 176268, with the
  stream rows and the stream-loop sender.
- **Counters,** per receiver channel, in core-local .bss: group proofs, status reads, a histogram of reads at success,
  cap hits, G 1 checks / passes / deferrals, and slots credited. They are written into the first VC0 receiver slot at
  the end of the receiver loop, then read back with hang_capture.py and proof_counts.py.
- **Perturbation, static:**
  - a group proof passing on read 1 has +18 instructions (7 -> 22, counting the slots-credited add);
  - each extra read +0;
  - larger buckets up to +7;
  - a cap hit +6;
  - a G 1 check +6 (deferred) / +9 (passed).
- **Images (mock):** E0 and the test kernels are identical to the plain worktree's.
  - E1 grows by +380 to +440 B: the counter code plus a 33-store zeroing and register-allocation moves.
- **Scripts:** fabric_exp/port_tune/proof/ (proof_runs.sh, proof_counts.py, proof_report.py, yamls/).
- **Results:** fabric_exp/results/proof_20261105-033445/report.md. 11 rows x 2 passes, all passing, with no stale
  records. Passes agree to the second digit.

**Counts** (receivers that credited >= 1,000 slots; "a <- b": a receiver on chip a.eth, from chip b):

| row | receiver | G | proofs (a / b) | reads / proof | % at 1 / 2-4 / 5-8 / 9-32 read(s) | cap hits |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 1-hop E/W write 16 B | 0.6 (delivers) | 4 | 13,017 / 13,034 | 1.00 | 100 / 0 / 0 / 0 | 0 |
| 1-hop E/W write 4352 B | 0.6 | 4 | 12,596 / 12,612 | 4.15 | 20 / 4.5 / 75.3 / 0 | 0 |
| 1-hop E/W scatter 4352 B | 0.6 | 4 | 12,520 / 12,519 | 1.00 | 100 / 0 / 0 / 0 | 0 |
| 1-hop E/W atomic | 0.6 | 4 | 12,519 / 12,521 | 1.00 | 100 / 0 / 0 / 0 | 0 |
| E/W fused 4352 B (stream) | 0.6 | 4 | 12,020 / 12,019 | 1.00 | 100 / 0 / 0 / 0 | 0 |
| FWD atomic | 0.6 (forwards) / 2.4 (delivers 2 flows) | 4 | 23.7k / 24.0k per pass (forwarder / deliverer) | 1.00 / 1.00 | 100 / 0 / 0 / 0 | 0 |
| FWD fused 4352 B | same | 4 | 24.0k / 24.0k per pass | 1.00 / 1.00 | 100 / 0 / 0 / 0 | 0 |
| FWD scatter 4352 B | same | 4 | 23.9k / 24.0k per pass | 1.00 / 1.00 | 100 / 0 / 0 / 0 | 0 |
| FWD write 4352 B | same | 4 | 23.7k / 24.0k per pass | 1.00 / 2.99 | forwarder 100 / 0 / 0 / 0; deliverer 50 / 1.4 / 48.6 / 0 | 0 |
| 2D 3-hop write 4352 B | 2 forwarders / deliverer | 4 | 13.3-14.2k each, per pass | 1.00, 1.04 / 3.54 | deliverer 34.4 / 8.7 / 57 / 0 | 0 |

- **Totals:** 598,528 group proofs, 0 cap hits, none needing more than 8 reads.
- **Ring (half-ring multicast write 4352 B; first-level acks, so G 1, checked once):** 6.19 M checks, 587,557
  deferred.
  - Receivers that credit 200k slots: 0.0-0.1% deferred.
  - Receivers that credit 150k slots: 19.4-19.8% deferred, in both passes.

**Reading against section 5.2 (counts only; no causal claim):**
- **The cap is never reached on these rows.** No proof needs more than 8 reads, so a cap of 8 or of 128 would not
  change any proof's outcome on them. That fits c8 not being slower.
- **Where proofs take more than one read.** Only the deliverer of 4352 B writes needs more than one read, mostly 5-8:
  the 1-hop write, the forwarded write's deliverer and the 3-hop deliverer. Atomic, scatter and fused proofs, and
  every forwarder's proofs, pass on the first read.
- **c128's −2.1 to −2.3 on forwarded fused / atomic / scatter is not explained by proofs.** On those rows the proof
  loop runs once per proof, so its cap cannot matter. The 5.2 inference is marked WRONG.
- **What else c128 changes, a hypothesis:** the loop's bound, and with it the receiver image (code size, placement).
  Not isolated. Note that 176268 ran 22cdb1e55c3 on a07u28, and these counts are from 6dad5fec638 on a09u32.

## 8. Idle-pass sweep 4 / 8 / 16 / 32 / 64 on LoudBox and Galaxy (coordinator, 2026-11-05)

Requested after the 200G NX A/B (Slurm 180134, TODO): IDLE_PASSES_BEFORE_PARTIAL_CREDIT per arm, every other constant
as shipped. No recommendation here; the user decides.

**Build.** Tree tune_e7fe: e7fe1899049 + the telemetry switch + the port_tune host defines; router_receiver.hpp takes
`FABRIC_EXP_IDLE_PASSES` (fabric_exp/port_tune/port_tune_e7fe.patch). Offline gate (port_tune/gate_idlesw.txt): the
default arm is byte-identical to e7fe1899049 (1448 builds; with the define set to 4, 1442); i8 changes 517 of 1442
router images; rings and tori are identical except 24 of bh_torus_vc2's. Arms
i4 (shipped), i8, i16, i32, i64; two passes each, interleaved; TT_FABRIC_NO_BW_TELEMETRY=1.

**Jobs.**
- LoudBox 400G: lb2 180292 on a04u24. Suites 2d, 1d, burst, mix, vc1, nx4096, ring (control), lat (latency, ns).
  Full table: fabric_exp/results/idlesw_lb_180292.txt.
- Galaxy 200G: Slurm 180534 on a08u02. Suites nx (NeighborExchange 3072 / 4096 / 4352 B, 1 link), mesh, linear,
  ring. Full table: fabric_exp/results/idlesw_glx_180534.txt.
- Each table gives, per row and arm, the mean of the two passes, the pass spread, and the change against i4.

**Regression rule** (fabric_exp/port_tune/idlesw_report.py): more than 1 cycle (latency: 0.74 ns) or more than 0.5%,
with both passes beyond i4's spread.

### 8.1 Galaxy 200G: the NX gap

cycles/packet, mean of passes; main is 179756's 1-link rows (another job; 180134's main reference, same values).

| row | main | i4 | i8 | i16 | i32 | i64 |
|---|---|---|---|---|---|---|
| NX write 3072 | 177.37 | 178.54 (+1.17) | 177.45 (+0.08) | 177.42 (+0.05) | 177.45 (+0.08) | 177.47 (+0.10) |
| NX write 4096 | 235.72 | 239.28 (+3.56) | 236.68 (+0.96) | 235.75 (+0.03) | 235.75 (+0.03) | 235.74 (+0.02) |
| NX write 4352 | 249.55 | 253.12 (+3.57) | 250.73 (+1.18) | 250.10 (+0.55) | 250.10 (+0.55) | 250.11 (+0.56) |

- **Smallest value closing the gap:** 3072 B closes at 8; 4096 B at 16. 4352 B keeps +0.55 at every value from 16 up,
  as 180134's d64 arm did (+0.54).
- So 16 is the smallest value closing the gap as far as this constant closes it; 32 and 64 add nothing on these rows.
- Pass spreads are 0.00-0.10.
- Other Galaxy rows: within 0.3 cycles of i4 at every value. These are mesh 1-hop and 3-hop, linear 3-hop, ring 1-hop
  and full-ring multicast.
- **Galaxy regressions: none at any value.**

### 8.2 LoudBox 400G

Suite means of the change against i4, cycles/packet (lat: ns):

| suite (rows) | i8 | i16 | i32 | i64 |
|---|---|---|---|---|
| 2d (10) | −0.11 | −0.17 | −0.13 | −0.15 |
| 1d (16) | +0.02 | +0.03 | +0.03 | +0.01 |
| burst (44) | +0.04 | +0.19 | +0.42 | +0.49 |
| mix (33) | +1.15 | −0.10 | +0.47 | −0.03 |
| vc1 (13) | −0.07 | −0.08 | −0.09 | −0.07 |
| nx4096 (1) | +0.00 | +0.01 | +0.01 | +0.00 |
| ring control (14) | −0.01 | −0.02 | −0.01 | −0.03 |
| lat (12, ns) | +2.59 | +1.30 | +1.26 | +6.14 |

- mix's i8 and i32 means come from one bimodal row, FWD_MixCmd 3264: +38.9 at i8, pass spread 0.48; +19.4 at i32,
  spread 39.0.
- At 400G the NX row is flat at every value: 117.90.

**LoudBox regressions** (rule above):

- **i8 (7 rows):**
  - BurstEW_N8_G500 write 16: +1.56
  - BurstNS_N8_G500 write 16: +1.63
  - FWD_MixCmd 3264: +38.88 (bimodal)
  - latency, 1-hop 4096 B: unicast EW +9.63 ns (+0.57%); fused NS +10.37 (+0.61%); unicast NS +10.00 (+0.59%)
  - latency, 3-hop unicast 4096 B: +3.88 ns (+0.09%)
- **i16 (7 rows):**
  - BurstEW_N8_G500 write 16: +4.44 (+3.3%)
  - BurstNS_N8_G500 write 16: +4.37 (+3.3%)
  - five latency rows, +1.29 to +6.48 ns (≤ 0.34%)
- **i32 (9 rows):**
  - BurstEW_N8_G500 write 16: +5.21
  - BurstNS_N8_G500 write 16: +5.95
  - BurstEW_N32_G2000 write 16: +1.31
  - BurstEW_N8_G2000 write 16: +1.74
  - BurstNS_N8_G2000 write 4352: +1.17
  - latency, 1-hop fused 4096 B: EW +13.34 ns (+0.78%); NS +15.55 (+0.92%)
  - two more latency rows, +3.88 ns each
- **i64 (13 rows):**
  - BurstEW_N8_G500 write 16: +5.16
  - BurstNS_N8_G500 write 16: +6.01 (+4.5%)
  - BurstEW_N32_G2000 write 16: +1.99
  - BurstNS_N32_G2000 write 16: +2.27
  - BurstEW_N8_G2000 write 16: +1.73
  - BurstNS_N8_G2000 write 4352: +1.17
  - latency, 1-hop fused 16 B: EW +8.70 ns; NS +8.14
  - latency, 1-hop fused 4096 B: EW +16.67 ns; NS +18.14 (+1.07%)
  - latency, 1-hop unicast 4096 B: EW +9.63 ns; NS +8.15
  - latency, 3-hop unicast 4096 B: +3.88 ns

The latency rows' i4 spreads are up to 4.4 ns. A 1-cycle (0.74 ns) threshold flags them readily.

Any mechanism for these rows is a hypothesis: no arm here isolates one. An example is a partial group held longer
before credit on bursty traffic.
