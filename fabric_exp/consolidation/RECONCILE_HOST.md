# Host reconciliation: what the study host and the port host pass the router (2026-11-04, offline)

Why: the study's results (the TRID A/B, the slot knees, the completion-group study) were measured with the lead
router v4txa11 on the study host. A rule found there would land in the port, which was checked against main, never
against the study. This file lists every host-side decision that differs between the two, mechanically, from mock
captures of both hosts on the same configs. The router-source diff is RECONCILE_DEVICE.md's (another agent); device
code is out of scope here.

## 1. The two sides

| | STUDY | PORT |
| --- | --- | --- |
| source | /home/aagarwal/tt-metal at e569c5deb20 + its uncommitted host changes (incl. variants/v4txa11/host_per_vc_args.patch), host library = broker 769's rebuild (its router images equal the 649 host's with the new study switches unset, SLOT_ALLOCATOR §12.10's gate) | commit 22cdb1e55c3 (pushed: 7cd3c13a0db; the two differ only in 5 router headers). Built in my worktree /home/aagarwal/tt-metal-port-slots. Its local, never-committed slot patch (fabric_exp/port_slots/slot_override.patch) only adds the slot log and the explicit-row switches; they are unset in these captures. |
| router | v4txa11, swapped into the v2 slot (`fabric_erisc_router_v2`) | the port's router (`fabric_erisc_router`) |
| tree used | /data/aagarwal/frs-glx/trees/v4txa11_1104cg (base_1104cg = the live host + v4txa11) | the worktree above |
| run env (what the perf rows used) | cap_lib.sh `run_cap` / jobs_lib.sh `run`: `TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_EXP_FLAGS=13129088 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_METAL_FABRIC_OPT_LEVEL=O3 TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0`, fast dispatch; the knee / TRID / CG rows add explicit `TT_FABRIC_EXP_SLOTS_*`, `TT_FABRIC_EXP_TRID_ALLOC`, `TT_FABRIC_EXP_COMPLETION_GROUP` | the port agent's .scratch/val/val_lib.sh: `TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0`, fast dispatch; nothing else (the port reads no `TT_FABRIC_*` switch; its opt level is its router-kind default) |

Method: `fabric_exp/consolidation/reconcile/rc_capture.sh study|port` runs each config's first test on the mock
(rc_mock_vet.sh = sls_mock_vet.sh without its built-in study switches; both sides log slot rows and compile commands),
and `rc_diff.py` compares every router build: named compile-time args, positional compile-time args, `-D` defines,
`-O` level, and the slot rows each host logged. Builds are grouped by ERISC and direction; within a group a named arg
"differs" when its set of values over the group's builds differs. 20 configs:

- P150 (fabric_exp/a09u32_cluster_desc.yaml): line, ring, 2x4 mesh, mesh as 4x2 (descriptor), 2x4 XY torus, 4x2 XY
  torus (descriptor), Y torus, NeighborExchange; at 4352 B and 3072 B packet slots where the yaml allows.
- Galaxy mock: VC1 split 2x2+2x2 at 4352 and 3072 B, VC1 split 4x4+4x4, VC2 on at 4352 and 3072 B, 8x4 mesh, 8x4 XY
  torus.
- Not captured: P150x4 (no 4-chip cluster descriptor in the study's tooling); VC1+VC2 together (v4 does not support it
  on ring/torus; the port rejects it there).

Outputs: the full diff `reconcile/rc_diff_all.md` (the captures themselves stay in the session scratchpad, `rc/study`,
`rc/port`; rc_capture.sh re-creates them).

## 2. Per-config decision table

Everything not listed is identical: every named arg both hosts pass has the same value on every router build of all
20 configs, and the positional compile-time arg counts match. What differs:

### 2.1 Slot rows (positional CT args: per-channel slot counts and the buffer addresses that follow)

| config | packet slot | STUDY row (VC0 W, F, R; VC1 F, R; VC2 W, R) | PORT row | origin of the difference |
| --- | --- | --- | --- | --- |
| P150 2x4 mesh, mesh as 4x2, 2x4 / 4x2 / Y torus; Galaxy 8x4 mesh, 8x4 torus | 4352 B | 17, 16, 16 | 17, 16, 16 | none (both: main's tier table + spare slots to the worker) |
| same, P150 | 3072 B | 35, 16, 32 | 35, 16, 32 | none |
| P150 line, ring (VC0 only) | 4352 B | **26, 25, 32** | 16, 16, 32 | study-only: the 1D slot split under `TT_FABRIC_KERNEL_VERSION=2` (allocator `split_1d_slots`, RESULTS §98); the port runs main's tier table (MAINLINE_PLAN §1.4: first push = tier table) |
| P150 line, ring | 3072 B | **43, 42, 32** | 32, 32, 32 | same |
| P150 NeighborExchange (worker channel only) | 4352 B | 32 / 32 | 32 / 32 | none |
| Galaxy VC1 split 2x2+2x2 and 4x4+4x4 | 4352 B | **19, 9, 16; VC1 5, 4** | 51, 4, 8; VC1 2, 4 | study-only: the v4 VC0+VC1 design row under `TT_FABRIC_KERNEL_VERSION=2` (allocator `StudyRow`, memory "fabric slot defaults"); port: tier table |
| Galaxy VC1 split 2x2+2x2 | 3072 B | **53, 9, 16; VC1 5, 4** | 85, 4, 8; VC1 2, 4 | same (the design row applies below 4352 B too; the worker takes the rest) |
| Galaxy VC2 on | 4352 B | **17, 14, 16; VC2 2, 4** | 43, 7, 11; VC2 2, 4 | study-only: the VC0+VC2 design row; port: tier table |
| Galaxy VC2 on | 3072 B | **51, 14, 16; VC2 2, 4** | 77, 7, 11; VC2 2, 4 | same |

The routers' other allocations (five per config, e.g. `{16, 0, 0, 0} R 16`) are the same on both hosts.

### 2.2 Transaction IDs and completion groups

| config class | STUDY passes | PORT passes | same IDs on the router? |
| --- | --- | --- | --- |
| one receiver channel, no deadlock avoidance (mesh, line, NE) | `RX_CH_0_NUM_TRIDS=16`, `_TRID_START=0`, `_COMPLETION_GROUP_SIZE=4` | `NUM_TRANSACTION_IDS=16` | yes: the lead v4txa11 does not read `RX_CH_*` ID/group args (its ct_args reads none of them); it computes 16 / 8 / 4 itself, the rule the port host passes |
| deadlock avoidance (ring, torus) | `RX_CH_0_NUM_TRIDS=8`, group 1 (first-level acks) | `NUM_TRANSACTION_IDS=8` | yes |
| two or more receiver channels (VC1 split, VC2 on) | `RX_CH_0/1_NUM_TRIDS=4`, starts 0 / 4 | `NUM_TRANSACTION_IDS=4` | yes |

The completion group size is not a host arg on either router (both derive it from the slot count on device); the
study host's `RX_CH_*_COMPLETION_GROUP_SIZE` is informational for v4txa11 and consumed only by v4txa11id. The study's
per-channel ID rule (`TT_FABRIC_EXP_TRID_ALLOC`) and the group switch (`TT_FABRIC_EXP_COMPLETION_GROUP`) have no
counterpart on the port.

### 2.3 Bubble flow control injection threshold (ring and torus routers)

| config | STUDY | PORT | origin |
| --- | --- | --- | --- |
| deadlock-avoiding routers with a 32-slot far receiver: P150 ring (4352 and 3072 B), P150 tori at 3072 B | T = 2 (v4txa11's ct_args constant `BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS = 2`; not a host arg) | `BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS=16` | port commit df64d0ad677 (the ring-jam fix, T = max(2, R - 16), user-decided 2026-11-04, RING_JAM_FIX.md §10) |
| deadlock-avoiding routers with a 16-slot far receiver: tori at 4352 B, Galaxy torus | T = 2 | 2 | same arg, same value |

On the study's 1D split rows the ring receiver stays at 32, so a port running the study's ring row would also get
T = 16.

### 2.4 Args only the study passes (unused by the lead)

37 named args, on every config: `EXP_FLAGS=13129088`, `EXP_SEQ_UPDATE_TIMEOUT=1000`, `EXP_TX_DELAY` /
`EXP_RX_DELAY` / `EXP_DRISC_DELAY` / `EXP_STACK_PAINT` = 0, `EXP_PUSH_DRISC_*` = 0, `PHASE_PROFILE_*`
(`ITER_TIMING=1`, `MASK=4095`, the rest 0), `PEER_ERISC_NOC_X/Y`, `PEER_SENDER_CHANNEL_0..9_SERVICED`, and the
`RX_CH_0..2_*` ID / group args. v4txa11's ct_args reads none of them (checked by name), so they change the build's
hash, not its code (a hypothesis until RECONCILE_DEVICE.md's image comparison says the same). Origin: study-only host
code; MAINLINE_PLAN §3.7 does not port these switches.

### 2.5 Defines, opt level, keepalive

- `-DL1_ECC_HALT` on every port build, none on the study's: main's newer code (f0b788af3cb, 2026-10-31, "Halt a
  Blackhole Tensix on an L1 ECC error"; jit_build adds it to every kernel build). Whether it changes ERISC router code is
  a device question (RECONCILE_DEVICE.md); the define names Tensix-only behavior.
- Opt level: -O3 on both, every config (study: `TT_METAL_FABRIC_OPT_LEVEL=O3` in the run env; port: its default, O3
  except VC1 + VC2 together, which takes -Os; not captured).
- `ETH_TXQ_SEQ_UPDATE_TIMEOUT=1000` on both (study: from its env; port: its default).
- `ENABLE_FIRST_LEVEL_ACK_VC*`, `ENABLE_DEADLOCK_AVOIDANCE`, `FUSE_RECEIVER_FLUSH_AND_COMPLETION_PTR`,
  `SWITCH_INTERVAL`, command buffers, NoC ids, stream ids, connection and credit addresses: identical on every build.

## 3. Which study results were measured under a difference

| study result | measured under | transfers to the port by construction? |
| --- | --- | --- |
| SLOT_ALLOCATOR §12.2-12.9 knees, composition, crossover (explicit rows `TT_FABRIC_EXP_SLOTS_*`) | explicit rows (no default-row difference); mesh, line (no deadlock avoidance) and the VC1 split | host side yes: the port slot patch reproduces the switches with the same semantics. Router code differs (RECONCILE_DEVICE.md), so the numbers do not transfer by construction; §12.12 re-measures the forwarding and receiver needs on the port |
| §12.10-12.11 completion-group study | `TT_FABRIC_EXP_COMPLETION_GROUP` on v4txa11id | no: the port has no group override; only the default (G from R) exists there |
| TRID A/B (TRID_ALLOCATION §6.2-6.5): the per-channel rule, the 4x2 N/S +2 cycles, hypothesis C | v4txa11id with `TT_FABRIC_EXP_TRID_ALLOC`; the idoff arm = the uniform rule the port passes | the idoff (uniform) arm's host args equal the port's. The rule itself is not in the port. The +2-cycle finding and the spill in the route decode (§6.5) are study-code results; the port has main's branchy decode, so they do not transfer (re-measure if the rule is ported) |
| census / perf rows at default slots on mesh and tori at 4352 B | 17, 16, 16; IDs 16 / 8; T 2 | host yes (identical args); device differences decide |
| default-slot rows on lines and rings (census 1D, ring rows, half-ring multicast) | the study's 1D split (26, 25, 32 at 4352 B) | no: the port runs 16, 16, 32 |
| ring rows with a 32-slot receiver (rings, 3072 B tori), incl. the TRID A/B ring row and jam rows on v4txa11 | T = 2 | no: the port injects at T = 16 there |
| VC1 rows at default slots (fabric slot defaults 19, 9, 16, 5, 4; the TRID A/B VC1 rows; VC1 knees' context) and VC2-on rows (17, 14, 16 + 2, 4) | the study's design rows | no: the port runs the tier rows (51, 4, 8, 2, 4 / 43, 7, 11, 2, 4) |
| dynamic-slot and receiver-floor A/Bs (`TT_FABRIC_EXP_DYN_SLOTS`, floor) | study-only switches | no counterpart on the port |

Short list:
- **Carry over by construction (host side):** 2D mesh and torus at their default slots (4352 and 3072 B) and
  NeighborExchange: every host arg identical; transaction-ID counts on every config (same uniform rule); keepalive,
  opt level (-O3), first-level acks, deadlock-avoidance flags, NoC and command-buffer assignments everywhere; explicit-row
  experiments, given the port slot patch. Any remaining difference on these rows is device code.
- **Need re-measuring on the port:** anything at default 1D slots (lines, rings), anything at the VC1 or VC2 default
  rows, ring/torus rows with 32-slot receivers (T 2 against 16), the per-channel TRID rule and its N/S loss, the
  completion-group results, the dyn-slot / floor results. The knee numbers carry over only if the device code paths
  they exercise match (§12.12 tests the forwarding and receiver needs directly).
- **Not matchable on the port without new host code:** the study's default 1D / VC1 / VC2 rows (only with the local
  explicit-row patch), T = 2 on 32-slot rings, the per-channel ID rule, the group override; `-DL1_ECC_HALT` (main's).
