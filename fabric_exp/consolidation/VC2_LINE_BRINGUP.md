# VC2 line mode bring-up log (worktree /home/aagarwal/tt-metal-vc2line, branch aagarwal/vc2-line)

Design and decisions: VC2_LINE_IDEA.md. Base: `v4rxv2i3` (the lead) plus the VC1+VC2 gate of `v4rxc7v`, applied into the
worktree's real tree (the router is `fabric_erisc_router_v2.cpp`, the v2 slot); snapshot `fabric_exp/variants/vc2l_base`.
Everything uncommitted. Labels: measured (job id, offline build name), read from code (file:line), hypothesis.

Tooling (this worktree only; the main tree and its caches are not touched):
- `fabric_exp/vc2line/vc2l_offline.py`: coverage builds with the study's `coverage.py`/`offline_build.py` pointed at this
  worktree and at its own directory `/home/aagarwal/.tt-metal-vc2line-cov` (COV). Overlay base: `tt_metal/fabric` at
  checkpoint 12 (`git archive 5578d6292a6`, COV/base_ckpt12). The main tree's post-merge captures (cov_m) are only read;
  captures older than this work get line mode's new named args at neighbour-mode values (`add_vc2_line_args`). `variant
  NAME` snapshots every changed router source into `fabric_exp/variants/NAME`; `mock` captures with this worktree's host;
  `build`, `cmp` (byte compare of .text and of all loadable bytes).
- `fabric_exp/vc2line/mock_vet.sh YAML`: one CPU-only mock-Galaxy run per test (parse, host checks, kernel compiles; a
  mock run ends at program launch).
- `fabric_exp/vc2line/vc2l_glx.sbatch`: Galaxy runs (busy check, `cp -al` of the tree, `tt-smi -glx_reset` before the
  steps and after a timeout, KMD-lock watchdog, per-step timeouts). Dry-run with stubs (normal path and the hang path)
  before the first submission. `fabric_exp/vc2line/glx_vc2_line.yaml`: the line-mode tests.
- Galaxy tree `/data/aagarwal/frs-glx/trees/vc2l_p1`: a lean copy of this worktree (tt_metal, runtime, tests/tt_metal,
  ttnn, tt_stl, the libraries and test binaries of build_Release; 776 MB, 15k files; rsync 192 s, per-job `cp -al` 69 s),
  smoke-tested on the mock Galaxy (no local device: the study's smoke_runs.sh uses the local device).
- Phases 3 and 4: `fabric_exp/vc2line/vc2l_glx_mm.sbatch` (the Galaxy as two 2x2 meshes, one rank each, MPI over shared
  memory, hang capture armed; its default steps are phase 4's regression), `fabric_exp/vc2line/allvc_report.py` (phase 4's
  report), results in `fabric_exp/vc2line/results/`.

## Phase 1: VC2 line forwarding on one Galaxy mesh

### What line mode is (as built)

- Host flag `TT_METAL_FABRIC_VC2_LINE=1` (rtoption `fabric_vc2_line`; test yaml `fabric_setup: vc2_line: true`, which
  implies `use_vc2`), with `TT_METAL_ENABLE_FABRIC_VC2`. Off: neighbour mode, today's VC2 and today's images.
- Each cardinal router gets a second VC2 sender channel (compact `VC2_LOCAL_CHANNEL_START + 1`, a forwarding channel)
  fed by the router facing the opposite way on its chip. VC2's receiver channel decodes the 2D route (same header and
  `Routing2D` decode as VC0) and forwards only straight on (`live = 1 << opposite(my_direction)`) and/or delivers.
- Phase-1 limits (host `TT_FATAL` in `compute_intermesh_vc_config`, router `vc2_config_supported`): 2D Mesh only (rings
  need VC2 bubble flow control: phase 2), no VC1 (fan-in: phase 3), no express meshes (VC2's forwarding channel would be
  flat position 10 with VC1 + express or a Z boundary, past the 10-entry table; and no line goes through Z).
- The test infra refuses a VC2 flow that is not straight in line mode (`tt_fabric_test_context.cpp`, add_traffic_config:
  hops in more than one direction); checked on the mock Galaxy with a turning flow (TT_FATAL as expected).

### Device changes (router sources; diff against vc2l_base)

- `fabric_erisc_router_ct_args.hpp`: named args `VC2_LINE_MODE`, `NUM_DOWNSTREAM_SENDERS_VC2`, `VC2_DOWNSTREAM_EDM_SIZE`,
  `OPEN_DOWNSTREAM_VC2_CONNECTIONS`, `VC2_FREE_SLOTS_FROM_DOWNSTREAM_STREAM_ID`, `PACKED_DOWNSTREAM_VC2_SENDER_CHANNEL_IDS`
  (capture only). Transaction IDs (known problem 2): with three receiver channels, 4 IDs per channel also under deadlock
  avoidance (8 would put channel 2's at 16-23); a static_assert that every channel's IDs fit `NOC_MAX_TRANSACTION_ID`.
- `router_common.hpp`: the VC feature `vc_routes_as_line(vc)` (VC2 in line mode), `opposite_direction`; a line VC's
  downstream free-slots register is the downstream's VC2 forwarding channel's, its local copy is the one VC2 register.
- `router_downstream.hpp`: `DownstreamTraits<2>`; a line VC's live set is the opposite direction only. Command buffers
  (decision 5, static logic in the router): a line VC uses VC0's forwarding command buffers; where VC0's forwards are
  stateful (E/W routers inside a mesh: one live downstream, the opposite router, which is VC2's too) VC2's forwards are
  stateful on VC0's preset, and both VCs set the credit register on every forward (`forwards_set_credit_register`, one
  command-buffer register write per forward; before, only on deadlock-avoiding routers). Elsewhere VC2's forwards are
  stateless, as VC0's there.
- `router_receiver.hpp`: VC2's channel routes in line mode (`delivers_only[2] = !vc2_line_mode`), on its own downstream
  VC; the pass and `run` take VC2's downstreams as an optional last argument (absent in neighbour mode, so its code is
  unchanged).
- `router_sender.hpp`: `vc2_channels` = worker + forwarding; the forwarding channel joins `all_forwarding` (a record in
  the forwarding table), its free-slots register joins the pass's group read, and its group is visited after VC2's worker
  (`visit_forwarding_group<vc2_forwarding>`, VC2's far ring). Neighbour mode: masks and code unchanged.
- `router_init.hpp`, `router_single.hpp`, `fabric_erisc_router_v2.cpp`: VC2's downstream runtime args (after VC1's;
  teardown words after VC1's), init and open of VC2's connection, VC2's `Downstreams` built only where VC2 forwards
  (template helpers, so neighbour mode compiles none of it; kernel_main keeps its old call shape in neighbour mode).
- Also in the tree: the coordinator's `edm_fabric_worker_adapters.hpp` change (update_credits_and_slots templated on the
  sender type, for VC2 senders in benchmark mode), with the missing `template` keyword the main tree also got.

### Host changes

- `rtoptions` + inspector: `fabric_vc2_line` / `TT_METAL_FABRIC_VC2_LINE`.
- `IntermeshVCConfig::vc2_line` (fabric_builder_context.hpp/.cpp) and the phase-1 limits above.
- `router_wiring_rules.cpp`: VC2 senders = 1 + (line mode and cardinal facing); VC2 turn = the opposite direction.
- `fabric_builder_helpers`: `RouterProducerSlots` on VC2: worker channel 0, channel 1 fed by the opposite direction.
- `erisc_datamover_builder.cpp`: the new named args, `DISABLE_RX_CH2_FORWARDING` from the capture in line mode (1 in
  neighbour mode, as before), VC2's downstream runtime args in line mode, the capture-consistency check over all VCs.
- `fabric_stream_assignment`: line mode adds VC2's forwarding channel's free-slots register and VC2's downstream copy (2
  registers; 12 of 30 on a VC0+VC2 line mesh); the declared name set has 39 names.
- Slots (`fabric_static_sized_channels_allocator.cpp`, study rows at 4352 B, decision 6): VC0+VC2 line = VC0 17 (the
  rest), forwarding 12, receiver 16; VC2 worker 4, receiver 4, forwarding 4 (VC0 forwarding 14 -> 12 pays for VC2's
  forwarding channel and 2 more worker slots; receiver 4 keeps bubble flow control's 2 for phase 2; not tuned).
  `TT_FABRIC_EXP_SLOTS_VC2` takes a sixth value (VC2 forwarding) in line mode.
- Unit tests (fabric_unit_tests, run on a mock cluster): line-mode shape and turns, producer slots, stream registers;
  59/59 of the touched suites pass.

### Offline proof (CPU only)

- Mock captures with this worktree's host: bh_2d_vc2line (80 router builds), _1erisc (40), _allinstr (80; the mock
  crashes after the builds, as the main capture of bh_2d_allinstr does), glx_2d_vc2line (168), _1erisc (84); and
  neighbour mode bh_2d_vc2_wt, glx_2d_vc2_wt, glx_2d_wt (VC2 off).
- Host neutrality: this host's neighbour-mode named args equal the main captures' (bh_2d_vc2 80/80 configs, glx_2d
  168/168) plus the new names at exactly the values the offline injection uses (the positional slot args of bh_2d_vc2
  differ only by §25's study rows, which the checkpoint-12 host already has).
- Coverage -O3 (build p1b, all main captures + own): 2,684 of 2,814 ELFs build; the 130 failures are the `*_sc` stream
  credit configs, which fail by design (as for the base). All 284 line-mode ELFs build. Base vs p1b on the 2,400 ELFs
  both build: .text and all loadable bytes identical, 2,400 of 2,400 (an earlier shape of kernel_main, a lambda around
  the loop call, changed a register order in an init loop of 44 one-ERISC images; replaced).

### Code size (-O3, largest per ERISC, kernel config buffer 26,624 B; build p1b, size_report.py)

| config | E0 | E1 | E0 + E1 | least free stack E0 / E1 |
| --- | ---: | ---: | ---: | ---: |
| P150 2D, VC2 neighbour (unchanged) | 11,584 | 7,928 | 19,512 | 1,732 / 4,400 |
| P150 2D, VC2 line | 12,352 | 8,764 | 21,116 | 1,636 / 4,320 |
| P150 2D, VC2 line, all instrumentation | 14,500 | 10,032 | 24,532 | 1,508 / 4,128 |
| P150 2D, VC2 line, one ERISC | | 15,360 | 15,360 | 3,704 |
| Galaxy 2D, VC2 neighbour (unchanged) | 11,640 | 8,540 | 20,180 | 1,716 / 4,368 |
| Galaxy 2D, VC2 line | 12,480 | 10,016 | 22,496 | 1,620 / 4,224 |
| Galaxy 2D, VC2 line, one ERISC | | 17,268 | 17,268 | 3,592 |
| (VC1 + VC2 neighbour, unchanged) | 13,872 | 12,284 | 26,156 | 1,492 / 4,184 |

Line mode costs +1,600 to +2,300 B over neighbour mode; every line config fits without growing the buffer (decision 2:
not needed in phase 1). The tightest config is still VC1 + VC2 neighbour (468 B left), which phase 3 extends.

### Galaxy runs

- Job 166884 (bh-glx-110-a01u08; tree vc2l_p1; `tt-smi -glx_reset` before the steps: exit 0, tt-smi 6.6.0):
  - line (glx_vc2_line.yaml, two ERISCs): 12/12 passed, data-checked: unicast along rows (1-3 hops, both ways) and
    columns (2-7 hops, both ways), writes and fused atomic incs, 1 KiB and 4 KiB; both directions of a row and a column
    at once; multicast along a row (E, W) and a column (S 7 hops, N 7 hops, S 3 from the middle); VC0 2D traffic that
    turns on the same chips and links as VC2 lines (separate worker cores). The kernel cache shows every router built
    with VC2_LINE_MODE=1 and two VC2 sender channels.
  - line_1e (the same on one ERISC): 12/12.
  - nbr_vc2 (the VC2 neighbour yaml, neighbour mode): 10/10; nbr_vc2_1e (one ERISC): 10/10.
  - line_vc2 (the VC2 neighbour yaml with every router in line mode, env TT_METAL_FABRIC_VC2_LINE=1): 10/10.
  - nbr_sanity (test_fabric_mesh_sanity_common.yaml, VC2 off): 96/96.
  - line_sanity (the same yaml, VC2 on in line mode, VC2 idle): refused at once by the host: the yaml's first group is
    Linear, and the phase-1 check refused line mode on any topology but Mesh. Fixed: line mode applies only where VC2
    has channels (2D fabrics; 1D fabrics create none), and only a torus is refused. Vetted on the mock Galaxy (the
    Linear group builds with VC2_LINE_MODE=0, the Mesh group with 1), tree re-synced.
- Job 166892 (bh-glx-110-a01u02; same tree with the fix; `tt-smi -glx_reset` exit 0): line_sanity 96/96 and
  line_sanity_1e 96/96 (the mesh sanity yaml with VC2 on in line mode on every Mesh router: VC0's traffic, incl. its
  stateful E/W forwards that now set their credit register per forward, unaffected).
- Router images of both jobs = offline build p1b (router sources unchanged since; the fix is host-only).

### Problems found (phase 1)

- The coordinator's `edm_fabric_worker_adapters.hpp` change (benchmark-mode VC2 senders) lacked a `template` keyword
  and broke every test-kernel build; fixed here (the main tree had the same fix).
- Test infra: `use_vc2` (and now `vc2_line`) set the rtoption only to true and never back, so a later test in the same
  run keeps VC2 (pre-existing; line tests are in a yaml of their own).
- The study's Galaxy smoke test (smoke_runs.sh) runs on the local device; replaced here by a mock-Galaxy run of the
  /data tree.
- Not addressed (known, out of phase 1): line mode with VC1 plus express or a Z boundary would put VC2's forwarding
  channel at flat position 10 (the table has 10); the VC2 worker API's flat index vs the router's compact index on
  Z-boundary fabrics (VC2_LINE_IDEA.md problem 3); VC0/VC1 sharing forward data command buffer 2 (problem 1, phase 3).
- Not measured (performance pass later): VC0's cost on line-mode routers (E0: one more register in the pass's group
  read and one test per pass for VC2's forwarding group; E1 on E/W routers: VC0's credit register set per forward).

### For phase 2 (VC2 rings), read from code

- Injection flags: the non-express torus policy already makes VC2's worker an injection channel and VC2's forwarding
  channel not (same axis, never a turn; injection_policy.cpp); the gate is `bubble_flow_control_enabled_on_vc(2)`
  (fabric_builder_config.hpp), plus the receiver's 2-slot check per VC (erisc_datamover_builder.cpp; VC2's ring is 4).
- First-level acks on VC2: a named arg (`ENABLE_FIRST_LEVEL_ACK_VC2`), the sender's `Channel<CH>::first_level_ack` for
  VC2 and the receiver's `first_level_ack_on[2]`; the router's ack code is per channel already.
- Transaction IDs: VC0 + VC2 with deadlock avoidance is 2 channels x 8 = 16 IDs (fits); 3 channels get 4 each (fixed).
- Credit register per forward: deadlock-avoiding routers already set it on every forward.

## Phase 2: VC2 rings on a single-Galaxy torus

User decisions (2026-11-01): VC0's per-axis deadlock-avoidance rule for VC2 (rings on torus axes, lines on the other);
X-only and Y-only torus, then XY; `ENABLE_FIRST_LEVEL_ACK_VC2` mirroring VC0's, bubble flow control on VC2 (worker =
injection, 2 free far slots; forwarding 1); VC1 never; small-ring stress with a 2-slot VC2 receiver; fix the test
infra's VC2 rtoption carry-over; no performance pass yet.

### Changes

- Device: named arg `ENABLE_FIRST_LEVEL_ACK_VC2` (ct_args); the sender's `Channel<CH>::first_level_ack` and the
  receiver's `first_level_ack_on[2]` take it (the ack, completion and credit code is per channel already); a
  static_assert that an injection channel's VC has first-level acks; `vc2_config_supported` admits line mode with
  deadlock avoidance, and VC2 acks only in line mode. Nothing else: VC2's worker gets the 2-slot rule from the host's
  injection flag through `far_slots_needed_by`, as VC0's does.
- Host: `bubble_flow_control_enabled_on_vc(vc, vc2_line)` = VC0, or VC2 in line mode (fabric_builder_config.hpp; the
  injection-flag derivation in compute_mesh_router_builder.cpp uses it). The non-express torus policy already makes
  VC2's worker injection and its forwarding channel not (same axis, never a turn). In the EDM builder:
  `vc2_rings = vc2_forwards && base deadlock avoidance` (the router's axis is a ring); `ENABLE_FIRST_LEVEL_ACK_VC2 =
  vc2_rings`; deadlock avoidance stays on where VC2 has rings even when VC0's trimmed shape would relax it; the 2-slot
  receiver check covers VC2's receiver. The phase-1 torus refusal is gone (VC1 and express still refused).
- Test infra (user decision 4): each test group sets the VC2 rtoptions to the environment's value or its own
  fabric_setup's (`use_vc2`, `vc2_line`), so a group's VC2 no longer carries over to later groups.
- Slots: the line row (17/12/16 + VC2 4/4/4) applies on tori too (4 VC0 senders); the small ring is
  `TT_FABRIC_EXP_SLOTS_VC2=0,12,16,4,2,4` (VC2 receiver 2; checked on the mock: VC2 sender {4, 4}, receiver {2}).

### Offline proof (CPU only; build p2a)

- Mock captures: glx_torus_vc2line (XY, 120 router builds), glx_torus_x_vc2line (168), glx_torus_y_vc2line (120),
  glx_torus_vc2line_1erisc (60), bh_torus_vc2line (96), bh_torus_vc2line_allinstr (96), glx_torus_vc2 (neighbour mode
  on a torus, 120). Per-axis rule as captured: X-only torus, E/W routers deadlock avoidance + VC2 acks, N/S routers
  neither; Y-only the reverse; XY all; neighbour mode on a torus no VC2 acks and VC2's worker not injection (as before).
- p2a builds 2,936 of 3,066 ELFs (the 130 `*_sc` fail by design). Byte comparisons (.text and all loadable bytes):
  base vs p2a 2,400 / 2,400 identical (every neighbour-mode and VC2-off config); neighbour mode on a torus (a new
  capture, base built on it) 16 / 16 identical; phase 1 (p1b) vs p2a 2,684 / 2,684 identical (line mode on meshes
  unchanged).
- Unit tests: 59/59 of the touched suites (mock cluster).

### Code size (-O3, build p2a)

| config | E0 | E1 | E0 + E1 | least free stack E0 / E1 |
| --- | ---: | ---: | ---: | ---: |
| Galaxy torus XY, VC2 off (unchanged) | 10,820 | 6,984 | 17,804 | 1,828 / 4,416 |
| Galaxy torus XY, VC2 neighbour (unchanged) | 12,248 | 8,472 | 20,720 | 1,716 / 4,368 |
| Galaxy torus XY, VC2 line (rings, VC2 acks) | 13,340 | 10,068 | 23,408 | 1,604 / 4,208 |
| Galaxy torus X only, VC2 line | 12,880 | 10,016 | 22,896 | 1,604 / 4,224 |
| Galaxy torus Y only, VC2 line | 13,340 | 10,068 | 23,408 | 1,604 / 4,208 |
| Galaxy torus XY, VC2 line, one ERISC | | 18,108 | 18,108 | 3,592 |
| P150 torus XY, VC2 line | 13,204 | 8,928 | 22,132 | 1,620 / 4,304 |
| P150 torus XY, VC2 line, all instrumentation | 15,316 | 10,176 | 25,492 | 1,492 / 4,128 |

VC2 rings with first-level acks fit the 26,624 B buffer; the instrumented torus config is the tightest (1,132 B left).

### Tests

- `fabric_exp/vc2line/glx_vc2_ring.yaml` (generated by gen_ring_yaml.py; 18 test groups, vetted on the mock Galaxy):
  per torus (X only, Y only, XY): WrapUnicast (shortest paths over the wrap link, writes and fused atomic incs, 1 and 4
  KiB; no flow exactly half a ring away, where the test infra and the routing table may break the tie differently),
  FullRingMcast (around a whole ring, wrapping), AllChips (every chip of a ring multicasts around it in both directions
  at once), WithVC0 (the same VC2 load next to VC0 full-ring multicasts and wrapping and turning VC0 unicasts); on X-only
  and Y-only tori a VC2 line on the axis that is not a ring.
- The torus deadlock-stability suite and the short torus stability suite (copied from /data/aagarwal/frs-glx/yamls),
  in neighbour and line mode; a subset (Ring, Torus X, Y, XY) vetted on the mock in line mode.

### Galaxy runs

- Tree `/data/aagarwal/frs-glx/trees/vc2l_p2` (hard links from vc2l_p1, changed files re-synced: new inodes, vc2l_p1
  unchanged).
- Job 167247 (bh-glx-120-c04u08; `tt-smi -glx_reset` exit 0):
  - ring (glx_vc2_ring.yaml, two ERISCs): 34/34 passed (the 18 groups' parametrizations over the X-only, Y-only and XY
    tori). The kernel cache: 216 router builds with deadlock avoidance and VC2 first-level acks (the ring axes), 88
    without (the line axis of the X-only and Y-only tori), all with VC2 line mode.
  - ring_small (the same, VC2 receiver ring of 2 slots, `TT_FABRIC_EXP_SLOTS_VC2=0,12,16,4,2,4`; the log shows VC2
    sender {4, 4}, receiver {2}): 34/34.
  - ring_1e (one ERISC): 34/34.
- Job 167252 (bh-glx-120-c04u08; `tt-smi -glx_reset` exit 0), the existing torus validation, two ERISCs:
  - tds_nbr (test_fabric_2d_torus_deadlock_stability.yaml: Ring, Torus X, Y, XY multicasts from every chip; VC2
    neighbour mode): 87/87; tds_line (the same, VC2 line mode: 216 router builds with VC2 acks, 208 without): 87/87.
  - tst_nbr (glx_torus_stability_short.yaml: all-to-all on Mesh, Torus XY, Linear, Ring; VC2 neighbour mode): 56/56;
    tst_line (line mode): 56/56.
- Job directories 166884, 166892, 167247 and 167252 deleted after the results were taken (user, disk).

### Phase 2 status

VC2 rings work on the Galaxy torus (X only, Y only, XY), data-checked: wrap unicast, full-ring multicast, both
directions at once from every chip, next to VC0 torus traffic, with the default and a 2-slot VC2 receiver ring, on two
ERISCs and one. Neighbour mode and VC2-off images are unchanged (byte-identical), line mode on meshes is unchanged from
phase 1 (byte-identical). Not done (by decision): performance.

## Phase 3: VC0 and VC2 fanning into VC1 (single Galaxy split into meshes)

User decisions (2026-11-01): VC2 mirrors VC0: a VC2 packet for another mesh leaves only where its line meets the mesh
edge and fans into VC1 at the inter-mesh router; VC1 carries it in the destination mesh; VC2 lines do not continue
there. The command-buffer-2 fix (VC0's preset forwards vs VC1's stateless ones on E/W routers inside meshes larger than
2x2) waits for a separate audit on the main tree: no multi-mesh Galaxy tests on meshes larger than 2x2 with VC1. Code
size: build and measure first; grow the kernel config buffer if a config overflows (not -O2).

### Design (as built)

- Source mesh: nothing new on the device. The worker's route to another mesh is the route to this mesh's exit chip
  (`fabric_set_unicast_route`); VC2's receiver on the exit chip's edge router turns the packet's "deliver here" into an
  exit (the existing `exits` step of `Routing2D`, `exit_action<VC>`), which for a line VC is a forward straight on into
  the intermesh router's VC2 forwarding channel, which sends it over the intermesh link on VC2.
- Destination mesh, the intermesh router: VC2's receiver channel lands (`IS_RECEIVER_CHANNEL_<vc2>_INTERMESH_INGRESS`,
  the landing rewrite before the decode) and crosses over onto VC1 (`FABRIC_2D_VC0_CROSSOVER_TO_VC1` makes every receiver
  channel's downstream VC VC1 on such routers), on the same VC1 downstream objects as VC0's crossover: VC0 and VC2 share
  VC1 there (they may block each other at VC1; accepted). VC1 then carries it as it carries VC0's.
- Deadlock (read from code): the VC dependencies are VC0 -> VC1 <- VC2; VC1 receivers forward only into VC1, nothing
  drains from VC1 into VC0 or VC2, so the graph is acyclic; VC2 lines are acyclic on meshes (phase 1) and ring-guarded on
  tori (phase 2). No concern found.
- A route that would turn (or exit sideways) is a flow the host refuses (test infra, below); in watcher builds the
  receiver asserts that a line VC's route has no direction it cannot forward to.
- The intermesh router's own VC2 straight connection (to the router facing into its mesh) stays opened, though no
  channel forwards on it there: that router's VC2 forwarding channel waits for its producer at init.

### Changes

- Device: `vc2_config_supported` admits line mode with VC1; the turn assert (`Route::turns()`, router_routing.hpp).
  Nothing else: landing, crossover and exits were already per channel and VC.
- Host: VC2's receiver channel is intermesh ingress on intermesh routers in line mode; VC1's downstream connections open
  for VC2's crossover too; the VC1 refusal in line mode is gone (express still refused; a Z intermesh boundary would put
  VC2's forwarding channel past the flat table: the stream assignment refuses it). Slots: VC0+VC1+VC2 line row
  16 (the rest), 7, 16, 4, 4 + VC2 4, 4, 4 (81 slots at 4352 B; not tuned).
- Test infra: a VC2 flow to another mesh must go straight to its exit and leave in that direction (walks the routing
  tables from the source; checked where the source is the process's own); a turning flow (mesh 0 chip 0 -> mesh 1) is
  refused on the mock as intended.
- Kernel config buffer: `MEM_ERISC_KERNEL_CONFIG_SIZE` 26 KiB -> 29 KiB (dev_mem_map.h; HAL and firmware derive from it).
  Why 29: multi-mesh line mode needs 28,412 B of code at -O3 plus ~544 B of runtime args (68 words per ERISC); 28 KiB
  does not fit, 29 KiB leaves ~740 B. The fabric's channel buffering shrinks by 3 KiB: measured on the mock (slot log,
  now with the buffering space): 2D/torus 364,352 -> 361,280 B, still 81 slots of 4,448 B (study rows unchanged); 1D
  365,760 -> 362,688 B, 83 -> 82 slots (the 1D split: worker channel 26 -> 25 slots, forwarding 25, receiver 32). 30 KiB
  would cost 2D a slot (80), which also sends the VC0+VC1 row (81 slots by design) back to the tier table. Router code
  does not depend on the buffer size (p3a vs p3b on this worktree's captures: identical; the main tree's captures cannot
  take an overlaid dev_mem_map.h, which they include twice).
- Tooling: multi-mesh mock captures (two ranks) in vc2l_offline.py (P150 split and the Galaxy split); ttrun.py runs on a
  small venv in the worktree (python_env: click, pyyaml, loguru, pydantic), copied into the Galaxy tree.

### The Galaxy split

- `fabric_exp/vc2line/glx_2x2_split_mgd.textproto` + `glx_2x2_split_rank_bindings.yaml`: two 2x2 meshes from the first
  tray, cluster-descriptor chips {0, 1, 4, 5} | {2, 3, 6, 7} (in single_bh_galaxy_clus_desc, 0-1-2-3 and 4-5-6-7 are rows,
  0-4 .. 3-7 columns), joined by chip pairs 1-2 and 5-6 (two links each), as the P150 split.
- Why the command-buffer-2 problem cannot occur here (read from code, router_downstream.hpp forwards_are_stateful and
  router_routing.hpp): VC0's forwards are stateful only on E/W routers on chips without intermesh links; such a router
  forwards VC1 only for a VC1 packet that arrived on it and continues along X in the same mesh. In a 2x2 mesh an X leg
  inside the mesh is one hop, so a VC1 packet arriving at an E/W router is at its destination column: it is delivered,
  never forwarded (landings decode Y first at the intermesh router, and no pass-through). So no VC1 stateless forward
  ever runs on a router with VC0's preset.
- Placement on the mock (router-arg dump): mesh 0 exits at chip 1 (facing N) and chip 3 (facing E); mesh 1 at chip 0
  (N) and chip 2 (S). Straight VC2 flows to the other mesh: from an exit chip itself, or mesh 0's chip 2 through chip 3.

### Offline proof (CPU only; build p3a, router sources = p3b without the buffer header)

- Mock captures: bh_vc1_vc2line (94 router builds), _1erisc (47), _allinstr (96); glx_vc1_vc2line (78), _1erisc (40),
  glx_vc1_vc2 (neighbour, 80), glx_vc1 (VC2 off, 78), glx_vc1_vc2line_o3 (at -O3 with 29 KiB: both ranks reach launch).
- p3a builds 3,308 of 3,438 ELFs (the 130 `*_sc` fail by design). Identical (.text and loadable bytes): base vs p3a
  2,400 / 2,400; the Galaxy split in VC2-off and neighbour mode (new captures, base built on them) 150 / 150; phase 2
  (p2a) vs p3a 2,936 / 2,936 (single-mesh line mode, rings included).

### Code size (-O3, build p3a; current buffer 26,624 B, grown 29,696 B)

| config | E0 | E1 | E0 + E1 | least free stack E0 / E1 |
| --- | ---: | ---: | ---: | ---: |
| P150 split, VC2 off (unchanged) | 12,408 | 10,932 | 23,340 | 1,604 / 4,216 |
| P150 split, VC2 neighbour (unchanged) | 13,872 | 12,284 | 26,156 | 1,492 / 4,184 |
| P150 split, VC2 line | 14,644 | 13,768 | 28,412 | 1,380 / 3,992 |
| P150 split, VC2 line, one ERISC | | 23,348 | 23,348 | 3,168 |
| P150 split, VC2 line, all instrumentation | 16,936 | 15,304 | 32,240 | 1,252 / 3,832 |
| Galaxy split, VC2 off / neighbour | 12,400 / 13,864 | 10,932 / 12,284 | 23,332 / 26,148 | |
| Galaxy split, VC2 line | 14,636 | 13,768 | 28,404 | 1,380 / 3,992 |
| Galaxy split, VC2 line, one ERISC | | 23,340 | 23,340 | 3,168 |

Two-ERISC multi-mesh line mode overflows 26,624 B by ~1.8 KB (2.3 KB with runtime args): the buffer grows to 29 KiB.
All instrumentation at once (a study-only combination) needs ~33 KiB and stays out.

### Galaxy runs

- Tree `/data/aagarwal/frs-glx/trees/vc2l_p3` (hard links from vc2l_p2, changed files re-synced, plus fabric_exp/vc1 and
  python_env); multi-mesh runs via `fabric_exp/vc2line/vc2l_glx_mm.sbatch` (ttrun.py, one rank per mesh; dry-run with
  stubs, normal and hang paths; the hang path kills only processes whose command line names the job's own tree).
- Job 167319: no rank started: under Slurm, OpenMPI saw a one-task allocation ("All nodes which are allocated for this
  job are already filled"). The script now passes `--mpi-args --oversubscribe` (the mock runs already did).
- Job 167327 (bh-glx-120-c03u14; `tt-smi -glx_reset` exit 0; routers at -O3 unless noted), per rank (two ranks):
  - mm_fanin (glx_vc2_fanin.yaml: VC2 from mesh 0's and mesh 1's exit chips to every chip of the other mesh, writes and
    fused atomic incs, 1 and 4 KiB; the VC2 line mesh 0 chip 2 -> chip 3 -> mesh 1; VC2 1-hop over every intermesh link
    both ways): 10/10 and 10/10. The test infra's straight-line check passed: the Galaxy's placement is the mock's.
  - mm_fanin_1e (one ERISC): 10/10, 10/10; mm_fanin_os (the host's default -Os): 10/10, 10/10.
  - mm_sanity_line (vc1_sanity.yaml: VC0 multi-mesh all-to-all and inter-mesh neighbour exchange; VC2 line mode, idle):
    12/12, 12/12; mm_sanity_nbr (VC2 neighbour mode): 12/12, 12/12.
  - mm_alloc_line (vc1_alloc.yaml, VC2 line mode): 25/25, 25/25.
  - mm_vc2_line (the VC2 neighbour yaml within each mesh, line mode): 10/10, 10/10.
  - The kernel cache: every router with VC2_LINE_MODE=1; the 16 intermesh router builds with VC2's receiver channel as
    intermesh ingress. Loaded router sizes: -O3 E0 14,656 + E1 13,764 = 28,420 B; -Os 13,600 + 13,836 = 27,436 B. Both
    exceed the old 26,624 B buffer and run in the 29 KiB one; so the default -Os would not have fit 26 KiB either.
- Job 167335 (single mesh, tree vc2l_p3, i.e. with the 29 KiB buffer; `tt-smi -glx_reset` exit 0): line
  (glx_vc2_line.yaml) 12/12; nbr_sanity (mesh sanity, incl. its Linear group on 82 slots) 96/96; ring (glx_vc2_ring.yaml)
  34/34; tst_line (short torus stability, VC2 line mode) 56/56. The grown buffer changes no result.
- Job directories 167319, 167327, 167335 deleted after the results were taken.

### Phase 3 status: stopped at the command-buffer-2 fix

Done: the fan-in design, device and host changes, offline proofs, sizes, the buffer growth, and the Galaxy 2x2 split
(the only multi-mesh shape where the command-buffer-2 problem cannot occur). Waiting for the fix (audit on the main
tree): multi-mesh tests on meshes larger than 2x2 with VC1, e.g. the Galaxy as two 4x4 or four 4x2 meshes. Note for the
audit: in line mode VC2's forwards are stateful on VC0's preset exactly where VC0's are (E/W routers inside a mesh), so
on multi-mesh fabrics with meshes larger than 2x2 a VC1 stateless forward on command buffer 2 would clobber VC2's
forwards as well as VC0's; the fix should cover the preset VC2 shares (router_downstream.hpp, forwards_are_stateful and
forwards_set_credit_register).

## Phase 4: VC0, VC1 and VC2 carrying traffic at once (Galaxy 2x2 split)

The first test with all three VCs busy at once: one data-checked yaml with every flow concurrent, in three modes, with
stress and the hang capture armed, then a first performance look (no tuning). Claims are labelled measured (job), read
from code (file), or hypothesis.

### Tests and tooling

- `fabric_exp/vc2line/glx_allvc.yaml` (data-checked, test flow control on, VC2 line mode; unicast writes and fused atomic
  incs at 1 and 4 KiB, 5,000 packets per flow), every flow concurrent (21 sender entries):
  - VC0 inside each mesh: 2D unicasts that turn, both ways ([0,0] <-> [0,3], [1,1] <-> [1,2]), and a 2D multicast per mesh;
  - VC0 between the meshes: three flows each way, VC1 in the destination mesh after the crossover;
  - VC2 inside each mesh: four 1-hop lines;
  - VC2 between the meshes: from every exit chip (two destinations each) and from mesh 0's chip 2 (a line through chip 3),
    VC1 after the crossover.
  Both directions leave through the same exit routers and land on the same intermesh routers, so VC0's and VC2's
  crossovers share VC1's downstream connections while both are busy.
- `glx_allvc_perf.yaml` (benchmark, sync): one flow at a time and two flows on one path, each VC alone and VC0 with VC2,
  across the meshes (mesh 0 chip 2 VC0 and chip 3 VC2 to mesh 1 chip 1: one exit, one landing router, its VC1 shared)
  and inside a mesh (mesh 0 chip 2 VC0 and chip 0 VC2 to chip 1: one link); PerfAll: VC0 and VC2 in both meshes and
  across, one flow per sender chip and first hop. `glx_allvc_perf_nosync.yaml`: the several-flow tests with sync off
  (per-flow rates, below). `glx_allvc_perf_ab.yaml`: the VC2 flows alone, for the window A/B.
- `vc2l_glx_mm.sbatch`: steps allvc (two ERISCs, -O3), allvc_1e (one ERISC), allvc_os (the host's default -Os),
  allvc_small (VC2 receiver at 2 slots), allvc_r2/_r3 (repeats), perf, perfns, perfab/_vc1x2/_vc2x2, capself (a capture
  of the idle device: checks the capture's mechanics). Default STEPS is this list: the post-rebase regression (about 10
  minutes). The ranks talk over shared memory (`--mca btl self,sm`, job 167436 below).
- `allvc_report.py`: per-step results, slot rows, per-sender lines, shared paths, the A/B table.
- Test infra: `tt_fabric_test_bandwidth_profiler.cpp` is the main tree's working copy (its uncommitted change: every
  sender core logs "Sender core ... (VCn) on device d direction D link l: P packets in C cycles", and several sender
  cores may share a link), used from job 167449 on. Checkpoint 12's profiler writes one CSV row per test and rank
  (tt_fabric_test_context.cpp: append_to_csv(config, latest_results.back())) and aborts when two sender cores on one
  chip leave the same way.
- Results kept in `fabric_exp/vc2line/results/` (reports, per-sender lines, the hang decode); job directories deleted.

### Jobs

- 167436 (bh-glx-110-a05u14, tree vc2l_p4): no test ran. Both ranks sat in MPI start-up, before any device work
  (no kernel built), in their first collective (gdb, both ranks: PMPI_Comm_dup <- MPIContext::duplicate <-
  MPIContext::create <- MetalContext::instance, from test_tt_fabric's static init). ttrun's default for the ranks' traffic
  is TCP with lo and docker0 excluded; this node also has a flannel/calico overlay and about twenty bridges. Cancelled
  after 5 minutes. Since then the ranks (one node) use shared memory; mock and Galaxy runs work with it. That the TCP
  interface choice caused the stall is a hypothesis (not re-tested on that node).
- 167439 (bh-glx-120-c02u14) and 167440 (bh-glx-120-c04u02): stopped by the busy check before any device work (another
  uid's 7-day-old process; another user's long-running Claude daemon and shell). Later jobs pinned to bh-glx-120-c04u08.
- 167441 (c04u08, tree vc2l_p4): the data check in every mode, perf (checkpoint 12's profiler), capself.
- 167449 (c04u08, tree vc2l_p4b = p4 + the profiler + a PerfAllFlows test + the A/B steps): perf hung in PerfAllFlows,
  after the other tests had run (with per-sender lines); the hang path killed the job's processes, captured all 8 chips
  (133 MB, 0 read errors), reset (exit 0) and stopped. Below.
- 167468 (c04u08, tree vc2l_p4c = p4b without PerfAllFlows, with the no-sync yaml): the window A/B, per-flow rates
  without sync, perf repeated.
- Every `tt-smi -glx_reset` exit 0, 32 chips ("64 listed" in the job logs: `tt-smi -ls` prints each chip in two tables).

### Data check (job 167441; per rank, both ranks)

| step | mode | result | time |
| --- | --- | --- | ---: |
| allvc | two ERISCs, -O3 | 4/4, 4/4 | 40 s |
| allvc_1e | one ERISC, -O3 | 4/4, 4/4 | 40 s |
| allvc_os | two ERISCs, the host's default -Os | 4/4, 4/4 | 40 s |
| allvc_small | two ERISCs, -O3, VC2 receiver 2 slots | 4/4, 4/4 | 40 s |
| allvc_r2, allvc_r3 | repeats of allvc | 4/4, 4/4 each | 51 s, 40 s |

Every run validated its data (no "Skipping validation" line: tt_fabric_test_context.cpp, validate_results). capself:
the capture works on the Galaxy (8 chips, 30,212 reads, 0 errors, 7.2 s).

### Slot rows in use (TT_FABRIC_EXP_LOG_SLOTS; the same on both ranks)

- Default, the VC0+VC1+VC2 line row: VC0 senders {16, 7, 7, 7} receiver 16, VC1 senders {4, 4, 4} receiver 4, VC2
  senders {4, 4} (worker, forwarding) receiver 4: 81 slots of 4,448 B (361,280 B).
- allvc_small (`0,7,16,4,4,4,2,4`): VC2 receiver 2, VC0's worker takes the freed slots (18).
- A/B rows: VC1 x2 (`6,5,16,8,8,4,4,4`): VC0 {6, 5, 5, 5}, VC1 {8, 8, 8} receiver 8; VC2 x2 (`4,7,16,4,4,8,8,8`): VC0
  {4, 7, 7, 7}, VC2 {8, 8} receiver 8.
- Five more rows per rank (VC0 {4, 0, 0, 0} receiver 8, VC1 receiver 4) are the per-direction MUX-mode configs the
  builder creates and this fabric does not use (fabric_builder_context.cpp, create_edm_config(MUX, direction)).

### Performance (benchmark, -O3, 20,000 packets per flow, device clock 1,350 MHz)

Cycles per packet = a flow's cycles over its packets (the Galaxy link reference: about 248 cycles per 4,352 B packet).
With sync, a sender's cycles run to the end of the test's global sync (tt_fabric_test_sender.cpp, the timed region), so
in a several-flow test every sender of a mesh reports about the slowest flow: one-flow tests give that flow's rate, and
several-flow tests the combined rate; per-flow rates of several flows come from the no-sync tests (their one-flow
controls match the sync runs within 0.4%: no start-skew effect).

One flow alone (sync; jobs 167441, 167449 and 167468 agree within 0.6%):

| flow | path | 1 KiB | 4 KiB | 4 KiB GB/s |
| --- | --- | ---: | ---: | ---: |
| VC0 inside, mesh 0 chip 2 -> chip 1 | 2 VC0 hops (north, then east) | 353.5 | 402.7 | 13.7 |
| VC2 inside, mesh 0 chip 0 -> chip 1 | 1 VC2 hop | 465.0 | 507.1 | 10.9 |
| VC0 across, mesh 0 chip 2 -> mesh 1 chip 1 | 2 VC0 hops, landing, 2 VC1 hops | 525.7 | 578.6 | 9.6 |
| VC2 across, mesh 0 chip 3 -> mesh 1 chip 1 | 1 VC2 hop, landing, 2 VC1 hops | 554.2 | 601.9 | 9.2 |

Two flows on one path (no sync, per flow; job 167468; 4 KiB):

| shared path | VC0: alone -> with VC2 | VC2: alone -> with VC0 | both packets over the slower flow's cycles |
| --- | ---: | ---: | ---: |
| mesh 0 chip 3's exit + the landing router's VC1 | 578.9 -> 1,063.8 | 601.4 -> 1,102.8 | 551.4 (10.0 GB/s) |
| mesh 0 chip 0 -> chip 1 link (VC0 forwarded, VC2 injected) | 402.7 -> 453.3 | 507.1 -> 586.6 | 293.3 (18.9 GB/s) |

- Neither VC starves (measured): across the meshes the two crossovers split the shared path evenly (VC0 51%, VC2 49% of
  the packets per cycle at 4 KiB; 1 KiB the same), and together they move barely more than either alone (551 against
  579 and 602 cycles per packet). Inside a mesh VC0 finishes first (453 against 587); while both run, VC2 gets about
  42% (estimate, assuming VC2 runs at its alone rate after VC0 ends), and the link carries about 261 cycles per packet.
- All flows at once (NsAll, no sync, one flow per sender chip and first hop, 4 KiB; mesh 1's flows send 16,000
  packets): every flow completes; 1-hop VC0 238-316 cycles per packet, 1-hop VC2 506-523, the flows into VC1 764-965 for
  both VCs (VC0 963 and 807, VC2 965, 798, 786, 764). VC0 and VC2 crossovers into VC1 get similar rates.

Window A/B (job 167468; one flow per test, sync; each row changes one VC's windows, its slots taken from VC0 channels
that the flow does not use):

| flow | default | VC1 8/8 | VC2 8/8/8 |
| --- | ---: | ---: | ---: |
| VC2 inside, 1 hop, 4 KiB | 506.6 | 508.1 | 293.9 |
| VC2 inside, 1 hop, 1 KiB | 465.1 | 462.8 | 264.4 |
| VC2 across (VC2 hop, VC1 hops), 4 KiB | 603.4 | 598.7 | 551.7 |
| VC2 across, 1 KiB | 554.4 | 547.9 | 501.8 |

- VC2's 4-slot windows bound a VC2 flow inside a mesh (measured): 8/8/8 makes it 42% faster per packet at 4 KiB and 43%
  at 1 KiB, while VC1 8/8 leaves it unchanged (its control). At 1 KiB the 4-slot flow is nearly as slow per packet as at
  4 KiB (465 against 507): a window bound, not bandwidth.
- VC1's 4/4 is not the bound of the VC2 flow across the meshes at the default row (measured: VC1 8/8 changes it by
  -0.8%; VC2 8/8/8 by -8.6%). For VC0 flows across the meshes the A/B was not isolating (the VC1 x2 row takes its slots
  from VC0 channels on that flow's path) and was not run: open whether VC1's 4/4 bounds them. What bounds the flows
  across the meshes (about 550-600 cycles per packet alone, and their shared path at 551) is not established: a
  hypothesis would have to name a piece and be tested with an A/B.

### The PerfAllFlows hang (job 167449): the test infra's mux and sync core on one router channel

PerfAllFlows ran every flow of glx_allvc.yaml in benchmark mode with sync. Its first run stopped at "Waiting for
programs" until the 1,200 s timeout; the armed hang path captured and reset. Decode (`hang_decode.py` with the run's
cache, `results/167449_perf_hangcap_decode.txt`):

- Routers: no packet and no credit waiting anywhere ("every channel empty, every copy equal to its downstream
  register"), heartbeats moving: the routers were alive and empty.
- Workers: on mesh 0 chip 0 and mesh 1 chip 1, the two VC0 senders that leave the same way (the unicast across the mesh
  and the trunk part of the split 2D multicast) are fed through a test-infra mux core (`tt_fabric_mux`,
  HAS_MUX_CONNECTIONS=1) and sent 0 of 20,000; the mux ran but made 4 NoC writes and 2 reads in all. Every other sender
  sent all its packets. Those chips' sync cores wait for the two senders; every other chip's sync core waits for the one
  missing end-of-test sync packet.
- On those chips the sync core holds its own direct connection to the same router worker channel (its second sync
  multicast leaves the same way; it was the router channel's last producer).
- Read from code: with sync and without flow control the test infra keeps the sync cores in a separate connection
  manager (tt_fabric_test_context.cpp:705-712; unified, "for correct mux detection", only with both), so that router
  worker channel gets two independent clients, the mux and the sync core.
- The data-checked glx_allvc.yaml (flow control on, no sync) builds muxes as well (mock: four mux configs on rank 0, three
  on rank 1) and passed 6 of 6 on the Galaxy.

Mechanism (hypothesis, the step inside the mux handshake not shown): the mux and the sync core contend for one router
worker channel and the mux's senders never got through. Not a router stall. A hang rules the shape out: no benchmark
test with sync has two senders sharing a chip's first hop (PerfAllFlows removed); per-flow rates of all flows come from
NsAll (one flow per chip and first hop, no sync, no mux).

### Phase 4 status

Done: VC0, VC1 and VC2 carry traffic at once on the Galaxy 2x2 split, data-checked, in both ERISC modes and at -O3 and
-Os, with VC2 at 2 slots and repeats: 24 of 24 runs per rank pass. The perf look: no VC starves at VC1; VC2's 4-slot
windows bound VC2 flows inside a mesh; VC1's 4/4 does not bound the VC2 flow across meshes; VC0 across meshes untested.
Regression after the rebase: a new tree, then `sbatch --nodelist=<a free node> fabric_exp/vc2line/vc2l_glx_mm.sbatch`
(default STEPS) and `allvc_report.py <job>/results`.

## Perf: idle and full VCs (VC2 line mode as built; Galaxy, 2026-11-01)

A first look at total router throughput when several VCs are active, and at what idle VCs cost. Tree vc2l_p4d (=
vc2l_p4c, the phase-4 router and host, with the main tree's profiler change), routers at -O3, benchmark mode without
sync (each sender times its own loop; per-flow cycles from the profiler's per-sender lines), 40,000 packets per flow,
256 B, 1 KiB and 4352 B payloads. Cycles per packet in the device clock (1,350 MHz). The wire: ~248 cycles per 4352 B
packet; at smaller sizes "wire est" scales that by payload + 96 B (it ignores per-packet wire overhead, so it is
optimistic). Labels as before; "what binds" only from window A/Bs.

- Jobs (both on bh-glx-120-c04u08, one at a time, queued after the slot-study job 167464 with `--dependency`):
  - 167481, the Galaxy as one 8x4 mesh (`vc2l_glx.sbatch`, steps idle_*; `glx_idle_mesh.yaml`): VC2 off, neighbour
    (idle), line (idle), two ERISCs and one, VC0's depths pinned at 16/7/16 in every configuration
    (TT_FABRIC_EXP_SLOTS / _VC2), repeats of off and line. 8 of 8 steps, 15 of 15 tests each.
  - 167482, the 2x2 split (`vc2l_glx_mm.sbatch`): split_idle_* (`glx_idle_split.yaml`, VC1 compiled in and idle, VC0 and
    VC1 pinned at 16/7/16 and 4/4), full / full_vc1x2 / full_vc2x2 (`glx_full_split.yaml`), repeats. 11 of 11 steps.
  - Slot rows logged in every run, as pinned. Repeats within 0.9 cycles per packet (single mesh) and 0.5% (split). The
    single-mesh script now arms the hang capture too (unused: no hang). Mock-vetted first (every test reaches launch on
    every rank, rows as intended). Reports: `fabric_exp/vc2line/perf_vcs_report.py`; results in
    `fabric_exp/vc2line/results/16748[12]_*`; job directories deleted.

### 1. Idle-VC cost (VC0's cycles per packet; deltas against VC2 off)

One mesh (167481), two ERISCs:

| row | 256 B: off, +nbr, +line | 1 KiB | 4352 B |
| --- | --- | --- | --- |
| 1-hop E/W | 124.9, +10.5, +20.0 | 127.8, +9.5, +17.3 | 252.0, +0.0, +0.0 |
| 1-hop N/S | 125.3, +10.6, +19.6 | 128.1, +10.0, +17.0 | 252.0, +0.0, +0.0 |
| 3-hop E/W | 279.1, +22.9, +30.9 | 287.5, +22.6, +30.0 | 298.8, +34.6, +46.4 |
| 4-hop N/S | 390.6, +18.8, +23.3 | 397.6, +18.3, +24.1 | 403.8, +53.6, +60.0 |
| multicast E:3 | 291.7, +17.7, +26.4 | 299.8, +18.9, +25.8 | 312.8, +55.3, +64.5 |

One mesh, one ERISC: 1-hop E/W 199.2, +37.8, +53.4 (256 B); 251.1, +0.0, +2.4 (4352 B); 1-hop N/S 202.8, +38.6, +54.1;
251.0, +0.0, +6.5; 3-hop, 4-hop and multicast +38 to +45 (neighbour) and +58 to +70 (line) at every size.

The 2x2 split (167482, VC1 compiled in and idle in every column), two ERISCs: 1-hop E/W 141.8, +25.3, +35.5 (256 B);
143.5, +23.7, +33.9 (1 KiB); 252.0, +0.0, +0.0 (4352 B); 1-hop N/S the same within 1; 2-hop 310.4, +21.7, +24.6;
315.4, +21.8, +27.9; 334.2, +47.8, +55.2; the 2D multicast {E:1, S:1} as 1-hop. One ERISC: 1-hop E/W 266.7-267.1 at
every size (router-bound even at 4352 B), +38.6 (neighbour), +64.0 (line); N/S 253, +34, +47; 2-hop 454-492, +40 to
+45, +63 to +65.

Readings:
- VC1 compiled in and idle: VC0 1-hop on the split 141.8 / 143.5 against 124.9 / 127.8 on one mesh (+17 / +16 at 256 B
  / 1 KiB; 0 at 4352 B). Different routers (a 2x2 mesh against an 8x4 mesh's interior), so a cross-topology difference,
  not an A/B.
- The Galaxy wire hides an idle VC only where VC0 is at the wire: 1-hop at 4352 B (two ERISCs; one ERISC on one mesh,
  except +2.4 / +6.5 in line mode). At 256 B and 1 KiB the 1-hop rows are router-bound (125-143 cycles against a wire
  estimate of 20-63) and show the whole cost: neighbour +10 (one mesh) / +25 (split), line +17-20 / +34-36; on one
  ERISC +38 and +53 (one mesh), +38 and +64 (split). Multi-hop and multicast rows show it at every size, most at 4352 B
  (+46 to +65, line, two ERISCs).
- Line mode's idle cost is about twice neighbour mode's (one mesh, small sizes: +17-20 against +10): a forwarding sender
  channel and a forwarding receiver more.
- Against P150 (the lead, 2D E/W 4352 B: +7.7, main V4_RECEIVER_SHAPE §29-§30): P150's row is router-bound at that
  size, the Galaxy's is at the wire; the Galaxy's router-bound rows (256 B, 1 KiB) cost +10 for neighbour mode, of the
  same order. Which ERISC bears the cost was not tested here (P150's probe A/B found E0 for neighbour mode).

### 2. Full VCs and total throughput (the split, two ERISCs; default row VC0 16/7/16, VC1 4/4, VC2 4/4/4)

Totals while the flows overlap: the packets moved before the first flow ends, the others at their alone rates after it
(exact for two flows, a lower bound for three); flows that end later are not starved, they hold fewer slots.

Link L, mesh 1 chip 2 -> chip 3, from the intermesh router's chip: VC0 and VC2 from chip 2's workers, VC1 = mesh 0
chip 3's VC0 flow over the intermesh link, crossing over at chip 2:

| size | VC0 alone | VC2 alone | VC1 alone | VC0 + VC2: VC0, VC2; total | all three: VC0, VC2, VC1; total | wire est |
| ---: | ---: | ---: | ---: | --- | --- | ---: |
| 256 | 177.6 | 448.0 | 467.2 | 239.7, 478.2; 163.3 (8.3 pkt/us) | 327.0, 558.4, 543.3; 161.8 (8.3 pkt/us) | 20 |
| 1024 | 177.5 | 459.1 | 483.9 | 238.6, 487.9; 163.8 (8.2 pkt/us, 8.4 GB/s) | 325.8, 566.1, 553.0; 162.3 (8.5 GB/s) | 62 |
| 4352 | 252.1 | 507.2 | 556.5 | 320.2, 690.1; 252.0 (23.3 GB/s) | 397.4, 807.3, 810.7; >= 274 | 248 |

The intermesh link I, mesh 0 chip 3 -> mesh 1 chip 2 (both VCs' packets delivered at the landing chip): VC0 alone
222.6 / 222.8 / 252.1, VC2 alone 494.8 / 505.9 / 575.3, together 353.0 and 581.1 / 349.4 and 590.5 / 322.1 and 737.2;
totals 229.4 / 229.4 / 251.9 (256 B / 1 KiB / 4352 B).

One chip's links at once (RAll: mesh 1 chip 2 sends VC0 and VC2 east, north and south out of the mesh; VC1 arrives
for chip 3): each link's flows run as in the one-link tests (east: 397 / 807 / 811 at 4352 B, as all three on L; north:
322 / 691, as VC0 + VC2 on L's 320 / 690; south: 323 / 749, as on I); one chip's three links do not slow each other.
The chip moves >= 50.7 GB/s of payload at 4352 B (seven flows over the slowest flow's time).

Readings (measured unless marked):
- At 4352 B several VCs together carry the link at the wire (252), as VC0 alone does; VC2 alone (4 slots) and VC1
  alone (4/4) do not (507, 557).
- At 256 B and 1 KiB the total stays far below the wire: on L one VC 177.6 cycles per packet (7.6 pkt/us), two or three
  VCs ~162-164 (8.3 pkt/us, +9%), with the busy VCs' windows doubled ~139-140 (9.7 pkt/us, +27%); the wire would allow
  ~20-63 (est). On the intermesh router one VC 222.6, two 229.4, two with VC2's windows doubled 230.6: no gain from the
  second VC there.
- So at small sizes the windows cap part of the total (the A/B below), and the router's per-packet and per-pass work
  caps the rest (hypothesis: not isolated, no delay-hook variant on this tree). VC0 alone on L (177.6) equals the split's
  idle-line 1-hop row (177.3): VC0 alone pays VC1's and VC2's idle per-pass work (~+52 over a VC0-only router's 125).
- Shares while both run (VC0 + VC2 on L): with VC0's 16-slot worker against VC2's 4, VC0 moves ~79% of the packets at
  4352 B and ~68% at 256 B (VC2 finishes 27% / 47% of its packets in the overlap); with VC2 at 8/8/8, ~65% / ~55% (VC2
  53% / 82%). Hypothesis: shares follow the windows. Neither VC starves.
- Context, P150 (main jobs 333/340): VC0 alone 134.6, with VC2 busy 163.6, VC2 alone 858 (2 slots) or 240 (8),
  combined ~126 at 4352 B (about the wire). On the Galaxy too the combined link reaches the wire at 4352 B.

### 3. What binds (window A/Bs, job 167482)

| flow (cycles per packet) | default | VC1 8/8 (`16,2,15,8,8,4,4,4`) | VC2 8/8/8 (`16,3,16,4,4,8,8,8`) |
| --- | ---: | ---: | ---: |
| VC2 on L, 256 B / 4352 B | 448.0 / 507.2 | 448.0 / 508.9 | 253.9 / 294.3 |
| VC2 on I, 256 B / 4352 B | 494.8 / 575.3 | 494.5 / 574.2 | 289.5 / 323.4 |
| VC1 on L (VC0 -> VC1), 256 B / 4352 B | 467.2 / 556.5 | 332.2 / 338.2 | 467.7 / 554.8 |
| VC0 on L, 256 B / 4352 B | 177.6 / 252.1 | 186.6 / 252.0 | 177.4 / 252.1 |
| VC0 on I, 256 B / 4352 B | 222.6 / 252.1 | 251.7 / 252.5 | 223.1 / 252.1 |

- VC2 flows: bound by VC2's 4-slot windows (doubling them: -42% to -44%; VC1's row leaves them).
- VC0 flows into VC1: bound by VC1's 4/4 windows (doubling them: -29% at 256 B, -39% at 4352 B; VC2's row leaves
  them). The VC1 x2 row also changed VC0's receiver (15) and forwarding channels (2) on that flow's path, which slow VC0
  flows (VC0 on I +29, on L +9 at 256 B), so the gain is VC1's. This settles phase 4's open question: VC1's 4/4 binds
  VC0 flows across the meshes; VC2 flows across them are bound by VC2's windows first (job 167468).
- Hypothesis (not isolated): the VC1 x2 row's 15-slot VC0 receiver (not a power of 2) or its 2-slot forwarding
  channels cost VC0 1-hop flows at small sizes.
- Not isolated: VC0 1-hop's small-size cap, the intermesh router's ~223, the multi-hop rows' idle-VC cost at 4352 B.

### For the line-VC census (VC2_LINE_1D_VCS.md)

- An idle line VC costs a VC0 flow ~17-20 cycles per packet at small sizes on a two-ERISC router, ~53 on one ERISC,
  nothing on 1-hop at 4352 B; with N line VCs and one busy, the others' idle cost adds (est, if additive).
- Several busy VCs share one router's per-packet capacity: the link's total at small sizes rises by ~9% with a second
  or third VC at today's windows and ~27% with doubled windows; at 4352 B the wire caps it. Native fan-in through VCs
  removes the mux core and its hop, but does not multiply a router's small-packet capacity.

## Line VCs on 1D fabrics, host side (2026-11-02)

Census: VC2_LINE_1D_VCS.md; design (reviewed and approved with its four recommendations) and implementation:
LINE_VCS_HOST_DESIGN.md (§9: as implemented, the checks, what the router side needs). Host only, offline checks only:
`TT_METAL_FABRIC_LINE_VCS=N` (1..3) makes VC0 .. N-1 line VCs on Linear and Ring fabrics; per-VC routing modes
(`builder_config::VcRouting`) with one validation table; new format-neutral named args for v4 (`VC<n>_SERVICED`,
`VC<n>_ROUTES_AS_LINE`, `RX_CH_<r>_FORWARDS_ON_VC`, `NUM_TRANSACTION_IDS_PER_RECEIVER_CHANNEL`,
`SENDER_CH_<i>_IS_WORKER_CHANNEL`); the worker API per VC. Checks: unit tests 73/73; mock captures at N = 1..3 (P150
Linear and Ring, Galaxy ring) pass every arg check (`line_vcs_dump_check.py`); existing configs identical (871/871
images of configs built on both sides; inputs identical in 36 of 38 cases, the other two only log noise). The router does
not compile N >= 2 yet (its gate and its own transaction-ID rule), as expected until its side lands.
