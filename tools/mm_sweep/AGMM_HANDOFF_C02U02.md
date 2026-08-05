# AGMM handoff → bh-glx-120-c02u02

Continuation of the Regime-A fused all-gather matmul (AGMM) work. Read this, then
`tools/mm_sweep/AGMM_DIRECT_L1_DESIGN.md` (the design + rationale) and
`tools/mm_sweep/REGIME_A_AGMM_DESIGN_SPEC.md` (the contract).

## 0. Why you are on this machine

The previous host (`bh-glx-120-b10u08`) has a **dead cross-tray ethernet link** (Tray 2 N3 ↔ Tray 4 N3,
eth channels 0-1 on chips 10 and 18). That breaks the X ring, so `FABRIC_1D_RING` cannot map and every
ring test fails. Four Galaxy resets produced an identical result, retrain count 0 — it is physical, not a
training flake. Line topology still works there.

`c02u02` routes all rings. Verified discriminator (identical binary, both hosts):

| test | b10u08 | c02u02 |
|---|---|---|
| `MeshGraphValidation.TestSingleGalaxyTorus{XY,X,Y}` | OK / OK / OK | OK / OK / OK |
| `RoutingTableValidation.TestSingleGalaxyTorusXY` | **FAILED** | **OK** |
| `RoutingTableValidation.TestSingleGalaxyTorusX`  | **FAILED** | **OK** |
| `RoutingTableValidation.TestSingleGalaxyTorusY`  | OK | OK |

`MeshGraphValidation` only parses descriptors (no hardware) — it passes on a broken system, so it proves
nothing. `RoutingTableValidation` builds routing against the real topology; that is the one that matters.

NOTE: c02u02 is not pristine either — chip 3 (Tray 1 N4) has 6/8 links up. It just does not sit on a
required wrap. Re-run the check below if ring behaviour ever looks strange.

## 1. Environment

`/data` is NFS-shared, so the worktree and the existing build are already here — **no rebuild needed
unless you change C++**:

    WT=/data/cglagovich/tt-metal/.claude/worktrees/splendid-questing-cookie

Branch `cglagovich/regime-a-ltxflux-opt` @ `92f05a8793a`. C++ test binaries run **natively** over NFS
(no container). Only Python needs one.

### Docker

    docker run -it --name agmm \
      --user $(id -u):$(id -g) \
      -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
      -v /dev/hugepages-1G:/dev/hugepages-1G \
      --device /dev/tenstorrent \
      -v /data/cglagovich:/data/cglagovich \
      -v /home/cglagovich:/home/cglagovich \
      -w $WT \
      ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest /bin/bash

### Build (only if C++ changed)

    docker exec -u $(id -u):$(id -g) -w $WT agmm ninja -C build_Release ttnn

**Then ALWAYS redeploy the .so** — ninja links `build_Release/ttnn/` but the runtime loads elsewhere, so
skipping this silently runs stale code:

    cp -p build_Release/ttnn/_ttnncpp.so build_Release/lib/_ttnncpp.so
    cp -p build_Release/ttnn/_ttnn.so    ttnn/ttnn/_ttnn.so

(A full `./build_metal.sh --build-all --enable-ccache` does this itself via its install step; every
incremental ninja after it re-arms the trap.)

### Python env — VERIFIED

The image pre-sets `PYTHON_ENV_DIR=/opt/venv`, which is root-owned, so a plain `./create_venv.sh` fails
with *"Environment directory exists but is not writable"* — **and exits 0 while printing the error**, so
it looks like it succeeded. Override the dir:

    docker exec -u $(id -u):$(id -g) -w $WT agmm ./create_venv.sh --env-dir $WT/python_env

**This works** — it resolves both known gaps (`tracy`, `graphviz`) and needs no `pip` of its own. Two
things to know:

- It takes **~25 minutes** over NFS and prints nothing for long stretches. Run it detached and wait it
  out; it is not hung. Do NOT poll for it with
  `pgrep -f "create_venv.sh --env-dir"` from a shell whose own command line contains that string — the
  pattern matches the poller and the wait never ends (cost ~30 min here).
- The venv's `bin/python3` symlink points at the **container's** interpreter, so it does not resolve on the
  host. Anything needing it — including `git commit`, whose `pre-commit` hook is a Python script — must run
  **inside** the container:

      docker exec -u $(id -u):$(id -g) -w $WT \
        -e PATH=$WT/python_env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        agmm git commit -F <msgfile>

## 2. Running the AGMM tests

**The PYTHONPATH is not the obvious one.** There is no `$WT/ttnn/__init__.py` — the package is at
`$WT/ttnn/ttnn` — so `PYTHONPATH=$WT` supplies no `ttnn` at all, and a PEP-660 editable finder in
`/opt/venv` (hard-wired to the **`resilient-marinating-piglet` worktree**) wins by default. That silently
runs a *different worktree's code*; it cost a full bogus test run and presents as an ordinary failure,
with the only tell a foreign path deep in a 60-frame traceback.

    export TT_METAL_HOME=$WT
    export PYTHONPATH=$WT/ttnn:$WT/tools:$WT
    export ARCH_NAME=blackhole

**Verify before trusting any run:**

    python3 -c "import ttnn,ttnn._ttnn as c; print(ttnn.__file__, c.__file__)"
    # both paths MUST contain splendid-questing-cookie

Then:

    # fused path (Phase 1, DRAM-staged) — the current default under test
    TT_AGMM_FUSED_GATHER=1 python3 -m pytest \
      tests/ttnn/unit_tests/operations/matmul/test_all_gather_regime_a_matmul_async.py -q --timeout=900

40 tests (tp4/tp8 × ring/line × shapes). Without `TT_AGMM_FUSED_GATHER=1` you get the Phase-0
composition (all_gather_async + single-chip matmul), which is the correctness oracle.

Fabric/link health check (native, no container, but **cwd must be the repo root** — `TT_METAL_HOME`
alone is not enough, `rtoptions` needs the cwd):

    cd $WT && ./build_Release/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=*SingleGalaxyTorus*
    cd $WT && ./build_Release/test/tt_metal/tt_fabric/test_system_health --gtest_filter=Cluster.ReportSystemHealth

## 3. Working-tree state — DONE, both changes are committed

    8ae5ebcff58  revert of the dedicated gather cores (verified 40/40 on ring AND line, tp4 + tp8)
    e986252b4a9  Phase 2 direct-L1 (28/40 under TT_AGMM_DIRECT_L1=1; 12 refused by scope, 0 failures)

Phase 2 measured **143.4 -> 120.7 us** on medium/tp4/ring/2-link. It does NOT reach the 86.9 us gate; the
remaining time and the next levers are in `AGMM_DIRECT_L1_DESIGN.md` under "Measured". The benchmark that
produced those numbers is `tools/mm_sweep/picker_gen/agmm_bench_worker.py` (four spec baselines +
`overlap_efficiency`, device makespan from the tracy device profiler).

The rest of this section describes the state as HANDED OVER, kept for context:

### Original working-tree state (all UNCOMMITTED at handover)

    M ttnn/.../all_gather_regime_a_matmul_async_program_factory.{cpp,hpp}
    M ttnn/.../device/kernels/in0_ring_reduce_writer.cpp
    ? tools/mm_sweep/AGMM_DIRECT_L1_DESIGN.md
    ? tools/mm_sweep/AGMM_HANDOFF_C02U02.md

Two independent changes:

1. **Revert of `e43eb6060b6`** (dedicated gather cores) — the user wants the all-gather handled by the
   same cores as the compute grid. Masters are back to ring group `p == 0`. Builds clean; **verified good
   on line topology** (`small-line-tp4` passes with real compute). The original commit measured
   149.9 → 149.4 us, i.e. it bought nothing, so this loses no performance.
2. **`TT_AGMM_DIRECT_L1` scaffolding** — env gate + `Ns>1` refusal + `DIRECT_L1` writer define.
   Compiles; no-op when unset. The kernel work is NOT started.

~~**First thing to do: re-run the full 40 on ring**~~ — done: 40/40, and the revert is committed
separately as `8ae5ebcff58`.

## 4. The actual task: Phase 2 direct-L1

Full design in `AGMM_DIRECT_L1_DESIGN.md`. Summary of the decisions already agreed with the user:

- **Keep the ring.** The user finds ring delivery effective; do not replace it with multicast.
- **No credits.** cb0 is already sized for the worker's complete gathered K/Pk slice, so a remote stripe
  written into its final cb0 slot is never overwritten. No bounded window, no flow-control handshake.
- **Fabric clients are all consumer cores**, not 8 masters: core `(kk,p)` sends its own cb0 slot 0 to
  core `(kk,p)` on the neighbour, so relay source == consume destination and no relay buffer is needed.
- **Ns>1 refused initially** (would duplicate fabric copies); DRAM staging still covers it.
- **Gated behind `TT_AGMM_DIRECT_L1=1`**, DRAM staging stays default and A/B oracle.

Why it matters: this shape is DRAM-bandwidth-bound, and Phase 1's staging round-trip costs 5.25 MB/device,
putting its **roofline at 97.5 us — above the 91.3 us gate**. Phase 1 cannot pass however well it is
scheduled. Direct-L1 keeps the activation out of DRAM: 28.2 MB → 77.6 us.

**Expectation management:** direct-L1 fixes the byte count, not the ~37 us dependency stall. Expect
~93 us, not 78. The stall is a structural property of the ring (see "the ring bound" in the design doc —
`makespan >= T_ready_max + G*delta`, independent of stripe permutation). That is also why the
`scratchpad/arrival_order_*.patch` idea was dropped: it is worth at most one hop.

## 5. Traps that have already cost real time

- **Wrong-worktree import** and **stale .so** — see §1/§2. Both silent.
- **Mux v2 self-terminates by counting `close()` calls** against a compile-time channel count. A mismatch
  is a *hang*, not an error. Has caused silent hangs three separate times.
- **Multicast rectangles** must get corners in the issuing NOC's traversal order; wrong order aims at an
  inverted rectangle nobody receives. Invisible at Pk=1 (1x1 rectangles make the swap a no-op).
- **Two different partitions of M** with nothing synchronising them caused both the PCC 0.984 and the
  PCC 0.79-0.98 rounds.
- **`override_runtime_arguments`**: anything not patched there is correct on invocation 1 and stale from
  invocation 2 on — the hardest failure to attribute.
- **Hardcoded compile-time accessor indices**: adding CT args displaced every `TensorAccessorArgs`, built
  cleanly, and silently failed all 40 tests on PCC.
- **Credit atomics need NOC0 coords** (`my_x[0]`, not `my_x[noc_index]`) — invisible on Blackhole, a hang
  on Wormhole.
- **Widening the fabric client set past 8 masters hits two mux ceilings**, both of which fire as
  program-creation errors only because they are now checked: **64 channels** per mux (one stream register
  each, a hard per-worker limit) and the mux's **L1 map** (channels x buffers x 4 KiB against 1.5 MB). A mux
  binds exactly ONE link, so `num_links` is the only way to add mux cores — you cannot spread one
  direction's clients over more muxes on the same link. See `AGMM_DIRECT_L1_DESIGN.md`, "Scope limits".
- **`run host ID` in the device-profiler CSV is per DEVICE, not per mesh op.** A tp=4 op is 4 runids. Group
  by runid and max over cores — what the single-chip worker does — and you silently time ONE device instead
  of the makespan. Setup work (`from_torch`, zeroed persistent buffers) also lands in the same CSV ahead of
  the measurement loop, so anchor the demux on the tail with a known ops-per-call.
- **Test fixture leaks fabric config** (known bug, worth fixing): in `open_cluster`, `set_fabric_config`
  and `open_mesh_device` sit *outside* the `try:`, so a failed open never resets to `DISABLED` and every
  later test dies with a bogus `Tried to override previous value of fabric config`. One real failure
  becomes 40. This is why "40 failed in 12s" means *one* thing broke, not forty.

## 6. Measurements to report (per the spec)

Four baselines at identical shapes/configs: single-chip full-K matmul, standalone all-gather, unfused
all-gather+matmul, and fused AGMM — plus `overlap_efficiency = max(T_mm, T_ag) / T_fused`, and fabric +
DRAM bytes. Use device FW duration from the tracy device profiler; **host dispatch time is not evidence
of overlap**. Galaxy RevB DRAM peak is **448 GB/s** (not the p150's 512).

`tools/mm_sweep/picker_gen/agmm_bench_worker.py` produces all of this. Reference numbers, medium
(256x5120x2560) / tp4 / ring / 2 links — the first column is what was handed over (b10u08), the second is
what this box measures with that worker:

    baseline                        handed over    c02u02
    matmul alone                       ~83.0        77.8 (1 chip) / 79.0 (on the mesh)
    standalone all_gather               41.3        36.6
    Phase-0 (unfused)                  125.3       115.1
    fused Phase 1 (DRAM-staged)        149.4       143.4
    fused Phase 2 (direct-L1)             --       120.7
    Phase 2, TT_AGMM_ABLATE=nowait        --       101.7

The two columns are consistently ~5% apart (different box), so do not mix them in one comparison. Note
`TT_AGMM_ABLATE=nowait` on the STAGED path measures 153.8 us here, i.e. SLOWER than the unablated 143.4 —
the opposite of the handed-over 113.2. Unexplained, and not chased; it makes the staged path's stall split
unusable on this box. The direct-L1 nowait number behaves normally.
