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

### Python env — UNVERIFIED, this is the open question

The image pre-sets `PYTHON_ENV_DIR=/opt/venv`, which is root-owned, so a plain `./create_venv.sh` fails
with *"Environment directory exists but is not writable"* — **and exits 0 while printing the error**, so
it looks like it succeeded. Override the dir:

    docker exec -u $(id -u):$(id -g) -w $WT agmm ./create_venv.sh --env-dir $WT/python_env

Not yet tested — verify it before trusting a test run. On b10u08 no venv was needed at all (an 8-day-old
container had the deps), but a *fresh* container is missing them. Known gaps found so far:
`tracy` (fixed by putting `$WT/tools` on PYTHONPATH) and then `graphviz` (blocked — the image venv has
no `pip`). If `--env-dir` works, both should resolve.

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

## 3. Working-tree state (all UNCOMMITTED)

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

**First thing to do: re-run the full 40 on ring** to confirm the revert is clean there too, then commit
it separately from the direct-L1 work.

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
- **Test fixture leaks fabric config** (known bug, worth fixing): in `open_cluster`, `set_fabric_config`
  and `open_mesh_device` sit *outside* the `try:`, so a failed open never resets to `DISABLED` and every
  later test dies with a bogus `Tried to override previous value of fabric config`. One real failure
  becomes 40. This is why "40 failed in 12s" means *one* thing broke, not forty.

## 6. Measurements to report (per the spec)

Four baselines at identical shapes/configs: single-chip full-K matmul, standalone all-gather, unfused
all-gather+matmul, and fused AGMM — plus `overlap_efficiency = max(T_mm, T_ag) / T_fused`, and fabric +
DRAM bytes. Use device FW duration from the tracy device profiler; **host dispatch time is not evidence
of overlap**. Galaxy RevB DRAM peak is **448 GB/s** (not the p150's 512).

Reference numbers, medium (256x5120x2560) / tp4 / ring / 2 links:

    matmul alone     ~83.0 us      standalone all_gather   41.3 us
    Phase-0 (unfused) 125.3 us     fused (Phase 1)        149.4 us
    TT_AGMM_ABLATE=nowait 113.2 us  (delta vs full = pure dependency stall)
