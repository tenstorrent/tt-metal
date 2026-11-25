# Handoff: host-initiated RDMA on a DRAM core, and DRAM kernel seats

Written 2026-11-25 for a Claude instance picking this work up on a different machine, working with Neel
Nyamagoudar (`nnyamagoudar-TT`). Nothing from the previous session carries over except what is on GitHub and
what is in this file. The companion file `HOST_RDMA_DRISC_PR_67119_DESCRIPTION.md`, next to this one, holds a
ready-to-paste description for PR #67119.

This file lives on the branch `nnyamagoudar/host-rdma-seats-local`, which is a testing branch and will never be
a PR. Do not copy this file onto a PR branch.

---

## 1. Background

### Whose work this is

Neel owns three issues, all children of #66015 ("enable fabric host-initiated RDMA interface"):

| Issue | What it asks for | State |
|---|---|---|
| #66251 | Move host RDMA to sideband compute (DRISC or x280) | **Current focus.** No description; "same coverage as the Tensix version" is the working definition of done |
| #66254 | Tune for 200G and Galaxy: a sweep flow, at least 20 GB/s at 4 KB on the host RDMA channel | Not started |
| #66995 | Host RDMA in the fabric test infrastructure: a stimulus worker kernel with host flow control | Not started |

Anything outside those three is not his to do. Do not propose it.

### Terms

- **TT-Fabric**: moves packets between chips over Ethernet. Each chip's Ethernet cores run a **router** kernel.
- **Host RDMA**: the host describes a memory copy from chip A to chip B as a 64-byte request; a **service
  kernel** on chip A carries it out. The payload never passes through the host. The driving use is KV-cache
  migration between Blackhole Galaxy systems.
- **DRISC / DRAM core**: the small RISC-V core in each Blackhole DRAM tile. A p150 has 8 DRAM banks with 3 DRAM
  tiles each. System firmware owns one per bank (the NoC0 endpoint), leaving 16 that tt-metal can program.
- **NIU mode**: each DRAM tile has one NoC interface per NoC. In **NOC2AXI** mode it forwards DRAM accesses and
  the core cannot initiate NoC traffic. In **stream** mode the core can initiate, and traffic terminates in its
  L1. DRISC firmware sets the mode once at boot. See `tt_metal/hw/inc/experimental/drisc_mode.h`.
- **Seat**: one DRAM core, as the DRISC data-mover service names them: `bank:noc`. A bank's NoC0 seat is its
  NOC1-endpoint core; its NoC1 seat is its free core. 16 seats per p150 chip. Seats are locked across processes
  with `flock` files under `/dev/shm/tt-drisc-seats` (`tt_metal/impl/drisc_service/seat_lock.hpp`).
- **DRISC service**: Siddhant Nadkarni's (`sidnadTT`) separate data-mover service that runs on DRAM cores and is
  shared between processes. It does not use fabric.

### Pull requests and branches

| Ref | What it is | State on 2026-11-25 |
|---|---|---|
| PR #66339 | The original host RDMA on a Tensix core (Sean Nijjar) | Merged, `a973eaa63bc` |
| PR #66978 | Sid: `dram_sender_logical_cores` as a function of a SoC descriptor | Merged, `3c0daec74ff` |
| PR #67011 | Sid: DRISC seat locks | Merged, `fe20b81814e` |
| `snadkarni/drisc_service/3-kernel` … `6-p2p`, `snadkarni/drisc_service_v2` | The rest of Sid's service | Pushed, no PRs |
| PR #67119 | **Neel's port of host RDMA to a DRAM core.** Head `nnyamagoudar/fabric-host-rdma-sideband-compute` | Draft, empty description, no reviews, head `80c8c9cf845` |
| `nnyamagoudar/dram-kernel-seats` | The seat flag, one commit on top of Sid's old `2-seat-locks` branch | Pushed at `4e512f25022`, **no PR**, needs rebasing onto `main` |
| `nnyamagoudar/host-rdma-seats-local` | Testing branch: port + seat locks + flag + test change. This file is here | Pushed |

### How Neel works. Follow these.

- **He designs; you guide.** Do not write real code unless asked. When asked, show the exact diff and get
  sign-off before applying it.
- **Commits and branches.** Do not commit, amend, push or switch branches unless he asks. When he does ask,
  commit under his name with **no Claude attribution** of any kind.
- **Never reset a board** (`tt-smi -r`) on your own. If something hangs, stop and report. Run anything new with
  a timeout so a hang aborts instead of wedging the chips.
- **Re-check the tree before every build or test run** (branch, HEAD, dirty files). He switches branches and
  commits mid-session.
- **Explain, don't quiz.** "Help me understand X" means a direct top-down explanation.
- **Do not claim code is missing after searching only `main`.** Check unmerged branches with
  `git ls-remote --heads origin | grep -i <term>` first. The DRISC service was found that way.
- **Measure on the machine in front of you.** Do not carry performance numbers from one machine to another.
- The GitHub CLI was not logged in on the old machine, so PRs were read through the web and opened by Neel.

### Reference material

- A layered explanation of host RDMA with code excerpts, written for Neel:
  https://claude.ai/artifact/L4Pow1YVEeLAtEZ9t8Coqa (private to his account).
- Blackhole hardware docs: `github.com/tenstorrent/tt-isa-documentation`. There is no Blackhole DRAM-tile page
  and the word "DRISC" does not appear; the NoC pages cover NIU registers, coordinates, alignment and atomics.

---

## 2. The problem and the implementation

### The problem

PR #66339 runs the host RDMA request service on a Tensix worker core, one per chip. #66251 asks for it to move
to a sideband core so the Tensix core is free for compute. DRISC was chosen over x280.

The instruction for this milestone was a **direct port**: whatever the Tensix service supports moves to a DRAM
core unchanged. No redesign around the GDDR DMA engine, no reads, no multi-domain.

A second problem followed from the first. A tt-metal process takes all 16 programmable DRAM cores on a chip
even though the service uses one, which shuts out Sid's DRISC service. The seat flag fixes that.

### How host RDMA works (unchanged from #66339)

Four layers:

1. **A dedicated router channel.** With `TT_METAL_ENABLE_FABRIC_HOST_RDMA=1`, every router gets one extra VC0
   sender channel, treated as a second worker channel with its own slots, connection semaphore and credit path.
2. **A lookup.** `ControlPlane::get_host_rdma_connection_info(node, eth_chan)` returns what a producer needs to
   connect to it.
3. **A shared layout** in the service core's L1 (`host_rdma_service_abi.hpp`): a 64-byte header, eight channel
   entries each with a 64-deep ring of 64-byte requests, and eight staging slots. Host and kernel never call each
   other; each reads what the other wrote. `produced` is written only by the host, `consumed` only by the kernel.
4. **Two programs.** `HostRdmaService` (`tt_metal/fabric/host_rdma/`) places the layout, launches the kernel and
   posts requests. The kernel (`tt_metal/fabric/impl/kernels/host_rdma_service.cpp`) is one non-blocking loop:
   retire acknowledged packets, send packets whose payload has arrived, start new ones. Up to eight packets are
   in flight, each on its own NoC transaction ID.

A packet's life: claim a window slot, build the fabric header in staging, NoC-read the payload in after it,
NoC-write the whole packet into the router, retire when that write is acknowledged. The last packet of a request
also increments a completion counter on the destination chip, which is how the host learns the data landed.

### What the port changed (PR #67119)

The request format, kernel loop, host API and routers are untouched. Eight files changed.

| Area | Tensix | DRAM core | Why |
|---|---|---|---|
| Launch | Mesh workload | `program->impl().compile`, `slow_dispatch::WriteRuntimeArgsToDevice`, `LaunchProgramAsync`, all with `force_slow_dispatch=true`; stop with `WaitProgramDone` | DRAM cores are outside normal dispatch. Copied from the tensor prefetcher |
| Host L1 access | Plain addresses | Add `hal.get_l1_noc_offset(DRAM)` to every read and write | A DRAM tile's L1 sits at a high NoC offset |
| Kernel build | Defaults | `DramConfig{.defines = get_fabric_kernel_defines(), .opt_level = O2}` | DRAM kernels get no fabric defines (builds a 1D header without them) and default to `-Os`, which cost a third of the throughput |
| Routing table | Already in L1 | Kernel copies it once at start-up from Tensix logical (0,0) | The control plane writes it to Tensix and Ethernet cores only |
| Route lookup | 3-argument `fabric_set_unicast_route` | New overload taking a table pointer | The fixed address is unrelated memory on a DRAM core |
| NoC | NoC 0 | The routers' credit NoC, checked at construction | See below |

Two findings shaped it:

- **Credits need no router change.** The router returns credits by writing a plain 32-bit L1 address, which
  cannot carry the DRAM-core L1 offset. But a plain address does land in DRAM-core L1 when it arrives on a NIU in
  stream mode. A probe test (`DramKernelPlainAddrFixture`, kernel `tensix_plain_addr_to_drisc_l1.cpp`) confirmed
  this for a write and an atomic increment on both NoCs. The resulting rule: **the service must run on a DRAM
  core whose NIU is in stream mode on the NoC the routers credit on.** That is NoC 0 with two ERISCs per router
  and NoC 1 with one (`requires_forced_assignment_to_noc1()`). `HostRdmaService` enforces it.
- **The first run sent packets that never arrived.** The service was alive, had consumed the request and had its
  credit back. The route was garbage because the kernel read the routing table from an address that holds
  unrelated memory on a DRAM core. The fix is the start-up copy above.

The only change outside host RDMA is the overload in `tt_metal/fabric/hw/inc/tt_fabric_api.h`. The original form
delegates to it. Router `.text` with and without the change is byte-identical.

`ServiceCore::Dram` is the default. The Tensix path is still in the code, unused, and is to be removed later.

### What the seat flag changed (`nnyamagoudar/dram-kernel-seats`)

`TT_METAL_DRAM_KERNEL_SEATS=3:1,7:0` makes tt-metal take only those seats on every chip. Eleven files, one
commit. The design leans on one fact: `metal_SocDescriptor::get_metal_dram_cores` is the single list every
DRAM-core loop in tt-metal reads, including the lock hook #67011 added.

- `rtoptions`: parses the comma-separated `bank:noc` list. Malformed or repeated seats fail at start-up.
- `dram_sender_topology`: `dram_seat_logical_core` maps a seat to a core (NoC0 seat = NOC1-endpoint core, NoC1
  seat = free core), matching the DRISC service's `bank_seats`.
- `metal_soc_descriptor`, `tt_cluster`, `metal_env`: `restrict_metal_dram_cores` narrows the list. Firmware load,
  reset, watcher, inspector and seat locking follow without further edits.
- `metal_env`: a named seat held by another process is fatal. So is a bank the chip lacks.
- `tt_metal.cpp` (`CreateDramKernel`): with the flag set, a kernel on an unreserved core is fatal.
- With the flag unset, every new path is skipped.

The host RDMA test was also changed to pick the first DRAM core tt-metal holds that satisfies the NoC rule
(commit `736e8793548` on the testing branch). That change does not depend on the flag.

### Things that are easy to get wrong

- `DramKernelFixture` tests skip unless `TT_METAL_SLOW_DISPATCH_MODE=1`. The fabric tests run in fast dispatch in
  CI, so run them without it for parity.
- Device open clears DRAM-core L1 **only** with `TT_METAL_CLEAR_L1=1`. Reset and firmware load always happen.
- NoC atomics are documented for Tensix and Ethernet L1 only. They worked against DRAM-core L1 in stream mode in
  the probe, but keep the completion counter on a Tensix core.
- Inline L1 writes from a DRAM core use the Tensix scratch location, inside the DRAM core's reserved first 64
  bytes (`noc_nonblocking_api.h`, around line 547). It works. Neel chose to leave it for now.
- DRAM kernel text shares a 26 KB window with about 13 KB of firmware. The service is 5,844 bytes.
- The commit hooks reformat files and fail the first commit attempt. Re-add and commit again.
- After merging `main`, the UMD submodule pointer may change: `git submodule update --init tt_metal/third_party/umd`.
  A stale installed runtime breaks kernel JIT; `ninja -C build install` fixes it.

---

## 3. Plan and status

### Verification already done

All on the old machine: eight Blackhole p150b in a 2x4 mesh, firmware bundle 19.16.1.

For the port, at `80c8c9cf845` (based on `main` at `fb5c1d54e8b`):

| Check | Result |
|---|---|
| `HostRdmaFabric2DFixture.*`, slow and fast dispatch | 4 of 4 each |
| Same, five repeated runs | 20 of 20 |
| Same, single-ERISC routers (`TT_METAL_DISABLE_MULTI_AERISC=1`) | 4 of 4 |
| `Fabric2D*Fixture.*` | 40 passed, 13 skipped for needing more hops than 2x4 |
| `FabricTrafficGeneratorKernelIntegrationTest.*` | 1 of 1 |
| `Fabric1D*Fixture.*` | 27 passed, 7 skipped |
| `*ChannelTrimming*`, both ERISC modes | 63 passed, 3 skipped, each |
| Router `.text` against `main`, feature off | identical, 128 binaries |
| Bandwidth, single hop, 64 KB | 9.21 GB/s on a DRAM core, 9.27 GB/s on Tensix |

For the flag, on the testing branch at `736e8793548`:

| Check | Result |
|---|---|
| Sid's seat tests (`MetalEnvSeatLocks.*`, `SeatLockFiles.*`) | 10 of 10 |
| New `MetalEnvDramKernelSeats.*` | 5 of 5 |
| Host RDMA with seats unset, `0:1`, `7:0`, `7:0,7:1`, `3:1,7:0` | 4 of 4 each |
| Host RDMA, single-ERISC, seat `7:1` | 4 of 4 |
| Host RDMA, single-ERISC, seat `7:0` | Refused as intended (wrong NoC) |

Two tests failed on that machine with and without the flag: `DramKernelFixture.DramKernelDRISCReadFromDRAMMcastToTensix`
and `DramKernelDRISCNocModeStress`. Whether they fail on plain `main` was not determined.

### Steps

| # | Step | Status |
|---|---|---|
| 1 | Port the service to a DRAM core | **Done.** In PR #67119 |
| 2 | Match the Tensix version's local coverage | **Done** on the old machine, see above |
| 3 | Implement the seat flag | **Done.** On `nnyamagoudar/dram-kernel-seats`, no PR |
| 4 | Bring PR #67119 up to date with `main` | **Not done on GitHub.** See below |
| 5 | Fill in PR #67119's description | **Drafted** in the companion file. Neel pastes it |
| 6 | Rebase the seat flag onto `main` and open its PR | **Not started** |
| 7 | Run on Galaxy (32 chips) | **Not started** |
| 8 | CI on `bh_loudbox` and `bh_quietbox_2`; Wormhole and simulator builds | **Not started.** Happens when the PRs are pushed and marked ready |
| 9 | Remove the unused Tensix path | **Not started.** Neel's stated intent |

### Step 4 in detail: the state that did not make it to GitHub

On the old machine the port branch was merged with `main` at `22e8a7c341a` (clean, no conflicts) and given the
seat-aware test commit. Those two commits were **never built, never run and never pushed**, because the machine
lost its build toolchain and went in for servicing. They do not exist anywhere else. Recreate them:

```
git fetch origin
git checkout nnyamagoudar/fabric-host-rdma-sideband-compute
git merge origin/main
git submodule update --init tt_metal/third_party/umd
git cherry-pick 736e8793548        # host_rdma test: run the service on a DRAM core metal takes
./build_metal.sh --build-metal-tests --enable-ccache
```

Then run the CI job and the seat tests before pushing:

```
export TT_METAL_HOME=$PWD
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="FabricTrafficGeneratorKernelIntegrationTest.*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="HostRdmaFabric2DFixture.*"
./build/test/tt_metal/unit_tests_context --gtest_filter="MetalEnvSeatLocks.*:SeatLockFiles.*"
```

What to watch for: `main` reorganised the public tt-metalium headers (#67012). The merge rewrote the include
paths in the host RDMA files without conflict, but nothing has compiled them. `main` now also contains the seat
locks, so a tt-metal process locks every DRAM core at start-up. The port was tested against that on the testing
branch and passed.

Merge, do not rebase, so the PR needs no force-push. Ask Neel before pushing.

### Step 6 in detail

`nnyamagoudar/dram-kernel-seats` sits on Sid's pre-merge branch. Now that #67011 is in `main`, take the single
flag commit (`4e512f25022`, the same change as `f6a8062b653` on the testing branch) onto a branch from current
`main`. It has not been built on its own base. The five new tests are in
`tests/tt_metal/tt_metal/context/test_metal_env_api.cpp`.

### Open questions for Neel

1. Does the seat flag stay a separate PR (the agreed plan), or get folded into #67119?
2. When two seats are named, is one host RDMA service per chip still the intent? The test uses one.
3. Has Sid started his own version of the flag? His note read "with your env flag … we'll add an exception",
   which could mean either side writes it. Neel decided building on Sid's branch was right; whether he has
   spoken to Sid is unknown.
4. Are DRAM programmable cores on for the CI machines? The host RDMA test skips when they are off, so the job
   would pass without running it.
5. Does the probe test (`DramKernelPlainAddrFixture`) stay in the PR? He said "keep for now".
