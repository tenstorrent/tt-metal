# X280 (Blackhole L2CPU)

This tree runs work on the SiFive X280 RISC-V cores inside the L2CPU tiles of a Tenstorrent
Blackhole chip, beside Metal's own workloads. It holds a C++ host runtime that binds the tiles to
a Metal mesh, the bare-metal firmware images it runs, and a characterization harness with six
benchmark families measuring how the X280 exchanges data with Tensix, Ethernet, DRAM and other
X280 cores.

## 1. What the X280 is

A Blackhole chip exposes **four L2CPU tiles**, each containing a SiFive
**X280 cluster of 4 harts** (16 RISC-V cores total per chip). Each
L2CPU tile is a NOC endpoint with its own LIM SRAM, L3 cache, DDR
front port, and three outbound NOC paths.

| Resource | Per-tile | Notes |
|---|---|---|
| X280 harts | 4 | rv64gcv (vector unit, VLEN=512) |
| L3 cache / LIM | 2 MB | LIM = L3 way 0 exposed as SRAM (WayEnable=0 default) |
| DDR via Front Port | shared | Cache-coherent path through L2CPU |
| NOC TLB windows | 224 × 2 MB + 32 × 128 GB | Outbound translation |
| DMA engine | 1 (Synopsys DW_ahb_dmac) | At periph offset `0x2FF80000` |

**L2CPU NOC coordinates on Blackhole:**

| Index | (NOC X, Y) |
|---|---|
| 0 | (8, 3) |
| 1 | (8, 9) |
| 2 | (8, 5) |
| 3 | (8, 7) |

**Memory map (X280 view):**

| Range | Region |
|---|---|
| `0x0000_0000 – 0x0000_FFFF` | Boot trampoline / vectors |
| `0x0200_0000 – 0x020F_FFFF` | CLINT (mtime/msip) |
| `0x0800_0000 – 0x081F_FFFF` | LIM SRAM (when WayEnable=0) |
| `0x4000_3000_0000 +` | DDR (cache-coherent) |
| `0x2FF8_0000` | DMA engine registers |
| `0xFFFFF7FEFFF10000` | L2CPU local register block (reset vectors, etc.) |

**Clock domains:**

| Clock | Source | Frequency | Drives |
|---|---|---|---|
| AICLK | ARC PLL, ramped via `arc_msg(0x52)` | ~1.35 GHz | NOC fabric, Tensix BRISC, NIU |
| X280 PLL | PLL4 (programmed by `X280Runtime`) | 1750 MHz typ. | X280 cores, `rdcycle` |
| l2sysclk | X280 PLL / 2 | 875 MHz | L2CPU AXI fabric, DMA engine |
| mtime | reference | 50 MHz | CLINT timer |

The host side opens a Metal mesh and then `X280Runtime::open` on it; the runtime prepares the
chip (LIM ECC priming, the cache/LIM partition, PLL4), boots a resident idle image on each tile
and runs active images through it ([docs/runtime.md](docs/runtime.md)). Images are freestanding
C or C++ programs built with tt-metal's SFPI compiler when tt-metal builds
([docs/device_images.md](docs/device_images.md)).

## 2. Build, boot and run a benchmark

The X280 tree builds only in a tt-metal build configured with the tests and
`TT_METAL_BUILD_X280`. The images are compiled at build time with the SFPI compiler tt-metal
fetches (`runtime/sfpi`, or `/opt/tenstorrent/sfpi` with `--use-system-sfpi`); no other RISC-V
toolchain is involved.

```bash
./build_metal.sh --build-metal-tests --configure-only
cmake -B build -DTT_METAL_BUILD_X280=ON
ninja -C build x280_images x280_abi_headers x280_runtime_host_tests x280_runtime_coordinate_tests \
  x280_bench_host_tests x280_bench_device_tests x280_bench_release_selector_tests \
  x280_transport_bandwidth_tensix_to_x280 x280_transport_bandwidth_x280_to_tensix \
  x280_transport_bandwidth_erisc_to_x280 \
  x280_p2p_latency_x280_same_tile x280_p2p_latency_x280_peer_tile x280_p2p_latency_tensix_peer_tile \
  x280_p2p_latency_erisc_peer_tile x280_p2p_latency_drisc_peer_tile \
  x280_pipeline_stage_route_compute x280_pipeline_stage_route_forward \
  x280_core_sync_director x280_core_sync_characterization \
  x280_packet_injection_tensix_to_x280 x280_packet_injection_erisc_to_x280_header \
  x280_eth_credit
ctest --test-dir build -R x280      # image placement, pinned-input guards, header refusals, host, offline and mock-cluster tests, results tools, STOP-poll scan; no device
```

Images land in `build/x280/` and executables in `build/test/tt_metal/x280/`. Every executable
takes its images from `build/x280/`; an `X280_*_FW_IMAGE` variable names another file.

**Prepare a device once.** LIM ECC priming commits the chip's cache/LIM partition and needs an
ASIC reset afterwards; it persists across later resets, so it is done once per device:

```bash
X280_ALLOW_LIM_ECC_PRIME=1 TT_VISIBLE_DEVICES=0 build/test/tt_metal/x280/x280_prime_test
env -u TT_VISIBLE_DEVICES tt-smi -r 0
```

**Run a benchmark.** Reset the device, then run one executable:

```bash
env -u TT_VISIBLE_DEVICES tt-smi -r 0
TT_VISIBLE_DEVICES=0 TT_METAL_RUNTIME_ROOT="$PWD" build/test/tt_metal/x280/x280_p2p_latency_x280_same_tile
```

- The benchmark harness boots resident idle itself on a chip whose L2CPUs are all held in
  reset, one batch per chip-reset lifetime, and sets PLL4 to `X280_PLL_MHZ` (default 1750)
  ([docs/harness.md](docs/harness.md#running-benchmarks)). A chip with only some L2CPUs out of
  reset skips until the next reset.
- The pipeline_stage executables have the runtime prepare their pinned input, once per reset of
  the device; later runs reuse it. Every benchmark opens at 14 LIM ways
  ([docs/harness.md](docs/harness.md#running-benchmarks)).
- `x280_eth_credit` uses chips 0 and 1 over their Ethernet link; run it with every card visible
  ([docs/benchmarks/eth_credit.md](docs/benchmarks/eth_credit.md#running-it)).
- Executables that launch Metal kernels on idle Ethernet or DRAM cores link slow dispatch in;
  nothing needs setting.
- Run with Metal's watcher off: its host NoC sanitizer rejects the runtime's ARC and L2CPU
  accesses.
- Scope every device invocation with `TT_VISIBLE_DEVICES` (UMD chip ids), and clear it for
  `tt-smi -r`, which with the variable inherited resets nothing and still exits 0.
  `/dev/tenstorrent/N` takes PCIe ids; UMD prints the mapping when it opens.

Each benchmark writes JSONL records under `X280_CHARACTERIZATION_OUTPUT_DIR`, else
`$TT_METAL_HOME/characterization`, else `./characterization`; `tools/tabulate_suites.py` turns
them into tables and `tools/run_characterization.py` collects a campaign from
`characterization_manifest.json` ([docs/measurement.md](docs/measurement.md#collection-and-reproduction)).

## 3. Documents

| Document | Holds |
|---|---|
| [docs/runtime.md](docs/runtime.md) | the host runtime: production use, open outcomes, priming and LIM shape, prepared pipeline inputs, ownership, sessions, reset semantics |
| [docs/device_images.md](docs/device_images.md) | building and writing images, the active-image contract, NoC access from firmware, the images and their build checks, the atomics micro-benchmark, constants |
| [docs/harness.md](docs/harness.md) | the characterization harness: architecture, targets, lifecycle, regions, the runtime as the harness uses it, measurement discipline, safety invariants, decisions, running benchmarks |
| [docs/measurement.md](docs/measurement.md) | the record contract, counters and clock evidence, unconfirmed hardware assumptions, collection and tabulation |
| [docs/benchmarks/design.md](docs/benchmarks/design.md) | the design shared by the families: conventions, naming, preconditions, and the p2p_latency, transport_bandwidth and pipeline_stage designs |
| [docs/benchmarks/transport_bandwidth.md](docs/benchmarks/transport_bandwidth.md) | transport_bandwidth: its two members and how to run them |
| [docs/benchmarks/p2p_latency.md](docs/benchmarks/p2p_latency.md) | p2p_latency: tile placement, the Metal peers, running it |
| [docs/benchmarks/pipeline_stage.md](docs/benchmarks/pipeline_stage.md) | pipeline_stage: routing lookup and forwarding micro-ops over the prepared input |
| [docs/benchmarks/core_sync.md](docs/benchmarks/core_sync.md) | core_sync: the cross-core barrier, its executables and its release characterization |
| [docs/benchmarks/packet_injection.md](docs/benchmarks/packet_injection.md) | packet_injection: Tensix and ERISC producers into an X280 consumer, protocol, layout, running it |
| [docs/benchmarks/eth_credit.md](docs/benchmarks/eth_credit.md) | eth_credit: credit flow control across a direct Ethernet link |
| [docs/results.md](docs/results.md) | where the published results of every family are |

## 4. Results

Measured on a Blackhole p150b at X280 PLL 1750 MHz with images built by gcc 15.2. The published
results are described in [docs/results.md](docs/results.md).

- **p2p_latency.** Two X280 harts on one tile round-trip a 4-byte token in 140 ns (7 `mtime`
  ticks) with a 40 ns turnaround. Across tiles every non-posted store crosses one full 12-hop
  lap whatever the placement, 260-340 ns round trip. To a Tensix, idle-ERISC or DRISC peer the
  X280 sees 360-420 ns.
- **transport_bandwidth.** X280 -> Tensix u32 posted stores from registers run at 1.800 GB/s,
  3.89 cycles per `sw`. Tensix -> X280 is Tensix-issue-bound at 17.8 ns per write through 256 B
  (14.4 GB/s), then the X280 sink holds about 17.5 GB/s from 384 B to 16 KiB at 1750 MHz; below
  about 1.44 GHz the sink binds at 256 B too.
- **pipeline_stage.** A routing-table lookup costs about 74 X280 cycles per item and a forward
  decision 42; recorded on the 4,160-byte prepared region, before it grew to 4,288 bytes.
- **core_sync.** An X280 director's multicast release spreads 16 Tensix workers' exits over
  about 122 debug-timer ticks, against about 11,600 for unicast.
- **packet_injection.** One Tensix producer sustains about 1.38 M headers/s at one credit per
  header; at C=8, K=16 the X280 consumer plateaus at about 4.9 M headers/s from Tensix and
  about 4.8 M from four or more idle ERISCs, bound by X280 service.
- **eth_credit.** Each X280 forwarder moves 0.30-0.35 M packets/s across the Ethernet link,
  flat in slot size, bound by its header write into the ERISC; 5.8 GB/s per forwarder at 16 KiB
  slots.
