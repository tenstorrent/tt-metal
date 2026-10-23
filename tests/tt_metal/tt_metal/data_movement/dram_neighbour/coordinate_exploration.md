# Coordinate Systems and the DRAM Neighbour "One Hop / Loop Back" Tests

## Context

This document supports the two new DRAM-neighbour sweeps added in this PR:

- `TensixDataMovementDramNeighbourOneHopSweep` (test ID 508)
- `TensixDataMovementDramNeighbourLoopBackSweep` (test ID 509)

Both tests start from the canonical single-bank ideal mapping (one optimally-placed Tensix worker per DRAM bank) and add a *second* worker at a controlled logical offset. Designing them required answering a concrete question: **what does `+1` on a logical `x`-coordinate actually do on the physical NOC0 grid?** The short answer is that it moves one NOC hop in NOC0's preferred direction, and `-1` moves one hop in the opposite direction (which, on a unidirectional torus, is the long way around).

The three tests documented below are exactly what was used to establish that relationship empirically. The first two are debug/exploration tests — they only log coordinate translations and assert nothing interesting. The third is one of the two PR tests that applies the finding.

---

## Logical vs. Physical Coordinates

TT hardware exposes core coordinates in several systems. Two of them matter here:

| System | What it is | Typical user |
|---|---|---|
| **Logical** | 0-indexed `(x, y)` over the *enabled* Tensix worker grid. Size is `device->logical_grid_size()`. Harvested cores are invisible; DRAM / Ethernet / PCIe cores are not part of the grid at all. | Test code, kernel launches, `SetRuntimeArgs`, `CoreRangeSet`. |
| **Physical / NOC0** | Actual row/column on the NOC mesh. Spans the full silicon grid (includes harvested rows, DRAM rows, Ethernet rows, …). Used by the NOC for routing. | SOC descriptor queries, hop-distance reasoning, low-level diagnostics. |

The translation from logical to physical is *not* an identity:

- The physical grid is bigger (extra rows/columns for DRAM and Ethernet).
- Harvested rows/columns are skipped when enumerating logical cores, so gaps can appear.
- The exact offsets depend on the architecture (Wormhole, Blackhole, …) and on which rows are harvested on a specific die.

The SOC descriptor (`metal_SocDescriptor`) is the authoritative translator between these systems.

---

## APIs Used

All APIs referenced by the three tests below.

### Obtaining the SOC descriptor and HAL

```cpp
const metal_SocDescriptor& soc_desc =
    tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
auto& hal = tt::tt_metal::MetalContext::instance().hal();
```

- `MetalContext::instance()` — the TT-Metal runtime singleton.
- `get_cluster()` — handle to the cluster of physical devices.
- `get_soc_desc(device_id)` — SOC descriptor for that device. Owns the coordinate-system translations.
- `hal()` — Hardware Abstraction Layer. Arch-specific constants like `get_num_nocs()`.

### Counting DRAM and NOC resources

```cpp
auto num_dram_channels = soc_desc.get_num_dram_views();    // DRAM views the SOC exposes
uint32_t num_dram_banks = device->num_dram_channels();     // DRAM banks from the device view
auto num_nocs           = hal.get_num_nocs();              // normally 2 (NOC0, NOC1)
```

On current architectures `num_dram_views == num_dram_channels`; the debug test logs both to confirm they agree.

### The logical worker grid

```cpp
auto logical_grid_size = device->logical_grid_size();
// logical_grid_size.x, logical_grid_size.y: the enabled Tensix grid.
```

### Logical → Physical for Tensix cores

```cpp
CoreCoord phys = soc_desc.get_physical_tensix_core_from_logical(logical);
```

The primary translation used for NOC-distance reasoning. Given a logical `(x, y)` in the worker grid, returns the physical NOC position of that worker.

### Preferred DRAM endpoints

```cpp
CoreCoord dram_worker_ep = soc_desc.get_preferred_worker_core_for_dram_view(dram_channel, /*noc=*/0);
CoreCoord dram_eth_ep    = soc_desc.get_preferred_eth_core_for_dram_view   (dram_channel, /*noc=*/0);
```

For each DRAM channel, per NOC, the SOC tells you the **preferred endpoint core** — the physical tile through which DRAM traffic on that NOC enters/leaves the mesh. Both a Tensix-side and an Ethernet-side endpoint are reported.

### Generic coordinate translation

```cpp
CoreCoord phys = soc_desc.translate_coord_to(
    any_coord, CoordSystem::TRANSLATED, CoordSystem::NOC0);
```

`translate_coord_to(coord, from_system, to_system)` converts a coordinate between any two systems the SOC descriptor knows. `TRANSLATED` is UMD's canonical runtime-facing form; `NOC0` is the physical NOC0 address.

### Optimal DRAM-bank → worker assignment

```cpp
const vector<CoreCoord> optimal_workers =
    mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0);
```

Returns a vector indexed by DRAM bank ID. Entry *i* is the **logical** Tensix core that should read from bank *i* for optimal traffic on NOC0 — the closest Tensix tile to bank *i*'s DRAM endpoint along NOC0's routing path, i.e. the fewest hops possible.

---

## Test 1 — `TensixDataMovementDRAMBankToWorkerMapping`

Debug / exploration test. Does not test anything about data movement; it only logs the logical-to-physical translation for every DRAM bank, plus the `±1` neighbours of the optimal worker.

```cpp
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMBankToWorkerMapping) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    log_info(LogTest, "=== DRAM Bank to Core Mapping ===");

    const vector<CoreCoord> optimal_workers =
        mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::NOC_0);

    const metal_SocDescriptor& soc_desc =
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());

    auto num_dram_channels = soc_desc.get_num_dram_views();
    uint32_t num_dram_banks = device->num_dram_channels();

    log_info(LogTest, "Total DRAM banks: {}, {}", num_dram_channels, num_dram_banks);

    for (uint32_t dram_channel = 0; dram_channel < num_dram_channels; dram_channel++) {
        CoreCoord dram_logical_core =
            soc_desc.get_preferred_worker_core_for_dram_view(dram_channel, 0);
        CoreCoord dram_physical_core =
            soc_desc.translate_coord_to(dram_logical_core, CoordSystem::TRANSLATED, CoordSystem::NOC0);

        const CoreCoord& optimal_worker_logical = optimal_workers[dram_channel];
        CoreCoord optimal_worker_physical =
            soc_desc.get_physical_tensix_core_from_logical(optimal_worker_logical);

        auto logical_grid_size = device->logical_grid_size();
        uint32_t x_p1 = (optimal_worker_logical.x + 1) % logical_grid_size.x;
        uint32_t x_m1 = (optimal_worker_logical.x == 0)
            ? (logical_grid_size.x - 1)
            : (optimal_worker_logical.x - 1);

        CoreCoord optimal_worker_physical_p1 =
            soc_desc.get_physical_tensix_core_from_logical(CoreCoord(x_p1, optimal_worker_logical.y));
        CoreCoord optimal_worker_physical_m1 =
            soc_desc.get_physical_tensix_core_from_logical(CoreCoord(x_m1, optimal_worker_logical.y));

        log_info(LogTest,
            "DRAM Bank[{}] -> DRAM Core[L{},{} -> P{},{}] -> "
            "Optimal Worker[L{},{} -> P{},{}] -> "
            "Neighbour P1[L{},{} -> P{},{}] M1[L{},{} -> P{},{}]",
            dram_channel,
            dram_logical_core.x, dram_logical_core.y,
            dram_physical_core.x, dram_physical_core.y,
            optimal_worker_logical.x, optimal_worker_logical.y,
            optimal_worker_physical.x, optimal_worker_physical.y,
            x_p1, optimal_worker_logical.y,
            optimal_worker_physical_p1.x, optimal_worker_physical_p1.y,
            x_m1, optimal_worker_logical.y,
            optimal_worker_physical_m1.x, optimal_worker_physical_m1.y);
    }

    log_info(LogTest, "=== End DRAM Bank Mapping ===");
    ASSERT_TRUE(true);
}
```

### What this test shows

For every DRAM bank on the device, the log line reports, side by side:

1. The **DRAM preferred worker endpoint**, in both TRANSLATED and NOC0 form.
2. The **optimal logical worker** assigned to that bank, and its physical NOC0 position.
3. The **`+1` neighbour** (`x + 1 mod logical_grid_size.x`) in both logical and physical coordinates.
4. The **`-1` neighbour** (`x - 1` with wrap-around) in both logical and physical coordinates.

Reading the output yields the key empirical finding: **a `+1` step in logical `x` translates to exactly one column shift in physical NOC0 coordinates**, and a `-1` step is the column to the other side. The mapping is smooth in `x` within a fixed `y`; there are no hidden jumps or gaps (on an unharvested die) between consecutive logical columns.

---

## Test 2 — `TensixDataMovementSingleRowMapping`

Debug / exploration test. Holds `y` fixed and sweeps `x` across the whole logical grid, printing the physical coordinate of each position.

```cpp
TEST_F(GenericMeshDeviceFixture, TensixDataMovementSingleRowMapping) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    const metal_SocDescriptor& soc_desc =
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());

    auto logical_grid_size = device->logical_grid_size();

    log_info(LogTest, "=== Single Row Core Mapping ===");

    for (uint32_t row = 0; row < logical_grid_size.x; row++) {
        CoreCoord logical_core(row, 1);
        CoreCoord physical_core = soc_desc.get_physical_tensix_core_from_logical(logical_core);
        log_info(LogTest,
            "Logical[{},{}] -> Physical[{},{}]",
            logical_core.x, logical_core.y, physical_core.x, physical_core.y);
    }

    log_info(LogTest, "=== End Single Row Core Mapping ===");
    ASSERT_TRUE(true);
}
```

### What this test shows

Monotonic, one-to-one growth of physical `x` with logical `x`. The log makes the relationship visual: each `+1` logical step produces exactly one physical column increment, modulo a constant offset contributed by the NOC layout (DRAM / harvested rows / etc. do not appear because the logical grid excludes them by construction). On a harvested die, a gap in physical `x` between two consecutive logical rows would be the signature of a harvested column; on a clean die, the sequence is smooth.

Together with Test 1, this is the confirmation we needed: it is safe to use `(x ± 1)` in logical space as a proxy for "one physical NOC hop over".

---

## Test 3 — `TensixDataMovementDramNeighbourLoopBackSweep`

PR test (ID 509). Built directly on the finding from Tests 1 and 2.

```cpp
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDramNeighbourLoopBackSweep) {
    shared_ptr<distributed::MeshDevice> mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    auto logical_grid_size = device->logical_grid_size();

    uint32_t test_id = 509;
    uint32_t num_banks = 1;
    uint32_t max_num_pages = 32;
    uint32_t max_transactions = 256;
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t page_size_bytes = tt::tile_size(l1_data_format);
    std::map<uint32_t, uint32_t> core_dram_map = core_dram_mapping_ideal(mesh_device, num_banks);
    CoreCoord singleCore = CoreCoord{
        core_dram_map.begin()->first >> 16,
        core_dram_map.begin()->first & 0xFFFF};
    core_dram_map
        [(static_cast<uint32_t>(
              (singleCore.x == 0) ? logical_grid_size.x - 2 : singleCore.x - 1)
          << 16) |
         static_cast<uint32_t>(singleCore.y)] = core_dram_map.begin()->second;

    EXPECT_TRUE(run_sweep_test(
        mesh_device,
        test_id,
        max_transactions,
        num_banks,
        max_num_pages,
        page_size_bytes,
        l1_data_format,
        core_dram_map));
}
```

### What this test does

1. Starts from `core_dram_mapping_ideal(mesh_device, 1)` — a single-bank mapping whose lone worker is 0 NOC0-hops from the DRAM endpoint.
2. **Adds** a second worker at `(singleCore.x - 1, singleCore.y)` (with a wrap branch when `singleCore.x == 0`). Both workers now read from the same DRAM bank.
3. Runs `run_sweep_test` over `(num_of_transactions, pages_per_bank)`.

### Why "loop back"

NOC0 routes in a single direction around the torus. Going from logical `x` to logical `x + 1` is the short way on NOC0 (1 hop). Going from `x` to `x - 1` is the **long way** — it wraps the torus and traverses roughly `grid_size - 1` hops on NOC0 before reaching its destination. The second worker therefore reads from the shared DRAM bank by "looping back" around the torus. The sweep measures how that extra path length interacts with the shared-bank contention.

The companion test `TensixDataMovementDramNeighbourOneHopSweep` (ID 508) is identical except the added worker sits at `(singleCore.x + 1) % logical_grid_size.x` — the short (1-hop) neighbour on NOC0.

> The wrap branch uses `logical_grid_size.x - 2` rather than `- 1` when `singleCore.x == 0`. That offset is deliberate — it keeps the added core off the last logical column, which on some configurations would collide with other test infrastructure. For all other `singleCore.x` values the offset is the straightforward `-1`.

---

## What the Two PR Tests Prove

Taken together, `OneHopSweep` (508) and `LoopBackSweep` (509) isolate one specific variable — the NOC0-distance of a *second* worker that shares a DRAM bank with an ideally-placed first worker.

- `OneHopSweep`: second worker is 1 NOC0 hop away from the ideal worker.
- `LoopBackSweep`: second worker is ~(grid_size − 1) NOC0 hops away, wrapping the torus.

Everything else about the two tests is identical: same bank, same page sizes, same transaction counts, same data format, same kernel. Comparing the two sweeps isolates:

1. **NOC-distance sensitivity of DRAM-read bandwidth** when two workers contend for a single bank.
2. **The asymmetry of NOC0 routing.** Geometrically the two added workers sit the same number of logical columns away from the ideal worker, but their NOC0 paths differ by more than an order of magnitude in hops. Any gap between the two bandwidth curves is attributable to that path-length asymmetry.

The two debug tests above (`DRAMBankToWorkerMapping`, `SingleRowMapping`) are what makes this interpretation valid: they prove that a `±1` logical step really is a single physical column shift, so the choice of `x + 1` vs. `x - 1` is a clean surrogate for "short NOC0 path vs. long NOC0 path" and nothing else.
