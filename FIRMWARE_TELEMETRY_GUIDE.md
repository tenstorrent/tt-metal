# Firmware Telemetry Initialization Guide

## Objective

Populate fabric telemetry `static_info` fields and enable bandwidth telemetry in ERISC firmware.

## Context

Bandwidth telemetry metrics exist in tt-telemetry but don't appear in output because:
1. Firmware doesn't populate `static_info` fields (mesh_id, device_id, direction, fabric_config)
2. Firmware doesn't set `supported_stats` bitmask to enable dynamic_info
3. L1 telemetry structures aren't zeroed on initialization

Without these, `dynamic_info` is unavailable and bandwidth calculations can't run.

## Branch Setup

```bash
cd /path/to/tt-metal
git checkout main
git pull
git checkout -b kkfernandez/fabric-telemetry-firmware-init
```

## Files to Modify

### 1. Telemetry Structure Definition
**Location:** `tt_metal/hw/inc/fabric_telemetry_msgs.h`

This defines the structures. Already correct, but review:
- `struct StaticInfo` - has mesh_id, device_id, direction, fabric_config, supported_stats
- `enum DynamicStatistics` - defines BANDWIDTH = 0x02 bitmask

### 2. ERISC Firmware Implementation

**Find ERISC firmware files:**
```bash
find tt_metal/hw/firmware -name "*erisc*.cc" -o -name "*erisc*.cpp"
```

Likely locations:
- `tt_metal/hw/firmware/src/` (architecture-specific)
- Look for initialization routines where telemetry structures are set up

**Key functions to find:**
- ERISC initialization/main
- Fabric router setup
- Telemetry structure initialization (if exists)

### 3. Control Plane / Fabric Context

**Location:** Fabric control plane code that knows mesh topology

```bash
grep -r "mesh_id\|device_id" tt_metal --include="*.cpp" --include="*.hpp" | grep -i fabric | head -20
```

The control plane already has this information - need to pass it to firmware.

## Implementation Steps

### Step 1: Find Where Telemetry Structures Live in Firmware

```bash
# Search for FabricTelemetry usage in firmware
grep -r "FabricTelemetry\|fabric_telemetry" tt_metal/hw/firmware --include="*.cc" --include="*.cpp"

# Find L1 memory layout
grep -r "telemetry.*l1\|L1.*telemetry" tt_metal/hw/firmware --include="*.cc" --include="*.h"
```

Look for where `FabricTelemetry` or `FabricTelemetryStaticOnly` structs are allocated in L1.

### Step 2: Zero Telemetry Structures on Init

Add to ERISC initialization routine (early in main or init function):

```cpp
// In ERISC init routine, before any telemetry use
extern FabricTelemetry fabric_telemetry __attribute__((section(".l1_data")));

void init_fabric_telemetry() {
    // Zero entire structure to prevent garbage values
    memset(&fabric_telemetry, 0, sizeof(fabric_telemetry));

    // Will populate static_info below
}
```

### Step 3: Populate static_info Fields

You need to determine these values from fabric router configuration:

```cpp
void populate_fabric_static_info() {
    // These values come from fabric router configuration
    // They're set during fabric initialization via control plane

    // mesh_id: Which mesh this router belongs to (usually 0 for single mesh)
    fabric_telemetry.static_info.mesh_id = get_mesh_id();  // TODO: implement

    // device_id: Logical device ID within the mesh
    fabric_telemetry.static_info.device_id = get_device_id();  // TODO: implement

    // direction: Physical direction of this link (NORTH=0, SOUTH=1, EAST=2, WEST=3, etc.)
    fabric_telemetry.static_info.direction = get_link_direction();  // TODO: implement

    // fabric_config: Configuration bits (link width, speed, operational mode)
    fabric_telemetry.static_info.fabric_config = 0;  // Start with 0, add config bits as needed

    // supported_stats: Bitmask of what dynamic stats are available
    // BANDWIDTH = 0x02, ROUTER_STATE = 0x01, HEARTBEAT_TX = 0x04, HEARTBEAT_RX = 0x08
    fabric_telemetry.static_info.supported_stats =
        BANDWIDTH | ROUTER_STATE | HEARTBEAT_TX | HEARTBEAT_RX;  // Enable all stats
}
```

### Step 4: Find How to Get Topology Info

**Critical:** ERISC firmware needs to know its position in the mesh. This information comes from:

1. **Fabric router context** - set during initialization by control plane
2. **Device coordinates** - may be in firmware already
3. **Passed as parameters** - control plane may send this when configuring routers

**Search for existing topology info in firmware:**
```bash
# Look for mesh/device ID in firmware
grep -r "mesh\|device.*id\|logical.*id" tt_metal/hw/firmware --include="*.cc" --include="*.h"

# Look for direction/link info
grep -r "direction\|NORTH\|SOUTH\|EAST\|WEST" tt_metal/hw/firmware --include="*.cc" --include="*.h"

# Check fabric router config structures
grep -r "router.*config\|fabric.*config" tt_metal/hw/firmware --include="*.cc" --include="*.h"
```

### Step 5: Implementation Pattern

Based on what you find, the pattern will be:

**If topology info is already in firmware:**
```cpp
// Just copy from existing variables to telemetry struct
fabric_telemetry.static_info.mesh_id = g_mesh_id;  // Global set during init
fabric_telemetry.static_info.device_id = g_device_id;
fabric_telemetry.static_info.direction = g_link_direction;
```

**If topology info comes from control plane:**
```cpp
// Control plane sends this during fabric router configuration
// Look for where router gets configured and capture these values

void configure_fabric_router(uint16_t mesh_id, uint8_t device_id, uint8_t direction, ...) {
    // Store for telemetry
    fabric_telemetry.static_info.mesh_id = mesh_id;
    fabric_telemetry.static_info.device_id = device_id;
    fabric_telemetry.static_info.direction = direction;
    fabric_telemetry.static_info.supported_stats = BANDWIDTH | ROUTER_STATE | HEARTBEAT_TX | HEARTBEAT_RX;
}
```

**If you need to determine direction from channel ID:**
```cpp
// Channel ID often maps to direction
// e.g., on Wormhole: channels 0-7 might map to different directions
uint8_t get_direction_from_channel(uint8_t channel_id) {
    // This mapping is architecture-specific
    // Check HAL or architecture docs
    // Example (verify against actual hardware):
    const uint8_t DIRECTION_NORTH = 0;
    const uint8_t DIRECTION_SOUTH = 1;
    const uint8_t DIRECTION_EAST = 2;
    const uint8_t DIRECTION_WEST = 3;

    // Map channel to direction (architecture-specific)
    // This is just an example - verify actual mapping
    if (channel_id < 2) return DIRECTION_NORTH;
    if (channel_id < 4) return DIRECTION_SOUTH;
    if (channel_id < 8) return DIRECTION_EAST;
    return DIRECTION_WEST;
}
```

## Where to Get Help

### Talk to Fabric Team

**Key people (from CODEOWNERS):**
- @ubcheema (Umar Cheema) - Fabric and ERISC owner
- @aliuTT (Brian Liu) - ERISC firmware
- @SeanNijjar - Fabric telemetry original author

**Questions to ask:**
1. Where does ERISC firmware receive mesh_id and device_id during initialization?
2. How does firmware know which direction (NORTH/SOUTH/EAST/WEST) a link points?
3. Is there existing router configuration code we should hook into?
4. What's the relationship between channel_id and physical direction?

### Reference Code

**Check how control plane sets up fabric:**
```bash
grep -r "configure.*fabric\|initialize.*fabric" tt_metal/fabric --include="*.cpp"
```

**Check how routing tables are built:**
```bash
# Routing tables know topology
grep -r "routing.*table\|FabricNodeId" tt_metal/fabric --include="*.cpp" | head -20
```

The routing code already knows mesh topology - find where it configures ERISC and add telemetry initialization there.

## Testing Plan

### Test 1: Verify Structures Are Zeroed

After changes, run validation test:
```bash
export TT_METAL_FABRIC_TELEMETRY=1
./build/test/tt_metal/tt_fabric/test_bandwidth_telemetry_validation --gtest_filter="*DetectUninitializedCounters"
```

Should show reasonable counter values (not 10^18 garbage).

### Test 2: Verify static_info Is Populated

Add temporary debug logging in telemetry reader to print static_info values:

```cpp
// In tt_telemetry or test code
auto snapshot = read_fabric_telemetry(cluster, hal, chip_id, channel);
log_info(LogTest, "static_info: mesh_id={}, device_id={}, direction={}, supported_stats=0x{:x}",
    snapshot.static_info.mesh_id,
    snapshot.static_info.device_id,
    snapshot.static_info.direction,
    snapshot.static_info.supported_stats);
```

Expected output:
- mesh_id: 0 (for single mesh systems)
- device_id: 0-7 (logical device in mesh)
- direction: 0-3 or similar (NORTH/SOUTH/EAST/WEST)
- supported_stats: 0x0F (all bits set) or at minimum 0x02 (BANDWIDTH)

### Test 3: Verify Bandwidth Metrics Appear

Start telemetry server:
```bash
export TT_METAL_FABRIC_TELEMETRY=1
./build/tt_telemetry/tt_telemetry_server --fsd=fsd.textproto --watchdog-timeout 60
```

Query metrics:
```bash
curl -s http://localhost:8080/api/metrics | grep -i bandwidth
```

Should see:
- `txBandwidthMBps{...}`
- `rxBandwidthMBps{...}`
- `txPeakBandwidthMBps{...}`
- `rxPeakBandwidthMBps{...}`

## Commit Strategy

Following CLAUDE.md guidelines:

```bash
# Commit 1: Zero telemetry structures
git add tt_metal/hw/firmware/...
git commit -m "Zero fabric telemetry structures on ERISC init"

# Commit 2: Populate static_info
git add tt_metal/hw/firmware/...
git commit -m "Populate fabric telemetry static_info fields"

# Commit 3: Set supported_stats bitmask
git add tt_metal/hw/firmware/...
git commit -m "Enable bandwidth telemetry via supported_stats"

# Push
git push origin kkfernandez/fabric-telemetry-firmware-init
```

## PR Description Template

```markdown
### Ticket
Addresses firmware portion of fabric bandwidth telemetry

### Problem description
Fabric bandwidth telemetry metrics exist in tt-telemetry but don't function because:
- ERISC firmware doesn't initialize telemetry structures (garbage values in L1)
- static_info fields (mesh_id, device_id, direction) not populated
- supported_stats bitmask not set, so dynamic_info is disabled

This prevents bandwidth metrics from appearing in telemetry output.

### What's changed
1. Zero fabric telemetry structures during ERISC initialization
2. Populate static_info fields from fabric router configuration
3. Set supported_stats bitmask to enable BANDWIDTH (and other dynamic stats)

This enables tt-telemetry to:
- Read valid counter values (no garbage)
- Access dynamic_info for bandwidth calculations
- Export bandwidth metrics to Prometheus/GUI

### Checklist
- [ ] All post commit CI passes
- [ ] Fabric tests pass
- [ ] Bandwidth metrics appear in telemetry server output
- [ ] static_info fields contain correct values for mesh topology
```

## Key Gotchas

1. **Architecture differences:** Wormhole vs Blackhole may have different:
   - Channel-to-direction mappings
   - Number of ERISC cores (WH=1, BH=2)
   - Telemetry structure layouts

2. **Timing:** Telemetry initialization must happen AFTER fabric router is configured

3. **Per-channel vs per-chip:** Each channel has its own telemetry structure - populate for all channels

4. **Coordinate systems:** Mesh coordinates vs physical coordinates - make sure you're using logical mesh IDs/device IDs, not physical chip IDs

## Success Criteria

After this PR:
1. ✓ No garbage counter values in logs
2. ✓ `supported_stats != 0` in telemetry reads
3. ✓ Bandwidth metrics appear in `http://localhost:8080/api/metrics`
4. ✓ static_info fields match actual mesh topology
5. ✓ Validation test passes with real bandwidth calculations

## Next Steps After This PR

Once firmware populates telemetry:
1. Run validation test with actual fabric workload
2. Build GUI for fabric bandwidth visualization
3. Deploy to ClosetBox cluster

## Questions to Resolve

While implementing, document answers to:
1. Where does mesh_id come from? (Control plane? Fixed at 0?)
2. How is device_id assigned within mesh? (Logical fabric node ID)
3. How to map channel_id → direction? (Check HAL or arch docs)
4. What should fabric_config contain? (Reserved for future use, start with 0)
5. Should we enable ALL stats (0x0F) or just BANDWIDTH (0x02)?

Good luck!
