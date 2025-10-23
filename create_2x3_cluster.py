#!/usr/bin/env python3
"""Create a cluster descriptor with only 6 chips for 2×3 mesh"""
import yaml

# Load full topology
with open("t3k_phys_topology.yaml", "r") as f:
    full_topology = yaml.safe_load(f)

# Select only chips 4,0,3,5,1,2 (6 chips for 2x3 mesh)
selected_chips = [4, 0, 3, 5, 1, 2]

# Create new topology with only these chips
partial_topology = {"arch": {}, "chips": {}, "ethernet_connections": []}

# Remap chip IDs: 4→0, 0→1, 3→2, 5→3, 1→4, 2→5
chip_remap = {4: 0, 0: 1, 3: 2, 5: 3, 1: 4, 2: 5}

# Copy architecture for selected chips
for old_id, new_id in chip_remap.items():
    if old_id in full_topology.get("arch", {}):
        partial_topology["arch"][new_id] = full_topology["arch"][old_id]

# Copy chip info for selected chips
for old_id, new_id in chip_remap.items():
    if old_id in full_topology.get("chips", {}):
        partial_topology["chips"][new_id] = full_topology["chips"][old_id]

# Copy and remap ethernet connections
for conn in full_topology.get("ethernet_connections", []):
    if len(conn) == 2:
        chip_a = conn[0]["chip"]
        chip_b = conn[1]["chip"]

        # Only include connections between selected chips
        if chip_a in chip_remap and chip_b in chip_remap:
            new_conn = [
                {"chip": chip_remap[chip_a], "chan": conn[0]["chan"]},
                {"chip": chip_remap[chip_b], "chan": conn[1]["chan"]},
            ]
            partial_topology["ethernet_connections"].append(new_conn)

print("2×3 Partial Cluster Topology:")
print(f"  Chips: {len(partial_topology['arch'])}")
print(f"  Connections: {len(partial_topology['ethernet_connections'])}")

# Save
with open("t3k_2x3_cluster_topology.yaml", "w") as f:
    yaml.dump(partial_topology, f, default_flow_style=False, sort_keys=False)

print("\n✅ Created: t3k_2x3_cluster_topology.yaml")
print("\nChip Remapping:")
for old_id, new_id in sorted(chip_remap.items(), key=lambda x: x[1]):
    print(f"  Physical chip {old_id} → Logical chip {new_id}")
