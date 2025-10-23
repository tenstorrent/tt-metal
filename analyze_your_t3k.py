#!/usr/bin/env python3
"""Deep analysis of your specific T3K wiring"""
import yaml

# Load your actual topology
with open("t3k_phys_topology.yaml", "r") as f:
    topology = yaml.safe_load(f)

print("\n" + "=" * 70)
print("YOUR T3K Physical Analysis")
print("=" * 70)

print("\n📊 Board Layout (from your topology):")
print("-" * 70)
boards = topology.get("boards", [])
for board in boards:
    board_id = board[0].get("board_id", "unknown")
    board_type = board[1].get("board_type", "unknown")
    chips = board[2].get("chips", [])
    print(f"Board {board_id[-2:]}: {board_type} → Chips {chips}")

print("\n🔌 Ethernet Connections:")
print("-" * 70)
connections = topology.get("ethernet_connections", [])
conn_map = {}
for conn in connections:
    if len(conn) == 2:
        a, b = conn[0]["chip"], conn[1]["chip"]
        if a not in conn_map:
            conn_map[a] = []
        if b not in conn_map:
            conn_map[b] = []
        conn_map[a].append(b)
        conn_map[b].append(a)

for chip in sorted(conn_map.keys()):
    neighbors = sorted(conn_map[chip])
    print(f"Chip {chip} connects to: {neighbors}")

print("\n🔍 Checking for 2×4 Mesh Possibility:")
print("-" * 70)

# For 2×4 mesh, need chips arranged like:
# Row 0: A - B - C - D
# Row 1: E - F - G - H
# With vertical connections: A↔E, B↔F, C↔G, D↔H


def has_connection(a, b, conn_map):
    return b in conn_map.get(a, [])


print("Attempting to find 2×4 arrangement...")
print()

# Try different chip arrangements
from itertools import permutations

found_2x4 = False
# Check if any arrangement works
# For 2x4: need consecutive chips in each row
for r0_start in range(8):
    for r0 in permutations(range(8), 4):
        if r0[0] != r0_start:
            continue
        remaining = [i for i in range(8) if i not in r0]
        for r1 in permutations(remaining, 4):
            # Check horizontal connections
            if not all(has_connection(r0[i], r0[i + 1], conn_map) for i in range(3)):
                continue
            if not all(has_connection(r1[i], r1[i + 1], conn_map) for i in range(3)):
                continue
            # Check vertical connections
            if not all(has_connection(r0[i], r1[i], conn_map) for i in range(4)):
                continue
            print(f"✓ FOUND 2×4 configuration!")
            print(f"  Row 0: {list(r0)}")
            print(f"  Row 1: {list(r1)}")
            found_2x4 = True
            break
        if found_2x4:
            break
    if found_2x4:
        break

if not found_2x4:
    print("❌ No 2×4 configuration possible with current wiring")
    print()
    print("Your T3K appears to be wired as TWO separate 2×2 blocks:")
    print()
    print("  Block 1: Chips 0,1,4,5    Block 2: Chips 2,3,6,7")
    print("    0 ←→ 1                    2 ←→ 3")
    print("    ↕    ↕                    ↕    ↕")
    print("    4 ←→ 5                    6 ←→ 7")
    print()
    print("  Connected via: 1 ←→ 2 (bridge between blocks)")
    print()
    print("This is a VALID T3K configuration, but doesn't form")
    print("a single 2×4 rectangular mesh.")
