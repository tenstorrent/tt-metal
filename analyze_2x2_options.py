#!/usr/bin/env python3
"""Analyze which 4-chip combinations form valid 2x2 meshes"""

# Physical ethernet connections from t3k_phys_topology.yaml
connections = {(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (6, 7)}

# Make bidirectional
all_connections = connections | {(b, a) for a, b in connections}


def is_valid_2x2_mesh(chips):
    """Check if 4 chips form a valid 2x2 mesh"""
    a, b, c, d = chips
    # For 2x2: A-B, A-C, B-D, C-D must all be connected
    required = [(a, b), (a, c), (b, d), (c, d)]
    return all((x, y) in all_connections for x, y in required)


# Test all possible 4-chip combinations
print("Valid 2×2 Mesh Configurations:\n")
valid_configs = []

from itertools import combinations

for chips in combinations(range(8), 4):
    if is_valid_2x2_mesh(chips):
        valid_configs.append(chips)
        print(f"✓ Chips {chips[0]}, {chips[1]}, {chips[2]}, {chips[3]}")
        print(f"  Layout: {chips[0]}↔{chips[1]}")
        print(f"          ↕ ↕")
        print(f"          {chips[2]}↔{chips[3]}\n")

print(f"\nTotal valid configurations: {len(valid_configs)}")
