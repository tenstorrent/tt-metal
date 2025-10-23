#!/usr/bin/env python3
"""Explain why 2x3 doesn't work"""

print("\n" + "=" * 70)
print("Why 2×3 Mesh Doesn't Work on Your T3K")
print("=" * 70)

print("\nYour Physical Layout:")
print("-" * 70)
print("     Col 0   Col 1   Col 2   Col 3")
print("Row 0:  4  ←→  0  ←→  3  ←→  7")
print("        ↕       ↕       ↕       ↕")
print("Row 1:  5  ←→  1  ←→  2  ←→  6")

print("\n🔍 Attempting 2×3 Configurations:")
print("-" * 70)

# Check some potential 2x3 configs
attempts = [
    ("Left 2×3", [4, 0, 3, 5, 1, 2], "4-0-3 / 5-1-2"),
    ("Right 2×3", [0, 3, 7, 1, 2, 6], "0-3-7 / 1-2-6"),
    ("Middle 2×3", [4, 0, 3, 5, 1, 2], "4-0-3 / 5-1-2"),
]

connections = {(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (6, 7)}

for name, chips, layout in attempts:
    print(f"\n{name}: {layout}")
    print(f"  Chips: {chips}")

    # Check required connections
    # Row 0: chips[0]-chips[1]-chips[2]
    # Row 1: chips[3]-chips[4]-chips[5]
    # Vert: chips[0]↔chips[3], chips[1]↔chips[4], chips[2]↔chips[5]

    required = [
        (chips[0], chips[1]),
        (chips[1], chips[2]),  # Top row
        (chips[3], chips[4]),
        (chips[4], chips[5]),  # Bottom row
        (chips[0], chips[3]),
        (chips[1], chips[4]),
        (chips[2], chips[5]),  # Vertical
    ]

    missing = []
    for a, b in required:
        conn = (min(a, b), max(a, b))
        if conn not in connections:
            missing.append((a, b))

    if missing:
        print(f"  ❌ Missing connection(s):")
        for a, b in missing:
            print(f"     • {a} ↔ {b}")
    else:
        print(f"  ✓ All connections present!")

print("\n\n✅ WHAT WORKS on Your T3K:")
print("=" * 70)
print("• 2×4 mesh: ALL 8 chips (4,0,3,7,5,1,2,6)")
print("• 2×2 mesh: Chips 0,1,4,5 OR chips 2,3,6,7")
print("• 1×4 mesh: Several combinations")
print("• 1×2 mesh: Many combinations")

print("\n💡 Recommendation:")
print("-" * 70)
print("Use the full 2×4 mesh to maximize performance!")
print("Command:")
print("  TT_MESH_GRAPH_DESC_PATH=t3k_mesh_graph_2x4.yaml python ...")
