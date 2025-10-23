#!/usr/bin/env python3
"""Analyze T3K based on physical coordinates from tt-topology"""

# From tt-topology -ls output
chips_coords = {0: (1, 0), 1: (1, 1), 2: (2, 1), 3: (2, 0), 4: (0, 0), 5: (0, 1), 6: (3, 1), 7: (3, 0)}

print("\n" + "=" * 70)
print("T3K Physical Layout (from tt-topology -ls)")
print("=" * 70)

# Arrange by grid position
print("\nPhysical Grid Layout:")
print("-" * 70)
print("     Col 0   Col 1   Col 2   Col 3")
for row in [0, 1]:
    row_str = f"Row {row}:"
    for col in [0, 1, 2, 3]:
        chip = [c for c, (x, y) in chips_coords.items() if x == col and y == row]
        if chip:
            row_str += f"   {chip[0]:2d}   "
        else:
            row_str += "    --   "
    print(row_str)

print("\n🔌 Physical Connections (from your topology):")
print("-" * 70)
connections = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (6, 7)]

for a, b in connections:
    coord_a = chips_coords[a]
    coord_b = chips_coords[b]
    print(f"Chip {a} {coord_a} ←→ Chip {b} {coord_b}")

print("\n🔍 Checking 2×4 Mesh Requirements:")
print("-" * 70)
print("For a proper 2×4 mesh in row-major order (4→5→?→?, 0→1→2→?)")
print()

# Check if we can form: 4-5-?-? on row 1 and 0-1-2-? on row 0
print("Expected 2×4 Layout:")
print("  4 - 0 - 3 - 7")
print("  |   |   |   |")
print("  5 - 1 - 2 - 6")
print()

# Check required connections for this layout
required = [
    (4, 0),
    (0, 3),
    (3, 7),  # Top row
    (5, 1),
    (1, 2),
    (2, 6),  # Bottom row
    (4, 5),
    (0, 1),
    (3, 2),
    (7, 6),  # Vertical
]

print("Checking connections:")
missing = []
for a, b in required:
    conn = (min(a, b), max(a, b))
    if conn in connections:
        print(f"  ✓ {a} ←→ {b}")
    else:
        print(f"  ❌ {a} ←→ {b} MISSING")
        missing.append((a, b))

if missing:
    print(f"\n❌ Missing {len(missing)} connection(s) for 2×4 mesh:")
    for a, b in missing:
        print(f"   • Chip {a} ↔ Chip {b}")
    print()
    print("Your T3K has the physical POSITIONS for 2×4,")
    print("but the ethernet WIRING doesn't support it!")
else:
    print("\n✅ All connections present - 2×4 mesh IS possible!")
    print("The tool's validation was incorrect!")
