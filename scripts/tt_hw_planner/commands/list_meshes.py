from __future__ import annotations

from ..hardware import HARDWARE


def cmd_list_meshes(args) -> int:
    print("Canonical mesh topology (probe with ttnn.open_mesh_device):")
    print()
    print(f"  {'BOX':<8} {'ARCH':<10} {'CHIPS':>5}  CANONICAL MESHES")
    print("  " + "-" * 76)
    for b in HARDWARE:
        shapes = ", ".join(f"[{r},{c}]" for r, c in b.mesh_shapes[:8])
        if len(b.mesh_shapes) > 8:
            shapes += ", ..."
        print(f"  {b.name:<8} {b.arch:<10} {b.chips:>5}  {shapes}")
    print()
    return 0
