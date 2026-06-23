"""Enumerate the (4,8) mesh: print the device-id grid by (row,col), then mark the
cluster_axis=0 dispatch rings (columns) and replicate groups (rows), and locate the
stuck devices 16/20/24/28 so we can say whether they are one CCL plane or one-per-plane.
"""
import ttnn

MS = (4, 8)
STUCK = {16, 20, 24, 28}


def grid_from(ids):
    return [list(ids[r * MS[1]:(r + 1) * MS[1]]) for r in range(MS[0])]


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(MS))
    try:
        ids = list(dev.get_device_ids())
        print("get_device_ids() (mesh order):", ids, flush=True)
        g = grid_from(ids)
        print("\n=== mesh grid: g[row][col] = device id (row-major) ===", flush=True)
        hdr = "R\\C " + " ".join(f"C{c:>2}" for c in range(MS[1]))
        print(hdr, flush=True)
        for r in range(MS[0]):
            cells = " ".join(f"{g[r][c]:>3}" for c in range(MS[1]))
            print(f"R{r}  {cells}", flush=True)

        print("\n=== cluster_axis=0 dispatch rings = COLUMNS (4 devices, along the 4 rows) ===", flush=True)
        for c in range(MS[1]):
            ring = [g[r][c] for r in range(MS[0])]
            mark = "  <-- all 4 stuck devices" if set(ring) == STUCK else ""
            print(f"  col{c} ring: {ring}{mark}", flush=True)

        print("\n=== replicate groups = ROWS (8 devices, along the 8 cols) ===", flush=True)
        for r in range(MS[0]):
            print(f"  row{r}: {g[r]}", flush=True)

        print("\n=== location of stuck devices 16/20/24/28 ===", flush=True)
        locs = []
        for r in range(MS[0]):
            for c in range(MS[1]):
                if g[r][c] in STUCK:
                    locs.append((g[r][c], r, c))
                    print(f"  dev {g[r][c]:>2} -> (row={r}, col={c})", flush=True)
        rows = sorted({r for _, r, _ in locs})
        cols = sorted({c for _, _, c in locs})
        print(f"\n  stuck span: rows={rows} cols={cols}", flush=True)
        if len(cols) == 1:
            print("  => SAME COLUMN -> one cluster_axis=0 dispatch ring -> ONE CCL plane", flush=True)
        elif len(rows) == 1:
            print("  => SAME ROW -> one replicate group", flush=True)
        else:
            print("  => spread across rows & cols -> NOT a single ring/group", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
