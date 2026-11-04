# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Map non-deterministic (ND) output elements back to the Tensix cores that produced them,
and describe the spatial pattern of the failing cores.

Pure torch, no ttnn, so it can be unit tested off-device (test_nd_core_map.py).

A test opts in by telling OpTestBase how its output is laid out across cores:
  nd_core_block      (rows, cols) of the output block each core writes
  nd_core_linear_x   optional: blocks are assigned to cores row-wise over a grid this many cores wide
                     (1D matmul with interleaved output), instead of block (r, c) -> core (c, r)
  nd_core_transpose  optional: block (r, c) -> core (r, c), for ops that put M on cores_x
                     (e.g. minimal_matmul when M > N)

Reading the pattern: in a 2D multicast matmul one input is forwarded core to core along each row
and the other down each column. So a single failing core points at that core, while a run of
cores along one row or column can also be a bad forwarding hop into the run on that axis.
"""

import torch


def per_block_mismatches(ref, out, block_h, block_w):
    """Count mismatching elements in each (block_h x block_w) output block.

    Returns an int tensor [blocks_y, blocks_x]. Partial blocks at the bottom/right edge are padded,
    so the last core still counts. Leading (batch) dims are summed.
    """
    diff = ref != out
    h, w = diff.shape[-2], diff.shape[-1]
    diff = diff.reshape(-1, h, w).sum(dim=0)
    blocks_y, blocks_x = -(-h // block_h), -(-w // block_w)
    diff = torch.nn.functional.pad(diff, (0, blocks_x * block_w - w, 0, blocks_y * block_h - h))
    return diff.reshape(blocks_y, block_h, blocks_x, block_w).sum(dim=(1, 3))


def block_to_core(by, bx, blocks_x, linear_x=None, transpose=False):
    """Logical core (x, y) that wrote output block (by, bx)."""
    if linear_x:
        idx = by * blocks_x + bx
        return idx % linear_x, idx // linear_x
    return (by, bx) if transpose else (bx, by)


def core_mismatches(ref, out, block, linear_x=None, transpose=False):
    """{(x, y): mismatching element count} for every core with at least one mismatch."""
    counts = per_block_mismatches(ref, out, *block)
    blocks_x = counts.shape[1]
    return {
        block_to_core(by, bx, blocks_x, linear_x, transpose): int(counts[by, bx])
        for by, bx in counts.nonzero().tolist()
    }


TILE = 32
FACE = 16
STRIP_FRAC = 0.5  # a row/column of a core's block counts as a strip when at least this much of it is wrong
DOMINANT_FRAC = 0.8  # a structure label needs this share of the core's wrong elements


def core_blocks(ref, out, block, linear_x=None, transpose=False):
    """{(x, y): (by, bx)} output block index of every failing core."""
    counts = per_block_mismatches(ref, out, *block)
    blocks_x = counts.shape[1]
    return {block_to_core(by, bx, blocks_x, linear_x, transpose): (by, bx) for by, bx in counts.nonzero().tolist()}


def _bits(t):
    """Raw bit pattern as an int tensor, for float formats we can reinterpret."""
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16).to(torch.int32) & 0xFFFF, 16
    if t.dtype == torch.float32:
        return t.view(torch.int32).to(torch.int64) & 0xFFFFFFFF, 32
    return None, 0


def block_structure(ref, out, by, bx, block):
    """Where the wrong elements sit inside one core's output block, to tell which stage corrupted them.

    In a matmul out = in0 x in1:
      row strips     whole output rows wrong       -> a bad in0 value (the input forwarded along rows)
      column strips  whole output columns wrong    -> a bad in1 value (the input forwarded down columns)
      whole tiles    complete 32x32 tiles wrong    -> a tile moved or written wrong (output writer, L1 overwrite)
      face column/row lane  the same position inside every 16x16 face, across the block -> a datapath
                     lane in unpack/math/Dest/pack, not an input value (strips that repeat every 16)
      scattered      isolated elements, often in particular faces or bit lanes -> math/Dest or pack
    Which RISC carries in0/in1 depends on the op: minimal_matmul, not transposed, moves in0 and writes
    output on NCRISC and moves in1 on BRISC; transposed (M > N) swaps them.
    """
    h, w = ref.shape[-2], ref.shape[-1]
    if ref.numel() != h * w:
        return {"label": "n/a (batched output)"}
    bh, bw = block
    r0, c0 = by * bh, bx * bw
    r1, c1 = min(r0 + bh, h), min(c0 + bw, w)
    rb, ob = ref.reshape(h, w)[r0:r1, c0:c1], out.reshape(h, w)[r0:r1, c0:c1]
    d = rb != ob
    n = int(d.sum())
    rows, cols = r1 - r0, c1 - c0
    if n == 0:
        return {"label": "none", "elements": 0}

    strip_rows = d.sum(dim=1) >= STRIP_FRAC * cols
    strip_cols = d.sum(dim=0) >= STRIP_FRAC * rows
    in_row_strips = int(d[strip_rows].sum())
    in_col_strips = int(d[:, strip_cols].sum())

    # Tiles of this block (pad partial edge tiles)
    th, tw = -(-rows // TILE), -(-cols // TILE)
    dt = torch.nn.functional.pad(d, (0, tw * TILE - cols, 0, th * TILE - rows)).reshape(th, TILE, tw, TILE)
    per_tile = dt.sum(dim=(1, 3))
    in_whole_tiles = int(per_tile[per_tile >= 0.9 * TILE * TILE].sum())

    # Position inside a face (0..15), summed over every face of every tile. A bad input value lands on
    # arbitrary rows/columns; a datapath lane defect keeps hitting the same face row or column.
    face_cols = dt.reshape(th, TILE, tw, 2, FACE).sum(dim=(0, 1, 2, 3))
    face_rows = dt.reshape(th, 2, FACE, tw, TILE).sum(dim=(0, 1, 3, 4))
    col_lanes = sorted(((i, int(v) / n) for i, v in enumerate(face_cols.tolist()) if v), key=lambda t: -t[1])
    row_lanes = sorted(((i, int(v) / n) for i, v in enumerate(face_rows.tolist()) if v), key=lambda t: -t[1])

    label = "scattered"
    for name, share in (
        ("whole tiles", in_whole_tiles / n),
        ("row strips", in_row_strips / n),
        ("column strips", in_col_strips / n),
    ):
        if share >= DOMINANT_FRAC:
            label = name
            break
    # Same face column (or row) repeating across most 16-wide face groups: a lane, not an input value.
    # Checked last so it overrides strips, which a lane defect also produces (1/16 of all columns wrong).
    # A single bad row or column also sits on one lane but in only one face group, so it stays a strip.
    col_groups = d.any(dim=0)
    col_groups = torch.nn.functional.pad(col_groups, (0, -(-cols // FACE) * FACE - cols)).reshape(-1, FACE).any(dim=1)
    row_groups = d.any(dim=1)
    row_groups = torch.nn.functional.pad(row_groups, (0, -(-rows // FACE) * FACE - rows)).reshape(-1, FACE).any(dim=1)
    if col_lanes and col_lanes[0][1] >= DOMINANT_FRAC and col_groups.float().mean() >= 0.5:
        label = f"face column lane {col_lanes[0][0]}"
    elif row_lanes and row_lanes[0][1] >= DOMINANT_FRAC and row_groups.float().mean() >= 0.5:
        label = f"face row lane {row_lanes[0][0]}"

    result = {"label": label, "elements": n, "col_lanes": col_lanes}

    # Which bit positions flipped, as a share of all flipped bits
    rbits, nbits = _bits(rb[d])
    obits, _ = _bits(ob[d])
    if nbits:
        x = rbits ^ obits
        flips = [int(((x >> b) & 1).sum()) for b in range(nbits)]
        total = sum(flips)
        if total:
            result["bits"] = sorted(((b, f / total) for b, f in enumerate(flips) if f), key=lambda t: -t[1])
    return result



def summarize_structures(structs):
    """Aggregate one core's block_structure() results over all failing checks into a signature."""
    structs = [s for s in structs if s.get("elements")]
    if not structs:
        return None
    labels = {}
    for s in structs:
        labels[s["label"]] = labels.get(s["label"], 0) + 1
    lanes = {}
    for s in structs:
        if s.get("col_lanes"):
            lane = s["col_lanes"][0][0]
            lanes[lane] = lanes.get(lane, 0) + 1
    bits, with_bits = {}, [s for s in structs if "bits" in s]
    for s in with_bits:
        for b, f in s["bits"]:
            bits[b] = bits.get(b, 0.0) + f / len(with_bits)
    elements = [s["elements"] for s in structs]
    return {
        "checks": len(structs),
        "labels": dict(sorted(labels.items(), key=lambda t: -t[1])),
        "top_col_lane": max(lanes.items(), key=lambda t: t[1]) if lanes else None,
        "elements_min": min(elements),
        "elements_max": max(elements),
        "bits": sorted(bits.items(), key=lambda t: -t[1])[:3],
    }


def describe_summary(s):
    """One-line rendering of summarize_structures()."""
    if not s:
        return "no structure"
    (label, n), *others = s["labels"].items()
    parts = [f"{label} in {n}/{s['checks']} checks"]
    if others:
        parts[0] += " (also " + ", ".join(f"{k} x{v}" for k, v in others) + ")"
    lane = s["top_col_lane"]
    if lane and not label.startswith("face column lane"):
        parts.append(f"most common face column {lane[0]} ({lane[1]}/{s['checks']} checks)")
    parts.append(f"{s['elements_min']}-{s['elements_max']} elements per check")
    if s["bits"]:
        parts.append("bits " + " ".join(f"{b}:{f:.0%}" for b, f in s["bits"]))
    return "; ".join(parts)


def _runs(values):
    """Split sorted ints into contiguous (start, end) runs."""
    runs = []
    for v in sorted(values):
        if runs and v == runs[-1][1] + 1:
            runs[-1][1] = v
        else:
            runs.append([v, v])
    return [tuple(r) for r in runs]


def _span(name, values, grid_len):
    runs = _runs(values)
    if len(runs) != 1:
        return f"{name}=" + ",".join(f"{a}..{b}" if a != b else f"{a}" for a, b in runs) + " (not contiguous)"
    a, b = runs[0]
    edges = []
    if grid_len is not None:
        if a == 0:
            edges.append(f"{name}=0")
        if b == grid_len - 1:
            edges.append(f"{name}={grid_len - 1}")
    reach = f", reaches edge {' and '.join(edges)}" if edges else ""
    return f"{name}={a}..{b}{reach}"


def classify(cores, grid=None):
    """One-line description of the failing-core pattern.

    cores: iterable of (x, y). grid: optional (grid_x, grid_y) so runs can be reported as reaching an edge.
    Patterns: none, single core, row run, column run, rectangle, whole grid, scattered.
    """
    cores = set(cores)
    gx, gy = grid if grid else (None, None)
    if not cores:
        return "none"
    xs, ys = {x for x, _ in cores}, {y for _, y in cores}
    n = len(cores)
    if n == 1:
        (x, y), = cores
        return f"single core ({x},{y})"
    if gx and gy and n == gx * gy:
        return f"whole grid ({n} cores)"
    if len(ys) == 1:
        return f"row y={next(iter(ys))}: {_span('x', xs, gx)} ({n} cores)"
    if len(xs) == 1:
        return f"column x={next(iter(xs))}: {_span('y', ys, gy)} ({n} cores)"
    rect = {(x, y) for x in range(min(xs), max(xs) + 1) for y in range(min(ys), max(ys) + 1)}
    if cores == rect:
        return f"rectangle x={min(xs)}..{max(xs)} y={min(ys)}..{max(ys)} ({n} cores)"
    listed = " ".join(f"({x},{y})" for x, y in sorted(cores, key=lambda c: (c[1], c[0]))[:12])
    more = f" +{n - 12} more" if n > 12 else ""
    return f"scattered ({n} cores in {len(ys)} rows, {len(xs)} columns): {listed}{more}"
