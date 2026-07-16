# SPDX-License-Identifier: Apache-2.0
"""
BABY DEBUG version of bigmesh_ops.py, with a teaching comment before EVERY line.

Each comment says:  ACTION (what the line does) / IN (what goes in) / OUT (what comes
out) / EXAMPLE (a concrete value). Running example throughout: we are rank 0 (launcher),
the mesh is (1,16) = 16 chips over 2 computers, rank 0 owns chips 0..7.

Same three ops as the real script (add, matmul, all_gather). Instead of a pausing
breakpoint (impossible here: tt-run runs TWO copies on TWO computers, no keyboard),
it NARRATES every step out loud with show(...).

Run it (env must be set, see GUIDE.md section 2):
    tt-run --tcp-interface ens18 \
      --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_1x16_experimental_bigmesh_mgd.textproto \
      --hosts t3k-node-a,t3k-node-b \
      python3 claude_job/dual_t3k_ops/scripts/bigmesh_ops_debug.py
tt-run tags each printed line with the computer that said it: [1,0]=rank 0, [1,1]=rank 1.
"""

# ACTION: load helper libraries.  IN: -  OUT: modules usable below.
# EXAMPLE: math.prod(...), os.cpu_count(), torch.randn(...), ttnn.add(...)
import math
import os

import torch
import ttnn

# ACTION: constant = Wormhole tile size.  IN: -  OUT: TILE=32 (every dim is a multiple of 32).
# EXAMPLE: a tensor of width TILE*16 = 512 splits into 16 tiles of 32.
TILE = 32

# ACTION: a global counter so each narrated step is numbered.  IN: -  OUT: _step starts at 0.
# EXAMPLE: after 3 show() calls, _step == 3 and the last line read "[STEP 3] ...".
_step = 0


def describe(v):
    """Turn any value into a short, friendly sentence about what it is."""
    # ACTION: is this a normal CPU tensor (numbers we can read)?
    # IN: v (anything)   OUT: True/False
    # EXAMPLE: a torch tensor -> True ; an on-chip ttnn tensor -> False
    if isinstance(v, torch.Tensor):
        # ACTION: grab the first 6 numbers as plain floats for a preview.
        # IN: v (a CPU tensor)   OUT: a short python list
        # EXAMPLE: [-1.126, -1.152, -0.251, -0.434, 0.849, 0.692]
        first = v.flatten().to(torch.float32)[:6].tolist()
        # ACTION: build the sentence describing a CPU tensor.
        # IN: v.shape, v.dtype, first   OUT: a string
        # EXAMPLE: "a CPU tensor ..., size=(1, 1, 32, 512), type=torch.float32, first numbers=[...]"
        return (
            f"a CPU tensor (numbers I can read), size={tuple(v.shape)}, "
            f"type={v.dtype}, first numbers={[round(x, 3) for x in first]}"
        )
    # ACTION: not a CPU tensor -> maybe an on-chip ttnn tensor; read its .shape if present.
    # IN: v   OUT: shp = the shape, or None if v has no shape
    # EXAMPLE: ttnn on-chip tensor -> shp=(1,1,32,32) ; a plain string -> None
    shp = getattr(v, "shape", None)
    # ACTION: if it has a shape, it's an on-chip tensor -> describe size only (numbers are on chips).
    # IN: shp, v.dtype   OUT: a string
    # EXAMPLE: "an ON-THE-CHIPS tensor ..., size=(1, 1, 32, 32), type=DataType.BFLOAT16"
    if shp is not None:
        return (
            f"an ON-THE-CHIPS tensor (its numbers live on the 16 chips, so I can "
            f"only see its size here, not the numbers), size={tuple(shp)}, "
            f"type={getattr(v, 'dtype', '?')}"
        )
    # ACTION: fallback for anything else (int, string, ...).  IN: v  OUT: its repr string.
    # EXAMPLE: repr(2) -> "2"
    return repr(v)


def show(rank, action, value=None, why=""):
    """Our 'printing breakpoint'. Only rank 0 narrates, so the output stays readable."""
    # ACTION: make the module counter writable in here.  IN: -  OUT: _step is the global one.
    global _step
    # ACTION: only rank 0 prints; rank 1 stays quiet to avoid double output.
    # IN: rank   OUT: return early if not rank 0
    # EXAMPLE: rank 1 -> just returns and prints nothing
    if rank != 0:
        return
    # ACTION: advance the step number.  IN: _step  OUT: _step+1
    # EXAMPLE: 4 -> 5, so the next line says "[STEP 5]"
    _step += 1
    # ACTION: print the headline "[STEP n] I just did: <action>".
    # IN: _step, action   OUT: one printed line
    # EXAMPLE: "[STEP 5] I just did: computed the CORRECT answer on the CPU: golden_add = a + b"
    print(f"\n[STEP {_step}] I just did: {action}", flush=True)
    # ACTION: if a reason was given, print it.  IN: why  OUT: an optional printed line
    # EXAMPLE: "          why:   this is the 'teacher's answer key' ..."
    if why:
        print(f"          why:   {why}", flush=True)
    # ACTION: if a value was given, print describe(value).  IN: value  OUT: an optional printed line
    # EXAMPLE: "          result: a CPU tensor ..., size=(1, 1, 32, 512), ..."
    if value is not None:
        print(f"          result: {describe(value)}", flush=True)


def local_coords_and_tensors(tt_tensor, mesh_device):
    """Which chip-positions are on MY computer, and what is each one's data (on CPU)?

    RUNNING EXAMPLE: rank 0, tt_tensor='ta' (16 pieces each (1,1,32,32)), mesh (1,16),
    rank 0 owns chips 0..7. Whole function -> IN: ta + the (1,16) mesh.
    OUT (rank 0): [(0, cpu(1,1,32,32)), ..., (7, cpu(1,1,32,32))]  (rank 1: numbers 8..15)
    """
    # ACTION: ask the tensor which mesh positions it sits on, as a list.
    # IN: tt_tensor (ta)   OUT: list of 16 coordinates, in order
    # EXAMPLE: [(0,0), (0,1), (0,2), ..., (0,15)]
    coords = list(tt_tensor.tensor_topology().mesh_coords())

    # ACTION: number each coordinate 0,1,2,... so we can say "chip #5" not "(0,5)".
    # IN: coords (16)   OUT: dict {coordinate -> number}
    # EXAMPLE: {(0,0):0, (0,1):1, ..., (0,15):15}
    coord_to_index = {coord: idx for idx, coord in enumerate(coords)}

    # ACTION: get the "mine / REMOTE" table for this rank (None if single-host).
    # IN: mesh_device   OUT: MeshDeviceView (1x16 grid; each cell = a real chip or REMOTE)
    # EXAMPLE (rank 0): [chip,chip,chip,chip,chip,chip,chip,chip, REMOTE,...,REMOTE]
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None

    # ACTION: fetch the actual on-chip pieces that physically live on THIS rank.
    # IN: tt_tensor (ta)   OUT: list of 8 ttnn tensors (rank 0 owns 8 chips -> 8, not 16)
    # EXAMPLE: [ttnn(1,1,32,32), ... x8]
    device_tensors = ttnn.get_device_tensors(tt_tensor)

    # ACTION: default plan = "walk through all coordinates".
    # IN: coords (16)   OUT: coord_iter = same 16 items
    # EXAMPLE: coord_iter = [(0,0), ..., (0,15)]
    coord_iter = coords

    # ACTION: we have 16 coords but only 8 pieces -> keep only LOCAL coords so they line up.
    #         view.is_local(c) = look up cell c in the table: is it a real chip?
    # IN: coords (16) + view   OUT: coord_iter shrunk to the 8 local coords
    # EXAMPLE: [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7)]
    if view is not None and len(device_tensors) != len(coords):
        coord_iter = [c for c in coords if view.is_local(c)]

    # ACTION: start an empty list to collect results.  IN: -  OUT: out=[]
    out = []

    # ACTION: pair each local coordinate with its data piece, one at a time.
    # IN: coord_iter (8) + device_tensors (8)   OUT: yields 8 pairs
    # EXAMPLE: iter1 -> coord=(0,0), dev_t=piece0 ; iter2 -> coord=(0,1), dev_t=piece1 ; ...
    for coord, dev_t in zip(coord_iter, device_tensors):
        # ACTION: safety skip — if a coordinate is somehow REMOTE, ignore it.
        # IN: coord   OUT: skip, or fall through
        # EXAMPLE: here all 8 are local -> nothing skipped
        if view is not None and not view.is_local(coord):
            continue
        # ACTION: copy this chip's piece to the CPU (to_torch), tag it with its number.
        # IN: coord=(0,0), dev_t=piece0 (on chip)   OUT: append pair (number, cpu_tensor)
        # EXAMPLE: (0, cpu(1,1,32,32) first numbers [-1.125, -1.156, ...])
        out.append((coord_to_index[coord], ttnn.to_torch(dev_t)))

    # ACTION: hand back the collected list.  IN: out  OUT: 8 pairs
    # EXAMPLE: [(0, cpu(1,1,32,32)), ..., (7, cpu(1,1,32,32))]
    return out


def full_tensor_to_cpu(tt_replicated):
    """After all_gather every chip holds the WHOLE tensor; copy one chip's copy to CPU."""
    # ACTION: get this rank's local (chip#, cpu-piece) pairs. After all_gather each piece
    #         is already the FULL tensor.
    # IN: tt_replicated (e.g. tadd_full, per chip (1,1,32,512))   OUT: list of local pairs
    # EXAMPLE (rank 0): [(0, cpu(1,1,32,512)), (1, cpu(1,1,32,512)), ..., (7, ...)]
    locs = local_coords_and_tensors(tt_replicated, tt_replicated.device())
    # ACTION: safety — if this rank owns no chip for this tensor, complain.
    # IN: locs   OUT: raise if empty
    # EXAMPLE: locs=[] -> RuntimeError
    if not locs:
        raise RuntimeError("no local device tensor to read")
    # ACTION: return the CPU tensor of the FIRST local pair = the whole answer.
    # IN: locs[0] = (0, cpu(1,1,32,512))   OUT: the cpu tensor part
    # EXAMPLE: cpu(1,1,32,512), first numbers [-1.68, 0.383, 0.75, ...]
    return locs[0][1]


def report_error(golden_full, out_full, label, rank):
    """Print the error rate between two FULL (un-sharded) tensors. No pass/fail threshold."""
    # ACTION: flatten the CORRECT answer to a 1-D float list.
    # IN: golden_full (1,1,32,512)   OUT: g = 16384 floats
    # EXAMPLE: g = [-1.678, 0.387, 0.753, ...]
    g = golden_full.flatten().to(torch.float32)
    # ACTION: flatten the chip answer the same way.
    # IN: out_full (1,1,32,512)   OUT: o = 16384 floats
    # EXAMPLE: o = [-1.68, 0.383, 0.75, ...]  (slightly off, because bf16)
    o = out_full.flatten().to(torch.float32)
    # ACTION: the mistake at each position = |chip - correct|.
    # IN: o, g   OUT: diff = 16384 non-negative floats
    # EXAMPLE: diff = [0.002, 0.004, 0.003, ...]
    diff = (o - g).abs()
    # ACTION: "error rate" = size of all mistakes / size of the correct answer (scale-free).
    # IN: diff, g   OUT: rel_l2 (a fraction)   clamp_min stops divide-by-zero.
    # EXAMPLE: rel_l2 = 0.002474  (i.e. 0.2474%)
    rel_l2 = (diff.norm() / g.norm().clamp_min(1e-12)).item()
    # ACTION: narrate that we compared, via show() (rank-0 only).
    # IN: rank, label   OUT: printed "[STEP n] ... compared ..." line
    # EXAMPLE: "[STEP 12] I just did: compared the chip answer to the CPU answer for ADD"
    show(
        rank,
        f"compared the chip answer to the CPU answer for {label}",
        why="rel_L2_err = size of all the mistakes / size of the correct answer",
        value=None,
    )
    # ACTION: on rank 0, print the actual numbers (error rate, average and worst mistake).
    # IN: rel_l2, diff   OUT: one printed ">>> ..." line
    # EXAMPLE: ">>> ADD: rel_L2_err=0.2474%  mean_abs_err=2.4615e-03  max_abs_err=2.6366e-02"
    if rank == 0:
        print(
            f"          >>> {label}: rel_L2_err={rel_l2 * 100:.4f}%  "
            f"mean_abs_err={diff.mean().item():.4e}  max_abs_err={diff.max().item():.4e}",
            flush=True,
        )


def main():
    # ACTION: fix the random seed so BOTH computers make identical inputs.
    # IN: 0   OUT: torch's randomness is now deterministic
    # EXAMPLE: torch.randn(...) gives the same numbers on rank 0 and rank 1
    torch.manual_seed(0)

    # ACTION: choose the routing mode BEFORE opening the mesh (2D routes the 1x16; 1D fails).
    # IN: FABRIC_2D   OUT: fabric is configured
    # EXAMPLE: skip this / use 1D -> "Could not find any forwarding direction ..."
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    # ACTION: turn on all 16 chips as ONE (1,16) mesh and train the cables. (Slow step.)
    # IN: MeshShape(1,16)   OUT: device = a handle to the 16-chip mesh
    # EXAMPLE: device.shape == (1, 16)
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 16))

    # ACTION: undo MPI_Init pinning torch to 1 CPU thread (so CPU goldens run fast).
    # IN: os.cpu_count()   OUT: torch uses all CPU cores
    # EXAMPLE: on a 32-core box -> torch.set_num_threads(32)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    # ACTION: ask "which copy am I?"  IN: -  OUT: rank = 0 on launcher, 1 on remote
    # EXAMPLE: rank == 0
    rank = int(ttnn.distributed_context_get_rank())
    # ACTION: ask "how many copies are there?"  IN: -  OUT: size = number of computers
    # EXAMPLE: size == 2
    size = int(ttnn.distributed_context_get_size())

    # ACTION: BOTH ranks print this -> proof two copies run at once (that's SPMD).
    # IN: rank, size   OUT: one printed line per computer
    # EXAMPLE: "HELLO, I am rank 0 of 2. ..."  and  "HELLO, I am rank 1 of 2. ..."
    print(f"HELLO, I am rank {rank} of {size}. (Both computers print this line — that is SPMD!)", flush=True)

    # ACTION: narrate that the mesh is open (rank-0 only).  IN: rank  OUT: a [STEP] line
    # EXAMPLE: "[STEP 1] I just did: opened the mesh of chips"
    show(
        rank,
        "opened the mesh of chips",
        value=None,
        why="turned on all 16 chips across 2 computers and trained the cables between them",
    )
    # ACTION: on rank 0, print the mesh size in numbers.
    # IN: device.shape, size   OUT: one printed line
    # EXAMPLE: "the mesh size is (1, 16) = 16 chips, across 2 computers"
    if rank == 0:
        print(
            f"          the mesh size is {tuple(device.shape)} = "
            f"{math.prod(tuple(device.shape))} chips, across {size} computers",
            flush=True,
        )

    # ACTION: how many pieces to cut into = number of mesh columns = 16.
    # IN: -   OUT: NCOL = 16
    # EXAMPLE: width 32*16 = 512 splits into 16 tiles
    NCOL = 16

    # ============================ PHASE 1: ADD ===============================
    # ACTION: narrate the start of phase 1.  IN: rank  OUT: a [STEP] line
    show(rank, "starting PHASE 1: ADD (a + b), one number at a time", why="warm-up, no cables needed")

    # ACTION: make input 'a', then round to bf16 = the SAME data type as the device.
    # IN: shape (1,1,32,512)  OUT: a = bfloat16 tensor
    # WHY bf16: we are testing the multi-device WIRING, not bf16 accuracy. If CPU and chips
    #           start from identical numbers, any leftover error means a distribution bug.
    a = torch.randn(1, 1, TILE, TILE * NCOL).to(torch.bfloat16)
    show(
        rank,
        "made input 'a' (bf16, SAME dtype as the device)",
        value=a,
        why="same bf16 numbers on both sides, so error means a wiring bug, not rounding",
    )

    # ACTION: make input 'b' in bf16 too.  IN: (1,1,32,512)  OUT: b = bfloat16 tensor
    b = torch.randn(1, 1, TILE, TILE * NCOL).to(torch.bfloat16)
    show(rank, "made input 'b' (bf16)", value=b)

    # ACTION: CORRECT answer in bf16 (bf16 + bf16 -> bf16), same as the device's add.
    # IN: a, b (bf16)  OUT: golden_add = a+b (1,1,32,512) bf16
    # EXPECT: a correct multi-device ADD should match this ALMOST EXACTLY (~0 error).
    golden_add = a + b
    show(
        rank,
        "computed golden_add = a + b (bf16, same dtype as device)",
        value=golden_add,
        why="same dtype both sides -> a correct sharded add should match ~exactly",
    )

    # ACTION: SEND 'a' to the chips, sliced into 16 column-pieces (dim=3), squished to bf16.
    # IN: a (1,1,32,512)  OUT: ta = on-chip tensor; each chip holds (1,1,32,32)
    # EXAMPLE: column i of a -> chip i
    ta = ttnn.from_torch(
        a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3),
    )
    show(
        rank,
        "SENT 'a' to the chips, sliced into 16 pieces along the last axis",
        value=ta,
        why="ShardTensorToMesh(dim=3): column i of 'a' goes to chip i (and squished to bf16)",
    )

    # ACTION: SEND 'b' the same way.  IN: b (1,1,32,512)  OUT: tb = on-chip, per chip (1,1,32,32)
    tb = ttnn.from_torch(
        b,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3),
    )
    show(rank, "SENT 'b' to the chips the same way", value=tb)

    # ACTION: PEEK at one real chip's piece so you can see a shard is small.
    # IN: ta  OUT: prints chip #0's piece shape + first numbers
    # EXAMPLE: "PEEK: chip #0's piece of 'a' is size (1, 1, 32, 32) ... [-1.125, -1.156, ...]"
    if rank == 0:
        idx, piece = local_coords_and_tensors(ta, device)[0]
        print(
            f"          PEEK: chip #{idx}'s piece of 'a' is size {tuple(piece.shape)} "
            f"(just 1 of 16 columns), first numbers={[round(x,3) for x in piece.flatten()[:4].tolist()]}",
            flush=True,
        )

    # ACTION: ask the chips to ADD their own pieces.  IN: ta, tb  OUT: tadd = on-chip (1,1,32,32) each
    # EXAMPLE: all 16 chips add at the same time, no cables used
    tadd = ttnn.add(ta, tb)
    show(
        rank,
        "asked the chips to ADD: tadd = ta + tb",
        value=tadd,
        why="each chip adds only its own little piece, all 16 at the same time",
    )

    # ACTION: GATHER every chip's piece over the cables so every chip holds the whole row.
    # IN: tadd (pieces (1,1,32,32))  OUT: tadd_full = on-chip, per chip (1,1,32,512)
    # EXAMPLE: gather on dim 3 = the same axis we cut on
    tadd_full = ttnn.all_gather(tadd, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)
    show(
        rank,
        "GATHERED every chip's piece back together over the cables",
        value=tadd_full,
        why="all_gather: now EVERY chip holds the whole answer, not just its slice",
    )

    # ACTION: wait for the chips to truly finish (chip work is lazy/async).
    # IN: device  OUT: all queued work is done
    ttnn.synchronize_device(device)
    show(rank, "waited for the chips to truly finish", why="chip work is 'lazy'; this makes sure it is done")

    # ACTION: on rank 0, copy ONE chip's full answer back to the CPU and narrate it.
    # IN: tadd_full  OUT: full = cpu(1,1,32,512)
    # EXAMPLE: full first numbers [-1.68, 0.383, 0.75, ...] (bf16, slightly off)
    if rank == 0:
        full = full_tensor_to_cpu(tadd_full)
        show(
            rank,
            "copied ONE chip's full answer back to the CPU",
            value=full,
            why="that one chip now has the whole tensor, so this is the full un-sliced answer",
        )
    # ACTION: compare chip answer vs CPU answer and print the error rate.
    # IN: golden_add + full-from-chips  OUT: prints ">>> ADD: rel_L2_err=..."
    # EXAMPLE: ">>> ADD: rel_L2_err=0.2474% ..."
    report_error(golden_add, full_tensor_to_cpu(tadd_full), "ADD", rank)

    # ============================ PHASE 2: MATMUL ============================
    show(rank, "starting PHASE 2: MATMUL (A times B)", why="each chip does part of a big matrix multiply")

    # ACTION: make A and B in bf16 = same data type as the device.
    # IN: A (1,1,512,128), B (1,1,128,128)  OUT: bfloat16 tensors
    A = torch.randn(1, 1, TILE * NCOL, 128).to(torch.bfloat16)
    B = torch.randn(1, 1, 128, 128).to(torch.bfloat16)
    # ACTION: CORRECT answer mirroring the hardware matmul: bf16 inputs, fp32 ACCUMULATE, bf16 out.
    # IN: A, B (bf16)  OUT: golden_mm = round( fp32(A) @ fp32(B) ) to bf16, shape (1,1,512,128)
    # NOTE: matmul canNOT be exactly 0 error even with same dtype — the CPU and the chip add the
    #       128 products in a DIFFERENT ORDER, leaving a tiny residual. That is not a wiring bug.
    golden_mm = (A.to(torch.float32) @ B.to(torch.float32)).to(torch.bfloat16)
    show(
        rank,
        "made A, B (bf16) and golden_mm (bf16 in, fp32 accumulate, bf16 out)",
        value=golden_mm,
        why="matches how Tensix matmul works; still a tiny residual from add-order, not a bug",
    )

    # ACTION: SEND A sliced by ROWS (dim=2).  IN: A (1,1,512,128)  OUT: tA = per chip (1,1,32,128)
    # EXAMPLE: each chip gets a 32-row block of A
    tA = ttnn.from_torch(
        A,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=2),
    )
    show(rank, "SENT A to the chips, sliced by ROWS (dim=2)", value=tA, why="each chip gets a 32-row block of A")

    # ACTION: SEND B as a FULL COPY on every chip (Replicate).  IN: B (1,1,128,128)
    #         OUT: tB = per chip (1,1,128,128)  EXAMPLE: every chip has all of B
    tB = ttnn.from_torch(
        B, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    show(
        rank,
        "SENT B to the chips as a FULL COPY on every chip (Replicate)",
        value=tB,
        why="every chip needs all of B to multiply its rows",
    )

    # ACTION: chips MULTIPLY their rows by whole B.  IN: tA, tB  OUT: tmm = per chip (1,1,32,128)
    tmm = ttnn.matmul(tA, tB)
    show(
        rank,
        "asked the chips to MULTIPLY: tmm = tA @ tB",
        value=tmm,
        why="each chip multiplies its 32 rows by the whole B, all at once",
    )

    # ACTION: GATHER the row-blocks (dim=2) into the full result.
    # IN: tmm (pieces (1,1,32,128))  OUT: tmm_full = per chip (1,1,512,128)
    tmm_full = ttnn.all_gather(tmm, dim=2, cluster_axis=1, topology=ttnn.Topology.Linear)
    # ACTION: wait for completion.  IN: device  OUT: work done
    ttnn.synchronize_device(device)
    show(rank, "gathered the row-blocks back into the full result", value=tmm_full)
    # ACTION: compare + print error rate for MATMUL.  IN: golden_mm + full  OUT: ">>> MATMUL: ..."
    # EXAMPLE: ">>> MATMUL: rel_L2_err=0.6588% ..."
    report_error(golden_mm, full_tensor_to_cpu(tmm_full), "MATMUL", rank)

    # ========================= PHASE 3: ALL_GATHER ===========================
    show(rank, "starting PHASE 3: ALL_GATHER (just moving data, no math)")

    # ACTION: make x in bf16 = same data type as the device.  IN: (1,1,32,512)  OUT: x = bfloat16
    x = torch.randn(1, 1, TILE, TILE * NCOL).to(torch.bfloat16)
    # ACTION: CORRECT answer = x itself (all_gather only MOVES data, no math).
    # IN: x (bf16)  OUT: golden_x = x
    # EXPECT: pure data movement + same dtype -> error should be EXACTLY 0 if wiring is correct.
    golden_x = x
    show(rank, "made x (bf16); correct answer = x itself (pure move -> expect EXACT 0)", value=golden_x)

    # ACTION: SEND x in 16 column-slices (dim=3).  IN: x (1,1,32,512)  OUT: tx = per chip (1,1,32,32)
    tx = ttnn.from_torch(
        x,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3),
    )
    show(rank, "SENT x to the chips in 16 slices", value=tx)

    # ACTION: GATHER (dim=3) so every chip holds the whole x. This all_gather IS the op tested.
    # IN: tx (pieces (1,1,32,32))  OUT: tg = per chip (1,1,32,512)
    tg = ttnn.all_gather(tx, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)
    # ACTION: wait for completion.  IN: device  OUT: work done
    ttnn.synchronize_device(device)
    show(rank, "GATHERED: every chip now holds the whole x", value=tg)
    # ACTION: compare + print error rate for ALL_GATHER.  IN: golden_x + full  OUT: ">>> ALL_GATHER: ..."
    # EXAMPLE: ">>> ALL_GATHER: rel_L2_err=0.1662% ..."
    report_error(golden_x, full_tensor_to_cpu(tg), "ALL_GATHER", rank)

    # ============================== CLEAN UP =================================
    # ACTION: both computers wait at a meeting point so they finish together.
    # IN: -  OUT: both ranks synchronized
    ttnn.distributed_context_barrier()
    show(rank, "both computers waited at a barrier (a meeting point)", why="so we finish together, tidily")

    # ACTION: release the 16 chips.  IN: device  OUT: mesh closed
    ttnn.close_mesh_device(device)
    # ACTION: turn the fabric off so the next job starts clean.  IN: DISABLED  OUT: fabric off
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    show(rank, "turned the chips off and disabled the fabric", why="leave the cluster clean for the next run")

    # ACTION: on rank 0, print the final banner.  IN: -  OUT: one printed line
    if rank == 0:
        print("\n=== DONE. Read the [STEP n] lines above from top to bottom. ===", flush=True)


# ACTION: standard python entry point — run main() only when this file is executed directly.
# IN: -  OUT: calls main()
# EXAMPLE: `python3 bigmesh_ops_debug.py` -> __name__ == "__main__" -> main() runs
if __name__ == "__main__":
    main()
