import itertools
import ttnn
import ttnn.operations.g6_zeros.g6_zeros_program_descriptor as pd

dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

print("tile sizes", [pd.page_bytes_of(d) for d in (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b)])


def simulate(plan):
    NB = plan.num_dram_banks
    P = plan.tensor_pages
    counts = bytearray(P)
    max_writes = 0
    for w in plan.cores:
        end = w.slot_begin + w.slots
        writes = 0
        for k in range(plan.banks_per_core):
            bank = w.bank_set_base + (w.first_bank_offset + k) % plan.banks_per_core
            slot = w.slot_begin
            while slot < end:
                first = slot * NB + bank
                if first >= P:
                    break
                block = min(w.block_slots, end - slot)
                contig = (P - first - 1) // NB + 1
                run = min(block, contig)
                for j in range(run):
                    counts[first + j * NB] += 1
                run_bytes = run * plan.page_bytes
                writes += -(-run_bytes // plan.zero_source_bytes)
                slot += w.block_slots
        max_writes = max(max_writes, writes)
    bad = [i for i, c in enumerate(counts) if c != 1]
    return bad, max_writes


cases = [
    ((16, 12, 128, 128), ttnn.float32),
    ((8, 12, 128, 128), ttnn.float32),
    ((32831, 1, 32, 256), ttnn.bfloat8_b),
    ((1, 4, 256, 128), ttnn.bfloat16),
    ((1, 1, 32, 32), ttnn.float32),
    ((1, 1, 32, 96), ttnn.float32),
    ((3, 1, 64, 160), ttnn.bfloat16),
]
n = 0
for bpc, mwc, bs, grid in itertools.product([1, 2, 8], [None, 8, 16, 64], [None, 1, 3], [(11, 10), (13, 10)]):
    pd.BANKS_PER_CORE, pd.MAX_WRITER_CORES, pd.BLOCK_SLOTS = bpc, mwc, bs
    for shape, dt in cases:
        try:
            plan = pd.make_plan(shape, dt, grid, 8)
        except RuntimeError as e:
            continue
        bad, mw = simulate(plan)
        assert not bad, (bpc, mwc, bs, grid, shape, bad[:5])
        n += 1
pd.BANKS_PER_CORE, pd.MAX_WRITER_CORES, pd.BLOCK_SLOTS = 1, None, None
for shape, dt in cases[:4]:
    plan = pd.make_plan(shape, dt, (11, 10), 8)
    bad, mw = simulate(plan)
    slots = [w.slots for w in plan.cores]
    print(
        shape,
        dt,
        "pages",
        plan.tensor_pages,
        "W",
        plan.num_writer_cores,
        "slots",
        min(slots),
        max(slots),
        "zsrc",
        plan.zero_source_bytes,
        "max writes/core",
        mw,
    )
print("coverage OK for", n, "plans")
ttnn.close_mesh_device(dev)
