# Odd row widths, tile-aligned planes and forced plans under the widened SUPPORTED["alignment"].
import sys, torch, ttnn

sys.path.insert(0, "tests/ttnn/unit_tests/operations/g6_indexed_fill")
from test_g6_indexed_fill import make_inputs, reference, to_device, read_back
from ttnn.operations.g6_indexed_fill import g6_indexed_fill, describe_plan, PLAN_CAPS, INPUT_TAGGERS

UMAX = 2**32 - 1
device = ttnn.open_device(device_id=0)
defaults = dict(PLAN_CAPS)
cases = [
    ("w17", (5, 1, 1, 17), [4, 1], {}),
    ("w17_oob", (5, 1, 1, 17), [UMAX, 2], {}),
    ("w50_r6", (3, 2, 3, 50), [2], {}),
    ("tile_aligned_r32", (3, 1, 32, 32), [1], {}),
    ("tile_aligned_K3", (4, 1, 32, 64), [3, 0, UMAX], {}),
    ("w100_chunks", (5, 2, 1, 100), [4, 1], {"max_row_chunk_bytes": 64}),
    (
        "w100_combined",
        (5, 2, 1, 100),
        [4, 1],
        {"ids_chunk_cap_entries": 1, "max_row_chunk_bytes": 64, "max_work_cores": 2, "staging_budget_bytes": 512},
    ),
    ("w1", (9, 1, 1, 1), [8, 0, 3], {}),
    (
        "w1_combined",
        (9, 1, 1, 1),
        [8, 0, 3],
        {"ids_chunk_cap_entries": 1, "max_work_cores": 2, "staging_budget_bytes": 128},
    ),
]
try:
    for name, xs, ids, caps in cases:
        PLAN_CAPS.clear()
        PLAN_CAPS.update(defaults)
        PLAN_CAPS.update(caps)
        torch.manual_seed(1)
        vs = (len(ids),) + xs[1:]
        tag = INPUT_TAGGERS["alignment"](((len(ids),), xs, vs), {})
        x, s = make_inputs(xs, vs)
        xd, sd = to_device(x, device), to_device(s, device)
        idd = to_device(torch.tensor(ids, dtype=torch.int64), device)
        p = describe_plan(idd, xd, sd)
        p.pop("roles")
        out = g6_indexed_fill(idd, xd, sd)
        ok = torch.equal(read_back(out), reference(ids, x, s))
        ok_in = torch.equal(read_back(xd), x) and torch.equal(read_back(sd), s)
        print(
            ("OK " if ok and ok_in else "FAIL ") + name,
            tag,
            {
                k: p[k]
                for k in (
                    "regime",
                    "num_cores",
                    "row_chunk_bytes",
                    "row_chunks_per_page",
                    "block_units",
                    "staging_depth",
                    "max_blocks_per_core",
                )
            },
        )
finally:
    PLAN_CAPS.clear()
    PLAN_CAPS.update(defaults)
    ttnn.close_device(device)
