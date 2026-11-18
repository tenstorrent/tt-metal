"""NAN49: what does the production fabric self-test (hy4_ccl_safe.hy4_fabric_selftest) say on this mesh right now?"""
import os

from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt.hy4_ccl_safe import hy4_fabric_selftest


def main():
    if os.environ.get("MESH4") == "1":
        from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh

        md, full = open_mesh(), None
    else:
        full, md = open_row()
    try:
        for i in range(2):
            r = hy4_fabric_selftest(md, n_calls=6)
            print(
                "RES",
                {
                    "call": i,
                    "per_op": r.per_op,
                    "num_links": r.num_links,
                    "pool_ag": r.pool_ag,
                    "sec": round(r.seconds, 1),
                },
                flush=True,
            )
    finally:
        (close_mesh(md) if full is None and os.environ.get("MESH4") == "1" else close_row(full, md))


if __name__ == "__main__":
    main()
