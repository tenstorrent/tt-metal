import pytest

ids = [l.strip() for l in open("/tmp/cq/mm.txt")]


class P:
    def pytest_collection_modifyitems(self, items):
        for it in items:
            if it.nodeid in ids:
                cs = it.callspec.params
                mc = cs["matmul_config"]
                td = mc.tile_dimensions
                print(
                    "ID",
                    it.nodeid,
                    cs["math_fidelity"].name,
                    "thr",
                    cs.get("throttle"),
                    "nb",
                    cs.get("num_blocks"),
                    f"r{td.rt_dim}c{td.ct_dim}k{td.kt_dim}",
                    mc.formats.input_format.name,
                    "->",
                    mc.formats.output_format.name,
                    mc.dest_sync.name,
                    mc.dest_acc.name,
                )


pytest.main(["-q", "--collect-only", "-p", "no:randomly"] + ids, plugins=[P()])
