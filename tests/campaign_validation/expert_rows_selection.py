"""Select unchanged original cases for the Device2 adoption comparison."""
import json
import os
from pathlib import Path

MODE = os.environ["FLASH_EXPERT_ROWS_CASES"]
EXPECTED = {"blackhole": 10, "n150": 1, "n300": 1}


def pytest_collection_modifyitems(session, config, items):
    keep, drop = [], []
    for item in items:
        p = item.callspec.params
        name = item.originalname
        if MODE == "blackhole":
            match = name in (
                "test_moe_compute_expert_rows_deterministic",
                "test_moe_compute_expert_rows_many_jobs_per_group",
            ) or (name == "test_moe_compute_expert_rows_row_tiles" and p["shape"] == "flash_next_e32_t128")
        elif MODE == "n150":
            match = (
                name == "test_moe_compute_expert_rows_row_tiles"
                and p["shape"] == "deepseek_e16_t128"
                and p["row_tiles"] == 2
                and p["kind"] == "hot_last"
            )
        else:
            assert MODE == "n300"
            match = name == "test_moe_compute_expert_rows_shapes" and p["shape"] == "deepseek_e16_t32"
        (keep if match else drop).append(item)
    items[:] = keep
    config.hook.pytest_deselected(items=drop)
    assert len(keep) == EXPECTED[MODE], [item.nodeid for item in keep]
    Path("generated").mkdir(exist_ok=True)
    Path("generated/flash-expert-collection.json").write_text(json.dumps([item.nodeid for item in keep], indent=2))


def pytest_runtest_call(item):
    import ttnn

    device = item.funcargs["mesh_device"]
    grid = device.compute_with_storage_grid_size()
    cores = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.RISCV_0_default)
    receipt = {
        "nodeid": item.nodeid,
        "grid": [grid.x, grid.y],
        "preferred_dram_workers": [[core.x, core.y] for core in cores],
    }
    print("FLASH_PLACEMENT " + json.dumps(receipt, sort_keys=True), flush=True)
    with Path("generated/flash-placement.jsonl").open("a") as stream:
        stream.write(json.dumps(receipt, sort_keys=True) + "\n")
