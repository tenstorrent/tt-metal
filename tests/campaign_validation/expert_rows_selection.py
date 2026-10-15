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
