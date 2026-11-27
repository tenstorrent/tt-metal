import torch, ttnn
import ttnn.operations.g6_reshard.g6_reshard_program_descriptor as pd
from ttnn.operations.g6_reshard import g6_reshard, last_plan
from tests.ttnn.unit_tests.operations.g6_reshard.test_g6_reshard import _configs, _make_input

device = ttnn.open_device(device_id=0)
try:
    in_cfg, out_cfg = _configs(device, "P")
    x, xh = _make_input(device, in_cfg)
    for name, knobs in [
        ("default", {}),
        ("runs_lag2", {"ISSUER_SPLIT": "runs", "ISSUER_LAG_RUNS": (0.0, 2.0)}),
        ("stagger", {"PIECE_ORDER": "stagger"}),
        ("local_last", {"PIECE_ORDER": "local_last"}),
    ]:
        saved = {k: getattr(pd, k) for k in knobs}
        for k, v in knobs.items():
            setattr(pd, k, v)
        g6_reshard(x, out_cfg)
        plan = last_plan()
        print("SET", name)
        for role in plan["roles"][:3] + plan["roles"][9:10]:
            print("  ", role["core"], role["block"], [(i["runs"], i["pieces"], i["segments"]) for i in role["issuers"]])
        for k, v in saved.items():
            setattr(pd, k, v)
finally:
    ttnn.close_device(device)
