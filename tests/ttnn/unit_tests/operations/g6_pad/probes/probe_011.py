import torch, ttnn
from ttnn.operations.g6_pad import SUPPORTED
from eval.golden_tests.g6_pad import feature_spec as fs
from eval.golden_tests.g6_pad.helpers import run_g6_pad
from eval.feature_matrix import apply_input_taggers
from ttnn.operations.g6_pad import INPUT_TAGGERS

for k, v in fs.TARGET.items():
    if k != "layout":
        SUPPORTED[k] = list(v)
device = ttnn.open_device(device_id=0)
ok = bad = 0
for cid, s in fs.SCENARIOS.items():
    if s["layout"] != ttnn.TILE_LAYOUT:
        continue
    axes = apply_input_taggers(INPUT_TAGGERS, (s,), {})
    new = any(
        axes[k] not in v
        for k, v in {
            "dtype": [ttnn.bfloat16],
            "memory_config": ["dram"],
            "alignment": ["w_non_aligned", "h_non_aligned"],
            "rank": [3, 4],
            "padded_dims": ["h", "hw"],
        }.items()
    )
    if not new:
        continue
    try:
        run_g6_pad((s,), device=device, **axes)
        r = "PASS"
        ok += 1
    except Exception as e:
        r = f"FAIL {type(e).__name__}: {str(e)[:200]}"
        bad += 1
    print(
        f"CASE {cid} {s['shape']} {s['dtype']} {s['memory_config']} {axes['padded_dims']} {axes['alignment']} r{axes['rank']}: {r}"
    )
print(f"SUMMARY pass={ok} fail={bad}")
ttnn.close_device(device)
