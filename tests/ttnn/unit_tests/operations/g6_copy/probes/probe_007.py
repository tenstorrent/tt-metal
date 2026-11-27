# Verifier probe: does the Phase 0 kernel pair already move every TARGET - SUPPORTED cell bit-exact?
# SUPPORTED / EXCLUSIONS are widened in memory only; the op file is not edited.
import sys, torch, ttnn
import ttnn.operations.g6_copy  # noqa: F401

op = sys.modules["ttnn.operations.g6_copy.g6_copy"]
from ttnn.operations.g6_copy import g6_copy, describe_plan
from eval.golden_tests.g6_copy.feature_spec import PRODUCTION_CASES, LOOSE_CASES
from eval.golden_tests.g6_copy import helpers as H

op.SUPPORTED["dtype"][:] = [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.int32, ttnn.uint32]
op.SUPPORTED["layout"][:] = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]
op.EXCLUSIONS[:] = []

device = ttnn.open_device(device_id=0)
ok = bad = 0
try:
    cases = [(c["inputs"], c["dtype"], c["layout"], {}) for c in PRODUCTION_CASES]
    cases += [(c["inputs"], c["dtype"], c["layout"], c.get("extras", {})) for c in LOOSE_CASES]
    cases += [
        (((2, 12, 128, 128),), ttnn.int32, ttnn.TILE_LAYOUT, {}),
        (((2, 12, 128, 128),), ttnn.uint32, ttnn.TILE_LAYOUT, {}),
        (((1, 3, 2560),), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, {}),
    ]
    for inputs, dtype, layout, extras in cases:
        shape = inputs[0]
        try:
            gen = torch.Generator().manual_seed(7)
            if extras.get("input_gen") == "adversarial":
                src = H.make_adversarial_source(dtype, shape, generator=torch.Generator().manual_seed(extras["seed"]))
            else:
                src = H.make_source(dtype, shape, generator=gen)
            dst = H.make_source(dtype, shape, generator=gen)
            H.copy_and_check(
                g6_copy,
                src,
                dst,
                dtype=dtype,
                layout=layout,
                device=device,
                gate=H.default_gate(dtype),
                poison_padding=extras.get("poison_padding"),
            )
            t = H.create_ttnn_input_tensor(H.make_source(dtype, shape), device, dtype=dtype, layout=layout)
            p = describe_plan(t)
            print(
                f"CASE PASS {shape} {dtype} {layout} pages={p['tensor_pages']} page_bytes={p['page_bytes']} "
                f"page_size={t.buffer_page_size()} cores={p['num_cores']} block={p['block_pages']} plan={p['plan']} l1={p['l1_bytes_per_core']} extras={extras}"
            )
            ok += 1
        except Exception as e:
            print(f"CASE FAIL {shape} {dtype} {layout} extras={extras}: {type(e).__name__}: {str(e)[:300]}")
            bad += 1
finally:
    ttnn.close_device(device)
print(f"SUMMARY pass={ok} fail={bad}")
