# Refinement 1: PCC / relative RMS of the four unlocked golden cells (golden operands + reference).
import torch, ttnn
import eval.golden_tests.gen_qkv_causal_conv1d_silu.helpers as H
from eval.golden_tests.gen_qkv_causal_conv1d_silu.feature_spec import LOOSE_CASES
from eval.metrics import compute_metrics
from ttnn.operations.gen_qkv_causal_conv1d_silu import gen_qkv_causal_conv1d_silu

captured = {}
orig_check = H.check_output


def spy(out, ref, **kw):
    m = compute_metrics(out, ref)
    captured.setdefault("m", []).append(m)
    return orig_check(out, ref, **kw)


H.check_output = spy
H.gen_qkv_causal_conv1d_silu = gen_qkv_causal_conv1d_silu

device = ttnn.open_device(device_id=0)
try:
    for case in LOOSE_CASES:
        cid = case["extras"]["case_id"]
        if cid not in ("conv1d-012", "conv1d-013", "conv1d-018", "conv1d-019"):
            continue
        captured.clear()
        axes = {k: v for k, v in case.items() if k not in ("inputs", "extras")}
        H.run_gen_qkv_causal_conv1d_silu(case["inputs"], device=device, extras=case["extras"], **axes)
        ms = captured["m"]
        print(
            f"CELL {cid} "
            + " | ".join(f"pcc={m.pcc:.6f} rel_rms={m.rel_rms:.5f}" if hasattr(m, "rel_rms") else str(m) for m in ms)
        )
finally:
    ttnn.close_device(device)
