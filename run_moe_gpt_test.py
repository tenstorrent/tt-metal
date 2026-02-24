#!/usr/bin/env python3
"""Standalone script to run moe_gpt test, bypassing conftest.py hook issue."""
import ttnn

device = ttnn.open_device(device_id=0)

from tests.ttnn.nightly.unit_tests.operations.experimental.test_moe_gpt import run_test_moe_gpt

accuracy_metrics = run_test_moe_gpt(device, M=32, K=2880, N=2880, E=4, L=1, check_accuracy=True, dump_outputs=False)

passing = True
for (layer_id, expert_id), metrics in accuracy_metrics.items():
    pcc = metrics["pcc"]
    status = "PASS" if pcc >= 0.984 else "FAIL"
    if status == "FAIL":
        passing = False
    print(f"  Layer {layer_id}, Expert {expert_id}: PCC={pcc:.6f} [{status}]")

print(f"\nOverall: {'PASS' if passing else 'FAIL'}")

ttnn.close_device(device)
