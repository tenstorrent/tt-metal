import torch, ttnn
from loguru import logger
from ttnn.operations.rms_norm.perf_experiments.pass2_batch_rows.test_pass2_batch_rows import _make_case, _check
from ttnn.operations.rms_norm.perf_experiments.pass2_batch_rows import run_op

device = ttnn.open_device(device_id=0)
try:
    # num_rounds=1 (c_rows=32) — isolate the batched chain from cross-round recycling
    x, rstd, gamma, expected = _make_case(device, per_w_t=4, ht_local=32)
    out = run_op(
        x, rstd, gamma, variant="batch_both", per_w_t=4, ht_local=32, c_rows=32, has_gamma=True, kernel_iters=1
    )
    pcc = _check(out, expected, "batch_both nr=1")
    logger.info(f"batch_both num_rounds=1 kernel_iters=1 PCC={pcc:.5f} OK")
finally:
    ttnn.close_device(device)
