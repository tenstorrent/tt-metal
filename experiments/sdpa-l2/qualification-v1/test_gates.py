# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression checks for qualification scoring, not kernel tests."""

import torch

from qualify import assess

torch.set_num_threads(2)
torch.manual_seed(29)
c = dict(heads=2, group="normal", distribution="normal", kv_len=1024)
e = torch.randn(1, 2, 1024, 128, dtype=torch.float64)
v = torch.randn_like(e)

_, failures = assess(e, e, c, "accurate", v)
assert not failures

# A bad head cannot be hidden by a good head.
a = e.clone()
a[0, 1] *= 1.006
heads, failures = assess(a, e, c, "accurate", v)
assert heads[0]["failed_gates"] == [] and "head1:l2" in failures

# An isolated bad row can fail even when aggregate L2 passes.
a = e.clone()
a[0, 1, 0] *= 1.05
heads, failures = assess(a, e, c, "accurate", v)
assert heads[1]["l2_pct"] < 0.5 and "head1:row_max" in failures

a = e.clone()
a[0, 1, :16] *= 1.025
heads, failures = assess(a, e, c, "accurate", v)
assert heads[1]["l2_pct"] < 0.5 and "head1:row_p99" in failures

# A nonzero result at an exactly zero reference cannot silently become NaN/pass.
zero_case = dict(c, distribution="zero_v")
_, failures = assess(torch.zeros_like(e), torch.zeros_like(e), zero_case, "accurate", v)
assert not failures
_, failures = assess(torch.ones_like(e), torch.zeros_like(e), zero_case, "accurate", v)
assert "head0:zero_v_exact" in failures

constant = torch.ones_like(e)
constant_case = dict(c, distribution="constant_v")
_, failures = assess(constant, constant, constant_case, "fast", v)
assert not failures
_, failures = assess(constant + 2 / 128, constant, constant_case, "fast", v)
assert "head0:structural_one_ulp" in failures

# Ordinary relative L2 can pass while the common-V residual guard catches drift.
common_case = dict(c, group="stress", distribution="common_v", offset=32)
common = 32 + 0.001 * e
heads, failures = assess(common + 0.01, common, common_case, "accurate", v)
assert heads[0]["l2_pct"] < 0.5 and "head0:common_v_residual" in failures

print("QUALIFICATION_GATES_SELF_TEST_PASS")
