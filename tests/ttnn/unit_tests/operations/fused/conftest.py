# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Session-scoped device fixture not applied: tests in this group use
# @pytest.mark.parametrize("device_params", ...) which conflicts with a
# shared session device. Refactor those tests first.
# Affected files: test_group_norm.py, test_group_norm_DRAM.py, test_softmax.py,
#   test_distributed_layernorm_exhaustive.py
# Additionally, several tests use mesh_device fixtures for distributed norms.
