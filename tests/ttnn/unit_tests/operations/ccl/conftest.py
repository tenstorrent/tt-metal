# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Session-scoped device fixture not applied: CCL tests use mesh_device
# fixtures (multi-device tests) and open_mesh_device directly. These require
# a shared mesh fixture, not a single-device shared fixture.
