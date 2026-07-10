# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Sparse-MLA (DSA) suite conftest. Registering the plugin HERE — instead of in the shared tests/
# conftest — scopes its perf/trace markers and the --ds-* MLA/indexer knobs to this subtree: they are
# only registered when the sparse tests are collected, so dense runs (e.g. tests/test_mla.py) never see
# them. Fixtures from the parent tests/ conftest (variant, config_only, mesh_device) still apply here.
pytest_plugins = ["models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_plugin"]
