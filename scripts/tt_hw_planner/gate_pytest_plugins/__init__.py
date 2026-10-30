# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugins the e2e gate loads with `-p`. This directory is put on the gate's PYTHONPATH, so it
holds plugins only -- anything else here would become importable by the model's own tests."""
