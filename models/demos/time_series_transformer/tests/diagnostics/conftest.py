# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Diagnostic scripts are developer tools, not part of the CI test suite.
# Exclude them from automatic collection to avoid import errors from
# stale references after refactors.
collect_ignore_glob = ["test_*.py"]
