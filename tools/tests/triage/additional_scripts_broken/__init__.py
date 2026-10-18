# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Test fixture. A consuming repo's scripts directory is often a package, so this file exists to prove
# triage never loads it as a triage script: it would be imported under the module name '__init__',
# running someone's package init out of context. Deliberately left with no code to execute.
