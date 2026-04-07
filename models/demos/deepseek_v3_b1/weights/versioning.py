# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Weight transform version constant.

Bump this whenever the preprocessing pipeline changes in a way that
invalidates existing cache artifacts (new shuffle, different TP concat
order, etc.).
"""

CURRENT_TRANSFORM_VERSION = 1
