# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Backwards-compatible re-export.

`h2d_socket_sync` now lives in the model-agnostic `models.common.prefill_runner.h2d_service`. This
shim keeps existing in-package importers (e.g. tests/test_embedding_socket.py) working with a single
source of truth — there is no separate implementation here anymore.
"""

from models.common.prefill_runner.h2d_service import h2d_socket_sync  # noqa: F401
