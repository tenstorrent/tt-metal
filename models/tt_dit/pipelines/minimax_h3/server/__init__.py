# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""HTTP server for the MiniMax-H3 pipeline (see docs/API.md in the bring-up, and `app.py`).

Nothing is imported eagerly here: `config` and `engine` are device-free and cheap, but `app` builds
the server configuration at import time, so importing this package must not drag it in.
"""
