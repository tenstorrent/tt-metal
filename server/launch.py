#!/usr/bin/env python3
"""Launcher for the Wan I2V server.

Runs uvicorn normally, then hard-exits with os._exit(0) after uvicorn's
graceful shutdown returns. This skips Python/C++ static destructors
where tt-metal's UMD throws `std::runtime_error` on a double mutex
unlock (SIGABRT / core dump) during interpreter teardown.

Our lifespan's finally block has already done the real cleanup
(close_mesh_device + set_fabric_config DISABLED), so bypassing the
static-destructor phase is safe.

Usage (from the tt-metal repo root):
    python -m server.launch
    # or
    python server/launch.py

Env overrides:
    HOST (default 0.0.0.0)
    PORT (default 8000)
    UVICORN_LOG_LEVEL (default info)
    UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN (default 60)
"""

from __future__ import annotations

import os
import sys

import uvicorn


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    log_level = os.environ.get("UVICORN_LOG_LEVEL", "info")
    grace = int(os.environ.get("UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN", "60"))

    config = uvicorn.Config(
        "server.server:app",
        host=host,
        port=port,
        workers=1,
        log_level=log_level,
        timeout_graceful_shutdown=grace,
        lifespan="on",
    )
    server = uvicorn.Server(config)

    try:
        server.run()
    finally:
        # Skip interpreter teardown: tt-metal's MetalContext static
        # destructor fires pthread_mutex_unlock on an already-unlocked
        # CHIP_IN_USE_* mutex and aborts. Our lifespan already closed
        # the mesh cleanly, so no state is lost here.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
