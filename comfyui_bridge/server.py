# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
ComfyUI Bridge Server - Unix socket server for tt-metal integration.

Listens on Unix domain socket and dispatches operations to OperationHandler.
Provides low-latency IPC between ComfyUI frontend and tt-metal backend.
"""

import os
import sys
import socket
import signal
import logging
import argparse
from typing import Optional
from .protocol import receive_message, send_error, send_success
from .handlers import OperationHandler
from sdxl_config import SDXLConfig
from utils.logger import setup_logger


class ComfyUIBridgeServer:
    """
    Unix socket server that bridges ComfyUI and tt-metal SDXL backend.

    Architecture:
        ComfyUI (frontend) -> Unix socket -> Bridge Server -> SDXLRunner -> TT Hardware
    """

    def __init__(self, socket_path: str, config: Optional[SDXLConfig] = None):
        """
        Initialize bridge server.

        Args:
            socket_path: Path to Unix domain socket (e.g., /tmp/tt-comfy.sock)
            config: SDXL configuration (default: create from env/defaults)
        """
        self.socket_path = socket_path
        self.config = config or SDXLConfig()
        self.handler: Optional[OperationHandler] = None
        self.sock: Optional[socket.socket] = None
        self.running = False

        # Setup logging
        self.logger = setup_logger("ComfyUIBridge")
        self.logger.info(f"Bridge server initialized: socket={socket_path}")

    def start(self):
        """
        Start the bridge server.

        This will:
        1. Initialize OperationHandler (does NOT load model yet)
        2. Create Unix domain socket
        3. Listen for connections and handle requests
        """
        self.logger.info("Starting ComfyUI Bridge Server...")

        # Initialize handler (model loaded on first init_model request)
        self.handler = OperationHandler(config=self.config)
        self.logger.info("OperationHandler created")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Create Unix socket
        self._create_socket()

        # Accept connections
        self.running = True
        self.logger.info("Bridge server ready, waiting for connections...")

        try:
            while self.running:
                try:
                    # Accept connection (blocking)
                    client_sock, _ = self.sock.accept()
                    self.logger.debug("Client connected")

                    # Handle client in same thread (sequential processing)
                    # For concurrent requests, implement threading/async here
                    self._handle_client(client_sock)

                except Exception as e:
                    if self.running:
                        self.logger.error(f"Error accepting connection: {e}", exc_info=True)
                    else:
                        break  # Server shutting down

        finally:
            self._cleanup()

    def _create_socket(self):
        """Create and bind Unix domain socket."""
        # Remove existing socket file if present
        if os.path.exists(self.socket_path):
            self.logger.warning(f"Removing existing socket: {self.socket_path}")
            try:
                os.unlink(self.socket_path)
            except Exception as e:
                self.logger.error(f"Failed to remove existing socket: {e}")
                raise

        # Create socket
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.socket_path)
        self.sock.listen(5)

        # Set permissions to allow ComfyUI to connect
        os.chmod(self.socket_path, 0o777)

        self.logger.info(f"Listening on Unix socket: {self.socket_path}")

    def _handle_client(self, client_sock: socket.socket):
        """
        Handle a single client connection.

        Receives request, dispatches to handler, sends response.

        Args:
            client_sock: Connected client socket
        """
        try:
            # Receive request
            request = receive_message(client_sock)

            operation = request.get("operation")
            data = request.get("data", {})
            request_id = request.get("request_id")

            self.logger.info(f"Handling operation: {operation} (request_id={request_id})")

            # Dispatch to handler
            result = self._dispatch_operation(operation, data)

            # Send success response
            send_success(client_sock, result)
            self.logger.debug(f"Operation {operation} completed successfully")

        except Exception as e:
            # Send error response
            error_msg = str(e)
            self.logger.error(f"Operation failed: {error_msg}", exc_info=True)
            try:
                send_error(client_sock, error_msg)
            except Exception as send_err:
                self.logger.error(f"Failed to send error response: {send_err}")

        finally:
            # Close client connection
            try:
                client_sock.close()
            except:
                pass

    def _dispatch_operation(self, operation: str, data: dict) -> dict:
        """
        Dispatch operation to appropriate handler method.

        Args:
            operation: Operation name
            data: Operation data

        Returns:
            Result dictionary

        Raises:
            ValueError: If operation is unknown
        """
        if operation == "ping":
            return self.handler.handle_ping(data)

        elif operation == "init_model":
            return self.handler.handle_init_model(data)

        elif operation == "full_denoise":
            return self.handler.handle_full_denoise(data)

        elif operation == "denoise_only":
            return self.handler.handle_denoise_only(data)

        elif operation == "vae_decode":
            return self.handler.handle_vae_decode(data)

        elif operation == "vae_encode":
            return self.handler.handle_vae_encode(data)

        elif operation == "unload_model":
            return self.handler.handle_unload_model(data)

        # Additional operations for future expansion
        elif operation == "encode_prompt":
            raise NotImplementedError("encode_prompt operation not yet implemented")

        elif operation == "full_inference":
            raise NotImplementedError("full_inference operation not yet implemented")

        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

        # Close socket to unblock accept()
        if self.sock:
            try:
                self.sock.close()
            except:
                pass

    def _cleanup(self):
        """Cleanup resources on shutdown."""
        self.logger.info("Cleaning up resources...")

        # Cleanup handler
        if self.handler:
            try:
                self.handler.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up handler: {e}")

        # Close socket
        if self.sock:
            try:
                self.sock.close()
            except:
                pass

        # Remove socket file
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
                self.logger.info(f"Removed socket: {self.socket_path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove socket: {e}")

        self.logger.info("Bridge server shutdown complete")


def main():
    """Main entry point for bridge server."""
    parser = argparse.ArgumentParser(description="ComfyUI Bridge Server for Tenstorrent Hardware")
    parser.add_argument(
        "--socket-path",
        type=str,
        default=os.getenv("TT_COMFY_SOCKET", "/tmp/tt-comfy.sock"),
        help="Path to Unix domain socket (default: /tmp/tt-comfy.sock)",
    )
    parser.add_argument("--device-id", type=int, default=0, help="Device ID to use (default: 0)")
    parser.add_argument("--dev", action="store_true", help="Enable dev mode (single worker, fast warmup)")

    args = parser.parse_args()

    # Setup environment
    if args.dev:
        os.environ["SDXL_DEV_MODE"] = "true"

    # Create config
    config = SDXLConfig()

    # Note: device_id from CLI is for single-device mode
    # For multi-device (T3K), use environment variables
    if args.device_id != 0:
        os.environ["TT_VISIBLE_DEVICES"] = str(args.device_id)

    # Create and start server
    try:
        server = ComfyUIBridgeServer(socket_path=args.socket_path, config=config)
        server.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
