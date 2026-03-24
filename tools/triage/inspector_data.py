#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    inspector_data [--inspector-rpc-port=<inspector_rpc_port>] [--inspector-rpc-host=<inspector_rpc_host>] [--inspector-log-path=<inspector_log_path>]

Options:
    --inspector-rpc-port=<inspector_rpc_port>  Port for the inspector RPC server. [default: 50051]
    --inspector-rpc-host=<inspector_rpc_host>  Host for the inspector RPC server. [default: localhost]
    --inspector-log-path=<inspector_log_path>  Path to the inspector log directory.

Description:
    Provides inspector data for other scripts.
    This script will try to connect to Inspector RPC.
    If RPC is not available, it will try to load serialized RPC data from the log directory.
    If RPC data is not available, it will try to parse inspector logs.

Owner:
    tt-vjovanovic
"""

from triage import triage_singleton, ScriptConfig, run_script
from parse_inspector_logs import get_data as get_logs_data, get_log_directory
import asyncio
import capnp
import os
from pathlib import Path
import threading
import time
import inspector_capnp

script_config = ScriptConfig(
    data_provider=True,
)

InspectorData = inspector_capnp.Inspector


class InspectorException(Exception):
    pass


class InspectorRpcRemoteException(InspectorException):
    pass


class InspectorRpcController(InspectorData):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.running = None
        self.queue = asyncio.Queue()
        self.loop = asyncio.new_event_loop()
        self.task = self.loop.create_task(self.__connect_client())
        self.background_thread = threading.Thread(target=self.__asyncio_background)
        self.background_thread.daemon = True
        self.background_thread.start()
        while not self.running:
            if self.task.done():
                self.stop()
                exception = self.task.exception()
                assert exception is not None
                raise exception
            time.sleep(0.01)  # Small sleep to avoid busy-waiting

    def __del__(self):
        if self.running:
            self.stop()

    def __asyncio_background(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def __connect_client(self):
        try:
            async with capnp.kj_loop():
                try:
                    connection = await capnp.AsyncIoStream.create_connection(host=self.host, port=self.port)
                    client = capnp.TwoPartyClient(connection)
                    self.inspector_rpc = client.bootstrap().cast_as(inspector_capnp.capnp_scheme.Inspector)
                    self.running = True
                except:
                    self.loop.stop()
                    raise
                while True:
                    request = await self.queue.get()

                    # If the request is None, stop has been initiated
                    if request is None:
                        self.running = False
                        break

        finally:
            self.loop.stop()

    async def __call_rpc(self, method_name: str, *args, **kwargs):
        if not self.running:
            raise RuntimeError("RPC client is not running")

        method = getattr(self.inspector_rpc, method_name)
        return await method(*args, **kwargs)

    REMOTE_EXCEPTION_TEXT_START = "remote exception: e.what() = "

    def __getattr__(self, name: str):
        def method(*args, **kwargs):
            try:
                return asyncio.run_coroutine_threadsafe(self.__call_rpc(name, *args, **kwargs), self.loop).result()
            except capnp.lib.capnp.KjException as e:
                if e.description.startswith(InspectorRpcController.REMOTE_EXCEPTION_TEXT_START):
                    message = e.description[len(InspectorRpcController.REMOTE_EXCEPTION_TEXT_START) :]
                    raise InspectorRpcRemoteException(message) from e
                raise InspectorException(f"Inspector RPC call '{name}' failed: {e.description}") from e
            except Exception as e:
                raise InspectorException(f"Inspector RPC call '{name}' failed: {e}") from e

        return method

    async def __async_stop(self):
        await self.queue.put(None)

    def stop(self):
        if self.running:
            asyncio.run_coroutine_threadsafe(self.__async_stop(), self.loop).result()
            self.background_thread.join()


class InspectorUnserializedMethod(InspectorException):
    pass


class InspectorRpcSerialized(InspectorData):
    def __init__(self, directory: str):
        self.__directory = Path(directory)
        self.__methods = inspector_capnp.capnp_scheme.Inspector.schema.methods
        if not self.__directory.exists() or not (self.__directory / "getPrograms.capnp.bin").exists():
            raise ValueError(f"Serialized RPC data not found in directory {directory}")

    def __getattr__(self, method_name: str):
        if method_name in self.__methods:
            serialized_path = self.__directory / f"{method_name}.capnp.bin"
            if not serialized_path.exists():
                raise InspectorUnserializedMethod(
                    f"Serialized file for method {method_name} not found at {serialized_path}"
                )
            method_name_cap = method_name[0].upper() + method_name[1:]
            with open(serialized_path, "rb") as f:
                results_schema = self.__methods[method_name].result_type
                results_name = f"{method_name_cap}Results"
                results_struct = capnp.lib.capnp._StructModule(results_schema, results_name)
                message = results_struct.read_packed(f)
                method = lambda: message
                setattr(self, method_name, method)
                return method
        else:
            raise AttributeError(f"Method {method_name} not found in Inspector RPC interface")

    def stop(self):
        # Do nothing
        pass


# TODO: parse_inspector_logs types should have different field names and different return types (not dictionary, but named tuple with array of elements)

# Default port from docopt; when this is used and rank env is set, effective port = base + rank (match C++ rtoptions).
_DEFAULT_INSPECTOR_RPC_PORT = 50051


def _get_rank_from_env() -> int:
    """Return MPI/mesh rank from process env for rank-aware Inspector port, or -1 if not set.

    Precedence (highest to lowest):
      1. OMPI_COMM_WORLD_RANK  -- OpenMPI standard
      2. PMI_RANK              -- MPICH / Hydra / Slurm PMI
      3. SLURM_PROCID          -- Slurm native (without PMI layer)
      4. PMIX_RANK             -- PMIx-aware launchers (OpenPMIx, PRRTE)
      5. TT_MESH_HOST_RANK     -- TT-specific fallback (may be duplicated across ranks)
    """
    for var in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "PMIX_RANK", "TT_MESH_HOST_RANK"):
        val = os.environ.get(var)
        if val is not None:
            try:
                return int(val)
            except (ValueError, OverflowError):
                continue
    return -1


@triage_singleton
def run(args, context) -> InspectorData:
    log_directory = args["--inspector-log-path"]
    rpc_port = args["--inspector-rpc-port"]
    rpc_host = args["--inspector-rpc-host"]
    try:
        rpc_port_int = int(rpc_port)
        if rpc_port_int <= 0 or rpc_port_int > 65535:
            raise ValueError(f"port out of range: {rpc_port_int}")
    except Exception as exc:
        raise ValueError(
            f"Invalid --inspector-rpc-port value '{rpc_port}'. Expected integer in range 1-65535."
        ) from exc

    # When default port was used (not explicitly overridden), use rank-aware port so timeout-invoked
    # triage on a non-zero rank connects to that rank's Inspector (C++ uses base + rank).
    if rpc_port == str(_DEFAULT_INSPECTOR_RPC_PORT):
        rank = _get_rank_from_env()
        if rank >= 0:
            effective_port = _DEFAULT_INSPECTOR_RPC_PORT + rank
            if effective_port > 65535:
                raise ValueError(
                    f"Inspector RPC port overflow: base_port={_DEFAULT_INSPECTOR_RPC_PORT} + rank={rank} exceeds 65535. "
                    "Set --inspector-rpc-port to a lower base port or reduce rank count."
                )
            rpc_port_int = effective_port

    # First try to connect to Inspector RPC
    rpc_error: Exception | None = None
    try:
        return InspectorRpcController(rpc_host, rpc_port_int)
    except Exception as exc:
        rpc_error = exc

    # Check for Inspector log directory
    log_directory = get_log_directory(log_directory)
    if not Path(log_directory).exists():
        raise ValueError(
            f"\n\tLog directory {log_directory} does not exist."
            f"\n\tMetal runtime is not running. Do not kill host process, but open triage in parallel."
            f"\n\tIf you have generated inspector logs, you can load them with --inspector-log-path"
            f"\n\tor defining TT_METAL_LOGS_PATH environment variable."
        )

    # Try to load serialized RPC data
    try:
        return InspectorRpcSerialized(log_directory)
    except Exception as serialized_exc:
        rpc_error_text = f"{type(rpc_error).__name__}: {rpc_error}" if rpc_error is not None else "unknown RPC failure"
        serialized_error_text = f"{type(serialized_exc).__name__}: {serialized_exc}"
        raise InspectorException(
            "There is no Inspector RPC data, cannot continue. "
            "Use --inspector-log-path to load saved Inspector data, or --inspector-rpc-host/--inspector-rpc-port "
            "to connect to a live Inspector. Ensure Inspector was enabled in Metal with TT_METAL_INSPECTOR=1 and "
            "TT_METAL_INSPECTOR_RPC=1.\n"
            f"RPC connection failure: {rpc_error_text}\n"
            f"Serialized data failure: {serialized_error_text}"
        )


if __name__ == "__main__":
    run_script()
