#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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
"""

from triage import triage_singleton, ScriptConfig, run_script
from parse_inspector_logs import get_data as get_logs_data, get_log_directory
import asyncio
import capnp
import os
import threading
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
                    message = e.description[len(InspectorRpcController.REMOTE_EXCEPTION_TEXT_START):]
                    raise InspectorRpcRemoteException(message)
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
        self.__directory = directory
        self.__methods = inspector_capnp.capnp_scheme.Inspector.schema.methods
        if not os.path.exists(directory) or not os.path.exists(os.path.join(directory, "getPrograms.capnp.bin")):
            raise ValueError(f"Serialized RPC data not found in directory {directory}")

    def __getattr__(self, method_name: str):
        if method_name in self.__methods:
            serialized_path = os.path.join(self.__directory, f"{method_name}.capnp.bin")
            if not os.path.exists(serialized_path):
                raise InspectorUnserializedMethod(f"Serialized file for method {method_name} not found at {serialized_path}")
            method_name_cap = method_name[0].upper() + method_name[1:]
            with open(serialized_path, "rb") as f:
                results_schema = self.__methods[method_name].result_type
                results_name = f"{method_name_cap}Results"
                results_struct = capnp.lib.capnp._StructModule(results_schema, results_name)
                message = results_struct.read_packed(f)
                method = lambda : message
                setattr(self, method_name, method)
                return method
        else:
            raise AttributeError(f"Method {method_name} not found in Inspector RPC interface")

    def stop(self):
        # Do nothing
        pass

# TODO: parse_inspector_logs types should have different field names and different return types (not dictionary, but named tuple with array of elements)


@triage_singleton
def run(args, context) -> InspectorData:
    log_directory = args["--inspector-log-path"]
    rpc_port = args["--inspector-rpc-port"]
    rpc_host = args["--inspector-rpc-host"]

    # First try to connect to Inspector RPC
    try:
        return InspectorRpcController(rpc_host, rpc_port)
    except:
        pass

    # Check for Inspector log directory
    log_directory = get_log_directory(log_directory)
    if not os.path.exists(log_directory):
        raise ValueError(f"Log directory {log_directory} does not exist. Please provide a valid path.")

    # Try to load serialized RPC data
    try:
        return InspectorRpcSerialized(log_directory)
    except:
        # Fall back to reading Inspector logs
        return get_logs_data(log_directory)

if __name__ == "__main__":
    run_script()
