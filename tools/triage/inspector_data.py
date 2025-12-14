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
"""

from dataclasses import dataclass
from triage import triage_singleton, ScriptConfig, run_script
from parse_inspector_logs import get_log_directory
import asyncio
import capnp
import os
import threading
import inspector_capnp

script_config = ScriptConfig(
    data_provider=True,
)


class InspectorException(Exception):
    pass


class InspectorRpcRemoteException(InspectorException):
    pass


class InspectorRpcController:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._running = None
        self._queue = asyncio.Queue()
        self._loop = asyncio.new_event_loop()
        self._task = self._loop.create_task(self.__connect_client())
        self._background_thread = threading.Thread(target=self.__asyncio_background)
        self._background_thread.daemon = True
        self._background_thread.start()
        while not self._running:
            if self._task.done():
                self.stop()
                exception = self._task.exception()
                assert exception is not None
                raise exception
        self.runtime_rpc: inspector_capnp.RuntimeInspector = self.__get_rpc_channel_wrapper("RuntimeInspector", inspector_capnp.RuntimeInspector)  # type: ignore
        self.ttnn_rpc: inspector_capnp.TtnnInspector = self.__get_rpc_channel_wrapper("TtnnInspector", inspector_capnp.TtnnInspector)  # type: ignore

    def __del__(self):
        if self._running:
            self.stop()

    def __asyncio_background(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def __connect_client(self):
        try:
            async with capnp.kj_loop():
                try:
                    connection = await capnp.AsyncIoStream.create_connection(host=self.host, port=self.port)
                    client = capnp.TwoPartyClient(connection)
                    self.inspector_rpc: inspector_capnp.InspectorChannelRegistry = self.__get_rpc_wrapper(
                        client.bootstrap(), inspector_capnp.InspectorChannelRegistry
                    )  # type: ignore
                    self._running = True
                except:
                    self._loop.stop()
                    raise
                while True:
                    request = await self._queue.get()

                    # If the request is None, stop has been initiated
                    if request is None:
                        self._running = False
                        break

        finally:
            self._loop.stop()

    def __get_rpc_channel_wrapper(self, channel_name: str, channel_type):
        channel = self.inspector_rpc.getChannel(channel_name).channel
        return self.__get_rpc_wrapper(channel, channel_type)

    def __get_rpc_wrapper(self, channel, channel_type):
        class ChannelWrapper:
            def __init__(self, rpc, controller: "InspectorRpcController"):
                self.rpc = rpc
                self.controller = controller

            def __getattr__(self, name: str):
                return self.controller._get_async_method_call(self.rpc, name)

        rpc = channel.cast_as(channel_type)
        return ChannelWrapper(rpc, self)

    async def __call_rpc(self, rpc, method_name: str, *args, **kwargs):
        if not self._running:
            raise RuntimeError("RPC client is not running")

        method = getattr(rpc, method_name)
        return await method(*args, **kwargs)

    REMOTE_EXCEPTION_TEXT_START = "remote exception: e.what() = "

    def _get_async_method_call(self, rpc, name: str):
        def method(*args, **kwargs):
            try:
                return asyncio.run_coroutine_threadsafe(
                    self.__call_rpc(rpc, name, *args, **kwargs), self._loop
                ).result()
            except capnp.lib.capnp.KjException as e:
                if e.description.startswith(InspectorRpcController.REMOTE_EXCEPTION_TEXT_START):
                    message = e.description[len(InspectorRpcController.REMOTE_EXCEPTION_TEXT_START) :]
                    raise InspectorRpcRemoteException(message)

        return method

    async def __async_stop(self):
        await self._queue.put(None)

    def stop(self):
        if self._running:
            asyncio.run_coroutine_threadsafe(self.__async_stop(), self._loop).result()
            self._background_thread.join()


class InspectorUnserializedMethod(InspectorException):
    pass


class InspectorRpcSerialized:
    def __init__(self, directory: str, rpc_type):
        self.__directory = directory
        self.__methods = rpc_type.schema.methods
        if not os.path.exists(directory):
            raise ValueError(f"Serialized RPC data not found in directory {directory}")

    def __getattr__(self, method_name: str):
        if method_name in self.__methods:
            serialized_path = os.path.join(self.__directory, f"{method_name}.capnp.bin")
            if not os.path.exists(serialized_path):
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


@dataclass
class InspectorData:
    inspector_channel_registry: inspector_capnp.InspectorChannelRegistry
    runtime_rpc: inspector_capnp.RuntimeInspector
    ttnn_rpc: inspector_capnp.TtnnInspector


@triage_singleton
def run(args, context) -> InspectorData:
    log_directory = args["--inspector-log-path"]
    rpc_port = args["--inspector-rpc-port"]
    rpc_host = args["--inspector-rpc-host"]

    # First try to connect to Inspector RPC
    try:
        controller = InspectorRpcController(rpc_host, rpc_port)
        return InspectorData(controller.inspector_rpc, controller.runtime_rpc, controller.ttnn_rpc)
    except:
        pass

    # Check for Inspector log directory
    log_directory = get_log_directory(log_directory)
    if not os.path.exists(log_directory):
        raise ValueError(f"Log directory {log_directory} does not exist. Please provide a valid path.")

    # Try to load serialized RPC data
    try:
        runtime_rpc_serialized: inspector_capnp.RuntimeInspector = InspectorRpcSerialized(
            os.path.join(log_directory, "RuntimeInspector"), inspector_capnp.RuntimeInspector
        )  # type: ignore
        ttnn_rpc_serialized: inspector_capnp.TtnnInspector = InspectorRpcSerialized(
            os.path.join(log_directory, "TtnnInspector"), inspector_capnp.TtnnInspector
        )  # type: ignore
        return InspectorData(None, runtime_rpc_serialized, ttnn_rpc_serialized)
    except:
        raise InspectorException(
            "There is no Inspector RPC data, cannot continue. "
            "Use --inspector-log-path to load saved Inspector data, or --inspector-rpc-host/--inspector-rpc-port "
            "to connect to a live Inspector. Ensure Inspector was enabled in Metal with TT_METAL_INSPECTOR=1 and "
            "TT_METAL_INSPECTOR_RPC=1."
        )


if __name__ == "__main__":
    run_script()
