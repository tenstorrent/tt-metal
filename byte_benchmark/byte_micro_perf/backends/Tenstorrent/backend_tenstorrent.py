import os
import sys
import json
import pathlib
import random
import shutil
import subprocess
import signal
from datetime import timedelta

from pydash import lines
import torch
import torch.distributed as dist
from tracy import signpost


FILE_DIR = pathlib.Path(__file__).parent.absolute()
BACKEND_DIR = FILE_DIR.parent
MICRO_PERF_DIR = BACKEND_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.backend import Backend
from backends.Tenstorrent.provider_tenstorrent import TENSTORRENT_PROVIDER
from core.utils import suppress_stdout_stderr

import ttnn


class TimeoutError(Exception):
    """Custom timeout exception for CCL initialization"""

    pass


def get_avg_duration(perf_data: dict, iterations: int) -> str:
    """Pretty-print program perf data mapping: chip_id -> set[ProgramAnalysisData]."""

    def _program_sort_key(p):
        uid = p.program_execution_uid
        return (uid.runtime_id, uid.trace_id, uid.trace_id_counter)

    total_duration = 0
    core_count = 0
    for device_id in sorted(perf_data.keys()):
        programs = sorted(list(perf_data[device_id]), key=_program_sort_key)
        for program in programs:
            uid = program.program_execution_uid
            analyses_items = sorted(program.program_analyses_results.items())
            if core_count == 0:
                core_count = program.core_count
            for name, res in analyses_items:
                if "FW" in name and "ns" in name:
                    total_duration += res.duration
                    break

    return total_duration / iterations


class BackendTenstorrent(Backend):
    # Class-level cache for device count to avoid repeated queries that lock devices
    _device_count_cache = None
    _device_ids_cache = None

    def __init__(self):
        super().__init__()
        self.avail_providers = TENSTORRENT_PROVIDER

        # Check if we should skip ttnn initialization (for CCL subprocess mode)
        # This prevents device lock contention when multiple processes are spawned
        skip_ttnn = os.environ.get("TT_SKIP_TTNN_INIT", "0") == "1"

        # Try to import ttnn for Tenstorrent hardware support
        self.ttnn_available = True
        self.ttnn_device = None

        print("Skip")

        if skip_ttnn:
            print("Skipping ttnn initialization (TT_SKIP_TTNN_INIT=1)")
        else:
            try:
                # import ttnn  # noqa: F401
                # if "blackhole" in ttnn.get_arch_name():
                #     self.total_memory = 24 * (1024 ** 3)
                #     self.free_memory =  22 * (1024 ** 3)
                # else:
                # Wormhole
                self.total_memory = 12 * (1024**3)
                self.free_memory = 12 * (1024**3)
                print(
                    f"TTNN initialized: total_memory={self.total_memory/(1024**3)}GB, free_memory={self.free_memory/(1024**3)}GB"
                )
            except ImportError as e:
                print(f"Warning: ttnn not available, falling back to CPU: {e}")
                pass
        print("TTNN available:", self.ttnn_available)

    def __del__(self):
        if self.numa_rank == 0:
            PROFILER_DIR = pathlib.Path.cwd().joinpath("profiling")
            if PROFILER_DIR.exists():
                shutil.rmtree(PROFILER_DIR)

    """
    device management related
    """

    def get_torch_device_name(self):
        # Tenstorrent uses CPU for PyTorch tensors, but computation is offloaded to TT devices
        # For now, use "cpu" as PyTorch device name
        # TODO: Update if tt_lib provides custom torch device integration
        return "cpu"

    def get_device_name(self, index=0):
        """Get Tenstorrent device name"""
        import ttnn

        arch_string = "Blackhole" if "blackhole" in ttnn.get_arch_name() else "Wormhole"
        return f"Tenstorrent {arch_string}"

    def get_device_properties(self, index=0):
        """Get device properties similar to torch.cuda.get_device_properties"""

        # Create a named tuple-like object for compatibility
        class DeviceProperties:
            def __init__(self):
                self.name = "Tenstorrentttnn.g"
                self.total_memory = 12 * (1024**3)  # Assume 12GB for Wormhole, adjust based on actual HW
                self.multi_processor_count = 80  # Placeholder

        return DeviceProperties()

    def get_mem_info(self, index=0):
        """Return (free_memory, total_memory) in bytes"""
        return self.free_memory, self.total_memory

    def get_device_count(self):
        """Return (device_count, list_of_device_indices)"""
        # Always check environment variable first to support CCL subprocess mode
        # This allows consistent device count even when ttnn is not initialized
        device_ids = [1]
        return len(device_ids), device_ids

    def set_device(self, device_index: int):
        """Set current TT device"""
        if self.ttnn_available:
            try:
                # TODO: Set active TT device
                # self.tt_lib.device.SetDevice(device_index)
                os.environ["TT_METAL_DEVICE_ID"] = str(device_index)
            except:
                pass

    def get_device(self):
        """Get current TT device index"""
        try:
            return int(os.environ.get("TT_METAL_DEVICE_ID", 0))
        except:
            return 0

    def device_synchronize(self, device):
        """Synchronize TT device execution"""
        if self.ttnn_available:
            ttnn.synchronize_device(device)
        # For CPU fallback, no sync needed

    def empty_cache(self):
        """Clear TT device memory cache"""
        if self.ttnn_available:
            try:
                # TODO: Clear TT device cache if API exists
                pass
            except:
                pass

    def get_backend_env(self):
        """Return backend environment information (similar to GPU backend)"""
        __torch_version = torch.__version__
        __ttnn_version = ""
        __driver_version = ""
        __firmware_version = ""

        # Get ttnn version
        if self.ttnn_available:
            try:
                import ttnn
                import importlib.metadata

                __ttnn_version = importlib.metadata.version("ttnn")
            except:
                __ttnn_version = "unknown"

        # Get driver and firmware versions from tt-smi (similar to nvidia-smi for GPU)
        try:
            tt_smi_output = subprocess.run(
                ["tt-smi", "-s"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
            )
            for line in tt_smi_output.stdout.split("\n"):
                if "Driver" in line or "KMD" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        __driver_version = parts[1].strip()
                elif "Firmware" in line or "FW" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        __firmware_version = parts[1].strip()
        except:
            # Fallback: try to get from environment or device query
            __driver_version = os.environ.get("TT_DRIVER_VERSION", "unknown")
            __firmware_version = os.environ.get("TT_FW_VERSION", "unknown")

        env_info = {
            "torch": __torch_version,
            "ttnn": __ttnn_version,
            "ttnn_available": self.ttnn_available,
            "driver": __driver_version,
            "firmware": __firmware_version,
        }

        # Add TT Metal environment variables
        tt_env_vars = ["TT_METAL_HOME", "ARCH_NAME", "TT_METAL_DEVICE_ID"]
        for var in tt_env_vars:
            if var in os.environ:
                env_info[var] = os.environ[var]

        return env_info

    """
    ccl related - distributed communication
    """

    def get_dist_module(self):
        """Return distributed communication module"""
        return dist

    def get_dist_backend(self):
        """Return distributed backend name"""
        # For Tenstorrent, use gloo as default (CPU-based)
        # TODO: If TT has custom CCL, return that backend name
        return "gloo"

    def initialize_ccl(self, rank: int, world_size: int):
        """
        Initialize CCL (Collective Communication Library) for Tenstorrent.

        Override the base class method to add timeout protection and proper
        error handling for gloo backend initialization.

        The gloo backend is used for CPU-based distributed communication
        since Tenstorrent handles device computation separately via ttnn.
        """
        import time

        os.environ["MASTER_PORT"] = os.environ["BACKEND_PORT"]
        dist_module = self.get_dist_module()
        dist_backend_name = self.get_dist_backend()

        print(f"[Rank {rank}] Initializing CCL with backend={dist_backend_name}, world_size={world_size}")

        # Use longer timeout for large tensor operations (default 1800s = 30 minutes)
        # Large all_reduce operations (e.g., 2GB) on CPU can take several minutes
        init_timeout = int(os.environ.get("TT_CCL_INIT_TIMEOUT", "1800"))
        try:
            dist_module.init_process_group(
                backend=dist_backend_name, world_size=world_size, rank=rank, timeout=timedelta(seconds=init_timeout)
            )
        except Exception as e:
            print(f"[Rank {rank}] Failed to initialize process group: {e}")
            raise

        print(f"[Rank {rank}] Process group initialized, testing all_reduce...")

        # Test all_reduce with CPU tensor (gloo backend requires CPU tensors)
        assigned_value = 1 if rank < world_size // 2 else -1
        data = torch.ones([1], dtype=torch.float32, device="cpu") * assigned_value

        try:
            dist_module.all_reduce(data, op=dist_module.ReduceOp.SUM)
            print(f"[Rank {rank}] CCL test all_reduce result: {data.item()}")
        except Exception as e:
            print(f"[Rank {rank}] CCL test all_reduce failed: {e}")
            raise

        return True

    def op_group_barrier(self, op_group=None, group_size=1):
        """
        Override barrier to ensure CPU tensors are used with gloo backend.

        The gloo backend only supports CPU tensors, so we must explicitly
        create CPU tensors regardless of what get_torch_device_name() returns.
        """
        dist_module = self.get_dist_module()
        if dist_module.is_initialized() and group_size > 1:
            # Always use CPU tensor for gloo backend
            dist_module.all_reduce(
                torch.tensor([1], dtype=torch.int32, device="cpu"), op=dist_module.ReduceOp.SUM, group=op_group
            )

    def core_perf(self, op_instance, warmup_iterations, prefer_iterations, tensor_list, profiling=True):
        """
        Performance measurement for Tenstorrent with hardware profiling support
        """
        op_group = op_instance.op_group
        group_size = op_instance.group_size

        if self.ttnn_available:
            try:
                import ttnn

                # Warmup
                for i in range(warmup_iterations):
                    index = random.randint(0, len(tensor_list) - 1)
                    op_instance.core_run(tensor_list[index])

                self.device_synchronize(op_instance.device)
                self.op_group_barrier(op_group=op_group, group_size=group_size)

                # Use ttnn events for precise timing
                import time

                start_time = time.perf_counter_ns()

                if profiling:
                    self.device_synchronize(op_instance.device)
                    ttnn.ReadDeviceProfiler(op_instance.device)

                for i in range(prefer_iterations):
                    op_instance.core_run(tensor_list[i % len(tensor_list)])

                self.device_synchronize(op_instance.device)
                end_time = time.perf_counter_ns()

                if profiling:
                    ttnn.ReadDeviceProfiler(op_instance.device)
                    latest_data = ttnn.get_latest_programs_perf_data()
                    latency_us = get_avg_duration(latest_data, prefer_iterations) / 1000.0
                else:
                    latency_us = (end_time - start_time) / 1e3 / prefer_iterations
                return latency_us, []

            except Exception as e:
                print(f"Warning: ttnn timing failed: {e}")
                pass

        # CPU timer fallback
        for i in range(warmup_iterations):
            index = random.randint(0, len(tensor_list) - 1)
            op_instance.core_run(tensor_list[index])

        self.device_synchronize(op_instance.device)
        self.op_group_barrier(op_group=op_group, group_size=group_size)

        import time

        start_time = time.perf_counter_ns()
        for i in range(prefer_iterations):
            op_instance.core_run(tensor_list[i % len(tensor_list)])
        self.device_synchronize(op_instance.device)
        end_time = time.perf_counter_ns()

        latency_us = (end_time - start_time) / 1e3 / prefer_iterations
        return latency_us, []
