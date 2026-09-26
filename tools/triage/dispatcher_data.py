#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dispatcher_data

Description:
    Provides dispatcher data noc locations on devices.
    Data include firmware path, kernel path, kernel offset, etc.

Owner:
    jbaumanTT
"""

from dataclasses import dataclass
from functools import cached_property
import os
import threading
from typing import Callable

from ttexalens.umd_device import TimeoutDeviceRegisterError

from inspector_capnp import BuildEnvData
from inspector_data import run as get_inspector_data, InspectorData
from metal_device_id_mapping import run as get_metal_device_id_mapping, MetalDeviceIdMapping
from elfs_cache import run as get_elfs_cache, ElfsCache
from triage import triage_singleton, ScriptConfig, run_script, log_check_location
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.elf import ElfFile, ElfVariable
from ttexalens.hardware.risc_debug import RiscLocation
from ttexalens.memory_access import create_l1_memory_access
from ttexalens._native_ttexalens import MemoryAccess
from ttexalens.context import Context
from triage import TTTriageError, triage_field, hex_serializer
from run_checks import run as get_run_checks
from run_checks import RunChecks, BlockType


script_config = ScriptConfig(
    data_provider=True,
    depends=["inspector_data", "elfs_cache", "run_checks", "metal_device_id_mapping"],
)

MAILBOX_CORRUPTED_MESSAGE = "Mailbox is likely corrupted, potentially due to NoC writes to an invalid location."


@dataclass
class DispatcherCoreData:
    # Level 0: Essential fields (always shown)
    watcher_kernel_id: int = triage_field("Kernel ID")
    kernel_name: str | None = triage_field("Kernel Name")
    subdevice: int = triage_field("Subdevice")
    go_message: str = triage_field("Go Message")
    preload: bool = triage_field("Preload")
    waypoint: str = triage_field("Waypoint")

    # Level 1: Detailed fields
    host_assigned_id: int | None = triage_field("Host Assigned ID", hex_serializer, verbose=1)
    watcher_previous_kernel_id: int = triage_field("Previous Kernel ID", verbose=1)
    previous_kernel_name: str | None = triage_field("Previous Kernel Name", verbose=1)
    kernel_offset: int | None = triage_field("Kernel Offset", hex_serializer, verbose=1)
    kernel_path: str | None = triage_field("Kernel Path", verbose=1)
    firmware_path: str = triage_field("Firmware Path", verbose=1)
    # New watcher/mailbox fields (verbose=1)
    dispatch_mode: str | None = triage_field("Dispatch Mode", verbose=1)
    brisc_noc_id: int | None = triage_field("BRISC NOC", verbose=1)
    enables: str | None = triage_field("Enables", verbose=1)
    subordinate_sync: str | None = triage_field("Subordinate Sync", verbose=1)
    watcher_enabled: bool | None = triage_field("Watcher Enabled", verbose=1)

    # Level 2: Internal debug fields
    launch_msg_rd_ptr: int = triage_field("RD PTR", verbose=2)
    kernel_config_base: int = triage_field("Base", hex_serializer, verbose=2)
    kernel_text_offset: int = triage_field("Offset", hex_serializer, verbose=2)
    kernel_xip_path: str | None = triage_field("Kernel XIP Path", verbose=2)

    # Non-triage fields
    mailboxes: ElfVariable | None = None
    # Host-assigned id from the previous launch message entry (best-effort).
    # Not serialized by default; used by scripts that need accurate previous-op tracking.
    previous_host_assigned_id: int | None = None
    # Inspector/control-plane-sourced block type for this core. Used by callers to reason about
    # active-vs-idle ETH without re-consulting the cluster descriptor.
    block_type: BlockType | None = None
    # Whether kernel_config.enables turned this specific risc on. False => idle by design (no kernel
    # launched on it). None => unknown (read failed / corrupt), so callers must not hide the core.
    risc_enabled_by_kernel: bool | None = None
    # Hint surfaced when find_kernel fails - explains the most likely cause (program cache off,
    # or workload destroyed despite cache being on) so callers can append it to "PC not in range" style errors.
    kernel_lookup_warning: str | None = None


ProcessorEnums = dict[str, dict[str, int | None]]


class DispatcherData:
    _dm0_elf: ElfFile
    _idle_erisc_elf: ElfFile | None
    _active_erisc_elf: ElfFile | None
    _drisc_elf: ElfFile | None

    def __init__(
        self,
        inspector_data: InspectorData,
        elfs_cache: ElfsCache,
        run_checks: RunChecks,
        metal_device_id_mapping: MetalDeviceIdMapping,
    ):
        self.inspector_data = inspector_data
        self.metal_device_id_mapping = metal_device_id_mapping
        self.programs = inspector_data.getPrograms().programs
        self.kernels = {kernel.watcherKernelId: kernel for program in self.programs for kernel in program.kernels}
        self.use_rpc_kernel_find = True

        # Caches that are populated on demand
        self.lock = threading.Lock()
        self._mailboxes_cache: dict[OnChipCoordinate, ElfVariable] = {}
        self._core_data_cache: dict[RiscLocation, DispatcherCoreData] = {}
        self._get_block_type: Callable[[OnChipCoordinate], BlockType | None] = run_checks.get_block_type

        # Cache build_env per device to avoid multiple RPC calls
        # Each device needs to have its own build_env to get the correct firmware path
        # Cache is keyed by unique_id for consistency
        self._build_env_cache: dict[int, BuildEnvData] = {}

        # Get the firmware paths from Inspector RPC build environment instead of relative paths
        # This ensures correct firmware paths for all devices and build configs
        # Prefill cache from no-arg RPC (ok if this fails - we'll fall back)
        try:
            all_build_envs = inspector_data.getAllBuildEnvs().buildEnvs
            for build_env_per_device in all_build_envs:
                # build_env_per_device.metalDeviceId is logical - remap to unique_id for cache key
                unique_id = metal_device_id_mapping.get_unique_id(build_env_per_device.metalDeviceId)
                self._build_env_cache[unique_id] = build_env_per_device.buildInfo
        except Exception:
            pass

        # Get the device ID from run_checks or inspector_data
        try:
            if not (run_checks and getattr(run_checks, "devices", None)):
                raise TTTriageError("RunChecks.devices not available. Ensure run_checks is a dependency or pass --dev.")
            # Use unique_id for device lookup
            device_unique_id = run_checks.devices[0].unique_id

            build_env = self._build_env_cache[device_unique_id]
            self._drisc_enabled_flag: bool | None = bool(build_env.dramProgrammableCoresEnabled)
            self._build_env = build_env

        except Exception as e:
            raise TTTriageError(
                f"Failed to get firmware path from Inspector RPC: {e}\n"
                "Make sure Inspector RPC is available or serialized RPC data exists.\n"
                "Set TT_METAL_INSPECTOR_RPC=1 when running your Metal application."
            )

        self._load_firmware_elfs(build_env, elfs_cache, run_checks.devices[0])

        # Access the value of enumerator for supported blocks
        self._ProgrammableCoreTypes_TENSIX = self._dm0_elf.get_enum_value("ProgrammableCoreType::TENSIX")
        self._ProgrammableCoreTypes_IDLE_ETH = self._dm0_elf.get_enum_value("ProgrammableCoreType::IDLE_ETH")
        self._ProgrammableCoreTypes_ACTIVE_ETH = self._dm0_elf.get_enum_value("ProgrammableCoreType::ACTIVE_ETH")
        self._ProgrammableCoreTypes_DRAM = self._dm0_elf.get_enum_value("ProgrammableCoreType::DRAM")

        # Go message states are constant values in the firmware elf, so we cache them
        def get_const_value(name) -> int:
            value = self._dm0_elf.get_constant(name)
            assert isinstance(value, int)
            return value

        self._go_message_states = {
            get_const_value("RUN_MSG_INIT"): "INIT",
            get_const_value("RUN_MSG_GO"): "GO",
            get_const_value("RUN_MSG_DONE"): "DONE",
            get_const_value("RUN_MSG_RESET_READ_PTR"): "RESET_READ_PTR",
            get_const_value("RUN_MSG_RESET_READ_PTR_FROM_HOST"): "RESET_READ_PTR_FROM_HOST",
        }
        self._launch_msg_buffer_num_entries = get_const_value("launch_msg_buffer_num_entries")

        # Subordinate sync states (used by NCRISC, TRISC0-2, ERISC1)
        self._sync_states = {
            get_const_value("RUN_SYNC_MSG_INIT"): "INIT",
            get_const_value("RUN_SYNC_MSG_GO"): "GO",
            get_const_value("RUN_SYNC_MSG_DONE"): "DONE",
            get_const_value("RUN_SYNC_MSG_LOAD"): "LOAD",
            get_const_value("RUN_SYNC_MSG_WAITING_FOR_RESET"): "WAITING_FOR_RESET",
            get_const_value("RUN_SYNC_MSG_INIT_SYNC_REGISTERS"): "INIT_SYNC_REGISTERS",
        }

        # Dispatch mode constants
        self._DISPATCH_MODE_DEV = self._dm0_elf.get_enum_value("dispatch_mode::DISPATCH_MODE_DEV")
        self._DISPATCH_MODE_HOST = self._dm0_elf.get_enum_value("dispatch_mode::DISPATCH_MODE_HOST")

        # Watcher enable constants (not used in firmware elf, so can't be retrieved from elf)
        self._WATCHER_ENABLED = 3
        self._WATCHER_DISABLED = 2

    def _load_firmware_elfs(self, build_env: BuildEnvData, elfs_cache: ElfsCache, device) -> None:
        raise NotImplementedError

    @cached_property
    def _enum_values_tenisx(self) -> ProcessorEnums:
        raise NotImplementedError

    @cached_property
    def _enum_values_eth(self) -> ProcessorEnums:
        raise NotImplementedError

    @cached_property
    def _enum_values_dram(self) -> ProcessorEnums:
        raise NotImplementedError

    @cached_property
    def _subordinate_sync_index(self) -> dict[str, int]:
        raise NotImplementedError

    def _firmware_elf_path(self, build_env: BuildEnvData, proc_name: str, block_type: BlockType | None) -> str:
        raise NotImplementedError

    def processor_index(self, risc_name: str, neo_id: int | None, enum_values: ProcessorEnums) -> int | None:
        raise NotImplementedError

    def _enables_bit_letters(self, block_type: BlockType | None) -> str:
        raise NotImplementedError

    def kernel_load_offset(
        self,
        proc_name: str,
        block_type: BlockType | None,
        kernel_config_base: int,
        kernel_text_offset: int,
    ) -> int:
        # For most blocks, the kernel is loaded at an offset from the config base, so we add them together to get the actual load address.
        # The & 0xFFFFFFFF is needed to wrap around to 32 bits, since the offset can be negative and Python ints are unbounded.
        return (kernel_config_base + kernel_text_offset) & 0xFFFFFFFF

    def has_xip_kernel_elf(self, proc_name: str) -> bool:
        return True

    def _fallback_kernel_elf_path(self, kernel, proc_name: str, block_type: BlockType | None) -> str | None:
        # When Inspector did not record the kernel ELF path, this method provides a fallback based on
        # the kernel's directory and processor name.
        return None

    def _get_build_env_for_device(self, device_unique_id: int) -> BuildEnvData:
        """Get build_env for a specific device, with caching"""
        if device_unique_id not in self._build_env_cache:
            raise TTTriageError(
                "Failed to get firmware path from Inspector RPC. "
                "Make sure Inspector RPC is available or serialized RPC data exists. "
                "Set TT_METAL_INSPECTOR_RPC=1 when running your Metal application."
            )
        return self._build_env_cache[device_unique_id]

    def _kernel_missing_hint_for_device(self, metal_device_id: int) -> str | None:
        mesh_devices = self.inspector_data.getMeshDevices().meshDevices
        containing = [md for md in mesh_devices if metal_device_id in md.devices]
        disabled = [md.meshId for md in containing if not md.programCacheEnabled]
        if disabled:
            return (
                f"Program cache is disabled on MeshDevice(s) {disabled} containing this device. "
                f"Enable program cache to see the callstack."
            )
        return (
            "No host-side live program owns the kernel on this device -"
            " the program should remain alive on host while its kernel is running."
        )

    def find_kernel(self, watcher_kernel_id):
        # Try to get kernel from RPC inspector data first, then fallback to cached kernels
        # RPC kernel find won't work if we are not connected to RPC, but are reading serialized data or logs
        if self.use_rpc_kernel_find:
            try:
                return self.inspector_data.getKernel(watcher_kernel_id).kernel
            except Exception:
                pass
        if watcher_kernel_id in self.kernels:
            self.use_rpc_kernel_find = False
            return self.kernels[watcher_kernel_id]
        raise TTTriageError(f"Kernel {watcher_kernel_id} not found in inspector data.")

    @staticmethod
    def _inspector_kernel_elf_path(kernel, proc_type: int) -> str | None:
        """Per-processor ELF path resolved by Inspector at compile time, indexed by processor index.

        Returns None when unavailable (older RPC/serialized data without the field, processor index out of
        range, or an empty entry for a processor this kernel doesn't use), so callers can fall back.
        """
        elf_paths = getattr(kernel, "processorElfPaths", None)
        if elf_paths is None or proc_type < 0 or proc_type >= len(elf_paths):
            return None
        return elf_paths[proc_type] or None

    def drisc_enabled(self) -> bool:
        # Casting to bool to avoid using ternary operator
        return bool(self._drisc_enabled_flag)

    def risc_enabled(self, risc_name: str) -> bool:
        if risc_name == "drisc":
            return self.drisc_enabled()
        return True

    def is_idle_in_default_view(self, risc_location: RiscLocation) -> bool:
        """Risc hidden unless --all-cores: finished (Go=DONE) or never enabled by the program."""
        d = self.get_cached_core_data(risc_location)
        return d.go_message == "DONE" or d.risc_enabled_by_kernel is False

    def get_cached_core_data(self, risc_location: RiscLocation) -> DispatcherCoreData:
        location = risc_location.location
        key = risc_location
        with self.lock:
            value = self._core_data_cache.get(key)
        if value is None:
            with self.lock:
                value = self._core_data_cache.get(key)
                if value is None:
                    mailboxes = self._mailboxes_cache.get(location)
                    if mailboxes is None:
                        mailboxes = self.read_mailboxes(location)
                        self._mailboxes_cache[location] = mailboxes
                    value = self.get_core_data(risc_location, mailboxes=mailboxes)
                    self._core_data_cache[key] = value
        return value

    def l1_memory_access(self, location: OnChipCoordinate) -> MemoryAccess:
        return create_l1_memory_access(location)

    def read_mailboxes(self, location: OnChipCoordinate) -> ElfVariable:
        block_type = self._get_block_type(location)
        l1_mem_access = self.l1_memory_access(location)
        fw_elf: ElfFile | None
        match block_type:
            case "tensix":
                fw_elf = self._dm0_elf
            case "idle_eth":
                fw_elf = self._idle_erisc_elf
            case "active_eth":
                fw_elf = self._active_erisc_elf
            case "dram":
                fw_elf = self._drisc_elf
            case _:
                raise TTTriageError(f"Unsupported block type: {block_type}")
        if fw_elf is None:
            raise TTTriageError(
                f"No firmware ELF for {block_type} blocks on this architecture, so their mailboxes cannot be read."
            )
        return fw_elf.read_global("mailboxes", l1_mem_access)

    def get_core_data(self, risc_location: RiscLocation, mailboxes: ElfVariable | None = None) -> DispatcherCoreData:
        location = risc_location.location
        risc_name = risc_location.risc_name
        neo_id = risc_location.neo_id
        # From inspector / the metal control plane, not location.device.active_eth_block_locations
        # (cluster descriptor / exalens). Everything below keys off this one source of truth.
        block_type = self._get_block_type(location)
        match block_type:
            case "tensix":
                programmable_core_type = self._ProgrammableCoreTypes_TENSIX
                enum_values = self._enum_values_tenisx
            case "idle_eth":
                programmable_core_type = self._ProgrammableCoreTypes_IDLE_ETH
                enum_values = self._enum_values_eth
            case "active_eth":
                programmable_core_type = self._ProgrammableCoreTypes_ACTIVE_ETH
                enum_values = self._enum_values_eth
            case "dram":
                if self._drisc_elf is None or not self._enum_values_dram or not self._drisc_enabled_flag:
                    raise TTTriageError("DRISC ELF not available for DRAM block type (Blackhole only)")
                programmable_core_type = self._ProgrammableCoreTypes_DRAM
                enum_values = self._enum_values_dram
            case _:
                raise TTTriageError(f"Unsupported block type: {block_type}")
        # Get the build_env for the device to get the correct firmware path
        # Each device may have different firmware paths based on its build configuration
        device_unique_id = location.device.unique_id
        build_env = self._get_build_env_for_device(device_unique_id)
        proc_name = risc_name.upper()
        proc_type = self.processor_index(risc_name, neo_id, enum_values)
        if proc_type is None:
            raise TTTriageError(f"Processor index for '{risc_name}' [neo: {neo_id}] not found in firmware ELF enums.")
        if mailboxes is None:
            mailboxes = self.read_mailboxes(location)

        # Refer to tt_metal/api/tt-metalium/dev_msgs.h for struct kernel_config_msg_t
        launch_msg_rd_ptr = int(mailboxes.launch_msg_rd_ptr)

        log_check_location(
            location,
            launch_msg_rd_ptr < self._launch_msg_buffer_num_entries,
            f"launch message read pointer {launch_msg_rd_ptr} >= {self._launch_msg_buffer_num_entries}. {MAILBOX_CORRUPTED_MESSAGE}",
        )

        previous_launch_msg_rd_ptr = (launch_msg_rd_ptr - 1) % self._launch_msg_buffer_num_entries

        kernel_config_base = -1
        kernel_text_offset = -1
        watcher_kernel_id = -1
        watcher_previous_kernel_id = -1
        kernel = None
        previous_kernel = None
        go_message_index = -1
        go_data = -1
        preload = False
        waypoint = ""
        host_assigned_id = None
        previous_host_assigned_id = None
        try:
            # Indexed with enum ProgrammableCoreType - tt_metal/hw/inc/*/core_config.h
            kernel_config_base = int(
                mailboxes.launch[launch_msg_rd_ptr].kernel_config.kernel_config_base[programmable_core_type]
            )
        except TimeoutDeviceRegisterError:
            raise
        except Exception:
            pass
        try:
            # Size 5 (NUM_PROCESSORS_PER_CORE_TYPE) - seems to be DM0,DM1,MATH0,MATH1,MATH2
            kernel_text_offset = int(mailboxes.launch[launch_msg_rd_ptr].kernel_config.kernel_text_offset[proc_type])
        except TimeoutDeviceRegisterError:
            raise
        except Exception:
            pass
        try:
            # enum dispatch_core_processor_classes
            watcher_kernel_id = int(mailboxes.launch[launch_msg_rd_ptr].kernel_config.watcher_kernel_ids[proc_type])
        except TimeoutDeviceRegisterError:
            raise
        except Exception:
            pass
        try:
            watcher_previous_kernel_id = int(
                mailboxes.launch[previous_launch_msg_rd_ptr].kernel_config.watcher_kernel_ids[proc_type]
            )
        except TimeoutDeviceRegisterError:
            raise
        except Exception:
            pass
        kernel_lookup_warning: str | None = None
        try:
            kernel = self.find_kernel(watcher_kernel_id)
        except Exception:
            if watcher_kernel_id != -1 and self.metal_device_id_mapping.has_unique_id(location.device.unique_id):
                metal_device_id = self.metal_device_id_mapping.get_metal_device_id(location.device.unique_id)
                kernel_lookup_warning = self._kernel_missing_hint_for_device(metal_device_id)
        try:
            previous_kernel = self.find_kernel(watcher_previous_kernel_id)
        except Exception:
            pass
        try:
            go_message_index = int(mailboxes.go_message_index)
            go_data = int(mailboxes.go_messages[go_message_index].signal)
        except TimeoutDeviceRegisterError:
            raise
        except Exception:
            pass
        try:
            preload = mailboxes.launch[launch_msg_rd_ptr].kernel_config.preload != 0
        except TimeoutDeviceRegisterError:
            raise
        except Exception:
            pass
        try:
            host_assigned_id = int(mailboxes.launch[launch_msg_rd_ptr].kernel_config.host_assigned_id)
        except TimeoutDeviceRegisterError:
            raise
        except Exception:
            pass
        try:
            previous_host_assigned_id = int(mailboxes.launch[previous_launch_msg_rd_ptr].kernel_config.host_assigned_id)
        except TimeoutDeviceRegisterError:
            raise
        except:
            pass
        try:
            waypoint_bytes = mailboxes.watcher.debug_waypoint[proc_type].waypoint.read_bytes()
            waypoint = waypoint_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")
        except TimeoutDeviceRegisterError:
            raise
        except Exception:
            pass

        # Read new watcher/mailbox fields
        dispatch_mode = None
        brisc_noc_id = None
        enables = None
        risc_enabled_by_kernel = None
        subordinate_sync = None
        watcher_enabled = None

        try:
            mode_val = mailboxes.launch[launch_msg_rd_ptr].kernel_config.mode
            if mode_val == self._DISPATCH_MODE_DEV:
                dispatch_mode = "DEV"
            elif mode_val == self._DISPATCH_MODE_HOST:
                dispatch_mode = "HOST"
            else:
                # Unexpected/unknown dispatch mode value; track for debugging but preserve behavior.
                log_check_location(
                    location,
                    False,
                    f"unexpected dispatch mode value '{mode_val}' in launch message",
                )
                dispatch_mode = str(mode_val)
        except Exception:
            log_check_location(
                location,
                False,
                f"failed to read dispatch mode from launch message. {MAILBOX_CORRUPTED_MESSAGE}",
            )

        try:
            brisc_noc_id = int(mailboxes.launch[launch_msg_rd_ptr].kernel_config.brisc_noc_id)
        except Exception:
            log_check_location(
                location,
                False,
                f"failed to read brisc noc id from launch message. {MAILBOX_CORRUPTED_MESSAGE}",
            )

        try:
            enables_val = int(mailboxes.launch[launch_msg_rd_ptr].kernel_config.enables)
            # Format enables like watcher: uppercase = enabled, lowercase = disabled.
            letters = self._enables_bit_letters(block_type)
            enables = ""
            for i, letter in enumerate(letters):
                enables += letter if (enables_val & (1 << i)) else letter.lower()
            # bit i set => processor i enabled; proc_type is this risc's processor index.
            risc_enabled_by_kernel = bool(enables_val & (1 << proc_type))
        except Exception:
            log_check_location(
                location,
                False,
                f"failed to read enables from launch message. {MAILBOX_CORRUPTED_MESSAGE}",
            )

        try:
            watcher_enable_val = mailboxes.watcher.enable
            if watcher_enable_val == self._WATCHER_ENABLED:
                watcher_enabled = True
            elif watcher_enable_val == self._WATCHER_DISABLED:
                watcher_enabled = False
            else:
                watcher_enabled = None  # Unknown state
                log_check_location(
                    location,
                    False,
                    f"unexpected watcher enable value: {watcher_enable_val}",
                )
        except Exception:
            watcher_enabled = None
            log_check_location(
                location,
                False,
                f"failed to read watcher enable from mailboxes. {MAILBOX_CORRUPTED_MESSAGE}",
            )

        # Subordinate sync is per-RISC (BRISC is the master, so no sync entry for it)
        try:
            if proc_name in self._subordinate_sync_index:
                sync_idx = self._subordinate_sync_index[proc_name]
                sync_val = int(mailboxes.subordinate_sync.map[sync_idx])
                subordinate_sync = self._sync_states.get(sync_val, str(sync_val))
        except Exception:
            log_check_location(
                location,
                False,
                f"failed to read subordinate sync from mailboxes. {MAILBOX_CORRUPTED_MESSAGE}",
            )

        # Construct the firmware path from the build_env instead of relative paths.
        firmware_path = os.path.realpath(self._firmware_elf_path(build_env, proc_name, block_type))

        kernel_path: str | None
        if kernel:
            # Prefer the per-processor ELF path resolved by Inspector at compile time. It is indexed by
            # processor index.
            kernel_path = self._inspector_kernel_elf_path(kernel, proc_type)
            if not kernel_path:
                kernel_path = self._fallback_kernel_elf_path(kernel, proc_name, block_type)
            kernel_path = os.path.realpath(kernel_path) if kernel_path else None
            kernel_xip_path = kernel_path + ".xip.elf" if kernel_path and self.has_xip_kernel_elf(proc_name) else None
            kernel_offset = self.kernel_load_offset(
                proc_name=proc_name,
                block_type=block_type,
                kernel_config_base=kernel_config_base,
                kernel_text_offset=kernel_text_offset,
            )
        else:
            kernel_path = None
            kernel_xip_path = None
            kernel_offset = None
        go_state = go_data
        go_data_state = self._go_message_states.get(go_state, str(go_state))

        return DispatcherCoreData(
            firmware_path=firmware_path,
            kernel_path=kernel_path,
            kernel_xip_path=kernel_xip_path,
            host_assigned_id=host_assigned_id,
            previous_kernel_name=previous_kernel.name if previous_kernel else None,
            kernel_offset=kernel_offset,
            kernel_name=kernel.name if kernel else None,
            launch_msg_rd_ptr=launch_msg_rd_ptr,
            kernel_config_base=kernel_config_base,
            kernel_text_offset=kernel_text_offset,
            watcher_kernel_id=watcher_kernel_id,
            watcher_previous_kernel_id=watcher_previous_kernel_id,
            subdevice=go_message_index,
            go_message=go_data_state,
            preload=preload,
            waypoint=waypoint,
            mailboxes=mailboxes,
            previous_host_assigned_id=previous_host_assigned_id,
            dispatch_mode=dispatch_mode,
            brisc_noc_id=brisc_noc_id,
            enables=enables,
            subordinate_sync=subordinate_sync,
            watcher_enabled=watcher_enabled,
            block_type=block_type,
            risc_enabled_by_kernel=risc_enabled_by_kernel,
            kernel_lookup_warning=kernel_lookup_warning,
        )


class DispatcherData1xx(DispatcherData):
    # One letter per ERISC an eth core has, for the Enables field. Architecture specific.
    _eth_enables_bit_letters: str
    # Whether this build runs a second ERISC on each active eth core.
    _is_2_erisc_mode: bool

    def _load_firmware_elfs(self, build_env: BuildEnvData, elfs_cache: ElfsCache, device) -> None:
        # The ELFs both Gen1 architectures have. Each one loads the rest on top of these.
        self._dm0_elf = elfs_cache[os.path.join(build_env.firmwarePath, "brisc", "brisc.elf")]
        self._idle_erisc_elf = elfs_cache[os.path.join(build_env.firmwarePath, "idle_erisc", "idle_erisc.elf")]

    def _enables_bit_letters(self, block_type: BlockType | None) -> str:
        # B=BRISC, N=NCRISC, T=TRISC on tensix; D=DRISC on dram; one E per ERISC on eth.
        if block_type == "tensix":
            return "BNT"
        if block_type == "dram":
            return "D"
        return self._eth_enables_bit_letters

    @cached_property
    def _enum_values_tenisx(self) -> ProcessorEnums:
        return {
            "ProcessorTypes": {
                "BRISC": self._dm0_elf.get_enum_value("TensixProcessorTypes::DM0"),
                "NCRISC": self._dm0_elf.get_enum_value("TensixProcessorTypes::DM1"),
                "TRISC0": self._dm0_elf.get_enum_value("TensixProcessorTypes::MATH0"),
                "TRISC1": self._dm0_elf.get_enum_value("TensixProcessorTypes::MATH1"),
                "TRISC2": self._dm0_elf.get_enum_value("TensixProcessorTypes::MATH2"),
            },
        }

    @cached_property
    def _subordinate_sync_index(self) -> dict[str, int]:
        # BRISC is the master, so it doesn't have a subordinate sync entry
        # For Tensix: NCRISC=0, TRISC0=1, TRISC1=2, TRISC2=3
        # For ETH (2-ERISC mode): ERISC1=0
        return {"NCRISC": 0, "TRISC0": 1, "TRISC1": 2, "TRISC2": 3, "ERISC1": 0}

    def _firmware_elf_path(self, build_env: BuildEnvData, proc_name: str, block_type: BlockType | None) -> str:
        name = proc_name.lower()
        if block_type == "dram":
            return os.path.join(build_env.firmwarePath, "drisc", "drisc.elf")
        if block_type == "active_eth":
            if name == "erisc":
                return os.path.join(build_env.firmwarePath, "erisc", "erisc.elf")
            if name == "erisc0":
                return os.path.join(build_env.firmwarePath, "active_erisc", "active_erisc.elf")
            if name == "erisc1":
                return (
                    os.path.join(build_env.firmwarePath, "subordinate_active_erisc", "subordinate_active_erisc.elf")
                    if self._is_2_erisc_mode
                    else os.path.join(build_env.firmwarePath, "active_erisc", "active_erisc.elf")
                )
            raise TTTriageError(f"Unsupported active ETH processor '{proc_name}' for firmware path.")
        if name in ("erisc", "erisc0"):
            return os.path.join(build_env.firmwarePath, "idle_erisc", "idle_erisc.elf")
        if name == "erisc1":
            return os.path.join(build_env.firmwarePath, "subordinate_idle_erisc", "subordinate_idle_erisc.elf")
        return os.path.join(build_env.firmwarePath, name, f"{name}.elf")

    def processor_index(self, risc_name: str, neo_id: int | None, enum_values: ProcessorEnums) -> int | None:
        return enum_values["ProcessorTypes"].get(risc_name.upper())

    def _fallback_kernel_elf_path(self, kernel, proc_name: str, block_type: BlockType | None) -> str | None:
        kernel_dir: str = kernel.path
        name = proc_name.lower()
        if block_type == "active_eth":
            if name == "erisc":
                return kernel_dir + "/erisc/erisc.elf"
            if name == "erisc0":
                return kernel_dir + "/active_erisc/active_erisc.elf" if self._is_2_erisc_mode else None
            if name == "erisc1":
                return (
                    kernel_dir + "/subordinate_active_erisc/subordinate_active_erisc.elf"
                    if self._is_2_erisc_mode
                    else kernel_dir + "/active_erisc/active_erisc.elf"
                )
            raise TTTriageError(f"Unsupported active ETH processor '{proc_name}' for kernel path.")
        if name in ("erisc", "erisc0"):
            return kernel_dir + "/idle_erisc/idle_erisc.elf"
        if name == "erisc1":
            return kernel_dir + "/subordinate_idle_erisc/subordinate_idle_erisc.elf"
        return kernel_dir + f"/{name}/{name}.elf"


class DispatcherDataWormhole(DispatcherData1xx):
    _eth_enables_bit_letters = "E"
    _is_2_erisc_mode = False
    _drisc_elf = None

    def _load_firmware_elfs(self, build_env: BuildEnvData, elfs_cache: ElfsCache, device) -> None:
        super()._load_firmware_elfs(build_env, elfs_cache, device)
        self._active_erisc_elf = elfs_cache[os.path.join(build_env.firmwarePath, "erisc", "erisc.elf")]

    @cached_property
    def _enum_values_eth(self) -> ProcessorEnums:
        # One ERISC per eth core, so both names triage knows it by are DM0. There is no
        # EthProcessorTypes::DM1 on this architecture, hence no ERISC1 entry: asking for one raises
        # rather than quietly resolving to the wrong processor.
        assert self._idle_erisc_elf is not None, "Wormhole always loads the idle ERISC firmware"
        dm0 = self._idle_erisc_elf.get_enum_value("EthProcessorTypes::DM0")
        return {"ProcessorTypes": {"ERISC": dm0, "ERISC0": dm0}}

    @cached_property
    def _enum_values_dram(self) -> ProcessorEnums:
        # Wormhole DRAM cores have no programmable RISC.
        return {}

    def has_xip_kernel_elf(self, proc_name: str) -> bool:
        # NCRISC is the one processor with no XIP ELF.
        return proc_name != "NCRISC"

    def kernel_load_offset(
        self,
        proc_name: str,
        block_type: BlockType | None,
        kernel_config_base: int,
        kernel_text_offset: int,
    ) -> int:
        if proc_name == "NCRISC":
            return 0xFFC00000
        if block_type == "active_eth":
            # Active ETH kernels are reached by the text offset alone.
            return kernel_text_offset
        return super().kernel_load_offset(proc_name, block_type, kernel_config_base, kernel_text_offset)


class DispatcherDataBlackhole(DispatcherData1xx):
    """Blackhole: one or two ERISCs per eth core depending on the build, and a DRISC on DRAM cores
    when the build enables programmable DRAM."""

    _eth_enables_bit_letters = "EE"

    def _load_firmware_elfs(self, build_env: BuildEnvData, elfs_cache: ElfsCache, device) -> None:
        super()._load_firmware_elfs(build_env, elfs_cache, device)
        self._active_erisc_elf = elfs_cache[os.path.join(build_env.firmwarePath, "active_erisc", "active_erisc.elf")]

        # There are 2 modes (1-ERISC and 2-ERISC); the subordinate ELF existing means 2-ERISC.
        self._is_2_erisc_mode = os.path.exists(
            os.path.join(build_env.firmwarePath, "subordinate_active_erisc", "subordinate_active_erisc.elf")
        )

        self._drisc_elf = None
        if self._drisc_enabled_flag:
            try:
                self._drisc_elf = elfs_cache[os.path.join(build_env.firmwarePath, "drisc", "drisc.elf")]
            except Exception:
                # DRISC firmware is optional; if it cannot be loaded, leave self._drisc_elf as None
                pass

    @cached_property
    def _enum_values_eth(self) -> ProcessorEnums:
        assert self._idle_erisc_elf is not None, "Blackhole always loads the idle ERISC firmware"
        dm0 = self._idle_erisc_elf.get_enum_value("EthProcessorTypes::DM0")
        dm1 = self._idle_erisc_elf.get_enum_value("EthProcessorTypes::DM1")
        processor_types: dict[str, int | None] = {"ERISC": dm0, "ERISC0": dm0}

        # ERISC1 is the second ERISC only in 2-ERISC mode; in 1-ERISC mode it behaves like DM0.
        # A firmware build with no DM1 enumerator at all gets no ERISC1 entry.
        if dm1 is not None:
            processor_types["ERISC1"] = dm1 if self._is_2_erisc_mode else dm0
        return {"ProcessorTypes": processor_types}

    @cached_property
    def _enum_values_dram(self) -> ProcessorEnums:
        # Only present when the build loaded DRISC firmware.
        if self._drisc_elf is None:
            return {}
        return {
            "ProcessorTypes": {
                "DRISC": self._drisc_elf.get_enum_value("DramProcessorTypes::DM0"),
            },
        }

    def kernel_load_offset(
        self,
        proc_name: str,
        block_type: BlockType | None,
        kernel_config_base: int,
        kernel_text_offset: int,
    ) -> int:
        if block_type == "dram":
            # DRAM kernel ELFs are linked at their actual load address (kernel_text_offset), not at
            # address 0 like Tensix kernels, so no base adjustment is needed.
            return kernel_text_offset
        return super().kernel_load_offset(proc_name, block_type, kernel_config_base, kernel_text_offset)


class DispatcherData2xx(DispatcherData):
    # TODO: For now this is the same as Quasar, but in the future, we should adapt it to any Gen2-specific differences.

    @staticmethod
    def _fold_uncached_l1_alias(address: int) -> int:
        # 4 MB-8 MB is the uncached alias of the same 4 MB of L1 for rocket cores, so for simplicity
        # we fold any address in that range back onto the 0-4 MB cached view.
        alias_base = 0x00400000
        alias_end = 0x00800000
        if alias_base <= address < alias_end:
            return address - alias_base
        return address

    class _UncachedL1AliasMemoryAccess(MemoryAccess):
        def __init__(self, inner: MemoryAccess):
            super().__init__()
            self._inner = inner

        def read(self, address: int, buffer) -> None:
            self._inner.read(DispatcherData2xx._fold_uncached_l1_alias(address), buffer)

        def write(self, address: int, data) -> None:
            self._inner.write(DispatcherData2xx._fold_uncached_l1_alias(address), data)

        def read_register(self, register_index: int) -> int:
            return self._inner.read_register(register_index)

        def write_register(self, register_index: int, value: int) -> None:
            self._inner.write_register(register_index, value)

    def _load_firmware_elfs(self, build_env: BuildEnvData, elfs_cache: ElfsCache, device) -> None:
        self._dm0_elf = elfs_cache[os.path.join(build_env.firmwarePath, "dm0", "dm0.elf")]
        # Quasar has no eth or dram firmware; the shared code guards on these being None.
        self._idle_erisc_elf = None
        self._active_erisc_elf = None
        self._drisc_elf = None

    @cached_property
    def _enum_values_tenisx(self) -> ProcessorEnums:
        # A tensix cluster is eight data movement cores plus four NEOs of four TRISCs each, and
        # TensixProcessorTypes numbers the compute processors after the eight DMs.
        num_dm_cores = 8
        num_neos = 4
        num_triscs_per_neo = 4
        processor_types: dict[str, int | None] = {
            f"ROCKET{i}": self._dm0_elf.get_enum_value(f"TensixProcessorTypes::DM{i}") for i in range(num_dm_cores)
        }
        for neo in range(num_neos):
            for trisc in range(num_triscs_per_neo):
                processor_types[f"E{neo}_TRISC{trisc}"] = self._dm0_elf.get_enum_value(
                    f"TensixProcessorTypes::E{neo}_MATH{trisc}"
                )
        return {"ProcessorTypes": processor_types}

    @cached_property
    def _enum_values_eth(self) -> ProcessorEnums:
        # TODO: Quasar triage does not cover eth blocks yet.
        return {"ProcessorTypes": {}}

    @cached_property
    def _enum_values_dram(self) -> ProcessorEnums:
        # TODO: Quasar triage does not cover dram blocks yet.
        return {}

    def _enables_bit_letters(self, block_type: BlockType | None) -> str:
        # TODO: Implement enables bit letters for Quasar.
        return ""

    @cached_property
    def _subordinate_sync_index(self) -> dict[str, int]:
        # TODO: Quasar's launch/sync layout is not the Gen1 master/subordinate map; until it is modelled,
        # report no subordinate slots rather than Gen1's, which would decode to nonsense.
        return {}

    def _firmware_elf_path(self, build_env: BuildEnvData, proc_name: str, block_type: BlockType | None) -> str:
        name = proc_name.lower()
        if name.startswith("rocket"):
            return os.path.join(build_env.firmwarePath, "dm0", "dm0.elf")
        return os.path.join(build_env.firmwarePath, name, f"{name}.elf")

    def processor_index(self, risc_name: str, neo_id: int | None, enum_values: ProcessorEnums) -> int | None:
        name = risc_name.lower()
        processor_types = enum_values["ProcessorTypes"]
        if name.startswith("rocket"):
            return processor_types.get(f"ROCKET{name[len('rocket'):]}")
        if name.startswith("trisc") and neo_id is not None:
            return processor_types.get(f"E{neo_id}_TRISC{name[len('trisc'):]}")
        return None

    def l1_memory_access(self, location: OnChipCoordinate) -> MemoryAccess:
        return self._UncachedL1AliasMemoryAccess(create_l1_memory_access(location))

    def kernel_load_offset(
        self,
        proc_name: str,
        block_type: BlockType | None,
        kernel_config_base: int,
        kernel_text_offset: int,
    ) -> int:
        return self._fold_uncached_l1_alias(
            super().kernel_load_offset(proc_name, block_type, kernel_config_base, kernel_text_offset)
        )


@triage_singleton
def run(args, context: Context) -> DispatcherData:
    inspector_data = get_inspector_data(args, context)
    elfs_cache = get_elfs_cache(args, context)
    run_checks = get_run_checks(args, context)
    metal_device_id_mapping = get_metal_device_id_mapping(args, context)

    if not run_checks.devices:
        raise TTTriageError("No devices to inspect, so there is no generation to pick dispatcher data for.")
    device = run_checks.devices[0]
    if device.is_wormhole():
        return DispatcherDataWormhole(inspector_data, elfs_cache, run_checks, metal_device_id_mapping)
    elif device.is_blackhole():
        return DispatcherDataBlackhole(inspector_data, elfs_cache, run_checks, metal_device_id_mapping)
    else:
        return DispatcherData2xx(inspector_data, elfs_cache, run_checks, metal_device_id_mapping)


if __name__ == "__main__":
    run_script()
