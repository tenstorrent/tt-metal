#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
import os
import threading

from inspector_data import run as get_inspector_data, InspectorData
from metal_device_id_mapping import run as get_metal_device_id_mapping, MetalDeviceIdMapping
from elfs_cache import run as get_elfs_cache, ElfsCache
from triage import triage_singleton, ScriptConfig, run_script, log_check_location
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.elf import ElfVariable
from ttexalens.memory_access import MemoryAccess
from ttexalens.context import Context
from triage import TTTriageError, triage_field, hex_serializer
from run_checks import run as get_run_checks
from run_checks import RunChecks

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
    kernel_path: str = triage_field("Kernel Path", verbose=1)
    firmware_path: str = triage_field("Firmware Path", verbose=1)

    # Level 2: Internal debug fields
    launch_msg_rd_ptr: int = triage_field("RD PTR", verbose=2)
    kernel_config_base: int = triage_field("Base", hex_serializer, verbose=2)
    kernel_text_offset: int = triage_field("Offset", hex_serializer, verbose=2)
    kernel_xip_path: str | None = triage_field("Kernel XIP Path", verbose=2)

    # Non-triage fields
    mailboxes: ElfVariable | None = None


class DispatcherData:
    def __init__(
        self,
        inspector_data: InspectorData,
        elfs_cache: ElfsCache,
        run_checks: RunChecks,
        metal_device_id_mapping: MetalDeviceIdMapping,
    ):
        self.inspector_data = inspector_data
        self.programs = inspector_data.getPrograms().programs
        self.kernels = {kernel.watcherKernelId: kernel for program in self.programs for kernel in program.kernels}
        self.use_rpc_kernel_find = True

        # Caches that are populated on demand
        self.lock = threading.Lock()
        self._mailboxes_cache: dict[OnChipCoordinate, ElfVariable] = {}
        self._core_data_cache: dict[tuple[OnChipCoordinate, str], DispatcherCoreData] = {}

        # Cache build_env per device to avoid multiple RPC calls
        # Each device needs to have its own build_env to get the correct firmware path
        # Cache is keyed by unique_id for consistency
        self._build_env_cache = {}

        # Get the firmware paths from Inspector RPC build environment instead of relative paths
        # This ensures correct firmware paths for all devices and build configs
        # Prefill cache from no-arg RPC (ok if this fails - we'll fall back)
        try:
            all_build_envs = inspector_data.getAllBuildEnvs().buildEnvs
            for build_env in all_build_envs:
                # build_env.metalDeviceId is logical - remap to unique_id for cache key
                unique_id = metal_device_id_mapping.get_unique_id(build_env.metalDeviceId)
                self._build_env_cache[unique_id] = build_env.buildInfo
        except Exception:
            pass

        # Get the device ID from run_checks or inspector_data
        try:
            if not (run_checks and getattr(run_checks, "devices", None)):
                raise TTTriageError("RunChecks.devices not available. Ensure run_checks is a dependency or pass --dev.")
            # Use unique_id for device lookup
            device_unique_id = run_checks.devices[0].unique_id

            build_env = self._build_env_cache[device_unique_id]
            # Use build_env for initial firmware paths
            brisc_elf_path = os.path.join(build_env.firmwarePath, "brisc", "brisc.elf")
            idle_erisc_elf_path = os.path.join(build_env.firmwarePath, "idle_erisc", "idle_erisc.elf")
            active_erisc_elf_name = "erisc" if run_checks.devices[0].is_wormhole() else "active_erisc"
            active_erisc_elf_path = os.path.join(
                build_env.firmwarePath, active_erisc_elf_name, active_erisc_elf_name + ".elf"
            )

            # On blackhole we have 2 modes (1-ERISC and 2-ERISC)
            # By checking if the subordinate active erisc elf exists, we can determine in which mode we are
            if run_checks.devices[0].is_blackhole():
                self._is_2_erisc_mode = os.path.exists(
                    os.path.join(build_env.firmwarePath, "subordinate_active_erisc", "subordinate_active_erisc.elf")
                )

        except Exception as e:
            raise TTTriageError(
                f"Failed to get firmware path from Inspector RPC: {e}\n"
                "Make sure Inspector RPC is available or serialized RPC data exists.\n"
                "Set TT_METAL_INSPECTOR_RPC=1 when running your Metal application."
            )

        self._brisc_elf = elfs_cache[brisc_elf_path]
        self._idle_erisc_elf = elfs_cache[idle_erisc_elf_path]
        self._active_erisc_elf = elfs_cache[active_erisc_elf_path]

        # Access the value of enumerator for supported blocks
        self._ProgrammableCoreTypes_TENSIX = self._brisc_elf.get_enum_value("ProgrammableCoreType::TENSIX")
        self._ProgrammableCoreTypes_IDLE_ETH = self._brisc_elf.get_enum_value("ProgrammableCoreType::IDLE_ETH")
        self._ProgrammableCoreTypes_ACTIVE_ETH = self._brisc_elf.get_enum_value("ProgrammableCoreType::ACTIVE_ETH")

        # Enumerators for tensix block
        self._enum_values_tenisx = {
            "ProcessorTypes": {
                "BRISC": self._brisc_elf.get_enum_value("TensixProcessorTypes::DM0"),
                "NCRISC": self._brisc_elf.get_enum_value("TensixProcessorTypes::DM1"),
                "TRISC0": self._brisc_elf.get_enum_value("TensixProcessorTypes::MATH0"),
                "TRISC1": self._brisc_elf.get_enum_value("TensixProcessorTypes::MATH1"),
                "TRISC2": self._brisc_elf.get_enum_value("TensixProcessorTypes::MATH2"),
            },
        }

        # Enumerators for eth block
        self._enum_values_eth = {
            "ProcessorTypes": {
                "ERISC": self._idle_erisc_elf.get_enum_value("EthProcessorTypes::DM0"),
                "ERISC0": self._idle_erisc_elf.get_enum_value("EthProcessorTypes::DM0"),
            },
        }

        # EthProcessorTypes::DM1 is only available on blackhole
        # ERISC1 behaves like DM0 if 1 ERISC mode is used
        if self._idle_erisc_elf.get_enum_value("EthProcessorTypes::DM1") is not None:
            self._enum_values_eth["ProcessorTypes"]["ERISC1"] = (
                self._idle_erisc_elf.get_enum_value("EthProcessorTypes::DM1")
                if self._is_2_erisc_mode
                else self._idle_erisc_elf.get_enum_value("EthProcessorTypes::DM0")
            )

        # Go message states are constant values in the firmware elf, so we cache them
        def get_const_value(name) -> int:
            value = self._brisc_elf.get_constant(name)
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

    def _get_build_env_for_device(self, device_unique_id: int):
        """Get build_env for a specific device, with caching"""
        if device_unique_id not in self._build_env_cache:
            raise TTTriageError(
                "Failed to get firmware path from Inspector RPC. "
                "Make sure Inspector RPC is available or serialized RPC data exists. "
                "Set TT_METAL_INSPECTOR_RPC=1 when running your Metal application."
            )
        return self._build_env_cache[device_unique_id]

    def find_kernel(self, watcher_kernel_id):
        # Try to get kernel from RPC inspector data first, then fallback to cached kernels
        # RPC kernel find won't work if we are not connected to RPC, but are reading serialized data or logs
        if self.use_rpc_kernel_find:
            try:
                return self.inspector_data.getKernel(watcher_kernel_id).kernel
            except:
                pass
        if watcher_kernel_id in self.kernels:
            self.use_rpc_kernel_find = False
            return self.kernels[watcher_kernel_id]
        raise TTTriageError(f"Kernel {watcher_kernel_id} not found in inspector data.")

    def get_cached_core_data(self, location: OnChipCoordinate, risc_name: str) -> DispatcherCoreData:
        key = (location, risc_name)
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
                    value = self.get_core_data(location, risc_name, mailboxes=mailboxes)
                    self._core_data_cache[key] = value
        return value

    def read_mailboxes(self, location: OnChipCoordinate) -> ElfVariable:
        l1_mem_access = MemoryAccess.create_l1(location)
        if location.device.get_block_type(location) == "functional_workers":
            # For tensix, use the brisc elf
            fw_elf = self._brisc_elf
        elif location in location.device.idle_eth_block_locations:
            # For idle eth, use the idle erisc elf
            fw_elf = self._idle_erisc_elf
        elif location in location.device.active_eth_block_locations:
            # For active eth, use the active erisc elf
            fw_elf = self._active_erisc_elf
        else:
            raise TTTriageError(f"Unsupported block type: {location.device.get_block_type(location)}")
        return fw_elf.read_global("mailboxes", l1_mem_access)

    def get_core_data(
        self, location: OnChipCoordinate, risc_name: str, mailboxes: ElfVariable | None = None
    ) -> DispatcherCoreData:
        if location.device.get_block_type(location) == "functional_workers":
            # For tensix, use the brisc elf
            programmable_core_type = self._ProgrammableCoreTypes_TENSIX
            enum_values = self._enum_values_tenisx
        elif location in location.device.idle_eth_block_locations:
            # For idle eth, use the idle erisc elf
            programmable_core_type = self._ProgrammableCoreTypes_IDLE_ETH
            enum_values = self._enum_values_eth
        elif location in location.device.active_eth_block_locations:
            # For active eth, use the active erisc elf
            programmable_core_type = self._ProgrammableCoreTypes_ACTIVE_ETH
            enum_values = self._enum_values_eth
        else:
            raise TTTriageError(f"Unsupported block type: {location.device.get_block_type(location)}")

        # Get the build_env for the device to get the correct firmware path
        # Each device may have different firmware paths based on its build configuration
        device_unique_id = location._device.unique_id
        build_env = self._get_build_env_for_device(device_unique_id)
        proc_name = risc_name.upper()
        proc_type = enum_values["ProcessorTypes"][proc_name]
        if mailboxes is None:
            mailboxes = self.read_mailboxes(location)

        # Refer to tt_metal/api/tt-metalium/dev_msgs.h for struct kernel_config_msg_t
        launch_msg_rd_ptr = mailboxes.launch_msg_rd_ptr

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
        try:
            # Indexed with enum ProgrammableCoreType - tt_metal/hw/inc/*/core_config.h
            kernel_config_base = mailboxes.launch[launch_msg_rd_ptr].kernel_config.kernel_config_base[
                programmable_core_type
            ]
        except:
            pass
        try:
            # Size 5 (NUM_PROCESSORS_PER_CORE_TYPE) - seems to be DM0,DM1,MATH0,MATH1,MATH2
            kernel_text_offset = mailboxes.launch[launch_msg_rd_ptr].kernel_config.kernel_text_offset[proc_type]
        except:
            pass
        try:
            # enum dispatch_core_processor_classes
            watcher_kernel_id = mailboxes.launch[launch_msg_rd_ptr].kernel_config.watcher_kernel_ids[proc_type]
        except:
            pass
        try:
            watcher_previous_kernel_id = mailboxes.launch[previous_launch_msg_rd_ptr].kernel_config.watcher_kernel_ids[
                proc_type
            ]
        except:
            pass
        try:
            kernel = self.find_kernel(watcher_kernel_id)
        except:
            pass
        try:
            previous_kernel = self.find_kernel(watcher_previous_kernel_id)
        except:
            pass
        try:
            go_message_index = mailboxes.go_message_index
            go_data = mailboxes.go_messages[go_message_index].signal
        except:
            pass
        try:
            preload = mailboxes.launch[launch_msg_rd_ptr].kernel_config.preload != 0
        except:
            pass
        try:
            host_assigned_id = mailboxes.launch[launch_msg_rd_ptr].kernel_config.host_assigned_id
        except:
            pass
        try:
            waypoint_bytes = mailboxes.watcher.debug_waypoint[proc_type].waypoint.read_bytes()
            waypoint = waypoint_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")
        except:
            pass

        # Construct the firmware path from the build_env instead of relative paths
        # This ensures we get the correct firmware path for this device and build config
        if location in location.device.active_eth_block_locations:
            if proc_name.lower() == "erisc":
                firmware_path = os.path.join(build_env.firmwarePath, "erisc", "erisc.elf")
            elif proc_name.lower() == "erisc0":
                firmware_path = os.path.join(build_env.firmwarePath, "active_erisc", "active_erisc.elf")
            elif proc_name.lower() == "erisc1":
                firmware_path = (
                    os.path.join(build_env.firmwarePath, "subordinate_active_erisc", "subordinate_active_erisc.elf")
                    if self._is_2_erisc_mode
                    else os.path.join(build_env.firmwarePath, "active_erisc", "active_erisc.elf")
                )

        else:
            if proc_name.lower() == "erisc" or proc_name.lower() == "erisc0":
                firmware_path = os.path.join(build_env.firmwarePath, "idle_erisc", "idle_erisc.elf")
            elif proc_name.lower() == "erisc1":
                firmware_path = os.path.join(
                    build_env.firmwarePath, "subordinate_idle_erisc", "subordinate_idle_erisc.elf"
                )
            else:
                firmware_path = os.path.join(build_env.firmwarePath, proc_name.lower(), f"{proc_name.lower()}.elf")
        firmware_path = os.path.realpath(firmware_path)

        if kernel:
            if location in location.device.active_eth_block_locations:
                if proc_name.lower() == "erisc":
                    kernel_path = kernel.path + "/erisc/erisc.elf"
                elif proc_name.lower() == "erisc0":
                    kernel_path = kernel.path + "/active_erisc/active_erisc.elf" if self._is_2_erisc_mode else None
                elif proc_name.lower() == "erisc1":
                    kernel_path = (
                        kernel.path + "/subordinate_active_erisc/subordinate_active_erisc.elf"
                        if self._is_2_erisc_mode
                        else kernel.path + "/active_erisc/active_erisc.elf"
                    )
            else:
                if proc_name.lower() == "erisc" or proc_name.lower() == "erisc0":
                    kernel_path = kernel.path + "/idle_erisc/idle_erisc.elf"
                elif proc_name.lower() == "erisc1":
                    kernel_path = kernel.path + "/subordinate_idle_erisc/subordinate_idle_erisc.elf"
                else:
                    kernel_path = kernel.path + f"/{proc_name.lower()}/{proc_name.lower()}.elf"
            kernel_path = os.path.realpath(kernel_path)
            # For NCRISC we don't have XIP ELF file
            kernel_xip_path = (
                kernel_path + ".xip.elf" if not (proc_name == "NCRISC" and location.device.is_wormhole()) else None
            )
            if proc_name == "NCRISC" and location.device.is_wormhole():
                kernel_offset = 0xFFC00000
            # In wormhole we only use text offset to calculate the kernel offset for active ETH
            elif location in location.device.active_eth_block_locations and location.device.is_wormhole():
                kernel_offset = kernel_text_offset
            else:
                kernel_offset = kernel_config_base + kernel_text_offset
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
        )


@triage_singleton
def run(args, context: Context):
    inspector_data = get_inspector_data(args, context)
    elfs_cache = get_elfs_cache(args, context)
    run_checks = get_run_checks(args, context)
    metal_device_id_mapping = get_metal_device_id_mapping(args, context)
    return DispatcherData(inspector_data, elfs_cache, run_checks, metal_device_id_mapping)


if __name__ == "__main__":
    run_script()
