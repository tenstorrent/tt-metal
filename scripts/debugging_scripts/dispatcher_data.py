#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dispatcher_data

Description:
    Provides dispatcher data noc locations on devices.
    Data include firmware path, kernel path, kernel offset, etc.
"""

from dataclasses import dataclass
import os
from inspector_data import run as get_inspector_data, InspectorData
from triage import triage_singleton, ScriptConfig, run_script
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.firmware import ELF
from ttexalens.parse_elf import mem_access
from ttexalens.tt_exalens_lib import parse_elf
from ttexalens.context import Context
from utils import ORANGE, RST
from triage import TTTriageError, combined_field, collection_serializer, triage_field, hex_serializer

script_config = ScriptConfig(
    data_provider=True,
    depends=["inspector_data"],
)


@dataclass
class DispatcherCoreData:
    launch_msg_rd_ptr: int = triage_field("RD PTR")
    kernel_config_base: int = triage_field("Base", hex_serializer)
    kernel_text_offset: int = triage_field("Offset", hex_serializer)
    watcher_kernel_id: int = combined_field("kernel_name", "Kernel ID:Name", collection_serializer(":"))
    kernel_offset: int | None = triage_field("Kernel Offset", hex_serializer)
    firmware_path: str = combined_field("kernel_path", "Firmware / Kernel Path", collection_serializer("\n"))
    kernel_path: str | None = combined_field()
    kernel_name: str | None = combined_field()
    subdevice: int = triage_field("Subdevice")
    go_message: str = triage_field("Go Message")
    preload: bool = triage_field("Preload")


class DispatcherData:
    def __init__(self, inspector_data: InspectorData, context: Context):
        self.inspector_data = inspector_data
        if inspector_data.kernels is None or len(inspector_data.kernels) == 0:
            raise TTTriageError("No kernels found in inspector data.")

        self._a_kernel_path = next(iter(inspector_data.kernels.values())).path
        brisc_elf_path = DispatcherData.get_firmware_elf_path(self._a_kernel_path, "brisc")
        idle_erisc_elf_path = DispatcherData.get_firmware_elf_path(self._a_kernel_path, "idle_erisc")

        # Check if firmware elf paths exist
        if not os.path.exists(brisc_elf_path):
            raise TTTriageError(f"BRISC ELF file {brisc_elf_path} does not exist.")

        if not os.path.exists(idle_erisc_elf_path):
            raise TTTriageError(f"IDLE ERISC ELF file {idle_erisc_elf_path} does not exist.")

        self._brisc_elf = parse_elf(brisc_elf_path, context)
        self._idle_erisc_elf = parse_elf(idle_erisc_elf_path, context)

        # Check if debug info is obtained correctly
        if not self._brisc_elf:
            raise TTTriageError(
                f"Failed to extract DWARF info from ELF file {brisc_elf_path}.\nRun workload with TT_METAL_RISCV_DEBUG_INFO=1 to enable debug info."
            )
        if not self._idle_erisc_elf:
            raise TTTriageError(
                f"Failed to extract DWARF info from ELF file {idle_erisc_elf_path}.\nRun workload with TT_METAL_RISCV_DEBUG_INFO=1 to enable debug info."
            )

        # Access the value of enumerator for supported blocks
        self._ProgrammableCoreTypes_TENSIX = self._brisc_elf.enumerators["ProgrammableCoreType::TENSIX"].value
        self._ProgrammableCoreTypes_IDLE_ETH = self._brisc_elf.enumerators["ProgrammableCoreType::IDLE_ETH"].value

        # Enumerators for tensix block
        self._enum_values_tenisx = {
            "ProcessorTypes": {
                "BRISC": self._brisc_elf.enumerators["TensixProcessorTypes::DM0"].value,
                "NCRISC": self._brisc_elf.enumerators["TensixProcessorTypes::DM1"].value,
                "TRISC0": self._brisc_elf.enumerators["TensixProcessorTypes::MATH0"].value,
                "TRISC1": self._brisc_elf.enumerators["TensixProcessorTypes::MATH1"].value,
                "TRISC2": self._brisc_elf.enumerators["TensixProcessorTypes::MATH2"].value,
            },
        }

        # Enumerators for eth block
        self._enum_values_eth = {
            "ProcessorTypes": {
                "ERISC": self._idle_erisc_elf.enumerators["EthProcessorTypes::DM0"].value,
                "ERISC0": self._idle_erisc_elf.enumerators["EthProcessorTypes::DM0"].value,
            },
        }

        # Blackhole has ERISC1 processor type
        try:
            self._enum_values_eth["ProcessorTypes"]["ERISC1"] = self._idle_erisc_elf.enumerators[
                "EthProcessorTypes::DM1"
            ].value
        except:
            pass

    def get_core_data(self, location: OnChipCoordinate, risc_name: str) -> DispatcherCoreData:
        loc_mem_reader = ELF.get_mem_reader(location)
        if location._device.get_block_type(location) == "functional_workers":
            # For tensix, use the brisc elf
            fw_elf = self._brisc_elf
            programmable_core_type = self._ProgrammableCoreTypes_TENSIX
            enum_values = self._enum_values_tenisx
        else:
            # For eth, use the idle erisc elf
            fw_elf = self._idle_erisc_elf
            programmable_core_type = self._ProgrammableCoreTypes_IDLE_ETH
            enum_values = self._enum_values_eth

        proc_name = risc_name.upper()
        proc_type = enum_values["ProcessorTypes"][proc_name]

        # Refer to tt_metal/api/tt-metalium/dev_msgs.h for struct kernel_config_msg_t
        launch_msg_rd_ptr = mem_access(fw_elf, "mailboxes->launch_msg_rd_ptr", loc_mem_reader)[0][0]

        def get_const_value(name):
            return mem_access(fw_elf, name, loc_mem_reader)[3]

        # Go message states are constant values in the firmware elf, so we cache them
        if not hasattr(self, "_go_message_states"):
            self._go_message_states = {
                get_const_value("RUN_MSG_INIT"): "INIT",
                get_const_value("RUN_MSG_GO"): "GO",
                get_const_value("RUN_MSG_DONE"): "DONE",
                get_const_value("RUN_MSG_RESET_READ_PTR"): "RESET_READ_PTR",
                get_const_value("RUN_MSG_RESET_READ_PTR_FROM_HOST"): "RESET_READ_PTR_FROM_HOST",
            }

        try:
            # Indexed with enum ProgrammableCoreType - tt_metal/hw/inc/*/core_config.h
            kernel_config_base = mem_access(
                fw_elf,
                f"mailboxes->launch[{launch_msg_rd_ptr}].kernel_config.kernel_config_base[{programmable_core_type}]",
                loc_mem_reader,
            )[0][0]

            # Size 5 (NUM_PROCESSORS_PER_CORE_TYPE) - seems to be DM0,DM1,MATH0,MATH1,MATH2
            kernel_text_offset = mem_access(
                fw_elf,
                f"mailboxes->launch[{launch_msg_rd_ptr}].kernel_config.kernel_text_offset[{proc_type}]",
                loc_mem_reader,
            )[0][0]

            # enum dispatch_core_processor_classes
            watcher_kernel_id = (
                mem_access(
                    fw_elf,
                    f"mailboxes->launch[{launch_msg_rd_ptr}].kernel_config.watcher_kernel_ids[{proc_type}]",
                    loc_mem_reader,
                )[0][0]
                & 0xFFFF
            )

            kernel = self.inspector_data.kernels.get(watcher_kernel_id)
            go_message_index = mem_access(fw_elf, f"mailboxes->go_message_index", loc_mem_reader)[0][0]
            go_data = mem_access(fw_elf, f"mailboxes->go_messages[{go_message_index}]", loc_mem_reader)[0][0]
            preload = (
                mem_access(fw_elf, f"mailboxes->launch[{launch_msg_rd_ptr}].kernel_config.preload", loc_mem_reader)[0][
                    0
                ]
                != 0
            )
        except:
            kernel_config_base = -1
            kernel_text_offset = -1
            watcher_kernel_id = -1
            kernel = None
            go_message_index = -1
            go_data = -1
            preload = False

        if proc_name.lower() == "erisc" or proc_name.lower() == "erisc0":
            firmware_path = self._a_kernel_path + "../../../firmware/idle_erisc/idle_erisc.elf"
        elif proc_name.lower() == "erisc1":
            firmware_path = self._a_kernel_path + "../../../firmware/subordinate_idle_erisc/subordinate_idle_erisc.elf"
        else:
            firmware_path = self._a_kernel_path + f"../../../firmware/{proc_name.lower()}/{proc_name.lower()}.elf"
        firmware_path = os.path.realpath(firmware_path)

        if kernel:
            if proc_name.lower() == "erisc" or proc_name.lower() == "erisc0":
                kernel_path = kernel.path + "/idle_erisc/idle_erisc.elf"
            elif proc_name.lower() == "erisc1":
                kernel_path = kernel.path + "/subordinate_idle_erisc/subordinate_idle_erisc.elf"
            else:
                kernel_path = kernel.path + f"/{proc_name.lower()}/{proc_name.lower()}.elf"
            kernel_path = os.path.realpath(kernel_path)
            if proc_name == "NCRISC" and location._device._arch == "wormhole_b0":
                kernel_offset = 0xFFC00000
            else:
                kernel_offset = kernel_config_base + kernel_text_offset
        else:
            kernel_path = None
            kernel_offset = None
        go_state = (go_data >> 24) & 0xFF
        go_data_state = self._go_message_states.get(go_state, str(go_state))

        return DispatcherCoreData(
            firmware_path=firmware_path,
            kernel_path=kernel_path,
            kernel_offset=kernel_offset,
            kernel_name=kernel.name if kernel else None,
            launch_msg_rd_ptr=launch_msg_rd_ptr,
            kernel_config_base=kernel_config_base,
            kernel_text_offset=kernel_text_offset,
            watcher_kernel_id=watcher_kernel_id,
            subdevice=go_message_index,
            go_message=go_data_state,
            preload=preload,
        )

    @staticmethod
    def get_firmware_elf_path(a_kernel_path: str, risc_name: str) -> str:
        firmware_elf_path = a_kernel_path + f"../../../firmware/{risc_name.lower()}/{risc_name.lower()}.elf"
        return os.path.realpath(firmware_elf_path)


@triage_singleton
def run(args, context: Context):
    inspector_data = get_inspector_data(args, context)
    return DispatcherData(inspector_data, context)


if __name__ == "__main__":
    run_script()
