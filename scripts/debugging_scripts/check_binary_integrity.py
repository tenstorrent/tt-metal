#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_binary_integrity

Description:
    Checking that code binaries are the way they were uploaded to the device (both firmware and kernel).
"""

from collections import namedtuple
from io import BytesIO
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from elfs_cache import run as get_elfs_cache, ElfsCache
from run_checks import run as get_run_checks
from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection
from elftools.elf.sections import Section as ELFSection
from elftools.elf.segments import Segment as ELFSegment
from elftools.elf.relocation import Relocation as ELFRelocation
import os
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_from_device
from triage import ScriptConfig, log_check, run_script
from sortedcontainers import SortedDict

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache"],
)


# TODO: This method should be removed and ElfFile::Impl::XIPify() should be used to store corrected binaries.
def apply_kernel_relocations(section: ELFSection) -> bytes:
    # Check for relocation section
    elf_file: ELFFile = section.elffile
    rel_section: RelocationSection = section.elffile.get_section_by_name(f".rela{section.name}")
    if rel_section is None:
        return section.data()

    # Create a stream to hold the section data
    section_address: int = section["sh_addr"]
    section_stream = BytesIO()
    section_stream.write(section.data())

    def write32(offset: int, value: int) -> None:
        section_stream.seek(offset - section_address)
        section_stream.write(value.to_bytes(4, byteorder="little"))

    def read32(offset: int) -> int:
        section_stream.seek(offset - section_address)
        return int.from_bytes(section_stream.read(4), byteorder="little")

    # Constants for RISC-V relocations
    R_RISCV_32 = 1
    R_RISCV_PCREL_HI20 = 23
    R_RISCV_PCREL_LO12_I = 24
    R_RISCV_PCREL_LO12_S = 25
    R_RISCV_HI20 = 26
    R_RISCV_LO12_I = 27
    R_RISCV_LO12_S = 28
    SHF_ALLOC = 1 << 1

    # We have to translate these two instructions
    insn_opc_auipc = 0x00000017
    insn_opc_lui = 0x00000037
    insn_mask_u = 0x0000007F
    mask_hi20 = 0x00000FFF
    mask_hi20_shift = 12
    mask_lo12_i = 0x000FFFFF
    mask_lo12_i_shift = 20
    mask_lo12_s = 0x01FFF07F
    mask_lo12_s_split = 5
    mask_lo12_s_shift_1 = 7
    mask_lo12_s_shift_2 = 25

    # Find important relocations that we should apply
    ComposedRelocation = namedtuple("ComposedRelocation", ["lo_relocs", "hi_reloc"])
    lo: dict[str, list[ELFRelocation]] = {
        "PCREL": [],
        "ABS": [],
    }
    composed: dict[str, SortedDict] = {  # SortedDict[int, ComposedRelocation]
        "PCREL": SortedDict(),
        "ABS": SortedDict(),
    }

    def in_segment(section: ELFSection, segment: ELFSegment) -> bool:
        return (
            section["sh_flags"] & SHF_ALLOC
            and section["sh_addr"] >= segment["p_vaddr"]
            and section["sh_addr"] + section["sh_size"] <= segment["p_vaddr"] + segment["p_memsz"]
        )

    segment_ix = -1
    for segment_index in range(elf_file.num_segments(), 0, -1):
        segment = elf_file.get_segment(segment_index - 1)
        if in_segment(section, segment):
            segment_ix = segment_index - 1
            break
    segment = elf_file.get_segment(segment_ix)
    symtab = section.elffile.get_section(rel_section["sh_link"])
    text_segment = None
    for i in range(elf_file.num_segments()):
        seg = elf_file.get_segment(i)
        if seg["p_type"] != "PT_RISCV_ATTRIBUTES":
            text_segment = seg
            break
    assert text_segment is not None, "Text segment not found in ELF file"
    is_from_text = segment == text_segment
    for i in range(rel_section.num_relocations(), 0, -1):
        relocation = rel_section.get_relocation(i - 1)
        # for relocation in rel_section.iter_relocations():
        symbol = symtab.get_symbol(relocation["r_info_sym"])
        symbol_section = (
            elf_file.get_section(symbol["st_shndx"])
            if isinstance(symbol["st_shndx"], int) and symbol["st_shndx"] < elf_file.num_sections()
            else None
        )
        is_to_text = symbol_section is not None and in_segment(symbol_section, text_segment)
        type = relocation["r_info_type"]
        kind = "PCREL"
        if (
            type == R_RISCV_LO12_I
            or type == R_RISCV_LO12_S
            or type == R_RISCV_PCREL_LO12_I
            or type == R_RISCV_PCREL_LO12_S
        ):
            if type == R_RISCV_LO12_I or type == R_RISCV_LO12_S:
                kind = "ABS"
            if kind != "ABS" or is_to_text:
                # PCrel relocs to text will not need translation,
                # but at this point we don't know the symbol as
                # these relocs point to the hi20 reloc.  Record
                # them all and filter later.
                lo[kind].append(relocation)
        elif type == R_RISCV_HI20 or type == R_RISCV_PCREL_HI20:
            if type == R_RISCV_HI20:
                kind = "ABS"
            if kind != "ABS" or is_to_text:
                assert (
                    relocation["r_offset"] not in composed[kind]
                ), f"Duplicate relocation {relocation['r_offset']} for kind {kind}"
                composed[kind][relocation["r_offset"]] = ComposedRelocation([], relocation)
        elif type == R_RISCV_32:
            if is_to_text:
                value = symbol["st_value"] + relocation["r_addend"] - elf_file.get_segment(0)["p_vaddr"]
                write32(relocation["r_offset"], value)
                # TODO: Save relocation?!?
                # TODO: auto& seg = GetSegments()[segment_ix];
                # TODO: seg.relocs.push_back(reloc.r_offset - seg.address);
                pass

    # Combine hi/lo relocs

    # We can't do abs ones in general with complete accuracy,
    # because there could be multiple possible matching hi
    # relocs. If we construct the CFG then it becomes more
    # accurate, but it's always going to be somewhat
    # heuristic. Let's hope CFG construction is unnecessary. A
    # first step in that direction might be to insert function
    # boundaries, to stop the search.
    for kind in ["PCREL", "ABS"]:
        comp = composed[kind]
        for lo_reloc in lo[kind]:
            # Find the matching hi-reloc by searching backwards. This
            # presumes block reordering hasn't done something to
            # break that.
            sym_ix = lo_reloc["r_info_sym"]
            if kind == "ABS":
                hi_reloc = None
                for o, c in comp.items():
                    if o > lo_reloc["r_offset"]:
                        break
                    if c.hi_reloc["r_info_sym"] == sym_ix:
                        hi_reloc = c
            else:
                hi_offset = symtab.get_symbol(sym_ix)["st_value"] + lo_reloc["r_addend"]
                hi_reloc = comp.get(hi_offset)
            assert (
                hi_reloc is not None
            ), f"No matching hi reloc for {kind} reloc at {lo_reloc['r_offset']} with symbol index {sym_ix}"
            hi_reloc.lo_relocs.append(lo_reloc)

    # Process composed relocations
    for kind in ["PCREL", "ABS"]:
        for o, c in composed[kind].items():
            assert len(c.lo_relocs) > 0, f"No lo relocs for composed {kind} reloc at {o}"
            hi_reloc = c.hi_reloc
            sym_ix = hi_reloc["r_info_sym"]
            symbol = symtab.get_symbol(sym_ix)
            symbol_section = (
                elf_file.get_section(symbol["st_shndx"])
                if isinstance(symbol["st_shndx"], int) and symbol["st_shndx"] < elf_file.num_sections()
                else None
            )
            is_to_text = symbol_section is not None and in_segment(symbol_section, text_segment)
            if kind == "PCREL" and is_to_text == is_from_text:
                # intra-text PCREL is ok.
                continue
            value = symbol["st_value"] + hi_reloc["r_addend"]
            if kind == "ABS":
                value -= o
                sym_ix = 0

            # Translate hi
            insn = read32(hi_reloc["r_offset"])
            insn &= mask_hi20  # Remove old immediate
            insn ^= insn_opc_auipc ^ insn_opc_lui  # Convert opcode
            # Insert new immediate
            insn |= ((value + (1 << 11)) >> 12) << mask_hi20_shift
            write32(hi_reloc["r_offset"], insn)
            # TODO: hi_reloc->r_info ^= ELF32_R_INFO(0, R_RISCV_HI20 ^ R_RISCV_PCREL_HI20);

            # Translate lo
            for lo_reloc in c.lo_relocs:
                type = lo_reloc["r_info_type"]
                is_form_i = type == (R_RISCV_PCREL_LO12_I if kind == "PCREL" else R_RISCV_LO12_I)
                insn = read32(lo_reloc["r_offset"])
                if is_form_i:
                    insn &= mask_lo12_i
                    insn |= (value & 0x0FFF) << mask_lo12_i_shift
                else:
                    # S form splits the immediate
                    insn &= mask_lo12_s
                    insn |= (value & ((1 << mask_lo12_s_split) - 1)) << mask_lo12_s_shift_1
                    insn |= ((value & 0x0FFF) >> mask_lo12_s_split) << mask_lo12_s_shift_2
                write32(lo_reloc["r_offset"], insn)

    # TODO: Do we need to update lo_reloc?
    #                 // We can't convert to PCREL with fidelity, as
    #                 // that involves adding a symbol. Instead, let's
    #                 // use a null symbol and an addend.
    #                 lo_reloc->r_info = ELF32_R_INFO(
    #                     sym_ix,
    #                     type ^ (is_form_i ? (R_RISCV_LO12_I ^ R_RISCV_PCREL_LO12_I)
    #                                       : (R_RISCV_LO12_S ^ R_RISCV_PCREL_LO12_S)));
    #                 lo_reloc->r_addend = kind == PCREL ? slot.second.hi_reloc->r_addend
    #                                                    : slot.second.hi_reloc->r_offset - lo_reloc->r_offset;
    #             }
    #         }
    #     }
    # }

    return section_stream.getvalue()


def check_binary_integrity(
    location: OnChipCoordinate, risc_name: str, dispatcher_data: DispatcherData, elfs_cache: ElfsCache
):
    dispatcher_core_data = dispatcher_data.get_core_data(location, risc_name)

    # Check firmware ELF binary state on the device
    log_check(
        os.path.exists(dispatcher_core_data.firmware_path),
        f"Firmware ELF file {dispatcher_core_data.firmware_path} does not exist.",
    )
    if os.path.exists(dispatcher_core_data.firmware_path):
        elf_file = elfs_cache[dispatcher_core_data.firmware_path].elf
        sections_to_verify = [".text"]
        for section_name in sections_to_verify:
            section = elf_file.get_section_by_name(section_name)
            if section is None:
                log_check(
                    False,
                    f"Section {section_name} not found in ELF file {dispatcher_core_data.firmware_path}.",
                )
            else:
                address: int = section["sh_addr"]
                data: bytes = section.data()
                read_data = read_from_device(location, address, num_bytes=len(data))
                log_check(
                    read_data == data,
                    f"{location.to_user_str()}: Data mismatch in section {section_name} at address 0x{address:08x} in ELF file {dispatcher_core_data.firmware_path}.",
                )

    # Check kernel ELF binary state on the device
    if dispatcher_core_data.kernel_path is not None:
        log_check(
            os.path.exists(dispatcher_core_data.kernel_path),
            f"Kernel ELF file {dispatcher_core_data.kernel_path} does not exist.",
        )

        # We cannot read 0xFFC00000 address on wormhole as we don't have debug hardware on NCRISC (only NCRISC has private code memory at that address).
        if (
            os.path.exists(dispatcher_core_data.kernel_path)
            and dispatcher_core_data.kernel_offset is not None
            and dispatcher_core_data.kernel_offset != 0xFFC00000
        ):
            elf_file = elfs_cache[dispatcher_core_data.kernel_path].elf
            sections_to_verify = [".text"]
            for section_name in sections_to_verify:
                section = elf_file.get_section_by_name(section_name)
                if section is None:
                    log_check(
                        False,
                        f"Section {section_name} not found in ELF file {dispatcher_core_data.kernel_path}.",
                    )
                else:
                    data = apply_kernel_relocations(section)
                    address: int = dispatcher_core_data.kernel_offset
                    read_data = read_from_device(location, address, num_bytes=len(data))
                    log_check(
                        read_data == data,
                        f"{location.to_user_str()}: Data mismatch in section {section_name} at address 0x{address:08x} in ELF file {dispatcher_core_data.kernel_path}.",
                    )


def run(args, context: Context):
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth"]
    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)
    run_checks = get_run_checks(args, context)
    run_checks.run_per_core_check(
        lambda location, risc_name: check_binary_integrity(location, risc_name, dispatcher_data, elfs_cache),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )


if __name__ == "__main__":
    run_script()
