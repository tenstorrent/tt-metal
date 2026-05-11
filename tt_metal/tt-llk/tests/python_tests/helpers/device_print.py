# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Parser for the DEVICE_PRINT buffer, port of dprint_parser.cpp.

Protocol (must match tt_metal/hw/inc/api/debug/device_print.h
and tt_metal/hw/inc/hostdev/device_print_common.h):

    DEVICE_PRINT_BUFFER_BASE = 0x16D800 (subject to change I suppose), with layout as follows:
    - struct Aux { uint32 wpos; uint32 rpos; uint8 risc_state[5]; uint32 lock; }
    - uint8 data[DPRINT_BUFFER_SIZE * PROCESSOR_COUNT - sizeof(struct Aux)]

    Each record:
    - [4-byte DevicePrintHeader]
    - [message_payload bytes of args, packed in size-descending order]
      where header = is_kernel:1 | risc_id:5 | message_payload:10 | info_id:16

    info_id indexes into the .device_print_strings_info ELF section, which is
    an array of DevicePrintStringInfo32: { format_ptr, file_ptr, line, pad }.
    Format strings live in .device_print_strings; CTSTR args are also
    pointers into that section.
"""

import re
import struct
from dataclasses import dataclass
from pathlib import Path

from helpers.logger import logger
from ttexalens.tt_exalens_lib import parse_elf, read_from_device, write_words_to_device

DPRINT_BUFFER_SIZE = 204
PROCESSOR_COUNT_TENSIX = 5
AUX_SIZE = 20
TOTAL_BUFFER_SIZE = (
    DPRINT_BUFFER_SIZE * PROCESSOR_COUNT_TENSIX
)  # enforced at compile time
DATA_SIZE = TOTAL_BUFFER_SIZE - AUX_SIZE
STRING_INFO_RECORD_SIZE = 16

# Sentinels (dprint_common.h/device_print.h).
DEVICE_PRINT_RESET_BUFFER_MAGIC = 0xF0E1D2C3
DEVICE_PRINT_WRITE_STALL_FLAG = 1 << 31
WRAP_INFO_ID = 0xFFFF
NEW_KERNEL_PAYLOAD_SENTINEL = 1023  # max_message_payload_size, used as "no payload" tag

# Map DevicePrintHeader.risc_id into human-readable names.
# Indexed by PROCESSOR_INDEX in dprint.h.
RISC_NAMES_TENSIX = {
    2: "UNPACK",
    3: "MATH",
    4: "PACK",
    5: "SFPU",  # Quasar
}

# Map type character -> (struct format, byte size, optional formatter override)
TYPE_TABLE: dict[str, tuple[str, int, object]] = {
    "b": ("b", 1, str),
    "B": ("B", 1, str),
    "h": ("h", 2, str),
    "H": ("H", 2, str),
    "i": ("i", 4, str),
    "I": ("I", 4, str),
    "q": ("q", 8, str),
    "Q": ("Q", 8, str),
    "f": ("f", 4, str),
    "d": ("d", 8, str),
    "?": ("B", 1, lambda v: "true" if v else "false"),
    "e": ("H", 2, lambda v: f"bf4(0x{v:04x})"),
    "E": ("H", 2, lambda v: f"bf8(0x{v:04x})"),
    "w": ("H", 2, lambda v: f"bf16(0x{v:04x})"),
    "p": ("I", 4, lambda v: f"0x{v:x}"),
    "s": ("I", 4, None),  # pointer into .device_print_strings
}

# Matches the {N,T[:spec]} tokens that device print writes into ELF format strings
# at compile time with update_format_string. Original source {} placeholders are
# rewritten to {N,T}, where:
#   N: slot index in the packed payload (possibly reordered)
#   T: type char from device_print_type<T>::value.type_char, OR the extended
#      form "/e_X_EnumName" / "/E_X_EnumName" for enum args (base char in group "base")
#   spec: optional fmtlib format specifier (width, precision, fill, type letter, ...)
# See Metal's parse_format_string and parse_placeholder.
PLACEHOLDER_RE = re.compile(
    r"\{(\d+),"
    r"(/[eE]_(?P<base>.)_[A-Za-z_:][A-Za-z_:0-9]*|.)"
    r"(?::(?P<spec>[^{}]*))?\}"
)


@dataclass
class StringInfo:
    format_string_ptr: int
    file_ptr: int
    line: int


class ElfStrings:
    """Caches .device_print_strings and .device_print_strings_info from an ELF."""

    def __init__(self, elf_path: str):
        self.elf_path = elf_path
        self._strings_addr: int | None = None
        self._strings_data: bytes = b""
        self._info_addr: int | None = None
        self._info_data: bytes = b""
        try:
            elf = parse_elf(elf_path, require_debug_symbols=False)
            for s in elf.sections:
                if s.name == ".device_print_strings":
                    self._strings_addr = s.address
                    self._strings_data = bytes(s.data)
                elif s.name == ".device_print_strings_info":
                    self._info_addr = s.address
                    self._info_data = bytes(s.data)
        except Exception:
            pass  # can't parse ELF; has_device_print will be False

    @property
    def has_device_print(self) -> bool:
        return self._info_addr is not None and self._strings_addr is not None

    def info_at(self, info_id: int) -> StringInfo | None:
        if not self.has_device_print:
            return None
        offset = info_id * STRING_INFO_RECORD_SIZE
        if offset + STRING_INFO_RECORD_SIZE > len(self._info_data):
            return None
        fmt_ptr, file_ptr, line, _pad = struct.unpack_from(
            "<IIII", self._info_data, offset
        )
        return StringInfo(fmt_ptr, file_ptr, line)

    def string_at(self, va: int) -> str:
        if self._strings_addr is None:
            return f"<no .device_print_strings (va=0x{va:x})>"
        offset = va - self._strings_addr
        if offset < 0 or offset >= len(self._strings_data):
            return f"<bad string ptr 0x{va:x}>"
        end = self._strings_data.find(b"\x00", offset)
        if end == -1:
            end = len(self._strings_data)
        return self._strings_data[offset:end].decode("utf-8", errors="replace")


def _decode_aux(buf: bytes) -> tuple[int, int]:
    wpos, rpos = struct.unpack_from("<II", buf, 0)
    return wpos, rpos


def _strip_stall(wpos: int) -> int:
    return wpos & ~DEVICE_PRINT_WRITE_STALL_FLAG


class DevicePrintParser:
    """Reads the on-device DEVICE_PRINT ring buffer and renders records as text.

    Construct with a {risc_id: elf_path} mapping (risc_id matches the value
    written into DevicePrintHeader.risc_id by the kernel, i.e. PROCESSOR_INDEX).
    Call drain(location) after a kernel run to pull and decode all pending
    records.
    """

    def __init__(
        self,
        elf_paths: dict[int, str | Path],
        buffer_base: int,
    ):
        self.buffer_base = buffer_base
        self.elfs: dict[int, ElfStrings] = {}
        for risc_id, p in elf_paths.items():
            self.elfs[risc_id] = ElfStrings(str(p))
        self._poll_rpos: int = 0

    def _walk_records(self, data_slice: bytes) -> list[str]:
        """Parse all complete records in data_slice, stopping at a wrap marker or truncation."""
        out: list[str] = []
        pos = 0
        while pos + 4 <= len(data_slice):
            header = struct.unpack_from("<I", data_slice, pos)[0]
            is_kernel = header & 0x1
            risc_id = (header >> 1) & 0x1F
            message_payload = (header >> 6) & 0x3FF
            info_id = (header >> 16) & 0xFFFF

            if (
                not is_kernel
                and not risc_id
                and info_id == WRAP_INFO_ID
                and message_payload == 0
            ):
                break  # wrap-around marker; caller handles crossing to offset 0

            risc_name = RISC_NAMES_TENSIX.get(risc_id, f"risc{risc_id}")
            elf = self.elfs.get(risc_id)

            if is_kernel and message_payload == NEW_KERNEL_PAYLOAD_SENTINEL:
                out.append(f"[{risc_name}] <new kernel id={info_id}>")
                pos += 4
                continue

            if pos + 4 + message_payload > len(data_slice):
                break  # truncated record; we'll re-read next poll

            if elf is None or not elf.has_device_print:
                out.append(
                    f"[{risc_name}] <no ELF for risc_id={risc_id}, info_id={info_id}>"
                )
                pos += 4 + message_payload
                continue

            info = elf.info_at(info_id)
            if info is None:
                out.append(
                    f"[{risc_name}] <bad info_id={info_id}> (payload={message_payload}B)"
                )
                pos += 4 + message_payload
                continue

            fmt = elf.string_at(info.format_string_ptr)
            file_str = elf.string_at(info.file_ptr) if info.file_ptr else "?"
            args_blob = data_slice[pos + 4 : pos + 4 + message_payload]
            rendered = self._render(fmt, args_blob, elf)
            kind = "kernel" if is_kernel else "fw"
            out.append(f"[{risc_name}|{kind}|{file_str}:{info.line}] {rendered}")
            pos += 4 + message_payload

        return out

    def drain(self, location: str = "0,0") -> list[str]:
        """One-shot read of the full buffer after kernel completion. Does not update device rpos."""
        raw = read_from_device(location, self.buffer_base, num_bytes=TOTAL_BUFFER_SIZE)
        wpos, rpos = _decode_aux(raw)
        wpos = _strip_stall(wpos)

        data = bytes(raw[AUX_SIZE:])
        if rpos > wpos:  # wrapped around
            return self._walk_records(data[rpos:]) + self._walk_records(data[:wpos])
        return self._walk_records(data[rpos:wpos])

    def poll(self, location: str = "0,0") -> list[str]:
        """Incremental drain: read new data since last poll, advance device rpos.

        Mirrors DevicePrintImpl::poll_one_core in dprint_server.cpp. Handles
        buffer wrap-around and kernel-stall (DEVICE_PRINT_WRITE_STALL_FLAG).
        Safe to call from the mailbox-wait loop — returns immediately if nothing new.
        """
        out: list[str] = []

        while True:
            aux_raw = read_from_device(location, self.buffer_base, num_bytes=AUX_SIZE)
            wpos_raw, _ = _decode_aux(aux_raw)

            stall = bool(wpos_raw & DEVICE_PRINT_WRITE_STALL_FLAG)
            wpos = _strip_stall(wpos_raw)

            if self._poll_rpos > wpos:
                # Kernel wrapped wpos back to 0; drain the tail then fall through to head.
                tail_size = DATA_SIZE - self._poll_rpos
                if tail_size > 0:
                    tail = bytes(
                        read_from_device(
                            location,
                            self.buffer_base + AUX_SIZE + self._poll_rpos,
                            num_bytes=tail_size,
                        )
                    )
                    out.extend(self._walk_records(tail))
                self._poll_rpos = 0

            if self._poll_rpos < wpos:
                chunk = bytes(
                    read_from_device(
                        location,
                        self.buffer_base + AUX_SIZE + self._poll_rpos,
                        num_bytes=wpos - self._poll_rpos,
                    )
                )
                out.extend(self._walk_records(chunk))
                self._poll_rpos = wpos

            if stall:
                # Buffer full; stall kernel and continue from offset 0.
                write_words_to_device(
                    location, self.buffer_base + 4, [DEVICE_PRINT_RESET_BUFFER_MAGIC]
                )
                self._poll_rpos = 0
                continue  # re-read wpos to catch data kernel wrote after reset

            break

        write_words_to_device(location, self.buffer_base + 4, [self._poll_rpos])
        return out

    def final_drain(self, location: str = "0,0") -> list[str]:
        """Last poll after kernel finishes. Resets poll state for the next run."""
        out = self.poll(location)
        self._poll_rpos = 0
        return out

    def _render(self, fmt: str, args_blob: bytes, elf: ElfStrings) -> str:
        placeholders = list(PLACEHOLDER_RE.finditer(fmt))
        if not placeholders:
            return fmt.replace("{{", "{").replace("}}", "}")

        # Determine type per reordered-slot index, then compute offsets in
        # ascending slot order; matches size-descending packing on device.
        type_for_ridx: dict[int, str] = {}
        for m in placeholders:
            ridx = int(m.group(1))
            type_for_ridx.setdefault(ridx, m.group(2))

        offsets: dict[int, int] = {}
        cur = 0
        for ridx in sorted(type_for_ridx):
            type_token = type_for_ridx[ridx]
            base_char = type_token[3] if type_token.startswith("/") else type_token
            entry = TYPE_TABLE.get(base_char)
            offsets[ridx] = cur
            cur += entry[1] if entry else 4

        parts: list[str] = []
        last_end = 0
        for m in placeholders:
            parts.append(fmt[last_end : m.start()])
            ridx = int(m.group(1))
            type_token = m.group(2)
            base_char = type_token[3] if type_token.startswith("/") else type_token
            entry = TYPE_TABLE.get(base_char)
            if entry is None:
                parts.append(f"<unknown type '{type_token}'>")
            else:
                struct_fmt, size, formatter = entry
                offset = offsets[ridx]
                if offset + size > len(args_blob):
                    parts.append("<truncated arg>")
                else:
                    val = struct.unpack_from("<" + struct_fmt, args_blob, offset)[0]
                    spec = m.group("spec") or ""
                    if base_char == "s":
                        rendered = elf.string_at(val)
                        parts.append(format(rendered, spec) if spec else rendered)
                    elif formatter is str:
                        # Plain numeric/float type: apply spec directly to the raw value.
                        try:
                            parts.append(format(val, spec) if spec else str(val))
                        except (ValueError, TypeError):
                            parts.append(str(val))
                    else:
                        # Custom formatter (bool, bfloat, pointer): format first, then
                        # apply spec as string padding/alignment if present.
                        rendered = formatter(val)
                        parts.append(format(rendered, spec) if spec else rendered)
            last_end = m.end()
        parts.append(fmt[last_end:])
        return "".join(parts).replace("{{", "{").replace("}}", "}")


# Any test file can import this to use device print.
def make_device_print_parser(configuration) -> DevicePrintParser:
    """Build a DevicePrintParser for the given configuration's ELFs.

    configuration.generate_variant_hash() must have been called before this.
    """
    from helpers.test_config import TestConfig  # local to avoid circular import

    # Maps KERNEL_COMPONENTS names to the TensixProcessorTypes index written into
    # DevicePrintHeader.risc_id (also the PROCESSOR_INDEX macro in dprint.h).
    COMPONENT_TO_RISC_ID: dict[str, int] = {
        "unpack": 2,
        "math": 3,
        "pack": 4,
        "sfpu": 5,
    }

    variant_elf_dir = (
        TestConfig.ARTEFACTS_DIR
        / configuration.test_name
        / configuration.variant_id
        / "elf"
    )
    elf_paths = {
        COMPONENT_TO_RISC_ID[name]: variant_elf_dir / f"{name}.elf"
        for name in TestConfig.KERNEL_COMPONENTS
    }
    return DevicePrintParser(elf_paths, TestConfig.DEVICE_PRINT_BUFFER_BASE)


def run_with_device_print(configuration):
    """Run a test variant with device print output.

    Returns: (TestOutcome, list[str]), where the list contains all dprint lines.
    """
    from helpers.test_config import TestConfig

    configuration.generate_variant_hash()
    configuration.build_elfs()  # ELFs must exist before the parser reads them
    parser = make_device_print_parser(configuration)
    all_lines: list[str] = []

    def _drain():
        batch = parser.poll(TestConfig.TENSIX_LOCATION)
        all_lines.extend(batch)
        for line in batch:
            logger.info(line)

    outcome = configuration.run(poll_callback=_drain)

    final = parser.final_drain(TestConfig.TENSIX_LOCATION)
    all_lines.extend(final)
    for line in final:
        logger.info(line)

    return outcome, all_lines
