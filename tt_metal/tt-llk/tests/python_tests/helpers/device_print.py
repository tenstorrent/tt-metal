# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Parser for the DEVICE_PRINT buffer, port of dprint_parser.cpp.

Protocol (must match tt_metal/hw/inc/api/debug/device_print.h
and tt_metal/hw/inc/hostdev/device_print_common.h):

    DEVICE_PRINT_BUFFER_BASE = TestConfig.DEVICE_PRINT_BUFFER_BASE (layout:
    - struct Aux { uint32 wpos; uint32 rpos; uint8 risc_state[5]; uint32 lock; }
    - uint8 data[DPRINT_BUFFER_SIZE * PROCESSOR_COUNT - sizeof(struct Aux)]

    Each record:
    - [4-byte DevicePrintHeader]
    - [message_payload bytes of args, packed in size-descending order]
      where header = is_kernel:1 | risc_id:5 | message_payload:10 | info_id:16

    info_id indexes into the .device_print_strings_info ELF section, which is
    an array of DevicePrintStringInfo: { format_ptr, file_ptr, line, pad }.
    Format strings live in .device_print_strings; CTSTR args are also
    pointers into that section.
"""

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from helpers.logger import logger
from ttexalens.tt_exalens_lib import parse_elf, read_from_device, write_words_to_device

# hostdev/device_print_structures.h: DevicePrintStringInfo is four uint32_t
# on 32-bit ELFs, and four uint64_t on 64-bit ELFs (Rocket cores on Quasar).
_STRING_INFO_LAYOUT: dict[int, tuple[int, str]] = {
    32: (16, "<IIII"),
    64: (32, "<QQQQ"),
}


# Aux struct (device_print_common.h): wpos:4 | rpos:4 | risc_state[N]:N padded
# up to 4-byte boundary | lock:4. N = PROCESSOR_COUNT, which varies per arch.
def aux_size_for(processor_count: int) -> int:
    return 4 + 4 + (processor_count + 3) // 4 * 4 + 4


# Sentinels (dprint_common.h/device_print.h).
DEVICE_PRINT_WRITE_STALL_FLAG = 1 << 31
WRAP_INFO_ID = 0xFFFF
NEW_KERNEL_PAYLOAD_SENTINEL = 1023  # max_message_payload_size, used as "no payload" tag


# Map DevicePrintHeader.risc_id (with PROCESSOR_INDEX passed to the build)
# into human-readable names. Derived from TestConfig.RISC_INFO.
def _risc_names_tensix() -> dict[int, str]:
    from helpers.test_config import TestConfig  # Local to avoid circular import

    return {risc_id: name for risc_id, name in TestConfig.RISC_INFO.values()}


# Map type character -> (struct format, byte size, optional formatter override).
# Pointer types ('s', 'p') are filled in per-ELF — see _type_table_for().
_BASE_TYPE_TABLE: dict[str, tuple[str, int, object]] = {
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
}


def _type_table_for(pointer_size: int) -> dict[str, tuple[str, int, object]]:
    """Return _BASE_TYPE_TABLE extended with the two pointer-sized entries.

    Only 's' (CTSTR pointer into the strings section) and 'p' (raw pointer,
    formatted as hex) need to match the ELF's pointer size. 32-bit ELFs
    use 'I'/4 bytes, and 64-bit ELFs use 'Q'/8 bytes.

    Each value is (struct_unpack_fmt, byte_size, formatter), where formatter is:
      - `str` for plain numerics; spec is applied directly to format()
      - a callable producing a string (bool, bfloat, pointer); spec is then
        applied to that string (alignment/padding only)
      - `None` for 's'; _render handles it specially by dereferencing the
        pointer through ElfStrings.string_at()

    The keys mirror the C++ type-char namespace produced by device_print_type<>
    in tt_metal/hw/inc/api/debug/device_print.h."""
    ptr_fmt = "I" if pointer_size == 4 else "Q"
    return {
        **_BASE_TYPE_TABLE,
        "p": (ptr_fmt, pointer_size, lambda v: f"0x{v:x}"),
        "s": (ptr_fmt, pointer_size, None),  # pointer into .device_print_strings
    }


# Matches the {N,T[:spec]} tokens that device print writes into ELF format strings
# at compile time with update_format_string. Original source {} placeholders are
# rewritten to {N,T}, where:
#   N: slot index in the packed payload (possibly reordered)
#   T: type char from device_print_type<T>::value.type_char, OR the extended
#      form "/e_X_EnumName" / "/E_X_EnumName" for enum args (base char in group "base")
#   spec: optional fmtlib format specifier (width, precision, fill, type letter, ...)
# See parse_format_string and parse_placeholder in tt_metal/impl/debug/dprint_parser.cpp.
PLACEHOLDER_RE = re.compile(
    r"\{(\d+),"
    r"(/(?P<flag>[eE])_(?P<base>.)_(?P<enum_name>[A-Za-z_][A-Za-z_0-9]*(?:::[A-Za-z_][A-Za-z_0-9]*)*)|.)"
    r"(?::(?P<spec>[^{}]*))?\}"
)


@dataclass(slots=True)
class StringInfo:
    format_string_ptr: int
    file_ptr: int
    line: int


@dataclass(frozen=True, slots=True)
class _Placeholder:
    """Compiled metadata for one placeholder in a format string.

    Built once per format string in an ELF and cached in ElfStrings,
    so high-rate prints don't re-run the regex / re-compute offsets.

    `kind` determines how the value is rendered:
      - "enum":    resolve through DWARF (uses enum_name, is_flag, use_full_name, cleaned_spec)
      - "string":  dereference into .device_print_strings (uses spec)
      - "plain":   apply spec directly to the unpacked value
      - "custom":  call formatter, then apply spec as string padding
      - "unknown": emit error_msg, no unpack
    Other fields are unused when not relevant for the kind.
    """

    kind: str
    # Pre-compiled struct.Struct for unpacking this arg from args_blob; faster
    # than calling struct.unpack_from(fmt, ...) which re-parses fmt each call.
    unpacker: Any = None
    size: int = 0
    offset: int = 0
    formatter: Any = None
    spec: str = ""
    enum_name: str = ""
    is_flag: bool = False
    use_full_name: bool = False
    cleaned_spec: str = ""
    error_msg: str = ""


@dataclass(frozen=True, slots=True)
class _RenderPlan:
    """Pre-built rendering plan for a format string.

    Iteration mode: append literals[i], render placeholders[i]
    from args_blob, repeat; finally append literals[-1].

    Literals are pre-cooked (fmtlib uses double braces
    for escaping: swap {{ for {, and }} for })
    so `_render` doesn't redo that on every record.

    When there are no placeholders, literals = [cooked_fmt] and
    `placeholders` is empty.
    """

    literals: list[str] = field(default_factory=list)
    placeholders: list[_Placeholder] = field(default_factory=list)


def _build_render_plan(
    fmt: str, type_table: dict[str, tuple[str, int, object]]
) -> _RenderPlan:
    """Parse fmt into a _RenderPlan. Pure function of (fmt, type_table)."""

    placeholders = list(PLACEHOLDER_RE.finditer(fmt))
    if not placeholders:
        return _RenderPlan(literals=[fmt.replace("{{", "{").replace("}}", "}")])

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
        entry = type_table.get(base_char)
        offsets[ridx] = cur
        cur += entry[1] if entry else 4

    literals: list[str] = []
    compiled: list[_Placeholder] = []
    last_end = 0
    for m in placeholders:
        literals.append(fmt[last_end : m.start()].replace("{{", "{").replace("}}", "}"))
        ridx = int(m.group(1))
        type_token = m.group(2)
        base_char = type_token[3] if type_token.startswith("/") else type_token
        entry = type_table.get(base_char)
        spec = m.group("spec") or ""

        if entry is None:
            compiled.append(
                _Placeholder(kind="unknown", error_msg=f"<unknown type '{type_token}'>")
            )
        else:
            struct_fmt, size, formatter = entry
            offset = offsets[ridx]
            unpacker = struct.Struct("<" + struct_fmt)
            if type_token.startswith("/"):
                # Enum: '#' in the spec means we should render with the
                # typename:: prefix. Consume the '#' here and tell
                # format_enum to use the full name.
                compiled.append(
                    _Placeholder(
                        kind="enum",
                        unpacker=unpacker,
                        size=size,
                        offset=offset,
                        spec=spec,
                        enum_name=m.group("enum_name"),
                        is_flag=(m.group("flag") == "E"),
                        use_full_name="#" in spec,
                        cleaned_spec=spec.replace("#", ""),
                    )
                )
            elif base_char == "s":
                compiled.append(
                    _Placeholder(
                        kind="string",
                        unpacker=unpacker,
                        size=size,
                        offset=offset,
                        spec=spec,
                    )
                )
            elif formatter is str:
                compiled.append(
                    _Placeholder(
                        kind="plain",
                        unpacker=unpacker,
                        size=size,
                        offset=offset,
                        spec=spec,
                    )
                )
            else:
                compiled.append(
                    _Placeholder(
                        kind="custom",
                        unpacker=unpacker,
                        size=size,
                        offset=offset,
                        formatter=formatter,
                        spec=spec,
                    )
                )
        last_end = m.end()

    literals.append(fmt[last_end:].replace("{{", "{").replace("}}", "}"))
    return _RenderPlan(literals=literals, placeholders=compiled)


class ElfStrings:
    """Caches .device_print_strings and .device_print_strings_info from an ELF."""

    def __init__(self, elf_path: str):
        self.elf_path = elf_path
        self._strings_addr: int | None = None
        self._strings_data: bytes = b""
        self._info_addr: int | None = None
        self._info_data: bytes = b""
        self._info_record_size: int = 16
        self._info_unpack_fmt: str = "<IIII"
        self.pointer_size: int = 4
        self.type_table: dict[str, tuple[str, int, object]] = _type_table_for(4)
        self._parsed_elf = None  # kept around for DWARF enum lookups
        self._enum_cache: dict[str, list[tuple[int, str]] | None] = {}
        # Per-format-string render plan cache. Keyed by the raw format string;
        # the type_table is fixed per-ELF, so the plan is reusable across
        # records that share a format string (e.g. tight DEVICE_PRINT loops).
        self._render_plan_cache: dict[str, _RenderPlan] = {}

        try:
            elf = parse_elf(elf_path, require_debug_symbols=False)
            self._parsed_elf = elf
            self.pointer_size = elf.elf.elfclass // 8
            self.type_table = _type_table_for(self.pointer_size)
            self._info_record_size, self._info_unpack_fmt = _STRING_INFO_LAYOUT[
                elf.elf.elfclass
            ]
            for s in elf.sections:
                if s.name == ".device_print_strings":
                    self._strings_addr = s.address
                    self._strings_data = bytes(s.data)
                elif s.name == ".device_print_strings_info":
                    self._info_addr = s.address
                    self._info_data = bytes(s.data)
        except Exception:
            # has_device_print will be false.
            logger.exception(
                "Failed to parse device print sections from %s; "
                'records from this RISC will say "no ELF".',
                elf_path,
            )

    def _enumerators(self, enum_name: str) -> list[tuple[int, str]] | None:
        """List of (value, name) for `enum_name` in DWARF declaration order.
        Return None if the enum isn't in the ELF's debug info."""

        if enum_name in self._enum_cache:
            return self._enum_cache[enum_name]
        result: list[tuple[int, str]] | None = None
        if self._parsed_elf is not None:
            try:
                die = self._parsed_elf.find_die_by_name(enum_name)
                if die is not None:
                    result = [(int(c.value), c.name) for c in die.iter_children()]
            except Exception:
                result = None
        self._enum_cache[enum_name] = result
        return result

    def format_enum(
        self, enum_name: str, value: int, is_flag: bool, use_full_name: bool = False
    ) -> str:
        """Resolve an enum value to its named form.
        Flag enums OR-combine matching bits with ' | '; regular enums return the
        single matching name. Unknown values (and missing DWARF) render as
        '(typename)value'. With use_full_name, each name is prefixed with 'typename::'.
        """
        prefix = f"{enum_name}::" if use_full_name else ""
        members = self._enumerators(enum_name)
        if not members:
            return f"({enum_name}){value}"

        # Regular enum, or flag enum with value 0: find first match.
        if not is_flag or value == 0:
            for eval_, name in members:
                if eval_ == value:
                    return f"{prefix}{name}"
            return f"({enum_name}){value}"

        # Flag enum: OR-combine matching bits.
        # Any leftover bits render as (typename)value.
        bits_left = value
        parts: list[str] = []
        for eval_, name in members:
            if eval_ != 0 and (bits_left & eval_) == eval_:
                parts.append(f"{prefix}{name}")
                bits_left &= ~eval_
        if bits_left != 0:
            parts.append(f"({enum_name}){bits_left}")
        return " | ".join(parts)

    @property
    def has_device_print(self) -> bool:
        return self._info_addr is not None and self._strings_addr is not None

    def get_render_plan(self, fmt: str) -> _RenderPlan:
        """Return the cached _RenderPlan for fmt, or build it on miss.
        Optimization for tight DEVICE_PRINT loops."""

        plan = self._render_plan_cache.get(fmt)
        if plan is None:
            plan = _build_render_plan(fmt, self.type_table)
            self._render_plan_cache[fmt] = plan
        return plan

    def info_at(self, info_id: int) -> StringInfo | None:
        """Look up the DevicePrintStringInfo record at index `info_id`.

        `info_id` is the 16-bit field from DevicePrintHeader; it indexes into
        the .device_print_strings_info ELF section as an array of fixed-size
        records (4 uint32s on 32-bit ELFs, 4 uint64s on 64-bit; see
        _STRING_INFO_LAYOUT).

        Return None if device print sections weren't loaded or the index points
        past the end of the array."""

        if not self.has_device_print:
            return None
        offset = info_id * self._info_record_size
        if offset + self._info_record_size > len(self._info_data):
            return None
        fmt_ptr, file_ptr, line, _ = struct.unpack_from(
            self._info_unpack_fmt, self._info_data, offset
        )
        return StringInfo(fmt_ptr, file_ptr, line)

    def string_at(self, va: int) -> str:
        """Read a string from .device_print_strings at virtual address `va`.

        The address is stored in the ELF (format string pointers in
        DevicePrintStringInfo, and CTSTR arguments in the payload;
        see sections.ld). Resolved by subtracting the section base address.

        Return a '<...>' placeholder if the section is missing or the VA falls outside it.
        We don't raise so a single bad pointer can't break an entire drain."""

        if self._strings_addr is None:
            return f"<no .device_print_strings (va=0x{va:x})>"
        offset = va - self._strings_addr
        if offset < 0 or offset >= len(self._strings_data):
            return f"<bad string ptr 0x{va:x}>"
        end = self._strings_data.find(b"\x00", offset)
        if end == -1:
            end = len(self._strings_data)

        return self._strings_data[offset:end].decode("utf-8", errors="replace")


def _decode_wpos_rpos(buf: bytes) -> tuple[int, int]:
    wpos, rpos = struct.unpack_from("<II", buf, 0)
    return wpos, rpos


def _strip_stall(wpos: int) -> int:
    return wpos & ~DEVICE_PRINT_WRITE_STALL_FLAG


class DevicePrintParser:
    """Reads the on-device DEVICE_PRINT ring buffer and renders records as text.

    Construct with a {risc_id: elf_path} mapping (risc_id matches the value written
    into DevicePrintHeader.risc_id by the kernel, and PROCESSOR_INDEX passed by the
    build to dprint.h). Call poll(location) during the run and final_drain(location)
    after to pull and decode all pending records.
    """

    def __init__(
        self,
        elf_paths: dict[int, str | Path],
        buffer_base: int,
        total_buffer_size: int,
        processor_count: int,
    ):
        self.buffer_base = buffer_base
        self.total_buffer_size = total_buffer_size
        self.aux_size = aux_size_for(processor_count)
        self.data_size = total_buffer_size - self.aux_size
        self.elfs: dict[int, ElfStrings] = {}
        for risc_id, p in elf_paths.items():
            self.elfs[risc_id] = ElfStrings(str(p))
        self._risc_names: dict[int, str] = _risc_names_tensix()

    def _walk_records(self, data_slice: bytes) -> list[str]:
        """Parse all complete records in data_slice,
        stopping at a wrap marker or truncation."""

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
                break  # Wrap-around marker; caller handles crossing to offset 0

            risc_name = self._risc_names.get(risc_id, f"risc{risc_id}")
            elf = self.elfs.get(risc_id)

            if is_kernel and message_payload == NEW_KERNEL_PAYLOAD_SENTINEL:
                out.append(f"[{risc_name}] <new kernel id={info_id}>")
                pos += 4
                continue

            if pos + 4 + message_payload > len(data_slice):
                break  # Truncated record; re-read next poll

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

            out.append(f"[{risc_name}|{file_str}:{info.line}] {rendered}")
            pos += 4 + message_payload

        return out

    # We'd spin forever if the kernel hangs while holding up the flag,
    # so we bound the poll loop; 64 is reasonable.
    _MAX_STALL_RESETS_PER_POLL: int = 64

    def poll(self, location: str = "0,0") -> list[str]:
        """Incremental drain: read new data since last poll, advance device rpos.
        Return immediately if there is no new data.
        See dprint_server.cpp:read_core_data.

        Kernel is the sole writer of wpos, host is the sole writer of rpos.
        Unlike Metal, we don't write RESET_BUFFER_MAGIC to rpos due to a race
        that happens if both host and device write to the same address.
        """
        out: list[str] = []

        aux_raw = read_from_device(location, self.buffer_base, num_bytes=self.aux_size)
        wpos, rpos = _decode_wpos_rpos(aux_raw)

        # Nothing to do: device has no new data.
        if wpos == rpos:
            return out

        for _ in range(self._MAX_STALL_RESETS_PER_POLL):
            stall = bool(wpos & DEVICE_PRINT_WRITE_STALL_FLAG)
            wpos = _strip_stall(wpos)

            if rpos > wpos:
                # Kernel wrapped wpos back to 0; drain the tail then fall through to head.
                tail_size = self.data_size - rpos
                if tail_size > 0:
                    tail = bytes(
                        read_from_device(
                            location,
                            self.buffer_base + self.aux_size + rpos,
                            num_bytes=tail_size,
                        )
                    )
                    out.extend(self._walk_records(tail))
                rpos = 0

            if rpos < wpos:
                chunk = bytes(
                    read_from_device(
                        location,
                        self.buffer_base + self.aux_size + rpos,
                        num_bytes=wpos - rpos,
                    )
                )
                out.extend(self._walk_records(chunk))
                rpos = wpos

            write_words_to_device(location, self.buffer_base + 4, [rpos])

            if not stall:
                return out

            # Kernel was stalled when we entered this iteration. Re-read
            # and loop: either it has cleared the stall and produced more
            # data, or it hasn't yet observed our rpos write and we retry.
            aux_raw = read_from_device(
                location, self.buffer_base, num_bytes=self.aux_size
            )
            wpos, rpos = _decode_wpos_rpos(aux_raw)

        raise RuntimeError(
            f"DevicePrintParser.poll(): stall flag still set after "
            f"{self._MAX_STALL_RESETS_PER_POLL} iterations, kernel likely hung."
        )

    def final_drain(self, location: str = "0,0") -> list[str]:
        """Last poll after kernel finishes."""
        return self.poll(location)

    def _render(self, fmt: str, args_blob: bytes, elf: ElfStrings) -> str:
        # Plan (regex scan + offset computation + literal cooking) is cached
        # on `elf` per format string, so this is one dict lookup on the hot path.
        plan = elf.get_render_plan(fmt)
        literals = plan.literals
        placeholders = plan.placeholders

        if not placeholders:
            return literals[0]

        parts: list[str] = []
        blob_len = len(args_blob)
        for i, ph in enumerate(placeholders):
            parts.append(literals[i])

            if ph.kind == "unknown":
                parts.append(ph.error_msg)
                continue

            if ph.offset + ph.size > blob_len:
                parts.append("<truncated arg>")
                continue

            val = ph.unpacker.unpack_from(args_blob, ph.offset)[0]
            spec = ph.spec

            if ph.kind == "enum":
                rendered = elf.format_enum(
                    ph.enum_name,
                    val,
                    is_flag=ph.is_flag,
                    use_full_name=ph.use_full_name,
                )
                cleaned_spec = ph.cleaned_spec
                parts.append(
                    format(rendered, cleaned_spec) if cleaned_spec else rendered
                )
            elif ph.kind == "string":
                rendered = elf.string_at(val)
                parts.append(format(rendered, spec) if spec else rendered)
            elif ph.kind == "plain":
                # Plain int/float: apply spec directly.
                try:
                    parts.append(format(val, spec) if spec else str(val))
                except (ValueError, TypeError):
                    # DEVICE_PRINT enforces spec correctness at compile time,
                    # but fmtlib isn't 1:1 with Python's format(), so we're
                    # conservative here.
                    parts.append(str(val))
            else:  # "custom"
                # Custom formatter (bool, bfloat, pointer): format first, then
                # apply spec as string padding/alignment if present.
                rendered = ph.formatter(val)
                try:
                    parts.append(format(rendered, spec) if spec else rendered)
                except (ValueError, TypeError):
                    parts.append(str(rendered))

        parts.append(literals[-1])
        return "".join(parts)


# Any test file can import this to use device print.
def make_device_print_parser(configuration) -> DevicePrintParser:
    """Build a DevicePrintParser for the given configuration's ELFs.

    configuration.prepare() must have been called before this so the ELFs exist.
    """
    from helpers.test_config import TestConfig  # Local to avoid circular import

    if TestConfig.PROCESSOR_COUNT == 0:
        raise RuntimeError(
            "TestConfig.setup_arch() must be called before make_device_print_parser()."
        )

    variant_elf_dir = (
        TestConfig.ARTEFACTS_DIR
        / configuration.test_name
        / configuration.variant_id
        / "elf"
    )
    elf_paths = {
        TestConfig.RISC_INFO[name][0]: variant_elf_dir / f"{name}.elf"
        for name in TestConfig.KERNEL_COMPONENTS
    }
    return DevicePrintParser(
        elf_paths,
        TestConfig.DEVICE_PRINT_BUFFER_BASE,
        TestConfig.DEVICE_PRINT_BUFFER_SIZE,
        TestConfig.PROCESSOR_COUNT,
    )
