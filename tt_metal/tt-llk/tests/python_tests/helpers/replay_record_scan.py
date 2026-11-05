# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Find Blackhole replay-buffer records in a built ELF that instruction gathering would see.

With gathering on (ENABLE_GATHERING), every record must sit inside disable_gathering() /
enable_gathering(), as ckernel::load_replay_buf emits it. The source lint
(infra/check_replay_record_helper.py) cannot see records the compiler makes on its own or a
helper that lost its bracket; only the binary shows them. TestConfig.build_elfs runs this on
every gathering build, and test_gathering_config.py pins it on known recording kernels.
"""

import struct


def _executable_words_with_addresses(elf_path):
    """Every 4-byte aligned word of every executable PROGBITS section, with the word address
    list and the address -> word index map (for following jumps)."""
    data = elf_path.read_bytes()
    assert data[:4] == b"\x7fELF" and data[4] == 1, f"{elf_path} is not a 32-bit ELF"

    (e_shoff,) = struct.unpack_from("<I", data, 0x20)
    e_shentsize, e_shnum, _ = struct.unpack_from("<HHH", data, 0x2E)

    words, word_addresses, index_of_address = [], [], {}
    for i in range(e_shnum):
        _, sh_type, sh_flags, sh_addr, sh_offset, sh_size = struct.unpack_from(
            "<IIIIII", data, e_shoff + i * e_shentsize
        )
        SHT_PROGBITS, SHF_EXECINSTR = 1, 0x4
        if sh_type != SHT_PROGBITS or not (sh_flags & SHF_EXECINSTR):
            continue
        for off in range(sh_offset, sh_offset + sh_size - 3, 4):
            index_of_address[sh_addr + (off - sh_offset)] = len(words)
            word_addresses.append(sh_addr + (off - sh_offset))
            words.append(struct.unpack_from("<I", data, off)[0])
    return words, (word_addresses, index_of_address)


def _executable_words(elf_path):
    """Every 4-byte aligned word of every executable PROGBITS section."""
    return _executable_words_with_addresses(elf_path)[0]


# Replay-record checks for ENABLE_GATHERING builds (tenstorrent/tt-llk#1701 item 9).
# RISC-V CSR fields: csr[31:20] rs1[19:15] funct3[14:12] rd[11:7] op[6:0]; rs1 is whatever
# register the compiler picked for the bit mask.
CFG0_CSR = 0x7C0
FUNCT3_CSRRS, FUNCT3_CSRRC = 0b010, 0b011
# disable_gathering() is csrrs, fence, csrrs, csrrc, fence, three TTNOPs, plus scheduling slack.
DISABLE_WINDOW_WORDS = 16
# Walk from a record to its enable_gathering() csrrc: counts Tensix words (RISC-V code is
# scheduled among them) and follows unconditional jumps (GCC cross-jumps common body tails).
BODY_WALK_LIMIT_WORDS = 512
TT_OPCODE_REPLAY = 0x04


def _cfg0_csr_op(word):
    """'s' / 'c' for csrrs / csrrc zero, 0x7c0, <any rs1>; None otherwise."""
    if word & 0x7F != 0x73 or (word >> 7) & 0x1F != 0 or word >> 20 != CFG0_CSR:
        return None
    return {FUNCT3_CSRRS: "s", FUNCT3_CSRRC: "c"}.get((word >> 12) & 0x7)


def _replay_records(words):
    """(index, length) of every REPLAY that records (load_mode=1), from the inline
    ``.ttinsn`` encoding: the Tensix word rotated left by two bits."""
    records = []
    for i, word in enumerate(words):
        if word & 0x3 == 0x3:  # 32-bit RISC-V opcode space: not a Tensix instruction
            continue
        insn = (word >> 2) | ((word & 0x3) << 30)
        if insn >> 24 == TT_OPCODE_REPLAY and insn & 0x1:
            records.append((i, (insn >> 4) & 0x3FF))
    return records


# A runtime-length record (_llk_pack_block_contiguous_mop_config_) is not an inline .ttinsn word:
# the compiler composes it as `lui rd, 0x4000` (REPLAY opcode), `addi rd, rd, 1` (load_mode), sw.
RISCV_OPCODE_LUI = 0x37
RISCV_OPCODE_OP_IMM = 0x13
RISCV_OPCODE_STORE = 0x23
RISCV_OPCODE_JAL = 0x6F
FUNCT3_SW = 0b010
REPLAY_OPCODE_LUI_IMM = (
    TT_OPCODE_REPLAY << 12
)  # lui immediate placing 0x04 in bits 31:24
RUNTIME_RECORD_COMPOSE_WORDS = 12
RUNTIME_RECORD_DISABLE_WINDOW_WORDS = 32


# RISC-V opcodes whose instruction writes rd (bits 11:7).
RISCV_OPCODES_WRITING_RD = {0x37, 0x17, 0x6F, 0x67, 0x03, 0x13, 0x33, 0x73}
FUNCT3_ADDI, FUNCT3_ORI = 0b000, 0b110


def _runtime_replay_records(words):
    """(index of the instruction-buffer store, None) of every REPLAY record composed at runtime.

    The value the ``lui`` starts is completed by the first later instruction that writes the same
    register; only an ``addi``/``ori`` of that register onto itself is followed. The word is a
    record only if that value has the load_mode bit (bit 0) set. A later, unrelated constant built
    in the same register (``lui rd, X; addi rd, rd, 1``) is not mistaken for this one.
    """
    records = []
    for i, word in enumerate(words):
        if word & 0x7F != RISCV_OPCODE_LUI or word >> 12 != REPLAY_OPCODE_LUI_IMM:
            continue
        rd = (word >> 7) & 0x1F
        end = min(i + RUNTIME_RECORD_COMPOSE_WORDS, len(words))
        value, compose = word & 0xFFFFF000, None
        for j in range(i + 1, end):
            w = words[j]
            if (
                not _is_tensix_word(w)
                and w & 0x7F in RISCV_OPCODES_WRITING_RD
                and (w >> 7) & 0x1F == rd
            ):
                funct3, rs1 = (w >> 12) & 0x7, (w >> 15) & 0x1F
                if (
                    w & 0x7F == RISCV_OPCODE_OP_IMM
                    and rs1 == rd
                    and funct3 in (FUNCT3_ADDI, FUNCT3_ORI)
                ):
                    imm = w >> 20
                    if imm & 0x800:
                        imm -= 0x1000
                    value = (
                        (value + imm)
                        if funct3 == FUNCT3_ADDI
                        else (value | (imm & 0xFFFFFFFF))
                    )
                    compose = j
                break  # rd is redefined: the lui's value ends here
        if compose is None or not value & 0x1:
            continue  # a playback word (no load_mode bit), e.g. a MOP template operand
        for j in range(compose + 1, end):
            if (
                words[j] & 0x7F == RISCV_OPCODE_STORE
                and (words[j] >> 12) & 0x7 == FUNCT3_SW
            ):
                records.append((j, None))
                break
    return records


def _is_tensix_word(word):
    """Inline ``.ttinsn`` words are the only words outside the 32-bit RISC-V opcode space."""
    return word & 0x3 != 0x3


def _jal_zero_target(words, index, addresses):
    """Word index of the target of ``jal zero, imm`` at ``index``, or None."""
    word = words[index]
    if word & 0x7F != RISCV_OPCODE_JAL or (word >> 7) & 0x1F != 0:
        return None
    imm = (
        ((word >> 31) & 0x1) << 20
        | ((word >> 12) & 0xFF) << 12
        | ((word >> 20) & 0x1) << 11
        | ((word >> 21) & 0x3FF) << 1
    )
    if imm & (1 << 20):
        imm -= 1 << 21
    word_addresses, index_of_address = addresses
    return index_of_address.get(word_addresses[index] + imm)


def _first_cfg0_op_after(words, index, addresses):
    """(op, tensix words seen) for the first cfg0 CSR op after word ``index``, following
    unconditional jumps; op is None if none is found within the walk limit."""
    tensix = 0
    position = index + 1
    for _ in range(BODY_WALK_LIMIT_WORDS):
        if position >= len(words):
            break
        target = _jal_zero_target(words, position, addresses)
        if target is not None:
            position = target
            continue
        op = _cfg0_csr_op(words[position])
        if op:
            return op, tensix
        if _is_tensix_word(words[position]):
            tensix += 1
        position += 1
    return None, tensix


def _unbracketed_records(
    words, records, addresses, disable_window_words=DISABLE_WINDOW_WORDS
):
    """The records not bracketed by disable_gathering()'s csrrs/csrrs/csrrc before and
    enable_gathering()'s csrrc as the first cfg0 CSR op after the body. A body that issues
    more Tensix instructions than its record length (an under-counted record: the tail would
    not be in the buffer) is reported too; a runtime-length record (length None) is only
    checked for the bracket.
    """
    unbracketed = []
    for index, length in records:
        before = [
            _cfg0_csr_op(w) for w in words[max(0, index - disable_window_words) : index]
        ]
        before = [op for op in before if op]
        disabled = before[-3:] == ["s", "s", "c"]
        after, tensix = _first_cfg0_op_after(words, index, addresses)
        enabled = after == "c"
        overlong = length is not None and tensix > length
        if not (disabled and enabled) or overlong:
            unbracketed.append(
                f"record at word {index} (len {'runtime' if length is None else length}): "
                f"disable_gathering before={disabled}, enable_gathering after={enabled}, "
                f"Tensix instructions before the enable={tensix}"
            )
    return unbracketed


def check_elf_records(elf_path):
    """All replay records of ``elf_path``, and the ones not bracketed."""
    words, addresses = _executable_words_with_addresses(elf_path)
    inline = _replay_records(words)
    runtime = _runtime_replay_records(words)
    unbracketed = _unbracketed_records(words, inline, addresses) + _unbracketed_records(
        words, runtime, addresses, RUNTIME_RECORD_DISABLE_WINDOW_WORDS
    )
    return inline + runtime, unbracketed
