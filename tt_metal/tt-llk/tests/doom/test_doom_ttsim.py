#!/usr/bin/env python3
import atexit
import ctypes
import os
import select
import struct
import sys
import termios

_DIR = os.path.dirname(os.path.abspath(__file__))

LIBTTSIM_PATH = os.path.join(_DIR, "libttsim_qsr.so")
DOOM_ELF = os.path.join(_DIR, "qsr_doom")

TILE_X = 2
TILE_Y = 2

SCREEN_WIDTH = 160
SCREEN_HEIGHT = 100

GPU_ADDR1 = 0x00020000
GPU_ADDR2 = 0x00024000
KB_ADDR = 0x0001F000
BUFFER_SWITCH_ADDR = 0x0001F004

KB_ESC = 0x80
KB_SPACE = 0x40
KB_D = 0x20
KB_S = 0x10
KB_A = 0x08
KB_W = 0x04
KB_SHIFT = 0x02
KB_CTRL = 0x01

CLOCKS_PER_POLL = 30000

DEBUG_REG_BASE = 0x1800000
SOFT_RESET_0 = DEBUG_REG_BASE + 0xB8
TRISC0_RESET_PC = DEBUG_REG_BASE + 0x100
TRISC_RESET_PC_OVERRIDE = DEBUG_REG_BASE + 0x110

BOOTSTRAP_ADDR = 0x3FFFF0
BOOTSTRAP_CODE = struct.pack("<II", 0x00400137, 0x00000067)

rgb_r = [((i & 0x07) >> 0) * 36 for i in range(256)]
rgb_g = [((i & 0x38) >> 3) * 36 for i in range(256)]
rgb_b = [((i & 0xC0) >> 6) * 85 for i in range(256)]


_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(LIBTTSIM_PATH)
        _lib.libttsim_tile_wr_bytes.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        _lib.libttsim_tile_wr_bytes.restype = None
        _lib.libttsim_tile_rd_bytes.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        _lib.libttsim_tile_rd_bytes.restype = None
        _lib.libttsim_clock.argtypes = [ctypes.c_uint32]
        _lib.libttsim_clock.restype = None
    return _lib


def tile_write(addr, data):
    buf = (ctypes.c_uint8 * len(data))(*data)
    _get_lib().libttsim_tile_wr_bytes(TILE_X, TILE_Y, addr, buf, len(data))


def tile_read(addr, size):
    buf = (ctypes.c_uint8 * size)()
    _get_lib().libttsim_tile_rd_bytes(TILE_X, TILE_Y, addr, buf, size)
    return bytes(buf)


def tile_write32(addr, val):
    tile_write(addr, struct.pack("<I", val))


def advance_clock(n):
    _get_lib().libttsim_clock(n)


def deassert_trisc0(entry_pc):
    tile_write32(TRISC_RESET_PC_OVERRIDE, 0x1)
    tile_write32(TRISC0_RESET_PC, entry_pc)
    tile_write32(SOFT_RESET_0, 0x7000)


def parse_elf_segments(path):
    with open(path, "rb") as f:
        ehdr = f.read(52)
        if ehdr[:4] != b"\x7fELF":
            raise ValueError("Not an ELF file")
        if ehdr[4] != 1:
            raise ValueError("Not a 32-bit ELF")

        e_entry = struct.unpack_from("<I", ehdr, 24)[0]
        e_phoff = struct.unpack_from("<I", ehdr, 28)[0]
        e_phentsize = struct.unpack_from("<H", ehdr, 42)[0]
        e_phnum = struct.unpack_from("<H", ehdr, 44)[0]

        segments = []
        for i in range(e_phnum):
            f.seek(e_phoff + i * e_phentsize)
            phdr = f.read(e_phentsize)
            p_type = struct.unpack_from("<I", phdr, 0)[0]
            if p_type != 1:
                continue
            p_offset = struct.unpack_from("<I", phdr, 4)[0]
            p_vaddr = struct.unpack_from("<I", phdr, 8)[0]
            p_filesz = struct.unpack_from("<I", phdr, 16)[0]
            p_memsz = struct.unpack_from("<I", phdr, 20)[0]

            f_pos = f.tell()
            f.seek(p_offset)
            data = f.read(p_filesz)
            f.seek(f_pos)

            if p_memsz > p_filesz:
                data += b"\x00" * (p_memsz - p_filesz)

            segments.append((p_vaddr, data))

        return e_entry, segments


def load_elf_to_device(elf_path):
    entry, segments = parse_elf_segments(elf_path)
    for vaddr, data in segments:
        chunk_size = 4096
        for off in range(0, len(data), chunk_size):
            tile_write(vaddr + off, data[off : off + chunk_size])
    return entry


old_termios = None


def setup_terminal():
    global old_termios
    old_termios = termios.tcgetattr(sys.stdin)
    new = termios.tcgetattr(sys.stdin)
    new[3] &= ~(termios.ICANON | termios.ECHO)
    new[6][termios.VMIN] = 0
    new[6][termios.VTIME] = 0
    termios.tcsetattr(sys.stdin, termios.TCSANOW, new)
    sys.stdout.write("\033[2J\033[H\033[?25l\033[?1049h")
    sys.stdout.flush()


def restore_terminal():
    if old_termios is not None:
        termios.tcsetattr(sys.stdin, termios.TCSANOW, old_termios)
    sys.stdout.write("\033[2J\033[H\033[?25h\033[?1049l")
    sys.stdout.flush()


def poll_keyboard():
    key_pressed = 0
    while select.select([sys.stdin], [], [], 0)[0]:
        buf = os.read(sys.stdin.fileno(), 32)
        for c in buf:
            if c == ord("w") or c == ord("W"):
                key_pressed |= KB_W
            elif c == ord("s") or c == ord("S"):
                key_pressed |= KB_S
            elif c == ord("a") or c == ord("A"):
                key_pressed |= KB_A
            elif c == ord("d") or c == ord("D"):
                key_pressed |= KB_D
            elif c == ord(" "):
                key_pressed |= KB_SPACE
            elif c == 27:
                key_pressed |= KB_ESC
            elif c == 13 or c == 10:
                key_pressed |= KB_CTRL
            elif c == 3:
                raise KeyboardInterrupt
    return key_pressed


screen_buffer = bytearray(SCREEN_WIDTH * SCREEN_HEIGHT)


def render_framebuffer(fb_data):
    parts = []
    parts.append("\033[H")
    for y in range(SCREEN_HEIGHT):
        for x in range(SCREEN_WIDTH):
            val = fb_data[y * SCREEN_WIDTH + x]
            parts.append("\033[38;2;%d;%d;%dm██" % (rgb_r[val], rgb_g[val], rgb_b[val]))
        parts.append("\033[0m\n")
    frame = "".join(parts).encode("utf-8")
    os.write(sys.stdout.fileno(), frame)


def run_doom():
    atexit.register(restore_terminal)

    lib = _get_lib()
    lib.libttsim_init()

    load_elf_to_device(DOOM_ELF)
    tile_write(BOOTSTRAP_ADDR, BOOTSTRAP_CODE)
    deassert_trisc0(BOOTSTRAP_ADDR)
    advance_clock(1000000)

    setup_terminal()

    prev_switch = 0xFF
    key_pressed = 0

    try:
        while True:
            advance_clock(CLOCKS_PER_POLL)

            buf_switch = tile_read(BUFFER_SWITCH_ADDR, 1)[0]

            if buf_switch != prev_switch:
                prev_switch = buf_switch
                key_pressed = 0

                read_addr = GPU_ADDR1 if buf_switch else GPU_ADDR2
                fb_data = tile_read(read_addr, SCREEN_WIDTH * SCREEN_HEIGHT)

                changed = False
                for i in range(SCREEN_WIDTH * SCREEN_HEIGHT):
                    if screen_buffer[i] != fb_data[i]:
                        screen_buffer[i] = fb_data[i]
                        changed = True
                if changed:
                    render_framebuffer(screen_buffer)

            keys = poll_keyboard()
            key_pressed |= keys
            tile_write(KB_ADDR, bytes([key_pressed]))

    except KeyboardInterrupt:
        pass
    finally:
        restore_terminal()


def test_doom():
    run_doom()


if __name__ == "__main__":
    run_doom()
