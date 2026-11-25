# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Pytest plugin: single-step one TRISC through a test variant via the tt-exalens GDB server and
# count what it issues.
#
# Usage (from tt_metal/tt-llk/tests):
#   DYN_GDB_CORE=math DYN_GDB_CLIENT=mi DYN_GDB_OUT=/path/out.jsonl DYN_GDB_STATUS=/path/status.jsonl \
#     PYTHONPATH=<dir of this file> pytest -p dyn_gdb_plugin "python_tests/test_x.py::test_x[...]" --timeout=0 -q
#
# The start is the same as in dyn_step_plugin.py: a core in reset cannot be halted, so the first
# word at the ELF entry (_start) of the chosen TRISC is replaced in L1 by `j .`; after START_TRISCS
# the core spins there, is halted, the word is restored, the instruction cache invalidated and the
# pipeline flushed to the entry. From there a GDB client takes over through the GDB server of
# tt-exalens (ttexalens/gdb/gdb_server.py), started inside this pytest process: it attaches to the
# core (vAttach), steps it one RISC-V instruction at a time (vCont;s; on Blackhole the server issues
# two hardware step commands per step), reads the PC and the base register of every store, and
# detaches (D, which resumes the core) after the core has stored KERNEL_COMPLETE to its mailbox.
#   DYN_GDB_CLIENT=mi   riscv-tt-elf-gdb from tests/sfpi, driven over its MI interface
#                       (this gdb is built without Python scripting)
#   DYN_GDB_CLIENT=rsp  minimal remote serial protocol client below, same server packets
# Per step the PC is looked up in the ELF disassembly (kept in memory only), as in dyn_step_plugin.py:
#   tti          tt* instruction (Tensix word embedded in the code)
#   tt           store whose address is in the instruction buffer (word built at runtime)
#   sfp          sfp* instruction
#   other_store  any other store (MMIO, L1, ...)
#   flow_errors  PC after a non-control-flow instruction is not the next address
# DYN_GDB_OUT gets the counts. DYN_GDB_STATUS gets only the variant id, completion flags, error
# counts and seconds, never instruction counts, addresses or client messages.

import json
import os
import re
import select
import socket
import subprocess
import time
from collections import Counter

CORE_NAMES = ["unpack", "math", "pack"]
MAILBOX_BASE = 0x1FFB8  # tests/helpers/src/trisc.cpp (non-coverage build)
KERNEL_COMPLETE = 0xFF  # helpers/device.py
INSTRN_BUFFER = (0xFFE40000, 0xFFE40000 + 3 * 0x10000)  # helpers/ld/memory.blackhole.ld
PC_REGISTER = 32
JUMP_TO_SELF = 0x0000006F  # j .
M32 = 0xFFFFFFFF
CONTROL_FLOW = re.compile(r"^(c\.)?(b\w*|j\w*|call|tail|ret|mret|ecall|ebreak)$")
STORES = {"sw", "sh", "sb", "c.sw", "c.swsp"}
REGISTERS = "zero ra sp gp tp t0 t1 t2 s0 s1 a0 a1 a2 a3 a4 a5 a6 a7 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 t3 t4 t5 t6".split()
REGISTER_INDEX = {name: i for i, name in enumerate(REGISTERS)}
REGISTER_INDEX.update({f"x{i}": i for i in range(32)})
REGISTER_INDEX["fp"] = 8
LINE = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)\s*(.*)$")
STORE_OPERANDS = re.compile(r"^(\w+),\s*(-?(?:0x[0-9a-fA-F]+|\d+))?\((\w+)\)")
STOPPED_PC = re.compile(r'frame=\{addr="0x([0-9a-fA-F]+)"')
REGISTER_VALUE = re.compile(r'value="0x([0-9a-fA-F]+)"')
STATUS_FIELDS = [
    "spin_pc_is_entry",
    "start_pc_is_entry",
    "completed",
    "completed_by_poll",
    "time_limit",
    "lost_halt",
    "flow_errors",
    "flow_same_pc",
    "flow_forward",
    "flow_backward",
    "unknown_pc",
    "unparsed_store",
    "client_error_type",
    "client_error_stage",
    "server_risc_debug_class",
    "server_halts",
    "settle_timeouts",
    "steps_not_sent",
    "stale_stop_replies",
    "mi_trace",
    "seconds",
]

CORE = os.environ.get("DYN_GDB_CORE", "math")
CLIENT = os.environ.get("DYN_GDB_CLIENT", "mi")
# 1 = no tt-exalens safety reads (is the core in reset?) between the two hardware step commands
# that BlackholeBabyRiscDebug.step issues; the server creates its RiscDebug objects with the
# class default BabyRiscDebugHardware.ENABLE_ASSERTS.
NO_ASSERTS = os.environ.get("DYN_GDB_NO_ASSERTS") == "1"
# 1 = after the two hardware step commands, BlackholeBabyRiscDebug.step polls until the core reports
# halted (as dyn_step_plugin.py does). Without it the server reads the status once and halts the core
# if it still runs, which can cut the step short.
SETTLE = os.environ.get("DYN_GDB_SETTLE") == "1"
# gdb_server.py (tt-exalens 0.3.31) queues an extra bare "T05" after every stop reply of a vCont and
# answers the next vCont with it without stepping, so every second step does nothing.
#   DYN_GDB_FIX_PENDING=1  drop that queued "T05" inside this process (the installed file is not changed)
#   otherwise              the rsp client recognises the bare "T05" (no thread id) and sends the step again
FIX_PENDING = os.environ.get("DYN_GDB_FIX_PENDING") == "1"
OUT = os.environ.get("DYN_GDB_OUT", "")  # required: where the counts go
STATUS = os.environ.get("DYN_GDB_STATUS", "")  # required: per-variant status records
MAX_SECONDS = float(os.environ.get("DYN_GDB_MAX_SECONDS", "3600"))
IO_TIMEOUT = 30.0
CORE_INDEX = CORE_NAMES.index(CORE)

state = {"config": None, "nodeid": None, "risc": None, "armed": False, "server": None}


def elf_entry(path):
    with open(path, "rb") as f:
        header = f.read(0x1C)
    return int.from_bytes(header[0x18:0x1C], "little")


def disassemble(path, objdump):
    text = subprocess.run(
        [objdump, "-d", "-w", path], capture_output=True, text=True, check=True
    ).stdout
    result = {}
    for line in text.splitlines():
        m = LINE.match(line)
        if m:
            size = len(m.group(2)) // 2
            result[int(m.group(1), 16)] = (
                m.group(3),
                m.group(4).split("#")[0].strip(),
                size,
            )
    return result


def risc_debug(location):
    import helpers.device as dev

    context = dev.check_context()
    device = context.devices[0]
    block = device.get_block(dev.convert_coordinate(location, 0, context))
    return block.get_risc_debug(f"trisc{CORE_INDEX}")


def append_json(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def write_record(record):
    record["nodeid"] = state["nodeid"]
    record["core"] = CORE
    record["client"] = CLIENT
    append_json(OUT, record)
    status = {"nodeid": state["nodeid"], "core": CORE, "client": CLIENT}
    status.update({k: record[k] for k in STATUS_FIELDS if k in record})
    append_json(STATUS, status)


def gdb_server():
    """GDB server of tt-exalens on a free local port, one per pytest process."""
    if state["server"] is None:
        import helpers.device as dev
        from ttexalens.gdb.gdb_communication import ServerSocket
        from ttexalens.gdb.gdb_server import GdbServer

        if FIX_PENDING:
            original_process_message = GdbServer.process_message

            def process_message(self, parser, writer):
                result = original_process_message(self, parser, writer)
                if self.vCont_pending_statuses == ["T05"]:
                    self.vCont_pending_statuses.clear()
                return result

            GdbServer.process_message = process_message

        server_socket = ServerSocket()
        server_socket.start()
        # debug_only_with_elfs: the server lists only cores with a loaded ELF instead of scanning
        # every core of the chip on each vCont.
        server = GdbServer(
            dev.check_context(), server_socket, debug_only_with_elfs=True
        )
        server.start()
        state["server"] = server
    return state["server"]


def gdb_process_id(server, rd, elf_path):
    """GDB process id of the stepped core; registers its ELF with the context if needed."""
    for attempt in range(2):
        for pid, process in server.available_processes.items():
            if (
                process.elf_path == elf_path
                and process.risc_debug.risc_location.risc_name
                == rd.risc_location.risc_name
            ):
                state["server_risc_debug_class"] = type(process.risc_debug).__name__
                return pid
        server.context.elf_loaded(rd.risc_location, elf_path)
    return None


class RspClient:
    """Minimal remote serial protocol client: only the packets the stepping needs."""

    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=IO_TIMEOUT)
        self.buffer = b""
        self.ack = True
        self.request("QStartNoAckMode")
        self.ack = False
        self.pid = None

    def request(self, payload):
        data = payload.encode()
        self.sock.sendall(b"$" + data + b"#" + f"{sum(data) % 256:02x}".encode())
        while True:
            start = self.buffer.find(b"$")
            end = self.buffer.find(b"#", start + 1) if start >= 0 else -1
            if start >= 0 and end >= 0 and len(self.buffer) >= end + 3:
                reply = self.buffer[start + 1 : end].decode(errors="replace")
                self.buffer = self.buffer[end + 3 :]
                if self.ack:
                    self.sock.sendall(b"+")
                return reply
            if start < 0:
                self.buffer = b""  # only acks so far
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("GDB server closed the connection")
            self.buffer += chunk

    def attach(self, pid):
        reply = self.request(f"vAttach;{pid:x}")
        if not reply.startswith("T"):
            raise RuntimeError(f"vAttach: {reply}")
        self.pid = pid
        self.thread = f"p{pid:X}.{pid:X}"  # GdbProcess.thread_id: virtual core id = pid

    def register(self, index):
        reply = self.request(f"p{index:x}")
        if reply.startswith("E"):
            raise RuntimeError(f"p{index:x}: {reply}")
        return int(
            reply, 16
        )  # gdb_server.py writes 'p' replies as a plain big-endian hex number

    def step(self):
        reply = self.request(f"vCont;s:{self.thread}")
        if reply == "T05":
            state["stale_stop_replies"] = state.get("stale_stop_replies", 0) + 1
            reply = self.request(f"vCont;s:{self.thread}")
        if not reply.startswith("T") or "thread:" not in reply:
            raise RuntimeError(f"vCont;s: {reply}")
        return self.register(PC_REGISTER)

    def close(self):
        try:
            if self.pid is not None:
                self.request(f"D;{self.pid:x}")
        finally:
            self.sock.close()


def mi_record_kind(line):
    """Record type of an MI output line without its content (no symbols, values or messages)."""
    kind = line.lstrip("0123456789")
    if kind[:1] in "^*=":
        reason = re.search(r'reason="([a-z-]+)"', kind)
        return kind.split(",", 1)[0] + (f"({reason.group(1)})" if reason else "")
    return {"~": "~console", "&": "&log", "@": "@target"}.get(kind[:1], kind[:6])


class MiClient:
    """riscv-tt-elf-gdb driven over its MI interface."""

    def __init__(self, gdb, port):
        self.proc = subprocess.Popen(
            [gdb, "--nx", "--quiet", "--interpreter=mi2"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.buffer = b""
        self.token = 0
        self.attached = False
        self.trace = []
        state["mi_trace"] = self.trace
        for command in [
            "-gdb-set confirm off",
            "-gdb-set pagination off",
            "-gdb-set sysroot /",  # read the ELF from the local file system, not over vFile
            "-gdb-set architecture riscv:rv32",
            f"-target-select extended-remote 127.0.0.1:{port}",
        ]:
            self.command(command)

    def line(self):
        deadline = time.time() + IO_TIMEOUT
        while b"\n" not in self.buffer:
            left = deadline - time.time()
            if left <= 0 or not select.select([self.proc.stdout], [], [], left)[0]:
                raise TimeoutError("no output from gdb")
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise ConnectionError("gdb exited")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        line = line.decode(errors="replace").rstrip("\r")
        self.trace.append(mi_record_kind(line))
        del self.trace[:-12]
        return line

    def command(self, text, wait_stopped=False):
        self.token += 1
        token = str(self.token)
        self.proc.stdin.write(f"{token}{text}\n".encode())
        self.proc.stdin.flush()
        result = None
        stopped = None
        while result is None or (wait_stopped and stopped is None):
            line = self.line()
            if line.startswith(token + "^"):
                result = line[len(token) :]
                if result.startswith("^error"):
                    raise RuntimeError(f"{text}: {result}")
            elif line.startswith("*stopped") and result is not None:
                stopped = (
                    line  # a *stopped before our result belongs to an earlier command
                )
        return result, stopped

    def attach(self, pid):
        self.command(f"-target-attach {pid}")
        self.attached = True

    def register(self, index):
        result, _ = self.command(f"-data-list-register-values x {index}")
        m = REGISTER_VALUE.search(result)
        if m is None:
            raise RuntimeError(f"register {index}: {result}")
        return int(m.group(1), 16)

    def step(self):
        _, stopped = self.command("-exec-step-instruction", wait_stopped=True)
        m = STOPPED_PC.search(stopped)
        return int(m.group(1), 16) if m else self.register(PC_REGISTER)

    def close(self):
        try:
            if self.attached:
                self.command("-target-detach")
            self.proc.stdin.write(b"-gdb-exit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()


def step_core(client, disassembly, location):
    import helpers.device as dev

    mailbox = MAILBOX_BASE + 4 * CORE_INDEX
    counts = Counter()
    deadline = time.time() + MAX_SECONDS
    start = time.time()
    finished = False
    expected_pc = None
    state["stage"] = "register"
    pc = client.register(PC_REGISTER)
    previous_pc = None
    while True:
        if expected_pc is not None and pc != expected_pc:
            counts["flow_errors"] += 1
            if pc == previous_pc:
                counts["flow_same_pc"] += 1
            elif pc > expected_pc:
                counts["flow_forward"] += 1
            else:
                counts["flow_backward"] += 1
        expected_pc = None
        previous_pc = pc
        instruction = disassembly.get(pc)
        if instruction is None:
            counts["unknown_pc"] += 1
        else:
            mnemonic, operands, size = instruction
            if not CONTROL_FLOW.match(mnemonic):
                expected_pc = pc + size
            if mnemonic.startswith("tt"):
                counts["tti"] += 1
            elif mnemonic.startswith("sfp"):
                counts["sfp"] += 1
            elif mnemonic in STORES:
                m = STORE_OPERANDS.match(operands)
                if m is None or m.group(3) not in REGISTER_INDEX:
                    counts["unparsed_store"] += 1
                else:
                    state["stage"] = "register"
                    base = client.register(REGISTER_INDEX[m.group(3)])
                    address = (base + int(m.group(2) or "0", 0)) & M32
                    if INSTRN_BUFFER[0] <= address < INSTRN_BUFFER[1]:
                        counts["tt"] += 1
                    elif address == mailbox:
                        finished = True
                    else:
                        counts["other_store"] += 1
        state["stage"] = "step"
        pc = client.step()
        counts["steps"] += 1
        if finished:
            counts["completed"] = 1
            break
        if counts["steps"] % 4096 == 0:
            if dev.read_word_from_device(location, mailbox, 0) == KERNEL_COMPLETE:
                counts["completed_by_poll"] = 1
                break
            if time.time() > deadline:
                counts["time_limit"] = 1
                break
    return counts, time.time() - start


def install():
    import helpers.device_io as dev_io
    import helpers.test_config as tc
    from helpers.llk_params import BriscCmd

    if NO_ASSERTS:
        from ttexalens.hardware.baby_risc_debug import BabyRiscDebugHardware

        BabyRiscDebugHardware.ENABLE_ASSERTS = False

    from ttexalens.hardware.blackhole.baby_risc_debug import BlackholeBabyRiscDebug

    original_step = BlackholeBabyRiscDebug.step
    original_halt = BlackholeBabyRiscDebug.halt

    def step(self):
        if state.get("stepping"):
            state["server_steps"] = state.get("server_steps", 0) + 1
        original_step(self)
        if SETTLE and state.get("stepping"):
            polls = 0
            while not self.debug_hardware.is_halted():
                polls += 1
                if polls > 100000:
                    state["settle_timeouts"] = state.get("settle_timeouts", 0) + 1
                    break

    def halt(self):
        if state.get("stepping"):
            state["server_halts"] = state.get("server_halts", 0) + 1
        return original_halt(self)

    BlackholeBabyRiscDebug.step = step
    BlackholeBabyRiscDebug.halt = halt

    original_run_elf_files = tc.TestConfig.run_elf_files
    original_wait = tc.TestConfig.wait_for_tensix_operations_finished
    original_commit = tc.commit_brisc_command

    def run_elf_files(self):
        state["config"] = self
        return original_run_elf_files(self)

    def commit_brisc_command(location="0,0", command=BriscCmd.IDLE_STATE, timeout=1):
        if command == BriscCmd.START_TRISCS and state["config"] is not None:
            entry = elf_entry(state["config"].temp_elfs[CORE_INDEX])
            original_word = dev_io.read_from_device(location, entry, num_bytes=4)
            dev_io.write_words_to_device(location, entry, [JUMP_TO_SELF])
            state["risc"] = (risc_debug(location), entry, original_word)
            state["armed"] = True
        return original_commit(location, command, timeout)

    def wait_for_tensix_operations_finished(self, timeout=2, poll_callback=None):
        if state["armed"]:
            state["armed"] = False
            rd, entry, original_word = state["risc"]
            hw = rd.debug_hardware
            location = tc.TestConfig.TENSIX_LOCATION
            rd.halt()
            record = {"spin_pc_is_entry": rd.read_gpr(PC_REGISTER) == entry}
            dev_io.write_to_device(location, entry, original_word)
            rd.invalidate_instruction_cache()
            hw.flush(entry)
            if not hw.is_halted():
                rd.halt()
            record["start_pc_is_entry"] = rd.read_gpr(PC_REGISTER) == entry
            if record["spin_pc_is_entry"] and record["start_pc_is_entry"]:
                elf_path = str(self.temp_elfs[CORE_INDEX])
                disassembly = disassemble(
                    elf_path, str(tc.TestConfig.TOOL_PATH / "riscv-tt-elf-objdump")
                )
                server = gdb_server()
                client = None
                try:
                    state["stage"] = "process"
                    pid = gdb_process_id(server, rd, elf_path)
                    if pid is None:
                        raise RuntimeError("core not listed by the GDB server")
                    port = server.server.port
                    state["stage"] = "connect"
                    if CLIENT == "mi":
                        client = MiClient(
                            str(tc.TestConfig.TOOL_PATH / "riscv-tt-elf-gdb"), port
                        )
                    else:
                        client = RspClient(port)
                    record["server_risc_debug_class"] = state.get(
                        "server_risc_debug_class"
                    )
                    state["stage"] = "attach"
                    client.attach(pid)
                    state.update(
                        stepping=True,
                        server_halts=0,
                        settle_timeouts=0,
                        server_steps=0,
                        stale_stop_replies=0,
                    )
                    counts, seconds = step_core(client, disassembly, location)
                    state["stepping"] = False
                    record.update(counts)
                    record["server_halts"] = state["server_halts"]
                    record["stale_stop_replies"] = state["stale_stop_replies"]
                    record["steps_not_sent"] = counts["steps"] - state["server_steps"]
                    record["settle_timeouts"] = state["settle_timeouts"]
                    record["seconds"] = round(seconds, 3)
                except Exception as e:
                    state["stepping"] = False
                    record["client_error_type"] = type(e).__name__
                    record["client_error_stage"] = state["stage"]
                    record["client_error"] = str(e)[:300]
                    if CLIENT == "mi":
                        record["mi_trace"] = list(state.get("mi_trace", []))
                finally:
                    if client is not None:
                        try:
                            state["stage"] = "detach"
                            client.close()
                        except Exception as e:
                            record.setdefault("client_error_type", type(e).__name__)
                            record.setdefault("client_error_stage", state["stage"])
            if rd.is_halted():
                rd.cont()
            write_record(record)
        return original_wait(self, timeout, poll_callback)

    tc.TestConfig.run_elf_files = run_elf_files
    tc.TestConfig.wait_for_tensix_operations_finished = (
        wait_for_tensix_operations_finished
    )
    tc.commit_brisc_command = commit_brisc_command


def pytest_configure(config):
    if not OUT or not STATUS:
        raise RuntimeError("dyn_gdb_plugin: set DYN_GDB_OUT and DYN_GDB_STATUS")


def pytest_runtest_setup(item):
    state["nodeid"] = item.nodeid
    if not state.get("installed"):
        install()
        state["installed"] = True
