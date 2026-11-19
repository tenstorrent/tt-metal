# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read-only installed MKL dispatch audit and normal-range scalar arithmetic proof.

The constants come from the installed math library, never from model outputs.
No binary patch, environment dispatch override, TTNN import, or hardware call.
"""
import ctypes
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_state_probe import code_compare

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"


def main():
    torch.set_num_threads(4)
    path = Path(torch.__file__).parent / "lib/libtorch_cpu.so"
    lib = ctypes.CDLL(str(path))
    lib.mkl_vml_serv_cpu_detect.restype = ctypes.c_int
    lib.mkl_vml_kernel_GetTTableIndex.argtypes = [ctypes.c_int]
    lib.mkl_vml_kernel_GetTTableIndex.restype = ctypes.c_int
    lib.vmlGetMode.restype = ctypes.c_uint
    cpu = lib.mkl_vml_serv_cpu_detect()
    index = lib.mkl_vml_kernel_GetTTableIndex(cpu)
    table = (ctypes.c_void_p * 16).in_dll(lib, "mkl_vml_kernel_sExp_ttab")
    target = ctypes.cast(lib.mkl_vml_kernel_sExp_EXHAynn, ctypes.c_void_p).value
    assert table[index] == target
    base = ctypes.cast(lib.vmsExp, ctypes.c_void_p).value - 0x9160A80
    assert target - base == 0xB9BD890

    def bits_at(address, count):
        return list(struct.unpack(f"{count}I", ctypes.string_at(base + address, count * 4)))

    constants = {
        name: bits_at(address, 1)[0]
        for name, address in (
            ("scale", 0x124D1D10),
            ("magic", 0x124D0C10),
            ("negative_magic", 0x124D0C20),
            ("table_mask", 0x124D1D20),
            ("ln2_hi_by128", 0x124D1D30),
            ("negative_ln2_lo_by128", 0x124D1D40),
            ("half", 0x124C1CB0),
            ("one_bits", 0x124C2770),
            ("exponent_mask", 0x124C2840),
            ("safe_abs", 0x124D1C50),
        )
    }
    table_bits = bits_at(0x13181980 + 0x300, 256)
    table_values = torch.tensor(table_bits, dtype=torch.int64).to(torch.int32).view(torch.float32).reshape(128, 2)
    scalar = {name: struct.unpack("f", struct.pack("I", value))[0] for name, value in constants.items()}

    def math_exp(x, rare="flush"):
        # All operators are separately rounded FP32, matching SSE mulps/addps/subps.
        z = x * scalar["scale"]
        z = z + scalar["magic"]
        raw = z.contiguous().view(torch.int32).long()
        ti = ((raw << 3) & constants["table_mask"]) // 8
        pair = table_values[ti]
        n = z + scalar["negative_magic"]
        r = x - n * scalar["ln2_hi_by128"]
        r = r + n * scalar["negative_ln2_lo_by128"]
        p = r * r
        p = p * scalar["half"]
        p = p + r
        result = p * pair[..., 0]
        result = result + pair[..., 1]
        result = result + pair[..., 0]
        exponent = (
            (((raw << 16) + constants["one_bits"]) & constants["exponent_mask"]).to(torch.int32).view(torch.float32)
        )
        result = result * exponent
        safe = x.abs() <= scalar["safe_abs"]
        if rare == "torch":
            return torch.where(safe, result, x.exp())
        return torch.where(safe, result, torch.zeros_like(result))

    meta_path = E / "v6_chunk64_ordered_state_device.json"
    meta = json.loads(meta_path.read_text())
    cpu_path = Path(meta["cpu_matrix_snapshot"])
    payload = torch.load(cpu_path, weights_only=False)
    case = payload["cases"][128]
    rows = []
    for chunk in range(2):
        cg = case["matrices"][f"chunk{chunk}/cumulative_g"]
        for name, x in [
            ("cumulative", cg),
            ("lower_difference", (cg - cg.transpose(-1, -2)).tril()),
            ("last_difference", cg[..., -1:, :] - cg),
        ]:
            expected = x.exp()
            pred = math_exp(x)
            safe = x.abs() <= scalar["safe_abs"]
            rows.append(
                dict(
                    chunk=chunk,
                    name=name,
                    safe_count=int(safe.sum()),
                    rare_count=int((~safe).sum()),
                    safe_fp32_code_differences=int(
                        (expected[safe].view(torch.int32) != pred[safe].view(torch.int32)).sum()
                    ),
                )
            )
    generator = torch.Generator().manual_seed(20261113)
    random = -87 * torch.rand(100000, generator=generator)
    random_expected = random.exp()
    random_pred = math_exp(random)
    random_differences = int((random_expected.view(torch.int32) != random_pred.view(torch.int32)).sum())
    states = []
    for rare in ("flush", "torch"):
        state = payload["initial"].clone()
        for start in (0, 64):
            key, value, g, beta = [
                case["operands"][name][..., start : start + 64, :] for name in ("key", "value", "g", "beta")
            ]
            kb, vb = key * beta, value * beta
            cg = g.cumsum(-2)
            decay = math_exp((cg - cg.transpose(-1, -2)).tril(), rare).tril()
            strict = torch.ones(64, 64, dtype=torch.bool).tril(-1)
            a = -((kb @ key.transpose(-1, -2)) * decay).masked_fill(~strict, 0)
            for i in range(1, 64):
                row, sub = a[..., i, :i].clone(), a[..., :i, :i].clone()
                a[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
            a = a + torch.eye(64)
            new = a @ vb - (a @ (kb * math_exp(cg, rare))) @ state
            last = cg[..., -1:, :]
            wk = key * math_exp(last - cg, rare)
            state = state * math_exp(last, rare) + wk.transpose(-1, -2) @ new
        states.append(dict(rare_policy=rare, final=code_compare(case["canonical"], state)))
    disassembly = subprocess.check_output(
        ["objdump", "-d", "--start-address=0xb9bd890", "--stop-address=0xb9bdf00", str(path)], text=True
    )
    dis_path = E / "v6_mkl_exp_normal_loop.disasm.txt"
    dis_path.write_text(disassembly)
    result = dict(
        classification="Read-only binary arithmetic diagnosis, no device-equivalence claim",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library=str(path),
        library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cpu_detect=cpu,
        dispatch_index=index,
        vml_mode=hex(lib.vmlGetMode()),
        selected_symbol="mkl_vml_kernel_sExp_EXHAynn",
        constants_bits=constants,
        table_hi_lo_bits=table_bits,
        table_address="vsexp_data+0x300",
        disassembly_sha256=hashlib.sha256(dis_path.read_bytes()).hexdigest(),
        actual_operand_checks=rows,
        random_normal_checks=100000,
        random_normal_fp32_code_differences=random_differences,
        full_state_exp_only_controls=states,
        limits="Normal finite arguments tested. Rare/underflow, positiveoverflow, NaN/Inf and arbitrary input-length tail paths not yet transcribed. Torch rare fallback is diagnostic-only; flush is an explicit candidate with recorded result, not an exact-exp claim.",
    )
    (E / "v6_mkl_exp_dispatch_cpu.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(rows=rows, random_differences=random_differences, states=states), indent=2))


if __name__ == "__main__":
    main()
