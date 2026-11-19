# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only causal substitutions using the saved same-operand device graph."""

import hashlib
import json
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_state_probe import code_compare

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"


def main():
    torch.set_num_threads(4)
    base = E / "v6_chunk64_sfpu_fma_compensated_device_v2.json"
    meta = json.loads(base.read_text())
    cpu_path, dev_path = Path(meta["cpu_matrix_snapshot"]), Path(meta["device_matrix_snapshot"])
    cpu = torch.load(cpu_path, weights_only=False)
    dev = torch.load(dev_path, weights_only=False)[128]["intermediates"]
    case = cpu["cases"][128]
    operands = case["operands"]
    rows = []
    controls = [
        (),
        ("cumulative_g",),
        ("decay_mask",),
        ("weighted_k",),
        ("cumulative_g", "decay_mask", "weighted_k"),
        ("A_initial",),
        ("A_final",),
        ("transformed_v",),
        ("transformed_k",),
        ("transformed_v", "transformed_k", "weighted_k"),
        ("v_new", "weighted_k"),
    ]
    for keys in controls:
        state = cpu["initial"].clone()
        for start in (0, 64):
            prefix = f"chunk{start // 64}/"

            def pick(name, value):
                return dev[prefix + name] if name in keys else value

            k, v, g, beta = [operands[name][..., start : start + 64, :] for name in ("key", "value", "g", "beta")]
            kb, vb = k * beta, v * beta
            cumulative = pick("cumulative_g", g.cumsum(-2))
            decay = pick("decay_mask", (cumulative - cumulative.transpose(-1, -2)).tril().exp().tril())
            strict = torch.ones(64, 64, dtype=torch.bool).tril(-1)
            a = pick("A_initial", -((kb @ k.transpose(-1, -2)) * decay).masked_fill(~strict, 0))
            for i in range(1, 64):
                row, sub = a[..., i, :i].clone(), a[..., :i, :i].clone()
                a[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
            a = pick("A_final", a + torch.eye(64))
            tv = pick("transformed_v", a @ vb)
            tk = pick("transformed_k", a @ (kb * cumulative.exp()))
            vn = pick("v_new", tv - tk @ state)
            last = cumulative[..., 63:64, :]
            wk = pick("weighted_k", k * (last - cumulative).exp())
            state = state * last.exp() + wk.transpose(-1, -2) @ vn
        rows.append(dict(device_substitutions=list(keys), final=code_compare(case["canonical"], state)))
    result = dict(
        classification="CPU diagnostic with saved device intermediates; not acceptance",
        controls=rows,
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (base, cpu_path, dev_path)},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    out = E / "v6_chunk64_intervention_cpu.json"
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(row["device_substitutions"], row["final"]["bf16_code_differences"])


if __name__ == "__main__":
    main()
