# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Write the winners of parse_matmul_chunk_sweep result files into tt/glm_chunk_matmul_configs.TUNED.

usage: python -m models.demos.deepseek_v3_d_p.utils.gen_glm_chunk_matmul_configs <result.json> [<result.json> ...]

Each result file is one (layout, chunk) sweep. Rewrites only the block between the BEGIN/END GENERATED markers.
"""

import json
import pprint
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "tt" / "glm_chunk_matmul_configs.py"
TABLES = {"tp", "bax", "gate", "shared"}


def entries(result_paths):
    tuned = {}
    for p in result_paths:
        for name, r in json.load(open(p)).items():
            s, best, default = r["spec"], r["best"], r["default"]
            if best is None or s["table"] not in TABLES:
                continue
            # The shared expert's gate/up config is computed by the model (tall out_subblock_h = per_core_M), a
            # tiling the sweep's candidate set did not contain -- so the swept "best" is not proven faster
            # than what the model already runs. Keep the model's. (The down projection's computed config WAS
            # among the candidates, so its winner is.)
            if name == "moe.shared_gate_up":
                continue
            rows = s["key_m"] if s["table"] in ("tp", "bax") else s["m"]
            tuned[(s["table"], name, rows)] = dict(
                desc={**best["cfg"], "grid": list(s["grid"])},
                act_mem=best["act_mem"],
                out_mem=best["out_mem"],
                out_dtype=best["out_dtype"],
                k=s["k"],
                n=s["n"],
                us=round(best["us"], 1),
                cores=best["cores"],
                default_us=round(default["us"], 1) if default and default["us"] else None,
            )
    return tuned


def main(paths):
    tuned = entries(paths)
    body = "TUNED: dict = " + pprint.pformat(dict(sorted(tuned.items())), width=118, sort_dicts=True) + "\n"
    src = TARGET.read_text()
    a = src.index("# BEGIN GENERATED")
    a = src.index("\n", a) + 1
    b = src.index("# END GENERATED")
    TARGET.write_text(src[:a] + body + src[b:])
    print(f"wrote {len(tuned)} entries to {TARGET}")


if __name__ == "__main__":
    main(sys.argv[1:])
