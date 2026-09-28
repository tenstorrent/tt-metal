# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""components.yaml: every step of every block type mapped to a TTNN implementation.

    components:
      - block_type: moe                 # a spec block type, or "model" for embedding / final norm / lm head
        step: attention                 # a step name of that block type's graph (or embed / final_norm / lm_head)
        tag: NATIVE | COMPOSED | CPU | OPGEN   # one TTNN op | several TTNN ops | stays on the host |
                                        # TTNN has no proper op: the implement task writes an op request for op-gen
                                        # (plan/op_request.py) and the step stays on the CPU bridge (F46)
        ttnn: "ttnn.transformer.chunked_scaled_dot_product_attention"
        reuse: models/demos/gemma4/tt/attention.py     # the repo code this starts from, or "none"
        searched: [repo_map: attention, grep chunked_scaled_dot_product_attention]   # required for COMPOSED / CPU / OPGEN
        op: indexer_key_pool             # OPGEN only (optional): the op name to request
        notes: ...

Findings (what the bring-up learned; the dashboard shows them) go to findings.yaml, so that appending one does not
invalidate the approved plan:
    findings: [{id, task, kind: PERF | ACCURACY | API | INFRA | CONTRACT, title, detail}]
"""

from __future__ import annotations

import yaml

TAGS = ("NATIVE", "COMPOSED", "CPU", "OPGEN")
NO_OP_TAGS = ("CPU", "OPGEN")  # no TTNN op to name
MODEL_STEPS = ("embed", "final_norm", "lm_head")


def load(spec) -> dict:
    p = spec.bringup_dir / "components.yaml"
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def validate(spec, ref) -> list[str]:
    """Schema errors, plus: every graph step of every block type is mapped exactly once."""
    data = load(spec)
    comps = data.get("components") or []
    errs, seen = [], set()
    graphs = {bt: [s.name for s in ref.block_graph(spec.representative_layer(bt))] for bt in spec.data["block_types"]}
    for c in comps:
        key = (c.get("block_type"), c.get("step"))
        if key in seen:
            errs.append(f"{key} mapped twice")
        seen.add(key)
        if c.get("tag") not in TAGS:
            errs.append(f"{key}: tag must be one of {TAGS}")
        if not c.get("ttnn") and c.get("tag") not in NO_OP_TAGS:
            errs.append(f"{key}: no ttnn op named")
        if "reuse" not in c:
            errs.append(f"{key}: say which repo code it starts from ('reuse', or 'none')")
        if c.get("tag") in ("COMPOSED", "CPU", "OPGEN") and not c.get("searched"):
            errs.append(f"{key}: a {c.get('tag')} tag needs 'searched' (what was looked for in the repo map and code)")
        bt, st = key
        if c.get("tag") == "OPGEN" and bt == "model":
            errs.append(f"{key}: OPGEN is for block steps (a component task defers); model-level steps cannot")
        if bt == "model":
            if st not in MODEL_STEPS:
                errs.append(f"{key}: model-level steps are {MODEL_STEPS}")
        elif bt not in graphs:
            errs.append(f"{key}: unknown block type")
        elif st not in graphs[bt]:
            errs.append(f"{key}: not a step of {bt} ({graphs[bt]})")
    for bt, steps in graphs.items():
        for st in steps:
            if (bt, st) not in seen:
                errs.append(f"({bt}, {st}) has no component entry")
    for st in MODEL_STEPS:
        if ("model", st) not in seen:
            errs.append(f"(model, {st}) has no component entry")
    return errs
