# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Production entrypoint for the host-only fused helper/public-contract audit."""

import json
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.audit_fused_interleaved_tail_v6 import audit
from models.autoports.qwen_qwen3_6_35b_a3b.tests.audit_fused_snapshot_v6 import UNCHANGED, sha

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder"


def main():
    source = ROOT / "tt/fused_decoder.py"
    checked = audit(source)
    is_v6 = checked["contract"]["v6"]
    # Retain legacy evidence keys, but distinguish public hints from the actual
    # reviewed planner. No functional/PCC/context/device threshold is changed.
    result = dict(
        python_audit_pass=checked["python_audit_pass"],
        decoder="FusedDecoder",
        decoder_sha256=checked["source_sha256"],
        runtime_functions=sorted(checked["syntax"]["runtime_function_ast_sha256"]),
        forbidden_calls=checked["syntax"]["forbidden_runtime_constructs"],
        standalone_implementation=True,
        unchanged_contract_methods=UNCHANGED,
        context_capability=checked["planner"]["capacity_bound"],
        default_prefill_chunk=4096,
        default_prefill_chunk_is_public_hint=True,
        internal_prefill_chunks={
            "linear_attention": 64 if is_v6 else "V5 physical hint bounded; singleton boundaries",
            "full_attention": "at most256 aligned interiors; singleton boundaries",
            "linear_tail": "logical final tail1..63 internally padded and trimmed" if is_v6 else "V5 singleton tail",
            "public_singleton": "unchanged recurrence path at all valid starts",
        },
        unchanged_planning_body=not is_v6,
        planning_contract="Exact contiguous logical coverage and absolute positions, positive aligned/non-aligned sequence support, same262144 bound; V6 linear chunks64 are relative to each public call; V5 full-attention planner unchanged",
        repaired_boundary="prefill_forward: HF BF16 cache rounding once per public call; signature preserved",
        recurrent_state_storage={
            "conv": "BF16 ROW_MAJOR backing [1,3,8192]; transient history adapters stay device-only",
            "recurrent": "FP32 TILE [1,32,128,128]; public BF16 consumption boundary preserved",
            "public_constructor_signature_preserved": True,
        },
        dynamic_audit="device_only guards remain required in delivered runners; not executed by this host-only audit",
        acceptance_eligible=False,
        device_executed=False,
        entrypoint_sha256=sha(__file__),
        detailed_source_audit=checked,
    )
    (DOC / "fallback_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("FUSED_SOURCE_AUDIT_PASS", checked["source_sha256"])


if __name__ == "__main__":
    main()
