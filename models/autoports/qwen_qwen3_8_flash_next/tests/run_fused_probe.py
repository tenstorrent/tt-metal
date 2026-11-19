# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run an unchanged functional acceptance harness with an explicit fused factory.

Selection happens before importing the harness, including harnesses with local
factory imports. The selection sidecar records actual constructed component types.
No functional model is substituted when a fused candidate is requested.
"""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

from models.autoports.qwen_qwen3_8_flash_next.tt import functional_decoder
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedDecoder, FusedSparseMoe

POLICIES = {
    "best_v2": dict(
        pack_shared=True,
        fuse_reduction=True,
        pack_gdn=True,
        fuse_qsa_gate=True,
        fuse_qsa_concat=True,
        fuse_precise_reductions=True,
    ),
    "best_rope": dict(
        pack_shared=True,
        fuse_reduction=True,
        pack_gdn=True,
        fuse_qsa_gate=True,
        fuse_qsa_concat=True,
        fuse_qsa_rope=True,
    ),
    "best_ple": dict(pack_shared=True, fuse_reduction=True, pack_gdn=True, fuse_ple=True, merge_norm_rhs=True),
    "best_v1": dict(pack_shared=True, fuse_reduction=True, pack_gdn=True, fuse_qsa_gate=True, fuse_qsa_concat=True),
    "baseline": None,
    "activation": {},
    "norm": dict(fuse_expert_activation=False, fuse_shared_activation=False, fuse_residual_norm=True),
    "qsa_gate": dict(fuse_qsa_gate=True),
    "qsa_concat": dict(fuse_qsa_concat=True),
    "qsa_both": dict(fuse_qsa_gate=True, fuse_qsa_concat=True),
}
POLICIES.update(
    {
        "best_conv": {**POLICIES["best_v2"], "fuse_gdn_conv": True},
        "best_layout": {**POLICIES["best_v2"], "fuse_gdn_layout": True},
        "best_norm": {**POLICIES["best_v2"], "merge_norm_rhs": True},
        "best_ple_v2": {**POLICIES["best_v2"], "fuse_ple": True},
        "best_qsa_pack": {**POLICIES["best_v2"], "pack_qsa": True},
        "best_rope_v2": {**POLICIES["best_v2"], "fuse_qsa_rope": True},
    }
)

# Preserve experimental policies when the delivered constructor default changes.
EXPERIMENT_DEFAULTS = dict(
    fuse_expert_activation=True,
    fuse_shared_activation=True,
    fuse_residual_norm=False,
    fuse_qsa_gate=False,
    fuse_qsa_concat=False,
    pack_shared=False,
    pack_experts=False,
    fuse_reduction=False,
    pack_gdn=False,
    pack_qsa=False,
    split_qsa_heads=False,
    merge_norm=False,
    merge_norm_rhs=False,
    fuse_ple=False,
    fuse_qsa_rope=False,
    decode_qsa_rope=False,
    fuse_precise_reductions=False,
    fuse_gdn_conv=False,
    fuse_gdn_conv_exact=False,
    fuse_gdn_layout=False,
    projection_decode_group=1,
    preserve_qkv_tree=False,
    fuse_indexer_valid_rows=False,
    fuse_indexer_row_compute=False,
    fuse_indexer_copy_init=False,
    fuse_indexer_score_sum=False,
    merge_indexer_score_heads=False,
    fuse_qsa_selector=False,
)
POLICIES.update(
    {f"best_group{n}": {**POLICIES["best_v2"], "projection_decode_group": n} for n in (2, 4, 8, 16, 32, 64)}
)
POLICIES["best_group8_conv"] = {**POLICIES["best_group8"], "fuse_gdn_conv": True}
POLICIES.update(
    {
        "best_group64_conv": {**POLICIES["best_group64"], "fuse_gdn_conv": True},
        "best_group64_conv_layout": {**POLICIES["best_group64"], "fuse_gdn_conv": True, "fuse_gdn_layout": True},
        "best_group64_layout": {**POLICIES["best_group64"], "fuse_gdn_layout": True},
        "best_group64_norm": {**POLICIES["best_group64"], "merge_norm_rhs": True},
        "best_group64_qsa_pack": {**POLICIES["best_group64"], "pack_qsa": True},
        "best_group64_rope": {**POLICIES["best_group64"], "fuse_qsa_rope": True},
    }
)


POLICIES = {name: None if policy is None else {**EXPERIMENT_DEFAULTS, **policy} for name, policy in POLICIES.items()}
POLICIES["final"] = {}
POLICIES.update({f"qkv_tree_group{n}": dict(qkv_decode_group=n) for n in (1, 2, 4, 8, 16, 32, 64)})
POLICIES["best_valid_rows"] = dict(fuse_indexer_valid_rows=True)
POLICIES["best_valid_compute"] = dict(fuse_indexer_row_compute=True)
POLICIES["best_copy_once"] = dict(fuse_indexer_copy_init=True)
POLICIES["best_copy_once_rows"] = dict(fuse_indexer_copy_init=True, fuse_indexer_row_compute=True)
POLICIES["best_score_sum"] = dict(fuse_indexer_valid_rows=True, fuse_indexer_score_sum=True)
POLICIES["best_score_merge"] = dict(fuse_indexer_valid_rows=True, merge_indexer_score_heads=True)
POLICIES["best_score_both"] = dict(
    fuse_indexer_valid_rows=True, fuse_indexer_score_sum=True, merge_indexer_score_heads=True
)
POLICIES["best_selector"] = {**POLICIES["best_score_both"], "fuse_qsa_selector": True}
for name in (
    "best_valid_rows",
    "best_valid_compute",
    "best_copy_once",
    "best_copy_once_rows",
    "best_score_sum",
    "best_score_merge",
    "best_score_both",
    "best_selector",
):
    POLICIES[name] = {
        "fuse_indexer_valid_rows": False,
        "fuse_indexer_row_compute": False,
        "fuse_indexer_copy_init": False,
        "fuse_indexer_score_sum": False,
        "merge_indexer_score_heads": False,
        "fuse_qsa_selector": False,
        **POLICIES[name],
    }


POLICIES["best_residual_group1"] = dict(POLICIES["best_selector"], residual_decode_group=1)

POLICIES["best_all_group1"] = dict(POLICIES["best_selector"], projection_decode_group=1)
POLICIES["best_compact_moe"] = dict(POLICIES["best_selector"], compact_experts=True)
POLICIES["best_compact_io"] = dict(POLICIES["best_selector"], fuse_gdn_io=True)
POLICIES["best_compact_both"] = dict(POLICIES["best_selector"], compact_experts=True, fuse_gdn_io=True)


POLICIES["best_compact_ple"] = dict(POLICIES["best_compact_both"], residual_decode_group=1)

POLICIES["best_direct_setup"] = dict(POLICIES["best_compact_both"], direct_packed_weights=True)
POLICIES["best_unpack"] = dict(POLICIES["best_compact_both"], fuse_projection_cast=True)
POLICIES["best_direct_unpack"] = dict(
    POLICIES["best_compact_both"], direct_packed_weights=True, fuse_projection_cast=True
)

for cores in (10, 20, 40):
    POLICIES[f"best_packed{cores}"] = dict(
        POLICIES["best_direct_unpack"], pack_experts=True, packed_expert_gate_cores=(10, cores // 10)
    )

POLICIES["best_packed_silu"] = dict(POLICIES["best_packed40"], fuse_packed_activation=True)

POLICIES["best_packed_silu20"] = dict(POLICIES["best_packed_silu"], packed_expert_gate_cores=(10, 2))
POLICIES["best_packed_silu10"] = dict(POLICIES["best_packed_silu"], packed_expert_gate_cores=(10, 1))
POLICIES["best_packed_silu_gr32"] = dict(POLICIES["best_packed_silu"], residual_decode_group=32)
POLICIES["best_packed_silu_gr1"] = dict(POLICIES["best_packed_silu"], residual_decode_group=1)

POLICIES["best_packed_early"] = dict(POLICIES["best_packed_silu"], early_packed_activation=True)


POLICIES["best_exact_qkv"] = dict(POLICIES["best_packed_early"], fuse_exact_qkv=True)


POLICIES["best_exact_qkv_cleanup"] = dict(POLICIES["best_exact_qkv"], fuse_moe_cleanup=True)


for limit in (2, 3, 4, 5, 6):
    POLICIES[f"best_tail{limit}"] = dict(POLICIES["best_exact_qkv_cleanup"], compact_prefill_limit=limit)

POLICIES["best_writer_v5"] = dict(POLICIES["best_exact_qkv_cleanup"], fuse_residual_write=True)
POLICIES["best_writer_tail2"] = dict(POLICIES["best_tail2"], fuse_residual_write=True)

LEGACY_FINAL_DEFAULTS = {
    "fuse_residual_write": False,
    "compact_prefill_limit": 1,
    "fuse_expert_activation": True,
    "fuse_shared_activation": True,
    "fuse_residual_norm": False,
    "fuse_qsa_gate": True,
    "fuse_qsa_concat": True,
    "pack_shared": True,
    "pack_experts": False,
    "packed_expert_gate_cores": None,
    "fuse_packed_activation": False,
    "early_packed_activation": False,
    "fuse_reduction": True,
    "fuse_moe_cleanup": False,
    "pack_gdn": True,
    "pack_qsa": True,
    "split_qsa_heads": False,
    "merge_norm": False,
    "merge_norm_rhs": False,
    "fuse_ple": False,
    "fuse_qsa_rope": True,
    "decode_qsa_rope": False,
    "fuse_precise_reductions": True,
    "fuse_gdn_conv": False,
    "fuse_gdn_conv_exact": True,
    "fuse_gdn_layout": True,
    "fuse_gdn_io": False,
    "compact_experts": False,
    "fuse_projection_cast": False,
    "direct_packed_weights": False,
    "projection_decode_group": 64,
    "residual_decode_group": None,
    "preserve_qkv_tree": True,
    "fuse_exact_qkv": False,
    "qkv_decode_group": None,
    "fuse_indexer_valid_rows": True,
    "fuse_indexer_row_compute": False,
    "fuse_indexer_copy_init": False,
    "fuse_indexer_score_sum": True,
    "merge_indexer_score_heads": True,
    "fuse_qsa_selector": True,
}
POLICIES = {
    name: policy if policy is None or name == "final" else {**LEGACY_FINAL_DEFAULTS, **policy}
    for name, policy in POLICIES.items()
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", required=True)
    p.add_argument("--variant", choices=tuple(POLICIES), required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    a = p.parse_args()
    policy = POLICIES[a.variant]
    original = functional_decoder.FunctionalDecoder
    records = []

    class Selected:
        @classmethod
        def from_state_dict(cls, *args, **kwargs):
            model = (
                original.from_state_dict(*args, **kwargs)
                if policy is None
                else FusedDecoder.from_state_dict(*args, **kwargs, **policy)
            )
            if policy is not None:
                assert isinstance(model, FusedDecoder) and isinstance(model.mlp, FusedSparseMoe)
            records.append(
                dict(
                    decoder=type(model).__name__,
                    moe=type(model.mlp).__name__,
                    qkv_projection=type(getattr(model.attn, "qkv", None)).__name__,
                    moe_cleanup=getattr(model.mlp, "fuse_moe_cleanup", False),
                    residual_writer=type(model.attn_mix).__name__,
                    compact_prefill_limit=getattr(model.mlp, "compact_prefill_limit", 1),
                    compact_experts=getattr(model.mlp, "compact_experts", False),
                    packed_experts=hasattr(model.mlp, "expert_packed"),
                    packed_activation=getattr(model.mlp, "fuse_packed_activation", False),
                    early_activation=getattr(model.mlp, "early_packed_activation", False),
                    gate_up_cores=list(model.mlp.program.decode_gate_up_cores),
                    expert_dtype=str(
                        (model.mlp.expert_packed if hasattr(model.mlp, "expert_packed") else model.mlp.gate).dtype
                    ),
                    projection_unpack=getattr(model.mlp.precise_router, "widen_in_unpack", False),
                    compact_gdn_io=getattr(model.attn, "compact_recurrence_io", False),
                    attention=type(model.chunked_attn).__name__,
                    residual=type(model.attn_mix).__name__,
                    qsa_scores=type(model.attn).__name__ if model.chunked_attn is not None else None,
                    qsa_projection=(
                        type(model.attn.project_index_qk).__name__ if model.chunked_attn is not None else None
                    ),
                    qsa_selector=getattr(model.chunked_attn, "fuse_selector", False),
                )
            )
            return model

    report = dict(
        variant=a.variant,
        policy=policy,
        probe=a.probe,
        constructed=records,
        completed=False,
        fused_source_sha256=hashlib.sha256(
            Path(sys.modules[FusedDecoder.__module__].__file__).read_bytes()
        ).hexdigest(),
    )
    try:
        with patch.object(functional_decoder, "FunctionalDecoder", Selected):
            module = importlib.import_module("models.autoports.qwen_qwen3_8_flash_next.tests." + a.probe)
            sys.argv = [module.__file__] + (a.arguments[1:] if a.arguments[:1] == ["--"] else a.arguments)
            module.main()
        assert records, "Probe never constructed a decoder"
        report["completed"] = True
    finally:
        a.selection.parent.mkdir(parents=True, exist_ok=True)
        a.selection.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
