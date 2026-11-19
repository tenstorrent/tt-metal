# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bind established correctness/profiler probes to the explicit optimized factory."""

import argparse
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import functional_decoder
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder, OptimizedSparseMoe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe", required=True)
    p.add_argument("--policy", type=Path)
    p.add_argument(
        "--factory",
        choices=("optimized", "fused"),
        default="optimized",
        help="Fused is an explicit diagnostic control only",
    )
    p.add_argument("--expert-dtype", choices=("bfloat16", "bfloat8_b", "bfloat4_b"), default="bfloat8_b")
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--split-matmul-activation", action="store_true")
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    a = p.parse_args()
    policy = json.loads(a.policy.read_text()) if a.policy else {}
    records = []

    def compute_fields(config):
        return {
            key: str(getattr(config, key))
            for key in ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc")
        }

    def projection(value, compute):
        tensor = value if isinstance(value, ttnn.Tensor) else getattr(value, "weight", None)
        if tensor is None and hasattr(value, "weights"):
            tensor = value.weights[0]
        return dict(
            implementation=type(value).__name__,
            dtype=str(tensor.dtype) if tensor is not None else None,
            memory=str(tensor.memory_config()) if tensor is not None else None,
            compute=compute_fields(getattr(value, "compute", compute)),
        )

    class Selected:
        @classmethod
        def from_state_dict(cls, *args, **kwargs):
            kwargs.update(policy)
            kwargs["expert_dtype"] = getattr(ttnn, a.expert_dtype)
            factory = OptimizedDecoder if a.factory == "optimized" else FusedDecoder
            model = factory.from_state_dict(*args, **kwargs)
            assert type(model) is factory
            if a.factory == "optimized":
                assert type(model.mlp) is OptimizedSparseMoe
            if a.split_matmul_activation:
                assert a.factory == "optimized" and not hasattr(model.mlp, "expert_packed")
                model.mlp.matmul_expert_activation = True
            dense = {}
            for owner, names in (
                (model.mlp, ("shared_packed", "shared_down")),
                (model.attn, ("zab", "out")),
                (model.chunked_attn, ("packed_qkv",)),
                (model.ple, ("key", "value")),
            ):
                for name in names:
                    if owner is not None and hasattr(owner, name):
                        dense[name] = projection(getattr(owner, name), model.mlp.compute)
            records.append(
                dict(
                    decoder=type(model).__name__,
                    moe=type(model.mlp).__name__,
                    expert_dtype=str(
                        (model.mlp.expert_packed if hasattr(model.mlp, "expert_packed") else model.mlp.gate).dtype
                    ),
                    expert_compute=str(getattr(model.mlp, "expert_compute", model.mlp.compute)),
                    expert_compute_fields=compute_fields(getattr(model.mlp, "expert_compute", model.mlp.compute)),
                    expert_down_dtype=str(model.mlp.down.dtype),
                    dense=dense,
                    program=vars(model.mlp.program),
                    policy=policy,
                    split_matmul_activation=a.split_matmul_activation,
                )
            )
            return model

    inner = list(a.arguments[1:] if a.arguments[:1] == ["--"] else a.arguments)
    requested = list(inner)
    # These existing probes expose the same base expert dtype in their reports.
    # Keep their metadata aligned with the explicit outer factory policy.
    if (
        a.probe
        in {
            "probe_batch",
            "probe_public_continuation",
            "probe_ple_decoder",
            "probe_ple_full_context",
            "probe_gdn_full_context",
            "profile_fused_ops",
        }
        and a.expert_dtype != "bfloat4_b"
    ):
        if "--expert-dtype" in inner:
            inner[inner.index("--expert-dtype") + 1] = a.expert_dtype
        else:
            inner.extend(("--expert-dtype", a.expert_dtype))
    report = dict(
        probe=a.probe,
        factory=a.factory,
        requested_inner_arguments=requested,
        effective_inner_arguments=inner,
        constructed=records,
        completed=False,
    )
    try:
        with patch.object(functional_decoder, "FunctionalDecoder", Selected):
            module = importlib.import_module("models.autoports.qwen_qwen3_8_flash_next.tests." + a.probe)
            sys.argv = [module.__file__] + inner
            module.main()
        assert records, "Probe did not construct the optimized path"
        report["completed"] = True
    finally:
        a.selection.parent.mkdir(parents=True, exist_ok=True)
        a.selection.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
