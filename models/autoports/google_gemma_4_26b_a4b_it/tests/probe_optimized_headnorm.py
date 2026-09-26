# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Actual-input sliding head-norm control using the existing runtime switch."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--headnorm-mode", choices=("native", "precise"), default="native")
    args, rest = parser.parse_known_args()
    output = Path(rest[rest.index("--output") + 1])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite head-norm evidence: {output}")
    factory = OptimizedDecoder.from_state_dict.__func__
    runtime = {}

    def create(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        attention = decoder.layer.self_attn
        if not attention.source.config.is_sliding:
            raise ValueError("This control requires a sliding attention layer")
        if not decoder.fuse_norm:
            raise ValueError("Native head normalization requires the existing fused norm backend")
        if getattr(attention.normalize, "__self__", None) is not decoder:
            raise ValueError("Attention must retain the decoder's bound normalize method")
        original = decoder.precise_heads
        decoder.precise_heads = args.headnorm_mode == "precise"
        runtime.update(
            mode=args.headnorm_mode,
            original_precise_heads=original,
            precise_heads=decoder.precise_heads,
            scope="Q/K/V head normalization in both prefill and decode",
            site="FusedDecoder.normalize via OptimizedDecoder.normalize",
            head_dim=attention.source.config.head_dim,
            hidden_size=decoder.config.hidden_size,
            external_q_weight_dtype=str(attention.q_weight.dtype),
            external_k_weight_dtype=str(attention.k_weight.dtype),
            compute=str(attention.compute),
            sharded_norm_site=decoder.sharded_norm_site,
            sharded_norms=decoder.use_sharded_norms,
            native_input_dtype="float32",
            learned_gamma="unchanged external multiply; not passed into rms_norm",
            changed_setup_attributes=["precise_heads"],
            runtime_source_modified=False,
        )
        decoder.precision_policy["head_norm_control"] = runtime
        return decoder

    previous_argv = sys.argv
    sys.argv = [sys.argv[0], *rest]
    failure = None
    try:
        with patch.object(OptimizedDecoder, "from_state_dict", classmethod(create)):
            run_optimized_decoder.main()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        sys.argv = previous_argv
        report = json.loads(output.read_text()) if output.exists() else {"decoder": "optimized"}
        report["head_norm_probe"] = vars(args)
        report["head_norm_runtime"] = runtime
        report["head_norm_probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["runtime_sha256"] = hashlib.sha256(
            Path(__file__).parents[1].joinpath("tt/optimized_decoder.py").read_bytes()
        ).hexdigest()
        if failure is not None:
            report["head_norm_error"] = failure
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
