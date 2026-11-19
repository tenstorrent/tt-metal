# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Process-local SDPA-decode-only precision discriminator; all gates unchanged."""

import argparse
import functools
import hashlib
import json
import sys
from pathlib import Path

import ttnn

POLICIES = {
    "hifi2_exact_fp32": ("HiFi2", False, True),
    "hifi2_approx_fp32": ("HiFi2", True, True),
    "hifi2_approx_bf16": ("HiFi2", True, False),
    "lofi_exact_fp32": ("LoFi", False, True),
}
CONFIG_FIELDS = (
    "math_fidelity",
    "math_approx_mode",
    "fp32_dest_acc_en",
    "packer_l1_acc",
    "dst_full_sync_en",
    "throttle_level",
)


def describe_config(config):
    return {
        name: (value if isinstance(value, (bool, int, float)) else str(value))
        for name in CONFIG_FIELDS
        if (value := getattr(config, name, None)) is not None
    }


def policy_config(original, policy):
    if policy == "control":
        return original
    assert original is not None, "Current production supplies an explicit SDPA compute config"
    fidelity, approximate, fp32 = POLICIES[policy]
    # Construct a new object: never mutate the shared self.compute used by norms,
    # RoPE, recurrence, or other operations. Preserve all unrelated config fields.
    values = {field: getattr(original, field) for field in CONFIG_FIELDS}
    values.update(
        math_fidelity=getattr(ttnn.MathFidelity, fidelity), math_approx_mode=approximate, fp32_dest_acc_en=fp32
    )
    return type(original)(**values)


def wrap_sdpa(original_operation, policy, observed):
    configs = {}

    @functools.wraps(original_operation)
    def wrapped(*operands, **kwargs):
        original = kwargs.get("compute_kernel_config")
        original_desc = describe_config(original)
        key = json.dumps(original_desc, sort_keys=True)
        if key not in configs:
            configs[key] = policy_config(original, policy)
        selected = configs[key]
        kwargs["compute_kernel_config"] = selected
        shape = tuple(operands[0].shape)
        observation_key = json.dumps([shape, original_desc], sort_keys=True)
        if observation_key not in observed:
            observed[observation_key] = dict(
                query_shape=list(shape),
                query_dtype=str(operands[0].dtype),
                original=original_desc,
                selected=describe_config(selected),
                host_invocations=0,
            )
        observed[observation_key]["host_invocations"] += 1
        return original_operation(*operands, **kwargs)

    return wrapped


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--sdpa-policy", choices=("control", *POLICIES), required=True)
    parser.add_argument("--recorded-batch", action="store_true")
    args, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index("--output") + 1]) if "--output" in remaining else None
    observed, restore_harness = {}, None
    operation = ttnn.transformer.paged_scaled_dot_product_attention_decode
    original_argv = sys.argv[:]
    error = None
    try:
        if args.recorded_batch:
            assert "--family" not in remaining and "--approximate" not in remaining
            from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_recorded_batch as harness
            from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder

            class RecordedCurrentDecoder(MultichipDecoder):
                """Current production methods; recorded harness policy attributes are inert."""

            restore_harness = (harness, "GdnFidelityCandidate", harness.GdnFidelityCandidate)
            harness.GdnFidelityCandidate = RecordedCurrentDecoder
        else:
            from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as harness
        ttnn.transformer.paged_scaled_dot_product_attention_decode = wrap_sdpa(operation, args.sdpa_policy, observed)
        sys.argv = [sys.argv[0], *remaining]
        harness.main()
        assert observed, "SDPA precision candidate did not reach its intended operation"
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        ttnn.transformer.paged_scaled_dot_product_attention_decode = operation
        if restore_harness is not None:
            setattr(*restore_harness)
        sys.argv[:] = original_argv
        if output is not None:
            sidecar = output.with_name(output.stem + "_sdpa_policy.json")
            sidecar.write_text(
                json.dumps(
                    dict(
                        policy=args.sdpa_policy,
                        harness="recorded_batch" if args.recorded_batch else "packed_stack",
                        changed_operation="ttnn.transformer.paged_scaled_dot_product_attention_decode",
                        changed_fields=[] if args.sdpa_policy == "control" else list(CONFIG_FIELDS[:3]),
                        observed=list(observed.values()),
                        error=error,
                        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        note="Host invocation counts exclude device trace replays; prefill SDPA and all other operations unchanged",
                    ),
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
