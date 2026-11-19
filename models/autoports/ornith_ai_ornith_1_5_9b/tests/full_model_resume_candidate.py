# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic controls only; never selected by the production generator."""

import argparse
import inspect
import sys
import textwrap

from models.autoports.ornith_ai_ornith_1_5_9b.tt import multichip_decoder as decoder


def install_canonical_bf8(*, norms, reductions):
    """Keep a long request's BF8 policy across its prefix/suffix fragments.

    This deliberately changes only resumed-prefill diagnostics. The reference
    start=0 invocation, ordinary decode and short standalone prompts are native.
    Prefix reuse across different policy domains is not solved by this control.
    """
    attribute = "_diagnostic_canonical_prefill_bf8"
    enabled = f'getattr(self, "{attribute}", False)'

    def replace_method(name, substitutions):
        source = textwrap.dedent(inspect.getsource(getattr(decoder.MultichipDecoder, name)))
        for old, new in substitutions:
            assert source.count(old) == 1, (name, old, "decoder source changed")
            source = source.replace(old, new)
        namespace = dict(decoder.__dict__)
        exec(  # noqa: S102 -- execute inspected local AST in isolated diagnostic namespace
            compile(source, f"<diagnostic_canonical_bf8_{name}>", "exec"), namespace
        )
        setattr(decoder.MultichipDecoder, name, namespace[name])

    if norms:
        replace_method(
            "norm",
            [
                (
                    "return self.decode_norm(x, name)",
                    (
                        "out = self.decode_norm(x, name)\n"
                        "        # Native decode projections/cache updates require BF16 storage.\n"
                        "        # Quantize values to the diagnostic policy, then preserve that storage.\n"
                        f"        return ttnn.typecast(ttnn.typecast(out, ttnn.bfloat8_b), out.dtype) if {enabled} else out"
                    ),
                ),
                (
                    'if x.shape[2] > 1 and name in ("input_layernorm", "post_attention_layernorm"):',
                    f'if (x.shape[2] > 1 or {enabled}) and name in ("input_layernorm", "post_attention_layernorm"):',
                ),
                ("if x.shape[2] >= 512:", f"if x.shape[2] >= 512 or {enabled}:"),
            ],
        )
    if reductions:
        replace_method(
            "reduce",
            [("reduced_precision = x.shape[2] >= 512", f"reduced_precision = x.shape[2] >= 512 or {enabled}")],
        )

    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import OrnithGenerator

    continuation = OrnithGenerator.prefill_logits_continuation

    def canonical_continuation(self, prompt_token_ids, start, *, positions=None, prefill_prefix=True):
        length = len(prompt_token_ids)
        layers = self.model.layers
        # This probe concerns a single original long chunk. Long multi-chunk
        # requests need absolute per-chunk policy state, not this request flag.
        if start and any(length > layer.chunk_size for layer in layers):
            raise ValueError("canonical_bf8 diagnostic requires one original chunk per layer")
        previous = [getattr(layer, attribute, False) for layer in layers]
        policies = []
        try:
            for layer in layers:
                original_rows = length if layer.kind == "linear_attention" else ((length + 31) // 32) * 32
                active = bool(start and original_rows >= 512)
                setattr(layer, attribute, active)
                policies.append(active)
            print(
                "CANONICAL_BF8_CONTEXT",
                {"length": length, "start": start, "norms": norms, "reductions": reductions, "layers": policies},
                flush=True,
            )
            return continuation(self, prompt_token_ids, start, positions=positions, prefill_prefix=prefill_prefix)
        finally:
            for layer, old in zip(layers, previous):
                setattr(layer, attribute, old)

    OrnithGenerator.prefill_logits_continuation = canonical_continuation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        choices=["bf16_prefill", "fixed32", "canonical_bf8", "canonical_bf8_norm", "canonical_bf8_reduce"],
        required=True,
    )
    args, rest = parser.parse_known_args()
    if args.candidate == "bf16_prefill":
        # Isolate shape-dependent activation/CCL quantization without changing
        # weights, math fidelity, caches, recurrence or the projection families.
        for name in ("norm", "reduce"):
            source = textwrap.dedent(inspect.getsource(getattr(decoder.MultichipDecoder, name)))
            source = source.replace("ttnn.bfloat8_b", "ttnn.bfloat16")
            namespace = dict(decoder.__dict__)
            exec(  # noqa: S102 -- inspected local diagnostic source
                compile(source, f"<diagnostic_bf16_{name}>", "exec"), namespace
            )
            setattr(decoder.MultichipDecoder, name, namespace[name])
    elif args.candidate.startswith("canonical_bf8"):
        install_canonical_bf8(
            norms=args.candidate != "canonical_bf8_reduce",
            reductions=args.candidate != "canonical_bf8_norm",
        )
    else:
        original = decoder.MultichipDecoder.from_state_dict.__func__

        def construct(cls, *args, **kwargs):
            kwargs["chunk_size"] = 32
            return original(cls, *args, **kwargs)

        decoder.MultichipDecoder.from_state_dict = classmethod(construct)
    print("DIAGNOSTIC_ONLY", args.candidate, flush=True)
    from readiness_check import run_prefill_resume_check

    if args.candidate.startswith("canonical_bf8"):
        # The harness normally loads generator.py under a fresh module name.
        # Use the patched canonical class for this diagnostic only.
        from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator

        run_prefill_resume_check._import_build_generator = lambda model_dir: build_generator

    sys.argv = [sys.argv[0], *rest]
    raise SystemExit(run_prefill_resume_check._main())


if __name__ == "__main__":
    main()
