# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Observe QSA GR and router tensors, reading them only after the first prefill completes."""
import argparse
import hashlib
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests import probe_qsa_decoder as probe


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--device-output-cache", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [probe.__file__, *remaining]
    original_case = probe.device_case
    original_metrics = probe.metrics
    rows = []
    context = {}
    intermediates = []

    def retain(name, **values):
        if context.get("capture"):
            context["device_records"].append((name, values))

    class Projection:
        def __init__(self, original, name):
            self.original, self.name = original, name

        def __getattr__(self, name):
            return getattr(self.original, name)

        def __call__(self, value):
            output = self.original(value)
            retain(self.name, input=value, output=output)
            return output

    def device_case(*values, **options):
        context["length"] = options["length"]
        context["metric_index"] = 0
        context["capture"] = True
        context["device_records"] = []
        model = values[0]
        from models.autoports.qwen_qwen3_8_flash_next.tt import decoder_components, fused_decoder

        with ExitStack() as stack:
            for name in ("attn_mix", "mlp_mix"):
                component = getattr(model, name)
                for projection in ("precise_down", "precise_up"):
                    stack.enter_context(
                        patch.object(
                            component, projection, Projection(getattr(component, projection), name + "." + projection)
                        )
                    )
                original_read = component.read

                def read(value, original=original_read, name=name):
                    mixed, injection = original(value)
                    retain(name + ".read", input=value, mixed=mixed, injection=injection)
                    return mixed, injection

                stack.enter_context(patch.object(component, "read", read))
            original_logits = model.mlp.router_logits

            def logits(value):
                output = original_logits(value)
                retain("router", input=value, logits=output)
                return output

            stack.enter_context(patch.object(model.mlp, "router_logits", logits))
            original_topk = decoder_components.aten_router_topk

            def topk(value):
                scores, indices = original_topk(value)
                retain("topk", logits=value, scores=scores, indices=indices)
                return scores, indices

            for module in (decoder_components, fused_decoder):
                stack.enter_context(patch.object(module, "aten_router_topk", topk))
            return original_case(*values, **options)

    def metrics(expected, actual, residual):
        result = original_metrics(expected, actual, residual)
        if context["metric_index"] == 0:
            import ttnn

            context["capture"] = False
            for name, values in context["device_records"]:
                intermediates.append(
                    dict(
                        length=context["length"],
                        name=name,
                        values={key: ttnn.to_torch(value).clone() for key, value in values.items()},
                        metadata={
                            key: dict(
                                shape=list(value.shape),
                                padded_shape=list(value.padded_shape),
                                dtype=str(value.dtype),
                                layout=str(value.layout),
                                memory=str(value.memory_config()),
                            )
                            for key, value in values.items()
                        },
                    )
                )
            context["device_records"].clear()
        rows.append(
            dict(
                length=context["length"],
                metric_index=context["metric_index"],
                expected=expected.clone(),
                actual=actual.clone(),
                residual=residual.clone(),
                metrics=result,
            )
        )
        context["metric_index"] += 1
        return result

    try:
        with patch.object(probe, "device_case", device_case), patch.object(probe, "metrics", metrics):
            probe.main()
    finally:
        args.device_output_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rows=rows,
                intermediates=intermediates,
                probe_sha256=hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest(),
                export_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope="Observer-only GR/router tensor retention during first prefill; host readback after completion. Original fixtures and gates unchanged; compare exported full outputs with uninstrumented controls.",
            ),
            args.device_output_cache,
        )


if __name__ == "__main__":
    main()
