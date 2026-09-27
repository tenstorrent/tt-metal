# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Retain TP4 prefill boundaries in the existing stack runner and then stop.

All ordinary stack CLI flags are accepted. The --output path receives boundary
JSON and a sibling .tensors.pt file. This intentionally retains tensor handles,
so a passing result can indicate that allocation/lifetime perturbation masks the
original failure. No host reads or extra device operations occur inside prefill.
"""

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import test_multichip_stack as stack


class DiagnosticComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs=2, default=[0, 5])
    args, _ = parser.parse_known_args()
    captures = []
    decoders = []
    completed_prefills = []
    original_factory = stack.MultichipDecoder.from_state_dict
    original_device_only = stack.device_only

    def retain(decoder, name, tensor, replicated=True):
        if getattr(decoder, "_diagnostic_prefill_active", False):
            captures.append(
                dict(
                    label=f"layer{decoder.layer_idx}/{name}",
                    layer=decoder.layer_idx,
                    tensor=tensor,
                    expected_replicated=replicated,
                )
            )

    def wrap_decoder(decoder):
        decoders.append(decoder)
        prefill = decoder.prefill_forward
        forward = decoder._forward
        allreduce = decoder.allreduce
        normalize = decoder.normalize
        tail = decoder._fused_tail

        def prefill_wrapper(hidden, **kwargs):
            decoder._diagnostic_prefill_active = True
            try:
                retain(decoder, "prefill_input_logical", hidden)
                result = prefill(hidden, **kwargs)
                retain(decoder, "prefill_output_logical", result)
                completed_prefills.append(decoder.layer_idx)
                return result
            finally:
                decoder._diagnostic_prefill_active = False

        def forward_wrapper(hidden, **kwargs):
            retain(decoder, "forward_input_padded", hidden)
            result = forward(hidden, **kwargs)
            retain(decoder, "forward_output_padded", result)
            return result

        def allreduce_wrapper(value, *, role="shared"):
            retain(decoder, f"{role}/collective_local_input", value, replicated=False)
            result = allreduce(value, role=role)
            retain(decoder, f"{role}/collective_output", result)
            return result

        def normalize_wrapper(value, epsilon, weight=None):
            site = (
                "input"
                if weight is decoder.input_norm_weight
                else (
                    "post_attention"
                    if weight is decoder.post_attention_norm_weight
                    else "common"
                    if weight is None
                    else "other"
                )
            )
            retain(decoder, f"normalize_{site}/input", value)
            result = normalize(value, epsilon, weight)
            retain(decoder, f"normalize_{site}/output", result)
            return result

        def tail_wrapper(residual, shared, routed, decode):
            retain(decoder, "tail/residual", residual)
            retain(decoder, "tail/shared", shared)
            retain(decoder, "tail/routed", routed)
            result = tail(residual, shared, routed, decode)
            retain(decoder, "tail/output", result)
            return result

        decoder.prefill_forward = prefill_wrapper
        decoder._forward = forward_wrapper
        decoder.allreduce = allreduce_wrapper
        decoder.normalize = normalize_wrapper
        decoder._fused_tail = tail_wrapper
        return decoder

    def factory(cls, *factory_args, **kwargs):
        return wrap_decoder(original_factory(*factory_args, **kwargs))

    def row_counts(mask):
        rows, width = mask.shape[-2:]
        return mask.reshape(-1, rows, width).sum(dim=(0, 2)).tolist()

    def inspect():
        host = {}
        boundaries = []
        first_bad = None
        for index, capture in enumerate(captures):
            tensor = capture["tensor"]
            label = capture["label"]
            key = f"{index:03d}/{label}"
            parts = [ttnn.to_torch(part).float().clone() for part in ttnn.get_device_tensors(tensor)]
            host[key] = parts
            mismatches = [part != parts[0] for part in parts]
            per_row = [row_counts(mask) for mask in mismatches]
            valid_rows = 33
            changed_valid = [int(mask[..., :valid_rows, :].sum()) for mask in mismatches]
            nonfinite = [int((~torch.isfinite(part)).sum()) for part in parts]
            entry = dict(
                index=index,
                label=label,
                expected_replicated=capture["expected_replicated"],
                logical_shape=list(tensor.shape),
                padded_shape=list(tensor.padded_shape),
                dtype=str(tensor.dtype),
                memory_config=str(tensor.memory_config()),
                nonfinite_per_rank=nonfinite,
                changed_from_rank0=[int(mask.sum()) for mask in mismatches],
                changed_valid_rows_from_rank0=changed_valid,
                changed_per_row_from_rank0=per_row,
                max_abs_diff_from_rank0=[float((part - parts[0]).abs().max()) for part in parts],
                first_mismatch_indices=[mask.nonzero()[:8].tolist() for mask in mismatches],
                sha256=[hashlib.sha256(part.contiguous().numpy().tobytes()).hexdigest() for part in parts],
            )
            if capture["expected_replicated"] and (any(changed_valid) or any(nonfinite)) and first_bad is None:
                first_bad = label
            boundaries.append(entry)
            print(
                "PREFILL_BOUNDARY",
                json.dumps(
                    dict(
                        label=label,
                        replicated=capture["expected_replicated"],
                        shape=entry["logical_shape"],
                        changed_valid=changed_valid,
                        changed_rows=[{i: n for i, n in enumerate(row) if n} for row in per_row],
                        nonfinite=nonfinite,
                    )
                ),
                flush=True,
            )

        padding = []
        for layer in args.layers:
            incoming_key = next(key for key in host if key.endswith(f"layer{layer}/prefill_input_logical"))
            padded_key = next(key for key in host if key.endswith(f"layer{layer}/forward_input_padded"))
            incoming, padded = host[incoming_key], host[padded_key]
            rows = incoming[0].shape[-2]
            padding.append(
                dict(
                    layer=layer,
                    valid_rows=rows,
                    changed_valid_after_padding=[
                        int((before != after[..., :rows, :]).sum()) for before, after in zip(incoming, padded)
                    ],
                    changed_per_valid_row_after_padding=[
                        row_counts(before != after[..., :rows, :]) for before, after in zip(incoming, padded)
                    ],
                    nonzero_padding_per_rank=[int((part[..., rows:, :] != 0).sum()) for part in padded],
                )
            )

        root = Path(stack.__file__).parents[1]
        report = dict(
            command=sys.argv,
            layers=args.layers,
            phase="prefill only; stopped before TP4 decode",
            first_bad_replicated_boundary=first_bad,
            pool_entries_per_layer={str(decoder.layer_idx): len(decoder._collective_buffers) for decoder in decoders},
            source_sha256={
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("tt/multichip_decoder.py", "tt/optimized_decoder.py", "tests/test_multichip_stack.py")
            },
            instrumentation="Retained TTNN handles only during forward; host reads after original device_only exited",
            limitation="Retention changes allocation lifetimes and may mask a reuse race",
            padding=padding,
            boundaries=boundaries,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tensor_path = args.output.with_suffix(".tensors.pt")
        torch.save(host, tensor_path)
        report["tensor_dump"] = str(tensor_path)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("PREFILL_DIAGNOSIS", json.dumps(dict(first_bad=first_bad, padding=padding)), flush=True)

    @contextmanager
    def device_only_wrapper():
        with original_device_only():
            yield
        if len(completed_prefills) == len(args.layers):
            inspect()
            raise DiagnosticComplete

    stack.MultichipDecoder.from_state_dict = classmethod(factory)
    stack.device_only = device_only_wrapper
    try:
        stack.main()
    except DiagnosticComplete:
        print("PREFILL_DIAGNOSTIC_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
