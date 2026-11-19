# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Observe BF16 split/tilize transport semantics for all 65536 seed patterns.

Eager diagnostic only: no trace, model quality or performance claim. Reports
actual uploaded coverage; seed coverage alone does not prove device coverage.
The existing 42-case hidden-row correctness gate is deliberately unchanged.
"""

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.hidden_row_mask import split_masked_hidden_rows
from models.autoports.qwen_qwen3_6_35b_a3b.tt.slot_state import copy_if

ROOT = Path(__file__).resolve().parents[1]
SLOTS = tuple(range(32))
WIDTH = 2048


def bits(tensor):
    assert tensor.dtype == torch.bfloat16
    return tensor.contiguous().view(torch.int16).to(torch.int32).bitwise_and(65535)


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def physical(tensor):
    return [ttnn.from_device(local).to_torch_with_padded_shape().clone() for local in ttnn.get_device_tensors(tensor)]


def bit_classes(words):
    exponent = (words >> 7) & 255
    fraction = words & 127
    sign = (words & 32768) != 0
    zero_exp = exponent == 0
    special = exponent == 255
    return {
        "positive_zero": words == 0,
        "negative_zero": words == 32768,
        "positive_subnormal": zero_exp & (fraction != 0) & ~sign,
        "negative_subnormal": zero_exp & (fraction != 0) & sign,
        "positive_normal": (exponent > 0) & (exponent < 255) & ~sign,
        "negative_normal": (exponent > 0) & (exponent < 255) & sign,
        "positive_infinity": words == 32640,
        "negative_infinity": words == 65408,
        "positive_quiet_nan": special & ((fraction & 64) != 0) & ~sign,
        "negative_quiet_nan": special & ((fraction & 64) != 0) & sign,
        "positive_signaling_nan": special & (fraction > 0) & ((fraction & 64) == 0) & ~sign,
        "negative_signaling_nan": special & (fraction > 0) & ((fraction & 64) == 0) & sign,
    }


def transitions(source, target):
    source, target = source.reshape(-1), target.reshape(-1)
    assert source.shape == target.shape and source.dtype == target.dtype == torch.int32
    changed = source != target
    groups = {}
    for label, selected in bit_classes(source).items():
        indices = (selected & changed).nonzero().reshape(-1)
        groups[label] = dict(
            observed_words=int(selected.sum()),
            distinct_source_patterns=int(torch.unique(source[selected]).numel()),
            changed_words=int(indices.numel()),
            distinct_changed_outputs=int(torch.unique(target[indices]).numel()),
            examples=[
                dict(index=int(i), source=f"0x{int(source[i]):04x}", output=f"0x{int(target[i]):04x}")
                for i in indices[:8]
            ],
        )
    pairs = torch.unique(torch.stack((source, target), dim=1), dim=0)
    order = torch.argsort(pairs[:, 0], stable=True)
    pairs = pairs[order]
    conflicting = (pairs[1:, 0] == pairs[:-1, 0]) & (pairs[1:, 1] != pairs[:-1, 1])
    return dict(
        words=source.numel(),
        distinct_source_patterns=int(torch.unique(source).numel()),
        changed_words=int(changed.sum()),
        distinct_source_output_pairs=len(pairs),
        single_valued_by_source_bits=not bool(conflicting.any()),
        source_sha256=digest(source),
        output_sha256=digest(target),
        sorted_unique_pairs_sha256=digest(pairs),
        classes=groups,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "readiness_vllm/hidden_row_transport_probe/result.json")
    args = parser.parse_args()
    maps_path = args.output.with_name(args.output.stem + "_maps.pt")
    if args.output.exists() or maps_path.exists():
        parser.error("Use fresh output and mapping paths; existing evidence is preserved")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "tt/hidden_row_mask.py", ROOT / "tt/slot_state.py", Path(__file__).resolve()]
    result = dict(
        status="RUNNING",
        command=sys.argv,
        mesh=[1, 4],
        logical_source_shape=[1, 32, WIDTH],
        physical_source_shape=[1, 32, WIDTH],
        upload="host BF16 TILE construction then copy_host_to_device_tensor into persistent replicated source",
        seed_patterns=65536,
        semantics="Observed bit mappings through padded host readback; no raw-copy correctness pass claim",
        limitations=[
            "Eager synthetic transport diagnostic, not model quality, trace correctness or performance.",
            "Actual uploaded distinct patterns are reported; host conversion may reduce seed-pattern coverage.",
            "Raw-copy versus old-chain mismatches are measurements, never accepted as equivalent.",
        ],
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        chips=[],
        mask_cases=[],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    mesh = None
    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        seed_words = torch.arange(65536, dtype=torch.int32).reshape(1, 32, WIDTH)
        seed = seed_words.to(torch.int16).view(torch.bfloat16)
        host = ttnn.from_torch(seed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        source = ttnn.empty(
            (1, 32, WIDTH),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.copy_host_to_device_tensor(host, source)
        uploaded = physical(source)
        assert len(uploaded) == 4 and all(tuple(value.shape) == (1, 32, WIDTH) for value in uploaded)
        source_words = [bits(value) for value in uploaded]

        def upload_mask(value, device=True):
            kwargs = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper)
            if device:
                kwargs.update(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.from_torch(value.contiguous(), **kwargs)

        mask = torch.ones(32, dtype=torch.int32)
        activity = upload_mask(mask)
        scalars = [upload_mask(mask[slot : slot + 1]) for slot in SLOTS]
        old = None
        candidate = None

        def reference():
            row_major = ttnn.to_layout(source, ttnn.ROW_MAJOR_LAYOUT)
            rows = tuple(ttnn.to_layout(row_major[:, row : row + 1, :], ttnn.TILE_LAYOUT) for row in SLOTS)
            outputs = []
            for row, value in enumerate(rows):
                value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
                destination = old[row] if old is not None else ttnn.empty_like(value)
                outputs.append(copy_if(value, destination, scalars[row], mesh, zero_if_inactive=True))
            return tuple(outputs)

        old_logical = candidate_logical = None
        for label, live in (("all", SLOTS), ("none", ()), ("even", SLOTS[::2]), ("return_all", SLOTS)):
            next_mask = torch.zeros(32, dtype=torch.int32)
            next_mask[list(live)] = 1
            if not torch.equal(next_mask, mask):
                ttnn.copy_host_to_device_tensor(upload_mask(next_mask, device=False), activity)
                for slot in SLOTS:
                    if mask[slot] != next_mask[slot]:
                        ttnn.copy_host_to_device_tensor(
                            upload_mask(next_mask[slot : slot + 1], device=False), scalars[slot]
                        )
                mask = next_mask
            old = reference()
            candidate = split_masked_hidden_rows(source, activity, SLOTS, mesh, outputs=candidate)
            observed_old = [physical(row) for row in old]
            observed_candidate = [physical(row) for row in candidate]
            logical_old = [torch.stack([bits(observed_old[row][rank])[0, 0] for row in SLOTS]) for rank in range(4)]
            logical_candidate = [
                torch.stack([bits(observed_candidate[row][rank])[0, 0] for row in SLOTS]) for rank in range(4)
            ]
            if label == "all":
                old_logical, candidate_logical = logical_old, logical_candidate
            case = dict(label=label, active_slots=list(live), chips=[])
            for rank in range(4):
                expected_old = torch.zeros((32, WIDTH), dtype=torch.int32)
                expected_candidate = torch.zeros_like(expected_old)
                expected_old[list(live)] = old_logical[rank][list(live)]
                expected_candidate[list(live)] = candidate_logical[rank][list(live)]
                case["chips"].append(
                    dict(
                        rank=rank,
                        old_logical_mask_mismatches=int((logical_old[rank] != expected_old).sum()),
                        candidate_logical_mask_mismatches=int((logical_candidate[rank] != expected_candidate).sum()),
                        old_nonzero_padding_words=sum(int((bits(row[rank])[:, 1:] != 0).sum()) for row in observed_old),
                        candidate_nonzero_padding_words=sum(
                            int((bits(row[rank])[:, 1:] != 0).sum()) for row in observed_candidate
                        ),
                        old_vs_candidate_padded_mismatches=sum(
                            int((bits(left[rank]) != bits(right[rank])).sum())
                            for left, right in zip(observed_old, observed_candidate)
                        ),
                    )
                )
            result["mask_cases"].append(case)
            save()
        final_source = [bits(value) for value in physical(source)]
        for rank in range(4):
            words = source_words[rank].reshape(32, WIDTH)
            predicted = words.clone()
            predicted[((words >> 7) & 255) == 0] = 0
            actual_old = old_logical[rank]
            source_classes = bit_classes(words)
            finite = ((words >> 7) & 255) != 255
            mismatch = predicted != actual_old
            result["chips"].append(
                dict(
                    rank=rank,
                    actual_distinct_uploaded_patterns=int(torch.unique(words).numel()),
                    actual_distinct_uploaded_finite_patterns=int(torch.unique(words[finite]).numel()),
                    seed_to_uploaded=transitions(seed_words, words),
                    uploaded_to_old=transitions(words, actual_old),
                    uploaded_to_candidate=transitions(words, candidate_logical[rank]),
                    exp_zero_to_positive_zero_otherwise_identity=dict(
                        exact_for_observed_patterns=not bool(mismatch.any()),
                        mismatched_words=int(mismatch.sum()),
                        finite_mismatched_words=int((mismatch & finite).sum()),
                        mismatch_by_class={key: int((mismatch & value).sum()) for key, value in source_classes.items()},
                        prediction_to_actual=transitions(predicted, actual_old),
                    ),
                    unchanged_source=torch.equal(source_words[rank], final_source[rank]),
                    source_equals_chip0=torch.equal(source_words[rank], source_words[0]),
                    old_output_equals_chip0=torch.equal(actual_old, old_logical[0]),
                    candidate_output_equals_chip0=torch.equal(candidate_logical[rank], candidate_logical[0]),
                )
            )
        torch.save(
            dict(
                seed=seed_words,
                uploaded=torch.stack(source_words),
                old=torch.stack(old_logical),
                candidate=torch.stack(candidate_logical),
            ),
            maps_path,
        )
        result.update(
            status="OBSERVED",
            mapping_artifact=str(maps_path),
            mapping_sha256=hashlib.sha256(maps_path.read_bytes()).hexdigest(),
            observed_all_65536_on_every_chip=all(
                row["actual_distinct_uploaded_patterns"] == 65536 for row in result["chips"]
            ),
            raw_copy_correctness_pass_claimed=False,
        )
    except BaseException as error:
        result.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
                result["mesh_closed"] = True
        except BaseException as error:
            result.update(status="FAIL", cleanup_error=repr(error), cleanup_traceback=traceback.format_exc())
            raise
        finally:
            save()
    print("HIDDEN_ROW_TRANSPORT_OBSERVED", args.output, flush=True)


if __name__ == "__main__":
    main()
