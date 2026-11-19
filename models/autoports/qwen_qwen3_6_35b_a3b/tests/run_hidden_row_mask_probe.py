# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact logical/padded BF16 split-mask probe with retained mixed-shape traces.

Requires the archived hidden_row_mask candidate to be applied. No weights,
attention, sampler or server are used; this is not model quality or performance.
"""

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.hidden_row_mask import split_masked_hidden_rows
from models.autoports.qwen_qwen3_6_35b_a3b.tt.slot_state import copy_if

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    ((0, 1), 2048),
    ((0, 1, 2, 3), 2048),
    (tuple(range(32)), 2048),
    ((31, 0), 64),
    ((31, 0, 17, 2), 64),
    (tuple(reversed(range(32))), 64),
)


def exact(left, right, label):
    assert left.shape == right.shape and left.dtype == right.dtype == torch.bfloat16, label
    a, b = left.contiguous().view(torch.int16), right.contiguous().view(torch.int16)
    if not torch.equal(a, b):
        unequal = a != b
        first = int(unequal.reshape(-1).nonzero()[0])
        raise AssertionError(f"{label}: {int(unequal.sum())} unequal BF16 words, first_flat_index={first}")


def physical(outputs):
    # This existing host API preserves implicit padding; ordinary to_torch
    # would expose only row0 and could miss another request's residual bytes.
    return [
        [ttnn.from_device(local).to_torch_with_padded_shape().clone() for local in ttnn.get_device_tensors(output)]
        for output in outputs
    ]


def equal_outputs(left, right, label):
    assert len(left) == len(right)
    for row, (a, b) in enumerate(zip(left, right)):
        assert len(a) == len(b) == 4
        for rank, (x, y) in enumerate(zip(a, b)):
            exact(x, y, f"{label}/row{row}/rank{rank}")


def seed_upload_report(seed, uploaded):
    """Report transport differences; the seed is not the extraction oracle."""
    assert seed.dtype == torch.bfloat16 and seed.ndim == 3 and seed.shape[0] == len(uploaded) == 4
    reports = []
    for rank, physical_source in enumerate(uploaded):
        assert physical_source.shape == (1, 32, seed.shape[-1])
        source = physical_source[0, : seed.shape[1]].contiguous()
        expected = seed[rank].contiguous().view(torch.int16)
        observed = source.view(torch.int16)
        unequal = expected != observed
        transitions = []
        if bool(unequal.any()):
            pairs, counts = torch.unique(
                torch.stack((expected[unequal].to(torch.int32), observed[unequal].to(torch.int32)), dim=1),
                dim=0,
                return_counts=True,
            )
            transitions = [
                dict(
                    seed_bits=f"0x{int(pair[0]) & 0xFFFF:04x}",
                    uploaded_bits=f"0x{int(pair[1]) & 0xFFFF:04x}",
                    count=int(count),
                )
                for pair, count in zip(pairs, counts)
            ]
        reports.append(
            dict(
                rank=rank,
                logical_words=expected.numel(),
                different_words=int(unequal.sum()),
                first_different_coordinates=unequal.nonzero()[:16].tolist(),
                bit_transitions=transitions,
                seed_sha256=hashlib.sha256(expected.numpy().tobytes()).hexdigest(),
                uploaded_logical_sha256=hashlib.sha256(observed.numpy().tobytes()).hexdigest(),
                uploaded_padded_sha256=hashlib.sha256(
                    physical_source.contiguous().view(torch.int16).numpy().tobytes()
                ).hexdigest(),
            )
        )
    return reports


def old_split_canonical_bits(value):
    """Independent CPU formula measured for every BF16 pattern on four chips.

    The old tilize boundary maps exponent-zero values (including -0) to +0;
    exponent-all-ones values lose the mantissa while preserving their sign.
    Normal finite bits are unchanged. Do not import the candidate kernel rule.
    """
    assert value.dtype == torch.bfloat16
    words = value.contiguous().view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
    exponent = words & 0x7F80
    canonical = torch.where(exponent == 0, 0, words)
    canonical = torch.where(exponent == 0x7F80, words & 0xFF80, canonical)
    return canonical.to(torch.int16).view(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "readiness_vllm/hidden_row_mask_probe/result.json")
    parser.add_argument("--trace-region-size", type=int, default=128 * 1024 * 1024)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path; existing evidence is preserved")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "tt/hidden_row_mask.py", ROOT / "tt/slot_state.py", Path(__file__).resolve()]
    result = dict(
        status="RUNNING",
        command=sys.argv,
        mesh=[1, 4],
        trace_region_size=args.trace_region_size,
        tracking=True,
        skip_program_cache=False,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        comparison="All padded BF16 bits versus old split/mask and measured BF16 canonicalization of uploaded source snapshots",
        cpu_oracle_rule="exp==0 -> +0; exp==0x7f80 -> bits&0xff80; all other bits unchanged",
        cpu_oracle_evidence="readiness_vllm/optimized_hidden_row_transport/result.json; all65536 patterns on all4 chips",
        limitations=[
            "Synthetic data movement microgate, not model/serving accuracy or performance.",
            "The oracle observes uploaded source through padded host readback; seed transport bit preservation is not assumed.",
            "Diagnostic readback synchronizes after nonblocking replay; no async-overlap claim.",
            "Control and candidate destinations are preallocated; production allocation lifetimes need full-model gates.",
        ],
        captures=0,
        releases=0,
        replays=0,
        variants=[],
        source_uploads=[],
        cases=[],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    mesh = None
    traces = []
    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)
        rng = torch.Generator().manual_seed(20261117)

        def upload(value, dtype, layout, *, sharded=False, device=True):
            mapper = ttnn.ShardTensorToMesh(mesh, dim=0) if sharded else ttnn.ReplicateTensorToMesh(mesh)
            kwargs = dict(dtype=dtype, layout=layout, mesh_mapper=mapper)
            if device:
                kwargs.update(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.from_torch(value.contiguous(), **kwargs)

        all_active = torch.ones(32, dtype=torch.int32)
        active = upload(all_active, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        scalars = [upload(all_active[s : s + 1], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) for s in range(32)]
        mask_current = all_active.clone()
        variants = []

        def reference(v):
            row_major = ttnn.to_layout(v["source"], ttnn.ROW_MAJOR_LAYOUT)
            rows = tuple(ttnn.to_layout(row_major[:, i : i + 1, :], ttnn.TILE_LAYOUT) for i in range(len(v["slots"])))
            outputs = []
            for row, (value, slot) in enumerate(zip(rows, v["slots"])):
                value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
                destination = v["reference"][row] if "reference" in v else ttnn.empty_like(value)
                outputs.append(copy_if(value, destination, scalars[slot], mesh, zero_if_inactive=True))
            return tuple(outputs)

        def check(v, snapshots, mask, generation, label):
            for row, slot in enumerate(v["slots"]):
                for rank, observed in enumerate(snapshots[row]):
                    expected = torch.zeros((1, 32, v["width"]), dtype=torch.bfloat16)
                    if mask[slot]:
                        expected[0, 0] = old_split_canonical_bits(v["uploaded"][generation][rank][0, row])
                    exact(expected, observed, f"{label}/oracle/row{row}/rank{rank}")

        # Allocate all persistent tensors and warm every candidate/reference,
        # diagnostic read and copy program before the first trace capture.
        for ordinal, (slots, width) in enumerate(VARIANTS):
            hosts = [torch.randn((4, len(slots), width), generator=rng).to(torch.bfloat16) for _ in range(2)]
            for host in hosts:
                host[..., 0] = -0.0
                host[..., 1] = 0.0
                # Finite edge seeds expose transport normalization separately;
                # extraction must match the measured old split/tilize semantics.
                edge_bits = torch.tensor([0x0001, 0x007F, 0x0080, 0x7F7F, -32767, -32640], dtype=torch.int16)
                host[..., 2:8] = edge_bits.view(torch.bfloat16)
            v = dict(slots=slots, width=width, host=hosts, generation=0)
            v["source"] = upload(hosts[0], ttnn.bfloat16, ttnn.TILE_LAYOUT, sharded=True)
            # Observe the constructor separately: it may execute device tilize,
            # unlike the host-construction/copy path used for replay refresh.
            initial_uploaded = physical((v["source"],))[0]
            result["source_uploads"].append(
                dict(
                    variant=ordinal,
                    generation=0,
                    path="device_constructor",
                    chips=seed_upload_report(hosts[0], initial_uploaded),
                )
            )
            v["host_inputs"] = [
                upload(host, ttnn.bfloat16, ttnn.TILE_LAYOUT, sharded=True, device=False) for host in hosts
            ]
            v["uploaded"] = []
            for generation in range(2):
                ttnn.copy_host_to_device_tensor(v["host_inputs"][generation], v["source"])
                snapshot = physical((v["source"],))[0]
                v["uploaded"].append(snapshot)
                result["source_uploads"].append(
                    dict(
                        variant=ordinal,
                        generation=generation,
                        path="replay_refresh",
                        chips=seed_upload_report(hosts[generation], snapshot),
                    )
                )
                save()
            ttnn.copy_host_to_device_tensor(v["host_inputs"][0], v["source"])
            equal_outputs([v["uploaded"][0]], physical((v["source"],)), f"restore0/{ordinal}")
            v["candidate"] = split_masked_hidden_rows(v["source"], active, slots, mesh)
            warmed_reference = v["reference"] = reference(v)
            observed = physical(v["candidate"])
            equal_outputs(physical(warmed_reference), observed, f"warm{ordinal}")
            check(v, observed, mask_current, 0, f"warm{ordinal}")
            addresses = [v["source"].buffer_address(), *[t.buffer_address() for t in v["candidate"]]]
            assert len(set(addresses)) == len(addresses), "Candidate outputs alias one another or source"
            v["zero"] = upload(torch.zeros((1, 1, width), dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            ttnn.copy(v["zero"], v["candidate"][0])
            equal_outputs(observed[1:], physical(v["candidate"][1:]), f"independent{ordinal}")
            split_masked_hidden_rows(v["source"], active, slots, mesh, outputs=v["candidate"])
            variants.append(v)
            result["variants"].append(
                dict(
                    slots=list(slots),
                    width=width,
                    independent_storage=True,
                    source_address=addresses[0],
                    output_addresses=addresses[1:],
                )
            )
            save()
        ttnn.synchronize_device(mesh)
        for ordinal, v in enumerate(variants):
            for owner in ("reference", "candidate"):
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                traces.append(trace)
                result["captures"] += 1
                try:
                    if owner == "reference":
                        v[owner] = reference(v)
                    else:
                        split_masked_hidden_rows(v["source"], active, v["slots"], mesh, outputs=v[owner])
                finally:
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                v[owner + "_trace"] = trace
            result["variants"][ordinal]["traces"] = {
                owner: str(v[owner + "_trace"]) for owner in ("reference", "candidate")
            }
        ttnn.synchronize_device(mesh)
        entries = int(mesh.num_program_cache_entries())
        result["prepared_program_entries"] = entries
        # A deliberately live allocation must be rejected before replay. This
        # control is freed immediately; the workload uses no acknowledgments.
        unsafe = ttnn.empty_like(variants[0]["candidate"][0])
        try:
            try:
                ttnn.execute_trace(mesh, variants[0]["candidate_trace"], cq_id=0, blocking=False)
            except RuntimeError as error:
                if "device buffer(s) still alive before" not in str(error):
                    raise
                result["tracker_positive_control"] = dict(rejected=True, message=str(error))
            else:
                raise AssertionError("Allocation tracker failed to reject the live positive control")
        finally:
            ttnn.deallocate(unsafe)
            del unsafe
        save()
        sequence = (
            ("all_active", tuple(range(32)), 0),
            ("unchanged", tuple(range(32)), 0),
            ("source_only", tuple(range(32)), 1),
            ("even_activity", tuple(range(0, 32, 2)), 1),
            ("slot31", (31,), 1),
            ("none", (), 0),
            ("return_all", tuple(range(32)), 0),
        )
        previous = None
        for label, selected, generation in sequence:
            mask = torch.zeros(32, dtype=torch.int32)
            mask[list(selected)] = 1
            mask_changed = not torch.equal(mask, mask_current)
            if mask_changed:
                ttnn.copy_host_to_device_tensor(upload(mask, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device=False), active)
                for slot in range(32):
                    if mask[slot] != mask_current[slot]:
                        ttnn.copy_host_to_device_tensor(
                            upload(mask[slot : slot + 1], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device=False),
                            scalars[slot],
                        )
                mask_current = mask.clone()
            for ordinal, v in enumerate(variants):
                changed_source = v["generation"] != generation
                if changed_source:
                    ttnn.copy_host_to_device_tensor(v["host_inputs"][generation], v["source"])
                    v["generation"] = generation
                for owner in ("reference", "candidate"):
                    ttnn.execute_trace(mesh, v[owner + "_trace"], cq_id=0, blocking=False)
                    result["replays"] += 1
                ttnn.synchronize_device(mesh)
                left, right = physical(v["reference"]), physical(v["candidate"])
                equal_outputs(left, right, f"{label}/variant{ordinal}")
                check(v, right, mask, generation, f"{label}/variant{ordinal}")
                equal_outputs([v["uploaded"][generation]], physical((v["source"],)), f"source/{label}/{ordinal}")
                retained_previous = previous is not None and previous[0] is not v
                if retained_previous:
                    equal_outputs(previous[1], physical(previous[0]["candidate"]), f"retained/{label}/{ordinal}")
                previous = (v, right)
                assert int(mesh.num_program_cache_entries()) == entries
                assert [int(t.buffer_address()) for t in v["candidate"]] == result["variants"][ordinal][
                    "output_addresses"
                ]
                result["cases"].append(
                    dict(
                        label=label,
                        variant=ordinal,
                        source_changed=changed_source,
                        activity_changed=mask_changed,
                        exact_padded_all_chips=True,
                        retained_previous_output=retained_previous,
                        program_entries=entries,
                    )
                )
                save()
            print("HIDDEN_ROW_MASK_PHASE_PASS", label, flush=True)
        result.update(status="PASS", case_count=len(result["cases"]), workload_releases=0)
    except BaseException as error:
        result.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if mesh is not None:
                for trace in traces:
                    ttnn.release_trace(mesh, trace)
                    result["releases"] += 1
                ttnn.close_mesh_device(mesh)
                result["mesh_closed"] = True
        except BaseException as error:
            result.update(status="FAIL", cleanup_error=repr(error), cleanup_traceback=traceback.format_exc())
            raise
        finally:
            save()
    print("HIDDEN_ROW_MASK_PROBE_PASS", args.output, flush=True)


if __name__ == "__main__":
    main()
