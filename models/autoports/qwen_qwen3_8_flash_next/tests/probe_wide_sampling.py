# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized sampler-only TP4 proof for sorted64, integer pairing and stale inputs.

This is a correctness diagnostic, not serving performance or model accuracy.
Requires the opt-in native sampling extension; never run beside a serving process.
"""

import argparse
import json
import math
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists() or args.draws < 256:
        parser.error("Use a fresh output and at least 256 draws")
    import torch

    import ttnn

    torch.set_num_threads(4)
    report = dict(passed=False, mesh=[1, 4], draws=args.draws, checks=[], started=time.time())
    mesh, trace = None, None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024, l1_small_size=32768)

        def upload(value, dtype, tiled=False):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT if tiled else ttnn.ROW_MAJOR_LAYOUT,
            )

        def refresh(destination, value):
            host = ttnn.from_torch(value.contiguous(), dtype=destination.dtype, layout=destination.layout)
            ttnn.copy_host_to_device_tensor(host, destination)

        def read_all(value):
            hosts = [ttnn.to_torch(rank) for rank in ttnn.get_device_tensors(value)]
            assert len(hosts) == 4 and all(torch.equal(host, hosts[0]) for host in hosts[1:]), "TP replicas disagree"
            return hosts[0]

        # All values are exactly representable and distinct; labels exercise
        # UINT32 beyond 65535, rather than a float cast that could hide pairing loss.
        perm = torch.randperm(256, generator=torch.Generator().manual_seed(82))
        values = ((torch.arange(256).float() - 128) / 128)[perm].reshape(1, 1, 1, 256).repeat(1, 1, 32, 1).bfloat16()
        ids = (torch.arange(256, dtype=torch.int64) * 7919 + 65535).reshape(1, 1, 1, 256).repeat(1, 1, 32, 1)
        vt, it = upload(values, ttnn.bfloat16, True), upload(ids, ttnn.uint32, True)

        def pair():
            selected, positions = ttnn.topk(vt, k=64, dim=-1, largest=True, sorted=True, stable=False)
            return selected, ttnn.gather(it, 3, positions)

        def check_pair(output, label):
            expected, positions = torch.topk(values.float(), 64, dim=-1, sorted=True)
            assert torch.equal(read_all(output[0]).float(), expected), label + " values"
            assert torch.equal(read_all(output[1]).long(), torch.gather(ids, -1, positions)), label + " labels"
            report["checks"].append(label)

        check_pair(pair(), "eager_global_top64_integer_pairing")
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        paired = pair()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        check_pair(paired, "traced_global_top64_integer_pairing")
        values = values.flip(-1).contiguous()
        ids = (ids.flip(-1) + 1000000).contiguous()
        refresh(vt, values)
        refresh(it, ids)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        check_pair(paired, "traced_changed_values_and_uint32_labels")
        ttnn.release_trace(mesh, trace)
        trace = None
        del paired, vt, it

        # Every physical row has disjoint labels, catching wrong-row reads at
        # both face boundaries and the second candidate tile.
        values = (-torch.arange(64).float() / 128).reshape(1, 1, 1, 64).repeat(1, 1, 32, 1).bfloat16()
        ids = torch.arange(64).reshape(1, 1, 1, 64) + 100000 * torch.arange(1, 33).reshape(1, 1, 32, 1)
        ks = ([1, 16, 17, 31, 32] + list(range(33, 41)) + [48, 64, 40]) * 2
        vt, it = upload(values, ttnn.bfloat16, True), upload(ids, ttnn.uint32)
        kt = upload(torch.tensor(ks), ttnn.uint32)
        pt = upload(torch.ones(32), ttnn.bfloat16)
        temp = upload(torch.ones(32), ttnn.bfloat16)
        seeds = upload(torch.arange(1, 33) * 7919, ttnn.uint32)
        user_ids = upload(torch.arange(32), ttnn.uint32)

        def sample():
            ttnn.manual_seed(seeds=seeds, user_ids=user_ids)
            return ttnn.sampling(vt, it, k=kt, p=pt, temp=temp, input_sorted=True)

        sample()  # compile before capture
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = sample()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        offsets = torch.arange(1, 33) * 100000
        for probability in (0.0, 1.0, 0.95):
            refresh(pt, torch.full((32,), probability))
            counts = torch.zeros(32, 64, dtype=torch.int64)
            for draw in range(1 if probability == 0.0 else args.draws):
                refresh(seeds, (torch.arange(1, 33) * 7919 + draw * 104729) % 1000000 + 1)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                selected = read_all(output).flatten().long()[:32] - offsets
                assert ((selected >= 0) & (selected < torch.tensor(ks))).all(), (probability, selected.tolist())
                counts[torch.arange(32), selected] += 1
            if probability == 0.0:
                assert counts[:, 0].tolist() == [1] * 32
            else:
                stats = []
                for row, k in enumerate(ks):
                    probs = torch.softmax(values[0, 0, row, :k].float(), dim=-1).bfloat16().float()
                    p_device = torch.tensor(probability).bfloat16().item()
                    cut = min(k, int(torch.searchsorted(probs.cumsum(0), p_device, right=True)) + 1)
                    ref = torch.zeros(64)
                    ref[:cut] = probs[:cut] / probs[:cut].sum()
                    empirical = counts[row].float() / args.draws
                    tv = 0.5 * (empirical - ref).abs().sum().item()
                    # Multinomial TV noise grows with support size. Bound its
                    # expectation by marginal variances, then use bounded
                    # differences with family-wise delta=0.001 over 64 rows.
                    expected_tv_bound = 0.5 * torch.sqrt(ref * (1 - ref) / args.draws).sum().item()
                    concentration = math.sqrt(math.log(64 / 0.001) / (2 * args.draws))
                    # One BF16 nucleus-edge candidate can differ through the
                    # approximate device exponential; all other support is strict.
                    edge_mass = ref[cut - 1].item()
                    bound = expected_tv_bound + concentration + edge_mass
                    assert counts[row, min(cut + 1, k) :].sum() == 0, ("outside nucleus", probability, row, cut)
                    assert tv <= bound, ("distribution", probability, row, k, tv, bound)
                    # p1 tests candidate33 exactly. For a truncated nucleus,
                    # require a second-tile candidate strictly inside the allowed
                    # one-candidate BF16 cutoff uncertainty, not only at its edge.
                    if cut > 32 and (probability == 1.0 or cut > 33):
                        assert counts[row, 32:cut].sum() > 0, ("missing second tile", probability, row, k)
                    stats.append(
                        dict(
                            row=row,
                            k=k,
                            nucleus_size=cut,
                            total_variation=tv,
                            bound=bound,
                            second_tile_draws=int(counts[row, 32:].sum()),
                        )
                    )
                report["checks"].append(
                    dict(
                        name="sorted64_distribution",
                        p=probability,
                        stats=stats,
                        bound_method="multinomial marginal variance + McDiarmid family delta0.001 + one BF16 edge mass",
                    )
                )

        # Same fixed seed + changed excluded values must preserve all draws.
        refresh(kt, torch.full((32,), 40))
        refresh(pt, torch.full((32,), 0.95))
        refresh(seeds, torch.arange(1, 33) * 7919)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        before = read_all(output).clone()
        changed = values.clone()
        changed[..., 40:] -= 20
        refresh(vt, changed)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        assert torch.equal(read_all(output), before), "excluded candidates affect k40 normalization"
        refresh(kt, torch.full((32,), 1))
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        assert torch.equal(read_all(output).flatten().long()[:32], offsets), "stale k tensor"
        changed_ids = ids + 7000000
        refresh(it, changed_ids)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        assert torch.equal(read_all(output).flatten().long()[:32], offsets + 7000000), "stale label tensor"
        report["checks"].extend(
            ["excluded_tail_output_invariance", "persistent_changed_k", "persistent_changed_labels"]
        )
        ttnn.release_trace(mesh, trace)
        trace = None

        # Logical H=1 shares the same physical tile convention.
        one_values = upload(values[:, :, :1, :], ttnn.bfloat16, True)
        one_ids = upload(ids[:, :, :1, :], ttnn.uint32)
        one_k = upload(torch.tensor([40]), ttnn.uint32)
        one_p = upload(torch.tensor([0.0]), ttnn.bfloat16)
        one_temp = upload(torch.tensor([1.0]), ttnn.bfloat16)
        one = ttnn.sampling(one_values, one_ids, k=one_k, p=one_p, temp=one_temp, input_sorted=True, seed=42)
        assert read_all(one).flatten()[0].item() == 100000
        report["checks"].append("logical_one_row_first_candidate")
        report["passed"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.time() - report["started"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)
    print(json.dumps(dict(passed=report["passed"], output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
