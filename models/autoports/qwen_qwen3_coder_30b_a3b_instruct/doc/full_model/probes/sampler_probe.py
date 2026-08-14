# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Why is split sampling 11 ms, and can the contract be fixed rather than dropped?

``perf_full_model.py`` at 48 layers measured the canonical ``Sampling1D`` split
path at **11.005 ms** against a 20.214 ms model trace -- 34.6% of the 31.826 ms
token-out decode it sat inside, which is exactly the case the full-model skill
says must be fixed at the LM-head/sampling contract before anything else is
tuned. (An earlier revision of this docstring, and of the documents, rounded
that share to "36%"; 11.005861550802365 / 31.826252466998994 = 0.34581.)

This sweeps the sampler alone at the shipped logits shape --
``[1, 1, 32, 37984]`` bf16 per die, the column-parallel LM head's output -- so
each leg costs seconds rather than a four-minute model load. No decoder layer is
involved and none is needed: the sampler sees only the logits.

Legs:

* ``split_padded``   -- what shipped: ``pad_to_power_of_2=True`` (37984 -> 65536)
                        then ``ttnn.topk(k=32)`` per die, all-gather, ``ttnn.sampling``
* ``split_unpadded`` -- the same without the power-of-two pad
* ``split_k<N>``     -- the same with a smaller ``max_top_k``
* ``force_argmax``   -- ``Sampling1D``'s own argmax path: all-gather the full
                        151936-wide vocabulary, untilize, ``ttnn.argmax``

Every leg is checked to produce the **same token** as a host argmax of the
gathered logits, so a speed win cannot come from being wrong. Timed warmed and
traced.

**About the archived log.** ``sampler_probe.log`` is the original sweep and
carries the four split legs only -- it predates the ``force_argmax`` leg and the
component breakdown below, which were added afterwards, and it is kept as-is
because it is the artifact the documents cite for 11.006 / 6.151 / 11.104 /
6.268. Re-running this file reproduces those four legs and adds the rest; it
also overwrites ``sampler_probe.json``, which is why no ``.json`` is archived
beside the log. The in-model force-argmax figure the documents quote (1.859 ms)
comes from ``perf_full_model.csv``, not from here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig  # noqa: E402
from models.common.modules.tt_ccl import TT_CCL  # noqa: E402

HERE = Path(__file__).resolve().parent
VOCAB = 151936
DEVICES = 4
LOCAL_VOCAB = VOCAB // DEVICES  # 37984
SLOTS = 32


def build_sampler(mesh, ccl, *, max_top_k, pad_to_power_of_2):
    sampler = Sampling1D.from_config(
        Sampling1DConfig(
            vocab_size=VOCAB,
            valid_vocab_size=VOCAB,
            mesh_device=mesh,
            tt_ccl=ccl,
            max_batch_size=SLOTS,
            max_top_k=max_top_k,
            num_gather_links=1,
            sampling_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            allow_force_argmax=True,
            num_argmax_gather_links=1,
            ag_topology=ttnn.Topology.Ring,
            pad_to_power_of_2=pad_to_power_of_2,
        )
    )
    sampler.load_device_buffers()
    return sampler


def traced_ms(mesh, fn, reps=20):
    fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, DEVICES), trace_region_size=50_000_000)
    results = {}
    try:
        ccl = TT_CCL(mesh)
        torch.manual_seed(0)
        # One global logit row per slot, split across the four dies exactly as
        # the column-parallel LM head splits it.
        host = torch.randn(1, 1, SLOTS, VOCAB) * 4.0
        expected = host[0, 0].argmax(dim=-1).tolist()
        logits = ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        assert int(logits.shape[-1]) == LOCAL_VOCAB, logits.shape

        def params(k_value):
            return (
                ttnn.from_torch(
                    torch.full((SLOTS,), k_value, dtype=torch.int32),
                    device=mesh,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                ),
                ttnn.from_torch(
                    torch.zeros(SLOTS, dtype=torch.bfloat16),
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                ),
                ttnn.from_torch(
                    torch.ones(SLOTS, dtype=torch.bfloat16),
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                ),
            )

        k1, p0, t1 = params(1)

        # ``max_top_k`` below 32 is **not swept**, and that is a finding rather
        # than an omission. The split path all-gathers a ``[1,1,32,max_top_k]``
        # candidate block; at 32 that is exactly one tile wide, and below it
        # ``ttnn.all_gather`` logs
        #
        #     Using slower composite all_gather: gather dim 3 is padded from 16
        #     to 32; size must be a multiple of the tile/shard extent
        #
        # and takes the composite path. At 16 that is merely slower -- it was
        # measured at 6.268 ms unpadded against 32's 6.151, i.e. *worse* despite
        # gathering half the candidates. At 8 the composite gather did not
        # return at all: the leg ran for over 20 minutes on the mesh before it
        # was killed. So 32 is both the fastest and the only tile-aligned
        # candidate width, and shrinking ``max_top_k`` is not a lever here.
        # ``k16`` is kept in the sweep because it is the measurement behind
        # "shrinking max_top_k is not a lever" -- removing it would leave the
        # claim in the prose with no leg in the probe, and this file's own
        # archived log (``sampler_probe.log``) contains all four. Only ``k8`` is
        # removed, and only because it hung the mesh.
        legs = [
            ("split_k32_padded", 32, True),
            ("split_k32_unpadded", 32, False),
            ("split_k16_padded", 16, True),
            ("split_k16_unpadded", 16, False),
        ]

        for name, max_top_k, pad in legs:
            try:
                sampler = build_sampler(mesh, ccl, max_top_k=max_top_k, pad_to_power_of_2=pad)

                def run(sampler=sampler):
                    return sampler.decode_forward(logits, k=k1, p=p0, temp=t1, enable_log_probs=False)[0]

                out = run()
                ttnn.synchronize_device(mesh)
                tokens = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1)[:SLOTS].tolist()
                exact = [int(a) == int(b) for a, b in zip(tokens, expected)]
                results[name] = {
                    "ms": traced_ms(mesh, run, args.reps),
                    "matches_host_argmax": f"{sum(exact)}/{SLOTS}",
                }
            except Exception as exc:  # noqa: BLE001 - recorded
                results[name] = {"error": repr(exc)}
            print(f"{name:<28} {results[name]}", flush=True)

        sampler = build_sampler(mesh, ccl, max_top_k=32, pad_to_power_of_2=True)

        def argmax_run():
            return sampler.decode_forward(logits, enable_log_probs=False)[0]

        out = argmax_run()
        ttnn.synchronize_device(mesh)
        tokens = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1)[:SLOTS].tolist()
        exact = [int(a) == int(b) for a, b in zip(tokens, expected)]
        results["force_argmax"] = {
            "ms": traced_ms(mesh, argmax_run, args.reps),
            "matches_host_argmax": f"{sum(exact)}/{SLOTS}",
        }
        print(f"{'force_argmax':<28} {results['force_argmax']}", flush=True)

        # Component breakdown of the shipped split path.
        padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 65536 - LOCAL_VOCAB)], value=-1e30)
        results["component_pad_to_65536_ms"] = traced_ms(
            mesh, lambda: ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 65536 - LOCAL_VOCAB)], value=-1e30), args.reps
        )
        results["component_topk32_over_65536_ms"] = traced_ms(mesh, lambda: ttnn.topk(padded, k=32, dim=-1), args.reps)
        results["component_topk32_over_37984_ms"] = traced_ms(mesh, lambda: ttnn.topk(logits, k=32, dim=-1), args.reps)
        results["component_full_vocab_all_gather_ms"] = traced_ms(
            mesh,
            lambda: ttnn.all_gather(
                logits, dim=3, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=ttnn.Topology.Ring
            ),
            args.reps,
        )
        for key in (
            "component_pad_to_65536_ms",
            "component_topk32_over_65536_ms",
            "component_topk32_over_37984_ms",
            "component_full_vocab_all_gather_ms",
        ):
            print(f"{key:<40} {results[key]:.4f} ms", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    (HERE / "sampler_probe.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
