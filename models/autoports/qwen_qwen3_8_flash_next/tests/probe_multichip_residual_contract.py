# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure a real-weight projection -> residual write -> complete next GR read.

Run under the stage's serialized, bounded hardware runner. This is a topology
probe, not a whole-decoder benchmark: the common producer uses the real layer0
GDN WO as an ordinary row-parallel BF16 matmul, whereas the shipping GDN uses
an exact projection. All candidates use identical operands and producer math.
The hidden-sharded candidate keeps the reduced block sharded through residual
add and distributed norm; it never immediately gathers the RS result.
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import linear
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    Collectives,
    GatheredProjection,
    MeshExactProjection,
    MeshLoader,
    MeshResidual,
)

DRAM = ttnn.DRAM_MEMORY_CONFIG
CANDIDATES = ("gather_fp32", "gather_bf16_widen", "native_ar_fp32", "hidden_rs_fp32")


class ProbeCollectives(Collectives):
    def __init__(self, device, policy):
        super().__init__(device)
        self.policy = policy

    def gather(self, x, dim=-1):
        return ttnn.all_gather(
            x,
            dim=dim % len(x.shape),
            cluster_axis=1,
            memory_config=DRAM,
            num_links=2,
            topology=ttnn.Topology.Ring,
        )

    def sum(self, x):
        if self.policy == "native_ar_fp32":
            return ttnn.all_reduce(
                ttnn.typecast(x, ttnn.float32),
                cluster_axis=1,
                memory_config=DRAM,
                num_links=2,
                topology=ttnn.Topology.Ring,
            )
        if self.policy == "gather_bf16_widen":
            return ttnn.sum(ttnn.typecast(self.gather(x, 1), ttnn.float32), dim=1, keepdim=True)
        # Keep this control explicit: the shipping Collectives.sum policy may
        # change while the four transport alternatives are being measured.
        return ttnn.sum(self.gather(ttnn.typecast(x, ttnn.float32), 1), dim=1, keepdim=True)


class HiddenResidual:
    """[1,4,T,640] residual; each rank owns one hidden slice of every stream."""

    def __init__(self, loader, prefix, config, ccl):
        w = loader.state_dict
        self.ccl, self.compute, self.eps = ccl, loader.compute, config.rms_norm_eps
        norm = w[prefix + "hc_norm.weight"].float().reshape(4, 2560) + 1
        self.weight = loader.shards([p.reshape(1, 4, 1, 640) for p in norm.chunk(4, dim=1)], dim=3, dtype=ttnn.float32)
        down = w[prefix + "input_mix_weight_down.weight"].float()
        self.down = GatheredProjection(
            MeshExactProjection(loader, list(down.chunk(4, dim=0)), dtype=ttnn.float32),
            ccl,
            valid_width=80,
        )
        up = w[prefix + "input_mix_weight_up.weight"].float().reshape(4, 2560, 320)
        self.up = MeshExactProjection(
            loader,
            [p.reshape(2560, 320) for p in up.chunk(4, dim=1)],
            dtype=ttnn.float32,
            block=256,
        )
        # All four scalars are needed to update the four local stream slices.
        self.inject = loader.tensor(
            torch.nn.functional.pad(w[prefix + "block_inject_weight.weight"].T, (0, 28)).reshape(1, 1, 10240, 32)
        )

    @staticmethod
    def write(residual, block, injection):
        update = ttnn.typecast(
            ttnn.multiply(ttnn.typecast(block, ttnn.float32), ttnn.typecast(injection, ttnn.float32)),
            ttnn.bfloat16,
        )
        return ttnn.typecast(
            ttnn.add(ttnn.typecast(residual, ttnn.float32), ttnn.typecast(update, ttnn.float32)),
            ttnn.bfloat16,
        )

    def read(self, residual):
        n = residual.shape[-2]
        value = ttnn.typecast(residual, ttnn.float32)
        local_mean = ttnn.mean(ttnn.multiply(value, value), dim=-1, keepdim=True)
        # Gather on a new rank axis, preserving the separate four-stream axis.
        rank_stats = self.ccl.gather(local_mean, dim=0)  # [4,4,T,1]
        mean = ttnn.multiply(ttnn.sum(rank_stats, dim=0, keepdim=True), 0.25)
        scale = ttnn.rsqrt(ttnn.add(mean, self.eps))
        normalized = ttnn.typecast(ttnn.multiply(ttnn.multiply(value, scale), self.weight), ttnn.bfloat16)
        full = self.ccl.gather(normalized, dim=3)  # [1,4,T,2560]
        full = ttnn.reshape(ttnn.permute(full, (0, 2, 1, 3)), (1, 1, n, 10240))
        down = ttnn.multiply(self.down(full), 0.25)
        mixing = ttnn.typecast(ttnn.silu(ttnn.typecast(down, ttnn.float32)), ttnn.bfloat16)
        mixing = ttnn.typecast(ttnn.sigmoid(ttnn.typecast(self.up(mixing), ttnn.float32)), ttnn.bfloat16)
        mixing = ttnn.permute(ttnn.reshape(mixing, (1, n, 4, 640)), (0, 2, 1, 3))
        products = ttnn.typecast(ttnn.multiply(mixing, normalized), ttnn.float32)
        mixed_local = ttnn.typecast(ttnn.multiply(ttnn.sum(products, dim=1, keepdim=True), 0.25), ttnn.bfloat16)
        mixed = self.ccl.gather(mixed_local, dim=3)
        injection = linear(full, self.inject, self.compute)[..., :4]
        injection = ttnn.multiply(
            ttnn.typecast(ttnn.sigmoid(ttnn.typecast(ttnn.multiply(injection, 0.25), ttnn.float32)), ttnn.bfloat16),
            2,
        )
        injection = ttnn.permute(ttnn.reshape(injection, (1, 1, n, 4)), (0, 3, 2, 1))
        return mixed, injection


def compare(expected, actual):
    if expected.shape != actual.shape:
        raise ValueError(f"Output contract mismatch: {expected.shape} != {actual.shape}")
    finite = bool(torch.isfinite(actual).all())
    if not finite:
        return dict(finite=False, pcc=None, max_absolute_error=None)
    a, b = expected.double().flatten(), actual.double().flatten()
    score = 1.0 if torch.equal(expected, actual) else float(torch.corrcoef(torch.stack((a, b)))[0, 1])
    if not math.isfinite(score):
        score = None
    return dict(
        finite=True, pcc=score, exact=torch.equal(expected, actual), max_absolute_error=float((a - b).abs().max())
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Optional saved layer0 probe operands; residual uses inputs[0]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2071)
    args = parser.parse_args()
    if any(n < 1 or n > 32 for n in args.lengths) or args.repeats < 1:
        parser.error("Each logical length must be 1..32 and repeats must be positive")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    weights = {}
    for prefix in ("attn_hyper_connection.", "mlp_hyper_connection.", "linear_attn.out_proj."):
        weights.update(load_weights(args.snapshot, 0, prefix))
    max_n = max(args.lengths)
    if args.reference:
        saved = torch.load(args.reference, map_location="cpu", weights_only=False)
        if saved.get("metadata", {}).get("layer") != 0:
            raise ValueError("Reference must describe layer0")
        host_residual = saved["inputs"][0][:, :, :max_n, :].contiguous()
        if tuple(host_residual.shape) != (1, 1, max_n, 10240):
            raise ValueError("Reference needs enough residual rows")
    else:
        host_residual = (torch.randn(1, 1, max_n, 10240) * 0.02).bfloat16()
    host_features = (torch.randn(1, 1, max_n, 6144) * 0.02).bfloat16()
    report = dict(
        passed=False,
        argv=sys.argv,
        seed=args.seed,
        snapshot=str(args.snapshot.resolve()),
        reference=None if args.reference is None else str(args.reference.resolve()),
        layer=0,
        mesh=[1, 4],
        regime="serial",
        rows=[],
        weights="real layer0",
        activations="seeded synthetic producer features; shared residual operands",
        scope="native row-parallel layer0 WO + collective + residual write + complete next MLP GR read",
        limitations=[
            "Not whole-decoder timing",
            "WO is a common native producer, not shipping GDN exact projection",
            "Eager warmed timing only; no trace acceptance is claimed",
        ],
        excluded_from_timing=[
            "weight/input setup",
            "first GR read providing injection",
            "boundary host reconstruction",
        ],
        per_rank_contracts={
            "producer_input": "BF16 TILE [1,1,T,1536]",
            "producer_weight": "BF16 TILE [1,1,1536,2560]",
            "producer_partial": "BF16 TILE [1,1,T,2560]",
            "stream_residual": "BF16 TILE [1,1,T,2560], one complete stream",
            "hidden_residual": "BF16 TILE [1,4,T,640], one hidden slice of every stream",
            "hidden_stats": "FP32 TILE [1,4,T,1], gather on dim0 then rank mean",
            "hidden_normalized_gather": "BF16 TILE [1,4,T,2560], follows distributed RMSNorm",
            "next_gr_down": "FP32 weights [96,10240], 80 valid output columns per rank",
            "next_gr_up": "FP32 weights [2560,320], stream or hidden-slice output ownership",
            "next_gr_mixed_output": "BF16 TILE replicated [1,1,T,2560]",
            "next_gr_injection_output": "BF16 TILE: stream [1,1,T,1]; hidden replicated [1,4,T,1]",
            "memory": "All tensors interleaved DRAM; logical T=1..32, physical T padded to32",
        },
    )
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        loader = MeshLoader(weights, mesh)
        ccls = {name: ProbeCollectives(mesh, name) for name in CANDIDATES}
        first = MeshResidual(loader, "attn_hyper_connection.", config, ccls["gather_fp32"])
        controls = {
            name: MeshResidual(loader, "mlp_hyper_connection.", config, ccls[name])
            for name in CANDIDATES
            if name != "hidden_rs_fp32"
        }
        hidden = HiddenResidual(loader, "mlp_hyper_connection.", config, ccls["gather_fp32"])
        projection = loader.shards(
            [p.T.reshape(1, 1, 1536, 2560) for p in weights["linear_attn.out_proj.weight"].chunk(4, dim=1)],
            dim=2,
        )

        def upload(value, dim):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=DRAM,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
            )

        def read(outputs, candidate):
            residual, mixed, injection = outputs
            residual = ttnn.to_torch(residual, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))
            if candidate == "hidden_rs_fp32":
                residual = residual.permute(0, 2, 1, 3).reshape(1, 1, residual.shape[-2], 10240)
            mixed = ttnn.to_torch(mixed, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
            if candidate == "hidden_rs_fp32":
                injection = ttnn.to_torch(injection, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
            else:
                injection = ttnn.to_torch(injection, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
            return {"residual": residual.clone(), "mixed": mixed.clone(), "injection": injection.clone()}

        for n in args.lengths:
            residual_host = host_residual[:, :, :n, :]
            residual = upload(residual_host, 3)
            fractured = upload(residual_host.reshape(1, n, 4, 2560).permute(0, 2, 1, 3), 3)
            features = upload(host_features[:, :, :n, :], 3)
            _, injection = first.read(residual)
            injection_all = ccls["gather_fp32"].gather(injection, dim=1)
            ttnn.synchronize_device(mesh)

            def run(candidate):
                partial = linear(features, projection, loader.compute)
                if candidate == "hidden_rs_fp32":
                    block = ttnn.reduce_scatter(
                        ttnn.typecast(partial, ttnn.float32),
                        dim=3,
                        cluster_axis=1,
                        memory_config=DRAM,
                        num_links=2,
                        topology=ttnn.Topology.Ring,
                    )
                    block = ttnn.typecast(block, ttnn.bfloat16)
                    updated = hidden.write(fractured, block, injection_all)
                    mixed, next_injection = hidden.read(updated)
                else:
                    block = ttnn.typecast(ccls[candidate].sum(partial), ttnn.bfloat16)
                    updated = controls[candidate].write(residual, block, injection)
                    mixed, next_injection = controls[candidate].read(updated)
                return updated, mixed, next_injection

            expected = read(run("gather_fp32"), "gather_fp32")
            order = ["gather_fp32"] + [c for c in args.candidates if c != "gather_fp32"]
            for candidate in order:
                actual = read(run(candidate), candidate)
                metrics = {key: compare(expected[key], actual[key]) for key in expected}
                # Warm the whole producer/consumer path before serialized host timing.
                warm = run(candidate)
                ttnn.synchronize_device(mesh)
                del warm
                samples = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    result = run(candidate)
                    ttnn.synchronize_device(mesh)
                    samples.append((time.perf_counter() - start) * 1000)
                    del result
                row = dict(
                    length=n,
                    candidate=candidate,
                    metrics=metrics,
                    samples_ms=samples,
                    median_ms=statistics.median(samples),
                    passed=all(m["finite"] and m["pcc"] is not None and m["pcc"] >= 0.995 for m in metrics.values()),
                )
                report["rows"].append(row)
                print(json.dumps({k: row[k] for k in ("length", "candidate", "median_ms", "passed")}), flush=True)
            baseline_ms = next(
                row["median_ms"] for row in report["rows"] if row["length"] == n and row["candidate"] == "gather_fp32"
            )
            for row in report["rows"]:
                if row["length"] == n:
                    row["speedup_vs_fp32_gather"] = baseline_ms / row["median_ms"]
            del residual, fractured, features, injection, injection_all
        report["passed"] = all(row["passed"] for row in report["rows"])
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        finally:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
