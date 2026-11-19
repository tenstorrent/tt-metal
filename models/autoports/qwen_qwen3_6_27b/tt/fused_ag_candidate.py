# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in decode MLP topology probe; no production-default registration.

Construction and prepare() belong before trace capture. One owner, one CQ.
Each family requires its own recorded device validation. See its stage-5 note.
"""

import ttnn


class FusedAGMLPCandidate:
    """Hidden-sharded add/norm -> AGMM -> local SwiGLU/down -> RS/add.

    ``separate`` is a distributed-norm control using the decoder's selected
    DRAM-sharded projection. ``minimal`` and ``mcast`` fuse AG with projection.
    Optional fused SwiGLU uses paired tiles, never contiguous gate/up halves.
    """

    def __init__(self, decoder, *, family="minimal", fuse_swiglu=False, grid=(10, 2), k_block=8):
        assert family in ("separate", "minimal", "mcast", "mcast1d")
        assert not fuse_swiglu or family == "minimal"
        assert decoder.hidden_width == 1280 and decoder.policy["packed_mlp"]
        assert tuple(decoder.mesh.shape) == (1, 4)
        self.decoder, self.family, self.fuse_swiglu = decoder, family, fuse_swiglu
        self.original_finish = decoder._finish
        self.buffers = {}
        self.sealed = False
        mesh = decoder.mesh
        full = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full.x - 1, full.y - 1))})
        self.semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
        self.barrier = ttnn.create_global_semaphore(mesh, cores, 0)
        self.fence_semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
        self.fence_barrier = ttnn.create_global_semaphore(mesh, cores, 0)
        self.num_links = min(2, decoder.ccl.num_links)
        assert grid[0] < full.x and grid[1] <= full.y
        assert grid[1] % self.num_links == 0
        self.workers_per_link = grid[1] // self.num_links
        self.config = ttnn.MinimalMatmulConfig(
            M_block_size=1,
            K_block_size=k_block,
            N_block_size=4,
            subblock_h=1,
            subblock_w=2,
            compute_with_storage_grid_size=grid,
        )
        # Setup-only conversion preserves existing quantized values. Remove the
        # decode reader padding (normally N=9216) before the fused interface.
        if fuse_swiglu:
            assert decoder.swiglu_weight is not None
            self.weight = decoder.swiglu_weight
        else:
            self.weight = ttnn.to_memory_config(decoder.weights["mlp.gate_up.weight"], ttnn.DRAM_MEMORY_CONFIG)
            self.weight = self.weight[:, :8704]
        assert tuple(self.weight.shape) == (5120, 8704)
        self.weight = ttnn.reshape(self.weight, (1, 1, 5120, 8704))
        self.mcast_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            in0_block_w=k_block,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=34,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )
        if family == "mcast1d":
            self.mcast_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=k_block,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=9,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
                num_global_cb_receivers=0,
            )

    def prepare(self, batches=(1, 32)):
        """Allocate every remote-write target before eager warmup/capture."""
        assert not self.sealed
        import torch

        for batch in batches:
            assert 1 <= batch <= 32
            if batch not in self.buffers:
                # Exact logical batch, common physical 32-row allocation.
                self.buffers[batch] = self.decoder._upload(torch.zeros(1, 1, batch, 5120, dtype=torch.bfloat16))
        self.fence_input = self.decoder._upload(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16))
        self.fence_output = self.decoder._upload(torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16))

    def seal(self):
        """Call after exact-shape eager warmup, before capture."""
        assert self.buffers
        self.sealed = True

    def _local_norm(self, x):
        m = self.decoder
        batch = x.shape[1]
        x4 = ttnn.reshape(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), (1, 1, batch, 1280))
        stats = ttnn.rms_norm_pre_all_gather(x4, compute_kernel_config=m.ckc, dtype=ttnn.bfloat16)
        stats = m._collective(ttnn.reshape(stats, tuple(stats.shape)[1:]))
        stats = ttnn.reshape(stats, (1, *tuple(stats.shape)))
        return ttnn.rms_norm_post_all_gather(
            x4,
            stats,
            epsilon=m.config.rms_norm_eps,
            weight=m.shard_norms["post_attention_layernorm.weight"],
            compute_kernel_config=m.ckc,
        )

    def _admission_fence(self):
        # Retain the stage-4 workaround: minimal's Linear barrier alone did
        # not protect asynchronous repeated launches. Include its cost in trace.
        m = self.decoder
        ttnn.experimental.all_gather_async(
            self.fence_input,
            dim=3,
            cluster_axis=1,
            mesh_device=m.mesh,
            topology=ttnn.Topology.Linear,
            num_links=self.num_links,
            multi_device_global_semaphore=self.fence_semaphores,
            barrier_semaphore=self.fence_barrier,
            persistent_output_tensor=self.fence_output,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def finish(self, x, attention, prefill=False, *, decode=False):
        # Explicit opt-in is decode only; all other calls retain their original
        # implementation, including arbitrary/nonaligned prompt lengths.
        if not decode:
            return self.original_finish(x, attention, prefill=prefill, decode=decode)
        m = self.decoder
        shape = tuple(x.shape)
        batch = shape[0] * shape[1]
        assert shape[-1] == 1280 and 1 <= batch <= 32
        assert batch in self.buffers, "Call prepare for this batch before tracing"
        x = ttnn.reshape(x, (1, batch, 1280))
        reduced = m._reduce(attention)
        reduced = ttnn.reshape(reduced, (1, batch, 1280))
        residual = ttnn.add(x, reduced)
        local = self._local_norm(residual)
        if self.family == "separate":
            gathered = m._collective(ttnn.reshape(local, (1, batch, 1280)))
            packed = m._linear(gathered, "mlp.gate_up.weight")
        elif self.family == "minimal":
            self._admission_fence()
            packed = ttnn.experimental.all_gather_minimal_matmul_async(
                local,
                self.weight,
                config=self.config,
                multi_device_global_semaphore=self.semaphores,
                barrier_semaphore=self.barrier,
                persistent_output_buffer=self.buffers[batch],
                num_links=self.num_links,
                num_workers_per_link=self.workers_per_link,
                cluster_axis=1,
                topology=ttnn.Topology.Linear,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=m.matmul_configs["mlp"],
                dtype=ttnn.bfloat16,
                force_transpose=False,
                transpose_core_grid=False,
                fuse_swiglu=self.fuse_swiglu,
            )[0]
            packed = ttnn.reshape(packed, (1, batch, 4352 if self.fuse_swiglu else 8704))
        else:
            _, packed = ttnn.experimental.all_gather_matmul_async(
                local,
                self.weight,
                persistent_output_buffer=self.buffers[batch],
                dim=3,
                multi_device_global_semaphore=self.semaphores,
                barrier_semaphore=self.barrier,
                all_gather_core_grid_offset=(0, 5) if self.family == "mcast1d" else (0, 2),
                num_links=1,
                num_workers_per_link=2,
                topology=ttnn.Topology.Linear,
                memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.mcast_config,
                compute_kernel_config=m.matmul_configs["mlp"],
                dtype=ttnn.bfloat16,
            )
            packed = ttnn.reshape(packed, (1, batch, 8704))
        if self.fuse_swiglu:
            act = packed
        elif m.policy.get("fused_packed_swiglu", False):
            from .packed_swiglu_candidate import packed_swiglu

            act = packed_swiglu(packed, round_silu=m.policy["round_silu"])
        else:
            gate, up = packed[:, :, :4352], packed[:, :, 4352:8704]
            act = (
                ttnn.multiply(ttnn.silu(gate), up)
                if m.policy["round_silu"]
                else ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            )
        output = ttnn.add(residual, m._reduce(m._linear(act, "mlp.down_proj.weight")))
        return ttnn.reshape(output, shape)

    def install(self):
        """Explicit parent hook; retain this object until captured traces die."""
        self.decoder._finish = self.finish
        return self


def main():
    """Owner-run complete-boundary comparison; never invoked by import."""
    import argparse
    import hashlib
    import json
    import time
    from pathlib import Path

    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import load_layer, pcc
    from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, choices=(0, 3), default=0)
    parser.add_argument("--batch", type=int, choices=(1, 32), default=32)
    parser.add_argument("--family", choices=("separate", "minimal", "mcast", "mcast1d"), default="minimal")
    parser.add_argument("--fuse-swiglu", action="store_true")
    parser.add_argument("--grid", type=int, nargs=2, default=(10, 2))
    parser.add_argument("--k-block", type=int, default=8)
    parser.add_argument("--policy", type=json.loads, default={})
    parser.add_argument("--activation-file", type=Path)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.replays > 0
    torch.set_num_threads(8)
    torch.manual_seed(28)
    root = Path(__file__).resolve().parents[1]
    cfg = Qwen3_5TextConfig(**json.loads((root / "doc/hf_config.json").read_text())["text_config"])
    weights = load_layer(args.layer)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
    try:
        decoder = MultichipDecoder.from_state_dict(
            weights,
            hf_config=cfg,
            layer_idx=args.layer,
            mesh_device=mesh,
            policy=args.policy,
        )
        candidate = FusedAGMLPCandidate(
            decoder,
            family=args.family,
            fuse_swiglu=args.fuse_swiglu,
            grid=tuple(args.grid),
            k_block=args.k_block,
        )
        candidate.prepare((args.batch,))
        if args.activation_file:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

            recorded = torch.load(args.activation_file, weights_only=True)
            prompt, token = (recorded[key][: args.batch].contiguous() for key in ("x", "d"))
            assert prompt.shape[0] == args.batch and tuple(token.shape) == (args.batch, 1, 5120)
            length = prompt.shape[1]
            pages = (length + 32) // 32
            state = (
                decoder.create_recurrent_state(args.batch)
                if decoder.kind == "linear_attention"
                else decoder.create_paged_state(torch.arange(args.batch * pages).reshape(args.batch, pages))
            )
            pk, dk = {}, {}
            if decoder.kind == "full_attention":
                rotary = Qwen3_5TextRotaryEmbedding(cfg)
                pk["rope"] = tuple(
                    decoder._upload(t.unsqueeze(1)) for t in rotary(prompt, torch.arange(length).expand(args.batch, -1))
                )
                dk["rope"] = tuple(
                    decoder._upload(t.unsqueeze(1)) for t in rotary(token, torch.full((args.batch, 1), length))
                )
                dk["current_pos"] = ttnn.from_torch(
                    torch.full((args.batch,), length, dtype=torch.int32),
                    device=mesh,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
            decoder.prefill_forward(decoder.upload_residual(prompt), state=state, **pk)
            boundary = {}

            def capture_boundary(residual, partial, prefill=False, *, decode=False):
                assert decode and not prefill
                boundary.update(residual=residual, attention=partial)
                return candidate.original_finish(residual, partial, decode=True)

            decoder._finish = capture_boundary
            try:
                decoder.decode_forward(decoder.upload_decode_residual(token), state=state, **dk)
            finally:
                del decoder._finish
            x, attention = boundary["residual"], boundary["attention"]
            fixture = "real weights and recorded text activations; actual decoder attention partials"
        else:
            x = decoder.upload_residual(torch.randn(1, args.batch, 5120, dtype=torch.bfloat16) * 0.1)
            # Row-projection partials: attention RS is included in both timings.
            attention = decoder._upload_shards(
                [torch.randn(1, args.batch, 5120, dtype=torch.bfloat16) * 0.025 for _ in range(4)]
            )
            fixture = "real weights; seeded synthetic residual and attention partials"

        def read(tensor):
            return torch.cat([ttnn.to_torch(v) for v in ttnn.get_device_tensors(tensor)], -1)

        baseline = lambda: candidate.original_finish(x, attention, decode=True)
        fused = lambda: candidate.finish(x, attention, decode=True)
        residual = read(ttnn.add(x, decoder._reduce(attention)))
        reference = read(baseline())

        def metrics(value):
            return {
                "pcc": pcc(reference, value),
                "contribution_pcc": pcc(reference - residual, value - residual),
                "min_user_pcc": min(pcc(reference[:, i], value[:, i]) for i in range(args.batch)),
                "finite": bool(torch.isfinite(value).all()),
            }

        report = {
            "family": args.family,
            "fuse_swiglu": args.fuse_swiglu,
            "batch": args.batch,
            "layer": args.layer,
            "grid": args.grid,
            "k_block": args.k_block,
            "policy": decoder.policy,
            "scope": "attention RS + residual add + norm + AG/projection + SwiGLU + down/RS + residual add",
            "fixture": fixture,
            "activation_file": str(args.activation_file) if args.activation_file else None,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "decoder_sha256": hashlib.sha256((root / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
            "runs": {},
            "status": "running",
            "pass": False,
        }
        for name, operation in (("baseline", baseline), ("candidate", fused)):
            print(f"PHASE {name} eager", flush=True)
            eager = read(operation())
            check = metrics(eager)
            report["runs"][name] = check
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            assert (
                check["finite"] and min(check[k] for k in ("pcc", "contribution_pcc", "min_user_pcc")) >= 0.995
            ), check
            for _ in range(2):
                warm = operation()
                ttnn.synchronize_device(mesh)
                del warm
            candidate.seal()
            print(f"PHASE {name} capture", flush=True)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = operation()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            first = read(out)
            print(f"PHASE {name} queued replay", flush=True)
            start = time.perf_counter()
            for _ in range(args.replays):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            check["trace_ms"] = (time.perf_counter() - start) * 1000 / args.replays
            last = read(out)
            check["replay"] = metrics(last)
            check["replay_equal"] = bool(torch.equal(first, last) and torch.equal(eager, last))
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            assert check["replay_equal"], check
            assert (
                check["replay"]["finite"]
                and min(check["replay"][k] for k in ("pcc", "contribution_pcc", "min_user_pcc")) >= 0.995
            ), check
            ttnn.release_trace(mesh, tid)
            del out
        report.update(status="pass", **{"pass": True})
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
