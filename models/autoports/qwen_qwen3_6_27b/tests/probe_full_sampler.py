# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact-vocabulary common-sampler ownership and trace reproducer, without layers."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import SamplingCCL
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder, _BlackholeCCLManager
from models.common.sampling.tt_sampling import TTSampling


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--negative-control", action="store_true")
    p.add_argument("--force-argmax", action="store_true")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--ties", action="store_true", help="Also prove returned tokens maximize tied logits")
    a = p.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=10000000)
    trace = None
    try:
        io = MultichipDecoder()
        io.mesh = mesh
        io.policy = dict(flat_ccl=True, persistent_ccl=True, ccl_decode_memory="DRAM")
        io.ccl = _BlackholeCCLManager(mesh, ttnn.get_num_links(mesh, 1), ttnn.Topology.Linear)
        io._ccl_buffers = {}

        class BorrowedCCL(SamplingCCL):
            def line_all_gather(self, tensor, **kwargs):
                value = io._collective(ttnn.reshape(tensor, tuple(tensor.shape)[1:]))
                return ttnn.reshape(value, (1, *tuple(value.shape)))

        ccl = (BorrowedCCL if a.negative_control else SamplingCCL)(SimpleNamespace(io=io))
        config = SimpleNamespace(
            vocab_size=248320,
            padded_vocab_size=248320,
            max_batch_size=1,
            max_top_k=32,
            cluster_shape=(1, 4),
            sampling_all_gather_axis=1,
            pad_logits_to_power_of_2=True,
            model_config={
                "SAMPLING_AG_CONFIG": dict(
                    allow_force_argmax=a.force_argmax, num_links=1, topology=ttnn.Topology.Linear
                )
            },
        )
        sampler = TTSampling(
            mesh, ccl, config, k=torch.ones(32, dtype=torch.int32), p=torch.zeros(32), temp=torch.ones(32)
        )
        logits = torch.full((1, 1, 32, 248320), -20.0, dtype=torch.bfloat16)
        winners = [1234 + i * 7000 for i in range(32)]
        for row, token in enumerate(winners):
            logits[0, 0, row, token] = 10.0
        x = ttnn.from_torch(
            logits,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )
        tokens = io._upload(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        sampler(x, tt_out_tok=tokens)
        print("SAMPLE_WARM_1", flush=True)
        try:
            sampler(x, tt_out_tok=tokens)
        except RuntimeError as error:
            if not a.negative_control:
                raise
            result = dict(
                expected_failure="Input Tensor is not allocated" in str(error), error=str(error).splitlines()[0]
            )
            assert result["expected_failure"]
            a.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            return
        if a.negative_control:
            raise AssertionError("Borrowed-output negative control did not fail")
        print("SAMPLE_WARM_2", flush=True)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            sampler(x, tt_out_tok=tokens)
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        outputs, expected_outputs = [], []
        for step in range(3):
            changed = [token + step for token in winners]
            logits.fill_(-20)
            for row, token in enumerate(changed):
                logits[0, 0, row, token] = 10
            host = ttnn.from_torch(
                logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3)
            )
            ttnn.copy_host_to_device_tensor(host, x)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            outputs.append(ttnn.to_torch(ttnn.get_device_tensors(tokens)[0]).flatten().tolist())
            expected_outputs.append(changed)
        assert outputs == expected_outputs, (outputs, expected_outputs)
        tied = []
        if a.ties:
            for maxima in ([17, 62083, 124169, 186247], list(range(70000, 70065))):
                logits.fill_(-20)
                logits[..., maxima] = 10
                host = ttnn.from_torch(
                    logits,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
                )
                ttnn.copy_host_to_device_tensor(host, x)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                actual = ttnn.to_torch(ttnn.get_device_tensors(tokens)[0]).flatten().tolist()
                assert all(token in maxima for token in actual), (maxima, actual)
                tied.append(
                    dict(
                        maxima=maxima,
                        actual=actual,
                        all_select_maximum=True,
                        hf_lowest_index_match=all(token == min(maxima) for token in actual),
                    )
                )
        ttnn.synchronize_device(mesh)
        begin = time.perf_counter()
        for _ in range(a.iterations):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        seconds = time.perf_counter() - begin
        result = dict(
            pass_greedy=True,
            outputs=outputs,
            expected=expected_outputs,
            trace_replays=3 + len(tied) + a.iterations,
            tied_cases=tied,
            force_argmax=a.force_argmax,
            measured_iterations=a.iterations,
            sampler_ms=seconds * 1000 / a.iterations,
        )
        a.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
