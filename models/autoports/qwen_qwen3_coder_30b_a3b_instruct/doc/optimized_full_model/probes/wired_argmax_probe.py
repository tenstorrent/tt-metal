# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Does the *wired-in* distributed argmax honour ``_sample_argmax``'s contract?

``distributed_argmax_probe.py`` established the reduction and its 1.82x on a
synthetic tensor. This file checks the thing that probe structurally could not:
the override as it actually sits in ``tt/model.py::_WatcherCleanSampling1D``,
called through ``Sampling1D.decode_forward`` with a caller-owned ``tt_out_tok``.

Three questions, in order of how badly a "no" would hurt:

1. **Buffer identity.** The traced decode loop binds one ``[1,1,1,32]`` uint32
   ROW_MAJOR tensor as *both* the sampler's output and the next step's token
   input. If ``_sample_argmax`` returns a new tensor instead of writing the one
   it was given, token feedback silently stops working and the model generates
   from a stale token forever. Checked as ``result is tt_out_tok`` **and**
   ``buffer_address()`` unchanged across the call, not assumed from the code.
2. **Same token as the base.** Both paths are run on the same logits through the
   same ``decode_forward`` entry point and compared slot by slot, against each
   other and against a host bf16 ``torch.argmax``.
3. **Faster, at the shipped shape, traced.**

Standalone: opens its own 1x4 mesh with synthetic logits at the shipped
``[1,1,32,37984]`` bf16/die. No 48-layer model -- the in-model proof is the
readiness runners and ``perf_full_model.py``. Writes its JSON beside itself;
nothing here touches ``doc/full_model/``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.model import _WatcherCleanSampling1D  # noqa: E402
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig  # noqa: E402
from models.common.modules.tt_ccl import TT_CCL  # noqa: E402

HERE = Path(__file__).resolve().parent
VOCAB = 151936
DEVICES = 4
LOCAL_VOCAB = VOCAB // DEVICES
SLOTS = 32
TOPOLOGY = ttnn.Topology.Ring


class _BaselineSampling1D(_WatcherCleanSampling1D):
    """Stage 05's shipped behaviour: the watcher-clean gather, base ``_sample_argmax``."""

    _sample_argmax = Sampling1D._sample_argmax


def build_sampler(cls, mesh, ccl):
    sampler = cls.from_config(
        Sampling1DConfig(
            vocab_size=VOCAB,
            valid_vocab_size=VOCAB,
            mesh_device=mesh,
            tt_ccl=ccl,
            max_batch_size=SLOTS,
            max_top_k=32,
            num_gather_links=1,
            sampling_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            allow_force_argmax=True,
            num_argmax_gather_links=1,
            ag_topology=TOPOLOGY,
            pad_to_power_of_2=False,
        )
    )
    sampler.load_device_buffers()
    return sampler


def traced_ms(mesh, fn, reps):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    for _ in range(5):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - t0) * 1e3)
    ttnn.release_trace(mesh, tid)
    return statistics.median(samples)


def read_tokens(tensor):
    return [int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).reshape(-1)[:SLOTS].tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=str, default=str(HERE / "wired_argmax_probe.json"))
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, DEVICES), trace_region_size=90_000_000)
    results = {}
    try:
        ccl = TT_CCL(mesh)
        torch.manual_seed(args.seed)
        host = torch.randn(1, 1, SLOTS, VOCAB) * 4.0
        expect_bf16 = [int(v) for v in host[0, 0].to(torch.bfloat16).float().argmax(dim=-1).tolist()]
        logits = ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        assert int(logits.shape[-1]) == LOCAL_VOCAB, logits.shape

        # The generator's decode token buffer, spelled exactly as tt/generator.py
        # allocates it: [1,1,1,32] uint32 ROW_MAJOR, replicated.
        def out_tok():
            return ttnn.from_torch(
                torch.zeros((1, 1, 1, SLOTS), dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        for name, cls in (
            ("baseline_force_argmax", _BaselineSampling1D),
            ("wired_distributed", _WatcherCleanSampling1D),
        ):
            sampler = build_sampler(cls, mesh, ccl)
            tok = out_tok()
            addr_before = tok.buffer_address()
            out, logprobs = sampler.decode_forward(logits, tt_out_tok=tok, enable_log_probs=False)
            ttnn.synchronize_device(mesh)
            tokens = read_tokens(tok)
            entry = {
                "returns_the_caller_buffer": out is tok,
                "buffer_address_unchanged": tok.buffer_address() == addr_before,
                "returned_dtype": str(out.dtype),
                "returned_layout": str(out.layout),
                "returned_shape": list(out.shape),
                "logprobs_is_none": logprobs is None,
                "matches_host_bf16": f"{sum(a == b for a, b in zip(tokens, expect_bf16))}/{SLOTS}",
                "tokens": tokens,
            }
            entry["ms"] = traced_ms(mesh, lambda s=sampler, t=tok: s.decode_forward(logits, tt_out_tok=t)[0], args.reps)
            results[name] = entry
            print(f"{name:<24} {json.dumps({k: v for k, v in entry.items() if k != 'tokens'})}", flush=True)

        base_tokens = results["baseline_force_argmax"]["tokens"]
        wired_tokens = results["wired_distributed"]["tokens"]
        results["agreement"] = {
            "wired_vs_baseline": f"{sum(a == b for a, b in zip(base_tokens, wired_tokens))}/{SLOTS}",
            "disagree_slots": [i for i in range(SLOTS) if base_tokens[i] != wired_tokens[i]],
        }
        results["speedup"] = results["baseline_force_argmax"]["ms"] / results["wired_distributed"]["ms"]
        print(json.dumps({k: v for k, v in results.items() if k in ("agreement", "speedup")}, indent=2), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
