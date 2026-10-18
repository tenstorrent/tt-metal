# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unselected native weighted expert-reduction experiments.

The native reader supports reduction dimension 1 directly, so the 80 MiB expert
activation stays tiled in DRAM. Only the 32 KiB routing tensor becomes row major.
Identity expert indices and a replicated all-local mapping are setup constants;
the native operation uses these to retain each slot's score, without sorting or
dispatching expert activations. The selected unified path's NaN masking remains
before this reduction.

Unlike the selected multiply-then-sum, the native FP32 hardware MAC does not pack
each product to BF16. Real-weight PCC and traced timing must qualify selection.
The adjacent AttnRes operation consumes tiled weights directly but allocates
three full candidate sets in L1. Its 512-expert call exceeds physical Blackhole
L1, so that candidate uses four 128-expert calls plus three output additions.
The slices introduce device traffic and BF16 partial-sum rounding; benchmark
evidence, not operation count, decides whether this adaptation is useful.

No device experiment has been run by the author of these host-only candidates.
"""

import hashlib
import inspect
import json
import textwrap
from pathlib import Path

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused

_BASE_MOE = fused.FusedMoE
_ORIGINAL_REDUCTION = "result = ttnn.sum(ttnn.multiply(down, ttnn.permute(routing, (0, 3, 2, 1))), dim=1, keepdim=True)"
_METHOD_HASHES = {}


def _replace_reduction(method):
    source = textwrap.dedent(inspect.getsource(method))
    if source.count(_ORIGINAL_REDUCTION) != 1:
        raise RuntimeError(f"{method.__qualname__} changed; reassess the reduction before testing")
    _METHOD_HASHES[method.__name__] = hashlib.sha256(source.encode()).hexdigest()
    replacement = source.replace(_ORIGINAL_REDUCTION, "result = self._weighted_reduce(down, routing)")
    namespace = dict(fused.__dict__)
    exec(compile(replacement, f"{__file__}:{method.__name__}", "exec"), namespace)
    return namespace[method.__name__]


class WeightedReduceMoE(_BASE_MOE):
    @classmethod
    def from_state_dict(cls, state, **kwargs):
        import torch

        mesh = kwargs["mesh_device"]
        if mesh.get_num_devices() != 1:
            raise ValueError("The weighted-reduction identity mapping is qualified only for a single device")
        obj = super().from_state_dict(state, **kwargs)
        if (obj.experts, obj.hidden) != (512, 2560):
            raise ValueError("Reassess native CB sizing before changing the E512/H2560 workload")
        indices = torch.arange(obj.experts, dtype=torch.int32).reshape(1, 1, 1, obj.experts)
        obj._reduce_indices = ttnn.from_torch(
            indices.expand(32, 1, 1, obj.experts).contiguous(),
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh,
        )
        obj._reduce_mapping = ttnn.from_torch(
            torch.zeros((1, obj.experts), dtype=torch.int32),
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh,
        )
        return obj

    def _weighted_reduce(self, down, routing):
        if tuple(down.shape) != (1, self.experts, 32, self.hidden):
            raise ValueError("Expected selected expert output [1,512,32,2560]")
        if tuple(routing.shape) != (1, 1, 32, self.experts):
            raise ValueError("Expected selected routing [1,1,32,512]")
        scores = ttnn.reshape(ttnn.to_layout(routing, ttnn.ROW_MAJOR_LAYOUT), (32, 1, 1, self.experts))
        outputs = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            down,
            self._reduce_indices,
            self._reduce_mapping,
            reduce_dim=1,
            split_size=self.hidden,
            cluster_axis=0,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            scores_tensor=scores,
            compute_kernel_config=self.compute,
        )
        if len(outputs) != 1:
            raise RuntimeError("The whole hidden dimension must remain one output tensor")
        return outputs[0]

    _group = _replace_reduction(_BASE_MOE._group)
    _group_unified = _replace_reduction(_BASE_MOE._group_unified)


class AttnWeightedReduceMoE(_BASE_MOE):
    """Four native E128 reductions; the direct E512 allocation cannot fit L1."""

    def _weighted_reduce(self, down, routing):
        if tuple(down.shape) != (1, self.experts, 32, self.hidden):
            raise ValueError("Expected selected expert output [1,E,32,H]")
        if tuple(routing.shape) != (1, 1, 32, self.experts):
            raise ValueError("Expected selected routing [1,1,32,E]")
        if (self.experts, self.hidden) != (512, 2560):
            raise ValueError("Reassess native CB sizing before changing the E512/H2560 workload")
        weights = ttnn.permute(routing, (0, 3, 2, 1))
        result = None
        for start in range(0, self.experts, 128):
            partial = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(
                down[:, start : start + 128, :, :],
                weights[:, start : start + 128, :, :],
                dim=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute,
            )
            result = partial if result is None else ttnn.add(result, partial)
        return result

    _group = _replace_reduction(_BASE_MOE._group)
    _group_unified = _replace_reduction(_BASE_MOE._group_unified)


def install(name="weighted_reduce"):
    candidates = {"weighted_reduce": WeightedReduceMoE, "attn_weighted_reduce": AttnWeightedReduceMoE}
    if name not in candidates:
        raise ValueError(f"Unknown weighted-reduction experiment {name!r}")
    fused.FusedMoE = candidates[name]
    return candidates[name]


def snapshot_provenance(directory):
    """Snapshot the candidate, source methods, and every native implementation file."""
    checkout = Path(__file__).resolve().parents[4]
    native = checkout / "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused"
    paths = [Path(__file__).resolve(), Path(fused.__file__).resolve()]
    paths.extend(sorted(path for path in native.rglob("*") if path.is_file()))
    attn_native = checkout / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_weighted_reduce_nc"
    paths.extend(sorted(path for path in attn_native.rglob("*") if path.is_file()))
    paths.append(
        checkout
        / "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/device/kernels/deepseek_moe_fast_reduce_nc_writer.cpp"
    )
    output = Path(directory) / "weighted_reduce_sources"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"original_method_sha256": dict(_METHOD_HASHES), "sources": {}}
    for index, path in enumerate(paths):
        data = path.read_bytes()
        snapshot = output / f"source_{index:02d}.txt"
        snapshot.write_bytes(data)
        manifest["sources"][str(path.relative_to(checkout))] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "snapshot": snapshot.name,
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
