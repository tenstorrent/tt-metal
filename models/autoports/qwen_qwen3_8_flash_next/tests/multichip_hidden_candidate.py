# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in QSA whole-decoder hidden-residual experiment; no default changes.

The local public tensor stays [1,1,T,2560], packing four640-wide stream slices
within each rank. This differs semantically from the default full-stream shard.
Setup/readback adapters transpose host mesh ownership once outside measurement.
There is no conversion to the old stream ownership inside the layer. PLE/GDN
and mixed-kind --stack are intentionally rejected by this initial experiment.
"""

import argparse
import sys
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_multichip_fused_contract import ProjectionFamilies
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_multichip_residual_contract import HiddenResidual
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as m


def pack_host_residual(value):
    """Global [stream,hidden] -> [rank,stream,local_hidden], preserving token order."""
    n = value.shape[-2]
    return value.reshape(1, 1, n, 4, 4, 640).permute(0, 1, 2, 4, 3, 5).reshape(1, 1, n, 10240).contiguous()


def unpack_host_residual(value):
    # Both axes have length4, so the same transpose is its inverse.
    return pack_host_residual(value)


class PartitionedHiddenInjection(m.FusedPreciseProjection):
    def __init__(self, loader, host, splits):
        self.parts, self.k, self.compute = splits, 10240 // splits, loader.compute
        weight = torch.nn.functional.pad(host.T, (0, 28)).reshape(1, splits, self.k, 32)
        self.weight = loader.tensor(weight)

    def __call__(self, x):
        n = x.shape[-2]
        x = ttnn.permute(ttnn.reshape(x, (1, n, self.parts, self.k)), (0, 2, 1, 3))
        partials = ttnn.matmul(
            x, self.weight, dtype=ttnn.float32, compute_kernel_config=self.compute, memory_config=m.DRAM
        )
        return ttnn.typecast(ttnn.sum(partials, dim=1, keepdim=True), ttnn.bfloat16)


class PackedHiddenResidual(HiddenResidual):
    @staticmethod
    def unpack(value):
        return ttnn.permute(ttnn.reshape(value, (1, value.shape[-2], 4, 640)), (0, 2, 1, 3))

    @staticmethod
    def pack(value):
        return ttnn.reshape(ttnn.permute(value, (0, 2, 1, 3)), (1, 1, value.shape[-2], 2560))

    def read(self, residual):
        return super().read(self.unpack(residual))

    def write(self, residual, block, injection):
        if block.shape[-1] != 640:
            raise ValueError("Hidden residual producer must return its local640 columns")
        return self.pack(super().write(self.unpack(residual), block, injection))


class QSAOutput:
    """Same legal native/fused families as the component probe, logical1..32.

    Every supported output buffer shape is prepared at setup. Each logical shape
    has its own physical32-row arena; only selected-family storage survives setup.
    The extra ownership is deliberately reported as candidate memory overhead."""

    def __init__(self, mesh, loader, weight, family, fidelity):
        self.family = family
        args = SimpleNamespace(
            weight_dtype="bfloat8_b",
            activation_dtype="bfloat16",
            math_fidelity=fidelity,
            in0_block_w=4,
            rs_out_block_w=5,
            num_links=2,
        )
        self.producer = ProjectionFamilies(mesh, loader, weight, args)
        self.buffers = {}

        def empty(n, width, dtype):
            return ttnn.empty((1, 1, n, width), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=m.DRAM)

        for n in range(1, 33):
            if family == "fused_rs_fp32":
                self.buffers[n] = (empty(n, 2560, ttnn.float32), empty(n, 640, ttnn.float32))
            elif family in ("ag_mm_fp32", "fused_ag_mm_fp32"):
                self.buffers[n] = (empty(n, 6144, ttnn.bfloat16),)
        # No permanent dual projection representation: release the unused one.
        if family in ("ag_mm_fp32", "fused_ag_mm_fp32"):
            del self.producer.row_weight
        else:
            del self.producer.column_weight
        self.producer.rs_intermediate = None
        self.producer.rs_output = None
        self.producer.ag_output = None

    def __call__(self, value):
        n = value.shape[-2]
        if not 1 <= n <= 32:
            raise ValueError("QSA chunks require logical1..32")
        if self.family == "fused_rs_fp32":
            self.producer.rs_intermediate, self.producer.rs_output = self.buffers[n]
        elif self.family in ("ag_mm_fp32", "fused_ag_mm_fp32"):
            self.producer.ag_output = self.buffers[n][0]
        return self.producer(self.family, value)


class FinalReduceScatter:
    """Only the MoE's final partial sum; its router keeps the original CCL."""

    @staticmethod
    def sum(value):
        return ttnn.reduce_scatter(
            ttnn.typecast(value, ttnn.float32),
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Ring,
            memory_config=m.DRAM,
        )


class AlreadyReduced:
    @staticmethod
    def sum(value):
        return value


def apply(*, family="fused_ag_mm_fp32", fidelity="HiFi4", injection_splits=10):
    if family not in ("rs_fp32", "fused_rs_fp32", "ag_mm_fp32", "fused_ag_mm_fp32"):
        raise ValueError(family)
    if injection_splits and (10240 % injection_splits or (10240 // injection_splits) % 32):
        raise ValueError("Injection splitting must preserve tile-aligned K")
    original_factory = m.MultichipDecoder.from_state_dict.__func__

    def build(cls, state_dict, **kwargs):
        if kwargs["layer_idx"] != 3:
            raise ValueError("Initial hidden-residual experiment supports only representative QSA layer3")
        model = original_factory(cls, state_dict, **kwargs)
        prefix = "model.language_model.layers.3."
        weights = {
            key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)
        } or state_dict
        loader = m.MeshLoader(weights, kwargs["mesh_device"])
        for role, prefix in (("attn_mix", "attn_hyper_connection."), ("mlp_mix", "mlp_hyper_connection.")):
            residual = PackedHiddenResidual(loader, prefix, model.config, model.ccl)
            if injection_splits:
                residual.inject = PartitionedHiddenInjection(
                    loader, weights[prefix + "block_inject_weight.weight"], injection_splits
                )
            setattr(model, role, residual)
        qsa = model.chunked_attn
        projection = QSAOutput(model.mesh_device, loader, weights["self_attn.o_proj.weight"], family, fidelity)
        attention = qsa.attention
        old_linear = attention.linear
        # A non-tensor marker releases the original row-weight copy. The native
        # caller sees this only through the intercepted final output projection.
        output_marker = object()
        attention.out = output_marker

        def linear(value, weight, compute):
            return projection(value) if weight is output_marker else old_linear(value, weight, compute)

        attention.linear = linear
        qsa.ccl = AlreadyReduced()
        model.mlp.ccl = FinalReduceScatter()
        model.hidden_projection = projection
        model.residual_layout_contract = "rank owns640 hidden columns from each of4 streams, packed[1,1,T,2560]"
        return model

    m.MultichipDecoder.from_state_dict = classmethod(build)


def apply_harness_adapter():
    """Setup/readback-only ownership transpose for the existing QSA harness.

    The strict BF16 global10240 predicate selects reference residuals and trace
    refresh tensors, not FP32 GR weight tensors. QSA state has different shapes.
    The harness forbids component mode so internal block diagnostics are not
    mistaken for public packed-residual outputs."""
    original_from_torch, original_to_torch = ttnn.from_torch, ttnn.to_torch

    def upload(value, *args, **kwargs):
        if (
            isinstance(value, torch.Tensor)
            and value.dtype == torch.bfloat16
            and value.ndim == 4
            and tuple(value.shape)[:2] == (1, 1)
            and value.shape[-1] == 10240
            and kwargs.get("mesh_mapper") is not None
        ):
            value = pack_host_residual(value)
        return original_from_torch(value, *args, **kwargs)

    def read(value, *args, **kwargs):
        result = original_to_torch(value, *args, **kwargs)
        if (
            len(value.shape) == 4
            and tuple(value.shape)[:2] == (1, 1)
            and value.shape[-1] == 2560
            and kwargs.get("mesh_composer") is not None
            and result.shape[-1] == 10240
        ):
            result = unpack_host_residual(result)
        return result

    ttnn.from_torch, ttnn.to_torch = upload, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", default="fused_ag_mm_fp32", choices=("rs_fp32", "fused_rs_fp32", "ag_mm_fp32", "fused_ag_mm_fp32")
    )
    parser.add_argument("--family-fidelity", choices=("HiFi4", "HiFi2", "LoFi"), default="HiFi4")
    parser.add_argument("--base-candidate", default="default")
    parser.add_argument("--hidden-injection-splits", type=int, default=10)
    args, rest = parser.parse_known_args()
    if "--stack" in rest or "--components" in rest or "--mode" not in rest or rest[rest.index("--mode") + 1] != "mesh":
        parser.error("Initial adapter requires --mode mesh; mixed-kind --stack and --components are unsupported")
    if "--layer" not in rest or rest[rest.index("--layer") + 1] != "3":
        parser.error("Initial adapter requires --layer3")
    from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_optimization_candidates import apply as apply_base

    if args.base_candidate != "default":
        apply_base(args.base_candidate)
    apply(family=args.family, fidelity=args.family_fidelity, injection_splits=args.hidden_injection_splits)
    apply_harness_adapter()
    sys.argv = [sys.argv[0], *rest]
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_multichip_decoder import main as probe

    probe()


if __name__ == "__main__":
    main()
