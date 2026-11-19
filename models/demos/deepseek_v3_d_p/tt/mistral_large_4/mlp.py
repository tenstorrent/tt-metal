# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Dense SwiGLU MLP, TP over the intermediate (plan.md attn_dense "mlp"; components.yaml notes).

Adapted from models/demos/deepseek_v3_d_p/tt/tt_ffn.py (TtFfn.forward). Per chip (SP row r of 8, TP column t of 4):

    x      [1, 1, 640, 1536] ffn_norm -> all_gather(axis 1) -> x_full [1, 1, 640, 6144]
    gate   x_full @ w1 (this column's quarter of the intermediate, column-parallel) [1, 1, 640, I/4]; up likewise w3
    h      silu(gate) * up (silu fused into the multiply)
    down   h @ w2 (the rows of this column's intermediate, row-parallel) -> partial [1, 1, 640, 6144] in
           MLP_PARTIAL_DTYPE -> reduce_scatter(axis 1) -> bf16 [1, 1, 640, 1536]
"""

import torch

import ttnn

from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import mm_configs, tp_gather
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import TP_AXIS
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings


def _compute_config(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, settings.get("MLP_MATH_FIDELITY")),
        math_approx_mode=False,
        fp32_dest_acc_en=settings.get("MLP_FP32_ACC"),
        packer_l1_acc=settings.get("MM_L1_ACC"),
    )


def _topology():
    return getattr(ttnn.Topology, settings.get("MLP_TOPOLOGY"))


class TtMlp:
    # read when they run, not when built: a sweep can change them between runs of one loaded model (PERF.md)
    @property
    def cfg(self):
        return _compute_config(self.mesh)

    @property
    def mid_dtype(self):
        return getattr(ttnn, settings.get("MLP_INTERMEDIATE_DTYPE"))

    @property
    def partial_dtype(self):
        return getattr(ttnn, settings.get("MLP_PARTIAL_DTYPE"))

    def __init__(self, mesh, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor):
        """w1, w3: [I, H]; w2: [H, I] (checkpoint order, host, any float dtype)."""
        self.mesh = mesh
        inter = w1.shape[0]
        assert inter % (ttnn.TILE_SIZE * mesh.shape[TP_AXIS]) == 0, f"intermediate {inter} does not split over TP"

        def put(t, dim):
            dims = [None, None]
            dims[TP_AXIS] = dim
            return ttnn.from_torch(
                t.float()[None, None].contiguous(),
                device=mesh,
                dtype=ttnn.bfloat8_b,  # rule 7: weights bfp8
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
            )

        # Kept [N, K] (checkpoint order) and read with transpose_b, so a bfp8 block runs along the contraction
        # (tt/attention.py: [K, N] blocks shared an exponent across output columns).
        self.w1 = put(w1, 2)  # [I/4, H]: this column's intermediate
        self.w3 = put(w3, 2)
        self.w2 = put(w2, 3)  # [H, I/4]: the same intermediate columns

    def _linear(self, x, w, dtype, name=None, sub=None):
        kw = {}
        if sub is not None:  # MOE_OVERLAP: on the shared expert's sub-device, its grid
            kw = dict(sub_device_id=sub.shared_id)
        return ttnn.matmul(
            x,
            w,
            transpose_b=True,
            dtype=dtype,
            program_config=mm_configs.get(name, x, w, transpose_b=True, grid=sub.shared_grid if sub else None),
            compute_kernel_config=self.cfg,
            memory_config=mm_configs.out_memory(name) if name == "mlp_down" else ttnn.DRAM_MEMORY_CONFIG,
            **kw,
        )

    def partial(self, x: ttnn.Tensor, sub=None) -> ttnn.Tensor:
        """x: ffn_norm [1, 1, S/8, H/4] bf16 per chip -> this column's partial [1, 1, S/8, H] in MLP_PARTIAL_DTYPE.
        sub (tt/sub_device.Overlap, MOE_OVERLAP): every op on the shared expert's sub-device (no collective: x's TP
        gather is the one the router and the experts already made)."""
        x_full = tp_gather.gather(x, _topology(), self.w1.shape[-1])
        gate = self._linear(x_full, self.w1, self.mid_dtype, "mlp_gate", sub)
        up = self._linear(x_full, self.w3, self.mid_dtype, "mlp_gate", sub)
        tp_gather.done(x_full, x)
        kw = dict(sub_core_grids=sub.shared_cores) if sub is not None else {}
        h = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=self.mid_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **kw,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        # the 4 TP partials are summed in MLP_PARTIAL_DTYPE: a bf16 reduce over TP biases the sum (known_issues)
        part = self._linear(h, self.w2, self.partial_dtype, "mlp_down", sub)
        ttnn.deallocate(h)
        return part

    def reduce(self, part: ttnn.Tensor, topology=None) -> ttnn.Tensor:
        """Sums the 4 TP partials (reduce_scatter over TP) -> [1, 1, S/8, H/4], bf16 (fp32 with RESIDUAL_FP32_IN)."""
        y = ttnn.reduce_scatter(
            part, dim=3, cluster_axis=TP_AXIS, num_links=settings.get("NUM_LINKS"), topology=topology or _topology()
        )
        ttnn.deallocate(part)
        if y.dtype != ttnn.bfloat16 and not settings.get("RESIDUAL_FP32_IN"):
            yb = ttnn.typecast(y, ttnn.bfloat16)
            ttnn.deallocate(y)
            y = yb
        return y

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: ffn_norm [1, 1, S/8, H/4] bf16 per chip -> mlp_out [1, 1, S/8, H/4] bf16 per chip."""
        return self.reduce(self.partial(x))
