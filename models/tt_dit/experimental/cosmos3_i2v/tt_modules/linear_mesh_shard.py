# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Mesh-sharded TTNN Linear replacement for tt-symbiote.

The stock `TTNNLinearLLamaBFloat16` calls `ttnn.to_device(host_tensor, mesh)`
without a mesh_mapper, so each chip ends up with a full copy of the weight.
For a 64B-param model that's ~128 GB per chip — way over BH's 32 GB DRAM.

This variant shards the weight along its output-feature dimension across one
axis of the mesh. With a (1, 32) or (4, 8) Blackhole Galaxy:

    weight shape (in_features, out_features) after preprocess
    sharded along dim=1 across cluster_axis=1
    per-chip footprint: in_features * (out_features / mesh_axis_size) * 2 bytes

For 64 B params total, sharded across 32 chips, that's 4 GB per chip.

**Limitations of this MVP:**
- forward() does NOT all-gather the sharded output. Running an actual forward
  through a model wrapped with this class will produce shape-mismatched
  activations downstream. The class exists to validate placement
  ("does the 64B fit on Galaxy?") not inference correctness.
- For inference correctness, add `ttnn.all_gather(out, dim=-1, cluster_axis=1)`
  in forward() and ensure the mesh fixture passes `line_params` (FABRIC_1D).
  See `models/tt_dit/layers/linear.py::ColParallelLinear` for the
  production pattern with persistent buffers + async CCL.
"""

from __future__ import annotations

from torch import nn

import ttnn
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.tt_dit.utils.matmul import get_matmul_core_grid


class TTNNLinearMeshShard(TTNNLinear):
    """Linear with weight sharded along output-features across mesh axis 1.

    Drop-in replacement for `nn.Linear` in a tt-symbiote `nn_to_ttnn` dict
    on a 2D mesh device. Sharding axis is fixed to mesh axis 1; weight is
    sharded across that axis with replication along mesh axis 0.

    For a (4, 8) mesh: each of 4 rows holds a full copy; within each row the
    8 chips hold disjoint 1/8 slices of the output features. 32-way total
    parallelism on memory is achievable by using a (1, 32) flat mesh.

    Weight precision is controlled by the `_weight_dtype` class attribute.
    Default is `ttnn.bfloat16`; use `make_sharded_linear_class(ttnn.bfloat8_b)`
    to get a subclass that quantizes weights to BFP8 at load time (halves
    DRAM + DRAM→L1 bandwidth, usually negligible PCC cost). Bias stays at
    bfloat16 — biases are tiny and quantization noise hurts proportionally.
    """

    # Override via make_sharded_linear_class() — putting the dtype on the
    # class (not the instance) lets tt-symbiote's `cls.from_torch(linear)`
    # construction path pick it up without plumbing extra kwargs through
    # register_module_replacement_dict.
    _weight_dtype = ttnn.bfloat16

    @classmethod
    def from_torch(cls, linear: nn.Linear):
        new_linear = cls(in_features=linear.in_features, out_features=linear.out_features)
        new_linear._fallback_torch_layer = linear
        return new_linear

    def preprocess_weights_impl(self):
        """Stage the torch weight in (in, out) layout for ttnn.from_torch."""
        if self.torch_layer is None:
            self._fallback_torch_layer = nn.Linear(self.in_features, self.out_features)
        # torch's nn.Linear stores weight as (out, in); ttnn expects (in, out)
        self._torch_weight_in_out = self.torch_layer.weight.t().contiguous()
        self._torch_bias = self.torch_layer.bias if self.torch_layer.bias is not None else None
        # Clear the unset attributes the parent expects so we don't crash
        # on later inspection.
        self.tt_weight_host = None
        self.tt_bias_host = None

    def move_weights_to_device_impl(self):
        """Shard the weight across mesh axis 1 (output-feature dim)."""
        mesh_shape = tuple(self.device.shape)
        if len(mesh_shape) != 2:
            raise RuntimeError(
                f"TTNNLinearMeshShard requires a 2D mesh, got shape {mesh_shape}. "
                "Use mesh_device fixture parameterized with a 2-tuple."
            )

        self.tt_weight = ttnn.from_torch(
            self._torch_weight_in_out,
            dtype=self._weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.device, dims=(None, 1), mesh_shape=mesh_shape),
        )
        # Free the host-side torch tensor — 27 shards worth otherwise.
        self._torch_weight_in_out = None

        if self._torch_bias is not None:
            self.tt_bias = ttnn.from_torch(
                self._torch_bias.reshape(1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.device, dims=(None, 1), mesh_shape=mesh_shape),
            )
            self._torch_bias = None
        else:
            self.tt_bias = None

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Sharded matmul; output stays sharded along the last dim.

        Per-chip math:
          input  : replicated full activation, shape (..., in_features)
          weight : sharded, shape (in_features, out_features / N_axis1)
          output : ttnn.linear(input, weight) → (..., out_features / N_axis1)
                   with TensorTopology placement = PlacementShard(dim=-1) on axis 1

        The output is INTENTIONALLY left sharded. When tt-symbiote's
        `to_torch_auto_compose` (models/common/auto_compose.py) converts the
        result back to torch — which it does at each ttnn→torch boundary —
        it sees `PlacementShard(dim=-1, dist_shape=8)` and concatenates the
        8 chip shards along the last dim into the full `(out_features)`
        tensor, exactly the shape a non-sharded Linear would have produced.

        If we instead called `ttnn.all_gather` here, each chip would hold a
        full copy but the topology metadata would still claim
        `PlacementShard(dim=-1, dist_shape=8)`, and the auto-composer would
        concatenate 8 full copies → 8x oversized output. The bug we hit on
        the full-transformer smoke test (shape [320, 40960] vs expected
        [320, 5120]) was exactly that. Round-tripping through host between
        Linears is the perf cost we accept for correctness; native all_gather
        on device requires also rewriting the tensor topology, which isn't
        straightforward through the simple ttnn API.
        """
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # If the input came from a previous sharded Linear in the same on-device
        # chain (no host roundtrip), it is sharded along the last dim and each
        # chip only holds in_features/N. The matmul below needs the full
        # in_features, so all_gather along mesh axis 1 to reassemble, then
        # mark the gathered tensor as Replicated.
        in_placements = list(input_tensor.tensor_topology().placements())
        if len(in_placements) >= 2 and isinstance(in_placements[1], ttnn.PlacementShard):
            input_tensor = ttnn.all_gather(
                input_tensor,
                dim=-1,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=1,
                topology=ttnn.Topology.Linear,
            )
            current = input_tensor.tensor_topology()
            input_tensor.update_tensor_topology(
                ttnn.TensorTopology(
                    current.distribution_shape(),
                    [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()],
                    current.mesh_coords(),
                )
            )

        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)
        input_tensor = ttnn.reshape(input_tensor, input_shape)

        # Per-chip matmul: (..., in_features) @ (in_features, out_features/N) → (..., out_features/N).
        # core_grid is clamped to 11x10 on BH Galaxy; without this the runtime can auto-pick the
        # full 12x10 and overdraw power from the PDU, tripping the tray.
        out_sharded = ttnn.linear(
            input_tensor,
            self.tt_weight,
            bias=self.tt_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=get_matmul_core_grid(self.device),
        )

        # ttnn.linear does NOT propagate the weight's sharding into the output's
        # TensorTopology metadata — the output inherits the input's (replicated)
        # placement, even though the actual data is sharded along the last dim.
        # Without this fix, `to_torch_auto_compose` reads the wrong metadata and
        # either returns a single chip's slice (1/N of true size) or duplicates
        # N times depending on what other ops have done. Explicitly rewrite the
        # topology so it reflects what the data actually is.
        current_topology = out_sharded.tensor_topology()
        corrected_topology = ttnn.TensorTopology(
            current_topology.distribution_shape(),
            [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)],
            current_topology.mesh_coords(),
        )
        out_sharded.update_tensor_topology(corrected_topology)

        per_chip_out_features = self.out_features // tuple(self.device.shape)[1]
        out_sharded = ttnn.reshape(out_sharded, input_tensor_shape[:-1] + [per_chip_out_features])
        return out_sharded


def make_sharded_linear_class(weight_dtype) -> type:
    """Return a `TTNNLinearMeshShard` subclass that loads weights at `weight_dtype`.

    Pass the returned class (not the base) into a tt-symbiote `nn_to_ttnn`
    replacement dict. The factory exists because `register_module_replacement_dict`
    calls `cls.from_torch(linear)` directly, so the dtype must live on the class
    rather than be passed per-instance.

    Common choices:
      - `ttnn.bfloat16` — full precision; identical to using `TTNNLinearMeshShard`
        directly. 16 GB/chip for the Cosmos3 64B trunk on a (4, 8) BH Galaxy.
      - `ttnn.bfloat8_b` — BFP8 block-float; halves per-chip DRAM (~8 GB/chip).
        Required to fit on WH LoudBox / T3K (12 GB per chip × 8 = 96 GB total).
      - `ttnn.bfloat4_b` — BFP4 block-float; quarter footprint but lossy. Apply
        selectively (FF1/FF3 only, never FF2 or output proj) — would need a
        per-tensor-group config rather than a single class.
    """
    suffix = str(weight_dtype).rsplit(".", 1)[-1]
    return type(
        f"TTNNLinearMeshShard_{suffix}",
        (TTNNLinearMeshShard,),
        {"_weight_dtype": weight_dtype},
    )
