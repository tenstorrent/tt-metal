# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3_70b_galaxy.tt.llama_ccl import tt_distributed_rmsnorm, tt_sharded_distributed_rmsnorm


class DistributedNorm(LightweightModule):
    """Wraps an inner RMSNorm and runs a distributed (col-sharded) RMSNorm forward.

    Parameters
    ----------
    norm : RMSNorm
        The wrapped per-device RMSNorm.  Must expose ``.weight`` and, when
        used in distributed mode, ``.weight_distributed``.
    args : ModelArgs
    tt_ccl : optional CCL handle.
    ccl_topology : optional topology hint.
    zero_centered : bool, default False
        When ``True``, bake ``w' = w + 1`` into the on-device weight tensors
        (``norm.weight`` and ``norm.weight_distributed`` if present).  This
        implements the HF ``Qwen3NextRMSNorm`` convention
        ``output = (1 + w) * normalize(x)`` without any extra on-device math
        in the forward path.  The transform is performed at construction
        time and is lossless (we never need the un-transformed weight back).
    """

    def __init__(self, norm, args, tt_ccl=None, ccl_topology=None, zero_centered: bool = False):
        self.norm = norm
        self.args = args
        self.tt_ccl = tt_ccl
        self.ccl_topology = ccl_topology
        self.zero_centered = zero_centered

        if zero_centered:
            self._bake_zero_centered_offset()
        if args.qk_norm:
            core_grid_ln, grid_offset = (5, 2), ttnn.CoreCoord(1, 0)
        else:
            core_grid_ln, grid_offset = (8, 2), ttnn.CoreCoord(2, 0)
        core_range = ttnn.CoreRange(
            grid_offset, ttnn.CoreCoord(core_grid_ln[1] + grid_offset.x - 1, core_grid_ln[0] + grid_offset.y - 1)
        )
        num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
        hidden_size_per_device_distributed_ln = args.dim // 4
        self.gather_in_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(1, 1, 32, hidden_size_per_device_distributed_ln // num_cores_ln),
            core_grid=ttnn.CoreRangeSet(
                {
                    core_range,
                }
            ),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        self.ln_prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(core_grid_ln[1], core_grid_ln[0]),
            subblock_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
            block_h=1,
            block_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
            inplace=False,
        )
        self.ln_sharded_stats_memcfg = None
        # self.ln_sharded_stats_memcfg = ttnn.create_sharded_memory_config(
        #     shape=[1, 1, 32, 32 * 4],
        #     core_grid=ttnn.CoreGrid(y=1, x=1),
        #     strategy=ttnn.ShardStrategy.WIDTH,
        # )
        # ttnn.create_sharded_memory_config(
        #     shape=[1, 1, 32, 32 * 4],
        #     core_grid=ttnn.CoreGrid(y=1, x=1),
        #     strategy=ttnn.ShardStrategy.WIDTH,
        # )
        # V2-decode-64L: use HiFi4 + fp32 dest accumulation to mirror v1's
        # DistributedNorm (models/demos/qwen3_6_galaxy/tt/distributed_norm.py
        # line ~138).  The default (HiFi2 + bf16 dest) accumulates the
        # sum-of-squares in bf16, losing precision on small-magnitude
        # activations; with 128 norm calls per single decode pass at 64L the
        # noise compounds and drops logits PCC from 0.9996 (4L) → 0.16 (64L).
        self.ln_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _bake_zero_centered_offset(self):
        """Bake the ``w' = w + 1`` transform into the on-device norm weight(s).

        Reads back each weight tensor on ``self.norm`` (``weight`` and, if
        present, ``weight_distributed``) to host, adds 1.0 in float, and
        re-uploads via ``ttnn.from_torch`` preserving dtype / layout /
        memory_config / mesh-mapper.  This is a one-shot transform performed
        at construction time so the forward path is identical to a standard
        RMSNorm — the +1 is fused into the stored weight.

        Lossless: we never need the un-transformed weight back.
        """
        mesh_device = self.args.mesh_device
        mesh_shape = list(mesh_device.shape)

        for attr, mesh_mapper in (
            ("weight", ttnn.ReplicateTensorToMesh(mesh_device)),
            (
                "weight_distributed",
                ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=mesh_shape),
            ),
        ):
            tt_w = getattr(self.norm, attr, None)
            if tt_w is None:
                continue

            dtype = tt_w.dtype
            layout = tt_w.layout
            memory_config = tt_w.memory_config()

            # Read the (replicated/sharded) on-device tensor back to host.  We
            # rely on the same mesh-mapper convention the inner RMSNorm used:
            # ``weight`` is replicated (any shard is fine), ``weight_distributed``
            # is column-sharded along dim 2.
            if attr == "weight":
                torch_w = ttnn.to_torch(tt_w, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
                # All shards are replicas — keep the first one.
                torch_w = torch_w[: torch_w.shape[0] // (mesh_shape[0] * mesh_shape[1])]
            else:
                # Concat column shards back into the full weight along dim 2.
                torch_w = ttnn.to_torch(
                    tt_w,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 2), mesh_shape=mesh_shape),
                )
                # ConcatMesh2dToTensor stacks the row-axis along dim 0 → take
                # the first row's worth of data (rows are replicated for norm).
                torch_w = torch_w[: torch_w.shape[0] // mesh_shape[0]]

            # Bake +1 in float32 to avoid bf16 rounding compounding.
            torch_w = torch_w.float() + 1.0

            new_tt_w = ttnn.from_torch(
                torch_w,
                device=mesh_device,
                dtype=dtype,
                layout=layout,
                memory_config=memory_config,
                mesh_mapper=mesh_mapper,
            )

            # Free the old on-device buffer and swap in the +1-baked weight.
            ttnn.deallocate(tt_w)
            setattr(self.norm, attr, new_tt_w)

    def forward(self, x, res, mode):
        """Apply a norm, possibly gathering inputs if required."""
        if mode == "decode":
            # fused_rms_minimal (the decode-mode sharded RMSNorm) fuses a residual
            # add and dereferences residual_input_tensor unconditionally — passing
            # None raises "bad optional access". qwen3.6 decode does its residual
            # add separately (unfused), so feed a zero residual: rmsnorm(x+0)=rmsnorm(x).
            if res is None:
                res = ttnn.zeros_like(x)
            # B1: the decode-mode fused RMSNorm (fused_rms_minimal) reads
            # ``tt_ccl.all_gather_buffers["LAYERNORM"]`` as its ``stats`` tensor. qwen3.6's
            # decode has always used the PREFILL DRAM norm, so the tt_ccl was only populated
            # with the prefill per-seqlen all-gather buffers (keys 4096/2048/.../32) and never
            # the decode "LAYERNORM" stats buffer → stats=None → "bad optional access". Build
            # + register it lazily on first decode-norm call (semaphores already exist). Shape
            # (1,1,32,128) WIDTH-sharded on core (1,0), replicated — mirrors get_all_gather_buffers.
            if "LAYERNORM" not in self.tt_ccl.all_gather_buffers:
                import torch as _torch

                _M = x.shape[-2]
                _go = ttnn.CoreCoord(1, 0)
                _stats_cfg = ttnn.create_sharded_memory_config(
                    shape=(_M, 128),
                    core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(_go, _go)]),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.tt_ccl.all_gather_buffers["LAYERNORM"] = ttnn.from_torch(
                    _torch.zeros((1, 1, _M, 128)),
                    device=self.tt_ccl.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=_stats_cfg,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.tt_ccl.mesh_device),
                )
            return tt_sharded_distributed_rmsnorm(
                x,
                res,
                epsilon=self.norm.eps,
                gamma=self.norm.weight_distributed,
                mesh_device=self.args.mesh_device,
                ln_sharded_input_memcfg=self.gather_in_mem_cfg,
                ln_sharded_progcfg=self.ln_prg_cfg,
                ln_sharded_stats_memcfg=self.ln_sharded_stats_memcfg,
                tt_ccl=self.tt_ccl,
                output_mem_config=self.norm.output_mem_config,
                ccl_topology=self.ccl_topology,
            )
        else:
            return tt_distributed_rmsnorm(
                x,
                epsilon=self.norm.eps,
                gamma=self.norm.weight_distributed,
                mesh_device=self.args.mesh_device,
                compute_kernel_config=self.ln_cfg,
                tt_ccl=self.tt_ccl,
            )
