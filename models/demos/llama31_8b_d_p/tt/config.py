# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`MeshConfig` — the parallelism decision and the three collective wrappers for Llama-3.1-8B prefill.

TP shards features along one mesh axis (default: columns); the other axis carries sequence-parallel
prefill, so **TP is the only knob** and SP is derived. Deployment target `(4, 8)`, TP=8 on the
columns, SP=4 on the rows (`bringup_log/00_MODEL_CARD.md` §4, `bringup_log/04_CCL_PLAN.md` §1).

**HF anchor:** none — this file holds no model math. It is the mesh half of the pair the repo has
converged on (`bringup_log/04_CCL_PLAN.md` §2): `MeshConfig` owns the parallelism decision and the
collective wrappers, `CCLManager` (`tt/ccl.py`) owns the persistent CCL resources. Modules call
`self.mesh_config.<collective>(t, self.ccl_manager, ...)` and never `ttnn.experimental.*` directly
— that is how semaphore-reuse bugs get in. One exception is logged: `DEC-028`.

**Template.** The **union** of the two in-repo copies, neither of which is a superset
(`bringup_log/03_OUTLINE.md` §2.1, `DEC-018`):

* `models/demos/minimax_m3/config.py:21` has `reduce_scatter:155`, but lacks the `sp` property and
  its `_validate` at `:42-45` lets sub-axis TP through with only a warning;
* `models/demos/gpt_oss_d_p/tt/config.py:19` has `sp:56` and the sub-axis refusal at `:44`, but has
  no `reduce_scatter`.

Deleted relative to both: `ep_axis`. Llama is dense — no MoE, no expert parallelism
(`bringup_log/00_MODEL_CARD.md` §3) — so the alias would be a name for a degree this model does not
have. `sp_axis` is derived directly.

Also here: `default_compute_kernel_config()`, the package's **one** compute-kernel config
(`DEC-030`). Recipe §2.4 requires an explicit config on every op that accepts one, and the danger
is not omitting the flag but inheriting a template's explicit `fp32_dest_acc_en=False`
(`models/demos/gpt_oss_d_p/tt/attention/config.py:71`), which costs 96x-1168x on a matmul. It lives
in this file rather than in `tt/model_config.py` because that file is P6.2's and this one is
already the package's config home. Same for `derive_head_dim()` (`DEC-032`): Llama's config has no
`head_dim` key, and `DEC-020` requires exactly one derivation in the package.
"""

from loguru import logger

import ttnn

# Recipe §2.4's measured winner for `ttnn.rms_norm` on this box: HiFi4 + `fp32_dest_acc_en=True`
# scores 0.9999971 against a 0.9999986 floor, where no config scores 0.9999652 and HiFi2 +
# `fp32_dest_acc_en=False` is marginally WORSE than no config at all (0.9999607). `MathFidelity`
# alone is a no-op; the flag is what removes ~25x of the error. `DEC-030`.
_DEFAULT_MATH_FIDELITY = ttnn.MathFidelity.HiFi4
_DEFAULT_MATH_APPROX_MODE = False
_DEFAULT_FP32_DEST_ACC_EN = True
_DEFAULT_PACKER_L1_ACC = False


def default_compute_kernel_config(mesh_device, *, fp32_dest_acc_en: bool = _DEFAULT_FP32_DEST_ACC_EN):
    """The package's compute-kernel config. Pass it to **every** op that accepts one (recipe §2.4).

    `fp32_dest_acc_en` is exposed as an argument for exactly two reasons, both required rather than
    convenient:

    * the in-suite A/B that turns §2.4's table into a measurement on this box rather than a
      quotation (`G-RMS`, `G-MLP`, `G-ATTN`; `DEC-014`'s falsifier);
    * the SP ring SDPA in P8, the one op in this model where `False` is mandatory rather than a
      preference (`BRINGUP_RECIPE.md:670-671`).

    Built through `ttnn.init_device_compute_kernel_config` rather than by naming a class:
    `ttnn.BlackholeComputeKernelConfig` **does not exist** — `ttnn/ttnn/__init__.py:305` exports
    only the Wormhole name and `ttnn/ttnn/types.py:61` shows they are the same object — so an
    "arch branch to pick the config class" would be a no-op (`BRINGUP_RECIPE.md:673-679`).
    """
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=_DEFAULT_MATH_FIDELITY,
        math_approx_mode=_DEFAULT_MATH_APPROX_MODE,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=_DEFAULT_PACKER_L1_ACC,
    )


# The deployment target (`bringup_log/00_MODEL_CARD.md` §4): (4,8) Blackhole Galaxy, TP=8 -> SP=4.
# TP=8 is an EQUALITY forced by the packed KV cache (one KV head per chip), not a bound — see
# `bringup_log/04_CCL_PLAN.md` §1.1. These two constants only *warn*; the refusal below is what
# rejects a shape that would produce a wrong tensor.
_VALIDATED_MESH_SHAPE = (4, 8)
_VALIDATED_TP = 8


class MeshConfig:
    """Prefill mesh parallelization. TP is the only knob; SP follows from the mesh shape."""

    def __init__(self, mesh_shape, tp, tp_axis: int = 1):
        """
        Args:
            mesh_shape: (rows, cols) — any mesh size.
            tp: tensor-parallel size; shards features along `tp_axis`.
            tp_axis: which mesh axis is TP (0=rows, 1=cols, default 1). The other axis carries
                sequence-parallel prefill (SP = size of that axis).
        """
        self.mesh_shape = tuple(mesh_shape)
        self.tp = tp
        self.tp_axis = tp_axis
        self.sp_axis = 0 if tp_axis == 1 else 1
        self.total_devices = self.mesh_shape[0] * self.mesh_shape[1]
        self._validate()

    def _validate(self):
        """Refuse sub-axis TP; warn when off the deployment target.

        `shard_mapper` always shards across the ENTIRE `tp_axis`, so a `tp` smaller than the axis
        builds head/feature counts from `tp` while the mapper still splits across all
        `tp_dim_size` devices — inconsistent per-device shapes, i.e. a wrong tensor. That is the one
        case that must raise (`models/demos/gpt_oss_d_p/tt/config.py:44`); being off
        `_VALIDATED_*` merely means untested, so it warns
        (`models/demos/minimax_m3/config.py:46-50`).

        `BRINGUP_RECIPE.md:1262-1264` states both "sub-axis TP ... **raises**" and "`MeshConfig`
        accepts any shape whose TP **divides** the column axis". 4 divides 8, so the two halves
        contradict each other; `bringup_log/03_OUTLINE.md` §5.1 takes the refusal as binding,
        because it is the half stated as a gate assertion and the half that prevents a wrong
        tensor.
        """
        tp_dim_size = self.mesh_shape[self.tp_axis]
        if self.tp != tp_dim_size:
            raise ValueError(
                f"TP({self.tp}) must equal mesh_{self.tp_axis}_size({tp_dim_size}); "
                f"sub-axis TP is unsupported (shard_mapper shards the full axis)."
            )
        if (self.mesh_shape, self.tp) != (_VALIDATED_MESH_SHAPE, _VALIDATED_TP):
            logger.warning(
                f"MeshConfig(mesh_shape={self.mesh_shape}, tp={self.tp}) is untested; only "
                f"mesh_shape={_VALIDATED_MESH_SHAPE}, tp={_VALIDATED_TP} (SP=4) is the "
                f"Llama-3.1-8B deployment target."
            )

    @property
    def sp(self) -> int:
        """Sequence-parallel degree (size of the non-TP axis)."""
        return self.mesh_shape[self.sp_axis]

    def shard_mapper(self, mesh_device, tensor_dim=None, mesh_dims=None):
        """Unified 2D shard mapper; `tensor_dim` shards that tensor dim along the TP axis only."""
        if mesh_dims is None:
            # Default: shard along the TP axis only. This is what makes every module SP-safe —
            # see `bringup_log/04_CCL_PLAN.md` §4 and the `mlp_2d.py` counter-example in
            # `bringup_log/02_SURVEY.md`.
            mesh_dims = (None, tensor_dim) if self.tp_axis == 1 else (tensor_dim, None)

        return ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=mesh_dims)

    def column_parallel(self, mesh_device):
        """Column-parallel weights: shard the output-feature dim (-1)."""
        return self.shard_mapper(mesh_device, tensor_dim=-1)

    def row_parallel(self, mesh_device):
        """Row-parallel weights: shard the input-feature dim (-2)."""
        return self.shard_mapper(mesh_device, tensor_dim=-2)

    def sequence_parallel(self, mesh_device):
        """Sequence sharding (-3), for the KV cache."""
        return self.shard_mapper(mesh_device, tensor_dim=-3)

    def shard_size(self, total_size):
        """Per-device size of a TP-sharded dimension. `4096/8 = 512`, `14336/8 = 1792`."""
        return total_size // self.tp

    def allreduce(self, tensor, ccl_manager, memory_config=None, pad_size=None, axis=0):
        """Tensor-parallel all-reduce, implemented as reduce-scatter + all-gather.

        Not `ttnn.experimental.all_reduce_async`: RS+AG is the repo's converged spelling
        (`models/demos/minimax_m3/config.py:77`, `models/demos/gpt_oss_d_p/tt/config.py:85`) and the
        two halves are what `bringup_log/04_CCL_PLAN.md` §5 enumerates as separate call sites.

        The caller checks whether communication is needed at all; at TP=1 there is nothing to do.
        """
        memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

        # Optional performance padding (the caller specifies it; no magic numbers here).
        padded = False
        if pad_size and tensor.shape[-2] >= 32:
            tensor_padded = ttnn.pad(tensor, [(0, 0), (0, 0), (0, 0), (0, pad_size)], 0)
            tensor.deallocate(True)
            tensor = tensor_padded
            padded = True

        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            dim=3,
            multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(),
            num_links=ccl_manager.num_links,
            memory_config=memory_config,
            topology=ccl_manager.topology,
            cluster_axis=axis,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )
        # Free the full-size input BEFORE the all-gather allocates its full-size output, so peak
        # live DRAM stays bounded under long-context prefill (`models/demos/minimax_m3/config.py:104-112`
        # records the OOM this prevents). Callers must NOT use `tensor` after this returns.
        tensor.deallocate(True)

        gathered = ttnn.experimental.all_gather_async(
            scattered,
            dim=3,
            cluster_axis=axis,
            mesh_device=ccl_manager.mesh_device,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=ccl_manager.num_links,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )
        scattered.deallocate(True)

        if padded:
            gathered_sliced = gathered[:, :, :, :-pad_size]
            gathered.deallocate(True)
            gathered = gathered_sliced
        return gathered

    def allgather(self, tensor, ccl_manager, memory_config=None, axis=0, dim=3, linear=False):
        """All-gather along mesh `axis`, concatenating on tensor `dim`.

        `linear=True` forces `Topology.Linear` for a single call without touching the manager's
        topology — the seam `models/demos/minimax_m3/config.py:114` uses.
        """
        memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

        return ttnn.experimental.all_gather_async(
            tensor,
            dim=dim,
            cluster_axis=axis,
            mesh_device=ccl_manager.mesh_device,
            topology=ttnn.Topology.Linear if linear else ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=ccl_manager.num_links,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )

    def reduce_scatter(self, tensor, ccl_manager, dim=3, axis=0, memory_config=None):
        """Reduce-scatter along mesh `axis`, scattering the sum on tensor `dim`.

        The scatter half of `allreduce`, exposed alone: it is what residual scheme B's module tails
        would close with (`DEC-025` takes scheme A for this iteration, and wires the
        `scatter_output` seam so B is a flag rather than a rewrite). Only
        `models/demos/minimax_m3/config.py:155` has this; the gpt-oss copy does not.
        """
        memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

        return ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            dim=dim,
            multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(),
            num_links=ccl_manager.num_links,
            memory_config=memory_config,
            topology=ccl_manager.topology,
            cluster_axis=axis,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )

    def __repr__(self):
        return f"MeshConfig({self.mesh_shape}, tp={self.tp}, sp={self.sp}, tp_axis={self.tp_axis})"


def derive_head_dim(hf) -> int:
    """`head_dim = hidden_size // num_attention_heads` — the package's ONE derivation (`DEC-032`).

    Llama's `config.json` has **no `head_dim` key** (`bringup_log/00_MODEL_CARD.md` §2), so the
    templates' `hf_config.head_dim` (`models/demos/gpt_oss_d_p/tt/model.py:64`) does not work here.
    `DEC-020` requires exactly one derivation in the package; it lives beside the other config
    helpers so `tt/rope.py` can keep the signature `bringup_log/03_OUTLINE.md` §2.5 gives it and
    P6.2's `ModelArgs` can expose the same value by calling this rather than re-deriving it.

    4096 // 32 = 128, which is tile-aligned.
    """
    hidden_size, num_heads = hf["hidden_size"], hf["num_attention_heads"]
    assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} is not divisible by {num_heads} heads"
    head_dim = hidden_size // num_heads
    assert head_dim % ttnn.TILE_SIZE == 0, f"head_dim {head_dim} is not tile-aligned"
    return head_dim
