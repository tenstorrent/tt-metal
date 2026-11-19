# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded per-decoder native normalization resources; default opt-in.

One decoder owns two phase resources for every logical B1..32. Device storage
does not grow with warmed batches. Input/post are the native ping/pong phases.
Only complete ordered decoder calls on CQ0 are supported; no runtime fallback.
"""

import math

KEYS = ("input_layernorm.weight", "post_attention_layernorm.weight")
FAMILIES = ("off", "dit", "minimal")
STATS_PAGE_LIMIT = 65536  # Explicit bound, checked against the native allocation.


def capacity(family, *, activation_dram=True):
    """Conservative persistent bytes/rank; native DiT page size is runtime-derived."""
    if family not in FAMILIES:
        raise ValueError(family)
    if family == "off":
        return {"dram_bytes_bound": 0, "l1_tensor_bytes": 0, "global_semaphores": 0}
    ag = 2 * 32 * 5120 * 2
    if family == "dit":
        # Four logical RM pages rounded to eight DRAM banks, twice.
        dram, l1, sems = 2 * 8 * STATS_PAGE_LIMIT, 0, 10
    else:
        dram, l1, sems = 2 * 1280 * 4, 2 * (32 * 1280 * 2 + 32 * 128 * 4), 8
    return {
        "dram_bytes_bound": dram + (ag if activation_dram else 0),
        "l1_tensor_bytes": l1 + (0 if activation_dram else ag),
        "global_semaphores": sems,
    }


def projection_keys(model):
    first = "packed_gdn_correction_joint" if model._packed_correction_candidate is not None else "packed_gdn"
    if model.kind == "full_attention":
        first = "packed_qkvg" if model.policy.get("packed_qkvg") else "packed_qkv"
    return (first, "mlp.gate_up.weight")


class FusedDecodeNormOwner:
    def __init__(self, model, *, layer_idx):
        import ttnn

        self.family = model.policy.get("distributed_decode_norm", "off")
        assert self.family in ("dit", "minimal")
        assert tuple(model.mesh.shape) == (1, 4) and model.hidden_width == 1280
        assert model.policy.get("dram_sharded") and model.policy.get("flat_decode")
        assert model.policy.get("packed_mlp")
        assert model.policy.get("minimal_swiglu") is not True, (
            "Fused decode norm canonical post layout requires packed MLP decode; "
            'minimal_swiglu=True consumes DRAM instead. Use minimal_swiglu="prefill" or False.'
        )
        assert model.policy.get("ccl_dtype") in (None, "bfloat16")
        assert isinstance(layer_idx, int) and 0 <= layer_idx < 64
        self.owner_id, self.mesh, self.layer_idx = id(model), model.mesh, layer_idx
        self.next_phase, self.open_batch = 0, None
        self.projections = projection_keys(model)
        self.canonical = tuple(model.projection_info[k]["input_mem"] for k in self.projections)
        self.gamma = tuple(model.shard_norms[k] for k in KEYS)
        assert all(t.dtype == ttnn.float32 and tuple(t.shape) == (1, 1, 1280) for t in self.gamma)
        grid = self.mesh.compute_with_storage_grid_size()
        full = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        create_sem = lambda cores: ttnn.create_global_semaphore(self.mesh, cores, 0)
        self.ag_sems = [[create_sem(full) for _ in range(2)] for _ in range(2)]
        self.ag_barriers = [create_sem(full) for _ in range(2)]
        self.links = model.ccl.num_links
        self.ag_memory = (
            ttnn.DRAM_MEMORY_CONFIG if model.policy.get("ccl_decode_memory") == "DRAM" else ttnn.L1_MEMORY_CONFIG
        )
        self.ag_outputs = [
            ttnn.empty(
                (1, 1, 32, 5120),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=self.ag_memory,
            )
            for _ in range(2)
        ]
        self.norm_outputs = []
        if self.family == "dit":
            self.norm_sems = [[create_sem(full) for _ in range(2)] for _ in range(2)]
            example = ttnn.empty(
                (1, 1, 32, 1280),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.stats = [
                ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(
                    example,
                    1,
                    self.mesh,
                    num_heads_per_device=1,
                    per_head_norm=False,
                    num_links=1,
                    weight=self.gamma[p],
                    row_capacity=32,
                )
                for p in range(2)
            ]
            for t in self.stats:
                assert t is not None and t.dtype == ttnn.float32 and t.layout == ttnn.ROW_MAJOR_LAYOUT
                assert tuple(t.shape)[:3] == (1, 1, 4)
                assert t.shape[-1] * 4 <= STATS_PAGE_LIMIT
            ttnn.deallocate(example)
        else:
            assert grid.x >= 8
            # Independent L1 owners cannot all occupy row0: 64*two FP32
            # stats alone would exceed sender-core L1. Spread layer ownership.
            self.worker_row = layer_idx % grid.y
            cores = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, self.worker_row), ttnn.CoreCoord(7, self.worker_row))}
            )
            stats_cores = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, self.worker_row), ttnn.CoreCoord(0, self.worker_row))}
            )
            self.input_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, (32, 160), ttnn.ShardOrientation.ROW_MAJOR),
            )
            stats_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(stats_cores, (32, 128), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(8, 1), subblock_w=1, block_h=1, block_w=5, inplace=False
            )
            self.norm_sems = [[create_sem(cores)] for _ in range(2)]
            self.stats = [
                ttnn.empty(
                    (1, 1, 32, 128),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=stats_memory,
                )
                for _ in range(2)
            ]
            self.norm_outputs = [
                ttnn.empty(
                    (1, 1, 32, 1280),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=self.input_memory,
                )
                for _ in range(2)
            ]
            # Setup only: preserve all actual FP32 affine values, no host readback.
            self.gamma = tuple(
                ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 40, 32)) for t in self.gamma
            )
        # These are host-only views of exactly the same physical tile storage.
        # Precreate every supported extent so capture never creates a new slot.
        self.ag_views = self._views(self.ag_outputs, 5120)
        self.norm_views = self._views(self.norm_outputs, 1280) if self.norm_outputs else {}
        self.warmed_batches = set()
        # Complete mesh-coherent initial semaphore stores before peer traffic.
        ttnn.synchronize_device(self.mesh)

    @staticmethod
    def _views(tensors, width):
        import ttnn

        result = {}
        for b in range(1, 33):
            result[b] = tuple(
                ttnn.reshape(t, ttnn.Shape((1, 1, b, width)), ttnn.Shape((1, 1, 32, width))) for t in tensors
            )
            for base, view in zip(tensors, result[b]):
                assert view.buffer_address() == base.buffer_address(), "Logical batch view allocated storage"
                assert tuple(view.padded_shape) == (1, 1, 32, width)
                assert view.memory_config() == base.memory_config()
                assert [t.buffer_address() for t in ttnn.get_device_tensors(view)] == [
                    t.buffer_address() for t in ttnn.get_device_tensors(base)
                ]
        return result

    def normalize(self, model, x, key):
        import ttnn

        assert id(model) == self.owner_id, "Shallow decoder clones require a new norm owner"
        assert key in KEYS and self.family == model.policy.get("distributed_decode_norm")
        phase = KEYS.index(key)
        assert phase == self.next_phase, "Fused norm requires complete input/post phase pairs"
        assert tuple(x.shape)[0] == 1 and x.shape[-1] == 1280 and 1 <= x.shape[1] <= 32
        assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT
        batch = x.shape[1]
        assert phase == 0 or batch == self.open_batch
        # Latch before dispatch: a failed partial normalization pair remains unusable; no
        # automatic reset could prove remote terminal traffic has drained.
        self.next_phase = -1
        x4 = ttnn.reshape(x, (1, 1, batch, 1280))
        if self.family == "dit":
            x4 = ttnn.to_memory_config(x4, ttnn.DRAM_MEMORY_CONFIG)
            local = ttnn.experimental.dit_fused_distributed_rmsnorm(
                x4,
                1,
                self.mesh,
                self.norm_sems[phase],
                topology=ttnn.Topology.Linear,
                epsilon=model.config.rms_norm_eps,
                num_heads_per_device=1,
                per_head_norm=False,
                weight=self.gamma[phase],
                dtype=ttnn.bfloat16,
                persistent_output_buffer=self.stats[phase],
                num_preferred_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=model.ckc,
                row_capacity=32,
            )
        else:
            x4 = ttnn.to_memory_config(x4, self.input_memory)
            local = ttnn.fused_rms_minimal(
                x4,
                self.program,
                1,
                self.mesh,
                self.norm_sems[phase][0],
                persistent_output_tensor=self.norm_views[batch][phase],
                num_links=1,
                topology=ttnn.Topology.Linear,
                dtype=ttnn.bfloat16,
                compute_kernel_config=model.ckc,
                memory_config=self.input_memory,
                epsilon=model.config.rms_norm_eps,
                weight=self.gamma[phase],
                stats=self.stats[phase],
            )
        local = ttnn.to_memory_config(local, self.ag_memory)
        gathered = ttnn.experimental.all_gather_async(
            local,
            dim=3,
            cluster_axis=1,
            mesh_device=self.mesh,
            topology=ttnn.Topology.Linear,
            num_links=self.links,
            multi_device_global_semaphore=self.ag_sems[phase],
            barrier_semaphore=self.ag_barriers[phase],
            memory_config=self.ag_memory,
            persistent_output_tensor=self.ag_views[batch][phase],
        )
        canonical = ttnn.to_memory_config(ttnn.reshape(gathered, (1, batch, 5120)), self.canonical[phase])
        self.next_phase, self.open_batch = (1, batch) if phase == 0 else (0, None)
        if phase == 1:
            self.warmed_batches.add(int(batch))
        return canonical

    def ownership(self):
        import ttnn

        assert self.next_phase == 0 and self.open_batch is None, "Incomplete norm pair"
        sems = [s for group in self.norm_sems + self.ag_sems for s in group] + self.ag_barriers
        tensors = self.stats + self.ag_outputs + self.norm_outputs
        if self.family == "minimal":
            tensors += list(self.gamma)
        records = []
        for t in tensors:
            shape = tuple(int(v) for v in t.padded_shape)
            records.append(
                {
                    "address": int(t.buffer_address()),
                    "shape": list(t.shape),
                    "rank_addresses": [int(s.buffer_address()) for s in ttnn.get_device_tensors(t)],
                    "padded_shape": list(shape),
                    "dtype": str(t.dtype),
                    "layout": str(t.layout),
                    "memory": str(t.memory_config()),
                    "logical_bytes": math.prod(t.shape) * (4 if t.dtype == ttnn.float32 else 2),
                }
            )
        return {
            "family": self.family,
            "phase_keys": list(KEYS),
            "capacity_rows": 32,
            "layer_idx": self.layer_idx,
            "worker_row": getattr(self, "worker_row", None),
            "batch_signature_device_allocations": 1,
            "projection_keys": list(self.projections),
            "semaphores": [int(ttnn.get_global_semaphore_address(s)) for s in sems],
            "persistent_tensors": records,
            "persistent_addresses": [r["address"] for r in records],
            "warmed_batches": sorted(self.warmed_batches),
            "metadata_views_share_backing": True,
            "capacity_bound": capacity(self.family, activation_dram=self.ag_memory == ttnn.DRAM_MEMORY_CONFIG),
        }


def install(model, *, layer_idx):
    """Setup-only, including shallow-clone setup; never call from a forward."""
    family = model.policy.get("distributed_decode_norm", "off")
    if family not in FAMILIES:
        raise ValueError(family)
    model._fused_decode_norm = FusedDecodeNormOwner(model, layer_idx=layer_idx) if family != "off" else None
