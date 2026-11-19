# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated TP4 distributed RMSNorm boundary; production defaults are untouched.

Imports are deferred for source-only preparation. Each owner has two independent
native stats/semaphore sets. Call both phases in order, including inside traces.
"""


def geometry(batch):
    assert 1 <= batch <= 32
    return {
        "mesh": [1, 4],
        "logical_input": [1, batch, 1280],
        "native_input": [1, 1, batch, 1280],
        "physical_rows": 32,
        "global_hidden": 5120,
        "gamma_dtype": "float32",
        "activation_dtype": "bfloat16",
        "phases": 2,
        "native_stats_buffers": 2,
        "public_alignment_required": False,
    }


class DistributedNormBoundary:
    def __init__(self, model, example, gamma, *, key, projection_key, family):
        import ttnn
        from .multichip_decoder import _BlackholeCCLManager

        assert family in ("control", "dit")
        assert tuple(model.mesh.shape) == (1, 4) and model.hidden_width == 1280
        assert example.dtype == ttnn.bfloat16 and gamma.dtype == ttnn.float32
        assert model.policy.get("ccl_dtype") in (None, "bfloat16"), "This bounded probe preserves BF16 CCL activations"
        self.model, self.mesh, self.key, self.family = model, model.mesh, key, family
        self.projection_key, self.batch = projection_key, example.shape[1]
        geometry(self.batch)
        assert tuple(example.shape) == (1, self.batch, 1280)
        canonical_key = projection_key
        if projection_key == "packed_gdn" and model._packed_correction_candidate is not None:
            canonical_key = "packed_gdn_correction_joint"
        self.canonical_memory = model.projection_info[canonical_key]["input_mem"]
        self.gamma = gamma
        self.stats, self.sems = [], []
        if family == "dit":
            self.ccl = _BlackholeCCLManager(self.mesh, 1, ttnn.Topology.Linear)
            self.sems = [self.ccl.get_ag_ping_pong_semaphore() for _ in range(2)]
            x4 = ttnn.reshape(example, (1, 1, self.batch, 1280))
            self.stats = [
                ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(
                    x4,
                    1,
                    self.mesh,
                    num_heads_per_device=1,
                    per_head_norm=False,
                    num_links=1,
                    weight=gamma,
                    row_capacity=32,
                )
                for _ in range(2)
            ]
            assert all(t is not None for t in self.stats)
            self.ag = _BlackholeCCLManager(self.mesh, model.ccl.num_links, ttnn.Topology.Linear)
            self.ag_sems = [self.ag.get_ag_ping_pong_semaphore() for _ in range(2)]
            self.ag_barriers = [self.ag.get_barrier_semaphore() for _ in range(2)]
            self.ag_memory = (
                ttnn.DRAM_MEMORY_CONFIG if model.policy.get("ccl_decode_memory") == "DRAM" else ttnn.L1_MEMORY_CONFIG
            )
            self.ag_outputs = [
                ttnn.empty(
                    (1, 1, self.batch, 5120),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=self.ag_memory,
                )
                for _ in range(2)
            ]

    def prepare(self, x, phase):
        import ttnn

        assert phase in (0, 1)
        if self.family == "control":
            full = self.model._norm(x, self.key, decode=True)
        else:
            native_input = ttnn.reshape(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), (1, 1, self.batch, 1280))
            local = ttnn.experimental.dit_fused_distributed_rmsnorm(
                native_input,
                1,
                self.mesh,
                self.sems[phase],
                topology=ttnn.Topology.Linear,
                epsilon=self.model.config.rms_norm_eps,
                num_heads_per_device=1,
                per_head_norm=False,
                weight=self.gamma,
                dtype=ttnn.bfloat16,
                persistent_output_buffer=self.stats[phase],
                num_preferred_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.model.ckc,
                row_capacity=32,
            )
            local = ttnn.to_memory_config(local, self.ag_memory)
            gathered = ttnn.experimental.all_gather_async(
                local,
                dim=3,
                cluster_axis=1,
                mesh_device=self.mesh,
                topology=ttnn.Topology.Linear,
                num_links=self.ag.num_links,
                multi_device_global_semaphore=self.ag_sems[phase],
                barrier_semaphore=self.ag_barriers[phase],
                memory_config=self.ag_memory,
                persistent_output_tensor=self.ag_outputs[phase],
            )
            full = ttnn.reshape(gathered, (1, self.batch, 5120))
        return ttnn.to_memory_config(full, self.canonical_memory)

    def project(self, canonical):
        return self.model._linear(canonical, self.projection_key)

    def ownership(self):
        return {
            "family": self.family,
            "stats": [int(t.buffer_address()) for t in self.stats],
            "gamma": int(self.gamma.buffer_address()),
            "activation_ag": [int(t.buffer_address()) for t in getattr(self, "ag_outputs", [])],
            "canonical_memory": str(self.canonical_memory),
            "row_capacity": 32,
            "native_sem_sets": len(self.sems),
            "calls_per_trace": 2,
        }
