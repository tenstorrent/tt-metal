# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated public-native GDN recurrence family; never registered as a default.

One owner per layer/state/CQ. Persistent FP32 T1 rings survive every decode;
all normalization, grouped-head expansion and output packing are device ops.
Imports and allocation are deferred until explicitly constructed by a probe.
"""


def wave_plan(batch, available_cores=110):
    assert 1 <= batch <= 32 and available_cores >= 12
    users = min(9, available_cores // 12)
    return [(start, min(batch, start + users)) for start in range(0, batch, users)]


def memory_estimate(batch, available_cores=110):
    waves = wave_plan(batch, available_cores)
    # One ROW_MAJOR index page, aligned to64B then spread over8DRAM banks.
    index_bytes = sum(8 * (((end - start) * 12 * 4 + 63) // 64) * 64 for start, end in waves)
    return {
        "state_fp32_bytes_per_rank": batch * 12 * 128 * 128 * 4,
        "state_bf16_control_bytes_per_rank": batch * 12 * 128 * 128 * 2,
        "state_increment_bytes_per_rank": batch * 12 * 128 * 128 * 2,
        "state_increment_all_48_layers_per_rank": batch * 12 * 128 * 128 * 2 * 48,
        "index_estimated_bytes_per_rank": index_bytes,
        "native_prepared_qkv_decay_beta_output_payload_bytes": batch * 12 * 18 * 4096,
        "compact_output_payload_bytes": ((batch * 12 + 31) // 32) * 4 * 4096,
        "persistent_compact_output_bytes_per_rank": ((batch * 12 + 31) // 32) * 4 * 4096,
        "native_cb_bytes_per_active_core": 130 * 4096,
        "limitations": [
            "Preparation/output allocations and trace-pinned temporaries need measured peaks.",
            "Index bank-spread estimate excludes allocator metadata and extra mapper padding.",
            "Probe reset snapshots and three controls are additional test-only resident memory.",
        ],
    }


class NativeRecurrentCandidate:
    """Public native op with setup-owned FP32 per-wave state and no restore copy.

    ``initial_state`` is FP32 TILE DRAM [B,12,128,128]. Constructor copies it;
    caller's initial state remains immutable. ``step`` accepts packed BF16
    ROW_MAJOR [B,1,2560], log-decay g and post-sigmoid beta [B,1,12].
    Returned FP32 [1,B*12,128] matches the custom recurrence epilogue contract.
    State is kept as wave-local rings. Production integration of public state
    access is deliberately absent; the probe inspects the wave tensors directly.
    """

    def __init__(self, initial_state):
        import torch
        import ttnn

        self.mesh = initial_state.device()
        assert tuple(self.mesh.shape) == (1, 4), "This probe requires the actual TP4 mesh"
        self.batch = initial_state.shape[0]
        assert tuple(initial_state.shape) == (self.batch, 12, 128, 128)
        assert initial_state.dtype == ttnn.float32 and initial_state.layout == ttnn.TILE_LAYOUT
        assert initial_state.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        grid = self.mesh.compute_with_storage_grid_size()
        self.waves = wave_plan(self.batch, grid.x * grid.y)
        self.ckc = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.state_tensors = []
        self.indices = []
        for start, end in self.waves:
            view = ttnn.reshape(initial_state[start:end], ((end - start) * 12, 128, 128))
            ring = ttnn.empty_like(view)
            ttnn.copy(view, ring)
            self.state_tensors.append(ring)
            self.indices.append(
                ttnn.from_torch(
                    torch.arange((end - start) * 12, dtype=torch.int32).reshape(1, -1),
                    device=self.mesh,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                )
            )
        self.state_addresses = tuple(t.buffer_address() for t in self.state_tensors)
        # This output belongs to the owner before any trace is captured. Returning
        # a freshly allocated output from a later independent trace would leave
        # a live allocation unsafe for an earlier trace's replay workspace.
        self.output = ttnn.empty(
            (1, self.batch * 12, 128),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.output_address = self.output.buffer_address()
        self.calls = 0

    def reset_from(self, initial_state):
        """Probe setup/phase boundary only; never inside measured step or trace."""
        import ttnn

        assert tuple(initial_state.shape) == (self.batch, 12, 128, 128)
        assert initial_state.dtype == ttnn.float32
        for (start, end), ring in zip(self.waves, self.state_tensors):
            source = ttnn.reshape(initial_state[start:end], ((end - start) * 12, 128, 128))
            ttnn.copy(source, ring)

    def step(self, packed, g, beta):
        import ttnn

        assert tuple(packed.shape) == tuple(packed.padded_shape) == (self.batch, 1, 2560)
        assert packed.dtype == ttnn.bfloat16 and packed.layout == ttnn.ROW_MAJOR_LAYOUT
        for gate in (g, beta):
            assert tuple(gate.shape) == (self.batch, 1, 12)
            assert gate.dtype in (ttnn.bfloat16, ttnn.float32) and gate.layout == ttnn.TILE_LAYOUT
        for tensor in (packed, g, beta):
            assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG and tensor.device() == self.mesh

        def norm(x):
            xf = ttnn.typecast(x, ttnn.float32)
            inv = ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(xf, xf), dim=-1, keepdim=True), 1e-6))
            return ttnn.multiply(xf, inv)

        rows = []
        for (start, end), ring, indices in zip(self.waves, self.state_tensors, self.indices):
            users = end - start
            raw = packed[start:end]
            # Reshape while still ROW_MAJOR, then tilize with owned zero padding.
            q = ttnn.to_layout(ttnn.reshape(raw[:, :, :512], (users, 1, 4, 128)), ttnn.TILE_LAYOUT)
            k = ttnn.to_layout(ttnn.reshape(raw[:, :, 512:1024], (users, 1, 4, 128)), ttnn.TILE_LAYOUT)
            v = ttnn.to_layout(ttnn.reshape(raw[:, :, 1024:2560], (users, 1, 12, 128)), ttnn.TILE_LAYOUT)
            output, updated = ttnn.transformer.fused_recurrent_gated_delta_rule(
                norm(q),
                norm(k),
                v,
                g[start:end],
                beta[start:end],
                scale=128**-0.5,
                initial_state=ring,
                initial_state_block_idx=indices,
                output_per_token_state=True,
                use_qk_l2norm=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.ckc,
            )
            assert updated.buffer_address() == ring.buffer_address(), "Native T1 ring must update in place"
            # Head/user padding differs from compact output. A real device pack
            # is necessary; this public-API control includes the complete cost.
            rows.append(ttnn.reshape(ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT), (users, 1, 1536)))
        packed_output = rows[0] if len(rows) == 1 else ttnn.concat(rows, dim=0)
        result = ttnn.to_layout(ttnn.reshape(packed_output, (1, self.batch * 12, 128)), ttnn.TILE_LAYOUT)
        ttnn.copy(result, self.output)
        assert tuple(t.buffer_address() for t in self.state_tensors) == self.state_addresses
        assert self.output.buffer_address() == self.output_address
        self.calls += 1
        return self.output
