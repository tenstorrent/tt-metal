import pytest
import torch

import ttnn


def build_topk_mask(S, E, k):
    m = torch.zeros([1, S, E], dtype=torch.bfloat16)
    for s in range(S):
        start = (s * k) % E
        idxs = [(start + i) % E for i in range(k)]
        m[0, s, idxs] = 1.0
    return m


def build_dummy_state_dict(num_experts: int, hidden_size: int, intermediate_size: int, seed: int = 0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    gate = torch.randn((num_experts, hidden_size, intermediate_size), generator=g, dtype=torch.bfloat16)
    up = torch.randn((num_experts, hidden_size, intermediate_size), generator=g, dtype=torch.bfloat16)
    # interleave gate/up along the last dim -> [E, H, 2I]
    gate_up = torch.empty((num_experts, hidden_size, 2 * intermediate_size), dtype=torch.bfloat16)
    gate_up[..., ::2] = gate
    gate_up[..., 1::2] = up

    gate_b = torch.randn((num_experts, 2 * intermediate_size), generator=g, dtype=torch.bfloat16)

    down = torch.randn((num_experts, intermediate_size, hidden_size), generator=g, dtype=torch.bfloat16)
    down_b = torch.randn((num_experts, hidden_size), generator=g, dtype=torch.bfloat16)

    return {
        "gate_up_proj": gate_up,
        "gate_up_proj_bias": gate_b,
        "down_proj": down,
        "down_proj_bias": down_b,
    }


class SparseExperts:
    def __init__(
        self,
        mesh_device,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_local_experts: int,
        num_experts_per_tok: int,
        state_dict: dict,
        dtype=ttnn.bfloat4_b,
        tensor_cache_path=None,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.mesh_device = mesh_device

        # For TP across columns: intermediate split along -1 (I)
        self.intermediate_size_per_device = self.intermediate_size // mesh_device.shape[1]

        # Unpack packed weights/biases (gate/up are interleaved in last dim)
        gate_proj = state_dict["gate_up_proj"][..., ::2].reshape(
            self.num_experts, self.hidden_size, self.intermediate_size
        )
        up_proj = state_dict["gate_up_proj"][..., 1::2].reshape(
            self.num_experts, self.hidden_size, self.intermediate_size
        )
        gate_proj_bias = state_dict["gate_up_proj_bias"][..., ::2].reshape(self.num_experts, 1, self.intermediate_size)
        up_proj_bias = state_dict["gate_up_proj_bias"][..., 1::2].reshape(self.num_experts, 1, self.intermediate_size)

        col_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)  # shard along I/H col-wise
        row_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-2)  # shard along I row-wise

        self.gate_proj = ttnn.as_tensor(
            gate_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.up_proj = ttnn.as_tensor(
            up_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.gate_proj_bias = ttnn.as_tensor(
            gate_proj_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.up_proj_bias = ttnn.as_tensor(
            up_proj_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        down_proj = state_dict["down_proj"].reshape(self.num_experts, self.intermediate_size, self.hidden_size)
        down_proj_bias = state_dict["down_proj_bias"].reshape(self.num_experts, 1, self.hidden_size)

        self.down_proj = ttnn.as_tensor(
            down_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=row_mesh_mapper,
            cache_file_name=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Row-parallel bias must not be replicated. Extend with zeros for TP devices on H dim.
        if mesh_device.shape[1] > 1:
            down_proj_bias = torch.cat(
                [down_proj_bias] + [torch.zeros_like(down_proj_bias)] * (mesh_device.shape[1] - 1),
                dim=-1,
            )

        self.down_proj_bias = ttnn.as_tensor(
            down_proj_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.alpha = 1.702

    def __call__(self, hidden_states, routing_weights):
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        hidden_states_4D = hidden_states
        print(hidden_states.shape)
        print(self.gate_proj.shape)

        routing_weights_rm = routing_weights

        output_tile = ttnn.Tile([32, 32])
        down_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),  # x along N, y along M
            in0_block_w=1,  # divides K_tiles (=32)
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,  # 1 * y(=1) = 1 M-tile (covers S=32)
            per_core_N=4,  # 4 * x(=8) = 32 N-tiles (covers I=1024)
            mcast_in0=True,
            fused_activation=None,
            fuse_batch=True,
        )
        # ---- gate ----
        gate = ttnn.sparse_matmul(
            hidden_states_4D,
            self.gate_proj,
            sparsity=routing_weights_rm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
        )
        print("Gate: ", gate.shape)
        gate = gate + self.gate_proj_bias

        # ---- up ----
        up = ttnn.sparse_matmul(
            hidden_states_4D,
            self.up_proj,
            sparsity=routing_weights_rm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
        )
        print("Up: ", up.shape)
        up = up + self.up_proj_bias

        # ---- glu & prepare down input ----
        glu = gate * ttnn.sigmoid(gate * self.alpha)
        down_in0 = (up + 1) * glu
        ttnn.deallocate(glu)
        ttnn.deallocate(up)
        ttnn.deallocate(gate)
        print(down_in0.shape)
        print("Made it past reshape")

        # ---- down ----
        down = ttnn.sparse_matmul(
            down_in0,
            self.down_proj,
            sparsity=routing_weights_rm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            program_config=down_pc,
        )
        ttnn.deallocate(down_in0)

        print("Down: ", down.shape)

        next_states = ttnn.sum(down, dim=1, keepdim=True)
        print("Next: ", next_states)

        return next_states


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test(mesh_device):
    # Model / run sizes
    S = 32  # tokens
    E = 256  # experts
    H = 4096  # hidden size
    I = 1408  # intermediate size per expert
    K = 8  # top-k experts per token (nnz per token)

    # Build dummy weights/biases in File 2's packed layout
    state_dict = build_dummy_state_dict(E, H, I, seed=17)

    # Instantiate SparseExperts
    experts = SparseExperts(
        mesh_device,
        hidden_size=H,
        intermediate_size=I,
        num_local_experts=E,
        num_experts_per_tok=K,
        state_dict=state_dict,
        dtype=ttnn.bfloat4_b,
    )

    hidden_states = ttnn.ones([S, 1, H], device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # routing_weights: [S, E] (torch -> device ROW_MAJOR inside call)
    sparsity = build_topk_mask(S, E, K)  # [1, S, E] torch bf16
    routing_weights = ttnn.from_torch(sparsity, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    print(routing_weights.shape)
    out = experts(hidden_states, routing_weights)

    print("Output (pre-collectives) shape:", out.shape)


if __name__ == "__main__":
    main()
