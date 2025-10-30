import torch

import ttnn


def build_topk_mask(S, E, k):
    m = torch.zeros([1, S, E], dtype=torch.bfloat16)
    for s in range(S):
        start = (s * k) % E
        idxs = [(start + i) % E for i in range(k)]
        m[0, s, idxs] = 1.0
    return m


def main():
    device = ttnn.open_device(device_id=0)

    S, E, K, N, nnz = 32, 256, 1024, 1024, 8

    # input_tensor_a = WEIGHTS (must be A), layout TILE, bf16
    weights = ttnn.ones([E, N, K], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # input_tensor_b = TOKENS (must be B), layout TILE, bf16
    tokens = ttnn.ones([S, 1, K], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Sparsity [1, S, E] (ROW_MAJOR bf16) – product of A&B batch dims matches
    sparsity = build_topk_mask(S, E, nnz)
    sparsity = ttnn.from_torch(sparsity, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    # 1-D multicast config (SUPPORTED by sparse_matmul); multicast input 0 (weights)
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),  # (x=N, y=M)
        in0_block_w=8,  # divides K_tiles (=32)
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,  # 8 rows × 1 = 8 M-tiles
        per_core_N=4,  # 8 cols × 4 = 32 N-tiles
        mcast_in0=True,  # <<< only mcast_in0 supported for sparse
        fused_activation=None,
        fuse_batch=True,
    )
    print(sparsity.shape)
    print(tokens.shape)
    print(weights.shape)

    out = ttnn.sparse_matmul(
        tokens,
        weights,
        sparsity=sparsity,
        # nnz=nnz * S,
        program_config=pc,
    )
    print("Output shape:", out.shape)  # typically [S, E, 1, N]

    # ttnn.close_device(device)


if __name__ == "__main__":
    main()
