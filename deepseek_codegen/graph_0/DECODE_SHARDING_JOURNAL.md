# Decode resharding / collective-fusion experiment (branch: mvasiljevic/deepseek-decode-sharding)

Base = mvasiljevic/deepseek-tm-optimization @ dc2b203 (EST 177.8 ms, 5.96x). Goal: cut the
latency-bound TP collectives (~27% of decode e2e) by fusing/restructuring, keeping I/O + PCC.

## Idea: fuse the per-matmul cross-device reductions
The qkv-down matmul's partial output was reduced 3x SEPARATELY (kv_a all_gather+sum, q_a
reduce_scatter[+all_gather for q_norm in L1], indexer all_gather+sum). Fuse into ONE
all_gather+sum of the full [32,2304], slice locally; q_a re-sharded via local mesh_partition
(L0, feeds TP q_b) or fed full to q_norm (L1). Numerically identical, argmax-gated.

| step | change | attn_0 | attn_1 | EST e2e | PCC | notes |
|------|--------|--------|--------|---------|-----|-------|
| base | (tm-opt HEAD) | 1.089 | 1.124 | 177.8 | argmax 100% | attn CCL 6/layer |
| s1 | fuse matmul_0 reductions (attn L0) | 0.989 | 1.124 | ~172 | ==base | -2 CCL/layer; mesh_partition ~free (1us) |
| s2 | fuse matmul_15 reductions (attn L1; q_a 2-CCL bonus) | 0.989 | **0.963** | **168.8** | ==base | -3 CCL L1; attn/layer 1.107->0.976; **6.28x** |

KEY: collective COUNT reduction works (latency-bound CCL). attn_1 dropped more (-161us) because its
q_a was a 2-CCL all_reduce (reduce_scatter+all_gather) into the q_norm — the fused full reduction
feeds q_norm directly. No #46208 (these are the qkv-down DRAM reductions). mesh_partition (replicated
-> sharded) is a free local op.
Next: same fusion pattern in dense MLP (mlp_0) + MoE shared-FFN/down reductions.
