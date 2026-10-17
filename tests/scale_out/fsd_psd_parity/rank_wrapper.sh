#!/usr/bin/env bash
# Rank-aware wrapper: set TT_METAL_MOCK_CLUSTER_DESC_PATH from the mapping for this MPI rank.
R="${OMPI_COMM_WORLD_RANK:-0}"
MAPDIR="$1"; shift         # dir containing per-rank path list
BIN="$1"; shift            # generate_rank_bindings binary
P=$(sed -n "$((R+1))p" "$MAPDIR/rank_paths.txt")
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$P"
exec "$BIN" "$@"
