# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduce minimal-matmul tile residency for larger non-aligned prefill chunks."""
import argparse
import sys
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_large_chunk_geometry as base

class ChunkBudget(base.LargeChunkGeometry):
    blocks=(4,8,8)
    @classmethod
    def from_state_dict(cls,*args,**kwargs):
        self=super().from_state_dict(*args,**kwargs)
        m,k,n=cls.blocks
        self.prefill_program=ttnn.MinimalMatmulConfig(M_block_size=m,K_block_size=k,N_block_size=n,
            subblock_h=1,subblock_w=2,compute_with_storage_grid_size=(11,10))
        return self

if __name__ == "__main__":
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument("--blocks",default="4,8,8")
    args,rest=ap.parse_known_args();sys.argv=[sys.argv[0],*rest]
    ChunkBudget.blocks=tuple(map(int,args.blocks.split(",")))
    base.LargeChunkGeometry=ChunkBudget
    base.main()
