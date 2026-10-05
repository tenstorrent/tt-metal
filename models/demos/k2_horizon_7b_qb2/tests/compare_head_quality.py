"""Isolate terminal reduction geometry on the exact shared chat suite."""

import gc
from pathlib import Path

import torch

import ttnn

from ..tt.generator import K2Generator
from ..tt.model import K2Model
from .run_qualitative_extended import run

OUT = Path("models/demos/k2_horizon_7b_qb2/doc/optimized_full_model").resolve()


def main():
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    try:
        gen = K2Generator(mesh)
        large_k_head = gen.model.head_decode
        candidate = K2Model(mesh, override_num_layers=1, head_k=2, head_split_size=16384)
        small_k_head = candidate.head_decode
        del candidate
        gc.collect()
        for name, head, block, splits in [("k4_s8192", large_k_head, 4, 8), ("k2_s16384", small_k_head, 2, 4)]:
            gen._release_traces()
            gen.prefill_state = None
            gen.model.head_decode = head
            head.config.program_configs = [
                ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=block, per_core_M=1, per_core_N=8 if splits == 8 else 16, num_workers_per_dram_bank=1
                )
            ] * splits
            print("QUALITY_HEAD", name, flush=True)
            run(gen, output_name=str(OUT / ("quality_head_" + name + ".json")))
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
