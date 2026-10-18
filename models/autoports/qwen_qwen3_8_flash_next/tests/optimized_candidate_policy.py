"""Cumulative candidate policy; final runtime integration is independently validated."""
import ttnn

def install():
    from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder as mod
    if getattr(mod,'_best_candidate_installed',False):return
    mod._best_candidate_installed=True
    for cls in (mod.OptimizedGatedDeltaNet,mod.OptimizedQSA):
        cls.projection_dtype=ttnn.bfloat8_b
        cls.projection_fidelity=ttnn.MathFidelity.HiFi2
    mod.OptimizedMoE.dedicated_weight_dtype=ttnn.bfloat4_b
    mod.OptimizedMoE.down_dtype=ttnn.bfloat4_b
    from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_dram_candidate import install as dram
    dram((10,8),(8,24),2,result_memory='l1')
    from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_gdn_movement_candidate import MovementGDN
    MovementGDN.roll=True;MovementGDN.inplace=True;MovementGDN.composite_norm=True
    mod.OptimizedGatedDeltaNet=MovementGDN
    from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_hybrid_moe_candidate import HybridMoE
    from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_routing_candidate import RoutedMoE
    class HybridRoutedMoE(HybridMoE,RoutedMoE):
        prefill_compact=False
        prefill_inplace=True
        sparse_cores=20
        sparse_down_cores=40
        gate_block=40
        down_block=20
        reduce_bf8=True
        padded_topk=True
    mod.OptimizedMoE=HybridRoutedMoE
    from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_hyper_candidate import ShardedHyperConnection as HC
    HC.norm_cores=20;HC.keep_sharded=True;HC.adapt_flatten=True;HC.adapt_binary=True
    HC.dram_readers=2;HC.dram_cores=(20,2);HC.dram_blocks=(16,5);HC.fused_up=False
    mod.OptimizedHyperConnection=HC
    from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_ple_candidate import TunedPLE
    TunedPLE.weight_dtype=ttnn.bfloat8_b;TunedPLE.fidelity=ttnn.MathFidelity.HiFi2
    TunedPLE.norm_cores=40;TunedPLE.adapt_l1=True
    mod.OptimizedPLE=TunedPLE
    from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_qsa_sparse_candidate import SparseQSA
    SparseQSA.indexer=True;SparseQSA.mask_bounds=True;SparseQSA.decode_chunk=32
    mod.OptimizedQSA=SparseQSA
