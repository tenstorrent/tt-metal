# Best known config for Llama-3.3-70B TP32 (models/demos/llama3_70b_galaxy) on one BH Galaxy, Nov 18 2026.
# Source it before run_server.sh / run_sla.sh / prefill_profile.sh:  source sla_scripts/bh_tp32/env.sh
# Prefetcher + chunked prefill (existing switches)
export LLAMA_BH_PREFETCHER=1 LLAMA_BH_GLOBAL_CB_TILES=512 LLAMA_BH_BATCHED_PREFILL=1 GALAXY_PREFILL_NUM_LINKS=2 LLAMA_GALAXY_CHUNKED_PREFILL=1
# Decode: shared RoPE tiles, ring collectives, 2-link attention all-gather (decode layer ~259 -> ~214 us)
export LLAMA_DECODE_ROPE_CACHE=1 LLAMA_BH_CCL_TOPOLOGY=ring LLAMA_BH_RS_WORKERS=1 LLAMA_BH_ATTN_AG_LINKS=2
# Prefill (one 4K chunk 8.1 -> ~5.3 ms device time; 80-layer logits PCC 0.992 vs all off). SDPA LoFi drops PCC to 0.96: keep HiFi2.
export LLAMA_BH_PREFILL_RS_WORKERS=4 LLAMA_BH_SDPA_FIDELITY=HiFi2 LLAMA_BH_PREFILL_WO_RS=1 LLAMA_BH_PREFILL_FF2_MINIMAL=1
export LLAMA_BH_PREFILL_MM_TUNED=1 LLAMA_BH_PREFILL_FF13_BF8=1 LLAMA_BH_PREFILL_MINIMAL_2K=1
# Fabric: vLLM reads FABRIC (run_server.sh), pytest reads TT_FABRIC_OVERRIDE (conftest.py)
export FABRIC=${FABRIC:-FABRIC_1D_RING} TT_FABRIC_OVERRIDE=${TT_FABRIC_OVERRIDE:-FABRIC_1D_RING}
