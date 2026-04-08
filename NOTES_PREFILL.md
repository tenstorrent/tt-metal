# GPT-OSS Prefill MoE Integration Notes

## Current State (2026-04-07 16:00)
All tracks complete. Branch: sraizada/gpt-oss-prefill-moe-deepseek (2 commits)

### Results
- 36L demo batch=128: TTFT=66ms, 27.4 tok/s/user, 3504 tok/s throughput (no regression)
- 36L demo batch=1: TTFT=309ms, 19.0 tok/s/user, coherent text generation
- Isolated MoE PCC=0.9996 (random weights), PCC=0.9954 (real weights)
- Per-layer MoE: 8.3ms (seq=128), 14.5ms (seq=512) on mesh (4,8)

### Key Decision: use_deepseek_prefill defaults OFF
DeepSeekPrefillConfig creates TtRoutedExpert with separate weight copies
for all 36 layers (~13K weight transfers). This doubles model init time.
For batch=128, the seq_len mismatch fallback triggers anyway (each user
prefills individually with variable prompt lengths). Default OFF avoids
the weight loading overhead. Enable manually for single-user long-prefill.

### Next Steps for DeepSeek Prefill Performance
1. Lazy weight loading: create TtRoutedExpert on first forward, not at init
2. Weight sharing: reuse ThroughputExpertWeights instead of loading separately
3. Variable seq_len support: handle arbitrary seq_len without fallback
4. EP=8 ndg=1 profiling: now that combine is fixed, compare against EP=4
