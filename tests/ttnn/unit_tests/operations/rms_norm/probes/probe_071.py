import sys

sys.path.insert(
    0,
    "/localdev/mstaletovic/2026_08_11/1659_mstaletovic_agent_eval/clones/rms_norm_run1/tt-metal/ttnn/ttnn/operations/rms_norm/perf_experiments/worksplit_retune",
)
import sweep

R = sweep.rule_two_stage_gain
# default pick vs the RULED pick, 2 reps each, on every interleaved decode/guard shape
sweep.main(
    ["decode1024", "decode1280", "decode1536", "decode2304", "decode3072", "decode5120", "decode7168"],
    gs=["rule"],
    rule=R,
    reps=2,
)
sweep.main(["wtail"], gs=["rule", 11, 22, 55], rule=R, reps=2)
sweep.main(["rm_interleaved", "ragged5119", "wide4096"], gs=["rule"], rule=R, reps=2)
# sharded: the shard spec supplies the geometry, so _select_regime never runs -- verify
sweep.main(["wshard1024", "wshard7168", "bshard1024", "hshard512"], gs=["rule"], rule=R, reps=2)
