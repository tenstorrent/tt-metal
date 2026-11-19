# Fuse state decay with K prediction

The accumulator/grouped-init recurrence still performs two passes over every state tile: decay into retained FP32 DST, then read that DST again for K prediction. This source-only candidate eliminates the second read while preserving both multiply rounding steps and the four-K-tile accumulation order. The parent reports the current exact-norm decoder at approximately 2.055 ms for a 512-step B32 compact workload and recurrence at approximately 592 us. No latency result is claimed for this candidate.

The new immutable helper is `candidates/gdn_sfpu_decay_predict.py`. The frozen `decaypredict_on96.py` overlays it on `grouped96.py`. `_GDN_FUSE_DECAY_PREDICT` becomes compile-time argument 7. The control leaves the grouped-init arithmetic path intact. Existing IO kernels, two-head buffers, BF16 persistent-state policy, normalized/raw QK interfaces, and public allocation behavior are unchanged.

For each value column, the candidate loads the four state tiles into DST0–3 before arithmetic, then restores the F32 unpack format once. DST7 contains decay; DST5 receives each K tile; DST6 holds prediction. In the first K tile, scalar broadcast loads decay into LREG7. Subsequent K tiles reuse LREG7: column mask/replay uses LREG6, scratch uses LREG0/2, and the data lanes use LREG1/3/4/5. `copy_tile` changes none of these LREGs. Every primitive:

1. Broadcasts its K column and loads the original state band.
2. Uses SFPMUL to compute rounded FP32 decay and stores it into the retained state tile.
3. Uses the same FP32 data LREGs for a separate K multiply.
4. For K tiles 1–3, adds the previous FP32 prediction accumulator using the existing explicit SFPADD sequence.
5. Stores the prediction accumulator.

There is no SFPMAD, reduction reassociation, state rounding change, or overwrite of a future state's DST input. The final updated state still rounds to BF16 after current output readout. The subsequent delta, update, Q readout, reduction and persistence code remains unchanged.

The source removes 16 full-tile FP32 DST reads per head (64 KiB of logical on-chip DST traffic/head; 96 MiB across B32), 16 separate scalar-operation wrappers/head, and 12 repeated decay-scalar broadcasts/head. These are on-chip traffic counts, not DRAM savings or measured bandwidth. It also changes state-copy scheduling: four copies occur before SFPU work instead of one copy interleaved with each decay. Hardware must establish whether that scheduling offsets the savings.

Legality follows the existing stage-local column multiply/add primitive and native `ckernel_sfpu_binary_bcast.h`: `_broadcast_scalar_from_dest_` uses LREG0/2/14 and restores predication, leaving LREG6 intact; `_broadcast_stage3_with_data_prefetch_` uses LREG0/2 and the four data registers, leaving retained LREG7 intact. The first fused call initializes LREG7 separately for every value column, after each preceding reduction/typecast family. All standard SFPU start/stall/done wrappers are retained for fused calls. Separate multiply instructions retain the intermediate FP32 result before either the DST store or second multiply.

Validation performed without hardware:

```sh
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_decay_predict_cpu.py
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_decay_predict_candidate.py --base grouped96.py --output decaypredict_on96.py
```

The source/CPU check covers eight recurrent steps, FP32/BF16 persistent state and five initial scales, including zero state. It checks the fused decay/prediction schedule bitwise against its separate FP32 control and checks that IO/config/partition are unchanged. `decay_predict_cpu.json` records its result and helper SHA256. These checks do not emulate SFPU. Required device follow-up: own-input probe with changed inputs and B1/B3/B32, short traced timing, then the same 512-step sequence and watcher coverage as the parent baseline. Use `--control` and a fresh output name for the control candidate. Select only with measured performance and unchanged acceptance gates.
