# Native BF16 persistence candidate (CPU preparation only)

`tt/recurrent_pack_candidate.py` adds an isolated `cumulative_native_pack` source transform. Its baseline is **cumulative_exp_scalar**, not the original unoptimized recurrence. The sole source delta is the body of `_gdn_store_bf16_rne_`: replace six explicit integer rounding/masking instructions plus the scratch-register FP32 store with `TT_SFPSTORE(DATA,InstrModLoadStore::FP32,ADDR_MOD_7,address)`. The historical helper name and nearby baseline comments are intentionally retained so the generated-source diff is limited to that body; their claim that persistence was already rounded no longer describes this candidate.

No default, model, frozen decoder, existing variant, reader, writer, CB descriptor, state page/address mapping, NOC operation, or synchronization is changed. Live DATA registers remain unrounded FP32 for current Q readout. The `STATE_BF16=false` branch is unchanged and does not call the changed helper. Changed BF16 persistence can affect subsequent tokens even if the first output matches. This is an **unproven precision candidate**, not an accepted exact optimization.

## Source evidence and limits

Read alongside `debug_recurrent/AUTODEBUG.md`, “Exactness boundaries and other potential issues.” The frozen compute config (`tt/optimized_decoder.py`, `_fused_delta_row_compact_fp32_compute_config`, around line 4814) enables FP32 DEST, HiFi4 and full DST sync, disables approximate math, and keeps BF16 state CBs 5/11 on default unpack. It does not request stochastic rounding. After FP32 update/readout, the existing code calls `pack_reconfig_data_format(12,11)`, packs four state tiles into BF16 CB11, and restores FP32 CB12 for readout (around line 5527).

The actual Blackhole LLK sources support investigating native nearest-even conversion rather than assuming RTZ:

- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h:150` documents non-stochastic rounding as nearest-even; pack stochastic mode covers both the DEST-to-pack-source gasket and pack-source-to-destination conversion.
- `common/inc/cunpack_common.h:761` gives `configure_unpack_AB` default `fpu_srnd_en=false, pack_srnd_en=false`; lines 839–847 explicitly write the FPU, gasket and packer enable bits. `llk_lib/llk_unpack_common.h:96` likewise clears them for `StochRndType::None`.
- `common/inc/cpack_common.h:509` reconfigures source/destination formats and selects 32-bit DEST reads whenever FP32 accumulation is enabled. The `Round_10b_mant` branch targets Float16/A or FP8 conversion, not BF16. This code does not select a BF16 RTZ mode.
- `tt_metal/api/tt-metalium/program_descriptors.hpp:103` exposes no stochastic-rounding field on this generic compute descriptor.
- The explicit SFPU typecast (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h:273`) deliberately uses integer add/parity/mask and an FP32 store. Its comment says an old FP16B-store truncation approach was not equivalent. This candidate uses an FP32 store plus the existing BF16 pack; it does **not** reinstate the old FP16B store.

Those source facts do not prove bitwise equivalence for ties, subnormals, overflow or NaN payloads. Removing six SFPU instructions per 32 values removes 3,072 instructions/head from the source path. Whether that reduces elapsed microseconds depends on overlap and actual kernel bottlenecks. No gain or hardware correctness is claimed.

## Probe and ownership

`tests/probe_multichip_native_pack.py` compiles a private AST-derived copy of the existing focused recurrent probe. It substitutes the matched cumulative baseline and the one-body candidate. Exact output **and persisted-state** checks, per-rank/user checks, finite checks, changed-input trace checks and post-timing checks are retained unchanged. First mismatch records a failure and exits; no threshold relaxation occurs. The default matrix includes B1/2/3/10/17/31/32, BF16/FP32 state, zero-QKV/zero-state controls and the extra B3 one-core wrap. Both kernels warm before either trace capture; all device inputs, states and outputs are preallocated and retained through trace release. The reader/writer protocol is inherited unchanged.

The separate `--full-layer` entrypoint registers the new variant once for the isolated CLI process, then runs the existing full-layer runner with its unchanged 0.995 PCC and exact self-replay gates. Registration is restored on exit. Existing variants delegate to their original source function; no per-call monkeypatch, bound-owner hook or persistent-model mutation is introduced. Source generation counts are reported and a run fails if the candidate was unused. Full-layer policy JSON/file is preserved except `recurrent_variant=cumulative_native_pack` and `batched_conv_b1=true`; B1 needs that flag to reach the raw packed route. A matched B1 baseline must set the same flag and `recurrent_variant=cumulative_exp_scalar`. Legacy flat prefill tails retain the ordinary route.

CPU validation passed: all four grouped/poison source combinations, body-only delta, unchanged downstream FP32/pack/address/CB source blocks, registration restoration even on exceptions, private probe compilation, and a stub full-layer CLI forwarding/registration/report test. No Torch/TTNN import or device execution occurred. The normal strict probes and production defaults are untouched.

## Parent-serialized commands

Run from repository root with the existing hardware environment; use fresh artifact names. `D` below is only a task-specific shell variable. Run exact BF16 first; if it fails, preserve the failed JSON/log, run the separate FP32 control, and evaluate full-layer as a **precision** candidate. Do not relabel a strict failure as pass. These commands deliberately are separate, not chained with `&&`.

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_native_pack --cpu-check

timeout 180s env TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_native_pack --batches 32 --state-dtypes bf16 --modes normal --low-cores 0 --warmup 3 --timing-samples 7 --replays-per-sample 10 --output "$D/native_pack_exact_b32_bf16.json"

timeout 180s env TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_native_pack --batches 32 --state-dtypes fp32 --modes normal --low-cores 0 --output "$D/native_pack_exact_b32_fp32_control.json"

timeout 600s env TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_native_pack --batches 1 2 3 17 31 32 --state-dtypes bf16 fp32 --modes normal zero_qkv zero_state --low-cores 1 --output "$D/native_pack_exact_broad.json"

timeout 300s env TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_native_pack --full-layer --layer 0 --batch 32 --length 33 --trace --repeats 100 --reference models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/delivery_perf_ref_l0_b32_s33.pt --output "$D/native_pack_precision_l0_b32.json"

timeout 300s env TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_native_pack --full-layer --layer 0 --batch 1 --length 512 --trace --repeats 100 --reference models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/delivery_perf_ref_l0_b1_s512.pt --output "$D/native_pack_precision_l0_b1.json"
```

For comparison with the current selected policy, supply the same `--policy-file` or `--policy-json` used by the parent to both full-layer runs. The wrapper changes only the two flags described above. Use the ordinary `run_multichip_decoder` for the matched baseline with `recurrent_variant=cumulative_exp_scalar, batched_conv_b1=true`; retain all other flags. This preparation does not run a full model, vLLM, or any device command.
