# Grouped SFPU initialization candidate

Source-only hypothesis, based on `candidates/gdn_sfpu_accumulator_fused.py` and the frozen `accumulator_on78.py` decoder. Parent reports recurrence profile 88 at 592.676 us (BRISC 592.5, NCRISC 530.4, TRISC0/1/2 587.5/588.8/589.0 us). This candidate targets repeated math setup; it has no hardware result yet.

`candidates/gdn_sfpu_grouped_init.py` retains every low-level arithmetic body, IO kernel, CB, precision configuration, persistent-state allocation, reduction order, and BF16 rounding boundary. It removes setup repeated across four consecutive state-tile operations. Scalar and column controls are independent immutable compile-time arguments 5 and 6; existing accumulator/delta controls remain 3 and 4.

| Group per value column | Before | Candidate | Prerequisite |
| --- | --- | --- | --- |
| Four state decay multiplications | 4 scalar init calls | 1 | scalar control |
| Four K prediction products | 4 column init calls | 1 | column control and fused accumulator |
| Four delta/K state updates | 4 column init calls | 1 | column control and hoisted row broadcast |
| Four Q readout products | 4 column init calls | 1 | column control and fused accumulator |

Four value columns save 12 scalar and 36 column init calls per head. Each remaining init occurs at a group entry. In particular, the update group initializes column state after the row broadcast. The standalone beta multiply, normalization path, reductions, typecast, and all family transitions retain their setup. Scalar normalization interleaves Q scaling, so that path deliberately retains per-operation setup.

## Source legality

- `tt_metal/hw/inc/api/compute/sfpu_binary_bcast.h`: `sfpu_bcast_col_init` documents persistent column lane mask/replay shared between opcodes, initialized once; scalar init is required again after shared configuration changes.
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_bcast.h`, `_sfpu_binary_bcast_init_`: generic SFPU configuration and zero-stride MOD7 are set; column mode additionally builds LREG6 and records broadcast replay slot 0. Native column multiply and the existing local column multiply/add preserve LREG6 and replay. The existing local row-copy primitive writes LREG4–7, so its following column init cannot be removed.
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h`, `eltwise_unary_configure_addrmod`: datacopy uses MOD0/2/3 and non-SFPU movement. Intervening `copy_tile` does not change MOD7, SFPU lane configuration, LREG6, or SFPU replay.
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h` and `llk_math_eltwise_sfpu_common.h`: every retained operation wrapper still starts/stalls SFPU, establishes its DST base, executes the numerical body, and clears its DST base. Only redundant init calls are bypassed; those synchronization wrappers remain intact.
- `llk_math_eltwise_binary_sfpu.h` explicitly assigns MOD7 to avoid the MOD0/2 used by A2D. `_llk_math_eltwise_binary_sfpu_init_` resets counters; each regular datacopy operation and retained SFPU start/done protocol remains unchanged. Hardware own-input validation remains necessary to check this interleaving.

## Reproduction and acceptance

Run from the checkout root:

```sh
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_grouped_init_source.py
python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_grouped_init_candidate.py --base accumulator_on78.py --output groupedinit_on78.py --variant both
```

`--variant scalar`, `column`, and `control` produce independent immutable controls; choose new output names. The generator refuses overwrites. Overlaying the helper on a frozen decoder replaces only recurrence source/configuration functions at import time; no runtime monkeypatch, host state switch, or host fallback is introduced.

First compare own-input state and output with the accumulator baseline at B1/B3/B32, normal/zero-state/zero-KV cases, multiple changed inputs, and stable allocation. Then compare short traced latency and the 512-step sequence with identical input/weight/cache policy. Require unchanged PCC and measured improvement; do not select based on the number of removed calls. Preserve the known-passing baseline until correctness, watcher, and measured performance gates pass. No hardware was accessed while producing this candidate.
