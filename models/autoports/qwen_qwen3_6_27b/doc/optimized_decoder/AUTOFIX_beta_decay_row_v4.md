# V4 repair: both SFPU column groups in the first row band

`beta188_probe.json` preserves V3 failure: beta is approximately50% bit equal,
softplus and decay fail, native add remains exact, all padded values remain
finite and repeat equal. No whole-decoder performance was measured for V3.

The source-level error is the assumed size of one SFPU vector, not new
sigmoid/softplus arithmetic. Blackhole's
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_bcast.h`
documents the exact DST layout at lines22–42:

- One load/store covers4rows ×8columns.
- Slot+0 covers face columns0–7 of rows0–3.
- Slot+2 covers face columns8–15 of rows0–3.
- Slot+4 advances to rows4–7.

VectorMode::R selects the two top faces, but ITERATIONS=1 processes only the
first8columns in **each** face. Thus V3 visited columns0–7 and16–23, leaving
8–15 and24–31 untouched. This matches the approximatelyhalf-value equality
failure without requiring a changed polynomial or a face-to-face cursor fix.

V4 uses ITERATIONS=2 with VectorMode::R for both native unary calculations.
It covers all32logicalcolumns and also transforms unused padded rows1–3.
Nominal unary vector iterations decrease32→4 relative to passing V2, an8×
reduction in this portion of SFPU work. This is not an8× kernel or decoder
latency claim. FPU add, FP32 final multiplication, init calls, packing, CBs,
reader/writer and runtime pointers remain unchanged.

Immutable files:

- `candidates/gdn_beta_decay_fused_v4.py`
- `candidates/make_beta_decay_candidate_v4.py`
- `candidates/beta_decay_v4_on185.py`
- `candidates/beta_decay_v4_on185_source.json`

The frozen decoder changes only integrated185's `_gate178_COMPUTE` source
string. Syntax and AST checks pass. A CPU enumeration of the documented
row-band mapping confirms V3 covers16/32logicalcolumns and V4 covers32/32.
Hardware exactness is pending, not inferred from that enumeration.

Reuse the unchanged strict padded-output/intermediate-aware probe:

```sh
python -m models.autoports.qwen_qwen3_6_27b.tests.run_beta_decay_probe_v3 \
  --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/gdn_beta_decay_fused_v4.py \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/beta190_probe.json
```

The exact logical beta, decay, add and softplus requirements remain unchanged.
Padding must be finite and repeat equal, but need not match unspecified native
padding. V2 remains the correct control if this revision fails any gate.
