# chunk-gdn-tp4-queue

## Goal
Make one `ttnn.transformer.chunk_gated_delta_rule` call faster for a 2048-token prefill chunk at the two Qwen TP-4
shapes on one Blackhole chip: BH=12 (Qwen3.6-27B, 4 key heads / 12 value heads) and BH=16 (Qwen3.5-397B-A17B, 4 / 16).
The score is the device time of the whole call, every device op it launches, lower is better. Accuracy and the
op's contracts must hold.

## What the code does
Gated DeltaNet (GDN) linear attention, prefill. Per value head (BH = batch x value heads independent recurrences)
it carries a 128x128 fp32 state S over the sequence in chunks of C=32 tokens (NC = 64 chunks at T=2048):

```
per chunk:  decay = cumsum(g) ; L = tril(exp(decay_i - decay_j))
            T_inv = (I + strictly_lower(k_beta k^T * L))^-1        (the WY inverse)
            u = T_inv v_beta ; w = T_inv (k_beta * exp(decay)) ; intra = (q k^T) * L
            v_new = u - w S ; o = (q * exp(decay)) S + intra v_new
            S = S * exp(decay_last) + k_dec^T v_new
```

The work splits into a **prep** part per (head, chunk) that does not depend on S (everything up to T_inv, u, w,
intra, q_decay, k_dec_t, dl: seven fp32 intermediates) and a **scan** part that is a strict chain over the chunks of
a head (v_new, o, S). Inputs arrive flat token-major (q/k `[1,T,H*128]`, un-normalized: prep L2-normalizes q/k and
folds the 1/sqrt(K) scale in), GQA G = HV/H = 3 or 4 is mapped at read time.

Three programs implement it, chosen by `program_config` (None = the op decides):
- **fused** (`ChunkGdnDeviceOperation`, the default at both TP-4 shapes): one program. Producer cores run prep and
  NoC-write each chunk's seven intermediates straight into the receiver cores' CBs; receivers run the scan. The
  hand-off protocol (slots, credits, VALID flags, mcast to the NV receivers of a head) is specified in
  `device/chunk_gdn_handoff_protocol.md`.
- **phased**: a prep program then a scan program, intermediates through DRAM (~600 us at BH=12).
- **mono**: one core per head does everything.

**Geometry** (fused): NV receivers per head (each carries V/NV columns of S), NP producers per head round-robin over
the chunks, or a **producer pool** of P cores (home producers per head + extras that serve every head's chunks by a
fixed item map), the hand-off depth (slots in flight, 2 or 3), placement (row-local / row-major). **Cost model**:
`choose_fused_geometry` in `device/chunk_gdn_device_operation.cpp` predicts the time of every candidate and the op
runs the fastest (or phased if no fused geometry wins):

```
T = fill(BH, P) + max(chain, busiest home producer, busiest extra) * (1 + balance penalty) + tail
chain = (NC - 1) * (t_step_us(V/NV/32) + kChainSlopeUs * BH) + skew        producer = items * kWpUs (15.98 us)
```

Its constants are measured kernel times (`kWpUs` the prep item, `t_step_us` the scan step per V-slice width: 3.17 us
at Vtl=1, 2.63 at Vtl=2, `kFill*`, `kSkew*`, `kTailUs`, the balance and depth-2 round-trip terms, `kPhasedUs[]`).
`calibration/README.md` explains each and how to re-measure them.

What bounds the two cases today (from the model and the stored measurements on an 11x10 grid):
- **BH=12**: NV=2, NP=7 per head, row-local, depth 3 (108 cores), about 210 us. **Chain-bound**: 63 scan steps of
  ~2.67 us after a ~29 us fill. A faster scan step (or a cheaper hand-off round trip per step) moves it; a faster prep
  item frees producer cores for receivers.
- **BH=16**: producer pool, NV=2, P=78 (3 home per head + 30 extras, share ~27/78), depth 2 (110 cores), about
  271 us. **Production-bound**: 1024 prep items of ~16 us over 78 producers. A faster prep item (the WY-inverse
  solve, the matmuls, exp) moves it directly.
The baseline measured by this campaign's eval (all ops of the call) is in the campaign report.

## Where the code is
All under `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/`:
- `chunk_gated_delta_rule.cpp`: the public op: input handling (flat vs 4-D, typecasts, GQA, padding), the
  fused/phased/mono dispatch, output relayout (`output_head_major`).
- `device/chunk_gdn_device_operation.cpp`: validation, **the cost model** (`choose_fused_geometry`, `t_step_us`,
  constants), the binding `chunk_gdn_fused_geometry` the tests call.
- `device/chunk_gdn_fused_program_factory.cpp`: core sets, placement, item map, CBs, semaphores, kernel args.
- `device/kernels/compute/chunk_gdn_prep.cpp`, `chunk_gdn_scan.cpp`, `chunk_gdn_math.hpp`: the prep and scan compute
  (shared by fused and phased: that is why they stay bit-identical).
- `device/kernels/dataflow/reader_chunk_gdn_prep.cpp` (producer reads), `writer_chunk_gdn_fused.cpp` (producer
  hand-off), `reader_chunk_gdn_scan.cpp` (receiver, `GDN_FUSED_RECEIVER`), `chunk_gdn_handoff*.hpp`,
  `chunk_gdn_fused_map.hpp` (the item map), `chunk_gdn_gate_read.hpp`.
- `device/chunk_gdn_compute_config.hpp`: HiFi4 / fp32 dest enforced; the WY-inverse method (SFPU solve on Blackhole).
- `calibration/`: scripts that measure the cost model's constants with the device profiler (zones `tx_valid`,
  `prep_item`, `scan_step`, `rx_wait_valid`, ...).

LLK / SFPU (editable, JIT-compiled, no host rebuild):
- `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_triangle_solve.h`: the SFPU forward-substitution WY
  inverse (only this op uses it); entry in `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_triangle_solve_sfpu_entry.h`.
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h` and the binary SFPU ops the prep uses
  (`sfpu_mul_dst`, `sfpu_sub_dst`, `sfpu_exp_dst`). Shared with many ops: add a new function rather than changing
  an existing one.

Host files (`*.cpp` outside `kernels/`) need a rebuild (incremental, ccache); kernels and LLK headers do not.
`tests/ttnn/unit_tests/operations/transformers/test_chunk_gdn_fused_geometry.py` mirrors the cost model (an oracle
with the same constants, measured anchors, operating points); keep it in step with any model change.

## What is measured
`agent_orch/campaigns/chunk-gdn-tp4-queue/eval.py`:
1. Runs `tests/ttnn/unit_tests/operations/transformers/test_chunk_gdn_tp4_perf.py` under Tracy, one case at a time
   (`bh12`, `bh16`). Each case calls the op 3 + 10 times exactly as the 27B decoder does (flat bf16 q/k/v, fp32 g,
   bf16 beta, fp32 non-zero initial state, all DRAM interleaved TILE, `output_head_major=True`, default
   `program_config` and `wy_inverse`). A `gdn_call` signpost precedes every call; the value is the mean over the 10
   measured calls of the summed DEVICE KERNEL DURATION of every device op of the call (us). The case result also
   lists the per-op split (`ops`) and the largest core count.
2. Gates: `pcc_o >= 0.99999`, `pcc_fs >= 0.99999` (baseline 0.9999977 / 0.999999) against a float64 torch golden of the whole computation, and
   `repeat_exact` (all 13 calls bit-identical: a hand-off race shows up here).
3. Then runs the correctness suites `test_chunk_gdn_fused_geometry.py` (host-only), `test_chunk_gdn_fused.py`
   (fused == phased bit-exact over many geometries, default dispatch, repeats), `test_chunk_gdn_prims.py`,
   `test_chunk_gated_delta_rule.py`. Any failure invalidates the attempt.

## Known ideas and constraints
- Recent history (in this tree): the SFPU forward-substitution solve for T_inv (#57445), its L loads through the MATH
  RISC data cache (4.5 -> 4.0 us per item, #58928), the producer pool and recalibrated model (#58891), the prep
  constants built in L1 instead of read from DRAM, g/beta read token-major without a host relayout. Proposed
  elsewhere, not in this tree: the WY inverse on the FPU instead of the SFPU (#58974), an exp replay body in the LLK
  (#58929, item to 16.2 us).
- A kernel speed-up only pays fully when the cost model knows about it: re-measure the constant it changes (prep ->
  `kWpUs`, scan -> `t_step_us` / `kChainSlopeUs`, hand-off -> depth terms, reader/kickoff -> `kFill*`) and check the
  pick against its runner-ups (`calibration/calib_picks.py`). Pinning a geometry with an explicit
  `ChunkGdnFusedProgramConfig` in a quick experiment is a good way to see what a different split would give.
- The profiler's own zones (`DeviceZoneScopedN`) in the kernels are what calibration reads; keep them.
- `T_inv` precision matters over the 64-chunk recurrence: the solve and every intermediate stay fp32.
