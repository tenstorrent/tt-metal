# AutoFix: advertised-context capacity fixture

Source-only investigation. No device command or implementation edit was made by this investigator. Runtime observations below come from the parent's saved log. The repaired capacity rerun remains the required proof.

## Starting evidence

Command: `bash models/autoports/qwen_qwen3_8_27b/tests/run_multichip_experiment.sh capacity_l0_s262143 --layer 0 --length 262143 --capacity --repeats 5`.

The failed run is preserved as `capacity_overreservation_failure.log`. It prints both `PREFILL_BEGIN` and `PREFILL_DONE`, then fails during the second prefill at `optimized_decoder.py:945`, allocating a concat result of **2,684,344,320 bytes**. Each of eight banks needs 335,544,320 bytes; the failing bank has 36,702,528 bytes free and a largest free block of 27,595,904 bytes. This is an allocator failure, not a context/shape validation rejection.

The original fixture additionally reserved **20,158,234,624 bytes per device**, including a synthetic 9 GiB activation allowance, while running real full-length activations. It also retained the previous full-length `result` while evaluating the next `result = prefill()`.

## Verified source causes

1. **The old output remains live across reassignment.** The right-hand forward executes before Python replaces the previous `result`. A BF16 `[1,262144,5120]` tiled output costs **2,684,354,560 bytes = 2.5 GiB per device**. The current runner now deletes the initial output after copying it to host (`run_multichip_decoder.py:214`) and deletes each timed output (`:282`). The host `pre` reference does not require retaining that device allocation.
2. **The empty activation allowance double-counted real allocations.** The current runner instead reads `capacity_probe_reserved_bytes_per_device` (`:141`): persistent full-model weights/state/terminal/constants plus 2 GiB for trace/CCL. Real full-length input, actual chunk activations, concat staging and output are then allocated normally. This remains conservative about persistent memory because the probe also has its real decoder weights/state and state snapshots live.
3. **Unaligned concat needs more than three hidden streams.** `OptimizedDecoder.prefill_forward` retains every chunk output in `outputs` and concatenates along sequence (`optimized_decoder.py:920-945`). With 2048-token chunks, S=262143 ends with a logical 2047-token tile-padded chunk. `concat.cpp:93-139` detects padding on the concat dimension and untilizes **all** chunks, recursively concatenates the row-major copies, then retilizes the result. `MassagedOperation::operator()` retains both `formatted_input` and `op_output` throughout `post_format` (`common/common.hpp:225-231`). Thus original tiled chunks, row-major chunk copies, row-major concatenation, and newly tiled output can overlap, in addition to the full input. The requested 2,684,344,320 bytes is exactly `262143 * 5120 * 2`, matching the row-major concat output rather than the padded tiled output. The direct unaligned tiled concat path is for width concat (`concat.cpp:340-347`), so it does not avoid this sequence concat staging.

`tilize_with_val_padding.cpp:139-152` calls its native tilize operation directly for this rank; there is no additional full-size host transfer needed. Public concat does not accept a preallocated output (`concat.cpp:272`). No native or decoder change is needed to remove the fixture's two excess reservations.

## Conservative planning budget

All numbers are **per device**, B1, current TP4 projection/cache policy, and a nominal DRAM capacity of 34,178,731,008 bytes. The parent's corrected constant allowance is 128 MiB, covering tiled replicated norms and convolution taps rather than the old 16 MiB estimate.

| Component | Bytes / allowance |
| --- | ---: |
| All decoder projection copies | 6,886,195,200 |
| Full-context KV across 16 full-attention layers | 2,281,701,376 |
| Recurrent and convolution state across 48 linear layers | 38,486,016 |
| Untied TP4 BF16 embedding and LM-head weights | 1,271,398,400 |
| Norm and other constant allowance | 134,217,728 |
| Persistent subtotal | **10,611,998,720** |
| Synthetic capacity-probe reserve: persistent + 2 GiB trace/CCL | **12,759,482,368**, rounded to whole BFP8 tiles |

The source-derived single-layer concat peak is **five hidden streams, approximately 12.5 GiB**, before other scratch or trace/CCL. A **15 GiB** total activation/trace allowance covers that peak plus 2 GiB trace/CCL and 0.5 GiB scratch; it assumes only the current layer's full-length input remains live. This gives planned total **26,718,126,080 bytes**, with nominal headroom **7,460,604,928 bytes**.

For a conservative future full-model caller that also retains the original embedding input while later layers consume a different full-length result, add another 2.5 GiB stream. This lifetime occurs in the runner's two-layer `--stack` closure (`run_multichip_decoder.py:195-204`), where outer `inputs` remains live. Recommend an **18 GiB** activation/trace allowance for that case: six streams (15 GiB), 2 GiB trace/CCL, and 1 GiB scratch. The resulting planned total is **29,939,351,552 bytes**, leaving nominal headroom **4,239,379,456 bytes**. A 12 GiB total allowance understates even the five-stream unaligned concat case.

These are planning bounds for the inspected decoder path, not a measured full-model peak. They assume earlier intermediate outputs are released and the future terminal path does not materialize full-sequence vocabulary logits. Keep the capacity probe's synthetic reserve at persistent plus 2 GiB: reserving the full activation allowance again would reintroduce double counting. A separate stack/retained-input control can exercise the additional stream.

The runner's actual `trace_region_size` is only 40,000,000 bytes (`run_multichip_decoder.py:66`). Native `mesh_trace.cpp:56-80` checks total live padded trace bytes against this region across all banks. A synthetic 2 GiB trace/CCL allowance does **not** prove a 64-layer trace fits the runner's configured region. Full-model trace size and persistent CCL buffers still need measurement/configuration at that stage. Likewise, the post-prefill `get_memory_view` snapshot is not a peak allocation trace; use observed per-bank free/largest-free space when diagnosing fragmentation.

## Focused verification and status

Rerun the original S262143 command with the lifetime/reservation corrections, preserving the 262144 context contract and identical policy. Verify first and warm prefill, final-position decode, trace replay, and baseline PCC. Then run the serialized `run_multichip_capacity.py` controls for layers 0/3 and lengths 262143/262144; the exact-capacity length uses prefill-only because there is no further legal decode position.

The parent reports that the corrected linear-layer probes passed at both lengths and has adopted the conservative 18 GiB plan. Full-attention probes are in progress. The additional required retained-input control is a real linear/full stack at S262143 with the same persistent reservation: `bash models/autoports/qwen_qwen3_8_27b/tests/run_multichip_experiment.sh capacity_stack_l0_s262143 --layer 0 --length 262143 --capacity --stack --repeats 5`, preceded by the same command with the name suffixed `_baseline` and `--baseline`. This directly exercises the original prompt input remaining live while the second layer consumes a separate full-length output. It is pending at report time; the investigator has not independently run or inspected a successful stack result.

Prediction: removing the retained output and synthetic activation duplication permits the same native unaligned concat workload to complete. A repeated OOM should be investigated from the new allocation site and actual per-bank state, rather than reducing supported context or adding public alignment restrictions.

**Verdict:** both fixture over-reservations are verified by source and the failure location. The first successful full-length prefill refutes a blanket claim that this decoder cannot accept the advertised unaligned length. No decoder capability reduction is justified. The parent reports successful repaired linear-layer controls; full-attention and retained-input stack results plus a measured full-model trace/activation peak remain outstanding in this source-only report.

## Parent hardware follow-up

Resolution: fixed fixture lifetime/accounting, capability preserved. All eight
single-kind baseline/mesh capacity runs pass (`capacity_runs.log`). The two
real-layer stack also passes at262143 prefill plus final-position decode
(`capacity_stack_s262143{,_baseline}.json`), minimum PCC0.9999313, with
12,759,483,008 additional bytes reserved on every chip. The original input
remains live across the second decoder, covering the sixth-stream lifetime.
Current allocator capacity after the40MB trace region is34,138,688,512B;
final conservative plan29,939,351,552B leaves4,199,336,960B headroom.
