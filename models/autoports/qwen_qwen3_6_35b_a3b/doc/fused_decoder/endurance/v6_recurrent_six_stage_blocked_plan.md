# Recurrent state transport: blocked NoC experiment

The six-dispatch adapter is numerically correct but rejected for integration on measured performance. `v6_recurrent_six_stage_compilefix_all_device.json` passes all 48 combinations, 12 changing-input updates, matched 305 traced updates and restart determinism. Native warmed/traced medians are 0.248990440741/0.168848196045 ms; the candidate is 0.911760749295/0.894937319681 ms (5.30× slower in trace). Neither dispatch count nor exactness makes this a performance win.

The source exposes a testable transport hypothesis: each of 32 workers handles 16 key rows and a 32-value block across 32 heads. It gathers and scatters each row as two 64-byte pieces per head. This produces 32,768 original-state requests in each direction, despite moving only 2 MiB. Each state half-tile consists of sixteen consecutive such rows. There is also a scalar transpose on the reader and only 32 workers; these remain possible bottlenecks. No device profile currently apportions the 0.895 ms among these costs, so the transaction count is a hypothesis, not a diagnosed dominant cause.

`tests/v6_recurrent_six_stage_blocked_component.py` loads each 16-key × 32-value block using two 1,024-byte face reads per head, stages the complete block locally, then supplies the exact same grouped FP32 tiles to the frozen phase1 compute. The phase2 writer stages the inverse permutation and writes two 1,024-byte faces per head after consuming all sixteen output tiles. Each direction now issues 2,048 state requests (16× fewer) with unchanged 2 MiB payload. The new writer adds scalar local copies; this is an explicit tradeoff, fully included in the adapter timing.

Both compute strings, phase1 writer, phase2 reader, native 16-term/8-group reductions, FP32 pack points, public BF16 round/widen, delta/Q scaling, allocation shapes, and output alias API are unchanged. There are no new dispatches, arithmetic operations, host operations, persistent buffers or cached trajectories. Phase1 finishes every original-state read before phase2 begins; phase2 writes each state word once. Thus aliased and separate outputs retain the same ownership contract. Phase2 NoC writes finish before the generic operation completes.

Per-worker temporary CB usage rises from 44 to 104 KiB in phase1 and from 62 to 130 KiB in phase2 (debug: 108 and 150 KiB). Reader scratch is 17 × 4096 bytes, sufficient for 64 KiB state, 1024-byte K strip, 128-byte decay strip and up to 63 alignment bytes. Writer staging is 17 × 4096 bytes, sufficient for 64 KiB plus alignment. Only 32 workers participate. Persistent usage remains zero, and visible interleaved tensors retain their prior shapes/lifetimes. These figures are not a full-decoder capacity certification; full integration is prohibited until this component wins and its actual allocation context is tested.

CPU setup proves all 524,288 state words map identically through packetized gather, grouped tiles, and inverse scatter; each output word is written exactly once, and staging bounds cover worst-case alignment. It uses integer labels, so the address proof covers every raw bit pattern, including signed zero and nonfinite payloads. This proves transport mapping only, not device code generation or timing. Pinned autoflake 2.3.1, isort 5.13.2, and Black 23.10.1 leave the frozen files unchanged. `v6_recurrent_six_stage_blocked_transfer.json` pins sources and checks the exact source edit manifest; both arithmetic compute strings are byte-identical. All original b1b3/979e sources and measured evidence remain unchanged.

Parent-owned first gate:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_recurrent_six_stage_blocked_probe.py --run-device --cases actual227 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_recurrent_six_stage_blocked_actual_device.json
```

Then the complete exactness and matched evolving-state adapter benchmark:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_recurrent_six_stage_blocked_probe.py --run-device --both-round-modes --time --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_recurrent_six_stage_blocked_all_device.json
```

The harness retains all original runtime/native functions, conditions, bitwise checks, ordinary/random/zero/boundary and actual fixtures, alias/separate outputs, debug/normal execution, twelve changing-input updates, warmed timing, and traced matched trajectories. Only source pins, capacity metadata, and an additional CPU mapping proof differ. Timing includes the entire reader/staging/writer cost. Any bit difference is a rejection. A gain over the previous slow candidate alone is insufficient: the complete candidate must beat the same-run correct native traced baseline before further integration. If blocked transport remains much slower, record that result and retain native recurrence; fewer packets do not satisfy the user's performance gate.
