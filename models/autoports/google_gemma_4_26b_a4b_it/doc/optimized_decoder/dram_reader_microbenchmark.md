# Exact-input DRAM reader probe

Status: implementation, CPU checks, parent-executed isolated device trials and all22 whole-layer controls are complete. [Results](reader_layer_results.md) retain the existing reader policy; no material layer gain was measured. [The probe](../../tests/probe_optimized_dram_readers.py) implements the isolated part of installed optimize 0.1.14 OPT-015. Runtime source is untouched.

Capture the selected decoder's actual first continuation token after its real 4096-token prefill. The existing audited harness checks the HF prefill/decode PCC, exact trace replay, program cache, and transport metadata. The probe retains references during production `ttnn.linear` calls, then reads input, quantized weight, and output tensors after the harness finishes. It stores the exact input/output dtypes and every compute-config flag from those calls. Source/fixture hashes bind the saved `.pt`; a runtime change requires recapture. CPU `.pt` files are local ignored artifacts, with provenance copied into the compact JSON report.

The four roles are packed direct QKV, attention output, packed shared gate/up, and shared down. Decode output-projection input is explicitly required to be BF16. The discovery code unwraps an optional `.decode_source` prefill wrapper, while measured calls remain the actual decode linear. Weights are read back after production BFP quantization and repacked with whole-tile zero padding; exact input and weight equality is checked before timing. Outputs are sliced to logical N for PCC against both the actual production result and the CPU FP32 dot product of the captured quantized operands. A failing candidate remains recorded and cannot be selected merely because timing succeeded.

| Role | Logical K×N, sliding / full | Common physical N for readers 1/2/3 | Input storage cores | K block families |
| --- | --- | ---: | ---: | --- |
| QKV | 2816×8192 / 2816×9216 | 9216 | 8 | 1 and 11, ranked separately |
| Output | 4096×2816 / 8192×2816 | 3072 | 8 | 16 |
| Shared gate/up | 2816×4224, both | 4608 | 8 | 11 |
| Shared down | 2112×2816, both | 3072 | 6 | 11 |

Input/output storage, padded weights, fidelity, compute flags, and K block stay fixed within each reader family. The QKV K1 family is a bounded legal-small-block control for previously observed K11 L1 failures; reader1/K1 is not compared against reader2/K11 as a reader-only speed result. An isolated K11 allocation may differ from the layer's live-buffer allocation, so every winner still needs layer integration.

Each native trace contains one `ttnn.linear`, using the same preallocated output across reader traces. Upload, resharding, weight padding, output slicing and readback are outside the measured windows. First-use compilation, trace capture and two warm trace executions are excluded. Six deterministic reader permutations give each reader each position twice; each timing block executes 20 replays and checks exact repeated output afterward. The report labels synchronized host-wall samples separately from device durations. CPU postprocessing requires exactly 20 native matmul rows per marked window, checks actual input/weight/output dtype, fidelity and reader count, and derives device distributions, round-median spread, logical/physical weight GB/s, and percent of the explicit 512 GB/s peak assumption. These are weight-payload bandwidth estimates, not hardware traffic counters.

Legality comes from `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`: counts 1..3, Blackhole for multiple readers, NOC0, per-bank width divisible by the reader count, K/storage divisibility, and available reader placement. Setup/L1 exceptions retain the full error. Unexpected exceptions abort. The source requires no universal 4 KiB row or 64-core input minimum; those two values are reported as screening heuristics only.

The report includes weight tile payload/header bytes, per-bank tiles, per-reader row bytes, activation storage cores, and weight triple-buffer estimates. Source at `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:128` issues one split-reader request per K row with `NOC_MAX_BURST_SIZE`; Blackhole defines that as 16384 bytes in `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:286`. The single-reader page divisor follows `get_max_page_size_and_num_pages` in `ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp:357`. Reported burst/page counts are source-derived request estimates. Native BRISC/NCRISC/TRISC duration medians are retained for classifying a flat case; they do not prove a NoC contention cause by themselves.

Run the following after the selected runtime is frozen, once per `layer=0` and `layer=5`, under the parent-owned hardware procedure. The commands below are a plan, not an execution claim.

```bash
layer=0
evidence=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder
module=models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_dram_readers

python_env/bin/python -m "$module" --layer "$layer" \
  --input-fixture "$evidence/actual_text_layer${layer}_4096_128.pt" \
  --capture "$evidence/dram_reader_layer${layer}.inputs.pt" --capture-only \
  --report "$evidence/dram_reader_layer${layer}.capture.json"

python_env/bin/python -m tracy -r -p -v --op-support-count 100000 \
  --no-op-info-cache --disable-device-data-dump-to-files \
  --disable-device-data-push-to-tracy \
  -o "$evidence/tracy/dram_reader_layer${layer}/raw" -n readers \
  -m "$module" --layer "$layer" \
  --capture "$evidence/dram_reader_layer${layer}.inputs.pt" --profile \
  --report "$evidence/dram_reader_layer${layer}.json"
```

Use the generated native `ops_perf_results_*.csv` path from that Tracy run explicitly:

```bash
python_env/bin/python -m "$module" \
  --summarize-csv "$evidence/tracy/dram_reader_layer${layer}/ops.csv" \
  --report "$evidence/dram_reader_layer${layer}.json"
```

Copy the exact generated CSV to the shown `ops.csv` path first, recording its source path; the parser hashes it. `--roles output` or another role list can bound a repeat without recapturing. The default probe compares all four roles and both QKV K blocks. The benchmark JSON must be inspected for `all_legal_accuracy_passed`, exact invalidity evidence, native device distributions, and whole-layer integration status; a host-only report remains explicitly device-pending.

CPU verification performed: three `unittest` checks cover common all-reader padding/storage, balanced permutations, and rejection of wrong dtype/incomplete native windows. `py_compile` and repository `pre-commit run --files` pass for the two new test files. No hardware was run by the probe author.
