# Benchmark: Qwen/Qwen3.8-27B

Status: completed. Accuracy concurrency: 32.

| Benchmark | Samples / full | Metric | Subset % | Published full % | Difference pp | Source |
|---|---:|---|---:|---:|---:|---|
| gpqa_diamond_cot_zeroshot | 10 / 198 | exact_match,strict-match | 0.00 | — | — | Unavailable |
| gpqa_diamond_cot_zeroshot | 10 / 198 | exact_match,flexible-extract | 50.00 | 89.20 | -39.20 | [reference](https://huggingface.co/Qwen/Qwen3.8-27B) |

| Profile | Concurrent requests | Server slots | ISL / OSL | TTFT ms | TPOT ms | Decode tokens/s/user | Output tokens/s | Prefill FLOP roofline % (est.) | Decode DRAM roofline % (est.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single user | 1 | 1 | 4096 / 128 | 427.50 | 27.60 | 36.23 | 32.54 | 18.44 | 27.91 |
| 8 users | 8 | 8 | 4096 / 128 | 7073.42 | 48.41 | 20.66 | 77.44 | 8.74 | 19.38 |
| 16 users | 16 | 16 | 4096 / 128 | 14476.24 | 73.87 | 13.54 | 85.84 | 8.52 | 21.44 |
| 32 users | 32 | 32 | 4096 / 128 | 30836.02 | 114.21 | 8.76 | 90.33 | 8.01 | 21.01 |

Scores use fixed subsets; published figures cover the full dataset. Missing references are shown as unavailable. The bringup owner decides whether these results meet their needs.

Roofline estimates divide modeled work by full-phase elapsed wall time and the participating hardware’s peak rate. Both phases are required for every serving profile; — indicates incomplete accounting. HTTP concurrency is not a fixed device batch size.

Reference protocol notes:
- gpqa_diamond_cot_zeroshot, exact_match,flexible-extract: Published exact-model GPQA Diamond score; local10/198 zero-shot lm-eval variant and output cap may differ. Published complete evaluation recipe not established from model card.

## Run details

Subset: `fb01e42fbed84a24ad4f7c57b6355afa2abbd27e7cc3a27418e20b861fd6b5d7`. Configuration: [run_config.json](run_config.json).

| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |
|---|---:|---:|---:|---:|
| [gpqa_diamond_cot_zeroshot](gpqa_diamond_cot_zeroshot/results.json) | 10 | 4 (40.0%) | 4 | 2226.0 |

Accuracy tasks share one request pool; their wall times refer to the same interval.

| Concurrency | Completed / requested | Wall seconds | Requests/s |
|---:|---:|---:|---:|
| [1](perf-b1.json) | 8 / 8 | 31.47 | 0.25 |
| [8](perf-b8.json) | 24 / 24 | 39.67 | 0.61 |
| [16](perf-b16.json) | 48 / 48 | 71.57 | 0.67 |
| [32](perf-b32.json) | 96 / 96 | 136.03 | 0.71 |

| Concurrency | Latency | Mean ms | Median ms | p95 ms | p99 ms |
|---:|---|---:|---:|---:|---:|
| 1 | TTFT | 427.50 | 426.84 | 432.35 | 433.36 |
| 1 | TPOT | 27.60 | 27.60 | 27.61 | 27.62 |
| 1 | ITL | 27.60 | 27.62 | 28.14 | 28.56 |
| 1 | E2EL | 3932.80 | 3932.69 | 3938.29 | 3939.42 |
| 8 | TTFT | 7073.42 | 7075.34 | 7085.29 | 7085.59 |
| 8 | TPOT | 48.41 | 48.39 | 48.64 | 48.65 |
| 8 | ITL | 48.41 | 47.09 | 48.08 | 49.01 |
| 8 | E2EL | 13221.89 | 13224.20 | 13226.43 | 13227.27 |
| 16 | TTFT | 14476.24 | 14476.61 | 14494.58 | 14495.05 |
| 16 | TPOT | 73.87 | 73.84 | 74.11 | 74.24 |
| 16 | ITL | 73.87 | 72.35 | 73.61 | 76.76 |
| 16 | E2EL | 23857.18 | 23860.03 | 23868.22 | 23869.34 |
| 32 | TTFT | 30836.02 | 30831.44 | 30862.02 | 30863.06 |
| 32 | TPOT | 114.21 | 114.20 | 114.21 | 115.07 |
| 32 | ITL | 114.21 | 112.37 | 113.66 | 118.94 |
| 32 | E2EL | 45340.43 | 45329.31 | 45364.39 | 45364.89 |

Server configuration for 1 concurrent request(s): [record](perf-b1-server.json).

Server configuration for 8 concurrent request(s): [record](perf-b8-server.json).

Server configuration for 16 concurrent request(s): [record](perf-b16-server.json).

Server configuration for 32 concurrent request(s): [record](perf-b32-server.json).

Roofline inputs: [roofline.json](roofline.json).

- Concurrency 32, prefill: 92.1272 s. Stage11 full64 Qwen3.8-27B useful work / estimated DRAM accounting.

This is a model, never a hardware counter. API inputs describe ONE scheduler
prefill/decode invocation, not independent requests whose weight costs may be
summed. Sum returned work once for each measured invocation on the TP4 mesh.

Prefill derivation follows tests/roofline.py: 48 linear +16 full layers,
H5120, I17408,24 Q heads,4 KV heads,D256; linear16 QK/48 value heads,D128.
Dense projections cost two FLOPs/MAC. Causal QK+AV use actual prefix start and
length: pairs=L*start+L*(L+1)/2. Linear recurrence uses7*48*128*128 useful
ops/token; convolution/gates/norms/softmax use the inherited scalar estimates,
counting a transcendental as one operation. This is useful algorithmic work,
not the extra chunk-GDN implementation work or padding. Final norm/LM head runs
only on each row's last token when final=True, matching suppressed intermediate
prefill in generator_vllm.py. No vision encoder, MTP or MoE work is counted.

Decode DRAM derives physical tiled weights from multichip_decoder.py:
4 DRAM shards,8 banks,32-wide tiles; mix padding uses2 readers (linear local
logical width4160 includes padded a/b; physical4608; full3584). out K1536,N5120;
gateup K5120,N8704; down K4352,N5120. BFP4 tile576 bytes/1024 values. B1 uses
DRAM-sharded projections and L1 pooled all-reduce; B8/16/32 compact users into
one M tile and use native fused out/down reduce-scatter. Both read each weight
representation once per invocation, NOT once per live user. Separate gate/up
prefill copies are not counted as decode reads. Native views reuse the same
weights. BF8 LM head [5120,248320] uses1088-byte tiles and one shared M tile;
embedding is a sharded BF16 row lookup. Compulsory weights are a lower bound:
matmul block/core rereads are unknown and excluded, never inferred from time.

48 recurrent layers: state is TP-sharded FP32 [capacity,48,128,128] and BF16
history[capacity,3,10240]. Unmasked state uses in-place kernel read+write (2 transfers). Above9 rows
(the110-core grid divided by12 local value heads), split outputs concatenate
then copy home, adding4 transfers. Partial occupancy disables in-place state:
kernel read+write, preservation where read/read/write, copy read+write total7
transfers, plus2 concat transfers above9 rows.
Count physical capacity even for inactive rows since traces execute their state
math. Additional concatenation/slicing/transposition transfers are excluded.
16 attention layers: one BFP8 K/V prefix read per live row and one token write,
2176 bytes/token across four ASICs; this ignores SDPA rereads and masked slots.
Prefix read includes the current token. The write is a payload lower bound, not
an estimate of any read/modify/write amplification of a physical tile.

Additional explicit DRAM traffic: fused B>=8 out/down intermediates each write
and read a BF16 [32,5120] tensor per ASIC per layer; their outputs and column
all-gathers are L1 and are not charged as DRAM. B1 pooled all-reduce scratch is
L1, so no DRAM network-byte fiction is added. Terminal forces hidden to DRAM,
normalizes there and writes BF16 vocab-sharded logits (then sampler reads), all
padded to32 rows. Include final norm weights once per ASIC and embedding reads.
Decoder norms/activations mostly reside in L1: unknown spill and repeated
intermediate reads and sampling/CCL scratch are omitted. At each4096-token
decode boundary, include recurrent/history read+write for the prefix snapshot. Thus decode bytes are an explicit LOWER-BOUND estimate,
not measured bandwidth nor complete byte-exact operation accounting. All those
operations, communication and host gaps MUST remain inside the timing denominator.

Peak: four participating ASICs,120 physical Tensix cores each, nominal1.35GHz,
4096 LoFi FLOP/core/cycle;4*512GB/s GDDR6. Hardware source:
https://docs.tenstorrent.com/aibs/blackhole/p300.html (2 ASICs/card,1024GB/s/card).
4096 LoFi factor: ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp
performance model and prior tests/report_optimized_multichip.py. Use physical
peak, not the110-core runtime grid. LoFi is the dense BFP4/BFP8 projection peak;
HiFi2 head and HiFi4 sensitive kernels are slower. This ceiling is NOT an
achievable mixed-fidelity ceiling; see prefill_peak() for a fidelity-weighted
ideal ceiling including HiFi2 terminal and HiFi4 recurrence/scalar work.
 Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; matmul_device_operation.cpp performance model. Timing: Actual output-ready boundaries with outgoing-phase overlap ownership; explicitly tagged drain/state-apply/consumption spans; admitted-to-finished active host gaps; request-free residuals excluded; no added synchronization. [Evidence](phase-accounting-b32.json).
- Concurrency 32, decode: 43.8398 s. Stage11 full64 Qwen3.8-27B useful work / estimated DRAM accounting.

This is a model, never a hardware counter. API inputs describe ONE scheduler
prefill/decode invocation, not independent requests whose weight costs may be
summed. Sum returned work once for each measured invocation on the TP4 mesh.

Prefill derivation follows tests/roofline.py: 48 linear +16 full layers,
H5120, I17408,24 Q heads,4 KV heads,D256; linear16 QK/48 value heads,D128.
Dense projections cost two FLOPs/MAC. Causal QK+AV use actual prefix start and
length: pairs=L*start+L*(L+1)/2. Linear recurrence uses7*48*128*128 useful
ops/token; convolution/gates/norms/softmax use the inherited scalar estimates,
counting a transcendental as one operation. This is useful algorithmic work,
not the extra chunk-GDN implementation work or padding. Final norm/LM head runs
only on each row's last token when final=True, matching suppressed intermediate
prefill in generator_vllm.py. No vision encoder, MTP or MoE work is counted.

Decode DRAM derives physical tiled weights from multichip_decoder.py:
4 DRAM shards,8 banks,32-wide tiles; mix padding uses2 readers (linear local
logical width4160 includes padded a/b; physical4608; full3584). out K1536,N5120;
gateup K5120,N8704; down K4352,N5120. BFP4 tile576 bytes/1024 values. B1 uses
DRAM-sharded projections and L1 pooled all-reduce; B8/16/32 compact users into
one M tile and use native fused out/down reduce-scatter. Both read each weight
representation once per invocation, NOT once per live user. Separate gate/up
prefill copies are not counted as decode reads. Native views reuse the same
weights. BF8 LM head [5120,248320] uses1088-byte tiles and one shared M tile;
embedding is a sharded BF16 row lookup. Compulsory weights are a lower bound:
matmul block/core rereads are unknown and excluded, never inferred from time.

48 recurrent layers: state is TP-sharded FP32 [capacity,48,128,128] and BF16
history[capacity,3,10240]. Unmasked state uses in-place kernel read+write (2 transfers). Above9 rows
(the110-core grid divided by12 local value heads), split outputs concatenate
then copy home, adding4 transfers. Partial occupancy disables in-place state:
kernel read+write, preservation where read/read/write, copy read+write total7
transfers, plus2 concat transfers above9 rows.
Count physical capacity even for inactive rows since traces execute their state
math. Additional concatenation/slicing/transposition transfers are excluded.
16 attention layers: one BFP8 K/V prefix read per live row and one token write,
2176 bytes/token across four ASICs; this ignores SDPA rereads and masked slots.
Prefix read includes the current token. The write is a payload lower bound, not
an estimate of any read/modify/write amplification of a physical tile.

Additional explicit DRAM traffic: fused B>=8 out/down intermediates each write
and read a BF16 [32,5120] tensor per ASIC per layer; their outputs and column
all-gathers are L1 and are not charged as DRAM. B1 pooled all-reduce scratch is
L1, so no DRAM network-byte fiction is added. Terminal forces hidden to DRAM,
normalizes there and writes BF16 vocab-sharded logits (then sampler reads), all
padded to32 rows. Include final norm weights once per ASIC and embedding reads.
Decoder norms/activations mostly reside in L1: unknown spill and repeated
intermediate reads and sampling/CCL scratch are omitted. At each4096-token
decode boundary, include recurrent/history read+write for the prefix snapshot. Thus decode bytes are an explicit LOWER-BOUND estimate,
not measured bandwidth nor complete byte-exact operation accounting. All those
operations, communication and host gaps MUST remain inside the timing denominator.

Peak: four participating ASICs,120 physical Tensix cores each, nominal1.35GHz,
4096 LoFi FLOP/core/cycle;4*512GB/s GDDR6. Hardware source:
https://docs.tenstorrent.com/aibs/blackhole/p300.html (2 ASICs/card,1024GB/s/card).
4096 LoFi factor: ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp
performance model and prior tests/report_optimized_multichip.py. Use physical
peak, not the110-core runtime grid. LoFi is the dense BFP4/BFP8 projection peak;
HiFi2 head and HiFi4 sensitive kernels are slower. This ceiling is NOT an
achievable mixed-fidelity ceiling; see prefill_peak() for a fidelity-weighted
ideal ceiling including HiFi2 terminal and HiFi4 recurrence/scalar work.
 Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; matmul_device_operation.cpp performance model. Timing: Actual output-ready boundaries with outgoing-phase overlap ownership; explicitly tagged drain/state-apply/consumption spans; admitted-to-finished active host gaps; request-free residuals excluded; no added synchronization. [Evidence](phase-accounting-b32.json).
- Concurrency 16, prefill: 43.2669 s. Stage11 full64 Qwen3.8-27B useful work / estimated DRAM accounting.

This is a model, never a hardware counter. API inputs describe ONE scheduler
prefill/decode invocation, not independent requests whose weight costs may be
summed. Sum returned work once for each measured invocation on the TP4 mesh.

Prefill derivation follows tests/roofline.py: 48 linear +16 full layers,
H5120, I17408,24 Q heads,4 KV heads,D256; linear16 QK/48 value heads,D128.
Dense projections cost two FLOPs/MAC. Causal QK+AV use actual prefix start and
length: pairs=L*start+L*(L+1)/2. Linear recurrence uses7*48*128*128 useful
ops/token; convolution/gates/norms/softmax use the inherited scalar estimates,
counting a transcendental as one operation. This is useful algorithmic work,
not the extra chunk-GDN implementation work or padding. Final norm/LM head runs
only on each row's last token when final=True, matching suppressed intermediate
prefill in generator_vllm.py. No vision encoder, MTP or MoE work is counted.

Decode DRAM derives physical tiled weights from multichip_decoder.py:
4 DRAM shards,8 banks,32-wide tiles; mix padding uses2 readers (linear local
logical width4160 includes padded a/b; physical4608; full3584). out K1536,N5120;
gateup K5120,N8704; down K4352,N5120. BFP4 tile576 bytes/1024 values. B1 uses
DRAM-sharded projections and L1 pooled all-reduce; B8/16/32 compact users into
one M tile and use native fused out/down reduce-scatter. Both read each weight
representation once per invocation, NOT once per live user. Separate gate/up
prefill copies are not counted as decode reads. Native views reuse the same
weights. BF8 LM head [5120,248320] uses1088-byte tiles and one shared M tile;
embedding is a sharded BF16 row lookup. Compulsory weights are a lower bound:
matmul block/core rereads are unknown and excluded, never inferred from time.

48 recurrent layers: state is TP-sharded FP32 [capacity,48,128,128] and BF16
history[capacity,3,10240]. Unmasked state uses in-place kernel read+write (2 transfers). Above9 rows
(the110-core grid divided by12 local value heads), split outputs concatenate
then copy home, adding4 transfers. Partial occupancy disables in-place state:
kernel read+write, preservation where read/read/write, copy read+write total7
transfers, plus2 concat transfers above9 rows.
Count physical capacity even for inactive rows since traces execute their state
math. Additional concatenation/slicing/transposition transfers are excluded.
16 attention layers: one BFP8 K/V prefix read per live row and one token write,
2176 bytes/token across four ASICs; this ignores SDPA rereads and masked slots.
Prefix read includes the current token. The write is a payload lower bound, not
an estimate of any read/modify/write amplification of a physical tile.

Additional explicit DRAM traffic: fused B>=8 out/down intermediates each write
and read a BF16 [32,5120] tensor per ASIC per layer; their outputs and column
all-gathers are L1 and are not charged as DRAM. B1 pooled all-reduce scratch is
L1, so no DRAM network-byte fiction is added. Terminal forces hidden to DRAM,
normalizes there and writes BF16 vocab-sharded logits (then sampler reads), all
padded to32 rows. Include final norm weights once per ASIC and embedding reads.
Decoder norms/activations mostly reside in L1: unknown spill and repeated
intermediate reads and sampling/CCL scratch are omitted. At each4096-token
decode boundary, include recurrent/history read+write for the prefix snapshot. Thus decode bytes are an explicit LOWER-BOUND estimate,
not measured bandwidth nor complete byte-exact operation accounting. All those
operations, communication and host gaps MUST remain inside the timing denominator.

Peak: four participating ASICs,120 physical Tensix cores each, nominal1.35GHz,
4096 LoFi FLOP/core/cycle;4*512GB/s GDDR6. Hardware source:
https://docs.tenstorrent.com/aibs/blackhole/p300.html (2 ASICs/card,1024GB/s/card).
4096 LoFi factor: ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp
performance model and prior tests/report_optimized_multichip.py. Use physical
peak, not the110-core runtime grid. LoFi is the dense BFP4/BFP8 projection peak;
HiFi2 head and HiFi4 sensitive kernels are slower. This ceiling is NOT an
achievable mixed-fidelity ceiling; see prefill_peak() for a fidelity-weighted
ideal ceiling including HiFi2 terminal and HiFi4 recurrence/scalar work.
 Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; matmul_device_operation.cpp performance model. Timing: Actual output-ready boundaries with outgoing-phase overlap ownership; explicitly tagged drain/state-apply/consumption spans; admitted-to-finished active host gaps; request-free residuals excluded; no added synchronization. [Evidence](phase-accounting-b16.json).
- Concurrency 16, decode: 28.2855 s. Stage11 full64 Qwen3.8-27B useful work / estimated DRAM accounting.

This is a model, never a hardware counter. API inputs describe ONE scheduler
prefill/decode invocation, not independent requests whose weight costs may be
summed. Sum returned work once for each measured invocation on the TP4 mesh.

Prefill derivation follows tests/roofline.py: 48 linear +16 full layers,
H5120, I17408,24 Q heads,4 KV heads,D256; linear16 QK/48 value heads,D128.
Dense projections cost two FLOPs/MAC. Causal QK+AV use actual prefix start and
length: pairs=L*start+L*(L+1)/2. Linear recurrence uses7*48*128*128 useful
ops/token; convolution/gates/norms/softmax use the inherited scalar estimates,
counting a transcendental as one operation. This is useful algorithmic work,
not the extra chunk-GDN implementation work or padding. Final norm/LM head runs
only on each row's last token when final=True, matching suppressed intermediate
prefill in generator_vllm.py. No vision encoder, MTP or MoE work is counted.

Decode DRAM derives physical tiled weights from multichip_decoder.py:
4 DRAM shards,8 banks,32-wide tiles; mix padding uses2 readers (linear local
logical width4160 includes padded a/b; physical4608; full3584). out K1536,N5120;
gateup K5120,N8704; down K4352,N5120. BFP4 tile576 bytes/1024 values. B1 uses
DRAM-sharded projections and L1 pooled all-reduce; B8/16/32 compact users into
one M tile and use native fused out/down reduce-scatter. Both read each weight
representation once per invocation, NOT once per live user. Separate gate/up
prefill copies are not counted as decode reads. Native views reuse the same
weights. BF8 LM head [5120,248320] uses1088-byte tiles and one shared M tile;
embedding is a sharded BF16 row lookup. Compulsory weights are a lower bound:
matmul block/core rereads are unknown and excluded, never inferred from time.

48 recurrent layers: state is TP-sharded FP32 [capacity,48,128,128] and BF16
history[capacity,3,10240]. Unmasked state uses in-place kernel read+write (2 transfers). Above9 rows
(the110-core grid divided by12 local value heads), split outputs concatenate
then copy home, adding4 transfers. Partial occupancy disables in-place state:
kernel read+write, preservation where read/read/write, copy read+write total7
transfers, plus2 concat transfers above9 rows.
Count physical capacity even for inactive rows since traces execute their state
math. Additional concatenation/slicing/transposition transfers are excluded.
16 attention layers: one BFP8 K/V prefix read per live row and one token write,
2176 bytes/token across four ASICs; this ignores SDPA rereads and masked slots.
Prefix read includes the current token. The write is a payload lower bound, not
an estimate of any read/modify/write amplification of a physical tile.

Additional explicit DRAM traffic: fused B>=8 out/down intermediates each write
and read a BF16 [32,5120] tensor per ASIC per layer; their outputs and column
all-gathers are L1 and are not charged as DRAM. B1 pooled all-reduce scratch is
L1, so no DRAM network-byte fiction is added. Terminal forces hidden to DRAM,
normalizes there and writes BF16 vocab-sharded logits (then sampler reads), all
padded to32 rows. Include final norm weights once per ASIC and embedding reads.
Decoder norms/activations mostly reside in L1: unknown spill and repeated
intermediate reads and sampling/CCL scratch are omitted. At each4096-token
decode boundary, include recurrent/history read+write for the prefix snapshot. Thus decode bytes are an explicit LOWER-BOUND estimate,
not measured bandwidth nor complete byte-exact operation accounting. All those
operations, communication and host gaps MUST remain inside the timing denominator.

Peak: four participating ASICs,120 physical Tensix cores each, nominal1.35GHz,
4096 LoFi FLOP/core/cycle;4*512GB/s GDDR6. Hardware source:
https://docs.tenstorrent.com/aibs/blackhole/p300.html (2 ASICs/card,1024GB/s/card).
4096 LoFi factor: ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp
performance model and prior tests/report_optimized_multichip.py. Use physical
peak, not the110-core runtime grid. LoFi is the dense BFP4/BFP8 projection peak;
HiFi2 head and HiFi4 sensitive kernels are slower. This ceiling is NOT an
achievable mixed-fidelity ceiling; see prefill_peak() for a fidelity-weighted
ideal ceiling including HiFi2 terminal and HiFi4 recurrence/scalar work.
 Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; matmul_device_operation.cpp performance model. Timing: Actual output-ready boundaries with outgoing-phase overlap ownership; explicitly tagged drain/state-apply/consumption spans; admitted-to-finished active host gaps; request-free residuals excluded; no added synchronization. [Evidence](phase-accounting-b16.json).
- Concurrency 8, prefill: 21.1042 s. Stage11 full64 Qwen3.8-27B useful work / estimated DRAM accounting.

This is a model, never a hardware counter. API inputs describe ONE scheduler
prefill/decode invocation, not independent requests whose weight costs may be
summed. Sum returned work once for each measured invocation on the TP4 mesh.

Prefill derivation follows tests/roofline.py: 48 linear +16 full layers,
H5120, I17408,24 Q heads,4 KV heads,D256; linear16 QK/48 value heads,D128.
Dense projections cost two FLOPs/MAC. Causal QK+AV use actual prefix start and
length: pairs=L*start+L*(L+1)/2. Linear recurrence uses7*48*128*128 useful
ops/token; convolution/gates/norms/softmax use the inherited scalar estimates,
counting a transcendental as one operation. This is useful algorithmic work,
not the extra chunk-GDN implementation work or padding. Final norm/LM head runs
only on each row's last token when final=True, matching suppressed intermediate
prefill in generator_vllm.py. No vision encoder, MTP or MoE work is counted.

Decode DRAM derives physical tiled weights from multichip_decoder.py:
4 DRAM shards,8 banks,32-wide tiles; mix padding uses2 readers (linear local
logical width4160 includes padded a/b; physical4608; full3584). out K1536,N5120;
gateup K5120,N8704; down K4352,N5120. BFP4 tile576 bytes/1024 values. B1 uses
DRAM-sharded projections and L1 pooled all-reduce; B8/16/32 compact users into
one M tile and use native fused out/down reduce-scatter. Both read each weight
representation once per invocation, NOT once per live user. Separate gate/up
prefill copies are not counted as decode reads. Native views reuse the same
weights. BF8 LM head [5120,248320] uses1088-byte tiles and one shared M tile;
embedding is a sharded BF16 row lookup. Compulsory weights are a lower bound:
matmul block/core rereads are unknown and excluded, never inferred from time.

48 recurrent layers: state is TP-sharded FP32 [capacity,48,128,128] and BF16
history[capacity,3,10240]. Unmasked state uses in-place kernel read+write (2 transfers). Above9 rows
(the110-core grid divided by12 local value heads), split outputs concatenate
then copy home, adding4 transfers. Partial occupancy disables in-place state:
kernel read+write, preservation where read/read/write, copy read+write total7
transfers, plus2 concat transfers above9 rows.
Count physical capacity even for inactive rows since traces execute their state
math. Additional concatenation/slicing/transposition transfers are excluded.
16 attention layers: one BFP8 K/V prefix read per live row and one token write,
2176 bytes/token across four ASICs; this ignores SDPA rereads and masked slots.
Prefix read includes the current token. The write is a payload lower bound, not
an estimate of any read/modify/write amplification of a physical tile.

Additional explicit DRAM traffic: fused B>=8 out/down intermediates each write
and read a BF16 [32,5120] tensor per ASIC per layer; their outputs and column
all-gathers are L1 and are not charged as DRAM. B1 pooled all-reduce scratch is
L1, so no DRAM network-byte fiction is added. Terminal forces hidden to DRAM,
normalizes there and writes BF16 vocab-sharded logits (then sampler reads), all
padded to32 rows. Include final norm weights once per ASIC and embedding reads.
Decoder norms/activations mostly reside in L1: unknown spill and repeated
intermediate reads and sampling/CCL scratch are omitted. At each4096-token
decode boundary, include recurrent/history read+write for the prefix snapshot. Thus decode bytes are an explicit LOWER-BOUND estimate,
not measured bandwidth nor complete byte-exact operation accounting. All those
operations, communication and host gaps MUST remain inside the timing denominator.

Peak: four participating ASICs,120 physical Tensix cores each, nominal1.35GHz,
4096 LoFi FLOP/core/cycle;4*512GB/s GDDR6. Hardware source:
https://docs.tenstorrent.com/aibs/blackhole/p300.html (2 ASICs/card,1024GB/s/card).
4096 LoFi factor: ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp
performance model and prior tests/report_optimized_multichip.py. Use physical
peak, not the110-core runtime grid. LoFi is the dense BFP4/BFP8 projection peak;
HiFi2 head and HiFi4 sensitive kernels are slower. This ceiling is NOT an
achievable mixed-fidelity ceiling; see prefill_peak() for a fidelity-weighted
ideal ceiling including HiFi2 terminal and HiFi4 recurrence/scalar work.
 Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; matmul_device_operation.cpp performance model. Timing: Actual output-ready boundaries with outgoing-phase overlap ownership; explicitly tagged drain/state-apply/consumption spans; admitted-to-finished active host gaps; request-free residuals excluded; no added synchronization. [Evidence](phase-accounting-b8.json).
- Concurrency 8, decode: 18.5373 s. Stage11 full64 Qwen3.8-27B useful work / estimated DRAM accounting.

This is a model, never a hardware counter. API inputs describe ONE scheduler
prefill/decode invocation, not independent requests whose weight costs may be
summed. Sum returned work once for each measured invocation on the TP4 mesh.

Prefill derivation follows tests/roofline.py: 48 linear +16 full layers,
H5120, I17408,24 Q heads,4 KV heads,D256; linear16 QK/48 value heads,D128.
Dense projections cost two FLOPs/MAC. Causal QK+AV use actual prefix start and
length: pairs=L*start+L*(L+1)/2. Linear recurrence uses7*48*128*128 useful
ops/token; convolution/gates/norms/softmax use the inherited scalar estimates,
counting a transcendental as one operation. This is useful algorithmic work,
not the extra chunk-GDN implementation work or padding. Final norm/LM head runs
only on each row's last token when final=True, matching suppressed intermediate
prefill in generator_vllm.py. No vision encoder, MTP or MoE work is counted.

Decode DRAM derives physical tiled weights from multichip_decoder.py:
4 DRAM shards,8 banks,32-wide tiles; mix padding uses2 readers (linear local
logical width4160 includes padded a/b; physical4608; full3584). out K1536,N5120;
gateup K5120,N8704; down K4352,N5120. BFP4 tile576 bytes/1024 values. B1 uses
DRAM-sharded projections and L1 pooled all-reduce; B8/16/32 compact users into
one M tile and use native fused out/down reduce-scatter. Both read each weight
representation once per invocation, NOT once per live user. Separate gate/up
prefill copies are not counted as decode reads. Native views reuse the same
weights. BF8 LM head [5120,248320] uses1088-byte tiles and one shared M tile;
embedding is a sharded BF16 row lookup. Compulsory weights are a lower bound:
matmul block/core rereads are unknown and excluded, never inferred from time.

48 recurrent layers: state is TP-sharded FP32 [capacity,48,128,128] and BF16
history[capacity,3,10240]. Unmasked state uses in-place kernel read+write (2 transfers). Above9 rows
(the110-core grid divided by12 local value heads), split outputs concatenate
then copy home, adding4 transfers. Partial occupancy disables in-place state:
kernel read+write, preservation where read/read/write, copy read+write total7
transfers, plus2 concat transfers above9 rows.
Count physical capacity even for inactive rows since traces execute their state
math. Additional concatenation/slicing/transposition transfers are excluded.
16 attention layers: one BFP8 K/V prefix read per live row and one token write,
2176 bytes/token across four ASICs; this ignores SDPA rereads and masked slots.
Prefix read includes the current token. The write is a payload lower bound, not
an estimate of any read/modify/write amplification of a physical tile.

Additional explicit DRAM traffic: fused B>=8 out/down intermediates each write
and read a BF16 [32,5120] tensor per ASIC per layer; their outputs and column
all-gathers are L1 and are not charged as DRAM. B1 pooled all-reduce scratch is
L1, so no DRAM network-byte fiction is added. Terminal forces hidden to DRAM,
normalizes there and writes BF16 vocab-sharded logits (then sampler reads), all
padded to32 rows. Include final norm weights once per ASIC and embedding reads.
Decoder norms/activations mostly reside in L1: unknown spill and repeated
intermediate reads and sampling/CCL scratch are omitted. At each4096-token
decode boundary, include recurrent/history read+write for the prefix snapshot. Thus decode bytes are an explicit LOWER-BOUND estimate,
not measured bandwidth nor complete byte-exact operation accounting. All those
operations, communication and host gaps MUST remain inside the timing denominator.

Peak: four participating ASICs,120 physical Tensix cores each, nominal1.35GHz,
4096 LoFi FLOP/core/cycle;4*512GB/s GDDR6. Hardware source:
https://docs.tenstorrent.com/aibs/blackhole/p300.html (2 ASICs/card,1024GB/s/card).
4096 LoFi factor: ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp
performance model and prior tests/report_optimized_multichip.py. Use physical
peak, not the110-core runtime grid. LoFi is the dense BFP4/BFP8 projection peak;
HiFi2 head and HiFi4 sensitive kernels are slower. This ceiling is NOT an
achievable mixed-fidelity ceiling; see prefill_peak() for a fidelity-weighted
ideal ceiling including HiFi2 terminal and HiFi4 recurrence/scalar work.
 Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; matmul_device_operation.cpp performance model. Timing: Actual output-ready boundaries with outgoing-phase overlap ownership; explicitly tagged drain/state-apply/consumption spans; admitted-to-finished active host gaps; request-free residuals excluded; no added synchronization. [Evidence](phase-accounting-b8.json).
- Concurrency 1, prefill: 3.33372 s. Stage11 full64 Qwen3.8-27B useful work / estimated DRAM accounting.

This is a model, never a hardware counter. API inputs describe ONE scheduler
prefill/decode invocation, not independent requests whose weight costs may be
summed. Sum returned work once for each measured invocation on the TP4 mesh.

Prefill derivation follows tests/roofline.py: 48 linear +16 full layers,
H5120, I17408,24 Q heads,4 KV heads,D256; linear16 QK/48 value heads,D128.
Dense projections cost two FLOPs/MAC. Causal QK+AV use actual prefix start and
length: pairs=L*start+L*(L+1)/2. Linear recurrence uses7*48*128*128 useful
ops/token; convolution/gates/norms/softmax use the inherited scalar estimates,
counting a transcendental as one operation. This is useful algorithmic work,
not the extra chunk-GDN implementation work or padding. Final norm/LM head runs
only on each row's last token when final=True, matching suppressed intermediate
prefill in generator_vllm.py. No vision encoder, MTP or MoE work is counted.

Decode DRAM derives physical tiled weights from multichip_decoder.py:
4 DRAM shards,8 banks,32-wide tiles; mix padding uses2 readers (linear local
logical width4160 includes padded a/b; physical4608; full3584). out K1536,N5120;
gateup K5120,N8704; down K4352,N5120. BFP4 tile576 bytes/1024 values. B1 uses
DRAM-sharded projections and L1 pooled all-reduce; B8/16/32 compact users into
one M tile and use native fused out/down reduce-scatter. Both read each weight
representation once per invocation, NOT once per live user. Separate gate/up
prefill copies are not counted as decode reads. Native views reuse the same
weights. BF8 LM head [5120,248320] uses1088-byte tiles and one shared M tile;
embedding is a sharded BF16 row lookup. Compulsory weights are a lower bound:
matmul block/core rereads are unknown and excluded, never inferred from time.

48 recurrent layers: state is TP-sharded FP32 [capacity,48,128,128] and BF16
history[capacity,3,10240]. Unmasked state uses in-place kernel read+write (2 transfers). Above9 rows
(the110-core grid divided by12 local value heads), split outputs concatenate
then copy home, adding4 transfers. Partial occupancy disables in-place state:
kernel read+write, preservation where read/read/write, copy read+write total7
transfers, plus2 concat transfers above9 rows.
Count physical capacity even for inactive rows since traces execute their state
math. Additional concatenation/slicing/transposition transfers are excluded.
16 attention layers: one BFP8 K/V prefix read per live row and one token write,
2176 bytes/token across four ASICs; this ignores SDPA rereads and masked slots.
Prefix read includes the current token. The write is a payload lower bound, not
an estimate of any read/modify/write amplification of a physical tile.

Additional explicit DRAM traffic: fused B>=8 out/down intermediates each write
and read a BF16 [32,5120] tensor per ASIC per layer; their outputs and column
all-gathers are L1 and are not charged as DRAM. B1 pooled all-reduce scratch is
L1, so no DRAM network-byte fiction is added. Terminal forces hidden to DRAM,
normalizes there and writes BF16 vocab-sharded logits (then sampler reads), all
padded to32 rows. Include final norm weights once per ASIC and embedding reads.
Decoder norms/activations mostly reside in L1: unknown spill and repeated
intermediate reads and sampling/CCL scratch are omitted. At each4096-token
decode boundary, include recurrent/history read+write for the prefix snapshot. Thus decode bytes are an explicit LOWER-BOUND estimate,
not measured bandwidth nor complete byte-exact operation accounting. All those
operations, communication and host gaps MUST remain inside the timing denominator.

Peak: four participating ASICs,120 physical Tensix cores each, nominal1.35GHz,
4096 LoFi FLOP/core/cycle;4*512GB/s GDDR6. Hardware source:
https://docs.tenstorrent.com/aibs/blackhole/p300.html (2 ASICs/card,1024GB/s/card).
4096 LoFi factor: ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp
performance model and prior tests/report_optimized_multichip.py. Use physical
peak, not the110-core runtime grid. LoFi is the dense BFP4/BFP8 projection peak;
HiFi2 head and HiFi4 sensitive kernels are slower. This ceiling is NOT an
achievable mixed-fidelity ceiling; see prefill_peak() for a fidelity-weighted
ideal ceiling including HiFi2 terminal and HiFi4 recurrence/scalar work.
 Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; matmul_device_operation.cpp performance model. Timing: Actual output-ready boundaries with outgoing-phase overlap ownership; explicitly tagged drain/state-apply/consumption spans; admitted-to-finished active host gaps; request-free residuals excluded; no added synchronization. [Evidence](phase-accounting-b1.json).
- Concurrency 1, decode: 28.1047 s. Stage11 full64 Qwen3.8-27B useful work / estimated DRAM accounting.

This is a model, never a hardware counter. API inputs describe ONE scheduler
prefill/decode invocation, not independent requests whose weight costs may be
summed. Sum returned work once for each measured invocation on the TP4 mesh.

Prefill derivation follows tests/roofline.py: 48 linear +16 full layers,
H5120, I17408,24 Q heads,4 KV heads,D256; linear16 QK/48 value heads,D128.
Dense projections cost two FLOPs/MAC. Causal QK+AV use actual prefix start and
length: pairs=L*start+L*(L+1)/2. Linear recurrence uses7*48*128*128 useful
ops/token; convolution/gates/norms/softmax use the inherited scalar estimates,
counting a transcendental as one operation. This is useful algorithmic work,
not the extra chunk-GDN implementation work or padding. Final norm/LM head runs
only on each row's last token when final=True, matching suppressed intermediate
prefill in generator_vllm.py. No vision encoder, MTP or MoE work is counted.

Decode DRAM derives physical tiled weights from multichip_decoder.py:
4 DRAM shards,8 banks,32-wide tiles; mix padding uses2 readers (linear local
logical width4160 includes padded a/b; physical4608; full3584). out K1536,N5120;
gateup K5120,N8704; down K4352,N5120. BFP4 tile576 bytes/1024 values. B1 uses
DRAM-sharded projections and L1 pooled all-reduce; B8/16/32 compact users into
one M tile and use native fused out/down reduce-scatter. Both read each weight
representation once per invocation, NOT once per live user. Separate gate/up
prefill copies are not counted as decode reads. Native views reuse the same
weights. BF8 LM head [5120,248320] uses1088-byte tiles and one shared M tile;
embedding is a sharded BF16 row lookup. Compulsory weights are a lower bound:
matmul block/core rereads are unknown and excluded, never inferred from time.

48 recurrent layers: state is TP-sharded FP32 [capacity,48,128,128] and BF16
history[capacity,3,10240]. Unmasked state uses in-place kernel read+write (2 transfers). Above9 rows
(the110-core grid divided by12 local value heads), split outputs concatenate
then copy home, adding4 transfers. Partial occupancy disables in-place state:
kernel read+write, preservation where read/read/write, copy read+write total7
transfers, plus2 concat transfers above9 rows.
Count physical capacity even for inactive rows since traces execute their state
math. Additional concatenation/slicing/transposition transfers are excluded.
16 attention layers: one BFP8 K/V prefix read per live row and one token write,
2176 bytes/token across four ASICs; this ignores SDPA rereads and masked slots.
Prefix read includes the current token. The write is a payload lower bound, not
an estimate of any read/modify/write amplification of a physical tile.

Additional explicit DRAM traffic: fused B>=8 out/down intermediates each write
and read a BF16 [32,5120] tensor per ASIC per layer; their outputs and column
all-gathers are L1 and are not charged as DRAM. B1 pooled all-reduce scratch is
L1, so no DRAM network-byte fiction is added. Terminal forces hidden to DRAM,
normalizes there and writes BF16 vocab-sharded logits (then sampler reads), all
padded to32 rows. Include final norm weights once per ASIC and embedding reads.
Decoder norms/activations mostly reside in L1: unknown spill and repeated
intermediate reads and sampling/CCL scratch are omitted. At each4096-token
decode boundary, include recurrent/history read+write for the prefix snapshot. Thus decode bytes are an explicit LOWER-BOUND estimate,
not measured bandwidth nor complete byte-exact operation accounting. All those
operations, communication and host gaps MUST remain inside the timing denominator.

Peak: four participating ASICs,120 physical Tensix cores each, nominal1.35GHz,
4096 LoFi FLOP/core/cycle;4*512GB/s GDDR6. Hardware source:
https://docs.tenstorrent.com/aibs/blackhole/p300.html (2 ASICs/card,1024GB/s/card).
4096 LoFi factor: ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp
performance model and prior tests/report_optimized_multichip.py. Use physical
peak, not the110-core runtime grid. LoFi is the dense BFP4/BFP8 projection peak;
HiFi2 head and HiFi4 sensitive kernels are slower. This ceiling is NOT an
achievable mixed-fidelity ceiling; see prefill_peak() for a fidelity-weighted
ideal ceiling including HiFi2 terminal and HiFi4 recurrence/scalar work.
 Peak: https://docs.tenstorrent.com/aibs/blackhole/p300.html; matmul_device_operation.cpp performance model. Timing: Actual output-ready boundaries with outgoing-phase overlap ownership; explicitly tagged drain/state-apply/consumption spans; admitted-to-finished active host gaps; request-free residuals excluded; no added synchronization. [Evidence](phase-accounting-b1.json).

Total client-stage wall time: 1288.8 seconds.
